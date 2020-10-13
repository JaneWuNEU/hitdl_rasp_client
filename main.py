from threading import Thread,Event
import sys
sys.path.append("./")
import socket
import time
from multiprocessing import Process,Value
from utils.util import SocketCommunication,ControlBandwidth
from utils.static_info import Static_Info
import os
import pandas as pd
from datetime import datetime
from user import User
import numpy as np
from multiprocessing import Process,Manager
comm_sock = SocketCommunication()
import signal

def logout_users(user_list,recv_port_list):
    if len(user_list)!=0:
        i = 0
        for user in user_list:
            # 0. logout users by set the $activate_flag_list as -1
            # 1. release the recv_port
            try:
                recv_port = recv_port_list[i].value
                os.system('fuser -k -n tcp  '+str(recv_port))
                os.kill(user.run_model_pid,signal.SIGKILL)
                print("#################logout users##################",user.user_id,user.model_name)
            except Exception as e:
                print("error happens when logouts users",e)
            i = i+1
    #return user_list
def listen_notice(edge_notice_event,create_user_finish_event,edge_notice):
    '''
    listen to the notice from the edge.
    When the notice comes, change the state of the shared edge_notice, which triggers
    the following adjustment. What's more, there are two kind of notices
    1. create users ==>  remove the out-date $model_details of the $edge_notice and rewrite the newest info
    2. activate users==> refresh the $port_details of the $edge_notice
    '''
    listner_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listner_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    listner_sock.bind((Static_Info.LISTENER_IP,Static_Info.LISTENER_PORT))
    listner_sock.listen(5)
    # print("--------------welcome to hitdl-------------")
    while True:
        try:
            conn,add = listner_sock.accept()
            data = comm_sock.recv_data(conn)
            conn.close()
            print("----------------------listen notice gets data--------------------",data)
            if data['type'] == 'create':
                # 1. refresh the $model_details
                edge_notice['model_details'] = data['model_details']
                edge_notice['port_details'] = None
                edge_notice['type']='create'
                edge_notice_event.set()
            elif data['type'] == 'activate':
                # 2. refresh the $port_details
                while not create_user_finish_event.is_set():
                    time.sleep(0.05)
                edge_notice['port_details'] = data['port_details']
                edge_notice['type'] = 'activate'
                edge_notice['bandwidth'] = data['bandwidth']
                # print("-------bbbbbbbbbbb-----", edge_notice)
                edge_notice_event.set()
            elif data['type'] == 'remove_rasp':
                edge_notice["type"]="remove_rasp"
                edge_notice_event.set()

        except Exception as e:
            print("error happens when receving the edge's notice",e)



def create_users(user_list,model_details):
    '''
    controller maintains a user list and a core list
    Depend on the $model_details to create users.
    1. read the $ins_num and $user_num_per_ins to create users
    2. Intialize an user with necessary info according to $model_details
     Note that error happens when there is no sufficient cpu cores.
    3. start a new process for each user, and then it to specific number of CPU cores (marked as intra) defined in Static_info
    Note that user has different types and various intra. Meanwhile, CPU cores must be allocated carefully without no intersect
    between different users.

    '''
    user_list.clear()
    now_time = datetime.now().strftime("%d-%H-%M-%S")
    # 1. create users for models
    mobilenet_details = model_details["mobilenet"]
    user = User(mobilenet_details["k"], "mobilenet", Static_Info.SERVER_USER_IP, now_time)
    user_list.append(user)

    # 2.start process for each user and bind the specific ports.
    activate_flag_list = []
    recv_port_list = []
    request_records_list = []
    print("create users",user_list)
    for user in user_list:
        request_records = {}#manager.dict()
        activate_flag = Value("i",0)
        recv_port = Value("i",0)
        user_process = Process(target=user.run_model,args=[activate_flag,recv_port,request_records])
        user_process.start()
        user.run_model_pid = user_process.pid
        print("user's pid",user.run_model_pid, os.getpid())
        activate_flag_list.append(activate_flag)
        recv_port_list.append(recv_port)
        request_records_list.append(request_records)
    return activate_flag_list,recv_port_list,request_records_list

def activate_users(user_list,port_details,activate_flag_list):
    '''
    The main process shares a variable $activate_flag as type of  multiprocessing.Value
    with each user's process.
    When the $activate_flag is zero, it means the user just created without being activated.
    When the $activate_flag >0, it means the user is activated and is able to send data to the edge.
    The receving port of the edge is represent as the value of $activate_flag.
    When $activate_flag is -1, it means to logout the users
    '''
    i = 0
    for user in user_list:
        #print("=============",user.user_id)
        model_name = user.model_name
        #ins_id = user.ins_id
        ins_port = port_details[model_name][0]
        activate_flag_list[i].value = ins_port # activate users with their respective receiving port.
        print("###########activate users#######","user ",user.user_id,"ins_port",ins_port,"model_name",model_name)
        i = i+1


if __name__ == '__main__':
    # 0. start a thread to listen to the notice from the edge
    edge_notice_ev = Event()
    create_user_finish_event = Event()
    edge_notice = {'model_details':None,
                   'port_details':None,
                   'type':None}
    notice_thread = Thread(target=listen_notice,args=[edge_notice_ev,create_user_finish_event,edge_notice])
    notice_thread.start()

    # maintain a user list and a instance list. when a new round comes, remove the user and close the instance
    user_list = []
    activate_flag_list = None
    recv_port_list = []
    request_records_list = None
    bandwidth_controller = ControlBandwidth()
    start = time.time()
    i = 0
    manager = Manager()
    while True:
        if edge_notice_ev.is_set():
            #print("===============",edge_notice)
            if edge_notice['type'] == 'create':
                # 1. logout users
                logout_users(user_list,recv_port_list)
                # 2. create users
                #del user_list
                activate_flag_list,recv_port_list,request_records_list = create_users(user_list,edge_notice['model_details'])
                create_user_finish_event.set()
                edge_notice_ev.clear()
            elif edge_notice['type'] == 'activate':
                # 3. activate users
                #print("================recv notice===========",edge_notice)
                bandwidth_controller.change_bandwidth(edge_notice)
                activate_users(user_list,edge_notice['port_details'],activate_flag_list)
                create_user_finish_event.clear()
                edge_notice_ev.clear()
                i = i + 1
            elif edge_notice['type'] == 'remove_rasp':
                # 1. logout users
                try:
                    logout_users(user_list,recv_port_list)
                    # 2. create users
                    #del user_list
                except Exception as e:
                    print("%%%%%%%%%%%%%%%%%%logsout errors",e)

