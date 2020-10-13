import socket
from utils.static_info import Static_Info
from threading import Event,Thread
from queue import Queue
from utils.util import SocketCommunication
import time
import tensorflow as tf
import model_zoo.net.mobilenet_v1 as mobilenet_v1
import tensorflow.contrib.slim as slim
from utils.static_info import Static_Info
import numpy as np
from utils.model_info import ModelInfo
import codecs
import pickle
import os
import pandas as pd
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
import datetime
import  pickle
import struct
import sys
#from multiprocessing import Queue,Manager
class User:
    def __init__(self,k,model_name,user_ip,records_file):
        '''
        initialize member variables
        '''
        # ==== create necessary member variables ===
        self.k = k
        self.model_name = model_name
        self.recv_port = None
        self.logout_event = Event()
        self.request_records = {}
        self.model_info = ModelInfo()
        self.recv_socket = None
        self.run_model_pid = None
        self.records_file = records_file
        self.user_id = 0
        self.ins_id=0
        self.user_ip = user_ip

    def process_image(self,sess, out, endpoints,input_images,im,activate_flag,data_queue,request_records):
        '''
        process images and add records the process
        tips:
        '''

        pic_num = 0
        start_send_queue = False
        # 0. confirm the frame interval
        frame_interval = 1.0 / Static_Info.MOBILENET_FRAME_RATE
        image_id = -1
        while True:
            # 1. process the image
            start_time = time.time()
            if activate_flag.value == 0:  # have not been activated.
                result = sess.run(out, feed_dict={input_images: im})
                image_id = np.argmax(result[0])
                end_time = round(time.time(),3)
            elif activate_flag.value>0:
                if self.k !=0:
                    a = time.time()

                    output_layer = endpoints[self.model_info.get_layer_name_by_index(self.model_name, self.k)]
                    result = sess.run(output_layer, feed_dict={input_images: im})
                    b = time.time()
                    #print("========hybrid ===========", self.model_name, self.k,result.shape)
                else:
                    result = im
                image_id = 0
                data_queue.put({'data':result,"pic_num":pic_num})

            # 2. record the time cost of processing an image
            process_image_interval = time.time()-start_time
            request_records[str(pic_num)]={"start_time":start_time,"end_time":0, "image_id":image_id,"local_run_time":process_image_interval,
                                           "mobile_send_time":0, "mobile_recv_time":0,"edge_run_time":0,"queue_time":0,"edge_recv_time":0}

            # 3. check if the sleep is necessary so as to ensure the process in a fixed frame rate
            if np.round(frame_interval-process_image_interval,3)>0.003:
                time.sleep(np.round(frame_interval-process_image_interval,3))

            pic_num = pic_num+1

    def get_recv_port(self):
        return self.recv_port


    def bound_pid(self,pid):
        intra = len(self.core_id)
        core_str = ''
        for i in self.core_id:
            core_str = core_str+str(i)+","
        core_str = core_str[:core_str.rindex(",")]
        if intra == 1:
            os.system("taskset -cp "+core_str+" " + str(pid))
        elif intra == 2:
            os.system("taskset -cp "+core_str+" " + str(pid))

    def run_model(self,activate_flag,recv_port,request_records):
        """
        The user should run the complete model in specific CPU cores defined by $core_id after being created.
        Only the the edge activates the user can it run the model in a hybrird way.
        1.create a queue to share with the sending thread.
        2. listen to the signal from the main process
           when the signal is  -1, it means logouts.
           when the signal is 0, it means the user has not been activated.
           when the signal is >0, it means the recv port of some model ins.
        """
        # 0.1 start the threads to send and receive data
        data_queue = Queue()
        request_records ={}
        Thread(target=self.recv_data, args=[activate_flag, recv_port, request_records]).start()
        Thread(target=self.send_data,args=[activate_flag,data_queue,recv_port,request_records]).start()

        # recv_port, recv_socket = self.assign_recv_port()
        # 1. initialize the model
        sess_config = tf.ConfigProto(intra_op_parallelism_threads=4, log_device_placement=False)
        model_path = "model_zoo/weights/mobilenet_v1_1.0_224.ckpt"
        input_images = tf.placeholder(dtype=tf.float32, shape=[None, 224,224,3], name='input')
        im = np.load("input_data/mobilenet/mobilenet_input_guitar.npy")
        im = im[np.newaxis,:]
        with tf.Session(config=sess_config) as sess:
            with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope()):
                out, endpoints = mobilenet_v1.mobilenet_v1(inputs=input_images)
                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver()
                saver.restore(sess, model_path)
                self.process_image(sess, out, endpoints, input_images, im,activate_flag,data_queue,request_records)

    def send_data(self,activate_flag,data_queue,recv_port,request_records):
        sock_tools = SocketCommunication()
        print("going to send data")
        while recv_port.value==0:
            time.sleep(0.01)
            #print("rasp send data to edge")


        while True:
            if activate_flag.value >0:
                try:
                    a1 = time.time()
                    data = data_queue.get()
                    b1 = time.time()
                    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    m = time.time()
                    #print("创建socket",m-b1)
                    m = time.time()
                    client.connect((Static_Info.EDGE_IP,activate_flag.value))
                    c = time.time()
                    #print("=============建立连接===========",c-m)
                    upload_data = {"recv_port": recv_port.value,"data":data,"user_ip":self.user_ip,"user_id":self.user_id}
                    a = time.time()
                    upload_pickles = pickle.dumps(upload_data)
                    e = time.time()
                    sock_tools.send_data_bytes(client,upload_pickles)
                    d = time.time()
                    request_records[str(upload_data["data"]["pic_num"])]["mobile_send_time"] = round(d-a,3)
                    #print("get data",a1-b1,"建立连接",c-b1,"封装数据",a-c,"picke",e-a,"发送数据",d-e)
                except Exception as e:
                    print("error happens when user ", self.model_name, " ", self.user_id, "sends data to the edge", activate_flag.value, e)
                    #pass
            elif activate_flag.value == 0:
                time.sleep(0.05)
            else:
                break


    def assign_recv_port(self):
        if self.model_name == 'mobilenet':
            temp_revc_port = Static_Info.RASP_RECV_PORT_START+self.user_id*Static_Info.RECV_PORT_INTERVEL
        else:
            temp_revc_port = Static_Info.SERVER_RECV_PORT_START+self.user_id*Static_Info.RECV_PORT_INTERVEL
        # 1. open a socket to listen to the results from the edge
        recv_socket = None
        while True:
            try:
                recv_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                recv_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                recv_socket.bind((self.user_ip, temp_revc_port))
                recv_socket.listen(5)
                recv_port = temp_revc_port
                break
            except Exception as e:
                print("error happens when assigning recv ports ",temp_revc_port,"to user ",self.user_id,self.model_name,e)
                temp_revc_port = temp_revc_port+1
        return recv_port,recv_socket

    def recv_data(self,activate_flag,recv_port,shared_request_records):

        """
        Different types of users have different methods to assign ports.
        SERVER_USER as 1000+$user_id*$RECV_PORT_INTERVEL, RASP_USER as 2000+$user_id*$RECV_PORT_INTERVEL.
        Note that the exception may be thrown during opening a specific port marked as $wrong_port, when try a new port
        as $(wrong_port++) until an available port appears.
        """
        # 2. receive the results until the user logouts
        recv_port_id,recv_socket = self.assign_recv_port()
        recv_port.value = recv_port_id
        comm_sock = SocketCommunication()

        file_name = "user_" + str(self.user_id) + "_" + self.model_name + "_ins_" + str(
            self.ins_id) + ".txt"
        file_path = "records/" + self.records_file + "/" + self.model_name + "/ins_" + str(self.ins_id)
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        start_time = time.time()

        while True:
            if activate_flag.value>=0:

                conn,add = recv_socket.accept()
                last_recv_time = time.time()
                a = time.time()
                result = comm_sock.recv_data(conn)
                b = time.time()
                recv_time = b-a
                c = time.time()
                conn.close()
                end_time = time.time()
                #print("接收数据",b-a,"关闭连接",end_time-c)
                # 2.1 depend on the edge result to refresh the records
                try:
                    shared_request_records[str(result["pic_num"])]["mobile_recv_time"] = round(recv_time,3)
                    shared_request_records[str(result["pic_num"])].update(result)
                    if shared_request_records[str(result["pic_num"])]["end_time"] == 0:
                        shared_request_records[str(result["pic_num"])]["end_time"] = end_time
                    # "mobile_recv_time":0,"edge_run_time":0,"queue_time":0,"edge_recv_time":0
                    shared_request_records[str(result["pic_num"])]["edge_run_time"] = result["edge_run_time"]
                    shared_request_records[str(result["pic_num"])]["queue_time"] = result["queue_time"]
                    shared_request_records[str(result["pic_num"])]["edge_recv_time"] = result["edge_recv_time"]
                except Exception as e:
                    print("error happens when updating records",pic_num)
                if time.time() - start_time >= Static_Info.RECORDS_PERIODS:
                    pic_num_keys = list(shared_request_records.keys())
                    with open(file_path + "/"+file_name, "a") as f:
                        f.write("==========write the file============"+datetime.datetime.now().strftime("%H:%M:%S")+"==============\n")
                        print("write data",datetime.datetime.now().strftime("%H:%M:%S"))
                        records = ""
                        for pic_num in pic_num_keys:
                            try:
                                record = shared_request_records[pic_num]
                                records = records+"#"+pic_num+":"+str(record)+"\n"
                                shared_request_records.pop(pic_num)
                            except Exception as e:
                                print("error happens when saving records",pic_num)
                        f.writelines(records)
                    start_time = time.time()


                #print('user', self.user_id, "model", self.model_name, 'recv_data', str(result["pic_num"]))

        # 测试如果类继承与Process，main方法是在主进程运行还是子进程运行
