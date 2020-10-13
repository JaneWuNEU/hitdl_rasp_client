class Static_Info:
   # ====== edge and mobile controller ip
   EDGE_IP = "192.168.1.16"
   LISTENER_IP = '192.168.1.20'
   LISTENER_PORT= 10970

   # ======= user ip and port
   SERVER_USER_IP = "192.168.1.20"
   RASP_USER_IP_START = 20
   RECV_PORT_INTERVEL = 50
   SERVER_RECV_PORT_START = 2000
   RASP_RECV_PORT_START = 2000

   # ====== the amount of users on differnt devices namely stack_server and rasp
   SERVER_USER_NUM = 12
   RASP_USER_NUM = 3

   # ======= frame rate of each model
   MOBILENET_FRAME_RATE = 6

   #======the maximum number of CPU cores for SERVER_USER
   # =====the maximum number of RASP_USER
   SERVER_CORE_UPPER = 16
   RASP_USER_UPPER = 5

   #======== user bandwidth for models
   INCEPTION_USER_BANDWIDTH = 109.9    #Mbits/s
   RESNET_USER_BANDWIDTH = 109.9
   MOBILENET_USER_BANDWIDTH = 109.9

   SEND_FAILS_TIMES = 5
   ACCEPT_WAIT_TIME = 2

   RECORDS_PERIODS = 5
