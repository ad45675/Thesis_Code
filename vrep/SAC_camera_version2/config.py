"""
各個參數

By : ya0000000
2021/08/31
"""
import numpy as np

ON_TRAIN =True                  # TRUE:訓練模式 , FALSE:驗證模式
yolo_detect = True             # TRUE:有結合YOLO , FALSE:沒有結合YOLO
random_train = False             # TRUE:物體隨機放置 , FALSE:物體固定點放置
color_state = False             # TRUE:RGB state , FALSE:depth state
pre_train = False                # TRUE:有 pre_train , FALSE:沒有 pre_train
render = False                  # 可視化(沒用到)
num_object = 4                  # 環境中有幾個物體要替換
initial_joint=[0,0,0,0,0,0]     # robot原始姿態

"""
權重資訊
"""
PATH_EVAL = ['06121204', '199']  # for eval
# PATH_EVAL = ['04240939', '199']  # for  banana eval
#要放碩論的
# PATH_EVAL = ['09081712', '35']  # for eval

"""
強化學習參數
"""
method = 'sac'
sac = True
hidden_dim=[64,64]
reparameterize_critic=False
reparameterize_actor=True

store_color_state_dim  = [3,64,64]  # 若輸入是彩色圖
store_sate_dim = [1,64,64]          # 若輸入是深度圖
image_input = 64                    # 輸入影像大小
state_dim = 512                     # 特徵萃取後輸入sac的維度
action_dim = 2                      # sac輸出維度
a_bound = [1,1]                     # sac輸出最大為1

MAX_EPISODES = 1000                 # 疊代次數(共要拾取幾回合)
MAX_EP_STEPS = 100                  # 拾取次數(每回合拾取幾次)

A_LR = 0.001                        # actor learning rate
C_LR = 0.001                        # critic learning rate
gamma = 0.99                        # reward discount
reward_scale =0.01                  # entropy
tau = 0.005                         # soft update
MEMORY_CAPACITY = 200000            # replay buffer
batch_size = 64                     # 取樣批次
eval_iteration = 10                 # 存model

"""
[[camera info]]
yolo_detect : TRUE:有結合YOLO , FALSE:沒有結合YOLO
theta : vertical field of view
dis_far : 最遠可視深度
dis_near : 最近可深度
depth_scale : 用於還原的係數


"""
if(yolo_detect):
    resolutionX_C  = 512
    resolutionY_C = 424
    resolutionX_D  = 512
    resolutionY_D = 424

    RoiOffset_Y = 0
    RoiOffset_X = 0
    RoiOffset_Y_ = 100
    RoiOffset_X_ = 0
else:
    resolutionX_C = 512
    resolutionY_C = 424
    resolutionX_D = 512
    resolutionY_D = 424

    RoiOffset_Y = 0
    RoiOffset_X = 130
    RoiOffset_Y_ = 100
    RoiOffset_X_ = 0

#---角度
theta = 60
theta_ratio = resolutionX_C / resolutionY_C

#---深度資訊
dis_far = 4.5
dis_near = 0.5
depth_scale = 1000

#---存照片路徑
SAVE_IMAGE_PATH = 'C:/Users/user/Desktop/rl/vrep/SAC_camera_version'


#-----------------yolo 相關

show_yolo = False                               # --> 存照片然後讀RGB和深度圖片
yolo_Img_path = "imgTemp\\frame.jpg"            # --> 存照片的地方與名子
yolo_Dep_path = "imgTemp\\Dep_frame.jpg"        # --> 存照片的地方與名子
yolo_Det_Img_path = 'imgTemp\\Det_frame.jpg'    # --> 存照片的地方與名子





