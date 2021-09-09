"""
各個參數

By : ya0000000
2021/08/31
"""
import numpy as np

ON_TRAIN = False            # TRUE:上機訓練模式 , FALSE:驗證模式
clean = False               # TRUE:分類實驗 , FALSE:拾取特定物件
vrep_show = False           # TRUE:開v-rep , FALSE:不開v-rep
show_yolo = True            # TRUE:顯示圖片 , FALSE:不顯示圖片
initial_joint=[0,0,0,0,0,0] # v-rep robot原始姿態
"""
權重資訊
"""
MODEL_PATH = 'C:/Users/user/Desktop/rl/vrep/SAC_camera_version2'
PATH_EVAL = ['06121204', '180']  # for eval 180  '04082155', '190'
PATH_EVAL_Banana = ['04212333', '199']  # for banana eval '04212333', '199'
Hair_drier_PATH_EVAL = ['06201309', '199']  # for exp1

"""
強化學習參數
"""
method = 'sac'
sac = True
hidden_dim=[64,64]
reparameterize_critic=False
reparameterize_actor=True

store_sate_dim = [1,64,64]  # 輸入是深度圖
image_input = 64            # 輸入影像大小
state_dim = 512             # 特徵萃取後輸入sac的維度
action_dim = 2              # sac輸出維度
a_bound = [1,1]             # sac輸出最大為1

MAX_EPISODES = 20           # 疊代次數(主要for上機訓練)
MAX_EP_STEPS = 20           # 拾取次數(主要for上機訓練)

A_LR = 0.001                # actor learning rate
C_LR = 0.001                # critic learning rate
gamma = 0.99                # reward discount
reward_scale =0.01          # entropy
tau = 0.005                 # soft update
MEMORY_CAPACITY = 200000    # replay buffer
batch_size = 64             # 取樣批次
eval_iteration = 5          # 存model

"""  camera info   """
#---影像長寬(real kinect)
resolutionX  = 1920
resolutionY = 1080

# 前
"""
(當clean table時)
RoiOffset_X = 900
RoiOffset_Y = 50

RoiOffset_X_ = 400
RoiOffset_Y_ = 400

(當 train by hand 時)
RoiOffset_X = 950
RoiOffset_Y = 280

RoiOffset_X_ = 500
RoiOffset_Y_ = 400

"""
if (ON_TRAIN):
    # 線上訓練
    RoiOffset_X = 1000
    RoiOffset_Y = 300

    # 後
    RoiOffset_X_ = 600
    RoiOffset_Y_ = 400

else:
    # 特定物件拾取和分類
    RoiOffset_X = 900
    RoiOffset_Y = 50
    # 後
    RoiOffset_X_ = 400
    RoiOffset_Y_ = 400

"""  vrep camera info   """
#---角度
theta = 60
theta_ratio = resolutionX / resolutionY

#---深度資訊
dis_far = 1
dis_near = 0.01
depth_scale = 1000
"""  vrep camera info   """

#---存照片路徑
SAVE_IMAGE_PATH = 'C:/Users/user/Desktop/rl/vrep/SAC_camera_version_real_world'
#-----------------yolo 相關
yolo_Img_path = "imgTemp\\frame.jpg"
yolo_Dep_path = "imgTemp\\Dep_frame.jpg"
yolo_Det_Img_ROI_path = 'imgTemp\\Det_frame_ROI.jpg' # ROI
yolo_Det_Img_ORI_path = 'imgTemp\\Det_frame_Ori.jpg' # Original
# ---------------- for eval
yolo_Img_path_eval = "imgTemp\\frame_eval.jpg"
yolo_Dep_path_eval = "imgTemp\\Dep_frame_eval.jpg"
yolo_Det_Img_ROI_path_eval = 'imgTemp\\Det_frame_ROI_eval.jpg' # ROI
yolo_Det_Img_ORI_path_eval = 'imgTemp\\Det_frame_Ori_eval.jpg' # Original
#checkpoint_path = os.path.join('./model/' + path + '/net/' + str(int(i)))


""" 新校的 """
Hand_To_Eye = np.array([
    [-9.9141887327760370e-01 ,6.3831177771060010e-02, 1.1407978985484538e-01,  2.6867348182830278e+02  ],
    [2.4491688757729665e-02, 7.6652839708685572e-01, -6.4174323030418312e-01,  -5.8722103734133526e+01   ],
    [-1.2840862467436101e-01, -6.3903035702837574e-01, -7.5838738643613091e-01, 1.2132108439637479e+03],
    [0, 0, 0, 1]])

Eye_To_Hand = np.array([
    [ -9.9141887327760381e-01 ,-2.4491688757729738e-02 ,-1.2840862467436095e-01  , 4.2071649305933681e+02  ],
    [6.3831177771060260e-02,7.6652839708685649e-01, -6.3903035702837585e-01, 8.0314097403679239e+02 ],
    [1.1407978985484563e-01, -6.4174323030418368e-01, -7.5838738643613102e-01, 8.5174907426249706e+02],
    [0, 0, 0, 1]])

Color_Intrinsic_Matrix = np.array([
    [1.0454801485926105e+03, 0., 9.3496788610430747e+02],
    [0., 1.0424962581852583e+03, 5.3487433803128465e+02],
    [0., 0., 1.]
])
Discoeff = np.array(
    [2.1664844254275058e-02, 3.6657109783434116e-02, -7.0038831777495102e-03, -4.0304747448493940e-03,
     -2.1132198547319650e-01])

Depth_Intrinsic_Matrix = np.array([
    [363.558411, 0.000000, 255.820206],
    [0.000000, 363.558411, 209.001694],
    [0, 0, 1]])





