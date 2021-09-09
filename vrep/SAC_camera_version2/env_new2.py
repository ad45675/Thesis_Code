
"""
rl environment

By : ya0000000
2021/08/31
"""
import numpy as np
import os
import math
import time
import inverseKinematics as IK
from IK_FindOptSol import FindOptSol
from robot_vrep import my_robot
import config
import cv2 as cv
from yolo import *

def creat_path(path):
    if path_exsit(path=path):
        print(path+' exist')
    else:
        os.makedirs(path)
def path_exsit(path):
    if os.path.exists(path):
        return True
    else:
        return False

radtodeg = 180 / math.pi  # 弧度轉角度
degtorad = math.pi / 180  # 角度轉弧度
terminal_reward = 1000
finalpos = [0, 0, 180]


#這裡單位是 cm  吸嘴加0.125
DH_table = np.array([[0,            0.345,  0.08,   math.pi / 2],
					 [0+math.pi / 2 , 0,  0.27,     0],
					 [0,             0,     0.09,    math.pi / 2],
					 [0,            0.295,  0,       -math.pi / 2],
					 [0,            0,      0,       math.pi / 2],
					 [0,       0.102+0.125, 0,          0]])


def save_txt(data, fmt='%f'):
    f = open('C:/Users/user/Desktop/rl/data.txt', 'a')
    np.savetxt(f, data, fmt=fmt)
    f.close()

class robot_env(object):
    degtorad = math.pi / 180
    state_dim = config.state_dim
    action_dim = config.action_dim

    def __init__(self):
        self.radtodeg = 180 / math.pi  # 弧度轉角度
        self.degtorad = math.pi / 180  # 角度轉弧度
        self.my_robot = my_robot()
        self.my_robot.connection()
        self.yolo = YOLOV3()
        self.yolo_coco = YOLOV3_coco()
        self.random_flag = 1
        self.object_flag = 0
        self.random_train = config.random_train
    def initial(self):
        self.my_robot.stop_sim()
        self.my_robot.start_sim()

    def reset(self,i):

        """  拿 camera 位置，角度，矩陣 """
        self.cam_position, self.cam_orientation ,self.cam_rotm, self.cam_pose= self.my_robot.get_depth_camera_pose()
        """  robot初始姿態 """
        self.joint = [0, 0, 0, 0, -1.57, 0]
        self.my_robot.move_all_joint(self.joint)
        print('reset')

        """  物體是否要隨機擺放 """
        if (self.random_train):
            if (i+1) % 100 == 0:  # 每100回合換一次物體
                #self.object_flag = np.random.randint(3, size=1)
                self.object_flag = self.object_flag - 1
                if self.object_flag <= -1:
                    self.object_flag =np.random.randint(3,size=1)
            print('object',self.object_flag)

            if self.random_flag == 1:
                self.my_robot.random_object(self.object_flag)     # 物體位置隨機放
            else:
                self.my_robot.no_random_object(self.object_flag)  # 物體位置固定放
        else:
            self.my_robot.no_random_object(self.object_flag)      # 物體位置固定放
        time.sleep(0.2)

        """ --- 隨機選擇yolo偵測的物體 或 yolo偵測到的第一個物體 --- """
        # self.index = np.random.randint(config.num_object, size=1)
        self.index = 0

        return self.get_state()

    def get_state(self):
        # -----************************Img Initial************************-----#
        """  拿 彩色圖，深度資訊(16位元)，深度圖(8位元) """
        self.color_img, self.depth_img, self.depth_img_for_show = self.my_robot.get_camera_data()

        """  存照片然後讀RGB和深度圖片 """
        if config.show_yolo:  # (存照片)
            self.my_robot.arrayToImage(self.color_img)
            self.my_robot.arrayToDepthImage(self.depth_img_for_show)
            RGB_Img = cv.imread(config.yolo_Img_path)  # 讀RGB圖片 (480, 640, 3)
            Dep_Img = cv.imread(config.yolo_Dep_path)  # 讀深度圖片 (480, 640, 3)


        ROI = self.color_img[config.RoiOffset_Y:(config.resolutionY_C-config.RoiOffset_Y_), config.RoiOffset_X:(config.resolutionX_C-config.RoiOffset_X_)]


        # /////////////////////////////////////////////////////////////////////#
        #                            YOLO Detect                              #
        # /////////////////////////////////////////////////////////////////////#
        if(config.yolo_detect):
            if(self.object_flag == 3):   # coco dataset
                self.Yolo_Det_frame, self.coordinate, self.cls, self.label, self.Width_and_Height = self.yolo_coco.detectFrame(ROI)  # 得到框的中心點
            else:                        # cubic dataset
                self.Yolo_Det_frame, self.coordinate,self.cls,self.label,self.Width_and_Height = self.yolo.detectFrame(ROI)  # 得到框的中心點
        else:                            # 沒有yolo
            self.Yolo_Det_frame = ROI
            self.coordinate =np.array([[int((self.Yolo_Det_frame.shape[1]/2)),int((self.Yolo_Det_frame.shape[0]/2))]])
            self.cls =np.array([0])
            self.label =['cubic']
            self.Width_and_Height =np.array([[self.Yolo_Det_frame.shape[1] ,self.Yolo_Det_frame.shape[0]]])

        """  若yolo沒偵測到就重置物件並重新取得狀態 """
        while not (self.Width_and_Height.any()):
            # self.my_robot.random_object(self.object_flag)
            self.my_robot.no_random_object(self.object_flag)
            time.sleep(0.2)
            self.get_state()
            print('No Object !!! ')

        """  顯示yolo結果 """
        if config.show_yolo:
            color = (0, 0, 255)  # BGR
            cv2.circle(self.Yolo_Det_frame, (self.coordinate[self.index][0], self.coordinate[self.index][1]), 2, color, -1)
            cv2.circle(self.color_img, (self.coordinate[self.index][0]+config.RoiOffset_X, self.coordinate[self.index][1]+config.RoiOffset_Y), 2, color, -1)
            # cv2.imshow('color_img', self.color_img)
            cv2.imshow('yolo',self.Yolo_Det_frame)
            # cv2.waitKey(0)
            cv2.imwrite(config.yolo_Det_Img_path, np.array(self.Yolo_Det_frame))  # 储存检测结果图
            cv2.imwrite(config.yolo_Det_Img_path, np.array(self.color_img))  # 储存检测结果图


        # /////////////////////////////////////////////////////////////////////#
        #                            YOLO END                                 #
        # /////////////////////////////////////////////////////////////////////#

        """  彩色影像的state """
        if(config.color_state):

            """ yolo邊界框中心點座標 """
            color_coordinate = np.zeros((2, 1), np.float64)
            color_coordinate[0] = self.coordinate[self.index][0] + config.RoiOffset_X
            color_coordinate[1] = self.coordinate[self.index][1] + config.RoiOffset_Y

            """ yolo邊界框對角線座標 """
            color_left = np.array([color_coordinate[0] - self.Width_and_Height[self.index][0] / 2,
                                 color_coordinate[1] + self.Width_and_Height[self.index][1] / 2])
            color_right = np.array([color_coordinate[0] + self.Width_and_Height[self.index][0] / 2,
                                  color_coordinate[1] - self.Width_and_Height[self.index][1] / 2])

            """ ---- 畫RGB圖顯示 ---- """
            # cv2.circle(self.color_img, (color_coordinate[0], color_coordinate[1]), 2, (0, 255, 0), -1)
            # cv2.rectangle(self.color_img, (color_left[0], color_left[1]), (color_right[0], color_right[1]), (0, 255, 0), 2)
            # cv2.imshow('depth123', self.depth_img_for_show)
            # cv2.imwrite(config.yolo_Dep_path, np.array(self.depth_img))  # 储存检测结果图
            # cv2.waitKey(0)
            """ ---- 畫RGB圖顯示 ---- """

            """ y 的座標 與 x 的座標 """
            cy = np.array([int(color_coordinate[1] - self.Width_and_Height[self.index][1] / 2),
                           int(color_coordinate[1] + self.Width_and_Height[self.index][1] / 2)])
            cx = np.array([int(color_coordinate[0] - self.Width_and_Height[self.index][0] / 2),
                           int(color_coordinate[0] + self.Width_and_Height[self.index][0] / 2)])

            cy = np.clip(cy, 0, 424)
            cx = np.clip(cx, 0, 512)

            """ 拿到邊界框範圍內的影像 """
            self.ROI = self.color_img[cy[0]:cy[1], cx[0]:cx[1]]

            """ resize 為 64*64 """
            color_img = cv.resize(self.ROI, (64,64), interpolation=cv.INTER_CUBIC)

            """ 
                transpose 為將img的data重新排列 
                img為[h,w,channel]
                pytorch 輸入為 [batch,channel,h,w]
            """
            color_img = color_img.transpose((2,0,1))
            self.color_img_input = color_img[np.newaxis, ...]
            s = self.color_img_input


        # ********* 深度影像的state *********
        else:

            """ yolo邊界框中心點 與 對角座標 """
            dep_coordinate = np.zeros((2, 1), np.float64)
            dep_coordinate[0] = self.coordinate[self.index][0] + config.RoiOffset_X
            dep_coordinate[1] = self.coordinate[self.index][1] + config.RoiOffset_Y
            dep_left = np.array([dep_coordinate[0] - self.Width_and_Height[self.index][0] / 2, dep_coordinate[1] + self.Width_and_Height[self.index][1] / 2])
            dep_right = np.array([dep_coordinate[0] + self.Width_and_Height[self.index][0] / 2, dep_coordinate[1] - self.Width_and_Height[self.index][1] / 2])

            """ ---- 畫深度圖 ---- """
            # cv2.circle( self.depth_img_for_show , (dep_coordinate[0], dep_coordinate[1]), 2, (196, 114, 68), -1)
            # cv2.rectangle( self.depth_img_for_show , (dep_left[0], dep_left[1]), (dep_right[0], dep_right[1]), (196, 114, 68), 2)
            # cv2.imshow('depth123', self.depth_img_for_show)
            # cv2.imwrite(config.yolo_Dep_path, np.array(self.depth_img))  # 储存检测结果图
            # cv2.waitKey(0)
            """ ---- 畫深度圖 ---- """

            """ y 的座標 與 x 的座標 """
            cy = np.array([int(dep_coordinate[1] - self.Width_and_Height[self.index][1] / 2), int(dep_coordinate[1] + self.Width_and_Height[self.index][1] / 2)])
            cx = np.array([int(dep_coordinate[0] - self.Width_and_Height[self.index][0] / 2), int(dep_coordinate[0] + self.Width_and_Height[self.index][0] / 2)])

            cy = np.clip(cy, 0, 424)
            cx = np.clip(cx, 0, 512)

            """ 拿到邊界框範圍內的影像 """
            self.ROI = self.depth_img[cy[0]:cy[1], cx[0]:cx[1]]

##########################################################################################
            """顯示深度state(碩論出圖)"""
            # self.depth_img_for_show = self.depth_img_for_show[cy[0]:cy[1], cx[0]:cx[1]]
            # self.color_img= self.color_img[cy[0]:cy[1], cx[0]:cx[1]]
            # # print(self.ROI.shape) #(137, 131)
            # depth_img_64_show = cv.resize( self.depth_img_for_show,(64,64),interpolation = cv.INTER_CUBIC)
            # cv2.imshow('self.depth_img_64_show', depth_img_64_show)
            # cv2.imshow('self.depth_img_for_show', self.depth_img_for_show)
            # cv2.imwrite('imgTemp\\depth.png', self.depth_img_for_show)  # 储存检测结果图
            # cv2.imwrite('imgTemp\\depth_64.png', depth_img_64_show)  # 储存检测结果图
            # cv2.imwrite('imgTemp\\color.png', self.color_img)  # 储存检测结果图
            # cv2.waitKey(0)
########################################################################################
            """ resize 為 64*64 """
            depth_img = cv.resize( self.ROI,(64,64),interpolation = cv.INTER_CUBIC)
            self.depth_img_input = depth_img[np.newaxis, ...]
            s = self.depth_img_input
            s = s[np.newaxis, ...]
        return s


    def step(self, action):

        #action 是 U,V


        done = False
        reward = 0
        success = 0
        suction_value = 0

        """  SAC 之動作輸出映射至感興趣物件影像平面上的位移向量 """
        u1 = 1 + math.floor(action[0] * (self.Width_and_Height[self.index][0]/2 - 0.99))  #-------四捨五入(64 ~ 1)
        v1 = 1 + math.floor(action[1] * (self.Width_and_Height[self.index][1]/2 - 0.99))  #-------四捨五入(64 ~ 1)

        """  最終拾取點座標 u,v """
        u = int(u1 + self.coordinate[self.index][0] + config.RoiOffset_X)
        v = int(v1 + self.coordinate[self.index][1] + config.RoiOffset_Y)

        """  將點顯示在彩色與深度圖上 """
        if config.show_yolo:
            # cv.circle(self.Yolo_Det_frame, (u, v), 5, (255, 0, 255), -1)
            cv.circle( self.depth_img_for_show , (u, v), 5, (255, 0, 255), -1)
            cv.circle(self.color_img, (u, v), 5, (255, 255, 0), -1)
            cv.imshow('color_img',  self.color_img )
            cv.imshow('self.depth_img_for_show',self.depth_img_for_show)
            # 按下任意鍵則關閉所有視窗
            cv.waitKey(0)


        """ 座標(u,v)的深度 """
        pixel_depth = self.my_robot.get_depth(u,v,self.depth_img)

        #------------------------------------------------座標轉換------------------------------------------------#
        """ (u,v)轉到camera frame """
        x, y = self.my_robot.uv_2_xyz(pixel_depth, u, v, config.resolutionX_C, config.resolutionY_C, config.theta)
        img_coor = np.array([x,y, pixel_depth])
        """ camera frame轉到 robot frame """
        world_coor = self.my_robot.coor_trans_AtoB(self.cam_position, self.cam_orientation, img_coor)

        """ 逆向運動學求各軸角度 """
        [tip_Jangle, flag, Singularplace] = IK.InverseKinematics(finalpos, world_coor , DH_table)
        joint,num_of_sol = FindOptSol(tip_Jangle, self.joint,Singularplace)

        if num_of_sol == 0:  # 選到奇異點的解扣分
            reward = reward-0.01
            done= True

        """ vrep移動robot到目標角度 """
        self.my_robot.move_all_joint(joint)
        self.joint = joint  # 更新目前joint位置

        time.sleep(1)
        """ 啟動吸嘴 """
        suction_value = self.my_robot.enable_suction(True)
        time.sleep(0.2)

        """ 抬高手臂 """
        joint = [-0.01142807, -0.2 , 0.03640932,  0.       ,  -1.03999794, -0.01142807]
        self.my_robot.move_all_joint(joint)
        self.joint = joint
        time.sleep(0.5)

        """ 取得cubic現在位置看是否拾取成功 """
        cuboid_pos_now= self.my_robot.get_cuboid_pos(self.object_flag)  # dim=3
        time.sleep(0.2)

        """ 若cubic z位置在0.15之上代表成功反之失敗  """
        if (abs(cuboid_pos_now[2]) > 0.15 ):
            success = 1
        else:
            success = 0

        """ 關掉吸嘴 """
        suction_value = self.my_robot.enable_suction(False)
        time.sleep(0.2)

        """ 若成功則結束此回合；若未成功則需查看cubic是否超過制定範圍，超過則重置物體  """
        if (success >= 1):
            done = True
            print('lift done')
        else:
            if (abs(cuboid_pos_now[1]) > 0.32 or cuboid_pos_now[0] < 0.2 or cuboid_pos_now[0] > 0.6):
                self.my_robot.random_object(self.object_flag) # 如果還沒done 物件又超出範圍則重置物體
                time.sleep(0.2)

        """ 計算獎勵值 """
        reward = reward+success

        """ 獲得下一刻狀態 """
        s_ = self.get_state()


        return s_, reward, done







if __name__ == '__main__':
  pass
##################### record data #####################
# error_record = np.reshape(error, (1, 6))
# # print(joint_out_record)
# path = './Trajectory/'
# name = 'error_record.txt'
# f = open(path + name, mode='a')
# np.savetxt(f, error_record, fmt='%f')
# f.close()
##################### record data #####################

