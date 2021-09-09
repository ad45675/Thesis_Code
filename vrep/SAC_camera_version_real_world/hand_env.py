

import numpy as np
import os
import math
import time
import inverseKinematics as IK
import Kinematics as FK
from IK_FindOptSol import FindOptSol
from robot_vrep import my_robot
import config
import cv2 as cv
from yolo import *
from Kinect_new import Kinect
from mapper import *
import sys

if config.show_yolo:
    import pygame

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
other_feature_dim = config.other_feature_dim
terminal_reward = 1000
finalpos = [0, 0, 180]


#這裡單位是 m  大吸嘴+0.125
DH_table = np.array([[0,            0.345,  0.08,   math.pi / 2],
					 [0+math.pi / 2 , 0,  0.27,     0],
					 [0,             0,     0.09,    math.pi / 2],
					 [0,            0.295,  0,       -math.pi / 2],
					 [0,            0,      0,       math.pi / 2],
					 [0,       0.102+0.125, 0,          0]])


def save_txt(path, name, data, fmt='%f'):
    f = open(path + name, 'w')
    np.savetxt(f, data, fmt=fmt)
    f.close()

class hand_robot_env(object):
    # joint_bound
    degtorad = math.pi / 180
    state_dim = config.state_dim
    action_dim = config.action_dim
    SamplingTime = 0.01

    def __init__(self):
        print("---------- You are in hand_env 線上學習 ----------")

        if (config.vrep_show):
            """ ---- vrep initial ---- """
            self.my_robot = my_robot()
            self.my_robot.connection()
            print('vrep  : initial ~ ~ ')
        self.radtodeg = 180 / math.pi  # 弧度轉角度
        self.degtorad = math.pi / 180  # 角度轉弧度
        self.joint_cmd = np.zeros((6,),np.float)
        self.vs = np.zeros((6,),np.float)

        """ 存照片"""
        self.PATH = time.strftime('%m%d%H%M')
        creat_path('C:\\Users\\user\\Desktop\\yolo_picture\\exp1\\' + self.PATH)
        self.save_pic_time = 0

    def initial_yolo_kinect(self, kinect, yolo_coco):
        """ ---- kinect initial ---- """
        self.kinect = kinect
        """ ---- YOLO initial ---- """
        # self.yolo = yolo
        self.yolo_coco = yolo_coco

    def initial(self):
        self.my_robot.stop_sim()
        self.my_robot.start_sim()


    def reset(self):

        self.joint = [0, 0.524, -0.349, 0, -0.785, 0]
        if (config.vrep_show):
            self.my_robot.move_all_joint(self.joint)
        print('reset')



        pygame.init()
        self.screen = pygame.display.set_mode((300, 300))
        self.screen.fill((255, 255, 255))
        pygame.display.set_caption("YOLO + SAC")
        font = pygame.font.SysFont("cambriacambriamath", 20)
        self.text = font.render('Press : ', True, (0, 0, 0))
        self.text1 = font.render('Q : Bye!', True, (0, 0, 0))
        self.text2 = font.render('0 : success', True, (0, 0, 0))
        self.text3 = font.render('9 : fail', True, (0, 0, 0))



        return self.get_state()

    def get_state(self):
        """
        state : Depth image from kinect after yolo detect 64*64

        """
        """ ---- 得到彩色圖 ---- """
        color = self.kinect.Get_Color_Frame()
        self.color = color.copy()

        """ ---- 得到深度圖 ---- """
        self.mImg16bit, self.DepthImg = self.kinect.Get_Depth_Frame()
        """ ---- ROI ---- """
        ROI =  self.color[self.kinect.RoiOffset_Y:self.kinect.h_color-config.RoiOffset_Y_, self.kinect.RoiOffset_X:self.kinect.w_color-config.RoiOffset_X_]  # (880*1020*3)

        # ROI = self.color[280:1080 - 400, 950:1920 - 500]
        # ROI = color
        """ ---- YOLOv3偵測吹風機 ---- """
        self.Yolo_Det_frame, self.coordinate, self.cls, self.label, self.Width_and_Height = self.yolo_coco.detectFrame(ROI)  # 得到框的中心點
        cv2.imshow('Yolo_Det_frame', self.Yolo_Det_frame)
        cv2.waitKey(1)


        while not (self.Width_and_Height.any()):
            self.get_state()
            print('No Object !!! ')

        scale = 1

        # -----------------------影像的state-------------------

        """ --- 彩色圖四邊形座標 --- """
        self.color_center = np.array([int(self.coordinate[0][0] / scale) + self.kinect.RoiOffset_X, int(self.coordinate[0][1] / scale)+self.kinect.RoiOffset_Y])
        right_coor = np.array([int(self.color_center[0] - self.Width_and_Height[0][0] / 2),int(self.color_center[1] + self.Width_and_Height[0][1] / 2)])
        left_coor = np.array([int(self.color_center[0] + self.Width_and_Height[0][0] / 2), int(self.color_center[1] - self.Width_and_Height[0][1] / 2)])

        """ --- 深度圖四邊形座標 --- """
        depth_center = color_point_2_depth_point( self.kinect._kinect, self.kinect.DepthSpacePoint,  self.kinect._kinect._depth_frame_data,self.color_center)
        depth_rect_right = color_point_2_depth_point( self.kinect._kinect,  self.kinect.DepthSpacePoint,  self.kinect._kinect._depth_frame_data,right_coor)
        depth_rect_left = color_point_2_depth_point( self.kinect._kinect,  self.kinect.DepthSpacePoint,  self.kinect._kinect._depth_frame_data,left_coor)

        cv2.rectangle(self.DepthImg,  (depth_rect_right[0],depth_rect_right[1]) , (depth_rect_left[0],depth_rect_left[1]), (0, 255, 0), 2)
        # cv2.imshow("self.DepthImg",self.DepthImg)
        # cv2.waitKey(0)

        cy = np.array([int(depth_rect_left[1]), int(depth_rect_right[1])])
        cx = np.array([int(depth_rect_right[0]), int(depth_rect_left[0])])

        cy = np.clip(cy, 0, 424)
        cx = np.clip(cx, 0, 512)

        # ----- uint8 的 Depth
        self.Depth_ROI_for_show = self.DepthImg[cy[0]:cy[1], cx[0]:cx[1]]

        # ----- uint16 的 Depth
        mImg16bit = self.mImg16bit / self.kinect.Max_Distance
        mImg16bit = mImg16bit.reshape([self.kinect.h_depth, self.kinect.w_depth])
        Depth_ROI =mImg16bit[cy[0]:cy[1], cx[0]:cx[1]]
        # print('depth',Depth_ROI.shape)

        Depth_ROI = cv.resize(Depth_ROI,(64,64),interpolation = cv.INTER_CUBIC)
        self.depth_img_input = Depth_ROI[np.newaxis, ...]

        # -----------------------影像的state-------------------

        s = self.depth_img_input
        s = s[np.newaxis, ...]


        return s


    def step(self, action, record):

        """ action 是 彩色圖的 U,V"""

        done = False
        reward = 0
        success = 0
        cuboid_out = 0

        u1 = 1 + math.floor(action[0] * (self.Width_and_Height[0][0]/2 - 0.99))  #-------四捨五入(64 ~ 1)
        v1 = 1 + math.floor(action[1] * (self.Width_and_Height[0][1]/2 - 0.99))  #-------四捨五入(64 ~ 1)



        u = int(u1 + self.coordinate[0][0] + config.RoiOffset_X)
        v = int(v1 + self.coordinate[0][1] + config.RoiOffset_Y)



        if config.show_yolo:
            # 彩色圖上畫點
            # cv.circle(self.color_for_yolo, (u, v), 2, (255, 255, 0), -1)
            # yolo圖上畫點
            cv.circle( self.Yolo_Det_frame, (u-config.RoiOffset_X,v-config.RoiOffset_Y), 2 , (255, 255, 0), -1)
            # u,v 在深度圖的座標
            Depth_RL = color_point_2_depth_point(self.kinect._kinect, self.kinect.DepthSpacePoint,
                                                 self.kinect._kinect._depth_frame_data, (u, v))
            # 深度圖上畫點
            cv.circle(self.DepthImg, (Depth_RL[0],Depth_RL[1]), 2, (255, 255, 0), -1)

            cv2.imshow('RL Choose action', self.Yolo_Det_frame)
            cv2.imshow('RL Choose action Depth', self.DepthImg)
            cv2.imwrite('C:\\Users\\user\\Desktop\\yolo_picture\\exp1\\'+self.PATH +'\\'+'Origina'+str(self.save_pic_time)+'.png',self.color )  # 存彩色图片
            cv2.imwrite('C:\\Users\\user\\Desktop\\yolo_picture\\exp1\\'+self.PATH +'\\'+'rl_action_RGB'+str(self.save_pic_time)+'.png',self.Yolo_Det_frame)  # 存彩色图片
            cv2.imwrite('C:\\Users\\user\\Desktop\\yolo_picture\\exp1\\'+self.PATH +'\\'+'rl_action_Depth' + str(self.save_pic_time) + '.png',self.DepthImg)  # 存深度图片


        World_coor =  self.kinect.Color_Pixel_To_World_Coor( u, v, self.mImg16bit) # mm

        World_coor = World_coor*0.001
        World_coor[1] = World_coor[1] - 0.0135 #0.0135

        World_coor = np.squeeze(World_coor)

        tip_Jangle, flag, Singularplace = IK.InverseKinematics(finalpos, World_coor, DH_table)
        joint, num_of_sol = FindOptSol(tip_Jangle, self.joint, Singularplace)

        """ -------------------------------- 傳送資料 --------------------------------  """
        folder_data = 'C:/Users/user/Desktop/碩論/Robot_File/'
        joint = np.reshape(joint, (1, 6))
        file_name = ['JointCmd.txt', 'flag.txt']
        data_set = ['joint', '[flag]']
        for i in range(len(file_name)):
            save_txt(path=folder_data, name=file_name[i], data=eval(data_set[i]))
        print(joint)
        print('資料已傳送')
        """ -------------------------------- 傳送資料 --------------------------------  """


        if (config.vrep_show):
            self.my_robot.set_object_pos(self.my_robot.Dummy7,World_coor)
            print('move')
            self.my_robot.move_all_joint(joint)
            time.sleep(0.5)

        cv2.waitKey(0)
        """ -------------------------------- 手動輸入成功或失敗 --------------------------------  """

        while (True):
            key_pressed = pygame.key.get_pressed()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        sys.exit()
                        kinect.close_kinect()
                    if event.key == pygame.K_0:
                        success = 1
                        break
                    elif event.key == pygame.K_9:
                        success = 0
                        break
            x, y, y2, t = 0, 0, 30, 20
            self.screen.blit(self.text, (x, y))
            self.screen.blit(self.text1, (x, y2))
            self.screen.blit(self.text2, (x, y2 + t))
            self.screen.blit(self.text3, (x, y2 + 2 * t))
            pygame.display.update()
            break



        if (success >= 1):
            done = True
            print('lift done')


        reward = success

        s_ = self.get_state()
        self.save_pic_time = self.save_pic_time + 1

        return s_, reward, done




if __name__ == '__main__':
    pass


# self.my_robot.move_all_joint(self.joint)
##################### record data #####################
# error_record = np.reshape(error, (1, 6))
# # print(joint_out_record)
# path = './Trajectory/'
# name = 'error_record.txt'
# f = open(path + name, mode='a')
# np.savetxt(f, error_record, fmt='%f')
# f.close()
##################### record data #####################

