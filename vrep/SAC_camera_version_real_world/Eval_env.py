

"""
    robot_Eval_Env     特定物件拾取環境
    clean_table_Env    物件分類拾取環境
"""


import numpy as np
import os
import math
import time
import inverseKinematics as IK
import Kinematics as FK
from IK_FindOptSol import FindOptSol
import config
from yolo import *
from Kinect_new import Kinect
from mapper import *
from robot_vrep import my_robot


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


#這裡單位是 m   大吸嘴+0.125
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

class robot_Eval_Env(object):
    # joint_bound
    degtorad = math.pi / 180
    state_dim = config.state_dim
    action_dim = config.action_dim
    SamplingTime = 0.01

    def __init__(self):
        print("---------- You are in Eval_env ----------")
        self.radtodeg = 180 / math.pi  # 弧度轉角度
        self.degtorad = math.pi / 180  # 角度轉弧度

        self.joint_cmd = np.zeros((6,),np.float)
        self.vs = np.zeros((6,),np.float)

        """ 存照片"""
        self.PATH = time.strftime('%m%d%H%M')
        creat_path('C:\\Users\\user\\Desktop\\yolo_picture\\exp2\\' + self.PATH)
        self.save_pic_time = 0

        if (config.vrep_show):
            """ ---- Vrep initial ---- """
            self.my_robot = my_robot()
            self.my_robot.connection()

    def initial_yolo_kinect(self,kinect,yolo,yolo_coco):
        """ ---- kinect initial ---- """
        self.kinect = kinect
        """ ---- YOLO initial ---- """
        self.yolo = yolo
        self.yolo_coco = yolo_coco
        # self.yolo_open_image = yolo_open_image #這個先保留

    def initial(self):
        self.my_robot.stop_sim()
        self.my_robot.start_sim()

    def reset(self,want_boject,Yolo_Flag):
        # return state containing joint ,EFF ,target ,dis
        # robot to initial pos and random the target

        self.want_boject = want_boject
        self.joint = [0, 0.524, -0.349, 0, -0.785, 0]
        if (config.vrep_show):
            self.my_robot.move_all_joint(self.joint)
        self.Yolo_Flag = Yolo_Flag
        print('reset')


        return self.get_state()

    def get_state(self):
        """
        state : Depth image from kinect after yolo detect 64*64

        """

        """ --- 獲取彩色及深度圖片 --- """
        has_image = False
        while not (has_image):
            if self.kinect._kinect.has_new_color_frame():
                self.color = self.kinect.Get_Color_Frame()
                self.mImg16bit, self.DepthImg = self.kinect.Get_Depth_Frame()
                self.color_for_yolo = self.color.copy()
                # cv2.imshow('depth1123',self.DepthImg)
                # self.color_for_yolo =cv2.flip(self.color_for_yolo ,1)
                # cv2.imshow('color',  self.color_for_yolo )
                # cv2.imwrite('color_img.jpg',self.color_for_yolo)
                has_image = True
        # self.other_feature = np.zeros([64, 64], np.float)
        # # /////////////////////////////////////////////////////////////////////#
        # #                            YOLO Detect                              #
        # # /////////////////////////////////////////////////////////////////////#
        #
        """ --- 擷取ROI --- """
        ROI =  self.color_for_yolo[self.kinect.RoiOffset_Y:self.kinect.h_color-config.RoiOffset_Y_, self.kinect.RoiOffset_X:self.kinect.w_color-config.RoiOffset_X_]  # (880*1020*3)
        cv2.imshow('ROI',ROI)

        """ --- 看此物體是哪個 data set --- 0:cubic,1:coco,2:open image """
        if (self.Yolo_Flag)  == 0:
            self.Yolo_Det_frame, self.coordinate, self.cls,self.label,self.Width_and_Height = self.yolo.detectFrame(ROI)  # cubic偵測

        # cv2.imshow('Yolo_Det_frame', self.Yolo_Det_frame)

        elif  (self.Yolo_Flag)  == 1: # coco_data
            self.Yolo_Det_frame, self.coordinate, self.cls,self.label,self.Width_and_Height = self.yolo_coco.detectFrame(ROI)  # cooc偵測

        # cv2.imshow(" self.Yolo_Det_frame",  self.Yolo_Det_frame)
        # cv2.waitKey(0)
        # elif  (self.Yolo_Flag)  == 2: # open_image_data
        #     self.Yolo_Det_frame, self.coordinate, self.cls,self.label,self.Width_and_Height = self.yolo_open_image.detectFrame(ROI)  # open image偵測


        """ --- 確認YOLO有偵測到物體，沒有的話重新獲取狀態(s) --- """
        while not (self.Width_and_Height.any()):
            self.get_state()
            print('No Object !!! ')

        """ --- 尋找YOLO預測出的所有類別哪一個是我要的 --- """
        for i in range(len(self.cls)):
            if self.want_boject == self.cls[i]:
                self.index = i
                break

        scale = 1
        if config.show_yolo:
            color = (0, 0, 255)  # BGR
            # cv2.circle(self.Yolo_Det_frame, (self.coordinate[self.index][0], self.coordinate[self.index][1]), 2, color, -1)
            # cv2.circle(self.color_for_yolo, (self.coordinate[self.index][0]+config.RoiOffset_X, self.coordinate[self.index][1]+config.RoiOffset_Y), 2, color, -1)
            # cv2.imshow('color_img', self.color_img)
            # cv2.imshow('yolo',self.Yolo_Det_frame)

            # cv2.imwrite(config.yolo_Det_Img_path, np.array(self.Yolo_Det_frame))  # 储存检测结果图
            # cv2.imwrite(config.yolo_Det_Img_ROI_path_eval, np.array(self.Yolo_Det_frame))  # 储存检测结果图
            # cv2.imwrite(config.yolo_Det_Img_ORI_path_eval, np.array(self.color_for_yolo))  # 储存检测结果图 原圖



        # /////////////////////////////////////////////////////////////////////#
        #                            YOLO END                                 #
        # /////////////////////////////////////////////////////////////////////#

        # -----------------------影像的state-------------------

        """ --- 彩色圖四邊形座標 --- """
        self.color_center = np.array([int(self.coordinate[self.index][0] / scale) + self.kinect.RoiOffset_X,
                                      int(self.coordinate[self.index][1] / scale) + self.kinect.RoiOffset_Y])
        right_coor = np.array([int(self.color_center[0] - self.Width_and_Height[self.index][0] / 2),
                               int(self.color_center[1] + self.Width_and_Height[self.index][1] / 2)])
        left_coor = np.array([int(self.color_center[0] + self.Width_and_Height[self.index][0] / 2),
                              int(self.color_center[1] - self.Width_and_Height[self.index][1] / 2)])

        ########################################
        cy_color = np.array([int(left_coor[1]), int(right_coor[1])])
        cx_color = np.array([int(right_coor[0]), int(left_coor[0])])
        print('cor',cy_color,cx_color)

        color_ROI_for_show = self.color_for_yolo[cy_color[0]:cy_color[1], cx_color[0]:cx_color[1]]

        cv2.imshow('color',color_ROI_for_show)
        ########################################

        """ --- 深度圖四邊形座標 --- """
        depth_center = color_point_2_depth_point(self.kinect._kinect, self.kinect.DepthSpacePoint,
                                                 self.kinect._kinect._depth_frame_data, self.color_center)
        depth_rect_right = color_point_2_depth_point(self.kinect._kinect, self.kinect.DepthSpacePoint,
                                                     self.kinect._kinect._depth_frame_data, right_coor)
        depth_rect_left = color_point_2_depth_point(self.kinect._kinect, self.kinect.DepthSpacePoint,
                                                    self.kinect._kinect._depth_frame_data, left_coor)

        cv2.rectangle(self.DepthImg, (depth_rect_right[0], depth_rect_right[1]),
                      (depth_rect_left[0], depth_rect_left[1]), (0, 255, 0), 2)
        # cv2.imshow("self.DepthImg",self.DepthImg)
        # cv2.waitKey(0)

        cy = np.array([int(depth_rect_left[1]), int(depth_rect_right[1])])
        cx = np.array([int(depth_rect_right[0]), int(depth_rect_left[0])])

        cy = np.clip(cy, 0, 424)
        cx = np.clip(cx, 0, 512)

        self.Depth_ROI_for_show = self.DepthImg[cy[0]:cy[1], cx[0]:cx[1]]

       ########################################
        dis_far = 1
        dis_near = 0.01
        depth_scale = 1000

        self.Depth_ROI_for_show = self.Depth_ROI_for_show
        Depth_64_for_show = cv2.resize(self.Depth_ROI_for_show, (64, 64), interpolation=cv2.INTER_CUBIC)
        Depth_64_for_show = dis_near + (dis_far - dis_near) * Depth_64_for_show
        Depth_64_for_show = (Depth_64_for_show - np.min(Depth_64_for_show)) * 255 / (np.max(Depth_64_for_show) - np.min(Depth_64_for_show))  # 正規化 0~255
        Depth_64_for_show = Depth_64_for_show.astype(np.uint8)
        cv2.imshow("self.DepthImg", self.Depth_ROI_for_show )
        cv2.imshow("Depth_64_for_show", Depth_64_for_show)
        cv2.imwrite('depth.png', self.Depth_ROI_for_show)
        cv2.waitKey(0)

        ########################################

        # ----- uint16 的 Depth
        mImg16bit = self.mImg16bit / self.kinect.Max_Distance
        mImg16bit = mImg16bit.reshape([self.kinect.h_depth, self.kinect.w_depth])
        Depth_ROI =mImg16bit[cy[0]:cy[1], cx[0]:cx[1]]
        # print('depth',Depth_ROI.shape)

        Depth_ROI = cv2.resize(Depth_ROI,(64,64),interpolation = cv2.INTER_CUBIC)
        self.depth_img_input = Depth_ROI[np.newaxis, ...]

        # -----------------------影像的state-------------------


        s = self.depth_img_input
        s = s[np.newaxis, ...]
        #s.shape = [other_feature_dim]

        return s


    def step(self, action, record):

        #action 是 U,V

        joint_pos_out = np.zeros((6,),np.float)
        joint_cmd = np.zeros((6,), np.float)
        target_height= np.zeros((3,), np.float)


        done = False
        outbound = False
        reward = 0
        success = 0
        coutbound = 0
        cuboid_out = 0
        suction_value = 0

        u1 = 1 + math.floor(action[0] * (self.Width_and_Height[self.index][0]/2 - 0.99))  #-------四捨五入(64 ~ 1)
        v1 = 1 + math.floor(action[1] * (self.Width_and_Height[self.index][1]/2 - 0.99))  #-------四捨五入(64 ~ 1)


        u = int(u1 + self.coordinate[self.index][0] + config.RoiOffset_X)
        v = int(v1 + self.coordinate[self.index][1] + config.RoiOffset_Y)


        if config.show_yolo:
            cv2.circle(self.color_for_yolo, (u, v), 2, (255, 255, 0), -1)
            cv2.circle(self.Yolo_Det_frame, (u-config.RoiOffset_X, v-config.RoiOffset_Y), 3, (0, 0, 255), -1)

            Depth_RL = color_point_2_depth_point(self.kinect._kinect, self.kinect.DepthSpacePoint,
                                                        self.kinect._kinect._depth_frame_data, (u,v))
            cv2.circle(self.DepthImg, (Depth_RL[0],Depth_RL[1]), 2, (255, 255, 0), -1)
            # cv2.namedWindow('RL Choose action',cv2.WINDOW_NORMAL)
            # cv2.namedWindow('RL Choose action Depth', cv2.WINDOW_NORMAL)
            cv2.imshow('RL Choose action', self.Yolo_Det_frame)
            cv2.imshow('RL Choose action Depth', self.DepthImg)

            cv2.imwrite('C:\\Users\\user\\Desktop\\yolo_picture\\exp2\\'+self.PATH +'\\'+'ROI'+str(self.label[self.index])+str(self.save_pic_time)+'.png',self.color_for_yolo)  # 存图片
            cv2.imwrite('C:\\Users\\user\\Desktop\\yolo_picture\\exp2\\'+self.PATH +'\\'+'RL_C'+str(self.label[self.index])+str(self.save_pic_time)+'.png',self.Yolo_Det_frame)  # 存图片
            cv2.imwrite('C:\\Users\\user\\Desktop\\yolo_picture\\exp2\\'+self.PATH +'\\' +'RL_D'+str(self.label[self.index])+str(self.save_pic_time)+'.png', self.DepthImg)  # 存图片

            self.save_pic_time = self.save_pic_time + 1

        # self.mImg16bit = self.kinect.DepthDatafilter(self.mImg16bit)  # 深度濾波 5s (這會變慢建議不要)
        World_coor =  self.kinect.Color_Pixel_To_World_Coor( u, v, self.mImg16bit) # mm

        World_coor = World_coor*0.001
        World_coor[1] = World_coor[1] - 0.0155 #0.0135


        World_coor = np.squeeze(World_coor)

        tip_Jangle, flag, Singularplace = IK.InverseKinematics(finalpos, World_coor, DH_table)
        joint, num_of_sol = FindOptSol(tip_Jangle, self.joint, Singularplace)
        print("World_coor", World_coor)
        print("joint", joint)
        if (config.vrep_show):
            self.my_robot.set_object_pos(self.my_robot.Dummy7,World_coor)
            self.my_robot.move_all_joint(joint)
            # self.back_to_home()

        # if num_of_sol == 0:
        #     done = True
        #     reward = reward - 0.2

        s_ =self.get_state()

        return  joint,flag

    def back_to_home(self):

        joint = [-0.01142807, -0.56720771 , 0.03640932,  0.       ,  -1.03999794, -0.01142807]
        self.my_robot.move_all_joint(joint)
        time.sleep(1)



        # tip_Jangle, flag, Singularplace = IK.InverseKinematics(finalpos, Can_pos, DH_table)
        # joint, num_of_sol = FindOptSol(tip_Jangle, self.joint, Singularplace)
        joint = [-0.71737675 ,-0.81042242 , 0.41661738 , 0.         ,-1.17699128, -0.71737675]

        self.my_robot.move_all_joint(joint)
        time.sleep(5)

        home_pos = [0,0,0,0,0,0]
        self.my_robot.move_all_joint(home_pos)
        time.sleep(1)


    def close_kinect(self):
        self.kinect.close_kinect()

    def Color_Pixel_To_World_Coor_test(self, u, v,depth_z):

        """ ----- (u,v) is color space coor -----"""
        CameraPointC = np.zeros((4, 1), np.float32)
        pixel = np.array([u, v], np.int)
        depth_center = color_point_2_depth_point(self._kinect, self._kinect._DepthSpacePoint, self._kinect._depth_frame_data, pixel)
        CameraPointC[0] = (pixel[0] - config.Color_Intrinsic_Matrix[0][2]) * depth_z / config.Color_Intrinsic_Matrix[0][0]
        CameraPointC[1] = (pixel[1] - config.Color_Intrinsic_Matrix[1][2]) * depth_z / config.Color_Intrinsic_Matrix[1][1]
        CameraPointC[2] = depth_z
        CameraPointC[3] = 1
        """ ---- 以上是 pixel to camera frame ---- """

        World_coor = self.Eye_To_Hand.dot(CameraPointC)
        World_coor_ = np.array([World_coor[0], World_coor[1], World_coor[2]])

        return  World_coor_

class clean_table_Env(object):
    # joint_bound
    degtorad = math.pi / 180
    state_dim = config.state_dim
    action_dim = config.action_dim
    SamplingTime = 0.01

    def __init__(self):
        print("---------- You are in Eval_env ----------")
        self.radtodeg = 180 / math.pi  # 弧度轉角度
        self.degtorad = math.pi / 180  # 角度轉弧度

        self.joint_cmd = np.zeros((6,),np.float)
        self.vs = np.zeros((6,),np.float)

        """ 存照片"""
        self.PATH = time.strftime('%m%d%H%M')
        creat_path('C:\\Users\\user\\Desktop\\yolo_picture\\exp3\\' + self.PATH)
        self.save_pic_time = 0

        if (config.vrep_show):
            """ ---- Vrep initial ---- """
            self.my_robot = my_robot()
            self.my_robot.connection()

    def initial_yolo_kinect(self,kinect,yolo,yolo_coco):
        """ ---- kinect initial ---- """
        self.kinect = kinect
        """ ---- YOLO initial ---- """
        self.yolo = yolo
        self.yolo_coco = yolo_coco

    def initial(self):
        self.my_robot.stop_sim()
        self.my_robot.start_sim()

    def reset(self,want_boject,color,mImg16bit,DepthImg,Yolo_Det_frame,coordinate,cls,label,Width_and_Height):
        """
            傳入當前影像資訊
        """
        self.mImg16bit = mImg16bit
        self.DepthImg = DepthImg
        self.color_for_yolo = color
        self.Yolo_Det_frame = Yolo_Det_frame
        self.coordinate = coordinate
        self.cls = cls
        self.label = label
        self.Width_and_Height = Width_and_Height
        self.index = want_boject

        # robot 初始位置
        self.joint = [0, 0.524, -0.349, 0, -0.785, 0]

        if (config.vrep_show):
            self.my_robot.move_all_joint(self.joint)



        return self.get_state()

    def get_state(self):
        """
        state : Depth image from kinect after yolo detect 64*64

        """

        while not (self.Width_and_Height.any()):
            self.get_state()
            print('No Object !!! ')

        scale = 1

        # -----------------------影像的state-------------------

        """ --- 彩色圖四邊形座標 --- """
        self.color_center = np.array([int(self.coordinate[self.index][0] / scale) + self.kinect.RoiOffset_X,
                                      int(self.coordinate[self.index][1] / scale) + self.kinect.RoiOffset_Y])
        right_coor = np.array([int(self.color_center[0] - self.Width_and_Height[self.index][0] / 2),
                               int(self.color_center[1] + self.Width_and_Height[self.index][1] / 2)])
        left_coor = np.array([int(self.color_center[0] + self.Width_and_Height[self.index][0] / 2),
                              int(self.color_center[1] - self.Width_and_Height[self.index][1] / 2)])

        """ --- 深度圖四邊形座標 --- """
        depth_center = color_point_2_depth_point(self.kinect._kinect, self.kinect.DepthSpacePoint,
                                                 self.kinect._kinect._depth_frame_data, self.color_center)
        depth_rect_right = color_point_2_depth_point(self.kinect._kinect, self.kinect.DepthSpacePoint,
                                                     self.kinect._kinect._depth_frame_data, right_coor)
        depth_rect_left = color_point_2_depth_point(self.kinect._kinect, self.kinect.DepthSpacePoint,
                                                    self.kinect._kinect._depth_frame_data, left_coor)

        cv2.rectangle(self.DepthImg, (depth_rect_right[0], depth_rect_right[1]),
                      (depth_rect_left[0], depth_rect_left[1]), (0, 255, 0), 2)

        cy = np.array([int(depth_rect_left[1]), int(depth_rect_right[1])])
        cx = np.array([int(depth_rect_right[0]), int(depth_rect_left[0])])

        cy = np.clip(cy, 0, 424)
        cx = np.clip(cx, 0, 512)

        self.Depth_ROI_for_show = self.DepthImg[cy[0]:cy[1], cx[0]:cx[1]]

        # ----- uint16 的 Depth
        mImg16bit = self.mImg16bit / self.kinect.Max_Distance
        mImg16bit = mImg16bit.reshape([self.kinect.h_depth, self.kinect.w_depth])
        Depth_ROI =mImg16bit[cy[0]:cy[1], cx[0]:cx[1]]
        # print('depth',Depth_ROI.shape)

        Depth_ROI = cv2.resize(Depth_ROI,(64,64),interpolation = cv2.INTER_CUBIC)
        self.depth_img_input = Depth_ROI[np.newaxis, ...]

        # -----------------------影像的state-------------------


        s = self.depth_img_input
        s = s[np.newaxis, ...]
        #s.shape = [other_feature_dim]

        return s


    def step(self, action):

        #action 是 U,V

        u1 = 1 + math.floor(action[0] * (self.Width_and_Height[self.index][0]/2 - 0.99))  #-------四捨五入(64 ~ 1)
        v1 = 1 + math.floor(action[1] * (self.Width_and_Height[self.index][1]/2 - 0.99))  #-------四捨五入(64 ~ 1)


        u = int(u1 + self.coordinate[self.index][0] + config.RoiOffset_X)
        v = int(v1 + self.coordinate[self.index][1] + config.RoiOffset_Y)


        if config.show_yolo:
            cv2.circle(self.color_for_yolo, (u, v), 2, (255, 255, 0), -1)
            cv2.circle(self.Yolo_Det_frame, (u-config.RoiOffset_X, v-config.RoiOffset_Y), 2, (255, 255, 0), -1)

            Depth_RL = color_point_2_depth_point(self.kinect._kinect, self.kinect.DepthSpacePoint,
                                                        self.kinect._kinect._depth_frame_data, (u,v))
            cv2.circle(self.DepthImg, (Depth_RL[0],Depth_RL[1]), 2, (255, 255, 0), -1)

            cv2.imshow('RL Choose action', self.Yolo_Det_frame)
            cv2.imshow('RL Choose action Depth', self.DepthImg)

            cv2.imwrite('C:\\Users\\user\\Desktop\\yolo_picture\\exp3\\'+self.PATH +'\\'+'ROI'+str(self.save_pic_time)+'.png',self.color_for_yolo)  # 存图片
            cv2.imwrite('C:\\Users\\user\\Desktop\\yolo_picture\\exp3\\'+self.PATH +'\\'+'RL_C'+str(self.save_pic_time)+'.png',self.Yolo_Det_frame)  # 存图片
            cv2.imwrite('C:\\Users\\user\\Desktop\\yolo_picture\\exp3\\'+self.PATH +'\\' +'RL_D'+str(self.save_pic_time)+'.png', self.DepthImg)  # 存图片

            self.save_pic_time = self.save_pic_time + 1


        # self.mImg16bit = self.kinect.DepthDatafilter(self.mImg16bit)  # 深度濾波 5s
        World_coor =  self.kinect.Color_Pixel_To_World_Coor( u, v, self.mImg16bit) # mm


        World_coor = World_coor*0.001
        World_coor[1] = World_coor[1] - 0.0155 #0.0135

        World_coor = np.squeeze(World_coor)

        print('World_coor',World_coor)
        tip_Jangle, flag, Singularplace = IK.InverseKinematics(finalpos, World_coor, DH_table)
        joint, num_of_sol = FindOptSol(tip_Jangle, self.joint, Singularplace)



        if (config.vrep_show):
            self.my_robot.set_object_pos(self.my_robot.Dummy7,World_coor)
            print('move')
            self.my_robot.move_all_joint(joint)
            time.sleep(0.5)

        s_ =self.get_state()

        return joint,flag



    def close_kinect(self):
        self.kinect.close_kinect()




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




