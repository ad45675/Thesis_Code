from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime

from yolo import *

import sys
import numpy as np
import time
import cv2
import math
import copy
import ctypes
from mapper import *
import config
import pygame


def save_txt(path, name, data, fmt='%f', mode='w'):
    f = open(path + name, mode)
    np.savetxt(f, data, fmt=fmt)
    f.close()


def path_exsit(path):
    if os.path.exists(path):
        return True
    else:
        return False


def creat_path(path):
    if path_exsit(path=path):
        print(path + ' exist')
    else:
        os.makedirs(path)


PATH = time.strftime('%m%d%H%M')


class Kinect(object):
    def __init__(self):
        #PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Infrared)
        self._kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth)
        self.w_color = self._kinect.color_frame_desc.Width
        self.h_color = self._kinect.color_frame_desc.Height
        self.w_depth = self._kinect.depth_frame_desc.Width
        self.h_depth = self._kinect.depth_frame_desc.Height
        self.w_infrared = self._kinect.infrared_frame_desc.Width
        self.h_infrared = self._kinect.infrared_frame_desc.Height
        self.Max_Distance = self._kinect.DepthMaxDistance
        self.Min_Distance = self._kinect.DepthMinDistance
        self.DepthSpacePoint = _DepthSpacePoint
        print("Color res : ", self.w_color, "x", self.h_color)
        print("Depth res : ", self.w_depth, "x", self.h_depth)
        print("Infrared res : ", self.w_infrared, "x", self.h_infrared)
        print('Max_Distance_Depth ',self.Max_Distance ,'Min_Distance_Depth ',self.Min_Distance)

        self.csp_type = _ColorSpacePoint * np.int(1920 * 1080)
        self.csp = ctypes.cast(self.csp_type(), ctypes.POINTER(_DepthSpacePoint))

        self.depth_ori = None
        self.infrared_frame = None
        self.color_frame = None
        self.color = None
        self.depth = None
        self.depth_draw = None
        self.color_draw = None
        self.infrared = None
        self.first_time = True

        self.RoiOffset_X = config.RoiOffset_X
        self.RoiOffset_Y = config.RoiOffset_Y

        self.Color_Intrinsic_Matrix =config.Color_Intrinsic_Matrix
        self.Discoeff = config.Discoeff

        self.Depth_Intrinsic_Matrix = config.Depth_Intrinsic_Matrix

        self.Hand_To_Eye = config.Hand_To_Eye
        self.Eye_To_Hand = config.Eye_To_Hand

    def close_kinect(self):
        print("Kinect Bye Bye")
        return self._kinect.close()



    def Get_Depth_intrinsics(self, kinect):
        # 3 * 3 Matrix
        while (1):
            if self._kinect.has_new_depth_frame():
                intrinsics_matrix = kinect._mapper.GetDepthCameraIntrinsics()
                print('FocalLengthX : ', intrinsics_matrix.FocalLengthX)
                print('FocalLengthY : ', intrinsics_matrix.FocalLengthY)
                print('PrincipalPointX : ', intrinsics_matrix.PrincipalPointX)
                print('PrincipalPointY : ', intrinsics_matrix.PrincipalPointY)
                print('RadialDistortionFourthOrder : ', intrinsics_matrix.RadialDistortionFourthOrder)
                print('RadialDistortionSecondOrder : ', intrinsics_matrix.RadialDistortionSecondOrder)
                print('RadialDistortionSixthOrder : ', intrinsics_matrix.RadialDistortionSixthOrder)
                self.Depth_Intrinsic_Matrix = np.array([
                    [intrinsics_matrix.FocalLengthX, 0, intrinsics_matrix.PrincipalPointX],
                    [0, intrinsics_matrix.FocalLengthY, intrinsics_matrix.PrincipalPointY],
                    [0, 0, 1]])

                print('Depth Intrinsic Matrix : ', '\n', self.Depth_Intrinsic_Matrix)
                path = "D:\\Kinect_Program_yao\\Data_python"
                creat_path(path)
                save_txt(path, '.\\Depth_Intrinsic_Matrix.txt', self.Depth_Intrinsic_Matrix, fmt='%1.6f')

                break

    def Save_Color_intrinsics(self):
        # self.Color_Intrinsic_Matrix = np.array([
        #         [1.0454801485926105e+03 ,0.                     ,9.3496788610430747e+02 ],
        #         [0.                     ,1.0424962581852583e+03 ,5.3487433803128465e+02 ],
        #         [0.                     ,0.                     ,1.]
        # ])
        # self.Discoeff = np.array ([ 2.1664844254275058e-02, 3.6657109783434116e-02, -7.0038831777495102e-03, -4.0304747448493940e-03, -2.1132198547319650e-01])

        self.Discoeff = np.reshape(self.Discoeff, (1, len(self.Discoeff)))
        path = "D:\\Kinect_Program_yao\\Data_python"
        creat_path(path)
        save_txt(path, '.\\Color_Intrinsic_Matrix.txt', self.Color_Intrinsic_Matrix, fmt='%1.6f')
        save_txt(path, '.\\Discoeff.txt', self.Discoeff, fmt='%1.6f')

    def Get_Color_Frame(self):
        # if self._kinect.has_new_color_frame():
        frame = self._kinect.get_last_color_frame()
        gbra = frame.reshape([self.h_color, self.w_color, 4])
        color = gbra[:, :, 0:3]  # [:, ::-1, :](左右相反)
        return color

    def Get_Depth_Frame(self):

        # if self._kinect.has_new_depth_frame():
        mImg16bit = self._kinect.get_last_depth_frame()
        """ 深度濾波 """
        # mImg16bit_filter = self.DepthDatafilter(mImg16bit)
        # 轉為圖像排列
        image_depth_all = mImg16bit.reshape([self.h_depth, self.w_depth])
        # 轉為( n, m, 1) 形事
        image_depth_all = image_depth_all.reshape([self.h_depth, self.w_depth, 1])
        #--------------------------------------------------------------------------------
        image_depth_all[image_depth_all >= self.Max_Distance] = 0
        image_depth_all[image_depth_all <= self.Min_Distance] = 0
        image_depth_all = np.uint8((image_depth_all/self.Max_Distance) * 255)
        # cv2.imshow("image_depth_all",image_depth_all_test)
        # cv2.waitKey(1)
        # --------------------------------------------------------------------------------

        DepthImg = np.squeeze(image_depth_all)

        return mImg16bit, DepthImg

    def DepthDatafilter(self, mImg16bit):
        """
        沒必要就不要用，用了程式會變很慢
        """
        mImg16bit_filter = np.zeros(424 * 512, dtype=np.uint16)
        innerBandThreshold = 3
        outerBandThreshold = 7
        nRows = 424
        nCols = 512
        """ 第一次濾波 """
        for i in range(nRows):
            for j in range(nCols):
                Nowindex = i * self.w_depth + j
                # print("i : ",i,"w_depth : ",self.w_depth,"j : ",j,"index : ",Nowindex)
                if (mImg16bit[Nowindex] > self.Max_Distance or mImg16bit[Nowindex] < self.Min_Distance):
                    mImg16bit[Nowindex] = 0
        """ 第二次濾波 """
        for i in range(nRows):
            for j in range(nCols):
                Nowindex = i * self.w_depth + j
                # print("i : ",i,"w_depth : ",self.w_depth,"j : ",j,"index : ",Nowindex)
                """ 深度值為0的像素為候選像素 """
                if (mImg16bit[Nowindex] == 0):
                    """filtercollection用来计算滤波器內每個深度值對應的頻度，在後面我們將通過這個數值來確定給候選像塑一個甚麼深度值。"""
                    filterCollection = np.zeros((24, 2), dtype=np.uint32)

                    """ 內外層框妹非0巷速數量計數器, 在後面用來確定候選像素是否濾波"""
                    innerBandCount = 0
                    outerBandCount = 0
                    """
                    以下循環將會以候選像素為中心的 5*5 像素陣列進行遍歷. 在此定義了兩個邊界. 如果在這個陣列的像素為非零，那就記錄此深度值
                    並將其所在邊界計數器+1 ， 如果計數器超過設定的閥值， 那我們將取濾波器內統計的深度值的眾數(頻度最高的那個深度值)應用於候選像素上
                    """
                    for yi in range(-2, 3):
                        for xi in range(-2, 3):
                            """ 我們不要 xi !=0 or yi != 0 的情況，因為此時操作的就是候選像素"""
                            if (xi != 0 or yi != 0):
                                """ 確定操作像素在深度圖的位置"""
                                xSearch = j + xi
                                ySearch = i + yi
                                """ 檢查操作像素的位置是否超過了圖像邊界(候選像素在圖的邊緣)"""
                                if (
                                        xSearch >= 0 and xSearch < self.w_depth and ySearch >= 0 and ySearch < self.h_depth):
                                    Searchindex = xSearch + (ySearch * self.w_depth)
                                    """  我们只要非零向量 """
                                    if (mImg16bit[Searchindex] != 0):
                                        """ 計算每個深度值的頻度"""
                                        for k in range(24):
                                            if (filterCollection[k][0] == mImg16bit[Searchindex]):
                                                """
                                                如果在 filter collection 已經紀錄過這個深度，將此深度對應值加1
                                                """
                                                filterCollection[k][1] += 1
                                                break
                                            elif (filterCollection[k][0] == 0):
                                                """
                                                如果filter collection中沒有記錄這個深度，那麼紀錄.
                                                """
                                                filterCollection[k][0] = mImg16bit[Searchindex]
                                                filterCollection[k][1] += 1
                                                break
                                        """ 確定是內外哪個邊界內的像素不為零，對應計數器加一"""
                                        if (yi != 2 and yi != -2 and xi != 2 and xi != -2):
                                            innerBandCount += 1
                                        else:
                                            outerBandCount += 1
                    """
                    判斷計數器是否超過閥值，如果任意層內非零像素的數目超過了閥值就要將所有非零像素深度值對應的統計眾數
                    """
                    if (innerBandCount >= innerBandThreshold or outerBandCount >= outerBandThreshold):
                        frequency = 0
                        depth = 0
                        """ 這個循環將統計所有非零像素深度值對應的眾數"""
                        for k in range(24):
                            """ 當沒有記錄深度值時 (無非零深度值的像素)"""
                            if (filterCollection[k][0] == 0):
                                break
                            elif (filterCollection[k][1] > frequency):
                                depth = filterCollection[k][0]
                                frequency = filterCollection[k][1]
                        mImg16bit_filter[Nowindex] = depth
                    else:
                        mImg16bit_filter[Nowindex] = 0
                        """非0向量大於閥值"""
                else:
                    """ 如果像素的深度值不為0，保持原深度"""
                    mImg16bit_filter[Nowindex] = mImg16bit[Nowindex]
        return mImg16bit_filter
    def Color_Pixel_To_World_Coor(self,u,v,mImg16bit):

        """ ----- (u,v) is color space coor -----"""
        CameraPointC = np.zeros((4,1),np.float32)
        pixel = np.array([u,v],np.int)
        depth_center = color_point_2_depth_point(self._kinect, _DepthSpacePoint, self._kinect._depth_frame_data,pixel)
        depth_z = depth_space_2_world_depth(mImg16bit, depth_center[0], depth_center[1])
        CameraPointC[0] = -(pixel[0] - self.Color_Intrinsic_Matrix[0][2]) * depth_z / self.Color_Intrinsic_Matrix[0][0]
        CameraPointC[1] = (pixel[1] - self.Color_Intrinsic_Matrix[1][2]) * depth_z / self.Color_Intrinsic_Matrix[1][1]
        CameraPointC[2] = depth_z
        CameraPointC[3] = 1
        # print('depth',depth_z)
        """ ---- 以上是 pixel to camera frame(mm) ---- """

        World_coor = self.Eye_To_Hand.dot(CameraPointC)
        World_coor_ = np.array([World_coor[0],World_coor[1],World_coor[2]])

        return World_coor_

    def Camera_To_World_Coor(self,CameraPointC ):

        World_coor = self.Eye_To_Hand.dot(CameraPointC)

        return World_coor




    def Detect_Red_Point(self,color,mImg16bit,DepthImg):

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        RedPointC = np.zeros((3, 1), np.int)
        RedPointDC = np.zeros((3, 1), np.int)

        n = 0

        color_draw = color.copy()

        ROI = color_draw[self.RoiOffset_Y:self.h_color, self.RoiOffset_X:self.w_color]  # (880*1020*3)

        lower1 = (0, 100, 120)
        upper1 = (10, 255, 255)
        lower2 = (170, 100, 120)
        upper2 = (180, 255, 255)

        scale = 1
        hsv = cv2.cvtColor(np.asarray(ROI, dtype=np.uint8), cv2.COLOR_BGR2HSV)

        r1 = cv2.inRange(hsv, lower1, upper1)
        r2 = cv2.inRange(hsv, lower2, upper2)
        mask = r1 + r2

        dst = cv2.bitwise_and(ROI, ROI, mask=mask)

        dst = cv2.morphologyEx(dst, cv2.MORPH_BLACKHAT, kernel)
        img_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)


        # Find the contour of masked shapes
        contours = cv2.findContours(img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = contours[0]

        center = None

        # if there is a masked object
        if len(contours) > 0:
            # largest contour
            for i in range(len(contours)):
                rect_External = cv2.boundingRect(contours[i])
                area = rect_External[2] * rect_External[3]

                if (area > 0):
                    ((x, y), radius) = cv2.minEnclosingCircle(contours[i])
                    if (radius > 0):
                        cv2.circle(ROI, (int(x / scale), int(y / scale)), int(radius), (0, 255, 255), 3, 8, 0)
                        # cv2.imshow('roi',ROI)
                        # cv2.waitKey(0)
                        """ -------------------- 畫彩色圖 -------------------- """
                        cv2.circle(color_draw, (int(x / scale) + self.RoiOffset_X, int(y / scale) + self.RoiOffset_Y),int(radius), (0, 255, 255), 3, 8, 0)

                        """---------Map color image to gray & retrieve red point---------"""

                        color_center = np.array([int(x / scale) + self.RoiOffset_X, int(y / scale) + self.RoiOffset_Y])
                        depth_center = color_point_2_depth_point(self._kinect, _DepthSpacePoint, self._kinect._depth_frame_data,color_center)
                        # print('_DepthSpacePoint',_DepthSpacePoint)
                        # print('self._kinect._depth_frame_data', self._kinect._depth_frame_data)

                        depth_radius = int(radius * 424 / 1080)
                        """ -------------------- 畫深度圖 -------------------- """
                        cv2.circle(DepthImg, (depth_center[0], depth_center[1]), depth_radius, (0, 255, 255), 3, 8, 0)

                        """ -------------------- 座標計算 -------------------- """
                        mImg16bit = self.DepthDatafilter(mImg16bit)  # 深度濾波
                        depth_z = depth_space_2_world_depth( mImg16bit, depth_center[0], depth_center[1])

                        """ -------------------- 彩色座標 -------------------- """
                        RedPointC[0] = (color_center[0] - self.Color_Intrinsic_Matrix[0][2]) * depth_z / self.Color_Intrinsic_Matrix[0][0]
                        RedPointC[1] = (color_center[1] - self.Color_Intrinsic_Matrix[1][2]) * depth_z / self.Color_Intrinsic_Matrix[1][1]
                        RedPointC[2] = depth_z
                        """ -------------------- 深度座標 -------------------- """
                        RedPointDC[0] = (depth_center[0] - self.Depth_Intrinsic_Matrix[0][2]) * depth_z / self.Depth_Intrinsic_Matrix[0][0]
                        RedPointDC[1] = (depth_center[1] - self.Depth_Intrinsic_Matrix[1][2]) * depth_z / self.Depth_Intrinsic_Matrix[1][1]
                        RedPointDC[2] = depth_z
        else:
            RedPointC = [0,0,0]
            RedPointDC = [0,0,0]


        return RedPointC, RedPointDC,color_draw,DepthImg

    def YOLO3_Detect(self,yolo,color):
        yolo_Img_path = "imgTemp\\frame.jpg"
        yolo_Dep_path = "imgTemp\\Dep_frame.jpg"
        yolo_Det_Img_path = 'imgTemp\\Det_frame.jpg'

        color_draw = color.copy()

        ROI = color_draw[self.RoiOffset_Y:self.h_color, self.RoiOffset_X:self.w_color]  # (880*1020*3)
        Yolo_Det_frame, coordinate, cls, label, Width_and_Height = yolo.detectFrame(ROI)  # 得到框的中心點
        color = (0, 0, 255)  # BGR
        for i in range(len(label)):
            cv2.circle(Yolo_Det_frame, (coordinate[i][0], coordinate[i][1]), 3, color, -1)
        cv2.imwrite(yolo_Det_Img_path, np.array(Yolo_Det_frame))  # 储存检测结果图


        return Yolo_Det_frame, coordinate, cls, label




if __name__ == '__main__':
    # SavePic = False
    # text = []
    # pygame.init()
    # NumOfSavePic = 0
    # screen = pygame.display.set_mode((300, 300))
    # screen.fill((255, 255, 255))
    # pygame.display.set_caption("record red point")
    # font =  pygame.font.SysFont("cambriacambriamath", 20)
    #
    # text = font.render('Press : ', True, (0, 0, 0))
    # text1 = font.render('1 : Detect Red Point!', True, (0, 0, 0))
    # text2 = font.render('2 : Save Image', True, (0, 0, 0))
    # text3 = font.render('3 : Yolo Detect', True, (0, 0, 0))
    # text4 = font.render('4 : .........', True, (0, 0, 0))
    #
    #
    # text9 = font.render('9 : Close Kinect', True, (0, 0, 0))
    # textq = font.render('q : Bye', True, (0, 0, 0))
    #
    #
    # # # # 循環事件，按住一個鍵可以持續移動
    # pygame.key.set_repeat(200, 50)
    #
    # Kinect = Kinect()
    # yolo = YOLOV3()
    #
    # while True:
    #     key_pressed = pygame.key.get_pressed()
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             sys.exit()
    #         if event.type == pygame.KEYDOWN:
    #             if event.key == pygame.K_q:
    #                 sys.exit()
    #             if event.key == pygame.K_1:
    #                 RedPointC, RedPointDC, Red_Point_color, Red_Point_DepthImg = Kinect.Detect_Red_Point(color,mImg16bit,DepthImg)
    #                 print('RedPoint X : %3.f Y: %3.f Z: %3.f DX: %3.f DY: %3.f DZ: %3.f' % (RedPointC[0], RedPointC[1], RedPointC[2], RedPointDC[0], RedPointDC[1], RedPointDC[2]))
    #                 SavePic = False
    #                 print("Just  Detect Red Point")
    #                 Red_Point_color = cv2.resize(Red_Point_color, (int(0.5 * Red_Point_color.shape[1]), int(0.5 * Red_Point_color.shape[0])),interpolation=cv2.INTER_CUBIC)
    #                 cv2.imshow('contours Image2', Red_Point_color)
    #                 cv2.imshow('contours Image', Red_Point_DepthImg)
    #                 cv2.waitKey(1)
    #             elif event.key == pygame.K_2:
    #                 SavePic = True
    #                 if (SavePic):
    #                     creat_path('./Pic/' + PATH + '/color')
    #                     cv2.imwrite('./Pic/' + PATH + '/color/' + str(int(NumOfSavePic)) + '.jpg', color)
    #                     red_point_path = "C:\\Users\\user\\Desktop\\pykinect\\Data_python"
    #                     creat_path(red_point_path)
    #                     RedPointC_save = np.reshape(RedPointC, (1, 3))
    #                     RedPointDC_save = np.reshape(RedPointDC, (1, 3))
    #                     save_txt(red_point_path, '.\\red_point.txt', RedPointC_save, fmt='%f', mode='a')
    #                     save_txt(red_point_path, '.\\red_pointD.txt', RedPointDC_save, fmt='%f', mode='a')
    #                     NumOfSavePic += 1
    #                     SavePic = False
    #                     print("save picture", NumOfSavePic)
    #             elif event.key == pygame.K_3:
    #
    #                 Yolo_frame, center, cls, object_name = Kinect.YOLO3_Detect(yolo,color)
    #                 print(len(object_name) ,'object','\n')
    #                 for i in range(len(object_name)):
    #                     print(i+1,':', object_name[i])
    #                     print('-----------------------')
    #                 cv2.imshow("Yolo_Det_frame", Yolo_frame)
    #                 cv2.waitKey(1)
    #
    #             elif event.key == pygame.K_4:
    #                 pass
    #
    #             elif event.key == pygame.K_9:
    #                 Kinect.close_kinect()
    #                 sys.exit()
    #
    #
    #     x,y,y2,t = 0,0,30,20
    #     screen.blit(text, (x, y))
    #     screen.blit(text1, (x, y2))
    #     screen.blit(text2, (x, y2 + t))
    #     screen.blit(text3, (x, y2 + 2 * t))
    #     screen.blit(text4, (x, y2 + 3 * t))
    #
    #     screen.blit(text9, (x, y2 + 6 * t))
    #     screen.blit(textq, (x, y2 + 7 * t))
    #
    #     pygame.display.update()
    #     color = Kinect.Get_Color_Frame()
    #     mImg16bit,DepthImg = Kinect.Get_Depth_Frame()
    #
    #
    #     color_show = cv2.resize(color, (int(0.5 * color.shape[1]), int(0.5 * color.shape[0])),interpolation=cv2.INTER_CUBIC)
    #     cv2.imshow('color Image', color_show)
    #     cv2.imshow('DepthImg Image', DepthImg)
    #     cv2.waitKey(10)
    pass