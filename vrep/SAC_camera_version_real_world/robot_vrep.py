"""
yaooooo
this is vrep connect code
update time:11/25

"""

import numpy as np
import math
import sim as vrep
import time
import config
from Rot2RPY import Rot2RPY,euler2mat,Rot2RPY_version2
import cv2 as cv
import PIL.Image as Image
import os
# 配置關節資訊
jointNum = 6
baseName = 'my_robot'
jointName = 'joint'

# V-REP data transmission modes:
WAIT = vrep.simx_opmode_oneshot_wait
ONESHOT = vrep.simx_opmode_oneshot  # 非阻塞式,只想給vrep發送指令
STREAMING = vrep.simx_opmode_streaming  # 數據流模式
BUFFER = vrep.simx_opmode_buffer  # 非阻塞式
BLOCKING = vrep.simx_opmode_blocking  # 阻塞式必須等待從vrep返回信息


class my_robot(object):
    def __init__(self):
        self.radtodeg = 180 / math.pi  # 弧度轉角度
        self.degtorad = math.pi / 180  # 角度轉弧度
        self.time_step = 0.5  # sampling time
        self.joint_pos = np.zeros((jointNum,), np.float)
        self.joint_handle = np.zeros((jointNum,), np.int)
        self.plane = 0
        self.work_space = []
        self.EEF = []
        # self.initial_joint = [0.0, 0.524, -0.349, 0, -0.785, 0]  # 這是弧度,vrep裡是角度
        self.initial_joint2 = [0, 0, 0.2, 0, 0, 0]  # my_robot_initial
        self.initial_joint = [0.0, 0, 0, 0, 0, 0]  # my_robot_initial
        self.suction = "suctionPad_active"
        self.object_height = 0
        self.objects = None
        self.object_pos = None
        self.Cuboid_pos = []
        self.action_bound = [1, -1]
#------------------------camera info------------------------#
        self.radtodeg = 180 / math.pi  # 弧度轉角度
        self.degtorad = math.pi / 180  # 角度轉弧度
        self.width = config.resolutionX
        self.height = config.resolutionY
        self.theta = config.theta
        self.dis_far = config.dis_far
        self.dis_near = config.dis_near
        self.depth_scale = config.depth_scale
        self.save_image_path = config.SAVE_IMAGE_PATH

    def connection(self):

        vrep.simxFinish(-1)  # just in case, close all opened connections
        self.clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)  # Connect to CoppeliaSim
        # clientID stores the ID assingned by coppeliaSim, if the connection failed it will be assigned -1
        if self.clientID != -1:
            print('Connected to remote API server')
        else:
            time.sleep(0.2)
            print('fail to connect!!')
        self.show_msg('Python: Hello')
        time.sleep(0.5)

        # -------Setup the simulation
        vrep.simxSetFloatingParameter(self.clientID,
                                      vrep.sim_floatparam_simulation_time_step,
                                      self.time_step,  # specify a simulation time step
                                      ONESHOT)

        #vrep.simxSynchronous(self.clientID, True)  # if we need to be syncronous

    def disconnect(self):
        self.show_msg('Python: byebye')
        time.sleep(0.2)
        self.stop_sim()

    def stop_sim(self):
        vrep.simxStopSimulation(self.clientID, ONESHOT)

    def start_sim(self):
        # 讀取 robot id
        self.read_object_id()

        # 開始模擬
        vrep.simxStartSimulation(self.clientID, ONESHOT)

        #while vrep.simxGetConnectionId(self.clientID) != -1:
        # 讓模擬先走一步
        vrep.simxSynchronousTrigger(self.clientID)
        # 暫停溝通等等一次發送
        vrep.simxPauseCommunication(self.clientID, True)

        for i in range(jointNum):
            vrep.simxSetJointTargetPosition(self.clientID, self.joint_handle[i], self.initial_joint[i], ONESHOT)

        vrep.simxPauseCommunication(self.clientID, False)

        # self.random_object()
        #vrep.simxSynchronousTrigger(self.clientID)  # 进行下一步
        vrep.simxGetPingTime(self.clientID)  # 使得该仿真步走完

        self.suction_enable = False

    # def set_object_to_position(self,pos):
    #     vrep.simxSetObjectPosition(self.clientID, self.Cuboid, -1, pos, ONESHOT)

    def set_object_pos(self,handle,target_pos):
        vrep.simxSetObjectPosition(self.clientID, handle, -1, target_pos, BLOCKING)

    def random_object(self):
        # 得到工作空間範圍(plame大小)

        err, x_max = vrep.simxGetObjectFloatParameter(self.clientID, self.plane, vrep.sim_objfloatparam_modelbbox_max_x,
                                                      BLOCKING)
        err, x_min = vrep.simxGetObjectFloatParameter(self.clientID, self.plane, vrep.sim_objfloatparam_modelbbox_min_x,
                                                      BLOCKING)
        err, y_max = vrep.simxGetObjectFloatParameter(self.clientID, self.plane, vrep.sim_objfloatparam_modelbbox_max_y,
                                                      BLOCKING)
        err, y_min = vrep.simxGetObjectFloatParameter(self.clientID, self.plane, vrep.sim_objfloatparam_modelbbox_min_y,
                                                      BLOCKING)

        # 得到plane位置
        err, plane_pos = vrep.simxGetObjectPosition(self.clientID, self.plane, -1, BLOCKING)
        POS_MIN, POS_MAX = [plane_pos[0] + x_min, plane_pos[1] + y_min,0], [plane_pos[0] + x_max, plane_pos[1] + y_max,0]
        # print('min', POS_MIN, 'MAX', POS_MAX)
        pos = list(np.random.uniform(POS_MIN, POS_MAX))



        # 物體隨機擺放

        pos = [0.4962,-0.0054,0.05]
        #for eval
        # pos_eval=[ 0.483295, 0.009622, 0.049872]
        vrep.simxSetObjectPosition(self.clientID, self.Cuboid, -1, pos, ONESHOT)

    def get_cuboid_pos(self):
        self.Cuboid_pos = self.get_position(self.Cuboid)  # 讀物體位置
        err, cuboid_x = vrep.simxGetObjectFloatParameter(self.clientID, self.Cuboid, vrep.sim_objfloatparam_modelbbox_max_x,BLOCKING)
        err, cuboid_y= vrep.simxGetObjectFloatParameter(self.clientID, self.Cuboid, vrep.sim_objfloatparam_modelbbox_max_y, BLOCKING)
        cuboid_x_range = np.array([self.Cuboid_pos[0] + cuboid_x, self.Cuboid_pos[0] - cuboid_x])
        cuboid_y_range = np.array([self.Cuboid_pos[1] + cuboid_y, self.Cuboid_pos[1] - cuboid_y])

        # self.Cuboid_pos[2] += self.get_object_height(self.Cuboid)  # 得到物體表面位置
        self.Cuboid_pos[2] += 0.39  # 得到物體表面位置
        return (self.Cuboid_pos), cuboid_x_range , cuboid_y_range

    def get_EEF_pos(self):
        _, self.EEF_pos = vrep.simxGetObjectPosition(self.clientID, self.EEF, -1, BLOCKING)
        return self.EEF_pos

    def get_joint_pos(self):

        for i in range(jointNum):
            err, self.joint_pos[i] = vrep.simxGetJointPosition(self.clientID, self.joint_handle[i], BLOCKING)

        # print(self.joint_pos*self.radtodeg)

        return self.joint_pos

    def simxSetJointTargetPosition(self, new_joint_pos):
        for i in range(jointNum):
            vrep.simxSetJointTargetPosition(self.clientID, self.joint_handle[i], new_joint_pos, STREAMING)

    def get_object_height(self, handle):
        # 得到物體高度
        time.sleep(0.2)
        err, minval = vrep.simxGetObjectFloatParameter(self.clientID, handle, vrep.sim_objfloatparam_modelbbox_min_z,
                                                       BLOCKING)

        err, maxval = vrep.simxGetObjectFloatParameter(self.clientID, handle, vrep.sim_objfloatparam_modelbbox_max_z,
                                                       BLOCKING)

        return (maxval - minval) / 2

    def get_position(self, handle):
        # 得到物體位置3D
        err, pos = vrep.simxGetObjectPosition(self.clientID, handle, -1, BLOCKING)
        # err, pos = vrep.simxGetObjectPosition(self.clientID, handle, self.kinectDepth_handle, BLOCKING)
        return pos

    def orientation(self, handle):
        # 得到物體位置3D
        err, euler_angles = vrep.simxGetObjectOrientation(self.clientID, handle, -1,BLOCKING)
        return euler_angles

    def EEF_ori(self):

        euler_angles=self.orientation(self.EEF)

        return euler_angles

    def show_msg(self, message):
        """ send a message for printing in V-REP """
        vrep.simxAddStatusbarMessage(self.clientID, message, WAIT)
        return

    def read_object_id(self):
        _, self.Dummy7 = vrep.simxGetObjectHandle(self.clientID, 'Dummy7', BLOCKING)
        # 讀robot base id
        _,self.my_robot=vrep.simxGetObjectHandle(self.clientID, 'my_robot', BLOCKING)

        # 拿取joint id
        for i in range(jointNum):
            _, self.joint_handle[i] = vrep.simxGetObjectHandle(self.clientID, jointName + str(i + 1), BLOCKING)

        # 第一次讀取joint一定要streaming
        for i in range(jointNum):
            _, joint_pos = vrep.simxGetJointPosition(self.clientID, self.joint_handle[i], STREAMING)

        # 讀取 末端點 id
        _, self.EEF = vrep.simxGetObjectHandle(self.clientID, 'tip', BLOCKING)

        # 讀 cuboid id
        _, self.Cuboid = vrep.simxGetObjectHandle(self.clientID, 'Cuboid', BLOCKING)

        # 讀suction id
        res, self.suctionPad = vrep.simxGetObjectHandle(self.clientID, 'suctionPad', BLOCKING)

        # 讀plane id
        res, self.plane = vrep.simxGetObjectHandle(self.clientID, 'Plane', BLOCKING)

        # 讀camera id
        _, self.camera_handle = vrep.simxGetObjectHandle(self.clientID, 'kinect', BLOCKING)
        _, self.kinectRGB_handle = vrep.simxGetObjectHandle(self.clientID, 'kinect_rgb', BLOCKING)
        _, self.kinectDepth_handle = vrep.simxGetObjectHandle(self.clientID, 'kinect_depth', BLOCKING)

        # err, pos = vrep.simxGetObjectPosition(self.clientID, self.Cuboid,self.kinectDepth_handle, BLOCKING)


        print('handle available!!!')

    def move_all_joint(self, joint_angle):

        vrep.simxPauseCommunication(self.clientID, True)
        for i in range(jointNum):
            vrep.simxSetJointTargetPosition(self.clientID, self.joint_handle[i], joint_angle[i], ONESHOT)
        vrep.simxPauseCommunication(self.clientID, False)

    def move_4_joint(self, joint_angle):
        #MOVE JOINT 1,2,3,5
        vrep.simxPauseCommunication(self.clientID, True)

        vrep.simxSetJointTargetPosition(self.clientID, self.joint_handle[0], joint_angle[0], ONESHOT)
        vrep.simxSetJointTargetPosition(self.clientID, self.joint_handle[1], joint_angle[1], ONESHOT)
        vrep.simxSetJointTargetPosition(self.clientID, self.joint_handle[2], joint_angle[2], ONESHOT)
        vrep.simxSetJointTargetPosition(self.clientID, self.joint_handle[4], joint_angle[3], ONESHOT)

        vrep.simxPauseCommunication(self.clientID, False)

    def one_joint(self, i,joint_angle):
        # MOVE ONE JOINT
        vrep.simxSetJointTargetPosition(self.clientID, self.joint_handle[i], joint_angle, ONESHOT)



    def enable_suction(self, active):
        if active:
            vrep.simxSetIntegerSignal(self.clientID, self.suction, 1, ONESHOT)
            _,value = vrep.simxGetIntegerSignal(self.clientID, self.suction, BLOCKING)
        else:
            vrep.simxSetIntegerSignal(self.clientID, self.suction, 0, ONESHOT)
            _,value = vrep.simxGetIntegerSignal(self.clientID, self.suction, BLOCKING)

        return value


    def test_env(self):
        self.connection()
        self.start_sim()
        # self.get_joint_pos()
        lastCmdTime = vrep.simxGetLastCmdTime(self.clientID)  # 記錄當前時間
        vrep.simxSynchronousTrigger(self.clientID)  # 讓仿真走一步
        while vrep.simxGetConnectionId(self.clientID) != -1:
            currCmdTime = vrep.simxGetLastCmdTime(self.clientID)  # 記錄當前時間
            # dt = currCmdTime - lastCmdTime # 記錄時間間隔，用於控制
            # robot.get_joint_pos()
            vrep.simxPauseCommunication(self.clientID, True)

            # self.simxSetJointTargetPosition(120/self.radtodeg)

            for i in range(jointNum):
                vrep.simxSetJointTargetPosition(self.clientID, self.joint_handle[i], 30 / self.radtodeg, STREAMING)
            vrep.simxPauseCommunication(self.clientID, False)

            lastCmdTime = currCmdTime  # 記錄當前時間
            vrep.simxSynchronousTrigger(self.clientID)  # 進行下一步
            vrep.simxGetPingTime(self.clientID)  # 使得該仿真步走完




    def reset(self):
        self.stop_sim()
        self.start_sim()




#----------------------camera info-----------------------#

    def set_up_camera(self,camera_handle):
        # ----------------------------get camera pose
        _, cam_position = vrep.simxGetObjectPosition(self.clientID, camera_handle, -1, vrep.simx_opmode_blocking)
        _, cam_orientation = vrep.simxGetObjectOrientation(self.clientID, camera_handle,-1, vrep.simx_opmode_blocking)

        cam_trans = np.eye(4, 4)
        cam_trans[0:3, 3] = np.asarray(cam_position)
        # cam_orientation = [-cam_orientation[0], -cam_orientation[1], -cam_orientation[2]]
        # cam_orientation = [cam_orientation[0], cam_orientation[1], cam_orientation[2]]
        cam_rotm = np.eye(4, 4)



        cam_rotm[0:3, 0:3] = euler2mat(cam_orientation[0], cam_orientation[1], cam_orientation[2])  # 逆矩陣
        cam_pose = np.dot(cam_trans, cam_rotm)

        return cam_position,cam_orientation , cam_rotm, cam_pose
    def get_depth_camera_pose(self):
        return self.set_up_camera(self.kinectDepth_handle)

    def intri_camera(self):
        # ----------------------------get camera 內參
        fx = -self.width / 2.0 / (math.tan(self.theta * self.degtorad / 2.0))
        fy = -fx
        u0 = self.width / 2
        v0 = self.height / 2
        intri = np.array([
            [fx, 0, u0],
            [0, fy, v0],
            [0, 0, 1]])

        return intri
    def get_camera_data(self):
        # 從VREP得到圖片資訊
        # ---------------------------彩色圖片
        res, resolution, raw_image = vrep.simxGetVisionSensorImage(self.clientID, self.kinectRGB_handle, 0,BLOCKING)
        # color_img = np.array(raw_image, dtype=np.uint8)
        # color_img.shape = (resolution[1], resolution[0], 3)
        # # color_img = color_img.astype(np.float) / 255
        # # color_img[color_img < 0] += 1  # 這甚麼??
        # # color_img *= 255
        # # color_img = np.flipud(color_img)  # 翻轉列表
        # color_img = cv.flip(color_img, 0)
        #
        # color_img = color_img.astype(np.uint8)  # np.uint8[0,255]  如果是float 就是灰階圖片

#---------------------------------以上為 rgb 顯示, 以下為 bgr 顯示---------------------------------#
#----------------------------------------------------------------------------------------------#

        image_rgb_r = [raw_image[i] for i in range(0, len(raw_image), 3)]
        image_rgb_r = np.array(image_rgb_r)
        image_rgb_r = image_rgb_r.reshape(resolution[1], resolution[0])
        image_rgb_r = image_rgb_r.astype(np.uint8)

        image_rgb_g = [raw_image[i] for i in range(1, len(raw_image), 3)]
        image_rgb_g = np.array(image_rgb_g)
        image_rgb_g = image_rgb_g.reshape(resolution[1], resolution[0])
        image_rgb_g = image_rgb_g.astype(np.uint8)

        image_rgb_b = [raw_image[i] for i in range(2, len(raw_image), 3)]
        image_rgb_b = np.array(image_rgb_b)
        image_rgb_b = image_rgb_b.reshape(resolution[1], resolution[0])
        image_rgb_b = image_rgb_b.astype(np.uint8)

        color_img = cv.merge([image_rgb_b, image_rgb_g, image_rgb_r])     # merge圖像多個通道合併 ； split多通到圖像分離
        # 鏡像翻轉，opencv在這裡返回是一張翻轉的影像
        color_img = cv.flip(color_img, 0)        # flip 圖像翻轉 1:水平翻轉；0:垂直翻轉；-1:水平垂直翻轉

        # ---------------------------------深度圖---------------------------------#
        res, resolution, depth_buffer = vrep.simxGetVisionSensorDepthBuffer(self.clientID, self.kinectDepth_handle,BLOCKING)
        depth_img = np.array(depth_buffer)
        depth_img.shape = (resolution[1], resolution[0])

        depth_img = np.flipud(depth_img)  # 翻轉列表
        depth_img[depth_img < 0] = 0
        depth_img[depth_img > 1] = 0.9999

        depth_img = self.dis_near + (self.dis_far - self.dis_near)* depth_img  # 0.01124954
        # depth_img = (self.dis_far * self.dis_near / (self.dis_far - (self.dis_far - self.dis_near))) * depth_img  # 0.01124954
        # print('1',depth_img1,self.dis_far )
        # depth_img = 1. * self.dis_far*self.dis_near / (self.dis_far - (self.dis_far- self.dis_near) * depth_img)  #0.0102
        # print('2', depth_img)
        depth_img_for_show = (depth_img - np.min(depth_img)) * 255 / (np.max(depth_img) - np.min(depth_img))  # 正規化 0~255

        depth_img_for_show = depth_img_for_show.astype(np.uint8)

        depth_img_for_show = cv.cvtColor(depth_img_for_show, cv.COLOR_GRAY2BGR)

        return color_img, depth_img,depth_img_for_show

    def save_image_and_show(self,cur_color,cur_depth,depth_img_for_show,img_idx):
        ## 存影像圖片  將原本array轉成image
        img = Image.fromarray(cur_color.astype(np.uint8), mode='RGB')  # .convert('RGB')  #array到image的實現轉換
        img_path = os.path.join(self.SAVE_PATH_COLOR, str(img_idx) + '_rgb.png')
        img.save(img_path)
        ##   存深度圖
        depth_img = Image.fromarray(cur_depth.astype(np.uint8), mode='I')  # array到image的實現轉換
        depth_path = os.path.join(self.SAVE_PATH_COLOR, str(img_idx) + '_depth.png')
        depth_img.save(depth_path)
        ##   存深度圖(3channel)
        depth_img_for_show = Image.fromarray(cur_depth.astype(np.uint8), mode='RGB')  # array到image的實現轉換
        depth_show_path = os.path.join(self.SAVE_PATH_COLOR, str(img_idx) + '_depth_for_show.png')
        depth_img_for_show.save(depth_show_path)

        bg_depth=cv.imread(depth_path,-1)/10000
        depth_img_for_show = cv.imread(depth_img_for_show)
        bg_color=cv.imread(img_path)/255
        cv.imshow('color Image',bg_color)
        cv.imshow('depth 3channel Image', depth_img_for_show )
        cv.imshow('depth Image', bg_depth)
        # 按下任意鍵則關閉所有視窗
        cv.waitKey(0)
        cv.destroyAllWindows()

    def get_depth_from_RGB(self,num=5, resy=config.resolutionY, pixel=np.array([255, 177])):
        depth_path = os.path.join(self.SAVE_PATH_COLOR, str(num) + '_depth.png')
        bg_depth = cv.imread(depth_path, 0)
        bg_depth [bg_depth  < 0] = 0
        bg_depth [bg_depth > 1] = 0.9999


        # # -----翻轉照片
        # depth_img_flip = np.zeros([424, 512])
        # for i in range(424):
        #     for j in range(512):
        #         depth_img_flip[i][j] = depth_img[423 - i][j]
        # # -----翻轉照片

        pixel_depth =bg_depth [pixel[1]][pixel[0]]
        # print('de',pixel_depth)
        pixel_depth = self.dis_near + (self.dis_far-self.dis_near)*pixel_depth

        # pixel_depth_img_flip = depth_img_flip[resy - pixel[1]][pixel[0]]
        # pixel_depth_img_flip = dis_near + (dis_far-dis_near)*pixel_depth_img_flip

        return pixel_depth
    def get_depth(self,u,v ,depth):
        pixel_depth = depth[v][u]
        # pixel_depth = self.dis_near + (self.dis_far-self.dis_near)*pixel_depth
        return pixel_depth

    def arrayToImage(self,color_img):
        path = config.yolo_Img_path
        if os.path.exists(path):
            os.remove(path)
        cv.imwrite(path, color_img)

    def arrayToDepthImage(self,depth_img_for_show):
        path = config.yolo_Dep_path
        if os.path.exists(path):
            os.remove(path)
        cv.imwrite(path, depth_img_for_show)

    def uv_2_xyz(self,z,u,v,resx,resy,theta):
        v = resy - v

        x = z*(math.tan(theta*self.degtorad/2))*2*((resx/2-u)/resx)
        y = z * (math.tan(theta * self.degtorad / 2)) * 2 * ((v-resy / 2 ) / resx)

        return  x,y

    def xyz_2_uv(self,x, y, z, resx, resy, theta):
        # ------------xyz to pixel
        u = x * (resx / 2) * (-1 / math.tan(theta * self.deg2rad / 2.0)) * (1 / z) + resx / 2
        v = y * (resx / 2) * (1 / math.tan(theta * self.deg2rad / 2.0)) * (1 / z) + resy / 2

        # call funtion EX: u,v=xyz_2_uv(o_in_cam_vrep[0],o_in_cam_vrep[1],o_in_cam_vrep[2],512,424,70)
        # cam_intri = intri_camera()
        # xyz = np.array([x,y,z])
        # xyz = np.reshape(xyz,(3,1))
        # uv = (1 / z) * np.dot(cam_intri , xyz)  #------>用內參matrix算
        # u,v = uv[0],uv[1]
        v = resy - v
        return u, v

    def coor_trans_AtoB(self,A_pos, A_ori, point):
        # print('ori',ori)
        matrix = euler2mat(A_ori[0], A_ori[1], A_ori[2])

        t = np.array([[matrix[0][0], matrix[0][1], matrix[0][2], A_pos[0]],
                      [matrix[1][0], matrix[1][1], matrix[1][2], A_pos[1]],
                      [matrix[2][0], matrix[2][1], matrix[2][2], A_pos[2]]])
        # cam_cor = np.array([0.00054361257757454, -0.036422047104143, 0.39749592260824,1])
        point = np.array([point[0], point[1], point[2], 1])
        B_frame_coor = np.dot(t, point)

        return B_frame_coor

    def find_square(self,color_img):
        #--------------------------------
        #   輸入rgb影像並找到正方形所在位置  #
        #--------------------------------
        squares = []
        img = cv.GaussianBlur(color_img, (3, 3), 0)
        # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  #會找不到
        bin = cv.Canny(img, 30, 100, apertureSize=3)
        contours, _hierarchy = cv.findContours(bin, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        print("轮廓数量：%d" % len(contours))
        index = 0
        for cnt in contours:
            cnt_len = cv.arcLength(cnt, True)  # 计算轮廓周长
            cnt = cv.approxPolyDP(cnt, 0.02 * cnt_len, True)  # 多边形逼近
            # 条件判断逼近边的数量是否为4，轮廓面积是否大于1000，检测轮廓是否为凸的
            if len(cnt) == 4 and cv.contourArea(cnt) > 1000 and cv.isContourConvex(cnt):
                M = cv.moments(cnt)  # 计算轮廓的矩
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])  # 轮廓重心

                cnt = cnt.reshape(-1, 2)
                max_cos = np.max([angle_cos(cnt[i], cnt[(i + 1) % 4], cnt[(i + 2) % 4]) for i in range(4)])
                # 只检测矩形（cos90° = 0）
                if max_cos < 0.1:
                    # 检测四边形（不限定角度范围）
                    # if True:
                    index = index + 1
                    # cv.putText(color_img, ("#%d" % index), (cx, cy), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                    squares.append(cnt)
        return squares

def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )
if __name__ == '__main__':
    joint_pos =[]
    cuboid=[]
    robot = my_robot()
    robot.connection()
    robot.start_sim()
    World_coor = [0.6,0.2,0.7]
