import sim as vrep
import time
import math
import numpy as np
import cv2 as cv
import PIL.Image as Image
import os
from Rot2RPY import Rot2RPY,euler2mat,Rot2RPY_version2
import matplotlib.pyplot as mpl
from imutils import perspective
from imutils import contours
import imutils
import argparse
from scipy.spatial import distance as dist
#-----影像長寬
width=512
height=424

#-----角度(不知道要幹嘛跟內參有關)
theta=70
theta_ratio=width/height
deg2rad=math.pi/180

#-----深度圖的東西
dis_far=1
dis_near=0.01
depth_scale=1000

#-----存照片路徑
SAVE_PATH_COLOR='C:/Users/ad456/Desktop/myrobot/my_robot_dh'



def save_txt(path, name, data, fmt='%f'):
    f = open(path + name, 'w')
    np.savetxt(f, data, fmt=fmt)
    f.close()
def path_exsit(path):
    if os.path.exists(path):
        return True
    else:
        return False
def creat_path(path):
    if path_exsit(path=path):
        print(path+' exist')
    else:
        os.makedirs(path)



def set_up_camera(camera_handle):
    #----------------------------get camera pose
    _,cam_position=vrep.simxGetObjectPosition(clientID,camera_handle,-1,vrep.simx_opmode_blocking)
    _,cam_orientation=vrep.simxGetObjectOrientation(clientID,camera_handle,-1,vrep.simx_opmode_blocking)

    cam_trans=np.eye(4,4)
    cam_trans[0:3,3]=np.asarray(cam_position)
    cam_orientation=[-cam_orientation[0],-cam_orientation[1],-cam_orientation[2]]
    cam_rotm=np.eye(4,4)

    cam_rotm[0:3,0:3]=euler2mat(cam_orientation[0],cam_orientation[1],cam_orientation[2])#逆矩陣
    cam_pose=np.dot(cam_trans,cam_rotm)
    return cam_position,cam_pose,cam_rotm

def set_up_robot(robot_handle):
    _,robot_position=vrep.simxGetObjectPosition(clientID,robot_handle,-1,vrep.simx_opmode_blocking)
    _,robot_orientation=vrep.simxGetObjectOrientation(clientID,robot_handle,-1,vrep.simx_opmode_blocking)

    robot_trans = np.eye(4,4)
    robot_trans[0:3,3] = np.asarray(robot_position)
    robot_rotm = np.eye(4,4)
    robot_rotm[0:3,0:3] = np.linalg.inv(euler2mat(robot_orientation[0],robot_orientation[1],robot_orientation[2]))
    robot_pose=np.dot(robot_trans,robot_rotm)

    return robot_position,robot_pose

def intri_camera():
#----------------------------get camera 內參
    fx= -width/2.0/(math.tan(theta*deg2rad/2.0))
    fy = -fx
    u0 = width/2
    v0 = height/2
    intri=np.array([
                    [fx,0,u0],
                    [0,fy,v0],
                    [0, 0, 1]])

    return intri
#---------------------------讀影像資訊並轉為np array
def save_image(cur_depth,cur_color,img_idx):
    ##   存影像圖  array转换成image
    img=Image.fromarray(cur_color.astype(np.uint8),mode='RGB')   #.convert('RGB')  #array到image的實現轉換
    img_path=os.path.join(SAVE_PATH_COLOR,str(img_idx)+'_rgb.png')
    img.save(img_path)
    ##   存深度圖
    depth_img = Image.fromarray(cur_depth.astype(np.uint8),mode='RGB')  # array到image的實現轉換
    depth_path=os.path.join(SAVE_PATH_COLOR,str(img_idx)+'_depth.png')
    depth_img.save(depth_path)

    return img_path,depth_path
def get_camera_data(kinectRGB_handle,kinectDepth_handle):
    #---------------------------彩色圖片
    res,resolution,raw_image=vrep.simxGetVisionSensorImage(clientID,kinectRGB_handle,0,vrep.simx_opmode_blocking)
    color_img=np.array(raw_image,dtype=np.uint8)
    color_img.shape=(resolution[1],resolution[0],3)
    color_img=color_img.astype(np.float)/255
    color_img[color_img<0]+=1   #這甚麼??
    color_img*=255
    color_img=np.flipud(color_img)  #翻轉列表

    # color_img = cv.flip(color_img, 0)

    color_img=color_img.astype(np.uint8)  #np.uint8[0,255]  如果是float 就是灰階圖片


    #---------------------------深度圖片
    res,resolution,depth_buffer=vrep.simxGetVisionSensorDepthBuffer(clientID,kinectDepth_handle,vrep.simx_opmode_blocking)
    # print('depth_buffer',depth_buffer)
    depth_img=np.array(depth_buffer)
    depth_img.shape=(resolution[1],resolution[0])
    depth_img=np.flipud(depth_img)  #翻轉列表
    depth_img[depth_img<0]=0
    depth_img[depth_img>1]=0.9999

    depth_img = (dis_far * dis_near / (dis_far - (dis_far - dis_near) )) * depth_img # 0.01124954
    # print('123',(np.max(depth_img) - np.min(depth_img)))
    # depth_img = dis_far * depth_img * depth_scale                                  # 112.49544655

    depth_img = (depth_img - np.min(depth_img)) * 255 / (np.max(depth_img) - np.min(depth_img))  # 正規化 0~255

    depth_img = depth_img.astype(np.uint8)

    depth_img = cv.cvtColor(depth_img,cv.COLOR_GRAY2BGR)
    # print(depth_img)
    return color_img,depth_img


#---------------------------存圖片到資料夾
def save_image_and_show(kinectRGB_handle,kinectDepth_handle,img_indx):
    color_img,cur_depth=get_camera_data(kinectRGB_handle,kinectDepth_handle)
    img_path,depth_path=save_image(cur_depth,color_img,img_indx)
    #---------------------------從資料夾讀圖片並秀出來
    bg_depth=cv.imread(depth_path)
    bg_color=cv.imread(img_path)/255
    cv.imshow('color Image',bg_color)
    cv.imshow('depth Image', bg_depth)
    # 按下任意鍵則關閉所有視窗
    cv.waitKey(0)
    cv.destroyAllWindows()
    #---------------------------從資料夾讀圖片並秀出來

def pixel2myrobot(u,v,cur_depth,robot_position,cam_position,depth=0.0,is_dst=True):
    intri=intri_camera()

    # ---------------------------[u,v] to camera frame
    if is_dst==False:
        depth=cur_depth[int(u)][int(v)]/depth_scale
    depth=cur_depth[int(u)][int(v)]
    x=depth*(u-intri[0][2]/intri[0][0])
    y=depth*(v-intri[1][2]/intri[1][1])
    camera_coor=np.array([x,y,depth])

    # ---------------------------camera to robot frame
    camera_coor[2]=-camera_coor[2]
    print('camera_coor........',camera_coor)

    location=camera_coor+cam_position-np.asarray(robot_position)
    return location,depth

def get_pointcloud(color_img,depth_img,camera_intri):

    #get depth image size
    im_h = depth_img.shape[0]
    im_w = depth_img.shape[1]

    #project depth into 3D point cloud in camera coord

    pix_x, pix_y = np.meshgrid(np.linspace(0, im_w - 1, im_w), np.linspace(0, im_h - 1, im_h))
    # print('u,v',pix_y,pix_x)  #不知道是啥
    cam_pts_x = np.multiply(pix_x - camera_intri[0][2], depth_img / camera_intri[0][0])  # (u-u0)*z/kx
    cam_pts_y = np.multiply(pix_x - camera_intri[1][2], depth_img / camera_intri[1][1])  # (u-u0)*z/kx
    cam_pts_z = depth_img.copy()
    cam_pts_x.shape = (im_h * im_w, 1)
    cam_pts_y.shape = (im_h * im_w, 1)
    cam_pts_z.shape = (im_h * im_w, 1)

    rgb_pts_r = color_img[:, :, 0]
    rgb_pts_g = color_img[:, :, 1]
    rgb_pts_b = color_img[:, :, 2]
    rgb_pts_r.shape = (im_h * im_w, 1)
    rgb_pts_g.shape = (im_h * im_w, 1)
    rgb_pts_b.shape = (im_h * im_w, 1)

    cam_pts = np.concatenate((cam_pts_x,cam_pts_y,cam_pts_z),axis=1)  #(u,v),在相機座標
    rgb_pts = np.concatenate((rgb_pts_r,rgb_pts_g,rgb_pts_b),axis=1)  #點雲圖座標

    return cam_pts,rgb_pts

def get_heightmap(color_img,depth_img,camera_intri,cam_pos,workspace_limits,heightmap_resolution):
    #compute heightmap size
    heightmap_size = np.round(((workspace_limits[1][1]-workspace_limits[1][0])/heightmap_resolution,(workspace_limits[0][1]-workspace_limits[0][0])/heightmap_resolution))

    #get 3D point cloud from RGBD image
    surface_pts,color_pts = get_pointcloud(color_img,depth_img,camera_intri)
    print('robot ', surface_pts)
    #transfrom 3D point cloud from camera coord to robot coord
    surface_pts=np.transpose(np.dot(cam_pos[0:3,0:3],np.transpose(surface_pts))+np.tile(cam_pos[0:3,3:],(1,surface_pts.shape[0])))
    print('robot coor',surface_pts)

    #sort surface point by z values
    sort_z_ind=np.argsort(surface_pts[:,2])
    # print('z',sort_z_ind)
    surface_pts=surface_pts[sort_z_ind]
    color_pts = color_pts[sort_z_ind]

    #filter out surface points outside heightmap boundaries
    heightmap_valid_ind = np.logical_and(np.logical_and(np.logical_and(np.logical_and(surface_pts[:,0] >= workspace_limits[0][0], surface_pts[:,0] < workspace_limits[0][1]), surface_pts[:,1] >= workspace_limits[1][0]), surface_pts[:,1] < workspace_limits[1][1]), surface_pts[:,2] < workspace_limits[2][1])
    surface_pts=surface_pts[heightmap_valid_ind]
    color_pts=color_pts[heightmap_valid_ind]

    # #creat orthographic top-down-view RGB-D heightmaps
    # color_heightmap_r=np.zeros((heightmap_size[0],heightmap_size[1],1), dtype=np.uint8)
    # color_heightmap_g=np.zeros((heightmap_size[0],heightmap_size[1],1), dtype=np.uint8)
    # color_heightmap_b=np.zeros((heightmap_size[0],heightmap_size[1],1), dtype=np.uint8)
    # depth_heightmap=np.zeros(heightmap_size)
    # heightmap_pix_x = np.floor((surface_pts[:, 0] - workspace_limits[0][0]) / heightmap_resolution).astype(int)
    # heightmap_pix_y = np.floor((surface_pts[:, 1] - workspace_limits[1][0]) / heightmap_resolution).astype(int)
    # color_heightmap_r[heightmap_pix_y, heightmap_pix_x] = color_pts[:, [0]]
    # color_heightmap_g[heightmap_pix_y, heightmap_pix_x] = color_pts[:, [1]]
    # color_heightmap_b[heightmap_pix_y, heightmap_pix_x] = color_pts[:, [2]]
    # color_heightmap = np.concatenate((color_heightmap_r, color_heightmap_g, color_heightmap_b), axis=2)
    # depth_heightmap[heightmap_pix_y, heightmap_pix_x] = surface_pts[:, 2]
    # z_bottom = workspace_limits[2][0]
    # depth_heightmap = depth_heightmap - z_bottom
    # depth_heightmap[depth_heightmap < 0] = 0
    # depth_heightmap[depth_heightmap == -z_bottom] = np.nan
    #
    # return color_heightmap, depth_heightmap


def detect(image,depth,color='red'):
    bg_depth = depth
    bg_color = image
    if color == 'green':
        lower = (40, 60, 60)
        upper = (80, 255, 255)
    if color == 'blue':
        lower = (78, 158, 124)
        upper = (138, 255, 255)
    if color == 'red':
        lower = (0, 120, 70)
        upper = (10, 255, 255)
    if color == 'wt':
        lower = (6, 86, 29)
        upper = (255, 255, 64)


    blur=cv.GaussianBlur(bg_color,(5,5),0)
    cv.imshow('bg_color', bg_color)
    hsv = cv.cvtColor(blur,cv.COLOR_BGR2HSV)

    # bg_color = cv.cvtColor(np.asarray(bg_color,dtype=np.uint8),cv.COLOR_RGB2BGR)

    # small = cv.resize(bg_color,(0,0),fx=scale,fy=scale)

    #creat a mask for the green areas of the image
    mask=cv.inRange(hsv,lower,upper)
    bmask=cv.GaussianBlur(mask,(5,5),0)

    moments=cv.moments(mask)
    # print('bmask', moments)
    m00=moments['m00']
    centroid_x,centroid_y = None, None
    if m00!=0:
        centroid_x =round(moments['m10']/m00)
        centroid_y =round(moments['m01'] / m00)
        centroid_x_float =(moments['m10']/m00)
        centroid_y_float =(moments['m01'] / m00)
    print('centroid',centroid_x_float,centroid_y_float)
    # ctr=None
    if centroid_x!=None and centroid_y!=None:
        ctr=(centroid_x,centroid_y)
        # ctr = (0, 0)
        # print(',ctr',ctr)

    if ctr:
       cv.rectangle(bg_color,(ctr[0]-15,ctr[1]-15),(ctr[0]+15,ctr[1]+15),(0xff,0xf4,0x0d),2)
       cv.circle(bg_color, (int(ctr[0]), int(ctr[1] )), 2, (0, 0, 255), -1)

    cv.imshow('color Image', bg_color)
    # cv.imshow('depth Image', bmask)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return ctr

def contours(image,depth,color='red',draw=True):
    bg_depth = depth
    bg_color = image

    if color == 'green':
        lower = (40, 60, 60)
        upper = (80, 255, 255)
    if color == 'blue':
        lower = (86, 6, 6)
        upper = (255, 90, 255)
    if color == 'red':
        lower = (0, 120, 70)
        upper = (10, 255, 255)
    if color == 'wt':
        lower = (6, 86, 29)
        upper = (255, 255, 64)

    scale=1
    hsv = cv.cvtColor(np.asarray(bg_color,dtype=np.uint8),cv.COLOR_BGR2HSV)
    bg_color = cv.cvtColor(np.asarray(bg_color,dtype=np.uint8),cv.COLOR_RGB2BGR)
    small=cv.resize(hsv,(0,0),fx=scale,fy=scale) #圖像縮放
    mask=cv.inRange(small,lower,upper)
    #erosion and dilation to remove imperfections in masking
    mask=cv.erode(mask,None,iterations=2)  #腐蝕
    mask=cv.dilate(mask,None,iterations=2)  #膨脹


    #Find the contour of masked shapes
    contours=cv.findContours(mask.copy(),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    contours = contours[0]
    # print('cour',contours)
    center=None

    #if there is a masked object
    if len(contours)>0:
        #largest contour
        c = max(contours,key=cv.contourArea)

        #Radius
        ((x,y),radius) = cv.minEnclosingCircle(c)

        #moment of the largest contour
        moments=cv.moments(c)
        # print('m',moments)
        center=((moments['m10'] / moments['m00']),(moments['m01'] /moments['m00']))
        # print('center',center)
        if draw:
            #draw appropriate circles bg_depth
            if radius>2:
                cv.circle(bg_color,(int(x/scale),int(y/scale)),int(radius*1.25),(0,255,255),2)
                cv.circle(bg_color, (int(center[0] / scale), int(center[1] / scale)), 2, (0, 0, 255), -1)

                #  draw in depth image
                cv.circle(bg_depth,(int(x/scale),int(y/scale)),int(radius*1.25),(0,255,255),2)
                cv.circle(bg_depth, (int(center[0] / scale), int(center[1] / scale)), 2, (0, 0, 255), -1)
            cv.imshow('contours Image', bg_color)
            cv.imshow('contours depth Image', bg_depth)
            cv.waitKey(0)
            cv.destroyAllWindows()
    return center

def coor_trans_AtoB(A_pos,A_ori,point):
# print('ori',ori)
    matrix = euler2mat(A_ori[0],A_ori[1],A_ori[2])
# print('pos_in_cam',pos,'\n' ,matrix)
    t = np.array([[matrix[0][0],matrix[0][1],matrix[0][2],A_pos[0]],
                 [matrix[1][0],matrix[1][1],matrix[1][2],A_pos[1]],
                 [matrix[2][0],matrix[2][1],matrix[2][2],A_pos[2]]])
    # cam_cor = np.array([0.00054361257757454, -0.036422047104143, 0.39749592260824,1])
    point = np.array([point[0],point[1],point[2],1])
    B_frame_coor = np.dot(t,point)
    return B_frame_coor

def get_depth_from_vrep(depth_handle,pixel):
    _,res,depth_buffer = vrep.simxGetVisionSensorDepthBuffer(clientID,depth_handle,vrep.simx_opmode_blocking)
    depth_buffer = np.array(depth_buffer)
    depth_buffer.shape = (res[1],res[0])
    depth_buffer[depth_buffer < 0] = 0
    depth_buffer[depth_buffer > 1] = 0.9999
    # depth_buffer = (dis_far * dis_near / (dis_far - (dis_far - dis_near) )* depth_buffer)

    depth_buffer = dis_near + (dis_far-dis_near)*depth_buffer
    #depth = get_depth(blobcam,pixel = np.array([255,177]))
    return depth_buffer[pixel[1]][pixel[0]]


def get_depth_from_RGB(num=5, resy=424, pixel=np.array([255, 177])):
    depth_path = os.path.join(SAVE_PATH_COLOR, str(num) + '_depth.png')
    bg_depth = cv.imread(depth_path, 0)
    depth_img = bg_depth * 0.6088108288288111 / 255 + 0.391089171171188
    depth_img[depth_img < 0] = 0
    depth_img[depth_img > 1] = 0.9999

    # print('test',depth_img[244][254])
    # -----翻轉照片
    depth_img_flip = np.zeros([424, 512])
    for i in range(424):
        for j in range(512):
            depth_img_flip[i][j] = depth_img[423 - i][j]
    # -----翻轉照片

    pixel_depth = depth_img[pixel[1]][pixel[0]]
    # print('de',pixel_depth)
    pixel_depth = dis_near + (dis_far-dis_near)*pixel_depth

    pixel_depth_img_flip = depth_img_flip[resy - pixel[1]][pixel[0]]
    pixel_depth_img_flip = dis_near + (dis_far-dis_near)*pixel_depth_img_flip
    # cv.imshow('depth',depth_img_flip)
    # cv.waitKey(0)
    # ---EX : depth ,depth_flip= get_depth_from_RGB()
    return pixel_depth , pixel_depth_img_flip

def xyz_2_uv(x,y,z,resx,resy,theta):
    #------------xyz to pixel
    u = x * (resx/2) * (-1/math.tan(theta * deg2rad / 2.0))*(1/z) +resx/2
    v = y * (resx/2) *(1/math.tan(theta * deg2rad / 2.0))*(1/z) +resy/2

    #call funtion EX: u,v=xyz_2_uv(o_in_cam_vrep[0],o_in_cam_vrep[1],o_in_cam_vrep[2],512,424,70)
    # cam_intri = intri_camera()
    # xyz = np.array([x,y,z])
    # xyz = np.reshape(xyz,(3,1))
    # uv = (1 / z) * np.dot(cam_intri , xyz)  #------>用內參matrix算
    # u,v = uv[0],uv[1]
    v = resy-v
    return u,v

def uv_2_xyz(z,u,v,resx,resy,theta):
    #------------pixel to xyz
    v = resy-v

    x = z*(math.tan(theta*deg2rad/2))*2*((resx/2-u)/resx)
    y = z*(math.tan(theta*deg2rad/2))*2*((v-resy/2)/resx)
    #call funtion EX: x,y=uv_2_xyz(o_in_cam_vrep[2],256.5,261.999,512,424,70)

    # cam_intri = intri_camera()
    # intri_inver = np.linalg.inv(cam_intri)
    # uv = np.array([u,v,1])
    # xyz = np.dot(intri_inver, uv) * z
    return x,y


def show_image(color_image,depth_image):
    cv.imshow('color Image', color_image)
    cv.imshow('depth Image', depth_image)
    cv.waitKey(0)
    cv.destroyAllWindows()




# if __name__ == '__main__':

# Close all open connections (just in case)
vrep.simxFinish(-1)

# Connect to V-REP (raise exception on failure)
clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
if clientID == -1:
    raise Exception('Failed connecting to remote API server')
#
# #----------------------------第一台相機的handle
_,camera_handle=vrep.simxGetObjectHandle(clientID,'kinect', vrep.simx_opmode_blocking)
_,kinectRGB_handle=vrep.simxGetObjectHandle(clientID,'kinect_rgb',vrep.simx_opmode_blocking)
_,kinectDepth_handle=vrep.simxGetObjectHandle(clientID,'kinect_depth',vrep.simx_opmode_blocking)
#----------------------------第二台相機的handle
# _,camera_handle2=vrep.simxGetObjectHandle(clientID,'kinect2', vrep.simx_opmode_blocking)
# # _,kinectRGB_handle2=vrep.simxGetObjectHandle(clientID,'kinect_rgb2',vrep.simx_opmode_blocking)
# _,kinectDepth_handle2=vrep.simxGetObjectHandle(clientID,'kinect_depth2',vrep.simx_opmode_blocking)
# #----------------------------myrobot的handle
_,myrobot_handle=vrep.simxGetObjectHandle(clientID,'my_robot_base', vrep.simx_opmode_blocking)
_,Cuboid_handle=vrep.simxGetObjectHandle(clientID,'Sphere0', vrep.simx_opmode_blocking)
#-------------------print coor------------------#
# _,pos_in_cam=vrep.simxGetObjectPosition(clientID,Cuboid_handle,kinectDepth_handle,vrep.simx_opmode_blocking)
# _,ori = vrep.simxGetObjectOrientation(clientID,kinectRGB_handle,-1,vrep.simx_opmode_blocking)
# print('ori',ori)
# matrix = euler2mat(ori[0],ori[1],ori[2])
# print('pos_in_cam',pos_in_cam ,'\n' ,matrix)

def test_for_blobposition():

    #this code is for test get coor from vrep vision to world coor and for world coor to pixel

    #-------------------test for blobposition------------------#
    _,blobcam = vrep.simxGetObjectHandle(clientID,'blobTo3dPosition_sensor',vrep.simx_opmode_blocking)
    _,pos=vrep.simxGetObjectPosition(clientID,blobcam,-1,vrep.simx_opmode_blocking)
    _,ori = vrep.simxGetObjectOrientation(clientID,blobcam,-1,vrep.simx_opmode_blocking)

    #-------------------print coor------------------#

    vrep.simxStartSimulation(clientID,vrep.simx_opmode_oneshot)
    time.sleep(0.2)
    # save_image_and_show(kinectRGB_handle,kinectDepth_handle,3)
    # save_image_and_show(kinectRGB_handle2,kinectDepth_handle2,2)
    save_image_and_show(blobcam,blobcam,1)
    time.sleep(0.2)
    vrep.simxStopSimulation(clientID,vrep.simx_opmode_oneshot)


    #----------------from pixel to world coor---------------##
    depth_path=os.path.join(SAVE_PATH_COLOR,str(1)+'_depth.png')
    color_path=os.path.join(SAVE_PATH_COLOR,str(1)+'_rgb.png')

    bg_color = cv.imread(color_path)
    bg_depth = cv.imread(depth_path,0)

    center = detect(bg_color,bg_depth)
    print('center',center)
    depth,depth_flip = get_depth_from_RGB(num=1, resy=424, pixel=center)
    print('depth',depth,depth_flip)
    x,y = uv_2_xyz(depth,center[0],center[1],512,424,70)
    img_coor = np.array([x,y,depth])
    print('img_coor',img_coor)
    world_coor = coor_trans_AtoB(pos,ori,img_coor)
    print('world_coor',world_coor)

    #----------------from coor to pixel---------------##

    u,v = xyz_2_uv(img_coor[0],img_coor[1],img_coor[2],512,resy = 424,theta = 70)
    print(u,v)










