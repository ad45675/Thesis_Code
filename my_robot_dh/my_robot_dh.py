import sim as vrep
import time
import numpy as np
from Rot2RPY import Rot2RPY,euler2mat,Rot2RPY_version2
from Kinematics import ForwardKinemetics as FK
import math

DegToRad = math.pi/ 180
RadToDeg = 180/math.pi
# Close all open connections (just in case)
vrep.simxFinish(-1)

# Connect to V-REP (raise exception on failure)
clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
if clientID == -1:
    raise Exception('Failed connecting to remote API server')

# Second, obtain the initial pose of the end effector (dummy7 frame)
# Get the handle of dummy7 object
# Start simulation
# vrep.simxStartSimulation(clientID, vrep.simx_opmode_oneshot)
dummy_handle=np.zeros((7,),np.float32)
joint_handle=np.zeros((7,),np.float32)
Rjoint_handle=np.zeros((7,),np.float32)
joint_ori=np.zeros((7,3),np.float32)
joint_pos=np.zeros((7,3),np.float32)
IK=np.zeros((7,3),np.float32)
def load_txt(path,):
    f = open(path , 'r')
    data = np.loadtxt(f)
    f.close()
    return data
path='C:/Users/user/Desktop/my_robot_dh/IK_ans/IK.txt'
# IK=load_txt(path)
# print('IK',IK)
#------------set_robot_configuaration----------------
# joint_target_angle=[0.37060997 ,-2.36066826, -2.86853279 , 0.57909029 , 1.36038727 , 0.53025717]
# joint_target_angle=[0.82753931, -1.01365489, -1.29381823, -1.53957381,  1.59910342 , 1.57079633]
# for j in range(108):
# vrep.simxStartSimulation(clientID, vrep.simx_opmode_oneshot)
joint_target_angle = [0,0,0,0,0,0]
# joint_target_angle=IK
#0,0.785,-0.785,0,-0.785,0
# --------------------get_handle --------------------#
for i in range(7):
	result, dummy_handle[i] = vrep.simxGetObjectHandle(clientID, 'Dummy'+str(i), vrep.simx_opmode_blocking)
	# result, joint_handle[i] = vrep.simxGetObjectHandle(clientID, 'joint' + str(i+1), vrep.simx_opmode_blocking)
for i in range(6):
	result, Rjoint_handle[i] = vrep.simxGetObjectHandle(clientID, 'joint' + str(i+1 ), vrep.simx_opmode_blocking)
	vrep.simxSetJointPosition(clientID, Rjoint_handle[i], joint_target_angle[i], vrep.simx_opmode_oneshot)
	time.sleep(0.5)
	res,joint_pos[i]=vrep.simxGetObjectPosition(clientID, Rjoint_handle[i],-1, vrep.simx_opmode_blocking)
	res,joint_ori[i] = vrep.simxGetObjectOrientation(clientID, Rjoint_handle[i], -1, vrep.simx_opmode_blocking)
joint_pos=np.round(joint_pos, 5)
print('---------joint_position_in_world_frame--------------')
for i in range(7):
	print('joint_pos',[i+1],joint_pos[i])
print('---------joint_orientation_in_world_frame--------------')
for i in range(7):
	print('joint_ori',[i+1],joint_ori[i])
# --------------------將 plane 位置訂在(0,0,0) --------------------#
result, plane= vrep.simxGetObjectHandle(clientID, 'Plane', vrep.simx_opmode_blocking)
vrep.simxSetObjectPosition(clientID, plane, -1, [0,0,0], vrep.simx_opmode_oneshot)

#--------------------------------------------------------------------------------------------------------------------------------
#加吸嘴要加上6.8cm  這裡單位是m
DH_table = np.array([[0,            0.345,  0.08,   math.pi / 2],
					 [0+math.pi / 2 , 0,  0.27,     0],
					 [0,             0,     0.09,    math.pi / 2],
					 [0,            0.295,  0,       -math.pi / 2],
					 [0,            0,      0,       math.pi / 2],
					 [0,       0.102+0.125, 0,          0]])
time.sleep(0.2)
# --------------------末端點的handle和位置和方位--------------------#
res,tip=vrep.simxGetObjectHandle(clientID,'tip',vrep.simx_opmode_blocking)
res,tip_pos=vrep.simxGetObjectPosition(clientID,tip,-1,vrep.simx_opmode_blocking)
res,tip_ori=vrep.simxGetObjectOrientation(clientID,tip,-1,vrep.simx_opmode_blocking)
# --------------------吸嘴handle和位置和方位 --------------------#
res,suctionDummy=vrep.simxGetObjectHandle(clientID,'suctionPadLoopClosureDummy1',vrep.simx_opmode_blocking)
res,suction_pos=vrep.simxGetObjectPosition(clientID,suctionDummy,-1,vrep.simx_opmode_blocking)
res,suction_ori=vrep.simxGetObjectOrientation(clientID,suctionDummy,-1,vrep.simx_opmode_blocking)
print('---------EEF_position and suction_position in_world_frame--------------')
for i in range(3):
	tip_pos[i] = round(tip_pos[i], 4)
	tip_ori[i] = round(tip_ori[i]*RadToDeg, 4)
	suction_pos[i]=round(suction_pos[i],4)
	suction_ori[i]=round(suction_ori[i]*RadToDeg,4)
print('tip_pos',tip_pos)
print('tip_ori',tip_ori)
print('suction_pos',suction_pos)
print('suction_ori',suction_ori)


#-------------當在原定姿態時的位置和角度---------------#
JointAngle=IK
JointAngle=[0,0,0,0,0,0]
Info ,EulerAngle_vrep,EulerAngle,Position=FK( JointAngle,DH_table )
# print('EulerAngle_vrep',EulerAngle_vrep)
##----------宣告----------##
position0 = np.zeros((3,), np.float32)
position1 = np.zeros((3,), np.float32)
position2 = np.zeros((3,), np.float32)
position3 = np.zeros((3,), np.float32)
position4 = np.zeros((3,), np.float32)
position5 = np.zeros((3,), np.float32)
position6 = np.zeros((3,), np.float64)
joint_dir0 = np.zeros((3,), np.float32)
joint_dir1 = np.zeros((3,), np.float32)
joint_dir2 = np.zeros((3,), np.float32)
joint_dir3 = np.zeros((3,), np.float32)
joint_dir4 = np.zeros((3,), np.float32)
joint_dir5 = np.zeros((3,), np.float32)
joint_dir6 = np.zeros((3,), np.float32)
EulerAngle0 = np.zeros(3)
##----------宣告----------##
for i in range(3):
	position0[i] = Info[0][0][i]
	position1[i] = Info[0][1][i]
	position2[i] = Info[0][2][i]
	position3[i] = Info[0][3][i]
	position4[i] = Info[0][4][i]
	position5[i] = Info[0][5][i]
	position6[i] = Info[0][6][i]

joint_dir0 = [0.0, 0.0, 0.0]
#matrix轉尤拉角 zyx
joint_dir1[0], joint_dir1[1], joint_dir1[2] = Rot2RPY(Info[1][1])
joint_dir2[0], joint_dir2[1], joint_dir2[2] = Rot2RPY(Info[1][2])
joint_dir3[0], joint_dir3[1], joint_dir3[2] = Rot2RPY(Info[1][3])
joint_dir4[0], joint_dir4[1], joint_dir4[2] = Rot2RPY(Info[1][4])
joint_dir5[0], joint_dir5[1], joint_dir5[2] = Rot2RPY(Info[1][5])
joint_dir6[0], joint_dir6[1], joint_dir6[2] = Rot2RPY(Info[1][6])

## matrix轉尤拉角 XYZ
# joint_dir1[0], joint_dir1[1], joint_dir1[2] = Rot2RPY_version2(Info[1][1])
# joint_dir2[0], joint_dir2[1], joint_dir2[2] = Rot2RPY_version2(Info[1][2])
# joint_dir3[0], joint_dir3[1], joint_dir3[2] = Rot2RPY_version2(Info[1][3])
# joint_dir4[0], joint_dir4[1], joint_dir4[2] = Rot2RPY_version2(Info[1][4])
# joint_dir5[0], joint_dir5[1], joint_dir5[2] = Rot2RPY_version2(Info[1][5])
# joint_dir6[0], joint_dir6[1], joint_dir6[2] = Rot2RPY_version2(Info[1][6])

# for i in range(3):
# 	joint_dir1[i]=DegToRad*joint_dir1[i]
# 	joint_dir2[i]=DegToRad*joint_dir2[i]
# 	joint_dir3[i]=DegToRad*joint_dir3[i]
# 	joint_dir4[i]=DegToRad*joint_dir4[i]
# 	joint_dir5[i]=DegToRad*joint_dir5[i]
# 	joint_dir6[i]=DegToRad*joint_dir6[i]
# 	joint_dir1[i]=joint_dir1[i]
# 	joint_dir2[i]=joint_dir2[i]
# 	joint_dir3[i]=joint_dir3[i]
# 	joint_dir4[i]=joint_dir4[i]
# 	joint_dir5[i]=joint_dir5[i]
# 	joint_dir6[i]=joint_dir6[i]

dummy_pos=[position0,position1,position2,position3,position4,position5,position6]
dummy_dir=[joint_dir0,joint_dir1,joint_dir2,joint_dir3,joint_dir4,joint_dir5,joint_dir6]
print('-------當joint在[0,0,0,0,0,0]時的姿態，由FK計算出來------')
for i in range(7):
	print('joint_dir',[i+1],np.round(dummy_dir[i],4))
for i in range(7):
	print('joint_position',[i+1], dummy_pos[i])
print('----------------------------------------')


##-------------------------position------------------------------##
for i in range(7):
	result = vrep.simxSetObjectPosition(clientID, dummy_handle[i], -1, dummy_pos[i], vrep.simx_opmode_oneshot)
	# vrep.simxSetObjectPosition(clientID, joint_handle[i], -1, dummy_pos[i], vrep.simx_opmode_blocking)
	# vrep.simxSetObjectPosition(clientID, Rjoint_handle[i], plane, dummy_pos[i], vrep.simx_opmode_blocking)

##-------------------------orientation------------------------------##

for i in range(7):
	# res,ori[i]=vrep.simxGetObjectOrientation(clientID,dummy_handle[i],dummy_handle[0],vrep.simx_opmode_blocking)
	# vrep.simxSetObjectOrientation(clientID,dummy_handle[i], plane, dummy_dir[i], vrep.simx_opmode_oneshot)
	# vrep.simxSetObjectOrientation(clientID, Rjoint_handle[i],plane, dummy_dir[i], vrep.simx_opmode_oneshot)
	res,joint_ori[i]=vrep.simxGetObjectOrientation(clientID,Rjoint_handle[i],plane,vrep.simx_opmode_blocking)




res,Sphere=vrep.simxGetObjectHandle(clientID,'Sphere',vrep.simx_opmode_blocking)
res,Sphere_pos = vrep.simxGetObjectPosition(clientID, Sphere, -1, vrep.simx_opmode_blocking)

d_EEF=np.round(tip_pos[0]-Sphere_pos[0],5)
print('d_EEF',d_EEF)


#
vrep.simxStopSimulation(clientID, vrep.simx_opmode_oneshot)

# Before closing the connection to V-REP, make sure that the last command sent out had time to arrive. You can guarantee this with (for example):
vrep.simxGetPingTime(clientID)

# Close the connection to V-REP
vrep.simxFinish(clientID)