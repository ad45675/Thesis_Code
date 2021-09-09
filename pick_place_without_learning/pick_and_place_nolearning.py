import sim as vrep
from Rot2RPY import Rot2RPY,euler2mat,Rot2RPY_version2
import inverseKinematics as IK
import Kinematics as FK
from compute_robot_jacobian import compute_robot_jacobian
from robot_constraint_and_parameter import robot_constraint as  ITRI_Constraint
from robot_constraint_and_parameter import robot_parameter as ITRI_Parameter
from compute_robot_jacobian import compute_robot_jacobian
from s_curve import s_curve
from IK_FindOptSol import FindOptSol
from controller import controller
import numpy as np
import math
import os
# import matplotlib.pyplot as plt
import time

render=True

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
Rjoint_handle=np.zeros((7,),np.int)
joint_ori=np.zeros((7,3),np.float32)
joint_pos=np.zeros((7,3),np.float32)
joint_angles=np.zeros((6,),np.float32)

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
def load_txt(path,):
    f = open(path , 'r')
    data = np.loadtxt(f)
    f.close()
    return data
# path='C:/Users/user/Desktop/my_robot_dh/IK_ans/IK.txt'
# IK=load_txt(path)
# print('IK',IK)

joint_angle=np.zeros((6,),np.float32)
def get_joint_pos(joint_handle):
    for i in range(6):
        err, joint_angle[i] = vrep.simxGetJointPosition(clientID, joint_handle[i], vrep.simx_opmode_oneshot)
    return joint_angle
def get_object_height( handle):
    # 得到物體高度

    err, minval = vrep.simxGetObjectFloatParameter(clientID, handle, vrep.sim_objfloatparam_modelbbox_min_z,
                                                   vrep.simx_opmode_blocking)

    err, maxval = vrep.simxGetObjectFloatParameter(clientID, handle, vrep.sim_objfloatparam_modelbbox_max_z,
                                                   vrep.simx_opmode_blocking)

    return (maxval - minval) / 2


def enable_suction( active):
    if active:
        vrep.simxSetIntegerSignal(clientID, 'suctionPad_active', 1, vrep.simx_opmode_oneshot)
        _,value = vrep.simxGetIntegerSignal(clientID,'suctionPad_active', vrep.simx_opmode_blocking)
    else:
        vrep.simxSetIntegerSignal(clientID, 'suctionPad_active', 0, vrep.simx_opmode_oneshot)
        _,value = vrep.simxGetIntegerSignal(clientID, 'suctionPad_active',vrep.simx_opmode_blocking)
    return value
# --------------------get_vrep_handle --------------------#
joint_target_angle=[0,0,0,0,0,0]
for i in range(6):
# result, dummy_handle[i] = vrep.simxGetObjectHandle(clientID, 'Dummy'+str(i), vrep.simx_opmode_blocking)
    result, Rjoint_handle[i] = vrep.simxGetObjectHandle(clientID, 'joint' + str(i+1 ), vrep.simx_opmode_blocking)
    _, joint_angles = vrep.simxGetJointPosition(clientID, Rjoint_handle[i], vrep.simx_opmode_streaming)
    vrep.simxSetJointPosition(clientID, Rjoint_handle[i], joint_target_angle[i], vrep.simx_opmode_oneshot)

#--------------------------------------------------------------------------------------------------------------------------------


# --------------------物體位置和方位--------------------#
Cuboid_pos=(np.zeros(3,),np.float32)
_, Cuboid = vrep.simxGetObjectHandle(clientID, 'Cuboid', vrep.simx_opmode_blocking)
err, Cuboid_pos = vrep.simxGetObjectPosition(clientID, Cuboid , -1, vrep.simx_opmode_blocking)
Cuboid_pos[2] = get_object_height(Cuboid)+Cuboid_pos[2]  # 得到物體表面位置
# # --------------------末端點handle和位置和方位 --------------------#
res,tip=vrep.simxGetObjectHandle(clientID,'tip',vrep.simx_opmode_blocking)

# # --------------------地板放置位置 --------------------#
_, target_floor = vrep.simxGetObjectHandle(clientID, 'target_floor', vrep.simx_opmode_blocking)
err, target_floor_pos = vrep.simxGetObjectPosition(clientID, target_floor , -1, vrep.simx_opmode_blocking)
print('物體要放置的位置',target_floor_pos)


#---------------------------開始規劃軌跡-----------------------------#
# #---------系統模型
[DOF , DH_table , SamplingTime , RatedTorque , GearRatio]  = ITRI_Parameter()

# #---------系統限制

[ J_PosLimit , J_VelLimit , J_AccLimit , J_JerkLimit , C_PosLimit , C_VelLimit , C_AccLimit , C_JerkLimit ] = ITRI_Constraint( GearRatio )

## Task Define


FinalPos=[0,0,180]
# Cuboid_pos=[0.5540,-0.044,0.14980132058262824]
Cuboid_pos[2]=Cuboid_pos[2]+0.004
[tip_Jangle, flag,Singularplace] = IK.InverseKinematics(FinalPos, Cuboid_pos, DH_table)
initial_angle=get_joint_pos(Rjoint_handle)
Final_Jangle=FindOptSol(tip_Jangle,initial_angle,Singularplace)

#----- Forward Kinematics -----
J_InitPos = initial_angle
J_FinalPos = Final_Jangle
[InitInfo, EulerAngle_vrep,InitEulerAngle, C_InitPos] = FK.ForwardKinemetics(J_InitPos, DH_table)
[FinalInfo, EulerAngle_vrep,FinalEulerAngle, C_FinalPos] = FK.ForwardKinemetics(J_FinalPos, DH_table)

# Trajectory Generator
## scurve  joint 軌跡規劃
InitialAngle = J_InitPos
FinalAngle = J_FinalPos

acc_lim =J_AccLimit[0]
a_avg =0.75
vec_lim =J_VelLimit[0]
sampling_t=SamplingTime

[JointCmd , Time] = s_curve (InitialAngle , FinalAngle ,  acc_lim , a_avg , vec_lim , sampling_t)

# print('j',JointCmd[0][0:6,1])
#--------------------宣告--------------------#
Trajectory=np.zeros((len(Time),3),np.float64)
Trajectory_joint=np.zeros((len(Time),6),np.float64)
joint_out=np.zeros((len(Time)+1,6),np.float64)
joint_input=np.zeros((len(Time)+1,6),np.float64)
vs = np.zeros((len(Time)+1,6),np.float64)
joint_out[0]=InitialAngle
vrep_joint1=np.zeros((len(Time)+1,6),np.float64)
#--------------------宣告--------------------#

vs[0,0:6] = 0

success=False
vrep.simxStartSimulation(clientID, vrep.simx_opmode_oneshot)  #開始模擬
print('STEP1:準備下去吸取物體')
print('-------------------')
print('物體位置',Cuboid_pos)
value=enable_suction(False)
for i in range(len(Time)):

    joint_input[i] = np.transpose(JointCmd[0][0:6, i])
    joint_out[i+1,0:6],vs[i+1]=controller(joint_input[i],joint_out[i,0:6],vs[i])
    # print('j_in1', joint_input[0].shape, 'j_o1', joint_out[0,0:6].shape)

    if render:
        vrep.simxPauseCommunication(clientID, True)
        for j in range(6):
            vrep.simxSetJointTargetPosition(clientID, Rjoint_handle[j],joint_out[i+1, j], vrep.simx_opmode_oneshot)
        vrep.simxPauseCommunication(clientID, False)
        vrep.simxGetPingTime(clientID)

        vrep_joint1[i]=get_joint_pos(Rjoint_handle)

time.sleep(0.2)
value=enable_suction(True)

creat_path('./Trajectory/')
save_txt(path='./Trajectory/',name='Joint_input1.txt',data=joint_input,fmt='%f')
save_txt(path='./Trajectory/',name='joint_out1.txt',data=joint_out,fmt='%f')
save_txt(path='./Trajectory/',name='vrep_joint1.txt',data=vrep_joint1,fmt='%f')

print('STEPT2將物體抬起(要確認是否有吸到)')


##------------------------------------------------------第二階端規劃---------------------------------------------------##
##----------工作空間s-curve(物體往上抬30公分)---------##

heigh=0.3  #(m)要抬高幾公分
acc_lim=300
a_avg=0.75
vec_lim=20
#------------末端點現在位置 與 要到達的位置------------#
_,tip_position=vrep.simxGetObjectPosition(clientID, tip, -1, vrep.simx_opmode_blocking)

target_heigh=[tip_position[0],tip_position[1],tip_position[2]+heigh]

[Cartesian_Cmd , Time2] = s_curve (tip_position ,target_heigh ,  acc_lim , a_avg , vec_lim , sampling_t)

#get now joint pos
# joint_angles=get_joint_pos(Rjoint_handle)
now_joint_angles=joint_out[len(Time),0:6]

Trajectory2=np.zeros((len(Time2),3),np.float64)
joint_out2=np.zeros((len(Time2)+1,6),np.float64)

joint_out2[0]=now_joint_angles

joint_input2=np.zeros((len(Time2)+1,6),np.float64)
vs2 = np.zeros((len(Time2)+1,6),np.float64)
vs2[0,0:6] = 0

vrep_joint2=np.zeros((len(Time2)+1,6),np.float64)

print('卡式空間scurve,丟速度命令')
for i in range(len(Time2)):
    [Info, EulerAngle_vrep, NowEulerAngle, NowPosition] = FK.ForwardKinemetics(now_joint_angles, DH_table)
    ##-----position x,y,z
    Trajectory2[i] = NowPosition
    ##-----compute robot jacobian
    RobotJacobian=compute_robot_jacobian(DH_table,now_joint_angles)

    B=[Cartesian_Cmd[1][0][i],Cartesian_Cmd[1][1][i],Cartesian_Cmd[1][2][i],0,0,0]
    JointVelocity=np.linalg.inv(RobotJacobian) .dot(B)
    now_joint_angles=now_joint_angles+JointVelocity.transpose() * SamplingTime

    joint_input2[i] =now_joint_angles

    joint_out2[i+1,0:6],vs2[i+1]=controller(joint_input2[i],joint_out2[i,0:6],vs2[i])
    # print('j_in',joint_input2[0].shape,'j_o',joint_out2[0,0:6].shape)

    if render:
        vrep.simxPauseCommunication(clientID, True)
        for j in range(6):
            vrep.simxSetJointTargetPosition(clientID, Rjoint_handle[j],joint_out2[i+1,j], vrep.simx_opmode_oneshot)
        vrep.simxPauseCommunication(clientID, False)
        vrep.simxGetPingTime(clientID)
        vrep_joint2[i] = get_joint_pos(Rjoint_handle)

    # now_joint_angles = np.squeeze(now_joint_angles, axis=0).tolist()  # 降維再轉list
# print(now_joint_angles)
creat_path('./Trajectory/')
save_txt(path='./Trajectory/',name='Joint_input2.txt',data=joint_input2,fmt='%f')
save_txt(path='./Trajectory/',name='joint_out2.txt',data=joint_out2,fmt='%f')
save_txt(path='./Trajectory/',name='vrep_joint2.txt',data=vrep_joint2,fmt='%f')

#---------物體現在位置 跟 原本位置相減 看有沒有被吸起來
err, Cuboid_pos2 = vrep.simxGetObjectPosition(clientID, Cuboid, -1, vrep.simx_opmode_blocking)
Cuboid_pos2[2] = get_object_height(Cuboid) + Cuboid_pos2[2]  # 得到物體表面位置
pick_heigh = Cuboid_pos2[2] - Cuboid_pos[2]

if (pick_heigh > 0.25):
    success = True
    print('物體吸起來了~~準備移動')
else:
    success = False
    print('fail fail fail')


success = True  #測試用


if success:
    print('STEP3 移動到指定地點 軸空間(位置命令)')
    ##  inverse kinematic
    print('j_o2',joint_out2[(len(Time2))])

    InitialAngle = joint_out2[len(Time2)]
    FinalPos = [0, 0, 180]
    target_floor_pos[2]=target_floor_pos[2]+2*get_object_height(Cuboid)
    [target_floor_Jangle, flag,Singularplace] = IK.InverseKinematics(FinalPos,target_floor_pos, DH_table)
    FinalAngle= FindOptSol(target_floor_Jangle, InitialAngle,Singularplace)

    ##---------joint scurve----------##
    acc_lim =J_AccLimit[0]
    a_avg =0.75
    vec_lim =J_VelLimit[0]

    [JointCmd3 , Time3] = s_curve (InitialAngle , FinalAngle ,  acc_lim , a_avg , vec_lim , sampling_t)

    joint_input3=np.zeros((len(Time3)+1,6),np.float64)
    joint_out3=np.zeros((len(Time3)+1,6),np.float64)
    joint_out3[0]=InitialAngle
    vs3 = np.zeros((len(Time3) + 1, 6), np.float64)
    vs3[0, 0:6] = 0

    vrep_joint3 = np.zeros((len(Time3) + 1, 6), np.float64)

    for i in range(len(Time3)):

        joint_input3[i] = np.transpose(JointCmd3[0][0:6, i])
        joint_out3[i+1, 0:6],vs3[i + 1] = controller(joint_input3[i],joint_out3[i, 0:6], vs3[i])


        if render:
            vrep.simxPauseCommunication(clientID, True)
            for j in range(6):
                # print('JointCmd[0]',j,i, JointCmd[0][j, i])
                vrep.simxSetJointTargetPosition(clientID, Rjoint_handle[j],  joint_out3[i,j], vrep.simx_opmode_oneshot)
            vrep.simxPauseCommunication(clientID, False)
            vrep.simxGetPingTime(clientID)
            vrep_joint3[i] = get_joint_pos(Rjoint_handle)

    time.sleep(0.2)
    value=enable_suction(False)
    creat_path('./Trajectory/')
    save_txt(path='./Trajectory/', name='Joint_input3.txt', data=joint_input3, fmt='%f')
    save_txt(path='./Trajectory/', name='joint_out3.txt', data=joint_out3, fmt='%f')
    save_txt(path='./Trajectory/', name='vrep_joint3.txt', data=vrep_joint3, fmt='%f')

print('STEP4  BACK TO HOME  位置命令')
##  inverse kinematic

InitialAngle = joint_out3[len(Time3), 0:6]
FinalAngle=[0.0,0.524,-0.349,0.0,-0.785,0.0]
##---------joint scurve----------##

acc_lim = J_AccLimit[0]
a_avg = 0.75
vec_lim =J_VelLimit[0]

#--------------------------s_curve---------------------------#
[JointCmd4, Time4] = s_curve(InitialAngle, FinalAngle, acc_lim, a_avg, vec_lim, sampling_t)


#-------------------------- 宣告 -----------------------------#
Trajectory4 = np.zeros((len(Time4), 3), np.float64)
Trajectory_joint4 = np.zeros((len(Time4), 6), np.float64)
joint_out4=np.zeros((len(Time4)+1, 6), np.float64)

joint_out4[0]=InitialAngle

vs4 = np.zeros((len(Time4)+1,6),np.float64)
vs4[0,0:6] = 0
joint_input4=np.zeros((len(Time4)+1, 6), np.float64)

vrep_joint4 = np.zeros((len(Time4) + 1, 6), np.float64)


for i in range(len(Time4)):
    # [Info, EulerAngle_vrep, NowEulerAngle, NowPosition] = FK.ForwardKinemetics(JointCmd[0][0:6, i], DH_table)
    #
    # Trajectory3[i] = NowPosition
    joint_input4[i] = np.transpose(JointCmd4[0][0:6, i])
    joint_out4[i+1, 0:6], vs4[i + 1] = controller(joint_input4[i], joint_out4[i, 0:6], vs4[i])


    if render:
        vrep.simxPauseCommunication(clientID, True)
        for j in range(6):
            # print('JointCmd[0]',j,i, JointCmd[0][j, i])
            vrep.simxSetJointTargetPosition(clientID, Rjoint_handle[j],joint_out4[i, j], vrep.simx_opmode_oneshot)
        vrep.simxPauseCommunication(clientID, False)
        vrep.simxGetPingTime(clientID)
        vrep_joint4[i] = get_joint_pos(Rjoint_handle)
time.sleep(0.5)
creat_path('./Trajectory/')
save_txt(path='./Trajectory/',name='Joint_input4.txt',data=joint_input4,fmt='%f')
save_txt(path='./Trajectory/',name='joint_out4.txt',data=joint_out4,fmt='%f')
save_txt(path='./Trajectory/',name='vrep_joint4.txt',data=vrep_joint4 ,fmt='%f')


vrep.simxStopSimulation(clientID, vrep.simx_opmode_oneshot)


vrep.simxFinish(clientID)