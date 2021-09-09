import inverseKinematics as IK
import Kinematics as FK
from compute_robot_jacobian import compute_robot_jacobian
from robot_constraint_and_parameter import robot_constraint
from creat_trajectory import creat_trajectory
from s_curve import s_curve
from IK_FindOptSol import FindOptSol
import numpy as np
import math
import os
import matplotlib.pyplot as plt

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

#這裡單位是 cm  吸嘴加0.068m
DH_table = np.array([[0,            0.345,  0.08,   math.pi / 2],
					 [0+math.pi / 2 , 0,  0.27,     0],
					 [0,             0,     0.09,    math.pi / 2],
					 [0,            0.295,  0,       -math.pi / 2],
					 [0,            0,      0,       math.pi / 2],
					 [0,       0.102+0.068, 0,          0]])


degtorad = math.pi / 180  # 角度轉弧度
JointAngle1 = [   0 ,  10*degtorad,  -10*degtorad,   20*degtorad, 10*degtorad  ,10*degtorad  ]
JointAngle2 = [   0 ,  50*degtorad,  -60*degtorad,   30*degtorad, 20*degtorad  ,300*degtorad  ]


Info, EulerAngle_vrep, EulerAngle1, Position1 = FK.ForwardKinemetics(JointAngle1, DH_table)
Info, EulerAngle_vrep, EulerAngle2, Position2 = FK.ForwardKinemetics(JointAngle2, DH_table)

#----------------軌跡規劃------------------#

#--------------起始點和末端點---------------#
InitialPosition = Position1
FinalPosition = Position2
#------------起始姿態和終點姿態(degree)------#
InitialPose = EulerAngle1
FinalPose = EulerAngle2
#--------------Time-----------------------#
Time_ini = 0
Time_final = 1
SamplingTime=0.001

[ Xcmd , Ycmd , Zcmd , AlphaCmd , BetaCmd , GammaCmd , Time]=creat_trajectory(InitialPosition, FinalPosition , InitialPose, FinalPose, Time_ini, Time_final)

#---------逆向幾何解求解---------------------#
[invsol1, flag] = IK.InverseKinemetics(InitialPose, InitialPosition, DH_table)
[invsol2, flag] = IK.InverseKinemetics(FinalPose , FinalPosition , DH_table)

optimalsol1=FindOptSol(invsol1,JointAngle1)
optimalsol2=FindOptSol(invsol2,JointAngle2)

#--------robotjacobian 求解----------------#

Nowjoint=JointAngle1

JointPosRecord=np.zeros((1,1),np.float64)
JointDirRecord=np.zeros((1,1),np.float64)
a_set=[]
NowPos_Position_EulerAngle=[]
NowPos_Position_position=[]

for i in range(len(Time)): #1000


    Info, EulerAngle_vrep, NowEulerAngle, NowPosition = FK.ForwardKinemetics(Nowjoint, DH_table)

    NowPos_Position_EulerAngle.append(NowEulerAngle)

    NowPos_Position_position.append(NowPosition)

    #計算 T(fi) 矩陣 3*3
    T_RPY=np.array([
        [0,-math.sin(NowEulerAngle[0]*degtorad), math.cos(NowEulerAngle[0]*degtorad)*math.cos(NowEulerAngle[1]*degtorad)],
        [0,math.cos(NowEulerAngle[0]*degtorad),math.sin(NowEulerAngle[0]*degtorad)*math.cos(NowEulerAngle[1]*degtorad)],
        [1 , 0 ,- math.sin(NowEulerAngle[1] *degtorad)]
        ])

    #基於robot jacobian軌跡

    RobotJacobian = compute_robot_jacobian(DH_table, Nowjoint) #6*6

    A=np.array([[AlphaCmd[i][1]*degtorad],[BetaCmd[i][1]*degtorad],[GammaCmd[i][1]*degtorad]]) #3*1

    Omega = T_RPY.dot(A)  #3*1

    B=np.array([[Xcmd[i][1]],[Ycmd[i][1]],[Zcmd[i][1]],[Omega[0][0]],[Omega[1][0]],[Omega[2][0]]])  #6*1
    JointVelocity = np.linalg.inv(RobotJacobian) .dot(B)  #6*1
    Nowjoint= Nowjoint + JointVelocity.transpose() * SamplingTime
    Nowjoint=np.squeeze(Nowjoint, axis=0).tolist()  #降維再轉list

PoseRecord=[NowPos_Position_position,NowPos_Position_EulerAngle]

TrajectoryError=np.zeros((6,len(Time)),np.float64)


for j in range(len(Time)):
    TrajectoryError[0][j] = abs(PoseRecord[0][j][0] - Xcmd[j][0])
    TrajectoryError[1][j] = abs(PoseRecord[0][j][1] - Ycmd[j][0])
    TrajectoryError[2][j] = abs(PoseRecord[0][j][2] - Zcmd[j][0])
    TrajectoryError[3][j] = abs(PoseRecord[1][j][0] - AlphaCmd[j][0])
    TrajectoryError[4][j] = abs(PoseRecord[1][j][1] - BetaCmd[j][0])
    TrajectoryError[5][j] = abs(PoseRecord[1][j][2] - GammaCmd[j][0])


for i in range(6):
    plt.figure(i)
    # plt.subplot(3, 2, i)
    plt.plot(TrajectoryError[i, : ], ':', color='y', label='Ps',linewidth=2.0 )
    plt.xlabel('Time (sec)')
    plt.ylabel('Position (rad)')
    plt.title('PoseError' + str(int(i)))
    plt.grid(True)
    plt.legend()
    plt.legend(loc='upper right')
plt.subplots_adjust(hspace=1.5, wspace=0.5)
plt.show()


#--------------------inverse_kinematic-----------------------------
#丟deg
# InitialPose = np.array(
#     [90, -45, 180])
# InitialPosition = np.array(
#     [0, 0.251, 0.208793])


# Info,EulerAngle_vrep,EulerAngle,Position=FK.ForwardKinemetics( JointAngle1,DH_table)
#
# Position=[0.17255957424640656, -0.0017113301437348127, 0.21242006123065948]
# EulerAngle=[0 ,   0 ,180]
# [invsol, flag] = IK.InverseKinemetics(EulerAngle, Position, DH_table)
# print('invsol',invsol)
# # print('flag',flag)
# for i in range(8):
#     Info, EulerAngle_vrep, EulerAngle, Position = FK.ForwardKinemetics(invsol[i], DH_table)
#     print('IK_answer',i+1,Position)
#
# # optimalsol=FindOptSol(invsol,JointAngle1)
# # print('optimalsol',optimalsol)
# # Info,EulerAngle_vrep,EulerAngle,Position=FK.ForwardKinemetics( optimalsol,DH_table)
# # print('EulerAngle',EulerAngle)
# # print('Position',Position)
# creat_path('./IK_ans/')
# save_txt(path='./IK_ans/',name='IK.txt',data=invsol[7],fmt='%f')

# #以下是測逆向出來的8組解
# ##----------------------Forward_kinematic------------------------------
# Info=np.zeros((3*2,), np.int)
# EulerAngle=np.zeros((8,3), np.float)
# Position=np.zeros((8,3), np.float)
# # Info,EulerAngle,Position=FK.ForwardKinemetics( JointAngle,DH_table)
# for i in range(8):
#     JointAngle = a[i]
#
#     Info,EulerAngle[i],Position[i]=FK.ForwardKinemetics( JointAngle,DH_table)
#     # print('info',Info)
#
# print('------------------Forward_Kinemetics---------------------------')
# for i in range(8):
#
#     print('FK_EulerAngle', i+1, EulerAngle[i])
#     print('FK_Position',i+1,np.round(Position[i],4))
#     print('----------------------------------')


