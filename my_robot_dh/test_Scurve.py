import inverseKinematics as IK
import Kinematics as FK
from compute_robot_jacobian import compute_robot_jacobian
from robot_constraint_and_parameter import robot_constraint as  ITRI_Constraint
from robot_constraint_and_parameter import robot_parameter as ITRI_Parameter
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

degtorad = math.pi / 180  # 角度轉弧度

##---------系統模型
[DOF , DH_table , SamplingTime , RatedTorque , GearRatio]  = ITRI_Parameter()



##---------系統限制

[ J_PosLimit , J_VelLimit , J_AccLimit , J_JerkLimit , C_PosLimit , C_VelLimit , C_AccLimit , C_JerkLimit ] = ITRI_Constraint( GearRatio )

## Task Define

#----- Forward Kinematics -----

J_InitPos = [0 *degtorad , 10 * degtorad , -10 * degtorad , 20 *degtorad , 10 * degtorad , 10 * degtorad ]
J_FinalPos = [0 * degtorad , 50 * degtorad , -60 *degtorad , 30 * degtorad , 20 * degtorad , 300 * degtorad ]

[InitInfo, EulerAngle_vrep,InitEulerAngle, C_InitPos] = FK.ForwardKinemetics(J_InitPos, DH_table)
[FinalInfo, EulerAngle_vrep,FinalEulerAngle, C_FinalPos] = FK.ForwardKinemetics(J_FinalPos, DH_table)

## Trajectory Generator
## scurve軌跡規劃
InitialAngle = J_InitPos
FinalAngle = J_FinalPos
acc_lim =J_AccLimit[0]
a_avg =0.75
vec_lim = J_VelLimit[0]
sampling_t=SamplingTime
[JointCmd , Time] = s_curve (InitialAngle , FinalAngle ,  acc_lim , a_avg , vec_lim , sampling_t)
# print(JointCmd[0][0:6,1])
Trajectory=np.zeros((len(Time),3),np.float64)

for i in range(len(Time)):

    [ Info ,EulerAngle_vrep,  NowEulerAngle , NowPosition ] =FK.ForwardKinemetics(JointCmd[0][0:6,i],DH_table )

    Trajectory[i]=NowPosition
#    PoseRecord(i,:) = [ NowPosition NowEulerAngle ];

# creat_path('./Trajectory/')
# save_txt(path='./Trajectory/',name='Trajectory.txt',data=Trajectory,fmt='%f')



