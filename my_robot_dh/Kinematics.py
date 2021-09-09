import numpy as np
import math
import Rot2RPY

def ForwardKinemetics(JointAngle ,DH_table):
    # 讀取DH參數
    dof = len(DH_table)
    theta = np.zeros(dof)
    d = np.zeros(dof)
    a = np.zeros(dof)
    alpha = np.zeros(dof)

    for i in range(dof):
        a[i] = DH_table[i][2]
        alpha[i] = DH_table[i][3]
        d[i] = DH_table[i][1]
        theta[i] = DH_table[i][0]

    # 順向運動學
    JointPos = [[0,0,0]] # 各關節的座標位置矩陣 [ 3 x n ] , n = DOF
    JointDir = [[[1,0,0],[0,1,0],[0,0,1]]]  # 各關節的座標向量矩陣 [ 3 x 3n ], n = DOF  

    T = np.eye(4) 
    for i in range(0,dof):
        A = np.array([ 
            [math.cos(JointAngle[i]+theta[i])  ,      -1*math.sin(JointAngle[i]+theta[i])*math.cos(alpha[i])  ,      math.sin(JointAngle[i]+theta[i])*math.sin(alpha[i])  ,     a[i]*math.cos(JointAngle[i]+theta[i]) ],
            [math.sin(JointAngle[i]+theta[i])  ,     math.cos(JointAngle[i]+theta[i])*math.cos(alpha[i])      ,    -1*math.cos(JointAngle[i]+theta[i])*math.sin(alpha[i]) ,   a[i]*math.sin(JointAngle[i]+theta[i]) ],
            [          0                       ,       math.sin(alpha[i])                                     ,     math.cos(alpha[i])                                    ,               d[i]              ],
            [          0                       ,                      0                                       ,         0                                                 ,               1              ] ,
            ])
        T = T.dot(A)
        JointDir.append(T[0:3,0:3]) # 儲存關節的向量資訊

        JointPos.append(T[0:3,3]) # 儲存關節的座標位置
    # wrist center position
    End_Position = T[ 0 : 3 , 3 ] 
    Wrist_Center_R = T[ 0 : 3 , 0 : 3 ]
    # print(Wrist_Center_R)
    # 末端點位置
    Position = End_Position
    # 取 EulerAngle z-y-x
    EulerAngle = np.zeros(3)
    EulerAngle_vrep=np.zeros(3)
    [EulerAngle[0], EulerAngle[1], EulerAngle[2]] = Rot2RPY.  Rot2RPY(Wrist_Center_R)  #zyx
    [EulerAngle_vrep[0], EulerAngle_vrep[1], EulerAngle_vrep[2]] = Rot2RPY.Rot2RPY_version2(Wrist_Center_R)  #xyz
    Info = [JointPos,JointDir]

    return Info ,EulerAngle_vrep,EulerAngle,Position

