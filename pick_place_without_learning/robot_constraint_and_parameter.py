import numpy as np
import math

DegToRad = math.pi/ 180
RadToDeg = 180/math.pi


def robot_parameter():
    DOF=6

    DH_table = np.array([[0, 0.345, 0.08, math.pi / 2],
                         [0 + math.pi / 2, 0, 0.27, 0],
                         [0, 0, 0.09, math.pi / 2],
                         [0, 0.295, 0, -math.pi / 2],
                         [0, 0, 0, math.pi / 2],
                         [0, 0.102 + 0.068, 0, 0]])

    SamplingTime=0.001

    ##馬達額定轉矩
    RatedTorque = [1.3, 1.3, 1.3, 0.32, 0.32, 0.32]
    ##馬達齒輪比
    GearRatio = [120.0, 120.0, 120.0, 102, 80.0, 51.0]

    return DOF,DH_table, SamplingTime,RatedTorque ,GearRatio

def robot_constraint( GearRatio):

    vel_limit=np.zeros(6,)
    acc_limit=np.zeros(6,)
    jerk_limit=np.zeros(6,)

    pos_limit = np.array([[-170*DegToRad , 170*DegToRad ],
                          [-85*DegToRad , 130*DegToRad ],
                          [-110*DegToRad , 170*DegToRad ],
                          [-190*DegToRad , 190*DegToRad ],
                          [-125*DegToRad , 125*DegToRad ],
                          [-360*DegToRad , 360*DegToRad ]])
    for i in range(6):
        #[2.4933, 2.4933, 2.9333, 3.7400, 5.8667]
        vel_limit[i]=((2000/60)/0.7)/GearRatio[i]*2*math.pi  #rad
        acc_limit[i]=vel_limit[i]*3                          #rad
        jerk_limit[i]=acc_limit[i]*3                         #rad


    PosCLimit = [500, 500, 500, 5, 5, 5]

    VelCLimit = [100, 100, 100, 5, 5, 5]

    AccCLimit = [300, 300, 300, 5, 5, 5]

    JerkCLimit = [900, 900, 900, 5, 5, 5]

    return pos_limit , vel_limit , acc_limit ,jerk_limit, PosCLimit , VelCLimit , AccCLimit , JerkCLimit
