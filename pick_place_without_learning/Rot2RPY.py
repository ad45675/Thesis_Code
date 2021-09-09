import math
import numpy as np
RadToDeg = 180/math.pi
def euler2mat(a, b, g):
    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    Rz = np.array([[np.cos(g), -np.sin(g), 0], [np.sin(g), np.cos(g), 0], [0, 0, 1]])
    temp = np.matmul(Ry, Rz)
    R = np.matmul(Rx, temp)
    return R


# Rot2RPY_version2 是對的
# Rot2RPY 有點問題

def Rot2RPY_version2(T):    ## matrix轉尤拉角 XYZ
    R = T[0:3, 0:3]
    cy_thresh = 0
    _FLOAT_EPS_4 = np.finfo(float).eps * 4.0
    try:
        cy_thresh = np.finfo(R.dtype).eps * 4
    except ValueError:
        cy_thresh = _FLOAT_EPS_4
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = R.flat
    cy = math.sqrt(r33*r33 + r23*r23)
    if cy > cy_thresh: # cos(y) not close to zero, standard form
        z = math.atan2(-r12,  r11) # atan2(cos(y)*sin(z), cos(y)*cos(z))
        y = math.atan2(r13,  cy) # atan2(sin(y), cy)
        x = math.atan2(-r23, r33) # atan2(cos(y)*sin(x), cos(x)*cos(y))
    else: # cos(y) (close to) zero, so x -> 0.0 (see above)
        # so r21 -> sin(z), r22 -> cos(z) and
        z = math.atan2(r21,  r22)
        y = math.atan2(r13,  cy) # atan2(sin(y), cy)
        x = 0.0
    # E = np.array([x, y, z])



    x = x*RadToDeg
    y = y*RadToDeg
    z = z*RadToDeg
    return x,y,z


#matrix轉尤拉角 zyx
def Rot2RPY(matr3by3):
    # Rad to mDeg ---------------------------------------------
    RadTomDeg = 180000/math.pi
    RadToDeg = 180/math.pi


    # judge ABC Posture
    if ((abs(matr3by3[0,0])-1.7e-5) < 0 and (abs(matr3by3[1,0])-1.7e-5) < 0):
        if (matr3by3[2,0] >= 0):
            B1 = -math.pi/2
            A1 = 0
            C1 = math.atan2(-matr3by3[1,2], -matr3by3[0,2])
        elif (matr3by3[2,0] < 0):
            B1 = math.pi/2
            A1 = 0
            C1 = math.atan2(matr3by3[1,2], matr3by3[0,2])
    else:
        # B1 = math.atan2(-matr3by3[2,0],  math.sqrt(matr3by3[0,0]*matr3by3[0,0] + matr3by3[1,0]*matr3by3[1,0]))
        B1 = math.atan2(-matr3by3[2, 0], math.sqrt(matr3by3[2, 1] * matr3by3[2, 1] + matr3by3[2, 2] * matr3by3[2, 2]))

        if ( (math.cos(B1) + 0.9999999) < 0 ):  
             B1 = B1 + 0.0011111

        if (math.cos(B1) > 0.0111111):
            # A1 = math.atan2(matr3by3[2,1], matr3by3[2,2])
            # C1 = math.atan2(matr3by3[1,0], matr3by3[0,0])
            A1 = math.atan2(matr3by3[1,0], matr3by3[0,0])
            C1 = math.atan2(matr3by3[2,1], matr3by3[2,2])
        elif (math.cos(B1) < 0.0111111):
            # A1 = math.atan2(-matr3by3[2,1], -matr3by3[2,2])
            # C1 = math.atan2(-matr3by3[1,0], -matr3by3[0,0])
            A1 = math.atan2(-matr3by3[1,0], -matr3by3[0,0])
            C1 = math.atan2(-matr3by3[2,1], -matr3by3[2,2])

    A1 = A1*RadTomDeg
    B1 = B1*RadToDeg
    C1 = C1*RadTomDeg

    A1 = round(A1)
    C1 = round(C1)

    if (abs(A1) == 180000):
        A1 = 180
    else:
        A1 = A1/1000
    if (abs(C1) == 180000):
        C1 = 180
    else:
        C1 = C1/1000
    return A1,B1,C1
# 對z旋轉A1 對Y旋轉B1 對X旋轉C1