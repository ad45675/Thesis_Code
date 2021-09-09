import math
import numpy as np

# Transfer Function
# -------------------------------------------------------


def RPY2Rot(A, B, C):
    # Deg to Rad
    Deg2Rad = math.pi/180
    # Deg to Rad
    alpha = A * Deg2Rad
    beta = B * Deg2Rad
    gamma = C * Deg2Rad
#### 尤拉角轉旋轉矩陣順序 zyx
    Rzyx = [[math.cos(alpha)*math.cos(beta), math.cos(alpha)*math.sin(beta)*math.sin(gamma) - math.cos(gamma)*math.sin(alpha), math.sin(alpha)*math.sin(gamma) + math.cos(alpha)*math.cos(gamma)*math.sin(beta)],
            [math.cos(beta)*math.sin(alpha), math.cos(alpha)*math.cos(gamma) + math.sin(alpha)*math.sin(beta) *
             math.sin(gamma), math.cos(gamma)*math.sin(alpha)*math.sin(beta) - math.cos(alpha)*math.sin(gamma)],
            [-math.sin(beta),                math.cos(beta)*math.sin(gamma),                                                   math.cos(beta)*math.cos(gamma)]]

#### 尤拉角轉旋轉矩陣順序 xyz
    Rxyz = [[math.cos(gamma) * math.cos(beta), -math.cos(beta) * math.sin(gamma) ,math.sin(beta) ],
            [math.cos(alpha)*math.sin(gamma)+math.cos(gamma)*math.sin(alpha)*math.sin(beta),math.cos(alpha)*math.cos(gamma)-math.sin(alpha)*math.sin(beta)*math.sin(gamma), math.cos(beta)*math.sin(alpha)],
            [math.sin(alpha)*math.sin(gamma)-math.cos(alpha)*math.cos(gamma)*math.sin(beta),math.cos(gamma)*math.sin(alpha)+math.cos(alpha)*math.sin(beta)*math.sin(gamma),math.cos(alpha)*math.cos(beta)]]

    return Rzyx

# Adjust if angle > 180 or angle < -180
# -------------------------------------------------------------


def over180_rad(input_rad):
    angle = input_rad

    if input_rad > math.pi:
        # print('bigger')
        angle = input_rad - 2*math.pi

    if input_rad < -math.pi:
        # print('smaller')
        angle = input_rad + 2*math.pi

    return angle

# Calculate T: output T
# --------------------------------------------------------


def calculateT(theta, alpha, a, d):
    ### RotZ(4, 4) ###################
    RotZ = np.zeros((4, 4))

    RotZ[0] = [1, 0, 0, 0]
    RotZ[1] = [0, math.cos(alpha), -math.sin(alpha), 0]
    RotZ[2] = [0, math.sin(alpha), math.cos(alpha), 0]
    RotZ[3] = [0, 0, 0, 1]

    ### TranZ(4, 4) ###################
    TransZ = np.zeros((4, 4))

    TransZ[0] = [1, 0, 0, a]
    TransZ[1] = [0, 1, 0, 0]
    TransZ[2] = [0, 0, 1, 0]
    TransZ[3] = [0, 0, 0, 1]

    ### TransX(4, 4) ##################
    TransX = np.zeros((4, 4))

    TransX[0] = [1, 0, 0, 0]
    TransX[1] = [0, 1, 0, 0]
    TransX[2] = [0, 0, 1, d]
    TransX[3] = [0, 0, 0, 1]

    ### RotX(4, 4) ###################
    RotX = np.zeros((4, 4))

    RotX[0] = [math.cos(theta), -math.sin(theta), 0, 0]
    RotX[1] = [math.sin(theta), math.cos(theta), 0, 0]
    RotX[2] = [0, 0, 1, 0]
    RotX[3] = [0, 0, 0, 1]

    # print(RotZ)
    # print(RotX)
    # print(TransX)
    # print(TransZ)

    T = RotX@TransX@TransZ@RotZ

    return T

# Calculate J1: output J1, J1p
# use function: over180_rad
# -------------------------------------------------------------


def calculateJ1(EEx, EEy, EEz, z1, z2, z3, DH_table):


    # Read DH_table
    dof = len(DH_table)

    a = np.zeros(dof)
    alpha = np.zeros(dof)
    d = np.zeros(dof)
    theta = np.zeros(dof)

    for i in range(dof):
        a[i] = DH_table[i][2]
        alpha[i] = DH_table[i][3]
        d[i] = DH_table[i][1]
        theta[i] = DH_table[i][0]

    # Solve Angle1

    Ex = EEx - d[5]*z1
    Ey = EEy - d[5]*z2
    Ez = EEz - d[5]*z3
    # print('eey',Ey,Ex,'eey',EEy, d[5],z2)



    J1 = math.atan2(Ey, Ex)


    J1p = math.atan2(Ey, Ex) + math.pi

    # 保持在 正負 180 內
    J1 = over180_rad(J1)

    J1p = over180_rad(J1p)

    return J1, J1p

# Calculate J2 J3: output J2 J2p J3 J3p
# call function: calculateT over180_rad
# -----------------------------------------------------


def calculateJ2J3(Ex, Ey, Ez, J1, DH_table):
    # Read DH_table
    singular = False
    dof = len(DH_table)

    a = np.zeros(dof)
    alpha = np.zeros(dof)
    d = np.zeros(dof)
    theta = np.zeros(dof)

    for i in range(dof):
        a[i] = DH_table[i][2]
        alpha[i] = DH_table[i][3]
        d[i] = DH_table[i][1]
        theta[i] = DH_table[i][0]

    T1 = calculateT(J1, alpha[0], a[0], d[0])

    A1 = T1

    Bx = A1[0][3]
    By = A1[1][3]
    Bz = A1[2][3]

    # print('B', Bx,By,Bz)
    P = 0
    AB = a[0]
    # print('AB',AB)

    # if abs(Bx) > 0.01:
    if Ex*Bx < 0:  #(2,3)
        P = -math.sqrt(Ex*Ex+Ey*Ey) - AB

    else:          #(1,4)
        P = math.sqrt(Ex*Ex+Ey*Ey) - AB

    # elif abs(By) > 0.01:
    #     if Ey*By < 0: #(2,3)
    #         P = -math.sqrt(Ex*Ex+Ey*Ey) - AB
    # 
    #     else:         #(1,4)
    #         P = math.sqrt(Ex*Ex+Ey*Ey) - AB
    # 
    # else:
    #     P = math.sqrt(Ex*Ex+Ey*Ey) - AB


    OA = d[0]
    Q = Ez - OA
    BC = a[1]
    CD = a[2]
    DE = d[3]
    BE = math.sqrt(P*P + Q*Q)
    CE = math.sqrt(CD*CD + DE*DE)

    # check singular
    if((BC + CE < BE) or (CE - BC >BE)):
        J2, J2p, J3, J3p = 0, 0, 0, 0
        singular = True
        # break
    else:
        angleCBE = math.acos((BC * BC + BE * BE - CE * CE) / (2 * BC * BE))
        # except ValueError:
        #     print('IK fail')


        # 使用 arctan2 函數以修正 arctan 給出的角度值
        angleEBU = math.atan2(Q, P)

        J2 = -(math.pi/2 - (angleCBE + angleEBU))
        J2p = -(math.pi/2 - (angleEBU - angleCBE))
        # 保持在 正負 180 內
        J2 = over180_rad(J2)
        J2p = over180_rad(J2p)

        ans2=(BC**2 + CE**2 - BE**2)/(2*BC*CE)
        angleBCE = math.acos(ans2)
        angleECD = math.acos(CD/CE)
        J3 = angleBCE + angleECD - math.pi  # elbow up
        J3p = math.pi - (angleBCE - angleECD)  # elbow down
        # 保持在 正負 180 內
        J3 = over180_rad(J3)
        J3p = over180_rad(J3p)

        singular = False
    return J2, J2p, J3, J3p,singular

# Calculate J4, J5, J6: output J4 J4p J5 J5p J6 J6p
# call function:
# ------------------------------------------------------


def calculateJ4J5J6(EEx, EEy, EEz, x1, x2, x3, J1, J2, J3, DH_table):
    dof = len(DH_table)

    a = np.zeros(dof)
    alpha = np.zeros(dof)
    d = np.zeros(dof)
    theta = np.zeros(dof)

    for i in range(dof):
        a[i] = DH_table[i][2]
        alpha[i] = DH_table[i][3]
        d[i] = DH_table[i][1]
        theta[i] = DH_table[i][0]

    # 解 Angle4  Angle5 Angle6

    T1 = calculateT(J1,  alpha[0],  a[0],  d[0])
    T2 = calculateT(J2+(math.pi/2),  alpha[1],  a[1],  d[1])
    T3 = calculateT(J3,  alpha[2],  a[2],  d[2])
    A3 = T1@T2@T3

    newEEx = EEx - A3[0][3]
    newEEy = EEy - A3[1][3]
    newEEz = EEz - A3[2][3]

    xProjection = newEEx*A3[0][0] + newEEy*A3[1][0] + newEEz*A3[2][0]
    yProjection = newEEx*A3[0][1] + newEEy*A3[1][1] + newEEz*A3[2][1]
    zProjection = newEEx*A3[0][2] + newEEy*A3[1][2] + newEEz*A3[2][2]

    J4 = math.atan2(yProjection, xProjection)
    J4p = math.atan2(yProjection, xProjection) + math.pi
    # 增加一個機制 若超過180度 自動變負號
    J4 = over180_rad(J4)
    J4p = over180_rad(J4p)


    EExdir1 = xProjection*A3[0][0]
    EExdir2 = xProjection*A3[1][0]
    EExdir3 = xProjection*A3[2][0]

    EEydir1 = yProjection*A3[0][1]
    EEydir2 = yProjection*A3[1][1]
    EEydir3 = yProjection*A3[2][1]

    EEinXYprojection1 = EExdir1 + EEydir1
    EEinXYprojection2 = EExdir2 + EEydir2
    EEinXYprojection3 = EExdir3 + EEydir3

    EEinXYprojectionValue = math.sqrt(EEinXYprojection1*EEinXYprojection1 +
                                      EEinXYprojection2*EEinXYprojection2 + EEinXYprojection3*EEinXYprojection3)

    # 會用102 是因為末端長度是102，如果用newEEvalue，那是末端長度和前一軸長度相加
    J5temp1cos = (zProjection - d[3])/d[5]
    J5temp2cos = (newEEx*EEinXYprojection1 + newEEy*EEinXYprojection2 +
                  newEEz*EEinXYprojection3) / (d[5]*EEinXYprojectionValue)

    if J5temp1cos > 1:
        J5temp1cos = 1

    if J5temp2cos > 1:
        J5temp2cos = 1

    if J5temp1cos < -1:
        J5temp1cos = -1

    if J5temp2cos < -1:
        J5temp2cos = -1

    # 如果有算術誤差，將其消除
    J5temp1 = math.acos(J5temp1cos)
    J5temp2 = math.acos(J5temp2cos)

    if J5temp2 > math.pi/2:
        J5 = J5temp1
    else:
        J5 = J5temp1

    J5p = -J5

    # 增加一個機制 若超過180度 自動變負號
    J5 = over180_rad(J5)
    J5p = over180_rad(J5p)

    T4 = calculateT(J4,  alpha[3],  a[3],  d[3])
    T5 = calculateT(J5,  alpha[4],  a[4],  d[4])

    A5 = T1@T2@T3@T4@T5

    J6temp1cos = x1*A5[0][0] + x2*A5[1][0] + x3*A5[2][0]
    J6temp2cos = x1*A5[0][1] + x2*A5[1][1] + x3*A5[2][1]

    if J6temp1cos > 1:
        J6temp1cos = 1

    if J6temp2cos > 1:
        J6temp2cos = 1

    if J6temp1cos < -1:
        J6temp1cos = -1

    if J6temp2cos < -1:
        J6temp2cos = -1

    # 如果有算術誤差，將其消除
    J6temp1 = math.acos(J6temp1cos)
    J6temp2 = math.acos(J6temp2cos)

    if J6temp2 > math.pi/2:
        J6 = -J6temp1
        J6p = -J6temp1 + math.pi

    else:
        J6 = J6temp1
        J6p = J6temp1 + math.pi

    # 增加一個機制 若超過180度 自動變負號
    J6 = over180_rad(J6)
    J6p = over180_rad(J6p)

    return J4, J4p, J5, J5p, J6, J6p


def InverseKinematics(EulerAngle, Position, DH_table):
    SingularFlag = 0  # 看看是否在奇異點 0:奇異點, 1:可解點
    SingularFlag1 = False
    SingularFlag2 = False
    Singularplace = np.zeros(8)

    # 讀取DH參數
    dof = len(DH_table)

    a = np.zeros(dof)
    alpha = np.zeros(dof)
    d = np.zeros(dof)
    theta = np.zeros(dof)

    for i in range(dof):
        a[i] = DH_table[i][2]
        alpha[i] = DH_table[i][3]
        d[i] = DH_table[i][1]
        theta[i] = DH_table[i][0]

    ### calculate wrist center position ############################

    TEEFPosture = np.array([EulerAngle[0], EulerAngle[1], EulerAngle[2]])
    EEF_Matr = RPY2Rot(TEEFPosture[0], TEEFPosture[1], TEEFPosture[2])
##這裡position單位是cm
    nsdt = np.array([[EEF_Matr[0][0], EEF_Matr[0][1], EEF_Matr[0][2], Position[0]],
                     [EEF_Matr[1][0], EEF_Matr[1][1], EEF_Matr[1][2], Position[1]],
                     [EEF_Matr[2][0], EEF_Matr[2][1], EEF_Matr[2][2], Position[2]],
                     [0,              0,              0,              1, ]])



    z1 = nsdt[0][2]
    z2 = nsdt[1][2]
    z3 = nsdt[2][2]
    x1 = nsdt[0][0]
    x2 = nsdt[1][0]
    x3 = nsdt[2][0]
    EEx = nsdt[0][3]
    EEy = nsdt[1][3]
    EEz = nsdt[2][3]

    # -----------------------------------------------

    J1s = np.zeros(2)
    J2s = np.zeros(4)
    J3s = np.zeros(4)
    J4s = np.zeros(8)
    J5s = np.zeros(8)
    J6s = np.zeros(8)

    J1s[0], J1s[1] = calculateJ1(EEx, EEy, EEz, z1, z2, z3, DH_table)


    Ex = EEx - d[5]*z1
    Ey = EEy - d[5]*z2
    Ez = EEz - d[5]*z3

    J2s[0], J2s[1], J3s[0], J3s[1], SingularFlag1 = calculateJ2J3(Ex, Ey, Ez, J1s[0], DH_table)
    J2s[2], J2s[3], J3s[2], J3s[3], SingularFlag2 = calculateJ2J3(Ex, Ey, Ez, J1s[1], DH_table)


    JointAngle = np.zeros((8, 6))
    if not SingularFlag1:
        J4s[0], J4s[1], J5s[0], J5s[1], J6s[0], J6s[1] = calculateJ4J5J6( EEx, EEy, EEz, x1, x2, x3, J1s[0], J2s[0], J3s[0], DH_table)
        J4s[2], J4s[3], J5s[2], J5s[3], J6s[2], J6s[3] = calculateJ4J5J6( EEx, EEy, EEz, x1, x2, x3, J1s[0], J2s[1], J3s[1], DH_table)

        # # 返回參數的實部
        # J1s = np.real(J1s)
        # J2s = np.real(J2s)
        # J3s = np.real(J3s)
        # J4s = np.real(J4s)

        JointAngle[0] = [J1s[0], J2s[0], J3s[0], J4s[0], J5s[0], J6s[0]]
        JointAngle[1] = [J1s[0], J2s[0], J3s[0], J4s[1], J5s[1], J6s[1]]
        JointAngle[2] = [J1s[0], J2s[1], J3s[1], J4s[2], J5s[2], J6s[2]]
        JointAngle[3] = [J1s[0], J2s[1], J3s[1], J4s[3], J5s[3], J6s[3]]
        Singularplace[0] = 1
        Singularplace[1] = 1
        Singularplace[2] = 1
        Singularplace[3] = 1

    if not SingularFlag2:
        J4s[4], J4s[5], J5s[4], J5s[5], J6s[4], J6s[5] = calculateJ4J5J6(EEx, EEy, EEz, x1, x2, x3, J1s[1], J2s[2], J3s[2], DH_table)
        J4s[6], J4s[7], J5s[6], J5s[7], J6s[6], J6s[7] = calculateJ4J5J6(EEx, EEy, EEz, x1, x2, x3, J1s[1], J2s[3], J3s[3], DH_table)

        # J5s = np.real(J5s)
        # J6s = np.real(J6s)

        JointAngle[4] = [J1s[1], J2s[2], J3s[2], J4s[4], J5s[4], J6s[4]]
        JointAngle[5] = [J1s[1], J2s[2], J3s[2], J4s[5], J5s[5], J6s[5]]
        JointAngle[6] = [J1s[1], J2s[3], J3s[3], J4s[6], J5s[6], J6s[6]]
        JointAngle[7] = [J1s[1], J2s[3], J3s[3], J4s[7], J5s[7], J6s[7]]

        Singularplace[4] = 1
        Singularplace[5] = 1
        Singularplace[6] = 1
        Singularplace[7] = 1

    if(SingularFlag1 and SingularFlag2):
        SingularFlag = True

    return JointAngle, SingularFlag,Singularplace


# Parameters for testing
# DH_table = np.array([[0,         34.5,   8.0,     math.pi/2],
#                      [math.pi/2, 0,      27.0,    0],
#                      [0,         0,      9.0,     math.pi/2],
#                      [0,         29.5,   0,       -math.pi/2],
#                      [0,         0,      0,       math.pi/2],
#                      [0,         10.2,   0,       0]])

# InitialPose = np.array(
#     [-109.063000000000, -58.4261775843947, -71.842000000000])
# InitialPosition = np.array(
#     [42.856538283717398, -0.605789981061624, 71.754203625228271])

# JointAngle, SingularFlag = InverseKinemetics(
#     InitialPose, InitialPosition, DH_table)
# print('Joint Angle = ', JointAngle)
# print('Flag = ', SingularFlag)
