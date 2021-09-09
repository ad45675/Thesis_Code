import numpy as np
import math

def creat_trajectory(initial_point, final_point, initial_pose, final_pose,time_ini,time_final):
    Ti = time_ini
    Tf = time_final

    # ----位置----#
    Px_ini, Py_ini, Pz_ini = initial_point[0], initial_point[1], initial_point[2]
    alpha_ini, beta_ini, gamma_ini = initial_pose[0], initial_pose[1], initial_pose[2]
    Px_final, Py_final, Pz_final = final_point[0], final_point[1], final_point[2]
    alpha_final, beta_final, gamma_final = final_pose[0], final_pose[1], final_pose[2]

    # ----速度----#
    Vx_ini, Vy_ini, Vz_ini = 0, 0, 0
    V_alpha_ini, V_beta_ini, V_gamma_ini = 0, 0, 0
    Vx_final, Vy_final, Vz_final = 0, 0, 0
    V_alpha_final, V_beta_final, V_gamma_final = 0, 0, 0

    # ----加速度----#
    Ax_ini, Ay_ini, Az_ini = 0, 0, 0
    A_alpha_ini, A_beta_ini, A_gamma_ini = 0, 0, 0
    Ax_final, Ay_final, Az_final = 0, 0, 0
    A_alpha_final, A_beta_final, A_gamma_final = 0, 0, 0


    #------5皆多項式時間矩陣------#
    B = np.array([
        [math.pow(Ti, 5), 5 * math.pow(Ti, 4), 20 * math.pow(Ti, 3), math.pow(Tf, 5), 5 * math.pow(Tf, 4),
         20 * math.pow(Tf, 3)],
        [math.pow(Ti, 4), 4 * math.pow(Ti, 3), 12 * math.pow(Ti, 2), math.pow(Tf, 4), 4 * math.pow(Tf, 3),
         12 * math.pow(Tf, 2)],
        [math.pow(Ti, 3), 3 * math.pow(Ti, 2), 6 * Ti, math.pow(Tf, 3), 3 * math.pow(Tf, 2), 6 * Tf],
        [math.pow(Ti, 2), 2 * Ti, 2, math.pow(Tf, 2), 2 * Tf, 2],
        [Ti, 1, 0, Tf, 1, 0],
        [1, 0, 0, 1, 0, 0],
    ])


    # ------5皆多項式位置 速度 加速度 初始化矩陣 ------#
    R = np.array([
        [Px_ini, Vx_ini, Ax_ini, Px_final, Vx_final, Ax_final],
        [Py_ini, Vy_ini, Ay_ini, Py_final, Vy_final, Ay_final],
        [Pz_ini, Vz_ini, Az_ini, Pz_final, Vz_final, Az_final],
        [alpha_ini, V_alpha_ini, A_alpha_ini, alpha_final, V_alpha_final, A_alpha_final],
        [beta_ini, V_beta_ini, A_beta_ini, beta_final, V_beta_final, A_beta_final],
        [gamma_ini, V_gamma_ini, A_gamma_ini, gamma_final, V_gamma_final, A_gamma_final]
    ])

    A=R.dot(np.linalg.inv(B))

    SamplingTime=0.001

    a=0
    Xcmd_record=np.zeros((1000,3))
    Ycmd_record = np.zeros((1000,3))
    Zcmd_record = np.zeros((1000,3))
    Apha_cmd_record = np.zeros((1000,3))
    Beta_cmd_record = np.zeros((1000,3))
    Gamma_cmd_record = np.zeros((1000,3))
    Time=np.zeros((1000,1),np.float64)
    T=int((Tf-Ti)/SamplingTime)
    t=0
    for j in range(Ti,T):  #1000
        t = t +SamplingTime
        #-----位置------#
        Xcmd_record[a][0] = A[0][0] * math.pow(t, 5) + A[0][1] * math.pow(t, 4) + A[0][2] * math.pow(t, 3) + A[0][3] * math.pow(t, 2) + A[0][4] * t + A[0][5]
        Ycmd_record[a][0] = A[1][0] * math.pow(t, 5) + A[1][1] * math.pow(t, 4) + A[1][2] * math.pow(t, 3) + A[1][3] * math.pow(t, 2) + A[1][4] * t + A[1][5]
        Zcmd_record[a][0] = A[2][0] * math.pow(t, 5) + A[2][1] * math.pow(t, 4) + A[2][2] * math.pow(t, 3) + A[2][3] * math.pow(t, 2) + A[2][4] * t + A[2][5]
        Apha_cmd_record[a][0] = A[3][0] * math.pow(t, 5) + A[3][1] * math.pow(t, 4) + A[3][2] * math.pow(t, 3) + A[3][3] * math.pow(t, 2) + A[3][4] * t + A[3][5]
        Beta_cmd_record[a][0] = A[4][0] * math.pow(t, 5) + A[4][1] * math.pow(t, 4) + A[4][2] * math.pow(t, 3) + A[4][3] * math.pow(t, 2) + A[4][4] * t + A[4][5]
        Gamma_cmd_record[a][0] = A[5][0] * math.pow(t, 5) + A[5][1] * math.pow(t, 4) + A[5][2] * math.pow(t, 3) + A[5][ 3] * math.pow(t, 2) + A[5][4] * t + A[5][5]
        #----速度------#
        Xcmd_record[a][1] = 5 * A[0][0] * math.pow(t, 4) + 4 * A[0][1] * math.pow(t, 3) + 3 * A[0][2] * math.pow(t,2) + 2 * A[0][3] * math.pow(t, 1) + A[0][4]
        Ycmd_record[a][1] = 5 * A[1][0] * math.pow(t, 4) + 4 * A[1][1] * math.pow(t, 3) + 3 * A[1][2] * math.pow(t, 2) + 2 * A[1][3] * math.pow(t, 1) + A[1][4]
        Zcmd_record[a][1] = 5 * A[2][0] * math.pow(t, 4) + 4 * A[2][1] * math.pow(t, 3) + 3 * A[2][2] * math.pow(t,2) + 2 * A[2][3] * math.pow(t, 1) + A[2][4]
        Apha_cmd_record[a][1] = 5 * A[3][0] * math.pow(t, 4) + 4 * A[3][1] * math.pow(t, 3) + 3 * A[3][2] * math.pow(t, 2) + 2 * A[3][3] * math.pow(t, 1) + A[3][4]
        Beta_cmd_record[a][1] = 5 * A[4][0] * math.pow(t, 4) + 4 * A[4][1] * math.pow(t, 3) + 3 * A[4][2] * math.pow(t, 2) + 2 * A[4][3] * math.pow(t, 1) + A[4][4]
        Gamma_cmd_record[a][1] = 5 * A[5][0] * math.pow(t, 4) + 4 * A[5][1] * math.pow(t, 3) + 3 * A[5][2] * math.pow(t,2) + 2 * A[5][3] * math.pow(t, 1) + A[5][4]
        # ----加速度------#
        Xcmd_record[a][2] = 20 * A[0][0] * math.pow(t, 3) + 12 * A[0][1] * math.pow(t, 2) + 6 * A[0][2] * math.pow(t, 1) + 2 * A[0][3]
        Ycmd_record[a][2] = 20 * A[1][0] * math.pow(t, 3) + 12 * A[1][1] * math.pow(t, 2) + 6 * A[1][2] * math.pow(t, 1) + 2 * A[1][3]
        Zcmd_record[a][2] = 20 * A[2][0] * math.pow(t, 3) + 12 * A[2][1] * math.pow(t, 2) + 6 * A[2][2] * math.pow(t, 1) + 2 * A[2][3]
        Apha_cmd_record[a][2] = 20 * A[3][0] * math.pow(t, 3) + 12 * A[3][1] * math.pow(t, 2) + 6 * A[3][2] * math.pow(t, 1) + 2 * A[3][3]
        Beta_cmd_record[a][2] = 20 * A[4][0] * math.pow(t, 3) + 12 * A[4][1] * math.pow(t, 2) + 6 * A[4][2] * math.pow(t, 1) + 2 * A[4][3]
        Gamma_cmd_record[a][2] = 20 * A[5][0] * math.pow(t, 3) + 12 * A[5][1] * math.pow(t, 2) + 6 * A[5][2] * math.pow(t, 1) + 2 * A[5][3]
        Time[a][0] = t
        a=a+1
    return Xcmd_record,Ycmd_record,Zcmd_record, Apha_cmd_record, Beta_cmd_record,Gamma_cmd_record,Time
