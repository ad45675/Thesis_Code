import math
import numpy as np

def s_curve(InitialAngle,FinalAngle,acc_lim,a_avg,vec_lim,SamplingTime):

    # -----------------------------------------------------
    #  acc_lim           acceleration limitation (m/sec^2)
    #  acc_avg           average accleration = acc_lim * a_avg (m/sec^2)
    #  vec_lim           velocity limitation (m/sec)
    #  sampling_t     sampling t (s)
    # -----------------------------------------------------

    ##Calculate Displacement
    Axis = len(InitialAngle)

    Displacement=np.zeros((Axis),np.float64)
    for i in range(Axis):
        Displacement[i] = FinalAngle[i] - InitialAngle[i]

    Max_Dis = np.amax(Displacement)
    max_num = np.where(Max_Dis == Displacement)

    acc_avg = a_avg * acc_lim

    if (Displacement[max_num] < 0):
        acc_lim = -acc_lim
        vec_lim = -vec_iim
        acc_avg = -acc_avg

    ## Calculate t
    Ta = vec_lim / acc_avg
    Tb = 2 * vec_lim / acc_lim - Ta
    Tc = (Ta - Tb) / 2
    Ts = (Displacement[max_num] - vec_lim * Ta) / vec_lim
    if (Ts.all() < 0):
        Ts = 0
    t1 = Tc
    t2 = Tc + Tb
    t3 = Ta
    t4 = Ta + Ts
    t5 = Ta + Ts + Tc
    t6 = Ta + Ts + Tc + Tb
    t7 = Ta + Ts + Ta

    ## Displacement Function
    DisplacementFunction = 1/6 * math.pow(t1,3) + 1/6 * math.pow(t2,3) + 1/3 * math.pow(t3,3) +  1/6 * math.pow(t4,3) - 1/6 * math.pow(t5,3) - 1/6 * math.pow(t6,3) + 1/6 * math.pow(t7,3)\
    - 1/2 * (t1 * math.pow(t3,2)) - 1/2 * (t7 * math.pow(t1,2)) - 1/2 * t2 * math.pow(t3,2) + (t1 * t3 * t7)\
    - 1/2 * (t7 * math.pow(t2,2)) + (t2 * t3 * t7)\
    - 1/2 * (t7 * math.pow(t3,2)) - 1/2 * (t7 * math.pow(t4,2)) \
    + 1/2 * (t4 * math.pow(t7,2)) + 1/2 * (t7 * math.pow(t5,2)) \
    - 1/2 * (t5 * math.pow(t7,2)) + 1/2 * (t7 * math.pow(t6,2)) \
    - 1/2 * (t6 * math.pow(t7,2))

    ## Jerk
    K=np.zeros(Axis,)
    for i in range(Axis):
        K[i]=Displacement[i]/DisplacementFunction

    ## Caculation
    n=0
    T=math.ceil(t7/SamplingTime)
    a = np.zeros((Axis, T), np.float64)
    v = np.zeros((Axis, T), np.float64)
    s= np.zeros((Axis, T), np.float64)
    Time=np.zeros((T,),np.float64)
    for t in range(T):

        t = t * SamplingTime
        for i in range(Axis):
            if (t1 >= t):

                a[i][n] = K[i] * t
                v[i][n] = K[i] * (1 / 2 * math.pow(t, 3))
                s[i][n] = InitialAngle[i] + K[i] * (1 / 6) * math.pow(t, 3)

            elif (t2 >= t and t > t1):

                a[i][n] = K[i] * t1
                v[i][n] = K[i] * (-1 / 2 * math.pow(t1, 2)+t1*t)
                s[i][n] = InitialAngle[i] + K[i] * ((1 / 6) * math.pow(t1, 3)-1/2*t*math.pow(t1,2)+1/2*t1*math.pow(t,2))

            elif(t3>=t and t>t2):
                a[i][n] = -K[i] * t + K[i] * t1 + K[i] * t2
                v[i][n] = K[i] * (-1 / 2 * math.pow(t1, 2) - 1 / 2 * math.pow(t2, 2) + t * t2 + t * t1 - 1 / 2 * math.pow(t, 2))
                s[i][n] = InitialAngle[i] + K[i] * (1 / 6 * math.pow(t1, 3) + 1 / 6 * math.pow(t2, 3) - 1 / 2 * t * math.pow(t1, 2) - 1 / 2 * t * math.pow(t2, 2) + 1 / 2 * (t2 + t1) * math.pow(t, 2) - 1 / 6 * math.pow(t, 3))
            elif(t4 >= t and t > t3):
                a[i][n] = 0
                v[i][n] = K[i] * (-1 / 2 * math.pow(t1, 2) - 1 / 2 *math.pow(t2, 2) + t3 * t2 + t3 * t1 - 1 / 2 * math.pow(t3, 2))
                s[i][n] = InitialAngle[i] + K[i] * (1 / 6 * math.pow(t1, 3) + 1 / 6 * math.pow(t2, 3) + 1 / 3 * math.pow(t3, 3) - 1 / 2 * t1 * math.pow(t3, 2) - 1 / 2 * t2 * math.pow( t3, 2) + t * (- 1 / 2 * math.pow(t1, 2) - 1 / 2 * math.pow(t2, 2) - 1 / 2 * math.pow(t3,2) + t1 * t3 + t2 * t3))
            elif(t5 >= t and t > t4):
                a[i][n]= -K[i] * t + K[i] * t4
                v[i][n] = K[i] * ( -1 / 2 * math.pow(t1, 2) - 1 / 2 * math.pow(t2, 2) - 1 / 2 * math.pow(t3, 2) - 1 / 2 *math.pow(t4,2) + t1 * t3 + t2 * t3 + t * t4 - 1 / 2 * math.pow(t, 2))
                s[i][n]= InitialAngle[i] + K[i] * (1 / 6 * math.pow(t1, 3) + 1 / 6 *math.pow(t2, 3) + 1 / 3 * math.pow(t3, 3) + 1 / 6 * math.pow(t4,3) - 1 / 2 * t1 *math.pow(t3, 2) - 1 / 2 * t2 * math.pow(t3, 2)+ t * (-1 / 2 * math.pow(t1, 2) - 1 / 2 *math.pow(t2, 2) - 1 / 2 * math.pow(t3, 2) - 1 / 2 * math.pow( t4, 2) + t1 * t3 + t2 * t3)+ 1 / 2 * t4 * math.pow(t, 2) - 1 / 6 * math.pow(t, 3))
            elif(t6 >= t and t > t5):
                a[i][n]= -K[i] * t5 + K[i] * t4
                v[i][n]= K[i] * ( -1 / 2 * math.pow(t1, 2) - 1 / 2 * math.pow(t2, 2) - 1 / 2 * math.pow(t3, 2) - 1 / 2 * math.pow(t4,2) + 1 / 2 * math.pow( t5, 2) + t1 * t3 + t2 * t3 - t5 * t + t4 * t)
                s[i][n] = InitialAngle[i] + K[i] * (1 / 6 * math.pow(t1, 3) + 1 / 6 * math.pow(t2, 3) + 1 / 3 * math.pow(t3, 3) + 1 / 6 * math.pow(t4,3) - 1 / 6 * math.pow(t5, 3)- 1 / 2 * t1 * math.pow(t3, 2) - 1 / 2 * t2 * math.pow(t3, 2) + t * (-1 / 2 * math.pow(t1, 2) - 1 / 2 * math.pow(t2, 2) - 1 / 2 * math.pow(t3, 2) - 1 / 2 * math.pow( t4, 2) + 1 / 2 *math.pow(t5, 2) + t1 * t3 + t2 * t3) - 1 / 2 * (t5 - t4) *math.pow(t, 2))
            elif(t7 >= t and t > t6):
                a[i][n]= K[i] * t - K[i] * t5 - K[i] * t6 + K[i] * t4
                v[i][n] = K[i] * (-1 / 2 * math.pow(t1, 2) - 1 / 2 * math.pow(t2, 2) - 1 / 2 *math.pow(t3, 2) - 1 / 2 * math.pow(t4,2) + 1 / 2 * math.pow( t5, 2) + 1 / 2 * math.pow(t6, 2) + t1 * t3 + t2 * t3 - t5 * t + t4 * t - t6 * t + 1 / 2 * math.pow(t, 2))
                s[i][n] = InitialAngle[i] + K[i] * ( 1 / 6 * math.pow(t1, 3) + 1 / 6 * math.pow(t2, 3) + 1 / 3 * math.pow(t3, 3) + 1 / 6 * math.pow(t4,  3) - 1 / 6 * math.pow( t5, 3) - 1 / 6 * math.pow(t6, 3) - 1 / 2 * t1 * math.pow(t3, 2) - 1 / 2 * t2 * math.pow(t3, 2)+ t * (-1 / 2 * math.pow(t1, 2) - 1 / 2 * math.pow(t2, 2) - 1 / 2 * math.pow(t3, 2) - 1 / 2 * math.pow(  t4, 2) + 1 / 2 * math.pow(t5, 2) + 1 / 2 * math.pow(t6, 2) + t1 * t3 + t2 * t3) + math.pow(t, 2) * 1 / 2 * (-t5 + t4 - t6) + 1 / 6 *math.pow(t, 3))

        Time[n]=t
        n=n+1
    JointCmd = [s,v,a]

    return JointCmd ,Time