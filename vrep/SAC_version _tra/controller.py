
import numpy as np
import fullmodel_MN as robot
def controller(Pr,ps,vs):


    # input shape (6,)
    #parameters
    n = 6   # 關節數
    wm = 1000   # 量測取樣頻率 (Hz)
    tm = 1 / wm   # 量測取樣時間 (sec)


    accs = np.zeros((n,1), np.float)
    ps = np.reshape(ps, (6, 1))
    vs = np.reshape(vs, (6, 1))


    Kp = np.array([250, 250, 200, 250, 150, 300.0])
    Kv = np.array([100, 100, 100, 40, 30, 5])
    Kp = np.reshape(Kp, (6, 1))
    Kv = np.reshape(Kv, (6, 1))
    # Ps,Vs,As,Ts=[],[],[],[]

    # ts = Kv * (Kp * (np.reshape(np.transpose(Pr), (6, 1)) - ps) - vs)
    ts = Kv * (Kp * (np.reshape(Pr, (6, 1)) - ps) - vs)

    # ----------------------------- Savedata - -----------------------------


    # ------------------------ Directdynamicmodel - -----------------------
    as_old = accs
    vs_old = vs

    M, N = robot.FullModel_MN(np.squeeze(ps), np.squeeze(vs))
    accs = np.linalg.inv(M).dot(ts - N)
    # accs[3, 0]=0
    vs = vs + (accs + as_old) * (tm / 2)
    ps = ps + (vs + vs_old) * (tm / 2) #shape(6,1)


    out_joint=np.squeeze(ps)
    out_joint_vel=np.squeeze(vs)   #shape(6,)

    return out_joint,out_joint_vel

def velocity_controller(Vcmd, Accs, Vs, Ps, SamplingTime,error):

    # input shape(6,)
    #parameters
    n = 6   # 關節數
    wm = 1000   # 量測取樣頻率 (Hz)
    tm = 1 / wm   # 量測取樣時間 (sec)

    Vcmd = np.reshape(Vcmd,(6,1))
    Accs = np.reshape(Accs,(6,1))
    Vs = np.reshape(Vs,(6,1))
    Ps = np.reshape(Ps,(6,1))

    error.append(Vcmd - Vs)


    error = np.array(error)   #(6,1)
    error = np.squeeze(error,axis = 2)
    error = np.transpose(error)  #(6,None)


    sum_error = np.array([
                            [np.sum(error[0,:])],
                            [np.sum(error[1,:])],
                            [np.sum(error[2,:])],
                            [np.sum(error[3,:])],
                            [np.sum(error[4,:])],
                            [np.sum(error[5,:])]
                        ])  #shape(6,1)





    Kiv = np.array([100, 100, 100, 40, 30, 5])
    Kpv = np.array([100, 150, 150, 70, 70, 50])
    Kfv = np.array([1, 1, 1, 1, 1, 1])

    Kiv = np.reshape(Kiv, (6, 1))
    Kfv = np.reshape(Kfv, (6, 1))
    Kpv = np.reshape(Kpv, (6, 1))


    ts = Kpv*(Kiv*sum_error*SamplingTime - Vs + Kfv*Vcmd)

    # ------------------------ Directdynamicmodel - -----------------------

    as_old = Accs
    vs_old = Vs
    M, N = robot.FullModel_MN(np.squeeze(Ps), np.squeeze(Vs))
    accs = np.linalg.inv(M).dot(ts - N)

    Vs = Vs + (accs + as_old) * (SamplingTime / 2)
    Ps = Ps + (Vs + vs_old) * (SamplingTime / 2)

    out_joint = np.squeeze(Ps)
    out_joint_vel = np.squeeze(Vs)
    out_joint_accs = np.squeeze(accs)
    out_ts = np.squeeze(ts)

    return out_joint, out_joint_vel, out_joint_accs, out_ts

