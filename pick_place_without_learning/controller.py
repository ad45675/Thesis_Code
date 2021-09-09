
import numpy as np
import fullmodel_MN as robot
def controller(Pr,ps,vs):

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
    Ps,Vs,As,Ts=[],[],[],[]

    ts = Kv * (Kp * (np.reshape(np.transpose(Pr), (6, 1)) - ps) - vs)
    # ----------------------------- Savedata - -----------------------------
    # print(ps.shape)
    Ps.append(np.transpose(ps))
    Vs.append(np.transpose(vs))
    As.append (np.transpose(accs))
    Ts.append (np.transpose(ts))
    # ------------------------ Directdynamicmodel - -----------------------
    as_old = accs
    vs_old = vs

    M, N = robot.FullModel_MN(np.squeeze(ps), np.squeeze(vs))
    accs = np.linalg.inv(M).dot(ts - N)
    # accs[3, 0]=0
    vs = vs + (accs + as_old) * (tm / 2)
    ps = ps + (vs + vs_old) * (tm / 2)

    out_joint=np.squeeze(np.transpose(ps))
    out_joint_vel=np.squeeze(np.transpose(vs))


    return out_joint,out_joint_vel

def velocity_controller(Velcmd, Vs, Ps, SamplingTime):

    Kiv = np.array([100, 150, 150, 70, 70, 50])
    Kfv = np.array([1, 1, 1, 1, 1, 1])
    Kpv = np.array([150, 150, 100, 50, 100, 50])
    Kiv = np.reshape(Kiv, (6, 1))
    Kfv = np.reshape(Kfv, (6, 1))
    Kpv = np.reshape(Kpv, (6, 1))

    Vel_error = Velcmd - Vs

    ts = Kpv*(Kiv*Vel_error*SamplingTime - Vs + Kfv * Velcmd)

    as_old = accs
    vs_old = Vs
    # ------------------------ Directdynamicmodel - -----------------------
    M, N = robot.FullModel_MN(np.squeeze(Ps), np.squeeze(Vs))

    accs = np.linalg.inv(M).dot(ts - N)

    vs = Vs + (accs + as_old) * (tm / 2)
    ps = ps + (vs + vs_old) * (tm / 2)


