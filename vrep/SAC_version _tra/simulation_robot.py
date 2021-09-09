import numpy as np
import fullmodel_MN as robot
import matplotlib.pyplot as plt



def load_txt(path, name):
    f = open(path + name, 'r')
    data = np.loadtxt(f)
    f.close()
    return data




# load data
# EstimatedData = load_txt('./Data/', 'EstimatedData.txt') #實際資料
# Trajectory = load_txt( './Data/','TestTrajectory.txt' )   #軌跡命令


# Pa = EstimatedData[:, 0: 6]  # 實際角度 (rad)
# Va = EstimatedData[:, 6: 12]  # 實際速度 (rad/s)
# Aa = EstimatedData[:, 12: 18]  # 實際加速度 (rad/s^2)
# Ta = EstimatedData[:, 18: 24]  # 實際轉矩 (Nm)

# Pr = Trajectory[:, 0: 6]  # 角度命令 (rad)
# Vr = Trajectory[:, 6: 12]  # 速度命令 (rad/s)
# Ar = Trajectory[:, 12: 18]  # 加速度命令 (rad/s^2)


#parameters

n = 6   # 關節數
tf = 10   # 結束時間 (sec)
wm = 1000   # 量測取樣頻率 (Hz)
tm = 1 / wm   # 量測取樣時間 (sec)
#Tm = tm : tm : tf   # 量測時間 (sec)
Tm = np.arange(tm, tf+0.001, tm)
Nm = np.int(tf / tm)   # 量測資料數 (samples)

def simulation_robot(Pr,j):


    Kp = np.array([250, 250, 200, 250, 150, 350])
    Kv = np.array([100, 100, 100, 40, 30, 5])
    Kp = np.reshape(Kp, (6, 1))
    Kv = np.reshape(Kv, (6, 1))

    ps = np.transpose(Pr)
    ps = np.reshape(ps, (6, 1))
    vs = np.zeros((n,1), np.float)
    accs = np.zeros((n,1), np.float)



    Ps=np.zeros((Nm,n), np.float)
    Vs=np.zeros((Nm,n), np.float)
    As=np.zeros((Nm,n), np.float)
    Ts=np.zeros((Nm,n), np.float)



        # ---------------------------- Controllaw - ----------------------------

    ts=Kv*(Kp*(np.reshape(np.transpose(Pr), (6, 1))-ps)-vs)

        # ----------------------------- Savedata - -----------------------------
            #print(ps.shape)
    Ps[j, :] = np.transpose(ps)
    Vs[j, :] = np.transpose(vs)
    As[j, :] = np.transpose(accs)
    Ts[j, :] = np.transpose(ts)
        # ------------------------ Directdynamicmodel - -----------------------
    as_old = accs
    vs_old = vs





    M, N = robot.FullModel_MN(np.squeeze(ps), np.squeeze(vs))
    accs = np.linalg.inv(M).dot(ts - N)
    vs = vs + (accs + as_old) * (tm / 2)
            #print('tm, ', tm.shape)
    ps = ps + (vs + vs_old) * (tm / 2)

    return(ps)

# plt.figure(1)
# for i in range(1, 7):
#     plt.subplot(3, 2, i)
#     plt.plot(Tm, Ps[:, i - 1], ':', color='y', label='Ps',linewidth=2.0 )
#     plt.plot(Tm, Pa[:, i - 1], '-', color='m', label='Pa',linewidth=1.5)
#     plt.plot(Tm, Pr[:, i - 1], '--', color='c',  label='Pr',linewidth=1.0)
#     plt.xlabel('Time (sec)')
#     plt.ylabel('Position (rad)')
#     plt.title('Direct comparison joint' + str(int(i)))
#     plt.grid(True)
#     plt.legend()
#     plt.legend(loc='upper right')
# plt.subplots_adjust(hspace=1.5, wspace=0.5)
# plt.savefig('Direct comparisonjoint_pos ')
# # plt.show()
#
#
# plt.figure(2)
# for i in range(1, 7):
#     plt.subplot(3, 2, i)
#     plt.plot(Tm, Vs[:, i - 1], ':', color='y', label='Vs',linewidth=2.0 )
#     plt.plot(Tm, Va[:, i - 1], '-', color='m', label='Va',linewidth=1.5)
#     plt.plot(Tm, Vr[:, i - 1], '--', color='c',  label='Vr',linewidth=1.0)
#     plt.xlabel('Time (sec)')
#     plt.ylabel('Velocity (rad/s)')
#     plt.title('Direct comparison joint' + str(int(i)))
#     plt.grid(True)
#     plt.legend()
#     plt.legend(loc='upper right')
# plt.subplots_adjust(hspace=1.5, wspace=0.5)
# plt.savefig('Direct comparisonjoint_vel ')
# # plt.show()
#
# plt.figure(3)
# for i in range(1, 7):
#     plt.subplot(3, 2, i)
#     plt.plot(Tm, As[:, i - 1], ':', color='y', label='As',linewidth=2.0 )
#     plt.plot(Tm, Aa[:, i - 1], '-', color='m', label='Aa',linewidth=1.5)
#     plt.plot(Tm, Ar[:, i - 1], '--', color='c',  label='Ar',linewidth=1.0)
#     plt.xlabel('Time (sec)')
#     plt.ylabel('Acceleration (rad/s^2)')
#     plt.title('Direct comparison joint' + str(int(i)))
#     plt.grid(True)
#     plt.legend(loc='upper right')
# plt.subplots_adjust(hspace=1.5, wspace=1)
# plt.savefig('Direct comparisonjoint_acc ')
# # plt.show()
#
# plt.figure(4)
# for i in range(1, 7):
#     plt.subplot(3, 2, i)
#     plt.plot(Tm, Ts[:, i - 1], '--', color='#FFFF00', label='Ts')
#     plt.plot(Tm, Ta[:, i - 1], '-', color='#6495ED', label='Ta')
#     # plt.plot(Tm, Ar[:, i - 1], '--', color=(255/255, 221/255, 130/255),  label='Ar',linewidth=1.0)
#     plt.xlabel('Time (sec)')
#     plt.ylabel('Torque (Nm)')
#     plt.title('Direct comparison joint' + str(int(i)))
#     plt.grid(True)
#     plt.legend(loc='upper right')
# plt.subplots_adjust(hspace=1.5, wspace=1)
# plt.savefig('Direct comparisonjoint_torque ')
# plt.show()