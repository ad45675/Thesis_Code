"""
yaooooo
this is robot env  code
update time:03/15

將距離和角度正規化並修改reward和結束條件

state dim=5
0~3:joint pos
# 4~6:cuboid pos
# 7~9:EEF pos
4:dis
-------------------
action dim=4
joint pos
--------------------
joint_bound
joint 1=[-170 170]
joint 2=[-135  80]
joint 3=[-70  104]
joint 4=[-190 190]
joint 5=[-115 115]
joint 6=[-360 360]

"""

import numpy as np
import os
import math
import time
import inverseKinematics as IK
import Kinematics as FK
from IK_FindOptSol import FindOptSol
from simulation_robot import simulation_robot as id_robot
from robot_vrep import my_robot
from controller import *
import config
import compute_robot_jacobian


def creat_path(path):
    if path_exsit(path=path):
        print(path+' exist')
    else:
        os.makedirs(path)
def path_exsit(path):
    if os.path.exists(path):
        return True
    else:
        return False

radtodeg = 180 / math.pi  # 弧度轉角度
degtorad = math.pi / 180  # 角度轉弧度
#這裡單位是 cm  吸嘴加0.068m
DH_table = np.array([[0,            0.345,  0.08,   math.pi / 2],
					 [0+math.pi / 2 , 0,  0.27,     0],
					 [0,             0,     0.09,    math.pi / 2],
					 [0,            0.295,  0,       -math.pi / 2],
					 [0,            0,      0,       math.pi / 2],
					 [0,       0.102+0.068, 0,          0]])


def save_txt(data, fmt='%f'):
    f = open('C:/Users/user/Desktop/rl/data.txt', 'a')
    np.savetxt(f, data, fmt=fmt)
    f.close()

def robot_constraint():
    GearRatio = [120.0, 120.0, 120.0, 102, 80.0, 51.0]
    vel_limit=np.zeros(6,)
    acc_limit=np.zeros(6,)
    jerk_limit=np.zeros(6,)

    pos_limit = np.array([[-170*degtorad , 170*degtorad ],
                          [-85*degtorad , 130*degtorad ],
                          [-110*degtorad , 170*degtorad ],
                          [-190*degtorad , 190*degtorad ],
                          [-125*degtorad , 125*degtorad ],
                          [-360*degtorad , 360*degtorad ]])
    for i in range(6):
        #[2.4933, 2.4933, 2.9333, 3.7400, 5.8667]
        vel_limit[i]=((2000/60)/0.7)/GearRatio[i]*2*math.pi  #rad
        acc_limit[i]=vel_limit[i]*3                          #rad
        jerk_limit[i]=acc_limit[i]*3                         #rad

    return  pos_limit, vel_limit, acc_limit

class robot_env(object):
    # joint_bound
    degtorad = math.pi / 180
    state_dim = config.state_dim
    action_dim = config.action_dim
    SamplingTime = 0.01
    def __init__(self):
        self.radtodeg = 180 / math.pi  # 弧度轉角度
        self.degtorad = math.pi / 180  # 角度轉弧度
        self.my_robot = my_robot()
        self.my_robot.connection()
        self.joint_cmd = np.zeros((6,),np.float)
        self.vs = np.zeros((6,),np.float)


    def initial(self):
        self.my_robot.stop_sim()
        self.my_robot.start_sim()

    def reset(self):
        # return state containing joint ,EFF ,target ,dis
        # robot to initial pos and random the target

        self.joint = [0, 0, 0, 0, -1.57, 0]
        self.vel = [0, 0, 0, 0, 0, 0]
        self.accs = [0, 0, 0, 0, 0, 0]
        self.torque = [0, 0, 0, 0, 0, 0]
        self.error = []
        self.my_robot.move_all_joint(self.joint)
        time.sleep(0.2)
        print('reset')

        # 目標物隨機擺放
        self.my_robot.random_object()
        self.cubid_pos, self.cuboid_x_range , self.cuboid_y_range  = self.my_robot.get_cuboid_pos()  # dim=3
        self.enable_suction = 0

        return self.get_state()

    def get_state(self):


        # 順向運動學得到末端點資訊
        Info, EulerAngle_vrep, EulerAngle, EEF_pos = FK.ForwardKinemetics(self.joint, DH_table)
        self.EEF_pos = np.round(EEF_pos, 4)

        distance = np.linalg.norm(self.cubid_pos - self.EEF_pos)
        # gripper state
        object_rel_pos = self.cubid_pos - self.EEF_pos



        # s = np.concatenate([self.cubid_pos,self.EEF_pos,object_rel_pos])

        s = np.hstack([self.cubid_pos,self.EEF_pos,object_rel_pos,self.joint,self.vel])
        # dim = 21
        return s


    def step(self, action, record,done_dis):
        #action 卡式空間速度
        joint_pos_out = np.zeros((6,),np.float)
        joint_cmd = np.zeros((6,), np.float)
        target_height = np.zeros((3,), np.float)
        JointVelocity_cmd = np.zeros((6,), np.float)
        done = False
        reward = 0
        success = 0
        coutbound = 0
        cuboid_out = 0
        total_out = 0
        singular = 0

        #--------------------直接輸出速度
        C_x = self.EEF_pos[0]+action[0]
        C_y = self.EEF_pos[1]+action[1]
        C_z = self.EEF_pos[2]+action[2]

        C_Vx = action[0] * 1/300
        C_Vy = action[1] * 1/300
        C_Vz = action[2] * 1/300

        # C_Vx = np.clip(C_Vx, -0.5, 0.5)
        # C_Vy = np.clip(C_Vy, -0.5, 0.5)
        # C_Vz = np.clip(C_Vz, -0.5, 0.5)

        C_V = [C_Vx, C_Vy, C_Vz, 0, 0, 0]
        C_V = np.reshape(C_V,(6,1))
        # print(C_V)
        RobotJacobian = compute_robot_jacobian.compute_robot_jacobian(DH_table,self.joint)
        b=np.linalg.det(RobotJacobian)

        try:
            JointVelocity_cmd = np.linalg.inv(RobotJacobian).dot(C_V)
        except:
            singular += 1
            JointVelocity_cmd = np.array([0,0,0,0,0,0])


        joint ,vel ,accs, ts = velocity_controller(JointVelocity_cmd,self.accs,self.vel,self.joint,0.001,self.error)
        # print(joint)
        self.my_robot.move_all_joint(joint)

        # C_V_record = np.reshape(joint, (1, 6))
        # action_record = np.reshape(action, (1, 6))
        # path = 'C:\\Users\\user\\Desktop\\rl\\vrep\\SAC_camera_version2\\Trajectory\\'
        # name = 'joint.txt'
        # # name2 = 'action_record.txt'
        # f = open(path + name, mode='a')
        # # f2 = open(path + name2, mode='a')
        # np.savetxt(f, C_V_record, fmt='%f')
        # # np.savetxt(f2, action_record, fmt='%f')
        # f.close()
        # # f2.close()

        self.joint = joint
        self.vel = vel
        self.accs = accs
        self.torque = ts
        total_out = self.check_joint_bound(self.joint,self.vel,self.accs)

        #------------------------------------------看現在末端點位置
        # self.joint = [0,0,0,0,0,0]
        Info, EulerAngle_vrep, EulerAngle, EEF_pos = FK.ForwardKinemetics(self.joint, DH_table)

        distance = np.linalg.norm(self.cubid_pos - EEF_pos)
        # print(distance)
        if distance < done_dis :
            done = True
            success = 1
        else:
            success = 0

        reward = -distance + success - total_out*0.1 - singular*0.1

        if (record):
        #################### record data #####################
            C_V_record = np.reshape(C_V, (1, 3))
            action_record = np.reshape(action,(1,3))
            path = './Trajectory/'
            name = 'EEF_record.txt'
            name2 = 'action_record.txt'
            f = open(path + name, mode='a')
            f2 = open(path + name2, mode='a')
            np.savetxt(f, C_V_record, fmt='%f')
            np.savetxt(f2, action_record, fmt='%f')
            f.close()
            f2.close()
        #################### record data #####################




        s_ = self.get_state()

        return s_, reward, done

    def check_joint_bound(self, joint, vel, accs):
        joint_outbound, vel_outbound, acc_outbount = 0, 0, 0
        pos_limit, vel_limit, acc_limit = robot_constraint()
        for i in range(6):
            if (pos_limit[i][0] > joint[i] or joint[i] > pos_limit[i][1]):
                joint_outbound = joint_outbound + 1
            if (vel[i] > vel_limit[i]):
                vel_outbound = vel_outbound + 1
            if (accs[i] > acc_limit[i]):
                acc_outbount = acc_outbount + 1
        total_out_bound = joint_outbound + vel_outbound + acc_outbount
        return total_out_bound



    def check_c_space_bound(self,EEF):
        c_out=False
        if EEF[0]<0.25 or EEF[0]>0.6:
            c_out=True
        if EEF[1]<-0.324 or EEF[1]>0.374 or EEF[2] < 0:
            c_out=True
        return c_out
    def sample_action(self):
        return np.random.rand(4)  # 4 joints




if __name__ == '__main__':
    render=True
    env = robot_env()
    env.initial()
    env.reset(render)
    # time.sleep(10)
    # while True:
    time.sleep(5)
    action = np.array([0.03, 0.03, 0.03, -0.03], dtype=np.float32)
    env.step(action,render)
    # env.step(env.sample_action())
    print(action *env.radtodeg)
    time.sleep(10)

# self.my_robot.move_all_joint(self.joint)
##################### record data #####################
# error_record = np.reshape(error, (1, 6))
# # print(joint_out_record)
# path = './Trajectory/'
# name = 'error_record.txt'
# f = open(path + name, mode='a')
# np.savetxt(f, error_record, fmt='%f')
# f.close()
##################### record data #####################