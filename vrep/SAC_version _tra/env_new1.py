"""
yaooooo
this is robot env  code
update time:12/11

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
import math
import time
import inverseKinematics as IK
import Kinematics as FK
from IK_FindOptSol import FindOptSol
from simulation_robot import simulation_robot as id_robot
from robot_vrep import my_robot

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

class robot_env(object):
    # joint_bound
    degtorad = math.pi / 180
    joint1_bound = [-50 * degtorad, 50 * degtorad]#(-0.87~0.87)
    joint2_bound = [-80 * degtorad, 70 * degtorad]#(-1.430~1.22)
    joint3_bound = [-60 * degtorad, 60 * degtorad]#(-1.22~1.04)
    joint4_bound = [0 * degtorad, 0 * degtorad]
    joint5_bound = [-90 * degtorad, 3 * degtorad]#(-1.57~0)
    joint6_bound = [-360 * degtorad, 360 * degtorad]
    state_dim = 7
    action_dim = 4

    def __init__(self):
        self.radtodeg = 180 / math.pi  # 弧度轉角度
        self.degtorad = math.pi / 180  # 角度轉弧度
        self.my_robot = my_robot()
        self.my_robot.connection()

    def initial(self):
        self.my_robot.stop_sim()
        self.my_robot.start_sim()

    def reset(self):
        # return state containing joint ,EFF ,target ,dis
        # robot to initial pos and random the target

        #self.my_robot.stop_sim()
        #self.my_robot.start_sim()
        self.my_robot.move_all_joint([0,0,0,0,0,0])
        print('reset')

        #self.my_robot.start_sim()
        # self.my_robot.random_object()
        return self.get_state()

    def get_state(self):
        # state:{物體位置,末端點位置}
        # self.joint_pos = self.my_robot.get_joint_pos()  # dim=6
        # self.joint_pos=np.round(self.joint_pos,5)
        # Info, EulerAngle_vrep, self.EulerAngle, Position = FK.ForwardKinemetics(self.joint_pos , DH_table)
        # Position=np.round(Position,4)
        EEF_pos = self.my_robot.get_EEF_pos()  # dim=3
        EEF_pos = np.round(EEF_pos , 4)

        cubid_pos = self.my_robot.get_cuboid_pos()  # dim=3


        diffence = [(cubid_pos[0] - EEF_pos[0]), (cubid_pos[1] - EEF_pos[1]), (cubid_pos[2] - EEF_pos[2])]
        self.distance = np.sqrt(pow(diffence[0], 2) + pow(diffence[1], 2) + pow(diffence[2], 2))

        s = np.hstack((EEF_pos,  cubid_pos, self.distance ))
        # print('s',s)
        # 吸嘴狀態還沒加上去
        return s

    def step(self, action,j):
        optimalsol_set=[]

        # EEF_pos = self.my_robot.get_EEF_pos()  # dim=3
        joint_pos = self.my_robot.get_joint_pos()  # dim=6
        time.sleep(0.2)
        joint_pos[0] = joint_pos[0] + action[0]
        joint_pos[1] = joint_pos[1] + action[1]
        joint_pos[2] = joint_pos[2] + action[2]
        joint_pos[5] = joint_pos[5] + action[3]
        self.my_robot.move_all_joint(joint_pos)
        # EEF_pos[0]=EEF_pos[0]+action[0]
        # EEF_pos[1]=EEF_pos[1]+action[1]
        # EEF_pos[2]=EEF_pos[2]+action[2]



        # print('eef_pos2', EEF_pos)
        # action為末端點變化量.
        # Move the robot arm according to the action.
        # Info, EulerAngle_vrep, EulerAngle, Position = FK.ForwardKinemetics(joint_pos, DH_table)
        # EulerAngle=[0,0,180]
        # invsol, flag = IK.InverseKinemetics(EulerAngle,  EEF_pos, DH_table)
        #
        # time.sleep(0.2)
        # optimalsol = FindOptSol(invsol,joint_pos )
        # optimalsol_set.append(optimalsol)
        # save_txt(optimalsol_set, fmt='%f')
        # print('optimalsol',optimalsol)
        # self.joint_pos=id_robot(optimalsol,j)


        done = False
        reward = 0
        # suction_flag=False#是否吸取物體
        EEF_pos = self.my_robot.get_EEF_pos()  # dim=3
        cubid_pos = self.my_robot.get_cuboid_pos()
        diffence = [(cubid_pos[0] - EEF_pos[0]), (cubid_pos[1] - EEF_pos[1]), (cubid_pos[2] - EEF_pos[2])]
        distance = np.sqrt(pow(diffence[0], 2) + pow(diffence[1], 2) + pow(diffence[2], 2))
        # print('dis',distance)

        # joint_pos = self.my_robot.get_joint_pos()
        # # print('joint_pos',joint_pos)
        # joint_state = np.hstack((joint_pos[0], joint_pos[1], joint_pos[2], joint_pos[4]))   # dim4



        if self.distance<distance:
            reward=-distance-0.1
        else:
            reward=-distance+0.1

        self.distance=distance

        if (cubid_pos[0]==EEF_pos[0] and cubid_pos[1]==EEF_pos[1] and cubid_pos[2]==EEF_pos[2]):
            self.my_robot.enable_suction(True)
            if cubid_pos[2]>0.5:
                reward += 1
                done = True
            else:
                reward -=1
                done = True

        s_ = np.hstack((EEF_pos, cubid_pos,self.distance))

        return s_, reward, done

    def sample_action(self):
        return np.random.rand(4)  # 4 joints

    def render(self):

        pass


if __name__ == '__main__':
    env = robot_env()
    env.reset()
    # time.sleep(10)
    # while True:
    time.sleep(5)
    action = np.array([0, 0, 0, -0.5], dtype=np.float32)
    env.my_robot.move_4_joint(action)
    # env.step(env.sample_action())
    print(action *env.radtodeg)
    time.sleep(10)
