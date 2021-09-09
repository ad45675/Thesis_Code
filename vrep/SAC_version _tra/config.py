import numpy as np


state = 'cuboid_pos(xyz), EEF(xyz), object_rel_pos(xyz), suction flag'
action = 'dx, dy, dz, suction_flag'
reward_fun = 'reward = -distance -0.2*coutbound -0.5*cuboid_out + success'
# success = 1 , not success = -0.1
# 走兩步就吸

ON_TRAIN = True
render = True

method = 'sac'
sac = False
hidden_dim=[256,256]
reparameterize_critic=False
reparameterize_actor=True

state_dim = 21
action_dim = 3
a_bound = [1,1,1]
repeat_action = 3

PATH_EVAL = ['03121642', '199']  # for eval

MAX_EPISODES = 10000
MAX_EP_STEPS = 1000

A_LR = 0.001
C_LR = 0.001
gamma = 0.99  # reward discount
reward_scale =0.01

tau = 0.005  # soft replacement
MEMORY_CAPACITY = 200000
batch_size = 256

eval_iteration = 5  #存model

initial_joint=[0,0,0,0,0,0]


#checkpoint_path = os.path.join('./model/' + path + '/net/' + str(int(i)))





