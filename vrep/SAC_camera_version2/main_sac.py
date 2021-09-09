"""
主程式

By : ya0000000
2021/08/31
"""


from env_new2 import robot_env
from sac4 import SAC_agent
import config
import time
import matplotlib.pyplot as plt
from yolo import *
import pygame
import sys

MAX_EPISODES = config.MAX_EPISODES
MAX_EP_STEPS = config.MAX_EP_STEPS
eval_iteration = config.eval_iteration
PATH = time.strftime('%m%d%H%M')
ON_TRAIN = config.ON_TRAIN
# for eval
PATH_EVAL = config.PATH_EVAL

env=robot_env()
a_bound = config.a_bound
print('a_bound',a_bound)
s_dim = env.state_dim
a_dim = env.action_dim
print('s_dim', s_dim,'a_dim',a_dim)
agent = SAC_agent(s_dim, a_dim, a_bound)

if ON_TRAIN:
    Mode = 'Train'
else:
    Mode = 'Eval'
    record = True

def train():

    total_numsteps = 0
    env.initial()

    if config.pre_train:
        agent.load_models(PATH_EVAL)

    #------record memory------#
    step_set, reward_set, avg_reward_set, state_set ,action_set,distance_set, loss_q1_set,  loss_q2_set,loss_pi_set = [],[],[],[], [], [], [], [], []
    for i in range(MAX_EPISODES):
        s = env.reset(i)
        done = False
        ep_r = 0
        for j in range(MAX_EP_STEPS):

            a = agent.choose_action(s,ON_TRAIN=True)

            action_set.append(a) # 紀錄

            s_, r, done= env.step(a)

            """ 一次就拾取成功 +0.5 ； 其他 -0.1 """
            if (j == 0 and done ==True):
                r += 0.5
            if not done:
                r -= 0.1

            """ sac 存資料到 replay buffer """
            agent.remember(s, a ,r, s_, done)

            ep_r += r
            total_numsteps += 1


            """ sac 學習 """
            loss_critic1, loss_critic2, loss_actor1 = agent.learn()
            loss_q1_set.append(loss_critic1)  # 紀錄
            loss_q2_set.append(loss_critic2)  # 紀錄
            loss_pi_set.append(loss_actor1)   # 紀錄

            """ 更新state """
            s = s_

            # 紀錄
            if(config.color_state):
                store_state = np.reshape(s, (config.image_input * config.image_input*3))
                state_set.append(store_state)
            else:
                store_state = np.reshape(s,(config.image_input*config.image_input))
                state_set.append(store_state)


            if done or j == MAX_EP_STEPS - 1:
                print('episode: %i | %s | ep_r %.3f |step:%i' % (i, '...' if not done else 'done', ep_r, j + 1))
                step_set.append(j + 1)                  # 紀錄
                reward_set.append(ep_r)                 # 紀錄
                avg_reward_set.append(ep_r / (j + 1))   # 紀錄
                break

        """ 存 model """
        if i % eval_iteration == 0 and ep_r == 1.5:
            agent.save_models(PATH, i / eval_iteration)

    print(total_numsteps)

    """---------------------------------以下為存資料和畫圖---------------------------------"""

    save_txt(path='./model/' + PATH + '/train/', name='Total_steps.txt', data=[total_numsteps], fmt='%f')

    # ----- record data ----- #
    save_txt(path='./model/' + PATH + '/train/data/', name='state_set.txt', data=state_set, fmt='%1.3f') # 存state

    folder_data = './model/' + PATH + '/train/data/'
    file_name = ['step.txt', 'reward.txt', 'avgR.txt','critic1_loss.txt','critic2_loss.txt','actor_loss.txt']
    data_set = ['step_set', 'reward_set','avg_reward_set','loss_q1_set','loss_q2_set','loss_pi_set']
    for i in range(len(file_name)):
        save_txt(path=folder_data, name=file_name[i], data=eval(data_set[i]))


    # ----- plot fig -----
    folder_fig = './model/' + PATH + '/train/fig/'
    xlabel = ['Episode', 'Episode', 'Episode','Episode', 'Episode', 'Episode']
    ylabel = ['Step', 'Reward', 'avgR','loss','loss','loss']
    title = ['step', 'reward', 'avgR','cost_critic1','cost_critic2','cost_actor']
    for i in range(len(file_name)):
        plot_txt(path=folder_data, name=file_name[i], xlabel=xlabel[i], ylabel=ylabel[i], title=title[i],
                 save_location=folder_fig)


def Eval_vrep():
    total_numsteps = 0
    env.initial()
    agent.load_models(PATH_EVAL)

    for i in range(10):
        s = env.reset(i)
        done = False
        for j in range(3):
            a = agent.choose_action(s, ON_TRAIN=True)
            s_, r, done = env.step(a)
            s = s_
            if done :
                total_numsteps = total_numsteps +1
                break


    print('success times : ',total_numsteps)


def plot(data, xlabel, ylabel, title, save_location):
    plt.plot(np.arange(data.shape[0]), data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.savefig(save_location+title)
    plt.clf()
    plt.close()

def plot_txt(path, name, xlabel, ylabel, title, save_location):
    # path exist
    if path_exsit(path+name):
        # load data
        data = load_txt(path=path, name=name)
        # plot
        plot(data=data, xlabel=xlabel, ylabel=ylabel, title=title, save_location=save_location)
    else:
        print(path+name +' does not exist')

def load_txt(path, name):
    f = open(path + name, 'r')
    data = np.loadtxt(f)
    f.close()
    return data


def main():
    if ON_TRAIN:
        creat_path('./model/' + PATH + '/train/data')
        creat_path('./model/' + PATH + '/train/fig')
        creat_path('./model/' + PATH + '/test')
        save_parameter()
        start_time = time.time()
        train()
        end_time = time.time()
        cost_time = np.array([[end_time - start_time]])
        save_txt(path='./model/' + PATH + '/train/', name='time.txt', data=cost_time, fmt='%f')
    else:
        Eval_vrep()


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
def save_txt(path, name, data, fmt='%f'):
    f = open(path + name, 'w')
    np.savetxt(f, data, fmt=fmt)
    f.close()

def save_parameter():
    with open('./model/' + PATH + '/train/parameter.txt', 'w') as f:
        f.writelines("Method: {}\n".format('sac'))
        f.writelines("state: {}\n state_type :{}\n action: {}\n a_bound:{}\n".format(config.state_dim,config.color_state,config.action_dim,config.a_bound))
        f.writelines("Max Episodes: {}\nMax Episodes steps: {}\nObject_Num:{}\nPre_train:{}\n".format(MAX_EPISODES, MAX_EP_STEPS,config.num_object,config.pre_train))
        f.writelines("LR_A: {}\nLR_C: {}\nGAMMA: {}\n".format(config.A_LR, config.C_LR, config.gamma))
        f.writelines("reward_scale: {}\ntau: {}\nmemory_capacity: {}\n".format(config.reward_scale,config.tau,config.MEMORY_CAPACITY))
        f.writelines("BATCH_SIZE: {}\nhidden_dim: {}\n".format(config.batch_size,config.hidden_dim))


if __name__ == "__main__":
    main()


