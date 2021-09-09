
from env_new2 import robot_env
import numpy as np
from sac4 import SAC_agent
# from utils import plot_learning_curve
import config
import time
import os
import matplotlib.pyplot as plt
import math

MAX_EPISODES = config.MAX_EPISODES
MAX_EP_STEPS = config.MAX_EP_STEPS
repeat_action = config.repeat_action
eval_iteration = config.eval_iteration
PATH = time.strftime('%m%d%H%M')
ON_TRAIN = config.ON_TRAIN
initial_joint = config.initial_joint
# render = config.render



# for eval
PATH_EVAL = config.PATH_EVAL

env=robot_env()
#a_bound = [env.joint1_bound[1], env.joint2_bound[1], env.joint3_bound[1], env.joint5_bound[1]]
a_bound = config.a_bound
print('a_bound',a_bound)
s_dim = env.state_dim
a_dim = env.action_dim
print('s_dim', s_dim,'a_dim',a_dim)
agent = SAC_agent(s_dim, a_dim, a_bound)

if ON_TRAIN:
    Mode = 'Train'
    record = False
else:
    Mode = 'Eval'
    record = True

def train():
    # uncomment this line and do a mkdir tmp && mkdir video if you want to
    # record video of the agent playing the game.
    #env = wrappers.Monitor(env, 'tmp/video', video_callable=lambda episode_id: True, force=True)
    best_score = float("-inf")

    total_numsteps = 0
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models(PATH_EVAL)
    env.initial()
    done_dis = 0.08
    # agent.load_models(PATH_EVAL)
    #------record memory------#
    step_set, reward_set, avg_reward_set, state_set ,action_set,distance_set, loss_q1_set,  loss_q2_set,loss_pi_set = [],[],[],[], [], [], [], [], []
    for i in range(MAX_EPISODES):
        s = env.reset()
        done = False
        ep_r = 0

        for j in range(MAX_EP_STEPS):

            a_re = np.array([0.0, 0.0, 0.0], np.float64)
            for repeat in range(repeat_action):
                a = agent.choose_action(s, ON_TRAIN=True)
                a_re += a

            a = a_re/repeat_action

            action_set.append(a)

            s_, r, done= env.step(a,record,done_dis)

            # if not done:
            #     r-=0.5

            agent.remember(s, a ,r, s_, done)

            ep_r += r
            total_numsteps += 1
            # if ep_r<-250:
            #     done = True

            if not load_checkpoint:
                loss_critic1, loss_critic2, loss_actor1 = agent.learn()
                loss_q1_set.append(loss_critic1)
                loss_q2_set.append(loss_critic2)
                loss_pi_set.append(loss_actor1)

            s = s_

            state_set.append(s)


            if done or j == MAX_EP_STEPS - 1:
                print('episode: %i | %s | ep_r %.1f |step:%i' % (i, '...' if not done else 'done', ep_r, j + 1))
                step_set.append(j + 1)
                reward_set.append(ep_r)
                avg_reward_set.append(ep_r / (j + 1))

                done_dis = done_dis * 0.993
                if done_dis <= 0.01:
                    done_dis = 0.01
                # avg_score_set.append(avg_score )
                # best_score_set.append(best_score)
                break

        if i % eval_iteration == 0:
            agent.save_models(PATH, i / eval_iteration)

    print(total_numsteps)
    save_txt(path='./model/' + PATH + '/train/', name='Total_steps.txt', data=[total_numsteps], fmt='%f')

    # ----- record data ----- #
    folder_data = './model/' + PATH + '/train/data/'
    file_name = ['step.txt', 'reward.txt', 'avgR.txt','state_set.txt','critic1_loss.txt','critic2_loss.txt','actor_loss.txt']
    data_set = ['step_set', 'reward_set','avg_reward_set','state_set','loss_q1_set','loss_q2_set','loss_pi_set']
    for i in range(len(file_name)):
        save_txt(path=folder_data, name=file_name[i], data=eval(data_set[i]))


    # ----- plot fig -----
    folder_fig = './model/' + PATH + '/train/fig/'
    xlabel = ['Episode', 'Episode', 'Episode', 'Episode','Episode', 'Episode', 'Episode']
    ylabel = ['step', 'r', 'avgR','state','loss','loss','loss']
    title = ['step', 'reward', 'avgR','state','cost_critic1','cost_critic2','cost_actor']
    for i in range(len(file_name)):
        plot_txt(path=folder_data, name=file_name[i], xlabel=xlabel[i], ylabel=ylabel[i], title=title[i],
                 save_location=folder_fig)


def Eval():

    agent.load_models(PATH_EVAL)
    env.initial()
    s = env.reset()
    record = False
    for i in range(300):
        a_re = np.array([0.0, 0.0, 0.0, 0.0], np.float64)
        for repeat in range(repeat_action):
            a=agent.choose_action(s,ON_TRAIN = True)
            a_re += a
        print(i)
        a = a_re/repeat_action
        s_,r,done=env.step(a,record)
        s=s_
        if done:
            break
        # print(i)

def plot(data, xlabel, ylabel, title, save_location):
    plt.plot(np.arange(data.shape[0]), data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.savefig(save_location+title)
    plt.clf()
    plt.close()

    # if not load_checkpoint:
    #     x = [i+1 for i in range(MAX_EPISODES)]
    #     plot_learning_curve(x, score_history, figure_file)
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
        Eval()

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
        f.writelines("state: {}\naction: {}\n a_bound:{}\n".format(config.state_dim,config.action_dim,config.a_bound))
        f.writelines("Max Episodes: {}\nMax Episodes steps: {}\n".format(MAX_EPISODES, MAX_EP_STEPS))
        f.writelines("LR_A: {}\nLR_C: {}\nGAMMA: {}\n".format(config.A_LR, config.C_LR, config.gamma))
        f.writelines("reward_scale: {}\ntau: {}\nmemory_capacity: {}\n".format(config.reward_scale,config.tau,config.MEMORY_CAPACITY))
        f.writelines("BATCH_SIZE: {}\nhidden_dim: {}\n".format(config.batch_size,config.hidden_dim))


if __name__ == "__main__":

    main()