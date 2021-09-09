
"""
use to test my sac algorithem
"""
import torch as T
import config
from env3 import ArmEnv
import os
import numpy as np
import time
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()


ON_TRAIN = config.ON_TRAIN
sac= config.sac


if ON_TRAIN:
    Mode = 'Train'
else:
    Mode = 'Eval'


# set env
env = ArmEnv(Mode)
# set RL method (continuous)
s_dim = env.state_dim
a_dim = env.action_dim
a_bound = [[env.arm1_bound[1], env.arm2_bound[1]]]




if sac:
    from sac5 import SAC_agent
    Method = 'sac'
    rl = SAC_agent(s_dim, a_dim, a_bound)


else:
    from RL_brain_ddpg import DDPG
    # from sac2 import SAC_agent
    Method = 'ddpg'
    rl = DDPG(a_dim, s_dim, a_bound)




print( Method)

# for train
MAX_EPISODES = config.MAX_EPISODES
MAX_EP_STEPS = config.MAX_EP_STEPS
eval_iteration = config.eval_iteration
cost_iteration = config.cost_iteration
PATH = time.strftime('%m%d%H%M')

# for eval
PATH_EVAL = config.PATH_EVAL



load_checkpoint = False
def train():
    # start training
    loss_critic1, loss_critic2, loss_actor1 = 0,0,0
    view = False
    # ----- record memory ----- #
    step_set, reward_set, avg_reward_set, repeat_set,  loss_q1_set,  loss_q2_set,loss_pi_set = [], [], [], [], [], [],[]
    for i in range(MAX_EPISODES):
        s = env.reset()
        # writer.add_scalar('state', s.data[0], i, walltime=None)

        # env.render()
        ep_r, count_cost_store, count_repeat = 0, 0, 0
        for j in range(MAX_EP_STEPS):
            # view = False
            if view:
                env.render()
            if sac:
                a = rl.choose_action(s,ON_TRAIN =True)
            else:
                a = rl.choose_action(s)
            # print('a',a)


            inf = [i, j]

            s_, r, done = env.step(a)

            if sac:
                rl.remember(s, a, r, s_,done) #sac
            else:
                rl.store_transition(s, a, r, s_,done) #ddpg


            ep_r += r

            if not load_checkpoint:
                # start to learn once has fulfilled the memory
                loss_critic1,loss_critic2,loss_actor1 = rl.learn()
                # rl.learn()
                # cost_actor, cost_critic = rl.learn()
                # count_cost_store += 1
                # if count_cost_store % cost_iteration == 0:
                loss_q1_set.append(loss_critic1)
                loss_q2_set.append(loss_critic2)
                loss_pi_set.append(loss_actor1)
            else:
                loss_critic1, loss_critic2, loss_actor1 = 0, 0, 0
                print("loss = 0")


            if (s == s_).all():
                count_repeat += 1

            if done or j == MAX_EP_STEPS-1:

                print('Ep: %i | %s | repeat: %d | ep_r: %.1f | step: %i' % (i, '----' if not done else 'done', count_repeat, ep_r, j+1))
                step_set.append(j+1)
                reward_set.append(ep_r)
                avg_reward_set.append(ep_r/(j+1))
                repeat_set.append(count_repeat)

                break
            s = s_
        if i % eval_iteration == 0:
            if sac:
                # rl.save_models(PATH, i/eval_iteration)  # sac
                rl.save_models(PATH, i / eval_iteration)  # sac
            else:
                rl.save(PATH, i / eval_iteration) #ddpg

    # save data
    folder_data = './model/' + PATH + '/train/data/'
    # file_name = ['step.txt', 'reward.txt', 'avgR.txt', 'repeat.txt', 'cost_actor.txt', 'cost_critic.txt']
    # data_set = ['step_set', 'reward_set', 'avg_reward_set', 'repeat_set', 'cost_actor_set', 'cost_critic_set']
    file_name = ['step.txt', 'reward.txt', 'avgR.txt', 'repeat.txt','critic1_loss.txt','critic2_loss.txt','actor_loss.txt']
    data_set = ['step_set', 'reward_set', 'avg_reward_set', 'repeat_set','loss_q1_set','loss_q2_set','loss_pi_set']
    for i in range(len(file_name)):
        save_txt(path=folder_data, name=file_name[i], data=eval(data_set[i]))

    # plot fig
    folder_fig = './model/' + PATH + '/train/fig/'
    xlabel = ['Episode', 'Episode', 'Episode', 'Episode','episo','episo','episo']
    ylabel = ['step', 'r', 'avgR', 'repeat', 'loss', 'loss','loss']
    title = ['step', 'reward', 'avgR', 'repeat', 'cost_critic1','cost_critic2','cost_actor']
    for i in range(len(file_name)):
        plot_txt(path=folder_data, name=file_name[i], xlabel=xlabel[i], ylabel=ylabel[i], title=title[i],
                 save_location=folder_fig)

def save_parameter():

    with open('./model/' + PATH + '/train/parameter.txt', 'w') as f:
        f.writelines("Max Episodes: {}\nMax Episodes steps: {}\nEval iteration: {}\ncost iteration: {}\nMethod: {}\n".format(MAX_EPISODES, MAX_EP_STEPS, eval_iteration,cost_iteration,Method))
        if sac:
            f.writelines("BATCH_SIZE: {}\nhidden_sizes: {}\n".format(config.batch_size, config.hidden_sizes))
            f.writelines("LR_A: {}\nLR_C: {}\nGAMMA: {}\nTAU: {}\nMEMORY_CAPACITY: {}\n".format(config.A_LR, config.C_LR,config.gamma, config.tau,config.MEMORY_CAPACITY))
            f.writelines("reward_scale: {}\n".format(config.reward_scale))
        else:
            f.writelines("LR_A: {}\nLR_C: {}\nGAMMA: {}\nTAU: {}\nMEMORY_CAPACITY: {}\n".format(config.LR_A, config.LR_C,config.gamma, config.tau,config.MEMORY_CAPACITY))
            f.writelines("BATCH_SIZE: {}\nlayer_actor: {}\nneurons_actor: {}\nlayer_critiv: {}\nneurons_critiv: {}\n".format(config.batch_size, config.layer_actor, config.neurons_actor,config.layer_critic, config.neurons_critic))
        f.writelines("Tolerance: {}\n".format(config.Tolerance))

def Eval():
    if sac:
        print(PATH_EVAL)
        rl.load_models(PATH_EVAL)


    else:
        rl.restore(PATH_EVAL)
    env.render()
    env.viewer.set_vsync(True)
    s = env.reset()
    while True:
    # for i in range(5):
        env.render()
        a = rl.choose_action(s,ON_TRAIN = False)
        if sac:
            s, r, done = env.step(a)
        else:
            s, r, done = env.step(a, [0, 0])
        with SummaryWriter(log_dir='./sac_logs', comment='actor') as writer:
            s_in = T.tensor([s]).to(T.device('cuda:0' if T.cuda.is_available() else 'cpu')).float()
            a_in = T.tensor([a]).to(T.device('cuda:0' if T.cuda.is_available() else 'cpu')).float()
            writer.add_graph(rl.actor, (s_in))
            # writer.add_graph(rl.critic_1, (s_in,a_in))
            # writer.add_graph(rl.critic_1_target, (s_in, a_in))

def Eval_train():
    # special state
    point = 19
    index = load_txt(path='F:/tingyu/python/compare/', name='index.txt')
    validation = load_txt(path='F:/tingyu/python/compare/', name='validation.txt')
    creat_path('./model/' + PATH_EVAL[0] + '/eval_train/' + str(point + 1) + '/data')
    creat_path('./model/' + PATH_EVAL[0] + '/eval_train/' + str(point + 1) + '/fig')

    data = validation[int(index[point] - 1), :]

    step_set, reward_set, avg_reward_set = [], [], []
    for i in range(200):
        path = [PATH_EVAL[0], str(i * 5)]
        rl.restore(path)
        ep_r = 0
        s = env.validation_reset([0, 0, data[0], data[1], data[2], data[3]])
        for t in range(MAX_EP_STEPS):
            a = rl.choose_action(s)
            s_, r, done = env.step(a, [0, 0])
            s = s_
            ep_r += r

            if done or t == MAX_EP_STEPS - 1:
                step = t + 1
                break
        step_set.append(step)
        reward_set.append(ep_r)
        avg_reward_set.append(ep_r / step)
        print("Episode:{} step:{} ep_r:{} avgR_:{}".format(i + 1, step, ep_r, ep_r / step))
    # ----- record data ----- #
    folder_data = './model/' + PATH_EVAL[0] + '/eval_train/' + str(point + 1) + '/data/'
    file_name = ['step.txt', 'reward.txt', 'avgR.txt']
    data_set = ['step_set', 'reward_set', 'avg_reward_set']
    for i in range(len(file_name)):
        save_txt(path=folder_data, name=file_name[i], data=eval(data_set[i]))

    # plot data
    folder_fig = './model/' + PATH_EVAL[0] + '/eval_train/' + str(point + 1) + '/fig/'
    xlabel = ['Episode', 'Episode', 'Episode']
    ylabel = ['step', 'r', 'avgR']
    title = ['step', 'reward', 'avgR']
    for i in range(len(file_name)):
        plot_txt(path=folder_data, name=file_name[i], xlabel=xlabel[i], ylabel=ylabel[i], title=title[i],
                 save_location=folder_fig)

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


def save_model_parameter():
    # load model
    rl.restore(PATH_EVAL)
    path = './model/' + PATH_EVAL[0] + '/net/' + PATH_EVAL[1] + '/'

    s = env.reset()
    a = rl.choose_action(s)
    x = s[np.newaxis, :]

    for i in range(config.layer_actor):
        layer_name = 'layer_{}'.format(i+1)
        # --- get layer parameter --- #
        if i < config.layer_actor-1:
            weight, bias = get_layer_parameter(scope='Actor/eval/'+layer_name)
            x = np.maximum(0, np.dot(x, weight[0]) + bias)
        else:
            weight, bias = get_layer_parameter(scope='Actor/eval/a')
            x = np.tanh(np.dot(x, weight[0]) + bias)

        # --- save layer parameter --- #
        save_txt(path=path, name=layer_name + '_w.txt', data=weight[0])
        save_txt(path=path, name=layer_name + '_b.txt', data=bias[0])

    out = np.multiply(x, a_bound)
    print("tensorflow : {}\n out : {}".format(a, out))

    # record testing data
    num_data = 10
    test_data = np.zeros((num_data, s_dim + a_dim))
    for i in range(num_data):
        s = env.reset()
        a = rl.choose_action(s)
        test_data[i, 0:s_dim] = s
        test_data[i, s_dim:s_dim+a_dim] = a
    save_txt(path=path, name='test_data.txt', data=test_data)

# def get_layer_parameter(scope):
#     weight = rl.sess.run(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope+'/kernel:0'))
#     bias = rl.sess.run(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope+'/bias:0'))
#
#     return weight, bias

def validation_produce():

    validation_set = []
    for i in range(config.validation_size):
        s = env.reset_validation()
        validation_set.append(s)

    path = './model/' + PATH_EVAL[0] + '/test/'
    folder = 1
    while True:
        if path_exsit(path + '{}'.format(folder)):
            folder += 1
        else:
            os.mkdir(path + '{}'.format(folder))
            break
    path_data = path + '{}/'.format(folder)
    save_txt(path=path_data, name='validation.txt', data=validation_set)

def performance():
    rl.restore(PATH_EVAL)
    folder = 1
    path_data = './model/' + PATH_EVAL[0] + '/test/'+ '{}/'.format(folder)

    # load validation
    validation = load_txt(path=path_data, name='validation.txt')

    ep_r_set, ep_step_set, ep_distance_set = [], [], []
    total_r, total_step, total_distance = 0, 0, 0
    for i in range(validation.shape[0]):
        s = env.validation_reset(validation[i, :])
        ep_r = 0
        for t in range(MAX_EP_STEPS):
            a = rl.choose_action(s)
            s_, r, done = env.step(a, [0, 0])
            s = s_
            ep_r += r
            if done:
                break
        total_r += ep_r
        total_step += (t + 1)
        total_distance += env.distance
        ep_r_set.append(ep_r)
        ep_step_set.append(t + 1)
        ep_distance_set.append(env.distance)
    avgStep = total_step / validation.shape[0]
    avgReward = total_r / validation.shape[0]
    avgDistance = total_distance / validation.shape[0]
    print("avgStep:{}\t avgReward;{}\t distance:{}".format(avgStep, avgReward, avgDistance))

    # save data
    save_txt(path=path_data, name='step.txt', data=np.array(ep_step_set)[:, np.newaxis], fmt='%d')
    save_txt(path=path_data, name='reward.txt', data=np.array(ep_r_set)[:, np.newaxis], fmt='%f')
    save_txt(path=path_data, name='distance.txt', data=np.array(ep_distance_set)[:, np.newaxis], fmt='%f')
    f_performance = open(path_data + 'performance.txt', 'w')
    f_performance.writelines("avgStep:{}\navgReward:{}\navgDistance:{}\n".format(avgStep, avgReward, avgDistance))
    f_performance.close()

def save_txt(path, name, data, fmt='%f'):
    f = open(path + name, 'w')
    np.savetxt(f, data, fmt=fmt)
    f.close()

def load_txt(path, name):
    f = open(path + name, 'r')
    data = np.loadtxt(f)
    f.close()
    return data

def plot(data, xlabel, ylabel, title, save_location):

    plt.plot(np.arange(data.shape[0]), data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.savefig(save_location + title)
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

if __name__ == "__main__":

    main()
    #validation_produce()
    #save_model_parameter()  # save model parameter
    #Eval_train()  # special state
    #performance()  # average N situation



