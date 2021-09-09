
##-------------
#Deep-reinforcement-learning-with-pytorch
# self.value = nn.MSELoss()
# self.Q1= nn.MSELoss()
# self.Q2 = nn.MSELoss()

##-------------

import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from buffer import replaybuffer
from networks2 import ActorNetwork, CriticNetwork, ValueNetwork
import config

A_LR = config.A_LR
C_LR = config.C_LR
gamma = config.gamma
MEMORY_CAPACITY = config.MEMORY_CAPACITY
batch_size = config.batch_size
reward_scale = config.reward_scale
hidden_sizes=config.hidden_dim

class SAC_agent(object):
    def __init__(self, input_dims, n_actions, max_action):
        self.input_dims, self.n_actions, self.max_action = input_dims, n_actions, max_action
        self.memory = replaybuffer(MEMORY_CAPACITY, config.store_sate_dim, n_actions)

        self.actor = ActorNetwork(A_LR, input_dims, n_actions, max_action,hidden_sizes, name='actor')
        self.critic_1 = CriticNetwork(C_LR, input_dims, n_actions,hidden_sizes, name='critic_1')
        self.critic_2 = CriticNetwork(C_LR, input_dims, n_actions,hidden_sizes, name='critic_2')
        self.critic_1_target = CriticNetwork(C_LR, input_dims, n_actions, hidden_sizes, name='critic_1_target')
        self.critic_2_target = CriticNetwork(C_LR, input_dims, n_actions, hidden_sizes, name='critic_2_target')
        self.scale = reward_scale
        self.update_network_parameters(tau=1)

    def choose_action(self, observation, reparameterize=False, ON_TRAIN = False):
        state = T.Tensor(observation).to(self.actor.device)

        if ON_TRAIN:
            actions, _,_ = self.actor.sample_normal(state, reparameterize=False)
        else:
            _, _, actions = self.actor.sample_normal(state, reparameterize=False)


        return actions.cpu().detach().numpy()[0]
        #GPU類型轉成CPU，再繼續轉成numpy
        #is a cuda tensor so we have ri send it to the cpu we detach from the graph and turn it to the numpy array and take the zeroth elememt


    def remember(self,state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = config.tau
        critic_1_target_params = self.critic_1_target.named_parameters()
        critic_1_params = self.critic_1.named_parameters() #輸出參數的名稱（字符串）與這個參數（Parameter類）

        critic_1_target_state_dict = dict(critic_1_target_params)
        critic_1_dict = dict(critic_1_params)

        for name in critic_1_dict:
            critic_1_dict[name] = tau*critic_1_dict[name].clone() + \
                    (1-tau)*critic_1_target_state_dict[name].clone()

        self.critic_1_target.load_state_dict(critic_1_dict)

    def update_network_parameters2(self, tau=None):
        if tau is None:
            tau = config.tau
        critic_2_target_params = self.critic_2_target.named_parameters()
        critic_2_params = self.critic_2.named_parameters()  # 輸出參數的名稱（字符串）與這個參數（Parameter類）

        critic_2_target_state_dict = dict(critic_2_target_params)
        critic_2_dict = dict(critic_2_params)

        for name in critic_2_dict:
            critic_2_dict[name] = tau * critic_2_dict[name].clone() + \
                                  (1 - tau) * critic_2_target_state_dict[name].clone()

        self.critic_2_target.load_state_dict(critic_2_dict)

    def save_models(self, path, i):
        print('.... saving models ....')
        os.makedirs('./model/' + path + '/net/' + str(int(i)))
        checkpoint_path = os.path.join('./model/' + path + '/net/' + str(int(i)))
        self.actor.save_checkpoint(checkpoint_path)
        # self.value.save_checkpoint(checkpoint_path)
        # self.target_value.save_checkpoint(checkpoint_path)
        self.critic_1_target.save_checkpoint(checkpoint_path)
        self.critic_2_target.save_checkpoint(checkpoint_path)
        self.critic_1.save_checkpoint(checkpoint_path)
        self.critic_2.save_checkpoint(checkpoint_path)


    def load_models(self,path):
        print('.... loading models ....')

        self.actor.load_checkpoint(path)
        # self.value.load_checkpoint(path)
        # self.target_value.load_checkpoint(path)
        self.critic_1_target.load_checkpoint(path)
        self.critic_2_target.load_checkpoint(path)
        self.critic_1.load_checkpoint(path)
        self.critic_2.load_checkpoint(path)

    def learn(self):

        if self.memory.mem_cntr < batch_size:
            return 0, 0, 0

        state, action, reward, new_state, done = self.memory.sample_buffer(batch_size)

        # 創建tensor T.tensor(data, dtype=None, device=None,requires_grad=False)
        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.actor.device)
        state = T.tensor(state, dtype=T.float).to(self.actor.device)
        action = T.tensor(action, dtype=T.float).to(self.actor.device)

        # Bellman backup for Q functions
        with T.no_grad():
            # Target actions come from *current* policy
            action1, log_prob, _ = self.actor.sample_normal(state_, reparameterize=config.reparameterize_critic)  # 看有沒有sample
            # Target Q-values
            q1_pi_targ = self.critic_1_target.forward(state_, action1)
            q2_pi_targ = self.critic_2_target.forward(state_, action1)
            q_pi_targ = T.min(q1_pi_targ, q2_pi_targ)
            backup = reward + gamma * (q_pi_targ - self.scale * log_prob)

        #    # Set up function for computing SAC Q-losses
        q1 = self.critic_1.forward(state, action)
        q2 = self.critic_2.forward(state, action)
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()

        # loss_q = loss_q1 + loss_q2
        self.critic_1.zero_grad()
        loss_q1.backward(retain_graph=True)
        self.critic_1.optimizer.step()

        self.critic_2.zero_grad()
        loss_q2.backward(retain_graph=True)
        self.critic_2.optimizer.step()

        # Set up function for computing SAC pi loss
        action2, log_prob2, _ = self.actor.sample_normal(state, reparameterize=config.reparameterize_actor)
        q1_pi = self.critic_1.forward(state, action2)
        q2_pi = self.critic_2.forward(state, action2)
        q_pi = T.min(q1_pi, q2_pi)
        loss_pi = (self.scale * log_prob2 - q_pi).mean()
        # Entropy-regularized policy loss

        self.actor.optimizer.zero_grad()
        loss_pi.backward(retain_graph=True)
        self.actor.optimizer.step()

        self.update_network_parameters()
        self.update_network_parameters2()

        return loss_q1.item(), loss_q2.item(), loss_pi.item()
        # 呼叫optimizer的zero_grad方法，將所有參數的梯度緩衝區（buffer）歸零
        # 呼叫loss的backward()方法開始進行反向傳播
        # 呼叫optimizer的step()方法來更新權重。