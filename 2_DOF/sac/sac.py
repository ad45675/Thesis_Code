#最原始的
import os
import torch as T
import torch.nn.functional as F
import numpy as np
from buffer import replaybuffer
from networks import ActorNetwork, CriticNetwork, ValueNetwork
import config

A_LR = config.A_LR
C_LR = config.C_LR
gamma = config.gamma
MEMORY_CAPACITY = config.MEMORY_CAPACITY
batch_size = config.batch_size
reward_scale = config.reward_scale


class SAC_agent(object):
    def __init__(self, input_dims, n_actions, max_action):
        self.input_dims, self.n_actions, self.max_action = input_dims, n_actions, max_action
        self.memory = replaybuffer(MEMORY_CAPACITY, input_dims, n_actions)

        self.actor = ActorNetwork(A_LR, input_dims, n_actions, max_action, name='actor')
        self.critic_1 = CriticNetwork(C_LR, input_dims, n_actions, name='critic_1')
        self.critic_2 = CriticNetwork(C_LR, input_dims, n_actions, name='critic_2')
        self.value = ValueNetwork(C_LR, input_dims, name='value')
        self.target_value = ValueNetwork(C_LR, input_dims, name='target_value')

        self.scale = reward_scale
        # print(self.scale)
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        state = T.Tensor([observation]).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)

        return actions.cpu().detach().numpy()[0]   
        #GPU類型轉成CPU，再繼續轉成numpy     
        #is a cuda tensor so we have ri send it to the cpu we detach from the graph and turn it to the numpy array and take the zeroth elememt


    def remember(self,state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = config.tau
        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters() #輸出參數的名稱（字符串）與這個參數（Parameter類）

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau*value_state_dict[name].clone() + \
                    (1-tau)*target_value_state_dict[name].clone()

        self.target_value.load_state_dict(value_state_dict)


    def save_models(self):
        print('.... saving models ....')
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()

    def learn(self):

        if self.memory.mem_cntr< batch_size:
            return

        state,action,reward,new_state,done=self.memory.sample_buffer(batch_size)

        #創建tensor T.tensor(data, dtype=None, device=None,requires_grad=False)
        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.actor.device)
        state = T.tensor(state, dtype=T.float).to(self.actor.device)
        action = T.tensor(action, dtype=T.float).to(self.actor.device)

        value = self.value(state).view(-1)          # view函数旨在reshape张量形状
        value_ = self.target_value(state_).view(-1) # X.view(-1)中的-1本意是根据另外一个数来自动调整维度，但是这里只有一个维度，因此就会将X里面的所有维度数据转化成一维的，并且按先后顺序排列。
        value_[done] = 0.0

        actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        # value networks loss and backprop (V=Q-log)
        self.value.optimizer.zero_grad() #將梯度初始化為0
        value_target = critic_value - self.scale*log_probs
        # value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True) #反向傳播,create_graph参数的作用是，如果为True可計算高階微分
        self.value.optimizer.step() #更新優化器學習率

        # actor networks loss and backprop (with reparameterize)
        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        actor_loss=self.scale*log_probs-critic_value
        # actor_loss = log_probs - critic_value
        actor_loss=T.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        # critic networks loss amd backprop
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        # q_hat=self.scale*reward+gamma*value_  #Q_=aR+r*v_
        q_hat = reward + gamma * value_
        q1_old_policy = self.critic_1.forward(state, action).view(-1)  #action is from the replay buffer
        q2_old_policy = self.critic_2.forward(state, action).view(-1)       
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)        

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.update_network_parameters()