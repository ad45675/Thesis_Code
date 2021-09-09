# this is network code
# use pytorch to bulid network
#3 networks
#critic>> state + action-->q
#value >>    state-->v
#最原始的
import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal   #class torch.distributions.Normal(mean, std)
                                                #创建以 mean 和 std 为参数的正态分布(也称为高斯分布）
import config
import numpy as np

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, n_actions, fc1_dims=config.layer1_size, fc2_dims=config.layer2_size,
            name='critic', chkpt_dir='tmp/sac'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')   


        self.fc1 = nn.Linear(self.input_dims+n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        action_value = self.fc1(T.cat([state, action], dim=1))
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)

        q = self.q(action_value)

        return q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class ValueNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims=config.layer1_size, fc2_dims=config.layer1_size,
            name='value', chkpt_dir='tmp/sac'):
        super(ValueNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, fc2_dims)
        self.v = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)    

    def forward(self, state):
        state_value = self.fc1(state)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)

        v = self.v(state_value)

        return v

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))   

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims,n_actions, max_action, fc1_dims=config.layer1_size,
            fc2_dims=config.layer1_size, name='actor', chkpt_dir='tmp/sac'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        self.max_action =  max_action
        self.reparam_noise = 1e-6

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        self.mu = nn.Linear(self.fc2_dims, self.n_actions)    #mean of the distribution for policy
        # self.log_std = nn.Linear(self.fc2_dims, self.n_actions) #standard deviation
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)    

    # def forward(self, state):
    #     prob = self.fc1(state)
    #     prob = F.relu(prob)
    #     prob = self.fc2(prob)
    #     prob = F.relu(prob)
    #
    #     mu = self.mu(prob)  #mean of the distribution for policy
    #     log_std = self.log_std(prob) #standard deviation(標準差)
    #     log_std = T.clamp(log_std, min=self.reparam_noise, max=2) #T.clamp(input, min, max, out=None) → 限制在[MIN,MAX]範圍
    #     sigma = T.exp(log_std)
    #     # sigma  = T.clamp(sigma , min=-20, max=2)
    #     # print('sigma ',sigma )
    #     return mu, sigma


    # def sample_normal(self, state,  reparameterize=True):
    #     mu, sigma = self.forward(state)
    #     probabilities = Normal(mu, sigma) #class torch.distributions.Normal(mean, std)
    #
    #     # if reparameterize:
    #     #     # Only used for evaluating policy at test time.
    #     #     actions = probabilities.rsample()
    #     #     #而是先对标准正太分布N ( 0 , 1 ) N(0,1)N(0,1)进行采样，然后输出：
    #     #     #mean+std ×採樣值
    #     #
    #     # else:
    #     #     actions = mu
    #
    #     if reparameterize:
    #         actions = probabilities.rsample()
    #         #     #而是先对标准正太分布N ( 0 , 1 )进行采样，然后输出：
    #         #     #mean+std ×採樣值
    #         # print('actions ',actions )
    #     else:
    #         actions = probabilities.sample()#高斯分布
    #
    #     # action = T.tanh(actions)*T.tensor([1.1]).to(self.device)
    #     # calculate entropies
    #     log_probs = probabilities.log_prob(actions).sum(axis=-1) #正態分布
    #     log_probs-= (2 * (np.log(2) -actions - F.softplus(-2 * actions))).sum(axis=1)
    #     # log_probs -= T.log(1-actions.pow(2)+self.reparam_noise).sum(axis=-1)
    #     action = T.tanh(actions)
    #     # print(type(action .type))
    #     action = self.max_action *(action)
    #
    #     return action, log_probs

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        mu = self.mu(prob)
        sigma = self.sigma(prob)

        sigma = T.clamp(sigma, min=self.reparam_noise, max=1)
        sigma = T.exp(sigma)
        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)

        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        action = T.tanh(actions) * T.tensor(self.max_action).to(self.device)

        log_probs = probabilities.log_prob(actions).sum(axis=-1)

        # log_probs -= T.log(1-actions.pow(2)+self.reparam_noise).sum(axis=1)
        log_probs -= (2 * (np.log(2) - actions - F.softplus(-2 * actions))).sum(axis=1)

        # log_probs = log_probs.sum(axis=1, keepdim=True)


        return action, log_probs




    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))