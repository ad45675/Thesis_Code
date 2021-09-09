# this is network code
# use pytorch to bulid network
# 3 networks
# critic>> state + action-->q
# value >>    state-->v

import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal  # class torch.distributions.Normal(mean, std)
# 创建以 mean 和 std 为参数的正态分布(也称为高斯分布）
import config
import numpy as np

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []

    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, n_actions, hidden_sizes,name='critic'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.hidden_size = hidden_sizes
        self.name = name
        activation = nn.ReLU

        self.q = mlp([self.input_dims + n_actions] + list(hidden_sizes)+[1], activation)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        q = self.q(T.cat([state, action], dim=-1))
        # q = self.q(action_value) #shape torch.Size([32, 1])
        # q = (T.squeeze(q, -1)) #shape torch.Size([32])
        return T.squeeze(q, -1)

    def save_checkpoint(self, checkpoint_path):
        checkpoint_file = os.path.join(checkpoint_path, self.name + '_sac')
        T.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, path):
        checkpoint_file = './model/' + path[0] + '/net/' + path[1] + '/' + self.name + '_sac'
        self.load_state_dict(T.load(checkpoint_file))


class ValueNetwork(nn.Module):
    def __init__(self, beta, input_dims,hidden_sizes, name='value'):
        super(ValueNetwork, self).__init__()
        self.input_dims = input_dims
        self.name = name
        self.hidden_sizes = hidden_sizes
        activation = nn.ReLU

        self.v = mlp([self.input_dims] + list(hidden_sizes) + [1], activation)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        v = self.v(state)
        return v

    def save_checkpoint(self, checkpoint_path):
        checkpoint_file = os.path.join(checkpoint_path, self.name + '_sac')
        T.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, path):
        checkpoint_file = './model/' + path[0] + '/net/' + path[1] + '/' + self.name + '_sac'
        self.load_state_dict(T.load(checkpoint_file))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, n_actions, max_action, hidden_sizes, name='actor'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.name = name
        self.hidden_sizes = hidden_sizes
        self.max_action = max_action
        self.reparam_noise = 1e-6
        activation = nn.ReLU
        self.net = mlp([self.input_dims] + list(self.hidden_sizes), activation, activation)
        self.mu = nn.Linear(self.hidden_sizes[-1], self.n_actions)  # mean of the distribution for policy
        self.log_std = nn.Linear(self.hidden_sizes[-1], self.n_actions)  # standard deviation

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        net_out = self.net(state)

        mu = self.mu(net_out )  # mean of the distribution for policy
        log_std = self.log_std(net_out )  # standard deviation

        log_std = T.clamp(log_std, min=-20, max=2)  # T.clamp(input, min, max, out=None) → 限制在[MIN,MAX]範圍
        std = T.exp(log_std)  # (版2)

        return mu, std

    def sample_normal(self, state, reparameterize=True):
        mu, std  = self.forward(state)
        pi_distribution = T.distributions.Normal(mu, std)  # class torch.distributions.Normal(mean, std)

        mean = T.tanh(mu)* T.tensor(self.max_action).to(self.device)  # Only used for evaluating policy at test time.

        if reparameterize:
            pi_action = pi_distribution.rsample()
            # 而是先对标准正太分布N ( 0 , 1 )进行采样，然后输出：
            # mean+std ×採樣值
        else:
            pi_action = pi_distribution.sample()

        ## Compute logprob from Gaussian, and then apply correction for Tanh squashing.
        log_probs = pi_distribution.log_prob(pi_action).sum(axis=-1)  # 正態分布(版2)
        log_probs -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)  # (版2)

        # log_probs = pi_distribution.log_prob(pi_action)  #(版1)
        # log_probs -= T.log(1-pi_action.pow(2)+self.reparam_noise)#(版1)
        # log_probs = log_probs.sum(1, keepdim=True)#(版1)

        pi_action = T.tanh(pi_action)
        action = pi_action * T.tensor(self.max_action).to(self.device)
        return action , log_probs,mean

    def evaluate(self, state):  #增加的
        batch_mu, batch_std = self.forward(state)
        # print('log_std', batch_mu.shape,batch_std.shape)
        pi_distribution = T.distributions.Normal(batch_mu, batch_std)
        noise = Normal(0, 1)

        z = noise.sample()
        action = T.tanh(batch_mu + batch_std * z.to(self.device))
        log_prob = pi_distribution.log_prob(action).sum(axis=-1)  # 正態分布(版2)
        log_prob -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(axis=1)  # (版2)
        # log_prob = pi_distribution .log_prob(batch_mu + batch_std * z.to(self.device)) - T.log(1 - action.pow(2) + self.reparam_noise)
        return action, log_prob, z, batch_mu, batch_std

    def save_checkpoint(self, checkpoint_path):
        checkpoint_file = os.path.join(checkpoint_path, self.name + '_sac')
        T.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, path):
        checkpoint_file = './model/' + path[0] + '/net/' + path[1] + '/' + self.name + '_sac'
        self.load_state_dict(T.load(checkpoint_file))





