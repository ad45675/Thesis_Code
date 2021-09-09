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
        self.hidden_dim = hidden_sizes
        self.name = name
        init_w = 3e-3
        #----------cnn extractor----------#
        if (config.color_state):
            self.cnn1 = nn.Sequential(nn.Conv2d(3, 32, 8, 4), nn.ReLU())
        else:
            self.cnn1 = nn.Sequential(nn.Conv2d(1, 32, 8, 4), nn.ReLU())
        self.cnn2 = nn.Sequential(nn.Conv2d(32, 64, 4, 2), nn.ReLU())
        self.cnn3 = nn.Sequential(nn.Conv2d(64, 64, 3, 1), nn.ReLU())
        self.linear = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        #----------cnn extractor end----------#
        self.linear1 = nn.Linear(self.input_dims + n_actions, self.hidden_dim[0])
        self.linear2 = nn.Linear(self.hidden_dim[0], self.hidden_dim[1])

        self.q = nn.Linear(self.hidden_dim[1], 1)
        self.q.weight.data.uniform_(-init_w, init_w)
        self.q.bias.data.uniform_(-init_w, init_w)
        self.optimizer = optim.Adam(self.parameters(), lr=beta)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        #state shape為[None,2,64,64]

        if(config.color_state):
            img = state/255
            img = img.view(-1, 3, config.image_input, config.image_input)
        else:
            img = state[:,:1,:,:]
            img = img.view(-1,1,64,64)

        img = self.cnn1(img)
        img = self.cnn2(img)
        img = self.cnn3(img)
        img = img.view(-1,1024)
        img = self.linear(img)
        feature = img

        q = T.cat([feature, action], dim=1)
        q = F.relu(self.linear1(q))
        q = F.relu(self.linear2(q))
        q = self.q(q)

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

        init_w = 3e-3

        #----------cnn extractor----------#
        self.cnn1 = nn.Sequential(nn.Conv2d(1, 32, 8, 4), nn.ReLU())
        self.cnn2 = nn.Sequential(nn.Conv2d(32, 64, 4, 2), nn.ReLU())
        self.cnn3 = nn.Sequential(nn.Conv2d(64, 64, 3, 1), nn.ReLU())
        self.linear = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        #----------cnn extractor end----------#
        self.linear1 = nn.Linear(self.input_dims, self.hidden_dim[0])
        self.linear2 = nn.Linear(self.hidden_dim[0],self. hidden_dim[1])
        self.v = nn.Linear(self.hidden_dim[1], 1)

        self.v.weight.data.uniform_(-init_w, init_w)
        self.v.bias.data.uniform_(-init_w, init_w)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        # state shape為[None,2,64,64]

        img = state[:, :1, :, :]
        img = img.view(-1, 1, 64, 64)
        img = self.cnn1(img)
        img = self.cnn2(img)
        img = self.cnn3(img)
        img = img.view(-1, 1024)
        img = self.linear(img)
        feature = img

        #----------提取feature完
        v = F.relu(self.linear1(feature))
        v = F.relu(self.linear2(v))
        v = self.v(v)

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
        self.hidden_dim = hidden_sizes
        self.max_action = max_action
        self.reparam_noise = 1e-6
        # activation = nn.ReLU
        init_w = 3e-3
        # self.net = mlp([self.input_dims] + list(self.hidden_sizes), activation, activation)
        #----------cnn extractor----------#
        if(config.color_state):
            self.cnn1 = nn.Sequential(nn.Conv2d(3, 32, 8, 4), nn.ReLU())
        else:
            self.cnn1 = nn.Sequential(nn.Conv2d(1, 32, 8, 4), nn.ReLU())
        self.cnn2 = nn.Sequential(nn.Conv2d(32, 64, 4, 2), nn.ReLU())
        self.cnn3 = nn.Sequential(nn.Conv2d(64, 64, 3, 1), nn.ReLU())
        self.linear = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        #----------cnn extractor end----------#
        self.linear1 = nn.Linear(self.input_dims, self.hidden_dim[0])
        self.linear2 = nn.Linear(self.hidden_dim[0], self.hidden_dim[1])


        self.mu = nn.Linear(self.hidden_dim[1], self.n_actions)  # mean of the distribution for policy
        self.mu.weight.data.uniform_(-init_w, init_w)
        self.mu.bias.data.uniform_(-init_w, init_w)

        self.log_std = nn.Linear(self.hidden_dim[1], self.n_actions)  # standard deviation
        self.log_std .weight.data.uniform_(-init_w, init_w)
        self.log_std .bias.data.uniform_(-init_w, init_w)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        # state shape為[None,2,64,64]

        if(config.color_state):
            img = state/255
            img = img.view(-1, 3, config.image_input, config.image_input)
        else:
            img = state[:, :1, :, :]
            img = img.view(-1, 1, config.image_input, config.image_input)

        img = self.cnn1(img)
        img = self.cnn2(img)
        img = self.cnn3(img)
        img = img.view(-1, 1024)
        img = self.linear(img)
        feature = img


        net_out = F.relu(self.linear1(feature))
        net_out = F.relu(self.linear2(net_out))
        mu = self.mu(net_out )  # mean of the distribution for policy
        log_std = self.log_std(net_out )  # standard deviation

        log_std = T.clamp(log_std, min=-20, max=2)  # T.clamp(input, min, max, out=None) → 限制在[MIN,MAX]範圍
        std = T.exp(log_std)  # (版2)

        return mu, std

    def sample_normal(self, state, reparameterize=True):
        mu, std  = self.forward(state)
        pi_distribution = T.distributions.Normal(mu, std)  # class torch.distributions.Normal(mean, std)
        mean = T.tanh(mu) * T.tensor(self.max_action).to(self.device)  # Only used for evaluating policy at test time.

        if reparameterize:
            pi_action = pi_distribution.rsample()
            # 而是先对标准正太分布N ( 0 , 1 )进行采样，然后输出：
            # mean+std ×採樣值
        else:
            pi_action = pi_distribution.sample()


        log_probs = pi_distribution.log_prob(pi_action).sum(axis=-1)  # 正態分布(版2)
        log_probs -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)  # (版2)
        # log_probs -= T.log(T.tensor(self.max_action).to(self.device)*(1-pi_action.pow(2))+self.reparam_noise)#(版1)
        # log_probs = log_probs.sum(1, keepdim=True)#(版1)

        tan_action = T.tanh(pi_action)
        action = tan_action * T.tensor(self.max_action).to(self.device)
        return action , log_probs, mean

    def evaluate(self, state):  #增加的(沒有用到)
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