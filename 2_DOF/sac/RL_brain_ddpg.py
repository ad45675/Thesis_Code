import tensorflow as tf
import numpy as np
import config
import os
from buffer import replaybuffer

####################  hyper parameters  ####################

LR_A = config.LR_A    # learning rate for actor
LR_C = config.LR_C    # learning rate for critic
GAMMA = config.gamma     # reward discount
TAU = config.tau      # soft replacement
MEMORY_CAPACITY = config.MEMORY_CAPACITY  # memory size
BATCH_SIZE = config.batch_size  # batch size
layer_actor = config.layer_actor  # Actor Network layer
neurons_actor = config.neurons_actor  # Actor Network Design
layer_critic = config.layer_critic  # Critic Network layer
neurons_critic = config.neurons_critic  # Critic Network Design

class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound):
        # self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)  # ( s, s_, a, r )
        self.memory = replaybuffer(MEMORY_CAPACITY, s_dim, a_dim)
        # self.pointer = 0
        self.memory_full = False
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        # Create Network Architecture
        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)

        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [[tf.assign(ta, (1 - TAU) * ta + TAU * ea), tf.assign(tc, (1 - TAU) * tc + TAU * ec)]
                             for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]

        q_target = self.R + GAMMA * q_
        self.td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(self.td_error, var_list=self.ce_params)

        self.a_loss = tf.reduce_mean(q)    # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(- self.a_loss, var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[None, :]})[0]

    def learn(self):
        # soft target replacement
        self.sess.run(self.soft_replace)
        #
        # indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        # bt = self.memory[indices, :]
        # bs = bt[:, :self.s_dim]
        # ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        # br = bt[:, -self.s_dim - 1: -self.s_dim]
        # print('br',br.shape)
        # bs_ = bt[:, -self.s_dim:]

        if self.memory.mem_cntr < BATCH_SIZE:
            return

        state,action,reward,new_state,done=self.memory.sample_buffer(BATCH_SIZE)
        bs = state
        ba = action
        br = np.reshape(reward,(BATCH_SIZE,1))
        bs_ = new_state
        done=done


        cost_a = self.sess.run(self.a_loss, {self.S: bs})
        self.sess.run(self.atrain, {self.S: bs})
        cost_c = self.sess.run(self.td_error, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

        return cost_a, cost_c

    def store_transition(self, s, a, r, s_,done):
        # transition = np.hstack((s, a, [r], s_))
        # index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        # self.memory[index, :] = transition
        # self.pointer += 1
        # if self.pointer > MEMORY_CAPACITY:      # indicator for learning
        #     self.memory_full = True

        self.memory.store_transition(s, a, r, s_, done)

    def _build_a(self, x, scope, trainable):

        with tf.variable_scope(scope):
            for i, layer_dim in enumerate(neurons_actor, 1):
                layer_name = 'layer_{}'.format(i)
                x = tf.layers.dense(x, layer_dim, activation=tf.nn.relu, name=layer_name, trainable=trainable)
            a = tf.layers.dense(x, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)  # product 0~1 value
            # print('a',a)
            # print('scaled_a',tf.multiply(a, self.a_bound))
            return tf.multiply(a, self.a_bound, name='scaled_a')  # remap to workspace

    def _build_c(self, s, a, scope, trainable):

        with tf.variable_scope(scope):
            n_l1 = neurons_critic[0]
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            x = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            neurons = neurons_critic[1:]
            for i, layer_dim in enumerate(neurons, 2):
                layer_name = 'layer_{}'.format(i)
                x = tf.layers.dense(x, layer_dim, activation=tf.nn.relu, name=layer_name, trainable=trainable)
            return tf.layers.dense(x, 1, trainable=trainable)  # Q(s,a)

    def save(self, path, i):
        saver = tf.train.Saver(max_to_keep=0)
        os.makedirs('./model/'+path+'/net/'+str(int(i)))
        saver.save(self.sess, './model/'+path+'/net/'+str(int(i))+'/params', write_meta_graph=False)

    def restore(self, path):
        saver = tf.train.Saver()
        saver.restore(self.sess, './model/'+path[0]+'/net/'+path[1]+'/params')

