import numpy as np


ON_TRAIN = True



epsilon = 0.2

MAX_EPISODES = 1000
MAX_EP_STEPS = 200
eval_iteration = 10
cost_iteration = 50

LR_A = 0.001    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GAMMA = 0.9     # reward discount
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
BATCH = 32
neurons_actor = [64, 128, 64]
layer_actor = np.size(neurons_actor) + 1
neurons_critic = [64, 128, 256, 128]
layer_critic = np.size(neurons_critic) + 1

Tolerance = 5

# for performance
validation_size = 200

TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32