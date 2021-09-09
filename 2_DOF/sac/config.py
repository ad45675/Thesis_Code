import numpy as np



ON_TRAIN = True
sac = True
if sac:
    # PATH_EVAL = ['03091043', '99']  # yao
    PATH_EVAL = ['03102109', '9']  # yao
    hidden_sizes=[256,256]

    reparameterize_critic = True
    reparameterize_actor = True
    # PATH_EVAL = ['12011347', '10']
    A_LR = 0.001
    C_LR = 0.001
    gamma= 0.9  # reward discount
    reward_scale = 0.001  #0.001 最好

    tau = 0.01  # soft replacement
    MEMORY_CAPACITY = 100000

    # layer4_size = 64

    batch_size = 128

    Tolerance=5

    MAX_EPISODES = 100
    MAX_EP_STEPS = 200
    eval_iteration = 10
    cost_iteration = 50

else:
    # PATH_EVAL = ['04261644', '999']  # for eval
    PATH_EVAL = ['10190058', '99']  # yao

    MAX_EPISODES = 500
    MAX_EP_STEPS = 200
    eval_iteration = 10
    cost_iteration = 50

    LR_A = 0.001  # learning rate for actor
    LR_C = 0.001  # learning rate for critic
    gamma = 0.9  # reward discount
    tau = 0.01  # soft replacement

    MEMORY_CAPACITY = 10000
    batch_size = 32
    neurons_actor = [64, 128, 64]
    layer_actor = np.size(neurons_actor) + 1
    neurons_critic = [64, 128, 256, 128]
    layer_critic = np.size(neurons_critic) + 1
    Tolerance = 5

    # for performance
    validation_size = 50
