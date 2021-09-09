"""
Ting yu
"""
import numpy as np
import matplotlib.pyplot as plt
file = ['01201521', '01201536'] # critic SAC DDPG
# file = ['05061437', '05061458', '05061558', '05061738']  # actor

for i in range(len(file)):
    f = open('./model/' + file[i] + '/train/data/' + 'step.txt', 'r')
    data = (np.loadtxt(f))
    smooth_data = []
    for j in range(np.size(data)):
        if j == 0:
            smooth_data.append(data[j])
        else:
            smooth_data.append(smooth_data[-1]*0.9 + data[j]*0.1)
    plt.plot(smooth_data )

plt.title('finish step')
plt.ylabel('step')
plt.xlabel('episode')
plt.legend(['SAC', 'DDPG', '3', '4', '5'])
plt.grid(True)
# file_name = 'perfomance_compare_actor.png'
file_name = 'step.png'
plt.savefig('./model/perfomance' + file_name)
plt.show()
# plt.clf()
