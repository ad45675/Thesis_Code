
import matplotlib.pyplot as plt
import numpy as np
import os

#
# plt.rcParams['font.sans-serif'] = ['DFKai-SB']
# plt.rcParams['axes.unicode_minus'] = False




def load_txt(path, name):
    f = open(path + name, 'r')
    data = np.loadtxt(f)
    f.close()
    return data

def path_exsit(path):
    if os.path.exists(path):
        return True
    else:
        return False

def plot(data, xlabel, ylabel, title, save_location):
    plt.plot(np.arange(data.shape[0]), data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title,fontsize = 14,fontweight = 'bold')
    plt.grid(True)
    plt.savefig(save_location+title)
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


"""畫碩論圖6-17(no-yolo) """
# PATH = '07082329'
# light_rgb = ([135 ,206 ,250],[	255, 231 ,186],[255 ,193 ,193])
# """ rename """
# folder_data = './' + PATH + '/train/data/'
# file_name = ['step.txt', 'reward.txt', 'avgR.txt','critic1_loss.txt','critic2_loss.txt','actor_loss.txt']
# data_set = ['step_set', 'reward_set', 'avg_reward_set', 'loss_q1_set', 'loss_q2_set', 'loss_pi_set']
#
# # ----- plot fig -----
# folder_fig = './' + PATH + '/train/fig/'
# xlabel = ['Episode', 'Episode', 'Episode', 'Episode', 'Episode', 'Episode']
# ylabel = ['Grasp Times', 'Reward', 'avgR', 'loss', 'loss', 'loss']
# title = ['Grasp Times', 'Accumulative Reward', 'avgR', 'cost_critic1', 'cost_critic2', 'cost_actor']
# for i in range(len(file_name)):
#     plot_txt(path=folder_data, name=file_name[i], xlabel=xlabel[i], ylabel=ylabel[i], title=title[i],
#              save_location=folder_fig)




""" smooth
 '06121418' --> 畫碩論圖6-15
 '06221503' --> 畫碩論圖6-16(pre-train)
 '06230850' --> 畫碩論圖6-16(no-pre-train)
 '07082329' --> 畫碩論圖6-17(no-yolo)
 """

# #,'06121418'
# PATH = '07082329'
# # PATH = ['06230850','06221503']
# data_name =['Data1','Data2']
# path_label = ['no_pre_train','pre_train']
# path_loc=['upper right','right']
#
# data_type = 'step'
# save_pic_name = '位隨機置'+data_type
# color=['slateblue','darkorange','lightseagreen']
A = [0.5,0.5]   #透明度
# # light_rgb = ([135 ,206 ,250],[	255, 231 ,186],[255 ,193 ,193])
# # deep_rgb = ([	65, 105 ,225],[255, 165, 0],[255, 64 ,64])
#
# light_rgb = ([171, 130, 255],[	255, 231 ,186],[255 ,193 ,193])
# deep_rgb = ([147, 112, 219],[	255, 165, 0],[255, 64 ,64])
#
light_rgb = ([	167 ,138, 230],[0, 0, 0])
deep_rgb = ([167 ,138, 230],[0, 0, 0])
#
# for i in range(len(PATH)):
#     f = open('./' + PATH[i] + '/train/data/' + data_type+'.txt', 'r')
#     data = (np.loadtxt(f))
#     smooth_data = []
#     for j in range(np.size(data)):
#         if j == 0:
#             smooth_data.append(data[j])
#         else:
#             smooth_data.append(smooth_data[-1]*0.9 + data[j]*0.1)
#     plt.plot(np.arange(data.shape[0]), data,color=(light_rgb[i][0]/255,light_rgb[i][1]/255,light_rgb[i][2]/255,A[i]),linewidth=1.5)
# # plt.legend(['pre_train','no_pre_train', '3', '4', '5'])
#
# for i in range(len(PATH)):
#     f = open('./' + PATH[i] + '/train/data/' + data_type+'.txt', 'r')
#     data = (np.loadtxt(f))
#     smooth_data = []
#     for j in range(np.size(data)):
#         if j == 0:
#             smooth_data.append(data[j])
#         else:
#             smooth_data.append(smooth_data[-1] * 0.9 + data[j] * 0.1)
#     data_name[i], = plt.plot(smooth_data,color=(deep_rgb[i][0]/255,deep_rgb[i][1]/255,deep_rgb[i][2]/255,0.9),label =path_label[i]  )
#     for j in range(data.shape[0]):
#         if data[j]<-2:
#             print('i',i,'j',j,'data',data[j])
#
#
# if data_type == 'reward':
#     # plt.title('固定位置訓練',fontname='DFKai-SB',fontsize = 20,fontweight = 'bold')
#     plt.title('Accumulative Reward',fontsize = 14,fontweight = 'bold')
#
#     # my_y_ticks = np.arange(-10, 1.5, 2)
#     # plt.yticks(my_y_ticks)
#     plt.legend(handles=[data_name[0], data_name[1]], loc=path_loc[1])
#     plt.ylabel('Reward',fontsize = 12)
#     plt.xlabel('Episode',fontsize = 12)
#
#
# elif data_type=='step':
#     # plt.title('固定位置訓練',fontname='DFKai-SB',fontsize = 20,fontweight = 'bold')
#     plt.title('Accumulative Grasp Times', fontsize=14, fontweight='bold')
#
#     plt.ylabel('Grasp Times', fontsize=12)
#     plt.xlabel('Episode', fontsize=12)
#     plt.legend(handles=[data_name[0], data_name[1]], loc=path_loc[0])
#     # my_y_ticks = np.arange(0, 20,1 )
#     # plt.xticks(my_x_ticks)
#     # plt.yticks([1,5,10,15,20,25])
#
# plt.xlim(0,1000)
# # plt.ylim(-0.5,1.6)
#
# #设置坐标轴刻度
# my_x_ticks = np.arange(0, 1100, 100)
# # my_y_ticks = np.arange(-5, 5, 0.5)
# plt.xticks(my_x_ticks)
# # plt.yticks(my_y_ticks)
#
# # plt.legend(handles=[data_name[0],data_name[1]], loc=path_loc[0])
# # plt.legend(['DEPTH','RGB', '3', '4', '5'],loc = 'lower right')
#
# plt.grid(True)
# # file_name = 'perfomance_compare_actor.png'
# file_name = save_pic_name+'.png'
# plt.savefig(file_name)
# plt.show()
# plt.clf()


PATH = ['07082329']
for i in range(len(PATH)):
    f = open('C:/Users/user/Desktop/rl/vrep/SAC_camera_version2/model/' + PATH[i] + '/train/data/' + 'step.txt', 'r')
    data = (np.loadtxt(f))
    smooth_data = []
    for j in range(np.size(data)):
        if j == 0:
            smooth_data.append(data[j])
        else:
            smooth_data.append(smooth_data[-1]*0.9 + data[j]*0.1)

    plt.plot(data,color=(light_rgb[0][0]/255,light_rgb[0][1]/255,light_rgb[0][2]/255,0.3),linewidth=1.5)
    plt.plot(smooth_data,color=(deep_rgb[0][0]/255,deep_rgb[0][1]/255,deep_rgb[0][2]/255,0.9))





plt.title('Accumulative Grasp Times',fontsize = 14,fontweight = 'bold')
plt.ylabel('Grasp Times')
plt.xlabel('Episode')
# plt.legend(['1', '2', '3', '4', '5'])
plt.grid(True)
# file_name = 'perfomance_compare_actor.png'
file_name = 'step____.png'
plt.savefig(file_name)
plt.show()
# plt.clf()