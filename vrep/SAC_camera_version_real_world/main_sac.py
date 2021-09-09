"""
    IF train_by_hand 為上機訓練(fail)
    IF Eval     為特定物件拾取
    IF CLEAN    為分類實驗

"""

from Eval_env import robot_Eval_Env,clean_table_Env
from hand_env import hand_robot_env
import numpy as np
from sac4 import SAC_agent
import config
import time
import os
import matplotlib.pyplot as plt
import math
import cv2
from yolo import *

from Kinect_new import Kinect
import pygame
import sys

MAX_EPISODES = config.MAX_EPISODES
MAX_EP_STEPS = config.MAX_EP_STEPS
eval_iteration = config.eval_iteration
PATH = time.strftime('%m%d%H%M')
ON_TRAIN = config.ON_TRAIN
initial_joint = config.initial_joint




# for eval
PATH_EVAL = config.PATH_EVAL
PATH_EVAL_Banana = config.PATH_EVAL_Banana
Hair_drier_PATH_EVAL = config.Hair_drier_PATH_EVAL




a_bound = config.a_bound
s_dim = config.state_dim
a_dim = config.action_dim
agent = SAC_agent(s_dim, a_dim, a_bound)

if ON_TRAIN:
    Mode = 'Train'
    record = False
else:
    Mode = 'Eval'
    record = True

print('a_bound : ', a_bound)
print('s_dim : ', s_dim, 'a_dim : ', a_dim)
print('Mode : ', Mode )



def train_by_hand():

    """
    [實驗一]
    此用於線上學習

    """
    hand_env = hand_robot_env()
    """ ---- kinect initial ---- """
    kinect = Kinect()
    """ ---- YOLO initial ---- """
    # yolo = YOLOV3()
    # yolo_coco = YOLOV3_coco()
    yolo_coco = YOLOV3_train_by_hand()
    """ ---- pre-train model ---- """
    agent.load_models(Hair_drier_PATH_EVAL)

    hand_env.initial_yolo_kinect(kinect, yolo_coco)
    if(config.vrep_show):
        hand_env.initial()

    total_numsteps = 0
    load_checkpoint = False

    """ ---- 得到彩色圖 ---- """
    color = kinect.Get_Color_Frame()
    color = color.copy()
    """ ---- 得到深度圖 ---- """

    # cv2.imshow('color', color)
    # cv2.waitKey(1)


    #------record memory------#
    step_set, reward_set, avg_reward_set, state_set ,action_set,distance_set, loss_q1_set,  loss_q2_set,loss_pi_set = [],[],[],[], [], [], [], [], []
    for i in range(MAX_EPISODES):
        want_boject = 0 #想要第幾個物體 (這裡沒用到)

        """ ---- 狀態輸入 整張彩色深度圖， YOLO偵測圖， 類別，類別標籤(數字)，BBOX ---- """
        s = hand_env.reset()
        done = False
        ep_r = 0
        for j in range(MAX_EP_STEPS):


            a = agent.choose_action(s,ON_TRAIN=True)

            action_set.append(a)

            s_, r, done= hand_env.step(a,record)


            # if (success >= 1):
            #     done = True
            #     print('lift done')
            # r = success

            if (j == 0 and done == True):
                r += 0.5
            if not done:
                r -= 0.05


            agent.remember(s, a ,r, s_, done)

            ep_r += r
            total_numsteps += 1
            if ep_r < -2:
                done = True

            if not load_checkpoint:
                loss_critic1, loss_critic2, loss_actor1 = agent.learn()
                loss_q1_set.append(loss_critic1)
                loss_q2_set.append(loss_critic2)
                loss_pi_set.append(loss_actor1)

            s = s_
            store_state = np.reshape(s,(config.image_input*config.image_input))
            state_set.append(store_state)


            if done or j == MAX_EP_STEPS - 1:
                print('episode: %i | %s | ep_r %.1f |step:%i' % (i, '...' if not done else 'done', ep_r, j + 1))
                step_set.append(j + 1)
                reward_set.append(ep_r)
                avg_reward_set.append(ep_r / (j + 1))
                break
        if i % eval_iteration == 0 :
            agent.save_models(PATH, i / eval_iteration)

    Kinect().close_kinect()

    print(total_numsteps)
    save_txt(path='./model/' + PATH + '/train/', name='Total_steps.txt', data=[total_numsteps], fmt='%f')
    # ----- record data ----- #
    folder_data = './model/' + PATH + '/train/data/'
    file_name = ['step.txt', 'reward.txt', 'avgR.txt','critic1_loss.txt','critic2_loss.txt','actor_loss.txt']
    data_set = ['step_set', 'reward_set','avg_reward_set','loss_q1_set','loss_q2_set','loss_pi_set']
    for i in range(len(file_name)):
        save_txt(path=folder_data, name=file_name[i], data=eval(data_set[i]))


    # ----- plot fig -----
    folder_fig = './model/' + PATH + '/train/fig/'
    xlabel = ['Episode', 'Episode', 'Episode','Episode', 'Episode', 'Episode']
    ylabel = ['step', 'Reward', 'avgR','loss','loss','loss']
    title = ['step', 'reward', 'avgR','cost_critic1','cost_critic2','cost_actor']
    for i in range(len(file_name)):
        plot_txt(path=folder_data, name=file_name[i], xlabel=xlabel[i], ylabel=ylabel[i], title=title[i],
                 save_location=folder_fig)

def Eval():
    """
    [實驗二]
    此用於針對特定物品拾取

    """
    Eval_Env = robot_Eval_Env()
    """ ---- kinect initial ---- """
    kinect = Kinect()
    """ ---- YOLO initial ---- """
    yolo = YOLOV3()
    yolo_coco = YOLOV3_coco()
    # yolo_coco = YOLOV3_open_image()
    Eval_Env.initial_yolo_kinect(kinect,yolo,yolo_coco)
    if (config.vrep_show):
        Eval_Env.initial()

    Yolo_Flag = 0  #0 : cuboid, 1 : coco data, 2 : open image data

    pygame.init()
    screen = pygame.display.set_mode((300, 300))
    screen.fill((255, 255, 255))
    pygame.display.set_caption("YOLO + SAC")
    font = pygame.font.SysFont("cambriacambriamath", 20)
    text = font.render('Press : ', True, (0, 0, 0))
    text1 = font.render('Q : Bye!', True, (0, 0, 0))
    text2 = font.render('1 : I want cubic ', True, (0, 0, 0))
    text3 = font.render('2 : I want banana ', True, (0, 0, 0))
    text4 = font.render('3 : I want apple ', True, (0, 0, 0))
    text5 = font.render('4 : I want orange ', True, (0, 0, 0))
    text6 = font.render('5 : I want cup ', True, (0, 0, 0))
    text7 = font.render('w : success', True, (0, 0, 0))
    text8 = font.render('e : fail', True, (0, 0, 0))


    while True:
        # cv2.waitKey(1)
        x,y,y2,t = 0,0,30,20
        screen.blit(text, (x, y))
        screen.blit(text1, (x, y2))
        screen.blit(text2, (x, y2 + t))
        screen.blit(text3, (x, y2 + 2 * t))
        screen.blit(text4, (x, y2 + 3 * t))
        screen.blit(text5, (x, y2 + 4 * t))
        screen.blit(text6, (x, y2 + 5 * t))
        screen.blit(text7, (x, y2 + 6 * t))
        screen.blit(text8, (x, y2 + 7 * t))
        pygame.display.update()

        """kinect獲取彩色照片"""

        color = kinect.Get_Color_Frame()
        color = color.copy()

        """"""
        ROI =  color[kinect.RoiOffset_Y:kinect.h_color-config.RoiOffset_Y_, kinect.RoiOffset_X:kinect.w_color-config.RoiOffset_X_]  # (880*1020*3)

        """ --- 顯示YOLO畫面 --- """
        # Yolo_Det_frame, _, _,_,_= yolo.detectFrame(ROI)  # 得到框的中心點
        Yolo_Det_frame, _, cls,_,_= yolo_coco.detectFrame(ROI)  # 得到框的中心點

        cv2.imwrite('C:\\Users\\user\\Desktop\\yolo_picture\\yolo_final_pic\\pic9.png',Yolo_Det_frame)
        cv2.imshow("Yolo_Det_frame",Yolo_Det_frame)
        cv2.waitKey(0)

        key_pressed = pygame.key.get_pressed()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    Eval_Env.close_kinect()
                    sys.exit()
                if event.key == pygame.K_1:
                    """ --- cubic --- """

                    """ SAC Model 下載 """
                    agent.load_models(PATH_EVAL)
                    want_boject = 0 # ---> 此為names檔裡的標籤數字
                    Yolo_Flag = 0  # 0 : cuboid, 1 : coco data,3 : open image data
                    s = Eval_Env.reset(want_boject,Yolo_Flag)
                    a=agent.choose_action(s,ON_TRAIN=True)
                    joint, flag = Eval_Env.step(a,record)
                    """ -------------------------------- record data  --------------------------------  """
                    save_data(joint, flag)
                    print("資料已傳送")
                    """ -------------------------------- record data  --------------------------------  """
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    # 按下任意鍵則關閉所有視窗

                    break
                elif event.key == pygame.K_2:
                    """ --- banana --- """

                    """ SAC Model 下載 """
                    agent.load_models(PATH_EVAL_Banana)
                    want_boject = 46
                    Yolo_Flag = 1
                    s = Eval_Env.reset(want_boject,Yolo_Flag)
                    a = agent.choose_action(s, ON_TRAIN=True)
                    joint, flag  = Eval_Env.step(a, record)
                    """ -------------------------------- record data  --------------------------------  """
                    save_data(joint, flag)
                    """ -------------------------------- record data  --------------------------------  """
                    print("資料已傳送")
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    # 按下任意鍵則關閉所有視窗

                elif event.key == pygame.K_3:
                    """ --- apple --- """

                    """ SAC Model 下載 """
                    agent.load_models(PATH_EVAL)
                    want_boject = 47
                    Yolo_Flag = 1
                    s = Eval_Env.reset(want_boject,Yolo_Flag)
                    a = agent.choose_action(s, ON_TRAIN=False)
                    joint, flag  = Eval_Env.step(a, record)
                    """ -------------------------------- record data  --------------------------------  """
                    save_data(joint, flag)
                    """ -------------------------------- record data  --------------------------------  """
                    print("資料已傳送")
                    # 按下任意鍵則關閉所有視窗
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    # break
                elif event.key == pygame.K_4:
                    """ --- orange --- """

                    """ SAC Model 下載 """
                    agent.load_models(PATH_EVAL)
                    want_boject = 49
                    Yolo_Flag = 1
                    s = Eval_Env.reset(want_boject,Yolo_Flag)
                    a = agent.choose_action(s, ON_TRAIN=True)
                    joint, flag = Eval_Env.step(a, record)
                    """ -------------------------------- record data  --------------------------------  """
                    save_data(joint, flag)
                    """ -------------------------------- record data  --------------------------------  """
                    print("資料已傳送")
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    # 按下任意鍵則關閉所有視窗
                    # break
                elif event.key == pygame.K_5:
                    """ --- cup --- """

                    """ SAC Model 下載 """
                    agent.load_models(PATH_EVAL)
                    want_boject = 41
                    Yolo_Flag = 1
                    s = Eval_Env.reset(want_boject,Yolo_Flag)
                    a = agent.choose_action(s, ON_TRAIN=True)
                    joint, flag  = Eval_Env.step(a, record)
                    """ -------------------------------- record data  --------------------------------  """
                    save_data(joint, flag)
                    """ -------------------------------- record data  --------------------------------  """
                    print("資料已傳送")
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    # 按下任意鍵則關閉所有視窗
                    # break

                elif event.key == pygame.K_6:
                    """ --- egg --- """

                    """ SAC Model 下載 """
                    agent.load_models(PATH_EVAL)
                    want_boject = 336
                    Yolo_Flag = 1
                    s = Eval_Env.reset(want_boject,Yolo_Flag)
                    a = agent.choose_action(s, ON_TRAIN=True)
                    joint, flag  = Eval_Env.step(a, record)
                    """ -------------------------------- record data  --------------------------------  """
                    save_data(joint, flag)
                    """ -------------------------------- record data  --------------------------------  """
                    print("資料已傳送")
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    # 按下任意鍵則關閉所有視窗
                    # break

                elif event.key == pygame.K_w:
                    success = 1
                    if (config.vrep_show):
                        Eval_Env.back_to_home()
                    break
                elif event.key == pygame.K_e:
                    success = 0
                    if (config.vrep_show):
                        Eval_Env.back_to_home()
                    print("fail")
                    continue


def clean_table():
    """
    [實驗三]
    此用於清理桌面上物體並分類

    """
    Eval_Env = clean_table_Env()
    """ ---- kinect initial ---- """
    kinect = Kinect()
    """ ---- YOLO initial ---- """
    yolo = YOLOV3()
    yolo_coco = YOLOV3_coco()
    # yolo_open_image = YOLOV3_open_image()  #這個先保留

    Eval_Env.initial_yolo_kinect(kinect, yolo, yolo_coco)

    if (config.vrep_show):
        Eval_Env.initial()

    Yolo_Flag = 0  # 0 : cuboid, 1 : coco data

    pygame.init()
    screen = pygame.display.set_mode((300, 300))
    screen.fill((255, 255, 255))
    pygame.display.set_caption("clean_table")
    font = pygame.font.SysFont("cambriacambriamath", 20)
    text = font.render('Press : ', True, (0, 0, 0))
    text1 = font.render('Q : Bye!', True, (0, 0, 0))

    while True:
        x, y, y2, t = 0, 0, 30, 20
        screen.blit(text, (x, y))
        screen.blit(text1, (x, y2))
        pygame.display.update()

        color = kinect.Get_Color_Frame()
        mImg16bit, DepthImg = kinect.Get_Depth_Frame()
        color = color.copy()
        ROI = color[kinect.RoiOffset_Y:kinect.h_color - config.RoiOffset_Y_,
              kinect.RoiOffset_X:kinect.w_color - config.RoiOffset_X_]  # (880*1020*3)
        cv2.imwrite('C:\\Users\\user\\Desktop\\yolo_picture\\exp3\\' + str(PATH)+'.png',ROI)  # 存图片

        # cv2.imshow('roi',ROI)
        # cv2.waitKey(0)

        Yolo_Det_frame, coordinate_cubic,cls_cubic,label_cubic,Width_and_Height_cubic= yolo.detectFrame(ROI)
        if (coordinate_cubic[0][0] == 0 ):
            grasp_index = 0
        else:
            grasp_index = len(cls_cubic)

        Yolo_Det_frame, coordinate_coco,cls_coco,label_coco,Width_and_Height_coco= yolo_coco.detectFrame(ROI,grasp_index)


        if coordinate_cubic[0][0] == 0  :
            coordinate = coordinate_coco
            cls = cls_coco
            label = label_coco
            Width_and_Height = Width_and_Height_coco

        elif coordinate_coco[0][0] == 0:
            coordinate = coordinate_cubic
            cls = cls_cubic
            label = label_cubic
            Width_and_Height = Width_and_Height_cubic

        else:
            coordinate = np.concatenate([coordinate_cubic,coordinate_coco])
            cls = np.concatenate([cls_cubic, cls_coco])
            label = np.concatenate([label_cubic, label_coco])
            Width_and_Height = np.concatenate([Width_and_Height_cubic, Width_and_Height_coco])

        cv2.imshow('Yolo_Det_frame', Yolo_Det_frame)
        cv2.waitKey(0)
        # joint=np.zeros((len(cls),6),np.float)
        # flag = np.zeros((len(cls), 1), np.int)


        joint_sol = []
        Object_Num = []
        Object_class = [] #將物品分為是水果或不是水果


        for i in range (len(cls)): # yolo 偵測出的物體數目
            want_boject = i

            # --------看是不是香蕉決定要哪個 pre-train weight--------#
            if cls[want_boject] == 46:
                agent.load_models(PATH_EVAL_Banana)
            else:
                agent.load_models(PATH_EVAL)
            # --------看是不是香蕉決定要哪個 pre-train weight--------#

            s = Eval_Env.reset(want_boject, color,mImg16bit, DepthImg,Yolo_Det_frame,coordinate,cls,label,Width_and_Height)
            a = agent.choose_action(s, ON_TRAIN=True)
            joint, flag = Eval_Env.step(a)
            print(joint)
            if (flag==0):
                joint_sol.append(joint)
                Object_Num.append(flag)
                if cls[want_boject] == 0 or cls[want_boject] == 41 :
                    classification = 1 #cubic類
                    Object_class.append(classification)
                else:
                    classification = 2  #疏果類
                    Object_class.append(classification)
            else:
                print("第",i+1,'個物體')
                print('------------')

        Object_Num_Save = int(len(Object_Num))
        folder_data = 'C:/Users/user/Desktop/碩論/Robot_File/'
        file_name = ['JointCmd_Clean.txt','Object_Num.txt','Object_classification.txt']
        data_set = ['joint_sol','[Object_Num_Save]','Object_class']

        for i in range(len(file_name)):
            save_txt(path=folder_data, name=file_name[i], data=eval(data_set[i]))
        print('資料已傳送')
        # cv2.imshow("Yolo_Det_frame", Yolo_Det_frame)
        # cv2.waitKey(1)

        key_pressed = pygame.key.get_pressed()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    Eval_Env.close_kinect()
                    sys.exit()

def save_data( joint, flag):
    folder_data = 'C:/Users/user/Desktop/碩論/Robot_File/'
    joint = np.reshape(joint,(1,6))
    file_name = ['JointCmd.txt', 'flag.txt']
    data_set = ['joint', '[flag]']
    for i in range(len(file_name)):
        save_txt(path=folder_data, name=file_name[i], data=eval(data_set[i]))

def plot(data, xlabel, ylabel, title, save_location):
    plt.plot(np.arange(data.shape[0]), data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
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

def load_txt(path, name):
    f = open(path + name, 'r')
    data = np.loadtxt(f)
    f.close()
    return data



def main_by_hand():
    """
    上機用程式
    IF ON_TEAIN 為實驗一
    IF Eval     為實驗二
    IF CLEAN    為實驗三
    """
    if ON_TRAIN:
        creat_path('./model/' + PATH + '/train/data')
        creat_path('./model/' + PATH + '/train/fig')
        creat_path('./model/' + PATH + '/test')
        save_parameter()
        start_time = time.time()
        train_by_hand()
        end_time = time.time()
        cost_time = np.array([[end_time - start_time]])
        save_txt(path='./model/' + PATH + '/train/', name='time.txt', data=cost_time, fmt='%f')

    elif config.clean:
        clean_table()
    else:
        Eval()


def creat_path(path):
    if path_exsit(path=path):
        print(path+' exist')
    else:
        os.makedirs(path)

def path_exsit(path):
    if os.path.exists(path):
        return True
    else:
        return False
def save_txt(path, name, data, fmt='%f'):
    f = open(path + name, 'w')
    np.savetxt(f, data, fmt=fmt)
    f.close()

def save_parameter():
    with open('./model/' + PATH + '/train/parameter.txt', 'w') as f:
        f.writelines("Method: {}\n".format('sac'))
        f.writelines("state: {}\naction: {}\n a_bound:{}\n".format(config.state_dim,config.action_dim,config.a_bound))
        f.writelines("Max Episodes: {}\nMax Episodes steps: {}\n".format(MAX_EPISODES, MAX_EP_STEPS))
        f.writelines("LR_A: {}\nLR_C: {}\nGAMMA: {}\n".format(config.A_LR, config.C_LR, config.gamma))
        f.writelines("reward_scale: {}\ntau: {}\nmemory_capacity: {}\n".format(config.reward_scale,config.tau,config.MEMORY_CAPACITY))
        f.writelines("BATCH_SIZE: {}\nhidden_dim: {}\n".format(config.batch_size,config.hidden_dim))


if __name__ == "__main__":
    main_by_hand()



