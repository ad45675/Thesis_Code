"""
YOLOV3 : cubic

YOLOV3_coco : coco dataset

輸入 : 416*416 圖片

輸出 :
frame : 偵測後的圖片
coordinate : 感興趣物體座標
cls : 感興趣物體類別(數字)
label : 感興趣物體類別(文字)
Width_and_Height : 感興趣物體編階框長寬

By : ya0000000
2021/08/31

"""

from __future__ import division
import os
import cv2
import torch
import random
import argparse
import numpy as np
import pickle as pkl
import pandas as pd
import os.path as osp
import torch.nn as nn
from darknetUtils import *
from darknet import Darknet
from torch.autograd import Variable
import colorsys

class YOLOV3:


    """ only cuboid  """
    images = "imgTemp/frame.jpg"
    cfgfile = "C:\\Users\\user\\Desktop\\VSCLab\\碩論\\YOLOv3\\Yolo_weight\\only_cubic\\training_set.cfg"
    namefile = "C:\\Users\\user\\Desktop\\VSCLab\\碩論\\YOLOv3\\Yolo_weight\\only_cubic\\training_obj.names"
    weightsfile = "C:\\Users\\user\\Desktop\\VSCLab\\碩論\\YOLOv3\\Yolo_weight\\only_cubic\\training_set_20000.weights"
    colorflie = "C:\\Users\\user\\Desktop\\VSCLab\\碩論\\YOLOv3\\pallete"

    reso = int(416)
    num_classes = int(1)
    batch_size = int(1)
    confidence = float(0.1)
    nms_thesh = float(0.1)
    CUDA = torch.cuda.is_available()

    def __init__(self):
        namefile = self.namefile
        cfgfile = self.cfgfile
        weightsfile = self.weightsfile
        colorflie = self.colorflie
        CUDA = self.CUDA

        classes = load_classes(namefile)
        colors = pkl.load(open(colorflie, "rb"))
        # print(classes)

        # Set up the neural network
        print("Loading network.....")
        model = Darknet(cfgfile)
        model.load_weights(weightsfile)
        print("Network successfully loaded")

        model.net_info["height"] = 416
        inp_dim = int(model.net_info["height"])
        assert inp_dim % 32 == 0
        assert inp_dim > 32

        # If there's a GPU availible, put the model on GPU
        if CUDA:
            model.cuda()

        # Set the model in evaluation mode
        model.eval()

        self.inp_dim = inp_dim
        self.classes = classes
        self.colors = colors
        self.model = model

        print("init finish")

    def __del__(self):
        print("network end!")

    def write(self, x, results):
        # colors = self.colors
        classes = self.classes
        colorflie = self.colorflie

        c1 = tuple(x[1:3].int())
        c2 = tuple(x[3:5].int())

        img = results

        cls = int(x[-1])
        # color = random.choice(colors)
        # color = pkl.load(open(colorflie, "rb"))[2]
        # color = __get_colors(classnum)
        label = "{0}".format(classes[cls])

        # 不同的框，不同的颜色
        hsv_tuples = [(float(x) / 3, 1., 1.)
                      for x in range(3)]  # 不同颜色
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))  # RGB
        np.random.seed(10101)
        np.random.shuffle(colors)
        np.random.seed(None)

        if label == 'ball' :
            color = colors[0]
        elif label == 'cube':
            color = colors[1]
        else:
            color = colors[2]

        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        cv2.rectangle(img, c1, c2, color, 1)

        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2, color, -1)
        # print('c1','\n',c1,'c2',':',c2,'\n','t_size',t_size)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)

        """ --- 寫分數 --- """
        # cv2.putText(img, str(x[5]), (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)

        # 画中点
        # center = (int((c1[0]+c2[0])/2), int((c1[1]+c2[1])/2))
        # # 画两点
        # cv2.circle(img, center, 2, (0, 0, 255), -1)
        # print('cen',center)
        # cv2.imshow('img',img)
        # cv2.waitKey(0)
        # cv2.circle(img, (320, 240), 4, color, -1)
        # # 画图像中心点到目标点的直线
        # cv2.line(img, center, (320, 240), color)

        return img

    """
    输出的格式：(ind,x1,y1,x2,y2,s,s_cls,index_cls)
    ind是方框所属图片在这个batch中的序号
    x1,y1是在网络输入图片坐标系中，方框左上角的坐标
    x2,y2是方框右下角的坐标
    s是这个方框含有目标的得分
    s_cls是这个方框中所含目标最有可能的类别的概率得分
    index_cls是s_cls对应的这个类别在所有类别中所对应的序号
    """

    def detectFrame(self, frame):
        reso = self.reso
        inp_dim = self.inp_dim
        confidence = self.confidence
        num_classes = self.num_classes
        nms_thesh = self.nms_thesh
        # print('thre',self.nms_thesh)
        CUDA = self.CUDA
        model = self.model
        classes = self.classes

        img = prep_image(frame, inp_dim)
        im_dim = frame.shape[1], frame.shape[0]

        im_dim = torch.FloatTensor(im_dim).repeat(1, 2)
        output_ =[]
        if CUDA:
            im_dim = im_dim.cuda()
            img = img.cuda()

        with torch.no_grad():
            output = model(Variable(img), CUDA)
        output = write_results(output, confidence, num_classes, nms_conf=nms_thesh)

        label = []
        # type(output) != int 表示检测到了目标, 此时对目标画框后输出, 否则输出原图

        if type(output) != int:
            coordinate = np.zeros((output.shape[0], 2), np.int)  # shape (5,2)
            Width_and_Height = np.zeros((output.shape[0], 2), np.int)  # shape (5,2)
            cls = np.zeros((output.shape[0]), np.int)

            im_dim = im_dim.repeat(output.size(0), 1)
            scaling_factor = torch.min(int(reso) / im_dim, 1)[0].view(-1, 1)
            output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim[:, 0].view(-1, 1)) / 2
            output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim[:, 1].view(-1, 1)) / 2
            output[:, 1:5] /= scaling_factor

            for i in range(output.shape[0]):
                output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim[i, 0])
                output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim[i, 1])
                classnum = output[i, 2]
                # coordinate
                coordinate[i] = np.array([int((output[i][1] + output[i][3]) / 2), int((output[i][2] + output[i][4]) / 2)])
                cls[i] = int(output[i][-1])

                label.append("{0}".format(classes[cls[i]]))
                Width_and_Height[i] = np.array([abs(int((output[i][1] - output[i][3]))), abs(int((output[i][2] - output[i][4])))])
            list(map(lambda x: self.write(x, frame), output))

            # cv2.imshow("frame", frame)
            # cv2.waitKey(0)
        else:
            coordinate = np.array([[0,0]])
            Width_and_Height = np.array([[0, 0]])
            cls = np.array([0])


        return frame, coordinate,cls,label,Width_and_Height

class YOLOV3_coco:


    # """ 人家train好ㄉcoco dataset """
    images = "imgTemp/frame.jpg"
    cfgfile = "C:\\Users\\user\\Desktop\\rl\\vrep\\SAC_camera_version_real_world\\yolo_model\\yolov3.cfg"
    namefile = "C:\\Users\\user\\Desktop\\rl\\vrep\\SAC_camera_version_real_world\\yolo_model\\coco.names"
    weightsfile = "C:\\Users\\user\\Desktop\\rl\\vrep\\SAC_camera_version_real_world\\yolo_model\\yolov3.weights"
    colorflie = "C:\\Users\\user\\Desktop\\VSCLab\\碩論\\YOLOv3\\pallete"

    reso = int(416)
    num_classes = int(80)
    batch_size = int(1)
    confidence = float(0.1)
    nms_thesh = float(0.1)
    CUDA = torch.cuda.is_available()

    def __init__(self):
        namefile = self.namefile
        cfgfile = self.cfgfile
        weightsfile = self.weightsfile
        colorflie = self.colorflie
        CUDA = self.CUDA

        classes = load_classes(namefile)
        colors = pkl.load(open(colorflie, "rb"))
        # print(classes)

        # Set up the neural network
        print("Loading network.....")
        model = Darknet(cfgfile)
        model.load_weights(weightsfile)
        print("Network successfully loaded")

        model.net_info["height"] = 416
        inp_dim = int(model.net_info["height"])
        assert inp_dim % 32 == 0
        assert inp_dim > 32

        # If there's a GPU availible, put the model on GPU
        if CUDA:
            model.cuda()

        # Set the model in evaluation mode
        model.eval()

        self.inp_dim = inp_dim
        self.classes = classes
        self.colors = colors
        self.model = model

        print("init finish")

    def __del__(self):
        print("network end!")

    def write(self, x, results):
        # colors = self.colors
        classes = self.classes
        colorflie = self.colorflie

        c1 = tuple(x[1:3].int())
        c2 = tuple(x[3:5].int())

        img = results

        cls = int(x[-1])
        # color = random.choice(colors)
        # color = pkl.load(open(colorflie, "rb"))[2]
        # color = __get_colors(classnum)
        label = "{0}".format(classes[cls])

        # 不同的框，不同的颜色
        hsv_tuples = [(float(x) / 3, 1., 1.)
                      for x in range(3)]  # 不同颜色
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))  # RGB
        np.random.seed(10101)
        np.random.shuffle(colors)
        np.random.seed(None)

        if label == 'ball' :
            color = colors[0]
        elif label == 'cube':
            color = colors[1]
        else:
            color = colors[2]

        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        cv2.rectangle(img, c1, c2, color, 1)

        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2, color, -1)
        # print('c1','\n',c1,'c2',':',c2,'\n','t_size',t_size)
        label = 'banana'

        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)

        """ --- 寫分數 --- """
        # cv2.putText(img, str(x[5]), (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)


        return img

    """
    输出的格式：(ind,x1,y1,x2,y2,s,s_cls,index_cls)
    ind是方框所属图片在这个batch中的序号
    x1,y1是在网络输入图片坐标系中，方框左上角的坐标
    x2,y2是方框右下角的坐标
    s是这个方框含有目标的得分
    s_cls是这个方框中所含目标最有可能的类别的概率得分
    index_cls是s_cls对应的这个类别在所有类别中所对应的序号
    """

    def detectFrame(self, frame):
        reso = self.reso
        inp_dim = self.inp_dim
        confidence = self.confidence
        num_classes = self.num_classes
        nms_thesh = self.nms_thesh
        # print('thre',self.nms_thesh)
        CUDA = self.CUDA
        model = self.model
        classes = self.classes

        img = prep_image(frame, inp_dim)
        im_dim = frame.shape[1], frame.shape[0]
        im_dim = torch.FloatTensor(im_dim).repeat(1, 2)
        output_ =[]
        if CUDA:
            im_dim = im_dim.cuda()
            img = img.cuda()

        with torch.no_grad():
            output = model(Variable(img), CUDA)
        output = write_results(output, confidence, num_classes, nms_conf=nms_thesh)

        # for i in range(output.shape[0]):
        #     if output[i][5] > 0.7:
        #         output_.append(output)
        #         output_ = torch.tensor([output_.cpu().detach().numpy() for output_ in output_]).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        # print('**',output.shape)
        # for i in range(output.shape[0]):
        #     print('__',i,':',output_[i])

        # if (output == 0):
        #     coordinate = np.zeros((1,2),np.int) #shape (5,2)
        #     cls = np.zeros((1),np.int)
        # else:
        #     coordinate = np.zeros((output.shape[0], 2), np.int)  # shape (5,2)
        #     cls = np.zeros((output.shape[0]), np.int)

        label = []
        # type(output) != int 表示检测到了目标, 此时对目标画框后输出, 否则输出原图

        if type(output) != int:
            coordinate = np.zeros((output.shape[0], 2), np.int)  # shape (5,2)
            Width_and_Height = np.zeros((output.shape[0], 2), np.int)  # shape (5,2)
            cls = np.zeros((output.shape[0]), np.int)

            im_dim = im_dim.repeat(output.size(0), 1)
            scaling_factor = torch.min(int(reso) / im_dim, 1)[0].view(-1, 1)
            output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim[:, 0].view(-1, 1)) / 2
            output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim[:, 1].view(-1, 1)) / 2
            output[:, 1:5] /= scaling_factor

            for i in range(output.shape[0]):
                output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim[i, 0])
                output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim[i, 1])
                classnum = output[i, 2]
                # coordinate
                coordinate[i] = np.array([int((output[i][1] + output[i][3]) / 2), int((output[i][2] + output[i][4]) / 2)])
                cls[i] = int(output[i][-1])

                label.append("{0}".format(classes[cls[i]]))
                Width_and_Height[i] = np.array([abs(int((output[i][1] - output[i][3]))), abs(int((output[i][2] - output[i][4])))])
                # print(output[i][5])
                # print('-----------')
            list(map(lambda x: self.write(x, frame), output))

            # cv2.imshow("frame", frame)
            # cv2.waitKey(0)


        else:
            coordinate = np.array([[0,0]])
            Width_and_Height = np.array([[0, 0]])
            cls = np.array([0])


        return frame, coordinate,cls,label,Width_and_Height





