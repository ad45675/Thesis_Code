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
import config




grasp_index = 0

class YOLOV3:
    """ cuboid  """
    images = "imgTemp/frame.jpg"
    cfgfile = "C:\\Users\\user\\Desktop\\碩論\\YOLOv3\\Yolo_weight\\only_cubic\\training_set.cfg"
    namefile = "C:\\Users\\user\\Desktop\\碩論\\YOLOv3\\Yolo_weight\\only_cubic\\training_obj.names"
    weightsfile = "C:\\Users\\user\\Desktop\\碩論\\YOLOv3\\Yolo_weight\\only_cubic\\training_set_20000.weights"
    colorflie = "C:\\Users\\user\\Desktop\\碩論\\YOLOv3\\pallete"

    reso = int(416)
    num_classes = int(1)
    batch_size = int(1)
    confidence = float(0.5)
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

        if label == 'mouse':
            color = colors[0]
        elif label == 'scissors':
            color = colors[1]
        elif label == 'keyboard':
            color = colors[2]
        else:
            color = colors[0]


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
        # cv2.putText(img, str(sort_index), (x[1], x[4]), cv2.FONT_HERSHEY_PLAIN, 1, [0, 255, 255], 1)

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
        # self.Yolo_Flag = Yolo_Flag
        reso = self.reso
        inp_dim = self.inp_dim
        confidence = self.confidence
        num_classes = self.num_classes
        nms_thesh = self.nms_thesh
        CUDA = self.CUDA
        model = self.model
        classes = self.classes

        img = prep_image(frame, inp_dim)
        im_dim = frame.shape[1], frame.shape[0]
        im_dim = torch.FloatTensor(im_dim).repeat(1, 2)

        if CUDA:
            im_dim = im_dim.cuda()
            img = img.cuda()

        with torch.no_grad():
            output = model(Variable(img), CUDA)
        output = write_results(output, confidence, num_classes, nms_conf=nms_thesh)

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

                cv2.putText(frame, str(i+1), (output[i][1], output[i][4]), cv2.FONT_HERSHEY_PLAIN, 1, [0, 255, 255], 1)
                # cv2.circle(frame, ( coordinate[i][0], coordinate[i][1]), 2, (0, 0, 0), -1)

            list(map(lambda x: self.write(x, frame), output))
            grasp_index = output.shape[0] + 1

        else:
            coordinate = np.array([[0,0]])
            Width_and_Height = np.array([[0, 0]])
            cls = np.array([0])

        return frame, coordinate,cls,label,Width_and_Height


class YOLOV3_coco:
    # 参数

    """ coco """
    images = "imgTemp/frame.jpg"
    cfgfile = "C:\\Users\\user\\Desktop\\rl\\vrep\\SAC_camera_version_real_world\\yolo_model\\yolov3.cfg"
    namefile = "C:\\Users\\user\\Desktop\\rl\\vrep\\SAC_camera_version_real_world\\yolo_model\\coco.names"
    weightsfile = "C:\\Users\\user\\Desktop\\rl\\vrep\\SAC_camera_version_real_world\\yolo_model\\yolov3.weights"
    colorflie = "C:\\Users\\user\\Desktop\\碩論\\YOLOv3\\pallete"

    reso = int(416)
    num_classes = int(80)
    batch_size = int(1)
    confidence = float(0.01)
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

        # Set up the neural network
        print("Loading coco network.....")
        model = Darknet(cfgfile)
        model.load_weights(weightsfile)
        print("coco Network successfully loaded")

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
        self.apple =0
        print("coco init finish")

    def __del__(self):
        print("coco network end!")

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

        if label == 'mouse':
            color = colors[0]
        elif label == 'scissors':
            color = colors[1]
        elif label == 'keyboard':
            color = colors[2]
        else:
            color = colors[0]

        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        cv2.rectangle(img, c1, c2, color, 1)

        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2, color, -1)
        # print('c1','\n',c1,'c2',':',c2,'\n','t_size',t_size)

        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
        """ --- 寫分數 --- """
        # cv2.putText(img, str(x[5]), (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 0, 0], 1)



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

    def detectFrame(self, frame,grasp_index=0):

        reso = self.reso
        inp_dim = self.inp_dim
        confidence = self.confidence
        num_classes = self.num_classes
        nms_thesh = self.nms_thesh
        CUDA = self.CUDA
        model = self.model
        classes = self.classes

        img = prep_image(frame, inp_dim)
        im_dim = frame.shape[1], frame.shape[0]
        im_dim = torch.FloatTensor(im_dim).repeat(1, 2)


        if CUDA:
            im_dim = im_dim.cuda()
            img = img.cuda()

        with torch.no_grad():
            output_ = model(Variable(img), CUDA)

        output_ = write_results(output_, confidence, num_classes, nms_conf=nms_thesh)


# #------------------------只顯示我要的------------------------#
        object_num = 0
        a = []
        for i in range(output_.shape[0]):
            cls = int(output_[i][-1])
            if cls == 76: #scissors
                object_num = object_num + 1
                a.append(output_[i])
            elif cls == 47: #apple
                object_num = object_num + 1
                a.append(output_[i])
            elif cls == 49: #orange
                object_num = object_num + 1
                a.append(output_[i])
            elif cls == 46: #banana
                object_num = object_num + 1
                a.append(output_[i])
            elif cls == 41:  # cup
                object_num = object_num + 1
                a.append(output_[i])
            elif cls == 78:  # 吹風機
                object_num = object_num + 1
                a.append(output_[i])


        output = torch.FloatTensor(object_num ,8)
        for i in range(object_num):
            output[i] = a[i]
#
# #------------------------只顯示我要的------------------------#


        # if (output == 0):
        #     coordinate = np.zeros((1,2),np.int) #shape (5,2)
        #     cls = np.zeros((1),np.int)
        # else:
        #     coordinate = np.zeros((output.shape[0], 2), np.int)  # shape (5,2)
        #     cls = np.zeros((output.shape[0]), np.int)

        label = []

        if object_num != 0: #表示检测到了目标, 此时对目标画框后输出, 否则输出原图

            coordinate = np.zeros((output.shape[0], 2), np.int)  # shape (5,2)
            Width_and_Height = np.zeros((output.shape[0], 2), np.int)  # shape (5,2)
            cls = np.zeros((output.shape[0]), np.int)


            im_dim = im_dim.repeat(output.size(0), 1)
            scaling_factor = torch.min(int(reso) / im_dim, 1)[0].view(-1, 1)
            output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim[:, 0].view(-1, 1)) / 2
            output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim[:, 1].view(-1, 1)) / 2
            output[:, 1:5] /= scaling_factor


            for i in range(output.shape[0]):
                cls[i] = int(output[i][-1])

                output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim[i, 0])
                output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim[i, 1])
                classnum = output[i, 2]

                # coordinate
                coordinate[i] = np.array(
                    [int((output[i][1] + output[i][3]) / 2), int((output[i][2] + output[i][4]) / 2)])
                # cls[i] = int(output[i][-1])
                label.append("{0}".format(classes[cls[i]]))
                Width_and_Height[i] = np.array(
                    [abs(int((output[i][1] - output[i][3]))), abs(int((output[i][2] - output[i][4])))])



                grasp_index = grasp_index + 1

                cv2.putText(frame, str(grasp_index), (output[i][1], output[i][4]), cv2.FONT_HERSHEY_PLAIN, 1, [0, 255, 255], 1)

            list(map(lambda x: self.write(x, frame), output))


            # cv2.imshow("frame", frame)
        else:
            coordinate = np.array([[0, 0]])
            Width_and_Height = np.array([[0, 0]])
            cls = np.array([0])

        return frame, coordinate, cls, label, Width_and_Height

class YOLOV3_open_image:


    """ openimage """
    images = "imgTemp/frame.jpg"
    cfgfile = "C:\\Users\\user\\Desktop\\碩論\\YOLOv3\\Yolo_weight\\open_image_data\\yolov3-openimages.cfg"
    namefile = "C:\\Users\\user\\Desktop\\碩論\\YOLOv3\\Yolo_weight\\open_image_data\\openimages.names"
    weightsfile = "C:\\Users\\user\\Desktop\\碩論\\YOLOv3\\Yolo_weight\\open_image_data\\\yolov3-openimages.weights"
    colorflie = "C:\\Users\\user\\Desktop\\碩論\\YOLOv3\\pallete"


    reso = int(416)
    num_classes = int(601)
    batch_size = int(1)
    confidence = float(0.01)
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

        if label == 'mouse':
            color = colors[0]
        elif label == 'scissors':
            color = colors[1]
        elif label == 'keyboard':
            color = colors[2]
        else:
            color = colors[0]


        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        cv2.rectangle(img, c1, c2, color, 1)

        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2, color, -1)
        # print('c1','\n',c1,'c2',':',c2,'\n','t_size',t_size)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)

        """ --- 寫分數 --- """
        cv2.putText(img, str(x[5]), (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
        # cv2.putText(img, str(sort_index), (x[1], x[4]), cv2.FONT_HERSHEY_PLAIN, 1, [0, 255, 255], 1)


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
        # self.Yolo_Flag = Yolo_Flag
        reso = self.reso
        inp_dim = self.inp_dim
        confidence = self.confidence
        num_classes = self.num_classes
        nms_thesh = self.nms_thesh
        CUDA = self.CUDA
        model = self.model
        classes = self.classes

        img = prep_image(frame, inp_dim)
        im_dim = frame.shape[1], frame.shape[0]
        im_dim = torch.FloatTensor(im_dim).repeat(1, 2)

        if CUDA:
            im_dim = im_dim.cuda()
            img = img.cuda()

        with torch.no_grad():
            output = model(Variable(img), CUDA)
        output = write_results(output, confidence, num_classes, nms_conf=nms_thesh)

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

                cv2.putText(frame, str(i+1), (output[i][1], output[i][4]), cv2.FONT_HERSHEY_PLAIN, 1, [0, 255, 255], 1)
                # cv2.circle(frame, ( coordinate[i][0], coordinate[i][1]), 2, (0, 0, 0), -1)

            list(map(lambda x: self.write(x, frame), output))
            grasp_index = output.shape[0] + 1

            # cv2.imshow("frame", frame)

        else:
            coordinate = np.array([[0,0]])
            Width_and_Height = np.array([[0, 0]])
            cls = np.array([0])

        return frame, coordinate,cls,label,Width_and_Height

class YOLOV3_train_by_hand:

    #此用來辨識吹風機

    #
    # """ openimage """
    images = "imgTemp/frame.jpg"
    cfgfile = "C:\\Users\\user\\Desktop\\碩論\\YOLOv3\\Yolo_weight\\open_image_data\\yolov3-openimages.cfg"
    namefile = "C:\\Users\\user\\Desktop\\碩論\\YOLOv3\\Yolo_weight\\open_image_data\\openimages.names"
    weightsfile = "C:\\Users\\user\\Desktop\\碩論\\YOLOv3\\Yolo_weight\\open_image_data\\\yolov3-openimages.weights"
    colorflie = "C:\\Users\\user\\Desktop\\碩論\\YOLOv3\\pallete"

    reso = int(416)
    num_classes = int(601)
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

        if label == 'mouse':
            color = colors[0]
        elif label == 'scissors':
            color = colors[1]
        elif label == 'keyboard':
            color = colors[2]
        else:
            color = colors[0]


        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        cv2.rectangle(img, c1, c2, color, 1)

        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        # cv2.rectangle(img, c1, c2, color, -1)
        # print('c1','\n',c1,'c2',':',c2,'\n','t_size',t_size)
        # label = "{0}".format(classes[263])
        """ --- 寫類別 --- """
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [225, 25, 255], 1)

        """ --- 寫分數 --- """
        # cv2.putText(img, str(x[5]), (c1[0], c1[1]  ), cv2.FONT_HERSHEY_PLAIN, 1, [0, 10, 255], 1)
        # cv2.putText(img, str(sort_index), (x[1], x[4]), cv2.FONT_HERSHEY_PLAIN, 1, [0, 255, 255], 1)


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
        # self.Yolo_Flag = Yolo_Flag
        reso = self.reso
        inp_dim = self.inp_dim
        confidence = self.confidence
        num_classes = self.num_classes
        nms_thesh = self.nms_thesh
        CUDA = self.CUDA
        model = self.model
        classes = self.classes

        img = prep_image(frame, inp_dim)
        im_dim = frame.shape[1], frame.shape[0]
        im_dim = torch.FloatTensor(im_dim).repeat(1, 2)

        if CUDA:
            im_dim = im_dim.cuda()
            img = img.cuda()

        with torch.no_grad():
            output = model(Variable(img), CUDA)
        output = write_results(output, confidence, num_classes, nms_conf=nms_thesh)

        # if (output == 0):
        #     coordinate = np.zeros((1,2),np.int) #shape (5,2)
        #     cls = np.zeros((1),np.int)
        # else:
        #     coordinate = np.zeros((output.shape[0], 2), np.int)  # shape (5,2)
        #     cls = np.zeros((output.shape[0]), np.int)
        coco = False
        if coco:
            object_num = 0
            a = []
            for i in range(output.shape[0]):
                cls = int(output[i][-1])
                if cls == 76 :  # scissors
                    object_num = object_num + 1
                    a.append(output[i])

            output = torch.FloatTensor(object_num, 8)
            for i in range(object_num):
                output[i] = a[i]

        else:
            object_num = 0
            a = []
            for i in range(output.shape[0]):
                cls = int(output[i][-1])
                #if cls != 465  and  cls != 403 and cls != 407 and cls != 300:  # Furniture ,Vehicle  ,Weapon ,Musical instrument
                object_num = object_num + 1
                a.append(output[i])


            output = torch.FloatTensor(object_num, 8)
            for i in range(object_num):
                output[i] = a[i]



        label = []
        # type(output) != int 表示检测到了目标, 此时对目标画框后输出, 否则输出原图
        # if type(output) != int:
        if object_num != 0:
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


                cv2.putText(frame, str(i+1), (output[i][1], output[i][4]), cv2.FONT_HERSHEY_PLAIN, 1, [0, 255, 255], 1)
                # cv2.circle(frame, ( coordinate[i][0], coordinate[i][1]), 2, (0, 0, 0), -1)

            list(map(lambda x: self.write(x, frame), output))
            grasp_index = output.shape[0] + 1

            # cv2.imshow("frame", frame)



        else:
            coordinate = np.array([[0,0]])
            Width_and_Height = np.array([[0, 0]])
            cls = np.array([0])

        return frame, coordinate,cls,label,Width_and_Height

if __name__ == '__main__':
    yolo = YOLOV3()
    yolo_coco = YOLOV3_coco()
    img = cv2.imread("C:\\Users\\user\\Desktop\\mango.jpg")  # 获取图片
    # print(img)
    cv2.imshow("ori",img)
    # cv2.waitKey(0)

    frame, coordinate,cls,label,_ = yolo.detectFrame(img)  # 检测
    # frame, coordinate, _, _, _ = yolo_coco.detectFrame(frame)  # 检测
    print('coordinate',label)

    color = (0, 0, 255)  # BGR
    # center = (coordinate[0],coordinate[1])
    # cv2.circle(frame, (coordinate[0],coordinate[1]), 2, color, -1)
    # print(coordinate)
    cv2.imshow("yolo",frame)

    # cv2.imwrite('C:\\Users\\user\\Desktop\\yolo_picture\\YOLOv3.png',frame)  # 存图片
    # img2 = cv2.imread("C:\\Users\\user\\Desktop\\碩論\\YOLOv3.jpg",1)  # 获取图片
    # cv2.imshow("yolo2...", img2)
    cv2.waitKey(0)