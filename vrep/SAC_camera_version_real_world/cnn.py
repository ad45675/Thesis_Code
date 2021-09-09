import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn

#################################
#Torch.nn.Conv2d(in_channels，out_channels，kernel_size，stride=1，padding=0，dilation=1，groups=1，bias=True
#################################
class cnn_net(nn.Module):

    def __init__(self,img):
        super(cnn_net,self).__init__()
        self.conv1 = nn.Conv2d(img, 8, kernel_size=4,stride=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.pool = nn.MaxPool2d(2,2)

        self.linear1 = nn.linear(512,2)
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.linear1(x)
        return x
