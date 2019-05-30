import torchvision
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torchsummary import summary

class reslayer(nn.Module):
    def __init__(self,f_n):
        super(reslayer, self).__init__()
        self.conv1=torch.nn.Conv2d(f_n, f_n, 3,padding=1)
        self.bn1=torch.nn.BatchNorm2d(f_n,momentum=0.9)
        self.relu1=torch.nn.ReLU()#LeakyReLU(0.1)
        self.conv2=torch.nn.Conv2d(f_n, f_n, 3,padding=1)
        self.bn2=torch.nn.BatchNorm2d(f_n,momentum=0.9)
        self.relu2=torch.nn.ReLU()#LeakyReLU(0.1)
    def forward(self, x):
        y=self.conv1(x)
        y=self.bn1(y)
        y=self.relu1(y)
        y=self.conv2(y)
        y=self.bn2(y)
        y+=x
        y=self.relu2(y)
        return y
class net_G(nn.Module):
    def __init__(self):
        super(net_G, self).__init__()
        self.conv1=torch.nn.Conv2d(3, 64, 3,padding=1)
        self.relu1=torch.nn.ReLU()
        self.reslayer1=reslayer(64)
        self.reslayer2=reslayer(64)
        self.reslayer3=reslayer(64)
        self.reslayer4=reslayer(64)
        self.reslayer5=reslayer(64)
        self.reslayer6=reslayer(64)
        self.upconv1=nn.ConvTranspose2d(64, 64,
                                        kernel_size=4, stride=2,
                                        padding=1)
        self.relu2=torch.nn.ReLU()
        self.upconv2=nn.ConvTranspose2d(64,64,
                                        kernel_size=4, stride=2,
                                        padding=1)
        self.relu3=torch.nn.ReLU()
        self.conv2=torch.nn.Conv2d(64, 3, 3,padding=1)
    def forward(self, x):
        y=self.conv1(x)
        y=self.relu1(y)
        y=self.reslayer1(y)
        y=self.reslayer2(y)
        y=self.reslayer3(y)
        y=self.reslayer4(y)
        y=self.reslayer5(y)
        y=self.reslayer6(y)
        y=self.upconv1(y)
        y=self.relu2(y)
        y=self.upconv2(y)
        y=self.relu3(y)
        y=self.conv2(y)
        return y
""" net=net_G().cuda()
#net=dark_res(3,128,128).cuda()
summary(net, (3,64,64))
input_arr=torch.rand(1,3,64,64).cuda()
with SummaryWriter(comment='net') as w:
    w.add_graph(net,input_arr) """
