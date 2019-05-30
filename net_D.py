import torchvision
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torchsummary import summary

class convlayer(nn.Module):
    def __init__(self,in_nc,out_nc,stride):
        super(convlayer, self).__init__()
        self.conv1=torch.nn.Conv2d(in_nc, out_nc, 3,padding=1,stride=stride)
        self.bn1=torch.nn.BatchNorm2d(out_nc,momentum=0.9)
        self.relu1=torch.nn.LeakyReLU(0.1)
    def forward(self, x):
        y=self.conv1(x)
        y=self.bn1(y)
        y=self.relu1(y)
        return y

class net_D(nn.Module):
    def __init__(self,imgwh):
        super(net_D, self).__init__()
        self.conv1=torch.nn.Conv2d(3, 64, 3,padding=1)
        self.relu1=torch.nn.LeakyReLU(0.1)
        self.convlayer1=convlayer(64,64,2)
        self.convlayer2=convlayer(64,128,1)
        self.convlayer3=convlayer(128,128,2)
        self.convlayer4=convlayer(128,256,1)
        self.convlayer5=convlayer(256,256,2)
        self.convlayer6=convlayer(256,512,1)
        self.convlayer7=convlayer(512,512,2)
        self.fc1 = nn.Linear(imgwh*imgwh*2,1024)
        self.relu2=torch.nn.LeakyReLU(0.1)
        self.fc2= nn.Linear(1024,1)
        self.relu3=torch.nn.ReLU()
        self.conv2=torch.nn.Conv2d(64, 3, 3,padding=1)
        self.sigmoid=torch.nn.Sigmoid()
    def forward(self, x):
        y=self.conv1(x)
        y=self.relu1(y)
        y=self.convlayer1(y)
        y=self.convlayer2(y)
        y=self.convlayer3(y)
        y=self.convlayer4(y)
        y=self.convlayer5(y)
        y=self.convlayer6(y)
        y=self.convlayer7(y)
        y=y.view(y.size()[0],-1)
        y=self.fc1(y)
        y=self.relu2(y)
        y=self.fc2(y)
        y=self.sigmoid(y)
        return y
""" net=net_D(64).cuda()
#net=dark_res(3,128,128).cuda()
summary(net, (3,64,64))
input_arr=torch.rand(1,3,64,64).cuda()
with SummaryWriter(comment='net') as w:
    w.add_graph(net,input_arr) """
