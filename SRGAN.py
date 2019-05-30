import net_D,net_G
import torch
import torch.nn as nn
import os
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import random
import numpy as np
import cv2
import tqdm
from torch.nn import init
from torch.autograd import Variable
#from vis import Visualizer
from PIL import Image

class FeatureExtractor(nn.Module):
    def __init__(self, cnn, feature_layer=11):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(cnn.features.children())[:(feature_layer+1)])

    def forward(self, x):
        return self.features(x)

class SRGAN():
    def __init__(self,input_size,batchSize):        
        self.transforminv = transforms.Compose([transforms.Normalize(mean = [-2.118, -2.036, -1.804], # Equivalent to un-normalizing ImageNet (for correct visualization)
                                                                    std = [4.367, 4.464, 4.444]),
                                            transforms.ToPILImage(),
                                            transforms.Scale(input_size*4)])
        self.transform = transforms.Compose([transforms.RandomCrop(input_size*4),transforms.ToTensor()])
        self.scale = transforms.Compose([transforms.ToPILImage(),
                            transforms.Scale(input_size),
                            transforms.ToTensor(),
                            transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                std = [0.229, 0.224, 0.225])
                            ])
        self.normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                std = [0.229, 0.224, 0.225])
        self.input_size=input_size
        self.batchSize=batchSize
        self.net_G=net_G.net_G().cuda()
        self.net_D=net_D.net_D(input_size*4).cuda()
        self.net_F = FeatureExtractor(torchvision.models.vgg19(pretrained=True)).cuda()
        self.BCELoss = nn.BCELoss().cuda()
        self.MSELoss = nn.MSELoss().cuda()
        self.ones_const = Variable(torch.ones(batchSize, 1)).cuda()
        self.optim_G = torch.optim.Adam(self.net_G.parameters(), lr=0.00001)
        self.optim_D = torch.optim.Adam(self.net_D.parameters(), lr=0.00001) 
        self.init_weights()
    def init_weights(self, init_type='normal', init_gain=0.02):
        def init_func(m):  # define the initialization function
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, init_gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=init_gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=init_gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
                init.normal_(m.weight.data, 1.0, init_gain)
                init.constant_(m.bias.data, 0.0)
        print('initialize network with %s' % init_type)
    def load_model(self):
        self.net_G.load_state_dict(torch.load("model/net_G.pth"))
        self.net_D.load_state_dict(torch.load("model/net_D.pth")) 
    def save_model(self):
        torch.save(self.net_G.state_dict(),"model/net_G.pth")
        torch.save(self.net_D.state_dict(),"model/net_D.pth")
    def train(self,ep_nub):
        #visualizer = Visualizer(image_size=self.input_size*4)
        dataset = datasets.CIFAR100(root="dataroot", train=True, download=True, transform=self.transform)
        assert dataset
        for epoch in range(ep_nub):
            mean_generator_total_loss = 0.0
            mean_discriminator_loss = 0.0
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batchSize,shuffle=True, num_workers=4)
            for i, data in enumerate(dataloader):
                high_res_real, _ = data
                low_res = torch.FloatTensor(high_res_real.size()[0], 3, self.input_size, self.input_size)
                for j in range(high_res_real.size()[0]):
                    low_res[j] = self.scale(high_res_real[j])
                    high_res_real[j] = self.normalize(high_res_real[j])
                high_res_real = Variable(high_res_real.cuda())
                high_res_fake = self.net_G(Variable(low_res).cuda())
                target_real = Variable(torch.rand(self.batchSize,1)*0.5 + 0.7).cuda()
                target_fake = Variable(torch.rand(self.batchSize,1)*0.3).cuda()
                        ######### Train discriminator #########
                self.net_D.zero_grad()

                discriminator_loss = self.BCELoss(self.net_D(high_res_real), target_real) + \
                                    self.BCELoss(self.net_D(Variable(high_res_fake.data)), target_fake)
                mean_discriminator_loss += discriminator_loss.item()
                
                discriminator_loss.backward()
                self.optim_D.step()

                ######### Train generator #########
                self.optim_G.zero_grad()

                real_features = Variable(self.net_F(high_res_real).data)
                fake_features = self.net_F(high_res_fake)

                generator_content_loss = self.MSELoss(high_res_fake, high_res_real) + 0.006*self.MSELoss(fake_features, real_features)
                generator_adversarial_loss = self.BCELoss(self.net_D(high_res_fake), self.ones_const)
                generator_total_loss = generator_content_loss + 1e-3*generator_adversarial_loss
                mean_generator_total_loss += generator_total_loss.item()
                generator_total_loss.backward()
                self.optim_G.step()
                print(discriminator_loss.item(),"\t",generator_total_loss.item())
                # if(i%10==5):
                #     visualizer.show(low_res, high_res_real.cpu().data, high_res_fake.detach().cpu().data)
                if(i%100==0):
                    newIm= Image.new('RGB', (self.input_size*4*3, self.input_size*4), 'white')
                    lr_image = self.transforminv(low_res[0])
                    #lr_imagec = lr_image.crop((0, 0, self.input_size*4,self.input_size*4))
                    hr_image = self.transforminv(high_res_real.cpu().data[0])
                    #hr_imagec = hr_image.crop((0, 0, self.input_size*4,self.input_size*4))
                    fake_hr_image = self.transforminv(high_res_fake.detach().cpu().data[0])
                    #fake_hr_imagec = fake_hr_image.crop((0, 0, self.input_size*4,self.input_size*4))
                    newIm.paste(lr_image, (0 ,0))
                    newIm.paste(hr_image, (self.input_size*4 ,0))
                    newIm.paste(fake_hr_image, (self.input_size*4*2 ,0))
                    newIm.save("C:/Users/long/Desktop/SRGAN/rz/rz_img/"+("%05d" % epoch)+("%05d" % i)+".jpg")
                    self.save_model()
"""                 rinb=low_res.cpu().numpy().transpose((0,2,3,1))[0]*255
                r_in=cv2.resize(rinb.astype(np.uint8),(self.input_size*4,self.input_size*4))
                
                r_out=(high_res_real.cpu().numpy().transpose((0,2,3,1))[0]*255).astype(np.uint8)
                f_out=(high_res_fake.detach().cpu().numpy().transpose((0,2,3,1))[0]*255).astype(np.uint8)
                result=np.concatenate((r_in, r_out,f_out),axis=1)
                cv2.imwrite("rz/z_img/"+("%05d" % epoch)+".jpg",result) """
if __name__ == "__main__":
    sr=SRGAN(8,128)
    sr.train(1)