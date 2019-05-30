#!/usr/bin/env python

import argparse
import os
import sys

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
from torch.autograd import Variable

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms




transform = transforms.Compose([transforms.RandomCrop(8*4),
                                transforms.ToTensor()])

normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                std = [0.229, 0.224, 0.225])

scale = transforms.Compose([transforms.ToPILImage(),
                            transforms.Scale(8),
                            transforms.ToTensor(),
                            transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                std = [0.229, 0.224, 0.225])
                            ])

dataset = datasets.CIFAR100(root="dataroot", train=True, download=True, transform=transform)
assert dataset

dataloader = torch.utils.data.DataLoader(dataset, batch_size=2,
                                         shuffle=True, num_workers=0)

for epoch in range(2):

    for i, data in enumerate(dataloader):
        # Generate data
        high_res_real, _ = data
# Avoid closing
while True:
    pass
