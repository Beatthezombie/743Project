import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F


def GILoss(gt, pred, gt_prob, pred_prob):
    return F.binary_cross_entropy(gt_prob, pred_prob) + F.l1_loss(gt, pred)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# heavily inspired from https://github.com/CreativeCodingLab/DeepIllumination/blob/master/model.py
class UNetGI(nn.Module):
    def __init__(self, K=32):
        super(UNetGI, self).__init__()
        self.K = K
        
        #constants
        kernel_size = 4
        stride = 2
        padding = 1
        
        # batch norm
        self.bn = nn.BatchNorm2d(self.K)
        self.bn2 = nn.BatchNorm2d(self.K * 2)
        self.bn4 = nn.BatchNorm2d(self.K * 4)
        self.bn8 = nn.BatchNorm2d(self.K * 8)
        
        #drop out for deconv
        self.dropout = nn.Dropout(0.5)
        
        #activations
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        
        # encoder
        self.conv1 = nn.Conv2d(15,         self.K,     kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(self.K,     self.K * 2, kernel_size, stride, padding)
        self.conv3 = nn.Conv2d(self.K * 2, self.K * 4, kernel_size, stride, padding)
        self.conv4 = nn.Conv2d(self.K * 4, self.K * 8, kernel_size, stride, padding)
        self.conv5 = nn.Conv2d(self.K * 8, self.K * 8, kernel_size, stride, padding)
        self.conv6 = nn.Conv2d(self.K * 8, self.K * 8, kernel_size, stride, padding)
        self.conv7 = nn.Conv2d(self.K * 8, self.K * 8, kernel_size, stride, padding)
        self.conv8 = nn.Conv2d(self.K * 8, self.K * 8, kernel_size, stride, padding)
        #decoder
        self.deconv1 = nn.ConvTranspose2d(self.K * 8,     self.K * 8, kernel_size, stride, padding)
        self.deconv2 = nn.ConvTranspose2d(self.K * 8 * 2, self.K * 8, kernel_size, stride, padding)
        self.deconv3 = nn.ConvTranspose2d(self.K * 8 * 2, self.K * 8, kernel_size, stride, padding)
        self.deconv4 = nn.ConvTranspose2d(self.K * 8 * 2, self.K * 8, kernel_size, stride, padding)
        self.deconv5 = nn.ConvTranspose2d(self.K * 8 * 2, self.K * 4, kernel_size, stride, padding)
        self.deconv6 = nn.ConvTranspose2d(self.K * 4 * 2, self.K * 2, kernel_size, stride, padding)
        self.deconv7 = nn.ConvTranspose2d(self.K * 2 * 2, self.K,     kernel_size, stride, padding)
        self.deconv8 = nn.ConvTranspose2d(self.K * 2,     3,          kernel_size, stride, padding)
                
    def forward(self, x):
        encoder1 = self.conv1(x)
        encoder2 = self.bn2(self.conv2(self.leaky_relu(encoder1)))
        encoder3 = self.bn4(self.conv3(self.leaky_relu(encoder2)))
        encoder4 = self.bn8(self.conv4(self.leaky_relu(encoder3)))
        encoder5 = self.bn8(self.conv5(self.leaky_relu(encoder4)))
        encoder6 = self.bn8(self.conv6(self.leaky_relu(encoder5)))
        encoder7 = self.bn8(self.conv7(self.leaky_relu(encoder6)))
        encoder8 = self.conv8(self.leaky_relu(encoder7))

        decoder1 = self.dropout(self.bn8(self.deconv1(self.relu(encoder8))))
        decoder1 = torch.cat((decoder1, encoder7), 1)
        decoder2 = self.dropout(self.bn8(self.deconv2(self.relu(decoder1))))
        decoder2 = torch.cat((decoder2, encoder6), 1)
        decoder3 = self.dropout(self.bn8(self.deconv3(self.relu(decoder2))))
        decoder3 = torch.cat((decoder3, encoder5), 1)
        decoder4 = self.bn8(self.deconv4(self.relu(decoder3)))
        decoder4 = torch.cat((decoder4, encoder4), 1)
        decoder5 = self.bn4(self.deconv5(self.relu(decoder4)))
        decoder5 = torch.cat((decoder5, encoder3), 1)
        decoder6 = self.bn2(self.deconv6(self.relu(decoder5)))
        decoder6 = torch.cat((decoder6, encoder2),1)
        decoder7 = self.bn(self.deconv7(self.relu(decoder6)))
        decoder7 = torch.cat((decoder7, encoder1), 1)
        decoder8 = self.deconv8(self.relu(decoder7))
        x = self.tanh(decoder8)
        return x
    
    
    
class PatchGAN(nn.Module):
    def __init__(self, K=64):
        super(PatchGAN, self).__init__()
        self.K = K
        
        #constants
        kernel_size = 4
        stride = 2
        stride_end = 1
        padding = 1
        
        # batch norm
        self.bn2 = nn.BatchNorm2d(self.K * 2)
        self.bn4 = nn.BatchNorm2d(self.K * 4)
        self.bn8 = nn.BatchNorm2d(self.K * 8)
        
        #activations
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()
        
        self.conv1 = nn.Conv2d(18, self.K,             kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(self.K,     self.K * 2, kernel_size, stride, padding)
        self.conv3 = nn.Conv2d(self.K * 2, self.K * 4, kernel_size, stride, padding)
        self.conv4 = nn.Conv2d(self.K * 4, self.K * 8, kernel_size, stride_end, padding)
        self.conv5 = nn.Conv2d(self.K * 8, 1,          kernel_size, stride_end, padding)
        
        
    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.bn2(self.conv2(x)))
        x = self.leaky_relu(self.bn4(self.conv3(x)))
        x = self.leaky_relu(self.bn8(self.conv4(x)))
        x = self.sigmoid(self.conv5(x))
        return x