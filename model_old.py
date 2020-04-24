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

class UNetGI(nn.Module):
    def __init__(self, K=32):
        super(UNetGI, self).__init__()
        self.K = K
        # encoder
        self.el1 = EncoderLayer(15, self.K, use_bn=False)
        self.el2 = EncoderLayer(self.K, 2*self.K)
        self.el3 = EncoderLayer(2*self.K, 2*self.K)
        self.el4 = EncoderLayer(2*self.K, 4*self.K)
        self.el5 = EncoderLayer(4*self.K, 8*self.K)
        self.el6 = EncoderLayer(8*self.K, 8*self.K)
        self.el7 = EncoderLayer(8*self.K, 8*self.K)
        self.el8 = EncoderLayer(8*self.K, 8*self.K)
        # decoder
        self.dl1 = DecoderLayer(8*self.K, 8 * self.K, 8*self.K)
        self.dl2 = DecoderLayer(8*self.K, 8 * self.K, 8*self.K)
        self.dl3 = DecoderLayer(8*self.K, 8 * self.K, 8*self.K)
        self.dl4 = DecoderLayer(8*self.K, 4 * self.K, 4*self.K)
        self.dl5 = DecoderLayer(4*self.K, 2 * self.K, 2*self.K)
        self.dl6 = DecoderLayer(2*self.K, 2 * self.K, 2*self.K)
        self.dl7 = DecoderLayer(2*self.K,     self.K, self.K)
        self.dl8 = DecoderLayer(self.K, 3, 15, use_bn=False, use_tanh=True)
           
    def forward(self, x):
        # encoder
        x1 = x
        x2 = self.el1(x1)
        x3 = self.el2(x2)
        x4 = self.el3(x3)
        x5 = self.el4(x4)
        x6 = self.el5(x5)
        x7 = self.el6(x6)
        x8 = self.el7(x7)
        x = self.el8(x8)
        # decoder
        x = self.dl1(x, x8)
        x = self.dl2(x, x7)
        x = self.dl3(x, x6)
        x = self.dl4(x, x5)
        x = self.dl5(x, x4)
        x = self.dl6(x, x3)
        x = self.dl7(x, x2)
        x = self.dl8(x, x1)
        return x
    
class EncoderLayer(nn.Module): 
    def __init__(self, in_channels, out_channels, use_bn=True):
        super(EncoderLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bn = use_bn
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(self.out_channels)
        self.dropout = nn.Dropout(p=0.4)
        
    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        x = F.leaky_relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        #x = self.dropout(x)
        return x

class DecoderLayer(nn.Module): 
    def __init__(self, in_channels, out_channels, skip_channels, use_bn=True, use_tanh=False):
        super(DecoderLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bn = use_bn
        self.use_tanh = use_tanh
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #self.up = nn.ConvTranspose2d(self.in_channels , self.in_channels, kernel_size=2, stride=2)
        self.conv = nn.Conv2d(self.in_channels + skip_channels, self.out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(self.out_channels)
        self.dropout = nn.Dropout(p=0.4)
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.use_tanh:
            x = torch.sigmoid(x)
        else:
            x = F.relu(x)
            
        #x = self.dropout(x)
        return x
    
    
class PatchGAN(nn.Module):
    def __init__(self, K=64):
        super(PatchGAN, self).__init__()
        self.K = K
        
        self.l1 = PatchGanLayer(18, self.K, use_bn=False)
        self.l2 = PatchGanLayer(self.K, 2 * self.K)
        self.l3 = PatchGanLayer(2 * self.K, 4 * self.K)
        self.l4 = PatchGanLayer(4 * self.K, 8 * self.K)
        self.l5 = PatchGanLayer(8 * self.K, 1, use_bn=False, use_sig=True)
        
    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        return x
        
class PatchGanLayer(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=True, use_sig=False):
        super(PatchGanLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bn = use_bn
        self.use_sigmoid = use_sig
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(self.out_channels)
        
    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.use_sigmoid:
            x = torch.sigmoid(x)
        else:
            x = F.leaky_relu(x)
        return x