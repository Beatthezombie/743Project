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
import numpy as np
import os
import cv2
from enum import Enum
import random
from PIL import Image
import torchvision.transforms.functional as F
import torch

"""
Enumeration of buffer types available as png files
"""
class BufferType(Enum):
    albedo = 1
    normal = 2
    depth = 3
    direct = 4
    occlusion = 5
    specular = 6
    smoothness = 7
    vxgi = 8
    raytracing = 9
    pred = 10

buffer_file_prefix = {}  
buffer_file_prefix[BufferType.albedo] = 'Albedo'
buffer_file_prefix[BufferType.normal] = 'Normal'
buffer_file_prefix[BufferType.depth] = 'Depth'
buffer_file_prefix[BufferType.direct] = 'Direct'
buffer_file_prefix[BufferType.occlusion] = 'Occlusion'
buffer_file_prefix[BufferType.specular] = 'Specular'
buffer_file_prefix[BufferType.smoothness] = 'Smoothness'
buffer_file_prefix[BufferType.vxgi] = 'VXGI'
buffer_file_prefix[BufferType.raytracing] = 'Raytracing'
buffer_file_prefix[BufferType.pred] = 'Prediction'

def load_image(buffer_type, index, directory):
    filename = build_file_name(buffer_type, index)
    filename = directory + filename
    img = Image.open(filename)
    img.load()
    return img

def build_file_name(buffer_type, index):
    padded_number = str(index).zfill(5)
    result = buffer_file_prefix[buffer_type] + "_" + padded_number + ".png"
    return result

def fixup_tensor(tensor):
    if tensor.shape[0] == 4:
        tensor = tensor[0:3, :, :]
    else:
        if tensor.shape[0] == 1:
            tensor = tensor.type(torch.FloatTensor) * (1.0 / 65535)
    return tensor
    
class DGID(torch.utils.data.Dataset):
    def __init__(self, imgdir, train = True, image_size=512, train_percent=0.80):
        self.train = train
        self.imgdir = imgdir
        self.img_names = os.listdir(self.imgdir)
        self.image_size = image_size
        self.sample_count = len(self.img_names) // 8
        self.data_count = 0
        split_index = int(train_percent * self.sample_count )

        print("Loading images...")
        self.albedo = []
        self.normal = []
        self.depth = []
        self.direct = []
        self.occlusion = []
        self.specular = []
        self.smoothness = []
        self.vxgi = []
        self.raytracing = []
        
        for i in range(0, self.sample_count):
            if (i < split_index and self.train == True) or (i >= split_index and self.train == False):
                self.data_count += 1
                index = i + 1
                self.albedo.append(load_image(BufferType.albedo, index, self.imgdir))
                self.normal.append(load_image(BufferType.normal, index, self.imgdir))
                self.depth.append(load_image(BufferType.depth, index, self.imgdir))
                self.direct.append(load_image(BufferType.direct, index + 1, self.imgdir))
                self.occlusion.append(load_image(BufferType.occlusion, index, self.imgdir))
                self.specular.append(load_image(BufferType.specular, index, self.imgdir))
                self.smoothness.append(load_image(BufferType.smoothness, index, self.imgdir))
                self.vxgi.append(load_image(BufferType.vxgi, index, self.imgdir))
            else:
                continue
        print(self.data_count)
        print("Loading done!")
        
    def __len__(self):
        return self.data_count

    def __getitem__(self, index):

        tensorTransform = transforms.ToTensor()
        
        albedo = self.albedo[index]
        normal = self.normal[index]
        depth = self.depth[index]
        direct = self.direct[index]
        occlusion = self.occlusion[index]
        specular = self.specular[index]
        smoothness = self.smoothness[index]
        vxgi = self.vxgi[index]

        (w, h) = albedo.size

        albedo = F.resize(albedo, (self.image_size, self.image_size))
        normal = F.resize(normal, (self.image_size, self.image_size))
        depth = F.resize(depth, (self.image_size, self.image_size))
        direct = F.resize(direct, (self.image_size, self.image_size))
        occlusion = F.resize(occlusion, (self.image_size, self.image_size))
        specular = F.resize(specular, (self.image_size, self.image_size))
        smoothness = F.resize(smoothness, (self.image_size, self.image_size))
        vxgi = F.resize(vxgi, (self.image_size, self.image_size))
        
        albedo = fixup_tensor(tensorTransform(albedo))
        normal = fixup_tensor(tensorTransform(normal))
        depth = fixup_tensor(tensorTransform(depth))
        direct = fixup_tensor(tensorTransform(direct))
        occlusion = fixup_tensor(tensorTransform(occlusion))
        specular =fixup_tensor( tensorTransform(specular))
        smoothness = fixup_tensor(tensorTransform(smoothness))
        vxgi = fixup_tensor(tensorTransform(vxgi))
        
        input_buffer = torch.cat([albedo, normal, depth, direct, occlusion, specular, smoothness], 0)
        
        return input_buffer, vxgi, direct
