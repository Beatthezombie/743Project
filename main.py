import argparse
import os
import numpy as np
import time
import cv2

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
import sys
sys.path.append('ssim')
from pytorch_ssim import *
from dataset import *
from model import *
from utils import visualize_pred
from torch.autograd import Variable
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true')
args = parser.parse_args()

num_epochs = 500
batch_size = 4
param_lambda = 100
lr = 0.0005
beta1 = 0.5
beta2 = 0.999
#Create network
generator_net = UNetGI(K=128)
generator_net.apply(weights_init)
generator_net.cuda()

discriminator_net = PatchGAN(K=128)
discriminator_net.apply(weights_init)
discriminator_net.cuda()

criterion = nn.BCELoss()
criterion_l1 = nn.L1Loss()
criterion = criterion.cuda()
criterion_l1 = criterion_l1.cuda()
mse_loss = nn.MSELoss()
mse_loss = mse_loss.cuda()

cudnn.benchmark = True

label = torch.FloatTensor(batch_size)
real_label = 1
fake_label = 0
label.cuda()
label = Variable(label)

if not args.test:
    
    dataset = DGID("data/vikingvillage/", train = True, image_size=256, train_percent=0.80)
    dataset_test = DGID("data/vikingvillage/", train = False, image_size=256, train_percent=0.80)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0,  drop_last=True)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=0,  drop_last=True)
    
    optimizer_gen = optim.Adam(generator_net.parameters(), lr=lr,  betas=(beta1, beta2))
    optimizer_disc = optim.Adam(discriminator_net.parameters(), lr=lr,  betas=(beta1, beta2))
    
    start_time = time.time()
    
    e_mse = []
    e_ssim = []
    e_dx = []
    e_dgx = []
    epochs = []
    
    for epoch in range(num_epochs):
        #TRAIN
        epochs.append(epoch + 1)
        generator_net.train()
        discriminator_net.train()
            
        e_avg_mse = 0
        e_avg_ssim = 0
        e_avg_dx = 0
        e_avg_dgx = 0
        count = 0
        for i, data in enumerate(dataloader, 0):
            buffers, vxgi, direct = data
            buffers = buffers.cuda()
            vxgi = vxgi.cuda()
            direct = direct.cuda()
            gt_input = torch.cat([buffers, vxgi], dim=1)
            
            optimizer_disc.zero_grad()

            real_image_disc_pred = discriminator_net(gt_input)
            with torch.no_grad():
                label = label.resize_(real_image_disc_pred.size()).fill_(real_label).cuda()
            real_loss = criterion(real_image_disc_pred, label)
            real_loss.backward()
            real_prob = real_image_disc_pred.data.mean()
            e_avg_dx  += real_prob
            generated_img = generator_net(buffers)
            pred_input = torch.cat([buffers, generated_img.detach()], dim=1)
            fake_image_disc_pred = discriminator_net(pred_input)
            with torch.no_grad():
                label = label.resize_(fake_image_disc_pred.size()).fill_(fake_label).cuda()
            fake_loss = criterion(fake_image_disc_pred, label)
            fake_loss.backward()
            fake_prob = fake_image_disc_pred.data.mean()
            e_avg_dgx  += fake_prob
            error_discriminant = 0.5 * (fake_prob + real_prob)
            optimizer_disc.step()
            
            optimizer_gen.zero_grad()
            fake_image_disc_pred = discriminator_net(pred_input)
            with torch.no_grad():
                label = label.resize_(fake_image_disc_pred.size()).fill_(real_label).cuda()
            loss_net = criterion(fake_image_disc_pred, label) + param_lambda * criterion_l1(generated_img, vxgi)
            loss_net.backward()
            optimizer_gen.step()
            prob_gen = fake_image_disc_pred.data.mean()

            print('[{:03}] ({:03}/{:03}): time: {:.4f} Loss D: {:.4f} Loss G: {:.4f} D(x): {:.4f} D(G(z)) : {:.4f}/{:.4f}'.format(
                epoch, i, len(dataloader), time.time()-start_time, error_discriminant.data.item(), loss_net.data.item(), real_prob, fake_prob, prob_gen))
            count += 1
            
        visualize_pred("test_images/train_" + str(epoch), generated_img[0].detach().cpu().numpy(), vxgi[0].detach().cpu().numpy(), direct[0].detach().cpu().numpy())

        
        #TEST
        
        generator_net.eval()

        with torch.no_grad():
            avg_ssim = 0
            avg_mse = 0
            avg_time = 0
            i_count = 0
            for i, data in enumerate(dataloader_test, 0):
                buffers, vxgi, direct = data
                buffers = buffers.cuda()
                vxgi = vxgi.cuda()
                direct = direct.cuda()
                test_time = time.time()
                generated_img = generator_net(buffers)
                test_time = time.time() - test_time
                ssim_loss = SSIM()
                avg_ssim +=  ssim_loss(generated_img, vxgi).sum().item()
                avg_mse += mse_loss(generated_img, vxgi).sum().item()
                i_count += batch_size
                avg_time += test_time
                
        avg_ssim /= i_count
        avg_mse /= i_count
        avg_time /= i_count

        print('[{:03}]: Test result (avg): SSIM: {:.4f} MSE: {:.4f} time: {:.4f}'.format(
            epoch,avg_ssim, avg_mse, avg_time))
        
        visualize_pred("test_images/test_" + str(epoch), generated_img[0].detach().cpu().numpy(), vxgi[0].detach().cpu().numpy(), direct[0].detach().cpu().numpy())
         
            
        e_avg_dx /= float(count) 
        e_avg_dgx /= float(count)  
          
        e_mse.append(avg_mse)
        e_ssim.append(avg_ssim)
        e_dx.append(e_avg_dx)
        e_dgx.append(e_avg_dgx)
            
        #save weights
        if epoch%5 == 4:
            #save last network
            print('saving net...')
            torch.save(generator_net.state_dict(), 'models/generator_%d.pth' % (epoch))
            torch.save(discriminator_net.state_dict(), 'models/discriminator_%d.pth' % (epoch))
    
        # plot ssim, mse, dx and dgx
        
        plt.figure(figsize=(10, 10))
        plt.title('Global Illumination GAN metrics')
        
        fig, ((ax1, ax2, ax3, ax4)) = plt.subplots(4, 1, sharex='all')

        ax1.set_ylabel('D(x)')
        ax1.set_ylim(0, 1)
        ax1.plot(epochs, e_dx)

        ax2.set_ylabel('D(G(x))')
        ax2.set_ylim(0, 1)
        ax2.plot(epochs, e_dgx)

        ax3.set_ylabel('Test SSIM')
        ax3.set_ylim(0, 1)
        ax3.plot(epochs, e_ssim)
        
        #ax4.set_xticks(epochs)
        ax4.set_xlabel('Epochs')
        ax4.set_ylabel('Test MSE')
        ax4.plot(epochs, e_mse)
        plt.savefig('test_images/GAN_metrics.png')