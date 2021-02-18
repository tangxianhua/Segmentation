
import math
import torch
from torch import nn
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
from functools import partial
import torch
import torch.nn as nn

class DenseNet2D_down_block(nn.Module):
    def __init__(self, input_channels, output_channels, down_size):
        super(DenseNet2D_down_block, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
        self.conv21 = nn.Conv2d(input_channels + output_channels, output_channels, kernel_size=1,padding=0)
        self.conv22 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
        self.conv31 = nn.Conv2d(input_channels + 2 * output_channels, output_channels, kernel_size=1,padding=0)
        self.conv32 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=down_size)
        self.relu = nn.LeakyReLU()
        self.norm = nn.BatchNorm2d(output_channels)
        self.down_size = down_size
    def forward(self, x):
        if self.down_size != None:
            x = self.max_pool(x)
        x1 = self.relu(self.norm(self.conv1(x)))
        x21 = torch.cat((x, x1), dim=1)
        x22 = self.relu(self.norm(self.conv22(self.conv21(x21))))
        x31 = torch.cat((x21, x22), dim=1)
        out = self.relu(self.norm(self.conv32(self.conv31(x31))))
        return out
class DenseNet2D_down_blockM(nn.Module):
    def __init__(self, input_channels, output_channels, down_size):
        super(DenseNet2D_down_blockM, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
        self.conv21 = nn.Conv2d(input_channels + output_channels, output_channels, kernel_size=1,padding=0)
        self.conv22 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
        self.conv31 = nn.Conv2d(input_channels + 2 * output_channels, output_channels, kernel_size=1,padding=0)
        self.conv32 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=down_size)
        self.relu = nn.LeakyReLU()
        self.norm = nn.BatchNorm2d(output_channels)
        self.down_size = down_size
    def forward(self, x):
        if self.down_size != None:
            x = self.max_pool(x)
        x1 = self.relu(self.norm(self.conv1(x)))
        x21 = torch.cat((x, x1), dim=1)
        x22 = self.relu(self.norm(self.conv22(self.conv21(x21))))
        x31 = torch.cat((x21, x22), dim=1)
        out = self.relu(self.norm(self.conv32(self.conv31(x31))))
        return out
class DenseNet2D_up_block_concat(nn.Module):
    def __init__(self, skip_channels, input_channels, output_channels, up_stride ):
        super(DenseNet2D_up_block_concat, self).__init__()
        self.conv11 = nn.Conv2d(skip_channels + input_channels, output_channels, kernel_size=1,padding=0)
        self.conv12 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
        self.conv21 = nn.Conv2d(skip_channels + input_channels + output_channels, output_channels,kernel_size=1, padding=0)
        self.conv22 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU()
        self.norm = nn.BatchNorm2d(output_channels)
        self.up_stride = up_stride
    def forward(self, prev_feature_map, x):
        x = nn.functional.interpolate(x, scale_factor=self.up_stride, mode='nearest')#需要修改
        x = torch.cat((x, prev_feature_map), dim=1)
        x1 = self.relu(self.norm(self.conv12(self.conv11(x))))
        x21 = torch.cat((x, x1), dim=1)
        out = self.relu(self.norm(self.conv22(self.conv21(x21))))
        return out
class Xmodule(nn.Module):
    def __init__(self,input_channels, output_channels):
        super(Xmodule,self).__init__()
        self.conv = nn.Conv3d(input_channels+input_channels, output_channels, kernel_size=3, padding=1)#需要down一半
        self.norm = nn.BatchNorm3d(output_channels)
        self.relu = nn.LeakyReLU()#Parametric Relu
        self.dropout = nn.Dropout3d(p=0.25)#25%高斯dropout
    def forward(self, x,y):
        x = torch.cat((x, y), dim=1)
        x1 = self.conv1(x)
        x2 = self.norm(x1)
        x3 = self.relu(x2)
        out = self.dropout(x3)
        return out
class Changedim(nn.Module):
    def __init__(self,input_channels, output_channels):
        super(Changedim,self).__init__()
        self.conv = nn.Conv3d(input_channels+input_channels, output_channels, kernel_size=1, padding=0)#需要down一半
        self.norm = nn.BatchNorm3d(output_channels)
        self.relu = nn.LeakyReLU()#Parametric Relu
    def forward(self, x,y):
        x = torch.cat((x, y), dim=1)
        x1 = self.conv1(x)
        x2 = self.norm(x1)
        x3 = self.relu(x2)
        return x3

class DenseNetMy(nn.Module):
    def __init__(self, in_channels=1,  channel_size=8, concat=True):
        super(DenseNetMy, self).__init__()
        self.down_block1 = DenseNet2D_down_block(input_channels=in_channels, output_channels=channel_size,down_size=None)
        self.down_block2 = DenseNet2D_down_block(input_channels=8, output_channels=16,down_size=2)
        self.down_block3 = DenseNet2D_down_block(input_channels=16, output_channels=32,down_size=2)
        self.down_block4 = DenseNet2D_down_block(input_channels=32, output_channels=64,down_size=2)
        self.down_block5 = DenseNet2D_down_block(input_channels=64, output_channels=128,down_size=2)
        self.up_block1 = DenseNet2D_up_block_concat(skip_channels=64, input_channels=128,output_channels=64, up_stride=2)
        self.up_block2 = DenseNet2D_up_block_concat(skip_channels=32, input_channels=64,output_channels=32, up_stride=2)
        self.up_block3 = DenseNet2D_up_block_concat(skip_channels=16, input_channels=32,output_channels=16, up_stride=2)
        self.up_block4 = DenseNet2D_up_block_concat(skip_channels=8, input_channels=16,output_channels=8, up_stride=2)
        self.out_conv1 = nn.Sequential(nn.Conv3d(8, 1, kernel_size=1), nn.Sigmoid())
        self.concat = concat
    def forward(self, x):
        self.x1 = self.down_block1(x)#1, 16, 32, 256, 256]
        self.x2 = self.down_block2(self.x1)#[1, 16, 16, 128, 128]  32
        self.x3 = self.down_block3(self.x2)#[1, 16, 8, 64, 64]  64
        self.x4 = self.down_block4(self.x3)#[1, 16, 4, 32, 32]  128
        self.x5 = self.down_block5(self.x4)#[1, 16, 2, 16, 16]  256
        self.x6 = self.up_block1(self.x4, self.x5)#[1, 16, 4, 32, 32] 128
        self.x7 = self.up_block2(self.x3, self.x6)#[1, 16, 8, 64, 64]) 64
        self.x8 = self.up_block3(self.x2, self.x7)#[1, 16, 16, 128, 128]  32
        self.x9 = self.up_block4(self.x1, self.x8)#[1, 16, 32, 256, 256]  16
        out = self.out_conv1(self.x9)#[1, 1, 32, 256, 256]
        return out
class DenseNetMyX(nn.Module):
    def __init__(self, in_channels=1,  channel_size=8, concat=True):
        super(DenseNetMyX, self).__init__()
        self.down_block1 = DenseNet2D_down_block(input_channels=in_channels, output_channels=channel_size,down_size=None)
        self.down_block2 = DenseNet2D_down_block(input_channels=8, output_channels=16,down_size=2)
        self.down_block3 = DenseNet2D_down_block(input_channels=16, output_channels=32,down_size=2)
        self.down_block4 = DenseNet2D_down_block(input_channels=32, output_channels=64,down_size=2)
        self.down_block5 = DenseNet2D_down_block(input_channels=64, output_channels=128,down_size=2)
        self.up_block1 = DenseNet2D_up_block_concat(skip_channels=64, input_channels=128,output_channels=64, up_stride=2)
        self.up_block2 = DenseNet2D_up_block_concat(skip_channels=32, input_channels=64,output_channels=32, up_stride=2)
        self.up_block3 = DenseNet2D_up_block_concat(skip_channels=16, input_channels=32,output_channels=16, up_stride=2)
        self.up_block4 = DenseNet2D_up_block_concat(skip_channels=8, input_channels=16,output_channels=8, up_stride=2)
        self.out_conv1 = nn.Sequential(nn.Conv3d(8, 1, kernel_size=1), nn.Sigmoid())
        self.concat = concat
    def forward(self, x,y,z):
        x1 = self.down_block1(x)#1, 16,256, 256]
        c1 = self.down_block1(y)#1, 16,32, 256]
        s1 = self.down_block1(z)
        x2 = self.down_block2(x1)#[1, 32, 128, 128]
        c2 = self.down_block2(c1)
        s2 = self.down_block2(s1)
        x3 = self.down_block3(x2)#[1, 64, 64, 64]
        c3 = self.down_block3(c2)
        s3 = self.down_block3(s2)
        x4 = self.down_block4(x3)#[1, 128, 32, 32]
        c4 = self.down_block4(c3)
        s4 = self.down_block4(s3)
        x5 = self.down_block5(x4)#[1, 256, 16, 16]
        c5 = self.down_block5(c4)
        s5 = self.down_block5(s4)
        x6 = self.up_block1(x4, x5)#[1, 128, 32, 32]
        x7 = self.up_block2(x3, x6)#[1, 64, 64, 64]
        x8 = self.up_block3(x2, x7)#[1, 32, 128, 128]
        x9 = self.up_block4(x1, x8)#[1, 16, 256, 256]
        out = self.out_conv1(x9)#[1, 1, 256, 256]
        return out



class DenseNetMyMerge2(nn.Module):
    def __init__(self, in_channels=1,  channel_size=8, concat=True):
        super(DenseNetMyMerge2, self).__init__()
        self.down_block1 = DenseNet2D_down_block(input_channels=in_channels, output_channels=channel_size,down_size=None)
        self.down_block2 = DenseNet2D_down_block(input_channels=8, output_channels=16,down_size=2)
        self.down_block3 = DenseNet2D_down_block(input_channels=16, output_channels=32,down_size=2)
        self.down_block4 = DenseNet2D_down_block(input_channels=32, output_channels=64,down_size=2)
        self.down_block5 = DenseNet2D_down_block(input_channels=64, output_channels=128,down_size=2)
        self.down_block2M = DenseNet2D_down_blockM(input_channels=24, output_channels=16,down_size=2)
        self.down_block3M = DenseNet2D_down_blockM(input_channels=48, output_channels=32,down_size=2)
        self.down_block4M = DenseNet2D_down_blockM(input_channels=96, output_channels=64,down_size=2)
        self.down_block5M = DenseNet2D_down_blockM(input_channels=160, output_channels=128,down_size=2)
        self.up_block1 = DenseNet2D_up_block_concat(skip_channels=64, input_channels=128,output_channels=64, up_stride=2)
        self.up_block2 = DenseNet2D_up_block_concat(skip_channels=32, input_channels=64,output_channels=32, up_stride=2)
        self.up_block3 = DenseNet2D_up_block_concat(skip_channels=16, input_channels=32,output_channels=16, up_stride=2)
        self.up_block4 = DenseNet2D_up_block_concat(skip_channels=8, input_channels=16,output_channels=8, up_stride=2)
        self.out_conv1 = nn.Sequential(nn.Conv3d(8, 1, kernel_size=1), nn.Sigmoid())
        self.Cdown_block1 = DenseNet2D_down_block(input_channels=in_channels, output_channels=channel_size,down_size=None)
        self.Cdown_block2 = DenseNet2D_down_block(input_channels=8, output_channels=16,down_size=2)
        self.Cdown_block3 = DenseNet2D_down_block(input_channels=16, output_channels=32,down_size=2)
        self.Cdown_block4 = DenseNet2D_down_block(input_channels=32, output_channels=64,down_size=2)
        self.Cdown_block5 = DenseNet2D_down_block(input_channels=64, output_channels=128,down_size=2)
        self.changedim1C = Changedim(input_channels=24, output_channels=16)
        self.changedim2C = Changedim(input_channels=48, output_channels=32)
        self.changedim3C = Changedim(input_channels=96, output_channels=64)
        self.changedim4C = Changedim(input_channels=160, output_channels=128)
        self.Cup_block1 = DenseNet2D_up_block_concat(skip_channels=64, input_channels=128,output_channels=64, up_stride=2)
        self.Cup_block2 = DenseNet2D_up_block_concat(skip_channels=32, input_channels=64,output_channels=32, up_stride=2)
        self.Cup_block3 = DenseNet2D_up_block_concat(skip_channels=16, input_channels=32,output_channels=16, up_stride=2)
        self.Cup_block4 = DenseNet2D_up_block_concat(skip_channels=8, input_channels=16,output_channels=8, up_stride=2)
        self.Cout_conv1 = nn.Sequential(nn.Conv3d(8, 1, kernel_size=1), nn.Sigmoid())
        self.XM1 = Xmodule(input_channels=8, output_channels=16)
        self.XM2 = Xmodule(input_channels=16, output_channels=32)
        self.XM3 = Xmodule(input_channels=32, output_channels=64)
        self.XM4 = Xmodule(input_channels=64, output_channels=128)
        self.concat = concat
    def forward(self, x):
        self.x1 = self.down_block1(x)#1, 8, 32, 256, 256]
        self.Cx1 = self.Cdown_block1(x)#1, 8, 32, 256, 256]
        self.Module1 = self.XM1(self.x1,self.Cx1)
        self.x2 = self.down_block2(self.x1)#[1, 16, 16, 128, 128]
        self.x2Mo = self.down_block2M(torch.cat([self.x1,self.Module1], 1))
        self.change1 = self.changedim1C(self.Cx1,self.Module1)
        self.Cx2 = self.Cdown_block2(self.change1)#[1, 16, 16, 128, 128]
        self.Module2 = self.XM2(self.x2Mo,self.Cx2)
        self.x3 = self.down_block3(self.x2)#[1, 32, 8, 64, 64]
        self.x3Mo = self.down_block3M(torch.cat([self.x2, self.Module2], 1))
        self.change2 = self.changedim2C(self.Cx2, self.Module2)
        self.Cx3 = self.Cdown_block3(self.change2)#[1, 32, 8, 64, 64]
        self.Module3 = self.XM3(self.x3Mo, self.Cx3)
        self.x4 = self.down_block4(self.x3)#[1, 64, 4, 32, 32]
        self.x4Mo = self.down_block4M(torch.cat([self.x3, self.Module3], 1))
        self.change3 = self.changedim3C(self.Cx3, self.Module3)
        self.Cx4 = self.Cdown_block4(self.change3)#[1, 64, 4, 32, 32]
        self.Module4 = self.XM4(self.x4Mo, self.Cx4)
        self.x5 = self.down_block5(self.x4)#[1, 128, 2, 16, 16]
        self.x5Mo = self.down_block5M(torch.cat([self.x4, self.Module4], 1))
        self.change4 = self.changedim4C(self.Cx4, self.Module4)
        self.Cx5 = self.Cdown_block5(self.change4)#[1, 128, 2, 16, 16]
        self.x6 = self.up_block1(self.x4, self.x5)#[1, 64, 4, 32, 32]
        self.Cx6 = self.Cup_block1(self.Cx4, self.x5)#[1, 64, 4, 32, 32]
        self.x7 = self.up_block2(self.x3, self.x6)#[1, 32, 8, 64, 64]) 64
        self.Cx7 = self.Cup_block2(self.Cx3, self.Cx6)#[1, 32, 8, 64, 64])
        self.x8 = self.up_block3(self.x2, self.x7)#[1, 16, 16, 128, 128]
        self.Cx8 = self.Cup_block3(self.Cx2, self.Cx7)#[1, 16, 16, 128, 128]
        self.x9 = self.up_block4(self.x1, self.x8)#[1, 8, 32, 256, 256]  16
        self.Cx9 = self.Cup_block4(self.Cx1, self.Cx8)#[1, 8, 32, 256, 256]
        out = self.out_conv1(self.x9)#[1, 1, 32, 256, 256]
        Cout = self.Cout_conv1(self.Cx9)#[1, 1, 32, 256, 256]
        return out,Cout




