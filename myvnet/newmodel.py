import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from functools import partial

nonlinearity = partial(F.relu, inplace=True)


class DACblock(nn.Module):
    def __init__(self, channel):
        super(DACblock, self).__init__()
        self.dilate1 = nn.Conv3d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv3d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv3d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv3d(channel, channel, kernel_size=1, dilation=1, padding=0)

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out


class SPPblock(nn.Module):
    def __init__(self, in_channels):
        super(SPPblock, self).__init__()
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=2)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=3)
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 5, 5), stride=5)
        self.pool4 = nn.MaxPool3d(kernel_size=(1, 6, 6), stride=6)

        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x):
        # self.in_channels, h, w = x.size(1), x.size(2), x.size(3)
        self.layer1 = F.upsample(self.conv(self.pool1(x)), size=(1, 8, 8), mode='trilinear')
        self.layer2 = F.upsample(self.conv(self.pool2(x)), size=(1, 8, 8), mode='trilinear')
        self.layer3 = F.upsample(self.conv(self.pool3(x)), size=(1, 8, 8), mode='trilinear')
        self.layer4 = F.upsample(self.conv(self.pool4(x)), size=(1, 8, 8), mode='trilinear')

        out = torch.cat([self.layer1, self.layer2, self.layer3, self.layer4, x], 1)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm3d(in_channels // 4)
        self.relu1 = nonlinearity
        self.deconv2 = nn.ConvTranspose3d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm3d(in_channels // 4)
        self.relu2 = nonlinearity
        self.conv3 = nn.Conv3d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm3d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class residual_first(nn.Module):  # stride =1循环时
    def __init__(self, in_channels, out_channels, strideshape):
        super(residual_first, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)  # 不改变尺寸
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.norm = nn.BatchNorm3d(out_channels)
        self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=strideshape, padding=0)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.conv2(x1)
        x3 = self.norm(x2)
        x4 = self.relu(x3)
        xcut = self.shortcut(x)
        return x4 + xcut


class residual_notfirstxunhuan(nn.Module):  # 不循环时用2 stride = 2
    def __init__(self, in_channels, out_channels, strideshape):
        super(residual_notfirstxunhuan, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)  # 不改变尺寸
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.norm = nn.BatchNorm3d(out_channels)
        self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=strideshape, padding=0)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.norm(x1)
        x3 = self.relu(x2)
        x4 = self.conv2(x3)
        x5 = self.norm(x4)
        x6 = self.relu(x5)
        xcut = self.shortcut(x)
        return x6 + xcut


class conv1layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv1layer, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.norm = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.norm(x1)
        x3 = self.relu(x2)
        return x3


class residual_notfirstnotxunhuan(nn.Module):  # 不循环时用2 stride = 2
    def __init__(self, in_channels, out_channels, strideshape):
        super(residual_notfirstnotxunhuan, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)  # 改变尺寸stride用2
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.norm = nn.BatchNorm3d(out_channels)
        self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=strideshape, padding=0)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.norm(x1)
        x3 = self.relu(x2)
        x4 = self.conv2(x3)
        x5 = self.norm(x4)
        x6 = self.relu(x5)
        xcut = self.shortcut(x)
        return x6 + xcut


class CE_Net(nn.Module):
    def __init__(self):
        super(CE_Net, self).__init__()
        # resnet = models.resnet34(pretrained=True)#res34?

        self.conv1 = conv1layer(1, 16)  # 128
        # self.poo1 = nn.MaxPool3d(conv1)#64
        self.conv2first = residual_first(16, 32, 2)  # 64
        self.conv2xunhuan1 = residual_notfirstxunhuan(32, 32, 1)  # 64
        self.conv2xunhuan2 = residual_notfirstxunhuan(32, 32, 1)  # 64
        self.conv3 = residual_notfirstnotxunhuan(32, 64, 2)  # 32
        self.conv3xunhuan1 = residual_notfirstxunhuan(64, 64, 1)
        self.conv3xunhuan2 = residual_notfirstxunhuan(64, 64, 1)
        self.conv3xunhuan3 = residual_notfirstxunhuan(64, 64, 1)
        self.conv4 = residual_notfirstnotxunhuan(64, 128, 2)  # 16
        self.conv4xunhuan1 = residual_notfirstxunhuan(128, 128, 1)
        self.conv4xunhuan2 = residual_notfirstxunhuan(128, 128, 1)
        self.conv4xunhuan3 = residual_notfirstxunhuan(128, 128, 1)
        self.conv4xunhuan4 = residual_notfirstxunhuan(128, 128, 1)
        self.conv4xunhuan5 = residual_notfirstxunhuan(128, 128, 1)
        self.conv5 = residual_notfirstnotxunhuan(128, 256, 2)  # 8
        self.conv5xunhuan1 = residual_notfirstxunhuan(256, 256, 1)
        self.conv5xunhuan2 = residual_notfirstxunhuan(256, 256, 1)

        self.dblock = DACblock(256)  # 8
        self.spp = SPPblock(256)  # 8

        self.decoder4 = DecoderBlock(260, 128)  # 16
        self.decoder3 = DecoderBlock(128, 64)  # 32
        self.decoder2 = DecoderBlock(64, 32)  # 64
        self.decoder1 = DecoderBlock(32, 16)  # 128

        self.finaldeconv1 = nn.ConvTranspose3d(16, 8, 4, 2, 1)  # 256
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv3d(8, 8, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv3d(8, 1, 3, padding=1)

    def forward(self, x):
        # Encoder
        conv1 = self.conv1(x)  # 128
        pconv21 = self.conv2first(conv1)
        pconv22 = self.conv2xunhuan1(pconv21)
        conv2 = self.conv2xunhuan2(pconv22)  # 64

        pconv3 = self.conv3(conv2)
        pconv32 = self.conv3xunhuan1(pconv3)
        pconv33 = self.conv3xunhuan2(pconv32)
        conv3 = self.conv3xunhuan3(pconv33)  # 32

        pconv41 = self.conv4(conv3)  # 16
        pconv42 = self.conv4xunhuan1(pconv41)
        pconv43 = self.conv4xunhuan2(pconv42)
        pconv44 = self.conv4xunhuan3(pconv43)
        pconv45 = self.conv4xunhuan4(pconv44)
        conv4 = self.conv4xunhuan5(pconv45)

        pconv51 = self.conv5(conv4)  # 8
        pconv52 = self.conv5xunhuan1(pconv51)
        conv5 = self.conv5xunhuan2(pconv52)

        # Center
        e4 = self.dblock(conv5)  # 8
        e4 = self.spp(e4)  # 8

        # Decoder
        d4 = self.decoder4(e4) + conv4  # 16
        d3 = self.decoder3(d4) + conv3  # 32
        d2 = self.decoder2(d3) + conv2  # 64
        d1 = self.decoder1(d2)  # 128

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)


class CE_NetCon(nn.Module):
    def __init__(self):
        super(CE_NetCon, self).__init__()
        # resnet = models.resnet34(pretrained=True)#res34?

        self.Cconv1 = conv1layer(1, 16)  # 128
        # self.poo1 = nn.MaxPool3d(conv1)#64
        self.Cconv2first = residual_first(16, 32, 2)  # 64
        self.Cconv2xunhuan1 = residual_notfirstxunhuan(32, 32, 1)  # 64
        self.Cconv2xunhuan2 = residual_notfirstxunhuan(32, 32, 1)  # 64
        self.Cconv3 = residual_notfirstnotxunhuan(32, 64, 2)  # 32
        self.Cconv3xunhuan1 = residual_notfirstxunhuan(64, 64, 1)
        self.Cconv3xunhuan2 = residual_notfirstxunhuan(64, 64, 1)
        self.Cconv3xunhuan3 = residual_notfirstxunhuan(64, 64, 1)
        self.Cconv4 = residual_notfirstnotxunhuan(64, 128, 2)  # 16
        self.Cconv4xunhuan1 = residual_notfirstxunhuan(128, 128, 1)
        self.Cconv4xunhuan2 = residual_notfirstxunhuan(128, 128, 1)
        self.Cconv4xunhuan3 = residual_notfirstxunhuan(128, 128, 1)
        self.Cconv4xunhuan4 = residual_notfirstxunhuan(128, 128, 1)
        self.Cconv4xunhuan5 = residual_notfirstxunhuan(128, 128, 1)
        self.Cconv5 = residual_notfirstnotxunhuan(128, 256, 2)  # 8
        self.Cconv5xunhuan1 = residual_notfirstxunhuan(256, 256, 1)
        self.Cconv5xunhuan2 = residual_notfirstxunhuan(256, 256, 1)

        self.Cdblock = DACblock(256)  # 8
        self.Cspp = SPPblock(256)  # 8

        self.Cdecoder4 = DecoderBlock(260, 128)  # 16
        self.Cdecoder3 = DecoderBlock(128, 64)  # 32
        self.Cdecoder2 = DecoderBlock(64, 32)  # 64
        self.Cdecoder1 = DecoderBlock(32, 16)  # 128

        self.Cfinaldeconv1 = nn.ConvTranspose3d(16, 8, 4, 2, 1)  # 256
        self.Cfinalrelu1 = nonlinearity
        self.Cfinalconv2 = nn.Conv3d(8, 8, 3, padding=1)
        self.Cfinalrelu2 = nonlinearity
        self.Cfinalconv3 = nn.Conv3d(8, 1, 3, padding=1)

    def forward(self, x):
        # Encoder
        Cconv1 = self.Cconv1(x)  # 128
        Cpconv21 = self.Cconv2first(Cconv1)
        Cpconv22 = self.Cconv2xunhuan1(Cpconv21)
        Cconv2 = self.Cconv2xunhuan2(Cpconv22)  # 64

        Cpconv3 = self.Cconv3(Cconv2)
        Cpconv32 = self.Cconv3xunhuan1(Cpconv3)
        Cpconv33 = self.Cconv3xunhuan2(Cpconv32)
        Cconv3 = self.Cconv3xunhuan3(Cpconv33)  # 32

        Cpconv41 = self.Cconv4(Cconv3)  # 16
        Cpconv42 = self.Cconv4xunhuan1(Cpconv41)
        Cpconv43 = self.Cconv4xunhuan2(Cpconv42)
        Cpconv44 = self.Cconv4xunhuan3(Cpconv43)
        Cpconv45 = self.Cconv4xunhuan4(Cpconv44)
        Cconv4 = self.Cconv4xunhuan5(Cpconv45)

        Cpconv51 = self.Cconv5(Cconv4)  # 8
        Cpconv52 = self.Cconv5xunhuan1(Cpconv51)
        Cconv5 = self.Cconv5xunhuan2(Cpconv52)

        # Center
        Ce4 = self.Cdblock(Cconv5)  # 8
        Ce4 = self.Cspp(Ce4)  # 8

        # Decoder
        Cd4 = self.Cdecoder4(Ce4) + Cconv4  # 16
        Cd3 = self.Cdecoder3(Cd4) + Cconv3  # 32
        Cd2 = self.Cdecoder2(Cd3) + Cconv2  # 64
        Cd1 = self.Cdecoder1(Cd2)  # 128

        Cout = self.Cfinaldeconv1(Cd1)
        Cout = self.Cfinalrelu1(Cout)
        Cout = self.Cfinalconv2(Cout)
        Cout = self.Cfinalrelu2(Cout)
        Cout = self.Cfinalconv3(Cout)

        return F.sigmoid(Cout)


class conv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation_func=nn.ReLU):
        super(conv3d, self).__init__()
        self.conv = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size, padding=1))

    def forward(self, x):
        return self.conv(x)


class conv3d_x3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation_func=nn.ReLU):
        super(conv3d_x3, self).__init__()
        self.conv_1 = conv3d(in_channels, out_channels, kernel_size)
        self.conv_2 = conv3d(out_channels, out_channels, kernel_size)
        self.conv_3 = conv3d(out_channels, out_channels, kernel_size)
        self.relu = activation_func()
        self.norm = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        z_1 = self.conv_1(x)
        z_3 = self.conv_3(self.conv_2(z_1))
        zz = z_1 + z_3
        norm = self.norm(zz)
        return self.relu(norm)


class deconv3d_x3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, activation_func=nn.ReLU):
        super(deconv3d_x3, self).__init__()
        self.up = deconv3d_as_up(in_channels, out_channels, kernel_size, stride)
        self.lhs_conv = conv3d(out_channels // 2, out_channels, kernel_size)
        self.conv_x3 = conv3d_x3(out_channels, out_channels, kernel_size)

    def forward(self, lhs, rhs):
        rhs_up = self.up(rhs)
        lhs_conv = self.lhs_conv(lhs)
        rhs_add = crop(rhs_up, lhs_conv) + lhs_conv
        return self.conv_x3(rhs_add)


def crop(large, small):
    l, s = large.size(), small.size()
    offset = [0, 0, (l[2] - s[2]) // 2, (l[3] - s[3]) // 2, (l[4] - s[4]) // 2]
    return large[..., offset[2]: offset[2] + s[2], offset[3]: offset[3] + s[3], offset[4]: offset[4] + s[4]]


def conv3d_as_pool(in_channels, out_channels, kernel_size=3, stride=2, activation_func=nn.ReLU):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding=1),
        nn.BatchNorm3d(out_channels),
        activation_func())


def deconv3d_as_up(in_channels, out_channels, kernel_size=3, stride=2, activation_func=nn.ReLU):
    return nn.Sequential(
        nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride),
        nn.BatchNorm3d(out_channels),
        activation_func()
    )


def conv3dim(in_channels, out_channels, kernel_size=3, padding=1):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
    )


# mergewithcontour
class MergeNet(nn.Module):
    def __init__(self):
        super(MergeNet, self).__init__()
        self.Vconv_1 = conv3d_x3(1, 8)
        self.Vpool_1 = conv3d_as_pool(8, 16)
        self.Vconv_2 = conv3d_x3(16, 16)
        self.Vpool_2 = conv3d_as_pool(16, 32)
        self.Vconv_3 = conv3d_x3(32, 32)
        self.Vpool_3 = conv3d_as_pool(32, 64)
        self.Vconv_4 = conv3d_x3(64, 64)
        self.Vpool_4 = conv3d_as_pool(64, 128)
        self.Mconv_1 = conv3d_x3(1, 8)
        self.Mpool_1 = conv3d_as_pool(8, 16)
        self.Mconv_2 = conv3d_x3(16, 16)
        self.Mpool_2 = conv3d_as_pool(16, 32)
        self.Mconv_3 = conv3d_x3(32, 32)
        self.Mpool_3 = conv3d_as_pool(32, 64)
        self.Mconv_4 = conv3d_x3(64, 64)
        self.Mpool_4 = conv3d_as_pool(64, 128)

        self.Vbottom = conv3d_x3(128, 128)

        self.Vdeconv_4 = deconv3d_x3(128, 128)
        self.Vdeconv_3 = deconv3d_x3(128, 64)
        self.Vdeconv_2 = deconv3d_x3(64, 32)
        self.Vdeconv_1 = deconv3d_x3(32, 16)
        self.Vout = nn.Sequential(nn.Conv3d(16, 1, kernel_size=1), nn.Sigmoid())

        self.Mdeconv_4 = deconv3d_x3(128, 128)
        self.Mdeconv_3 = deconv3d_x3(128, 64)
        self.Mdeconv_2 = deconv3d_x3(64, 32)
        self.Mdeconv_1 = deconv3d_x3(32, 16)
        self.Mout = nn.Sequential(nn.Conv3d(16, 1, kernel_size=1), nn.Sigmoid())

    def forward(self, x):
        Vconv_1 = self.Vconv_1(x)
        Mconv_1 = self.Mconv_1(x)
        Vpool1 = self.Vpool_1(Vconv_1)
        Mpool1 = self.Mpool_1(Mconv_1)
        Vconv_2 = self.Vconv_2(Vpool1)
        Mconv_2 = self.Mconv_2(Mpool1)
        Vpool2 = self.Vpool_2(Vconv_2)
        Mpool2 = self.Mpool_2(Mconv_2)
        Vconv_3 = self.Vconv_3(Vpool2)
        Mconv_3 = self.Mconv_3(Mpool2)
        Vpool3 = self.Vpool_3(Vconv_3)
        Mpool3 = self.Mpool_3(Mconv_3)
        Vconv_4 = self.Vconv_4(Vpool3)
        Mconv_4 = self.Mconv_4(Mpool3)
        Vpool4 = self.Vpool_4(Vconv_4)
        Mpool4 = self.Mpool_4(Mconv_4)

        Vbottom = self.Vbottom(Vpool4)

        Vdeconv_4 = self.Vdeconv_4(Vconv_4, Vbottom)
        Mdeconv_4 = self.Mdeconv_4(Mconv_4, Vbottom)
        Vdeconv_3 = self.Vdeconv_3(Vconv_3, Vdeconv_4)
        Mdeconv_3 = self.Mdeconv_3(Mconv_3, Mdeconv_4)
        Vdeconv_2 = self.Vdeconv_2(Vconv_2, Vdeconv_3)
        Mdeconv_2 = self.Mdeconv_2(Mconv_2, Mdeconv_3)
        Vdeconv_1 = self.Vdeconv_1(Vconv_1, Vdeconv_2)
        Mdeconv_1 = self.Mdeconv_1(Mconv_1, Mdeconv_2)
        Vout = self.Vout(Vdeconv_1)
        Mout = self.Mout(Mdeconv_1)
        return Vout, Mout


class Xmodule2(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Xmodule2, self).__init__()
        self.conv = nn.Conv3d(input_channels + input_channels, output_channels, kernel_size=1, padding=0)  # 需要down一半
        self.norm = nn.BatchNorm3d(output_channels)
        self.dropout = nn.Dropout3d(p=0.25)  # 25%高斯dropout

    def forward(self, x, y):
        x = torch.cat((x, y), dim=1)
        x1 = self.conv(x)
        x2 = self.norm(x1)
        x3 = self.relu(x2)
        out = self.dropout(x3)
        return out
class changedim(nn.Module):
    def __init__(self,input_channels, output_channels):
        self.conv = nn.Conv3d(input_channels, output_channels, kernel_size=1, padding=0)
    def forward(self, x,y ):
        x = torch.cat((x, y), dim=1)
        x1 = self.conv(x)
        return x1

class MergeNet2(nn.Module):
    def __init__(self):
        super(MergeNet2, self).__init__()
        self.Vconv_1 = conv3d_x3(1, 8)
        self.Vpool_1 = conv3d_as_pool(8, 16)
        self.Vconv_2 = conv3d_x3(16, 16)
        self.Vconv_2M = conv3d_x3(32, 16)
        self.changedim1C = changedim(32,16)
        self.Vpool_2 = conv3d_as_pool(16, 32)
        self.Vconv_3 = conv3d_x3(32, 32)
        self.Vconv_3M = conv3d_x3(64, 32)
        self.changedim2C = changedim(64, 32)
        self.Vpool_3 = conv3d_as_pool(32, 64)
        self.Vconv_4 = conv3d_x3(64, 64)
        self.Vconv_4M = conv3d_x3(128, 64)
        self.changedim3C = changedim(128, 64)
        self.Vpool_4 = conv3d_as_pool(64, 128)
        self.Mconv_1 = conv3d_x3(1, 8)
        self.Mpool_1 = conv3d_as_pool(8, 16)
        self.Mconv_2 = conv3d_x3(16, 16)
        self.Mpool_2 = conv3d_as_pool(16, 32)
        self.Mconv_3 = conv3d_x3(32, 32)
        self.Mpool_3 = conv3d_as_pool(32, 64)
        self.Mconv_4 = conv3d_x3(64, 64)
        self.Mpool_4 = conv3d_as_pool(64, 128)

        self.Vbottom = conv3d_x3(128, 128)

        self.Vdeconv_4 = deconv3d_x3(128, 128)
        self.Vdeconv_3 = deconv3d_x3(128, 64)
        self.Vdeconv_2 = deconv3d_x3(64, 32)
        self.Vdeconv_1 = deconv3d_x3(32, 16)
        self.Vout = nn.Sequential(nn.Conv3d(16, 1, kernel_size=1), nn.Sigmoid())

        self.Mdeconv_4 = deconv3d_x3(128, 128)
        self.Mdeconv_3 = deconv3d_x3(128, 64)
        self.Mdeconv_2 = deconv3d_x3(64, 32)
        self.Mdeconv_1 = deconv3d_x3(32, 16)
        self.Mout = nn.Sequential(nn.Conv3d(16, 1, kernel_size=1), nn.Sigmoid())

    def forward(self, x):
        Vconv_1 = self.Vconv_1(x)
        Mconv_1 = self.Mconv_1(x)
        Vpool1 = self.Vpool_1(Vconv_1)#16
        Mpool1 = self.Mpool_1(Mconv_1)#16

        Module1 = self.xm1(Mpool1,Vpool1)#16
        Vconv_2 = self.Vconv_2(Vpool1)#16
        Vconv_2x2Mo = self.Vconv_2M(torch.cat([Vpool1,Module1], 1))#16
        Mchange1 = self.changedim1C(Mpool1, Module1)#32-16
        Mconv_2 = self.Mconv_2(Mchange1)
        Vpool2 = self.Vpool_2(Vconv_2)
        Vpool2x2Mo = self.Vpool_2(Vconv_2x2Mo)
        Mpool2 = self.Mpool_2(Mconv_2)

        Module2 = self.xm2(Mpool2, Vpool2x2Mo)
        Vconv_3 = self.Vconv_3(Vpool2)
        Vconv_3x3Mo = self.Vconv_3M(torch.cat([Vpool2,Module2], 1))
        Mchange2 = self.changedim2C(Mpool2, Module2)
        Mconv_3 = self.Mconv_3(Mchange2)
        Vpool3 = self.Vpool_3(Vconv_3)
        Vpool3x3Mo = self.Vpool_3(Vconv_3x3Mo)
        Mpool3 = self.Mpool_3(Mconv_3)

        Module3 = self.xm3(Mpool3, Vpool3x3Mo)
        Vconv_4 = self.Vconv_4(Vpool3)
        #Vconv_4x4Mo = torch.cat([Vpool3,Module3], 1)
        Mchange3 = self.changedim3C(Mpool3, Module3)
        Mconv_4 = self.Mconv_4(Mchange3)
        Vpool4 = self.Vpool_4(Vconv_4)
        #Vpool4x4Mo = self.Vpool_3(Vconv_4x4Mo)
        #Mpool4 = self.Mpool_4(Mconv_4)

        Vbottom = self.Vbottom(Vpool4)

        Vdeconv_4 = self.Vdeconv_4(Vconv_4, Vbottom)
        Mdeconv_4 = self.Mdeconv_4(Mconv_4, Vbottom)
        Vdeconv_3 = self.Vdeconv_3(Vconv_3, Vdeconv_4)
        Mdeconv_3 = self.Mdeconv_3(Mconv_3, Mdeconv_4)
        Vdeconv_2 = self.Vdeconv_2(Vconv_2, Vdeconv_3)
        Mdeconv_2 = self.Mdeconv_2(Mconv_2, Mdeconv_3)
        Vdeconv_1 = self.Vdeconv_1(Vconv_1, Vdeconv_2)
        Mdeconv_1 = self.Mdeconv_1(Mconv_1, Mdeconv_2)
        Vout = self.Vout(Vdeconv_1)
        Mout = self.Mout(Mdeconv_1)
        return Vout, Mout