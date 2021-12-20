###########################################################################
# Created by: CASIA IVA
# Email: jliu@nlpr.ia.ac.cn
# Copyright (c) 2018
###########################################################################
import pdb
import numpy as np
import torch
import torch.nn as nn
import math
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F
from torch.autograd import Variable
torch_ver = torch.__version__[:3]

__all__ = ['PAM_Module', 'CAM_Module']



"https://github.com/13952522076/SPANet/blob/master/models/cp_spp_se_resnet.py"
"""
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
 AdaptiveAvgPool2d-1             [-1, 64, 1, 1]               0
 AdaptiveAvgPool2d-2             [-1, 64, 2, 2]               0
 AdaptiveAvgPool2d-3             [-1, 64, 4, 4]               0
            Linear-4                    [-1, 4]           5,376
              ReLU-5                    [-1, 4]               0
            Linear-6                   [-1, 64]             256
           Sigmoid-7                   [-1, 64]               0
================================================================
Total params: 5,632
Trainable params: 5,632
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 16.00
Forward/backward pass size (MB): 0.01
Params size (MB): 0.02
Estimated Total Size (MB): 16.03
----------------------------------------------------------------
"""
class CPSPPSELayer(nn.Module):
    def __init__(self,in_channel, channel, reduction=16):
        super(CPSPPSELayer, self).__init__()
        if in_channel != channel:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channel, channel, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True)
            )
        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool2 = nn.AdaptiveAvgPool2d(2)
        self.avg_pool4 = nn.AdaptiveAvgPool2d(4)
        self.fc = nn.Sequential(
            nn.Linear(channel*21, channel // reduction, bias=False),
            nn.ReLU(inplace=True ),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.conv1(x) if hasattr(self, 'conv1') else x
        b, c, _, _ = y.size() # b: number; c: channel;
        y1 = self.avg_pool1(y).view(b, c)  # like resize() in numpy
        y2 = self.avg_pool2(y).view(b, 4 * c)
        y3 = self.avg_pool4(y).view(b, 16 * c)
        y = torch.cat((y1, y2, y3), 1)
        y = self.fc(y)
        b,out_channel = y.size()
        y = y.view(b, out_channel, 1, 1)
        return x * y 





"""
https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
Squeeze-and-Excitation Networks
"""
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py
# ref from CBAM
"""
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1          [-1, 1, 256, 256]              98
           Sigmoid-2          [-1, 1, 256, 256]               0
================================================================
Total params: 98
Trainable params: 98
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 16.00
Forward/backward pass size (MB): 1.00
Params size (MB): 0.00
Estimated Total Size (MB): 17.00
----------------------------------------------------------------
"""
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)




class PAM_Module(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key) # (B, H*W, H*W)
        attention = self.softmax(energy)  # (B, H*W, H*W)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class CAM_Module(Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out



# Adapted from https://github.com/yulunzhang/RCAN/blob/master/RCAN_TrainCode/code/model/rcan.py
## Channel Attention (CA) Layer
# Paper: Image Super-Resolution Using Very Deep Residual Channel Attention Networks
"""
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
 AdaptiveAvgPool2d-1             [-1, 64, 1, 1]               0
            Conv2d-2              [-1, 4, 1, 1]             260
              ReLU-3              [-1, 4, 1, 1]               0
            Conv2d-4             [-1, 64, 1, 1]             320
           Sigmoid-5             [-1, 64, 1, 1]               0
================================================================
Total params: 580
Trainable params: 580
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 16.00
Forward/backward pass size (MB): 0.00
Params size (MB): 0.00
Estimated Total Size (MB): 16.00
----------------------------------------------------------------
"""
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class MyChannelAttention(nn.Module):
    def __init__(self,in_channel, channel, reduction=16):
        super(MyChannelAttention, self).__init__()
        if in_channel != channel:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channel, channel, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True)
            )
        self.wa = SpatialAttention()
        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool2 = nn.AdaptiveAvgPool2d(2)
        self.avg_pool4 = nn.AdaptiveAvgPool2d(4)
        self.fc = nn.Sequential(
            nn.Linear(channel*21, channel // reduction, bias=False),
            nn.ReLU(inplace=True ),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.conv1(x) if hasattr(self, 'conv1') else x
        y = self.wa(y) * y
        b, c, _, _ = y.size() # b: number; c: channel;
        y1 = self.avg_pool1(y).view(b, c)  # like resize() in numpy
        y2 = self.avg_pool2(y).view(b, 4 * c)
        y3 = self.avg_pool4(y).view(b, 16 * c)
        y = torch.cat((y1, y2, y3), 1)
        y = self.fc(y)
        b,out_channel = y.size()
        y = y.view(b, out_channel, 1, 1)
        return y


# 我自己根据压缩理论构建的

class MyCA(nn.Module):
    def __init__(self, inchannels, reduction=4):
        super().__init__()
        self.reduction = reduction
        self.reduction_channels = inchannels // reduction
        self.q = nn.Sequential(nn.Conv2d(inchannels, self.reduction_channels, kernel_size=1),
                               nn.BatchNorm2d(self.reduction_channels),
                               nn.PReLU()
                               )
        self.k = nn.Sequential(nn.Conv2d(inchannels, self.reduction_channels, kernel_size=1),
                               nn.BatchNorm2d(self.reduction_channels),
                               nn.PReLU()
                               )
        self.v = nn.Sequential(nn.Conv2d(inchannels, self.reduction_channels, kernel_size=1),
                               nn.PReLU()
                               )
        self.inverse_mapping = nn.Sequential(nn.Conv2d(self.reduction_channels, inchannels, kernel_size=1),
                                             nn.PReLU())
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.q(x)
        k = self.k(x).view(B, self.reduction_channels, -1).permute(0, 2, 1)
        alpha = torch.bmm(q.view(B, self.reduction_channels, -1), k)  #[B, R, R]
        attention = self.softmax(alpha)
        v = self.v(x).view(B, self.reduction_channels, -1) #[B, R, HW]
        y = torch.bmm(attention, v).view(B, self.reduction_channels, H, W)
        y = self.inverse_mapping(y)
        return y


"""
inChannels=64; parameters=37185
"""
class LearnedConvC(nn.Module):
    def __init__(self, inChannels, reduction=4):
        super().__init__()
        self.metaConv = nn.Sequential(nn.Conv2d(inChannels, inChannels, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(inChannels),
                                      nn.PReLU())

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(inChannels, inChannels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(inChannels // reduction, inChannels, bias=False),
            nn.Sigmoid()
        )


    def forward(self, x):
        B, C, H, W = x.shape
        metaConv = self.metaConv(x)
        y = metaConv * metaConv 
        y = self.avg_pool(y).view(B, -1)
        y = self.fc(y).view(B, C, 1, 1)
        x = y * x 
        return x 


class MyWA(nn.Module):
    def __init__(self, inchannels, reduction=4):
        super().__init__()
        self.downsampling1 = nn.Sequential(nn.Conv2d(inchannels, inchannels, kernel_size=3, stride=2),
                                          nn.BatchNorm2d(inchannels))
        self.downsampling2 = nn.Sequential(nn.Conv2d(inchannels, inchannels, kernel_size=3, stride=2))
        self.encoder = nn.Sequential(nn.Conv2d(inchannels, inchannels, kernel_size=1),
                                     nn.PReLU())
        self.sigmoid = nn.Sigmoid()

        self.up_conv2 = nn.Sequential(nn.Conv2d(inchannels, inchannels, kernel_size=3))

    def forward(self, x):
        d_x = self.downsampling1(x)
        d_x = self.encoder(d_x)
        d_attention = self.sigmoid(d_x)
        d_y = self.downsampling2(x)
        d_y = d_y * d_attention
        y = F.interpolate(d_y, x.size()[2:])
        y = self.up_conv2(y)

        return y








# 我设计的MRAB
class MRAB(nn.Module):
    def __init__(self, inchannels, rate=3, WA=True, CA=True):
        super().__init__()
        self.WA = WA
        self.CA = CA
        self.conv1 = nn.Sequential(nn.Conv2d(inchannels, inchannels//4, kernel_size=3, padding=1),
                                   nn.PReLU())
        self.scale_conv1 = nn.Sequential(nn.Conv2d(inchannels, inchannels//4, kernel_size=3, padding=rate, dilation=rate),
                                        nn.PReLU())
        self.scale_conv2 = nn.Sequential(nn.Conv2d(inchannels, inchannels//4, kernel_size=3, padding=2, dilation=2))
        self.scale_conv3 = nn.Sequential(nn.Conv2d(inchannels, inchannels//4, kernel_size=3, padding=5, dilation=5))

        if WA:
            self.spatial_atten1 = MySpatialAttention(r_channels=8, kernel_size=inchannels//2)
            self.spatial_atten2 = MySpatialAttention(r_channels=8, kernel_size=inchannels//2)
            self.spatial_atten3 = MySpatialAttention(r_channels=8, kernel_size=inchannels // 2)
            self.spatial_atten4 = MySpatialAttention(r_channels=8, kernel_size=inchannels // 2)

        if CA:
            self.channel_atten = CPSPPSELayer(inchannels, inchannels, reduction=16)
        self.conv = nn.Sequential(nn.Conv2d(inchannels, inchannels, kernel_size=3, padding=1),
                                  nn.PReLU(),
                                  nn.Conv2d(inchannels, inchannels, kernel_size=3, padding=1),
                                  nn.PReLU()
                                  )

    def forward(self, x):
        # pdb.set_trace()
        #还可以尝试对x进行分割
        g = self.conv1(x)
        if self.WA:
            g = self.spatial_atten1(g)

        h = self.scale_conv1(x)
        if self.WA:
            h = self.spatial_atten2(h)

        e = self.scale_conv2(x)
        if self.WA:
            e = self.spatial_atten3(e)

        i = self.scale_conv3(x)
        if self.WA:
            i = self.spatial_atten4(i)

        h = torch.cat((g, h, e, i), dim=1)
        if self.CA:
            h = self.channel_atten(h)
        h = self.conv(h)
        return h + x

# 我设计的DMRAB
"""
在MRAB的基础上添加恒等映射，以构成类似dense block的效果
"""
class DMRAB1(nn.Module):
    def __init__(self, inchannels, outchannels, rate=3, batch_norm=False):
        super().__init__()
        rate = 3
        self.conv1 = nn.Sequential(nn.Conv2d(inchannels, inchannels//4, kernel_size=3, padding=1),
                                   nn.PReLU())
        self.scale_conv1 = nn.Sequential(nn.Conv2d(inchannels, inchannels//4, kernel_size=3, padding=3, dilation=3),
                                        nn.PReLU())
        self.scale_conv2 = nn.Sequential(nn.Conv2d(inchannels, inchannels//4, kernel_size=3, padding=2, dilation=2),
                                         nn.PReLU())
        self.scale_conv3 = nn.Sequential(nn.Conv2d(inchannels, inchannels//4, kernel_size=1),
                                         nn.PReLU())

        self.spatial_atten1 = MySpatialAttention(r_channels=8, kernel_size=inchannels//2)
        self.spatial_atten2 = MySpatialAttention(r_channels=8, kernel_size=inchannels//2)
        self.spatial_atten3 = MySpatialAttention(r_channels=8, kernel_size=inchannels // 2)
        self.spatial_atten4 = MySpatialAttention(r_channels=8, kernel_size=inchannels // 2)

        self.channel_atten = CPSPPSELayer(inchannels, inchannels, reduction=4)
        self.conv = nn.Sequential(nn.Conv2d(inchannels, inchannels, kernel_size=3, padding=1),
                                  nn.PReLU(),
                                  nn.Conv2d(inchannels, inchannels, kernel_size=3, padding=1)
                                  )

    def forward(self, x):
        # pdb.set_trace()
        #还可以尝试对x进行分割
        g = self.conv1(x)
        g = self.spatial_atten1(g)

        h = self.scale_conv1(x)
        h = self.spatial_atten2(h)

        e = self.scale_conv2(x)
        e = self.spatial_atten3(e)

        i = self.scale_conv3(x)
        i = self.spatial_atten4(i)

        h = torch.cat((g, h, e, i), dim=1)
        h = self.channel_atten(h) * h
        h = self.conv(h)
        return h + x


class DMRAB3(nn.Module):
    def __init__(self, inchannels, outchannels, rate=3, batch_norm=False):
        super().__init__()
        self.conv_in = nn.Sequential(nn.Conv2d(inchannels, inchannels, kernel_size=1), nn.PReLU())

        self.conv1 = nn.Sequential(nn.Conv2d(inchannels//4, inchannels//4, kernel_size=3, padding=1),
                                   nn.PReLU())
        self.scale_conv1 = nn.Sequential(nn.Conv2d(inchannels//4, inchannels//4, kernel_size=3, padding=3, dilation=3),
                                        nn.PReLU())
        self.scale_conv2 = nn.Sequential(nn.Conv2d(inchannels//4, inchannels//4, kernel_size=3, padding=2, dilation=2),
                                         nn.PReLU())
        self.scale_conv3 = nn.Sequential(nn.Conv2d(inchannels//4, inchannels//4, kernel_size=1),
                                         nn.PReLU())

        self.spatial_atten1 = MySpatialAttention(r_channels=8, kernel_size=inchannels//2)
        self.spatial_atten2 = MySpatialAttention(r_channels=8, kernel_size=inchannels//2)
        self.spatial_atten3 = MySpatialAttention(r_channels=8, kernel_size=inchannels // 2)
        self.spatial_atten4 = MySpatialAttention(r_channels=8, kernel_size=inchannels // 2)

        self.channel_atten = CPSPPSELayer(inchannels, inchannels, reduction=4)
        self.conv = nn.Sequential(nn.Conv2d(inchannels, inchannels, kernel_size=3, padding=1),
                                  nn.PReLU(),
                                  nn.Conv2d(inchannels, inchannels, kernel_size=3, padding=1)
                                  )

    def forward(self, x):
        # pdb.set_trace()
        #还可以尝试对x进行分割
        x = self.conv_in(x)

        g, h, e, i = torch.chunk(x, chunks=4, dim=1)
        g = self.conv1(g)
        g = self.spatial_atten1(g)

        h = self.scale_conv1(h)
        h = self.spatial_atten2(h)

        e = self.scale_conv2(e)
        e = self.spatial_atten3(e)

        i = self.scale_conv3(i)
        i = self.spatial_atten4(i)

        h = torch.cat((g, h, e, i), dim=1)
        h = self.channel_atten(h) * h
        h = self.conv(h)
        return h + x


class DMRAB4(nn.Module):
    def __init__(self, inchannels, outchannels, rate=3, batch_norm=False):
        super().__init__()
        self.conv_in = nn.Sequential(nn.Conv2d(inchannels, inchannels, kernel_size=1), nn.PReLU())
        rate = 2
        self.scale_conv1 = nn.Sequential(nn.Conv2d(inchannels//4, inchannels//4, kernel_size=(2*rate-1, 2*rate-1), padding=(2*rate-1)//2),
                                         nn.PReLU(),
                                   nn.Conv2d(inchannels//4, inchannels//4, kernel_size=3, padding=rate, dilation=rate),
                                   nn.PReLU(),
                                   nn.Conv2d(inchannels//4, inchannels//4, kernel_size=1),
                                    nn.PReLU())
        rate += 2
        self.scale_conv2 = nn.Sequential(nn.Conv2d(inchannels//4, inchannels//4, kernel_size=(2*rate-1, 2*rate-1), padding=(2*rate-1)//2),
                                         nn.PReLU(),
                                   nn.Conv2d(inchannels//4, inchannels//4, kernel_size=3, padding=rate, dilation=rate),
                                    nn.PReLU(),
                                   nn.Conv2d(inchannels//4, inchannels//4, kernel_size=1),
                                         nn.PReLU(),)
        rate += 2
        self.scale_conv3 = nn.Sequential(nn.Conv2d(inchannels//4, inchannels//4, kernel_size=(2*rate-1, 2*rate-1), padding=(2*rate-1)//2),
                                         nn.PReLU(),
                                   nn.Conv2d(inchannels//4, inchannels//4, kernel_size=3, padding=rate, dilation=rate),
                                         nn.PReLU(),
                                   nn.Conv2d(inchannels//4, inchannels//4, kernel_size=1),
                                         nn.PReLU(),)
        rate += 2
        self.scale_conv4 = nn.Sequential(nn.Conv2d(inchannels//4, inchannels//4, kernel_size=(2*rate-1, 2*rate-1), padding=(2*rate-1)//2),
                                         nn.PReLU(),
                                   nn.Conv2d(inchannels//4, inchannels//4, kernel_size=3, padding=rate, dilation=rate),
                                         nn.PReLU(),
                                   nn.Conv2d(inchannels//4, inchannels//4, kernel_size=1),
                                         nn.PReLU(),)

        self.spatial_atten1 = MySpatialAttention(r_channels=8, kernel_size=inchannels//2)
        self.spatial_atten2 = MySpatialAttention(r_channels=8, kernel_size=inchannels//2)
        self.spatial_atten3 = MySpatialAttention(r_channels=8, kernel_size=inchannels // 2)
        self.spatial_atten4 = MySpatialAttention(r_channels=8, kernel_size=inchannels // 2)

        self.channel_atten = CPSPPSELayer(inchannels, inchannels, reduction=4)
        self.conv = nn.Sequential(nn.Conv2d(inchannels, inchannels, kernel_size=3, padding=1),
                                  nn.PReLU(),
                                  nn.Conv2d(inchannels, inchannels, kernel_size=3, padding=1)
                                  )

    def forward(self, x):
        # pdb.set_trace()
        #还可以尝试对x进行分割
        x = self.conv_in(x)

        g, h, e, i = torch.chunk(x, chunks=4, dim=1)
        g = self.scale_conv1(g)
        g = self.spatial_atten1(g)

        h = self.scale_conv2(h)
        h = self.spatial_atten2(h)

        e = self.scale_conv3(e)
        e = self.spatial_atten3(e)

        i = self.scale_conv4(i)
        i = self.spatial_atten4(i)

        h = torch.cat((g, h, e, i), dim=1)
        h = self.channel_atten(h) * h
        h = self.conv(h)
        return h + x

class DMRAB5(nn.Module):
    def __init__(self, inchannels, outchannels, rate=3, batch_norm=False):
        super().__init__()
        self.conv_in = nn.Sequential(nn.Conv2d(inchannels, inchannels, kernel_size=1), nn.PReLU())
        rate = 2
        self.scale_conv1 = nn.Sequential(nn.Conv2d(inchannels//4, inchannels//4, kernel_size=(2*rate-1, 2*rate-1), padding=(2*rate-1)//2),
                                   nn.Conv2d(inchannels//4, inchannels//4, kernel_size=3, padding=rate, dilation=rate),
                                   nn.Conv2d(inchannels//4, inchannels//4, kernel_size=1))
        rate += 2
        self.scale_conv2 = nn.Sequential(nn.Conv2d(inchannels//4, inchannels//4, kernel_size=(2*rate-1, 2*rate-1), padding=(2*rate-1)//2),
                                   nn.Conv2d(inchannels//4, inchannels//4, kernel_size=3, padding=rate, dilation=rate),
                                   nn.Conv2d(inchannels//4, inchannels//4, kernel_size=1))
        rate += 2
        self.scale_conv3 = nn.Sequential(nn.Conv2d(inchannels//4, inchannels//4, kernel_size=(2*rate-1, 2*rate-1), padding=(2*rate-1)//2),
                                   nn.Conv2d(inchannels//4, inchannels//4, kernel_size=3, padding=rate, dilation=rate),
                                   nn.Conv2d(inchannels//4, inchannels//4, kernel_size=1))
        rate += 2
        self.scale_conv4 = nn.Sequential(nn.Conv2d(inchannels//4, inchannels//4, kernel_size=(2*rate-1, 2*rate-1), padding=(2*rate-1)//2),
                                   nn.Conv2d(inchannels//4, inchannels//4, kernel_size=3, padding=rate, dilation=rate),
                                   nn.Conv2d(inchannels//4, inchannels//4, kernel_size=1))

        self.spatial_atten1 = MySpatialAttention(r_channels=8, kernel_size=inchannels//2)
        self.spatial_atten2 = MySpatialAttention(r_channels=8, kernel_size=inchannels//2)
        self.spatial_atten3 = MySpatialAttention(r_channels=8, kernel_size=inchannels // 2)
        self.spatial_atten4 = MySpatialAttention(r_channels=8, kernel_size=inchannels // 2)

        self.channel_atten = CPSPPSELayer(inchannels, inchannels, reduction=4)
        # self.conv = nn.Sequential(nn.Conv2d(inchannels, inchannels, kernel_size=3, padding=1),
        #                           nn.PReLU(),
        #                           nn.Conv2d(inchannels, inchannels, kernel_size=3, padding=1),
        #                           nn.PReLU()
        #                           )

    def forward(self, x):
        # pdb.set_trace()
        #还可以尝试对x进行分割
        x = self.conv_in(x)

        g, h, e, i = torch.chunk(x, chunks=4, dim=1)
        g = self.scale_conv1(g)
        g = self.spatial_atten1(g)

        h = self.scale_conv2(h)
        h = self.spatial_atten2(h)

        e = self.scale_conv3(e)
        e = self.spatial_atten3(e)

        i = self.scale_conv4(i)
        i = self.spatial_atten4(i)

        h = torch.cat((g, h, e, i), dim=1)
        h = self.channel_atten(h) * h

        return h + x
        # h = self.conv(h)
        # return h + x

class DMRAB6(nn.Module):
    def __init__(self, inchannels, outchannels, rate=3, batch_norm=False):
        super().__init__()
        self.conv_in = nn.Sequential(nn.Conv2d(inchannels, inchannels, kernel_size=1), nn.PReLU())
        # pdb.set_trace()

        self.scale_conv1 = nn.Sequential(nn.Conv2d(inchannels//4, inchannels//4, kernel_size=(2*rate-1, 2*rate-1), padding=(2*rate-1)//2),
                                   nn.Conv2d(inchannels//4, inchannels//4, kernel_size=3, padding=rate, dilation=rate),
                                   nn.Conv2d(inchannels//4, inchannels//4, kernel_size=1))
        rate += 2
        self.scale_conv2 = nn.Sequential(nn.Conv2d(inchannels//4, inchannels//4, kernel_size=(2*rate-1, 2*rate-1), padding=(2*rate-1)//2),
                                   nn.Conv2d(inchannels//4, inchannels//4, kernel_size=3, padding=rate, dilation=rate),
                                   nn.Conv2d(inchannels//4, inchannels//4, kernel_size=1))
        rate += 2
        self.scale_conv3 = nn.Sequential(nn.Conv2d(inchannels//4, inchannels//4, kernel_size=(2*rate-1, 2*rate-1), padding=(2*rate-1)//2),
                                   nn.Conv2d(inchannels//4, inchannels//4, kernel_size=3, padding=rate, dilation=rate),
                                   nn.Conv2d(inchannels//4, inchannels//4, kernel_size=1))
        rate += 2
        self.scale_conv4 = nn.Sequential(nn.Conv2d(inchannels//4, inchannels//4, kernel_size=(2*rate-1, 2*rate-1), padding=(2*rate-1)//2),
                                   nn.Conv2d(inchannels//4, inchannels//4, kernel_size=3, padding=rate, dilation=rate),
                                   nn.Conv2d(inchannels//4, inchannels//4, kernel_size=1))

        self.spatial_atten1 = MySpatialAttention(r_channels=8, kernel_size=inchannels//2)
        self.spatial_atten2 = MySpatialAttention(r_channels=8, kernel_size=inchannels//2)
        self.spatial_atten3 = MySpatialAttention(r_channels=8, kernel_size=inchannels // 2)
        self.spatial_atten4 = MySpatialAttention(r_channels=8, kernel_size=inchannels // 2)

        self.channel_atten = CPSPPSELayer(inchannels, inchannels, reduction=4)
        # self.conv = nn.Sequential(nn.Conv2d(inchannels, inchannels, kernel_size=3, padding=1),
        #                           nn.PReLU(),
        #                           nn.Conv2d(inchannels, inchannels, kernel_size=3, padding=1),
        #                           nn.PReLU()
        #                           )

    def forward(self, x):
        # pdb.set_trace()
        #还可以尝试对x进行分割
        x = self.conv_in(x)

        g, h, e, i = torch.chunk(x, chunks=4, dim=1)
        g = self.scale_conv1(g)
        g = self.spatial_atten1(g)

        h = self.scale_conv2(h)
        h = self.spatial_atten2(h)

        e = self.scale_conv3(e)
        e = self.spatial_atten3(e)

        i = self.scale_conv4(i)
        i = self.spatial_atten4(i)

        h = torch.cat((g, h, e, i), dim=1)
        h = self.channel_atten(h) * h

        return h + x
        # h = self.conv(h)
        # return h + x


class DMRAB7(nn.Module):
    def __init__(self, inchannels, outchannels, rate=3, batch_norm=False):
        super().__init__()
        rate = 3
        # self.conv1 = nn.Sequential(nn.Conv2d(inchannels, inchannels//4, kernel_size=3, padding=1),
        #                            nn.PReLU())
        self.conv_in = nn.Sequential(nn.Conv2d(inchannels, inchannels, kernel_size=1),
                                     nn.PReLU())
        self.scale_conv1 = nn.Sequential(nn.Conv2d(inchannels//4, inchannels // 2, kernel_size=3, padding=3, dilation=3),
                                        nn.PReLU())
        self.scale_conv2 = nn.Sequential(nn.Conv2d(inchannels//4, inchannels // 2, kernel_size=3, padding=2, dilation=2),
                                         nn.PReLU())
        self.scale_conv3 = nn.Sequential(nn.Conv2d(inchannels//4, inchannels // 2, kernel_size=1),
                                         nn.PReLU())
        self.scale_conv4 = nn.Sequential(nn.Conv2d(inchannels//4, inchannels // 2, kernel_size=3, padding=1),
                                   nn.PReLU())

        self.conv_cat = nn.Sequential(nn.Conv2d(inchannels * 2, inchannels, kernel_size=1),
                                      nn.PReLU())

        self.spatial_atten1 = MySpatialAttention(r_channels=8, kernel_size=inchannels//2)
        self.spatial_atten2 = MySpatialAttention(r_channels=8, kernel_size=inchannels//2)
        self.spatial_atten3 = MySpatialAttention(r_channels=8, kernel_size=inchannels // 2)
        self.spatial_atten4 = MySpatialAttention(r_channels=8, kernel_size=inchannels // 2)

        self.channel_atten = CPSPPSELayer(inchannels, inchannels, reduction=4)
        self.conv = nn.Sequential(nn.Conv2d(inchannels, inchannels, kernel_size=3, padding=1),
                                  nn.PReLU(),
                                  nn.Conv2d(inchannels, inchannels, kernel_size=3, padding=1)
                                  )
        self.activate = nn.PReLU()

    def forward(self, x):
        # pdb.set_trace()
        #还可以尝试对x进行分割
        x = self.conv_in(x)

        g, h, e, i = torch.chunk(x, chunks=4, dim=1)

        h = self.scale_conv1(h)
        h = self.spatial_atten2(h)

        e = self.scale_conv2(g)
        e = self.spatial_atten3(e)

        i = self.scale_conv3(i)
        i = self.spatial_atten4(i)

        g = self.scale_conv4(g)
        g = self.spatial_atten1(g)

        h = torch.cat((h, e, i, g), dim=1)

        h = self.conv_cat(h)

        h = self.channel_atten(h) * h
        h = self.conv(h)
        return self.activate(h + x)










class MyRAB(nn.Module):

    def __init__(self, inlayer, outlayer, stride=1, batch_norm=False):
        super(MyRAB, self).__init__()

        if batch_norm:
            self.conv_block1 = nn.Sequential(
                nn.Conv2d(inlayer, outlayer, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(outlayer),
                nn.ReLU(inplace=True)
            )

            self.conv_block2 = nn.Sequential(
                nn.Conv2d(outlayer, outlayer, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(outlayer)
            )

        else:
            self.conv_block1 = nn.Sequential(
                nn.Conv2d(inlayer, outlayer, kernel_size=3, stride=stride, padding=1),
                nn.ReLU(inplace=True)
            )

            self.conv_block2 = nn.Sequential(
                nn.Conv2d(outlayer, outlayer, kernel_size=3, stride=stride, padding=1),
            )

        # self.CA = DictionaryCAttention(inlayer, reduction=4)
        # self.CA = DirectionAvg1(inlayer, reduction=4)
        # self.WA = MySpatialAttention(r_channels=inlayer, kernel_size=outlayer // 2)


        self.relu = nn.ReLU()

    def forward(self, x):

        residual = x

        x = self.conv_block1(x)
        x = self.conv_block2(x)

        # x = self.CA(x) * x
        # x = self.WA(x)

        x = x + residual

        # Not specified in the paper
        x = self.relu(x)

        return x





#我自己设计的 RIR
class RIR(nn.Module):
    def __init__(self, inchannels, rate, block_num=3):
        super().__init__()
        # self.RIRS = nn.ModuleList([MRAB(inchannels, rate, WA=True, CA=True) for i in range(block_num)])
        self.RIRS = nn.ModuleList([DMRAB7(inchannels, i % 2 + 1) for i in range(block_num)])

    def forward(self, x):
        input = x
        for i, _ in enumerate(self.RIRS):
            x = self.RIRS[i](x)
        return input + x



if __name__ == '__main__':
    # model = MRAB(64, 3).to('cuda')
    # model = CALayer(64).to('cuda')
    # model = MySpatialAttention(kernel_size=64)
    # model = DirectionAvg3(64)
    # model = SpatialAttention()
    # model = PAM_Module(64)
    model = LearnedConvC(64,64)
    # model = CPSPPSELayer(64, 32)

    model.to('cuda')
    # model = DirectionAvg3(24, 8)
    from torchsummary import summary
    # summary(model, input_size=(64, 256, 256))
    summary(model, input_size=(64, 256, 256))
    # model = PAM_Module(24)
    # x = torch.randn(8, 64, 256, 256).to('cuda')
    # y = model(x)
    # print("y.size:", y.size())