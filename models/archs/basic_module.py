import torch
import torch.nn as nn


def conv3x3(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

def conv1x1(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation()

        # self.block = nn.Identity()
        self.block = nn.Sequential(
            conv3x3(in_channels, out_channels),
            activation(),
            conv3x3(in_channels, out_channels),
        )
        if self.should_apply_shortcut:
            self.shortcut = nn.Sequential(
                conv1x1(in_channels, out_channels),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels

    def forward(self, x):
        x = self.block(x) + self.shortcut(x)
        return self.activation(x)



class ResBlockBN(ResBlock):
    def __init__(self, in_channles, out_channels, activation=nn.ReLU):
        super().__init__(in_channles, out_channels, activation=activation)
        self.block = nn.Sequential(
            conv3x3(in_channles, out_channels),
            nn.BatchNorm2d(out_channels),
            activation(),
            conv3x3(out_channels, out_channels),
            nn.BatchNorm2d(out_channels)
        )

class ResBlockLN(ResBlock):
    def __init__(self, in_channles, out_channels, activation=nn.ReLU, **kwargs):
        super().__init__(in_channles, out_channels, activation=activation)
        self.block = nn.Sequential(
            conv3x3(in_channles, out_channels),
            nn.LayerNorm(**kwargs),
            activation(),
            conv3x3(out_channels, out_channels),
            nn.LayerNorm(**kwargs)
        )

### FFA-Net
class NoconvexSelfLearnedFusion(nn.Module):
    def __init__(self, feat_planes=64, feat_nums=2,  ca_module=None, reduction=16):
        super().__init__()
        self.feat_nums = feat_nums
        if ca_module is None:
            self.ca_module = nn.Sequential(*[
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(feat_planes * feat_nums, feat_planes // reduction, 1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.dim // reduction, feat_planes * feat_nums, 1, padding=0, bias=True),
                nn.Sigmoid()
            ])
        else:
            self.ca_module = ca_module

    def forward(self, *x):
        assert len(x) == self.feat_nums, "The number of the features fusioned dosen't match self.feat_nums {}".format(self.feat_nums)
        feat_concat = torch.cat(x, dim=1)
        B, C, H, W = feat_concat.shape
        weights = self.ca_module(feat_concat)
        feat_concat = (feat_concat * weights).view(B, self.feat_nums, C // self.feat_nums, H, W)
        return torch.sum(feat_concat, dim=1)

### GDN
class NoconvexLearnedFusion(nn.Module):
    def __init__(self, feat_nums=2,  ca_module=None, reduction=16):
        super().__init__()
        self.feat_nums = feat_nums
        self.weights = nn.Parameters(torch.ones((1, self.feat_nums, 1, 1, 1), requires_grad=True))  # 1, feat_nums, 1, 1, 1

    def forward(self, *x):
        assert len(x) == self.feat_nums, "The number of the features fusioned dosen't match self.feat_nums {}".format(self.feat_nums)
        feat_stack = torch.stack(x, dim=1) # B, feat_nums, C, H, W
        feat_stack = feat_stack * self.weights
        return torch.sum(feat_stack, dim=1)


### Contrastive Learning for Compact Single Image Dehazing
class ConvexLearnedFusion(nn.Module):
    def __init__(self, ca_module=None, reduction=16):
        super().__init__()
        # self.feat_nums = feat_nums
        self.weights = nn.Parameters(torch.ones((1, 1, 1, 1), requires_grad=True))  # 1, 1, 1, 1

    def forward(self, *x):
        assert len(x) == 2, "The number of the features fusioned dosen't match 2 {}".format(self.feat_nums)
        feat_stack = torch.stack(x, dim=1) # B, feat_nums, C, H, W
        feat_stack = feat_stack * self.weights
        return torch.sum(feat_stack, dim=1)


### Dynamic Convolution: Attention Over Convolution Kernels
class ConvexSelfLearnedFusion(nn.Module):
    def __init__(self, feat_planes=64, feat_nums=2,  ca_module=None, reduction=16):
        super().__init__()
        self.feat_nums = feat_nums
        if ca_module is None:
            self.ca_module = nn.Sequential(*[
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(feat_planes * feat_nums, feat_planes // reduction, 1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.dim // reduction, feat_planes * feat_nums, 1, padding=0, bias=True),
                nn.Softmax()
            ])
        else:
            self.ca_module = ca_module

    def forward(self, *x):
        assert len(x) == self.feat_nums, "The number of the features fusioned dosen't match self.feat_nums {}".format(self.feat_nums)
        feat_concat = torch.cat(x, dim=1)
        B, C, H, W = feat_concat.shape
        weights = self.ca_module(feat_concat)
        feat_concat = (feat_concat * weights).view(B, self.feat_nums, C // self.feat_nums, H, W)
        return torch.sum(feat_concat, dim=1)



