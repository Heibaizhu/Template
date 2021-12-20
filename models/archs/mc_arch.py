import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .basic_module import *

# --- Build dense --- #
class MakeDense(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size=3):
        super(MakeDense, self).__init__()
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=kernel_size, padding=(kernel_size-1)//2)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


# --- Build the Residual Dense Block --- #
class RDB(nn.Module):
    def __init__(self, in_channels, num_dense_layer, growth_rate):
        """

        :param in_channels: input channel size
        :param num_dense_layer: the number of RDB layers
        :param growth_rate: growth_rate
        """
        super(RDB, self).__init__()
        _in_channels = in_channels
        modules = []
        for i in range(num_dense_layer):
            modules.append(MakeDense(_in_channels, growth_rate))
            _in_channels += growth_rate
        self.residual_dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(_in_channels, in_channels, kernel_size=1, padding=0)

    def forward(self, x):
        out = self.residual_dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out


class ToyNet(nn.Module):
    def __init__(self, pos_dim=1, **kwargs):
        super().__init__()
        pi = 3.1415
        coeff = torch.tensor([2 ** i * pi for i in range(pos_dim)]).view(1, pos_dim, 1, 1, 1)
        self.register_buffer('coeff', coeff)
        # self.L = 10
        planes = 6 * pos_dim

        self.gate = nn.Conv2d(planes, planes, kernel_size=5, padding=2)


        self.conv1 = nn.Sequential(nn.Conv2d(planes, 2 * planes, kernel_size=3, padding=1),
                                   nn.LayerNorm([2 * planes, 240, 240], elementwise_affine=False),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(2 * planes, planes * 4, kernel_size=3, padding=1),
                                   nn.LayerNorm([4 * planes, 240, 240], elementwise_affine=False),
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(planes * 4, planes * 6, kernel_size=3, padding=1),
                                   nn.LayerNorm([6 * planes, 240, 240], elementwise_affine=False),
                                   nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(planes * 6, planes * 8, kernel_size=3, padding=1),
                                   nn.LayerNorm([8 * planes, 240, 240], elementwise_affine=False),
                                   nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(planes * 8, planes * 4, kernel_size=3, padding=1),
                                   nn.LayerNorm([4 * planes, 240, 240], elementwise_affine=False),
                                   nn.ReLU())
        self.conv6 = nn.Sequential(nn.Conv2d(planes * 4, planes * 2, kernel_size=3, padding=1),
                                   nn.LayerNorm([2 * planes, 240, 240], elementwise_affine=False),
                                   nn.ReLU())
        self.conv7 = nn.Sequential(nn.Conv2d(planes * 2, 3, kernel_size=3, padding=1),
                                   nn.ReLU())


    def log_grad(self, tb_logger):
        def make_hook(tag):
            def hook_fn(grad):
                tb_logger.add_histogram(tag=tag, values=grad)
            return hook_fn

        self.conv7[0].weight.register_hook(make_hook(tag='conv7'))
        self.gate.weight.register_hook(make_hook(tag='gate'))


    def forward(self, x):
        #pos encoding
        B, C, H, W = x.shape
        x = x.view(B, 1, C, H, W)
        phase = x * self.coeff # (B, pos_dim, C, H, W)
        x = (torch.cat((torch.sin(phase), torch.cos(phase)), dim=1) * x).view(B, -1, H, W)

        x = self.gate(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        return {'dehazed': [x]}


class AODnet(nn.Module):
    def __init__(self, **kwargs):
        super(AODnet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=7, stride=1, padding=3)
        self.conv5 = nn.Conv2d(in_channels=12, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.b = 1


    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        cat1 = torch.cat((x1, x2), 1)
        x3 = F.relu(self.conv3(cat1))
        cat2 = torch.cat((x2, x3), 1)
        x4 = F.relu(self.conv4(cat2))
        cat3 = torch.cat((x1, x2, x3, x4), 1)
        k = F.relu(self.conv5(cat3))

        if k.size() != x.size():
            raise Exception("k, haze image are different size!")

        output = k * x - k + self.b
        return {'dehazed': [F.relu(output)]}

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, inplanes=3, outplanes=3, midplanes=64, bilinear=True):
        super(UNet, self).__init__()


        self.inc = DoubleConv(inplanes, midplanes)
        self.down1 = Down(midplanes, midplanes * 2)
        self.down2 = Down(midplanes * 2, midplanes * 4)
        self.down3 = Down(midplanes * 4, midplanes * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(midplanes * 8, midplanes * 16 // factor)
        self.up1 = Up(midplanes * 16, midplanes * 8 // factor, bilinear)
        self.up2 = Up(midplanes * 8, midplanes * 4 // factor, bilinear)
        self.up3 = Up(midplanes * 4, midplanes * 2 // factor, bilinear)
        self.up4 = Up(midplanes * 2, midplanes, bilinear)
        self.outc = OutConv(midplanes, outplanes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)

        return {'dehazed': x}

class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,
                                bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers - 2):
            layers.append(
                nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                          bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding,
                                bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return out

# --- Downsampling block in GridDehazeNet  --- #
class DownSample(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=2):
        super(DownSample, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=(kernel_size-1)//2)
        self.conv2 = nn.Conv2d(in_channels, stride*in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        return out


# --- Upsampling block in GridDehazeNet  --- #
class UpSample(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=2):
        super(UpSample, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, in_channels, kernel_size, stride=stride, padding=1)
        self.conv = nn.Conv2d(in_channels, in_channels // stride, kernel_size, stride=1, padding=(kernel_size - 1) // 2)

    def forward(self, x, output_size):
        out = F.relu(self.deconv(x, output_size=output_size))
        out = F.relu(self.conv(out))
        return out


# --- Main model  --- #
class GridDehazeNet(nn.Module):
    def __init__(self, in_channels=3, depth_rate=16, kernel_size=3, stride=2, height=3, width=6, num_dense_layer=4, growth_rate=16, attention=True):
        super(GridDehazeNet, self).__init__()
        self.rdb_module = nn.ModuleDict()
        self.upsample_module = nn.ModuleDict()
        self.downsample_module = nn.ModuleDict()
        self.height = height
        self.width = width
        self.stride = stride
        self.depth_rate = depth_rate
        self.coefficient = nn.Parameter(torch.Tensor(np.ones((height, width, 2, depth_rate*stride**(height-1)))), requires_grad=attention)
        self.conv_in = nn.Conv2d(in_channels, depth_rate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.conv_out = nn.Conv2d(depth_rate, in_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.rdb_in = RDB(depth_rate, num_dense_layer, growth_rate)
        self.rdb_out = RDB(depth_rate, num_dense_layer, growth_rate)

        rdb_in_channels = depth_rate
        for i in range(height):
            for j in range(width - 1):
                self.rdb_module.update({'{}_{}'.format(i, j): RDB(rdb_in_channels, num_dense_layer, growth_rate)})
            rdb_in_channels *= stride

        _in_channels = depth_rate
        for i in range(height - 1):
            for j in range(width // 2):
                self.downsample_module.update({'{}_{}'.format(i, j): DownSample(_in_channels)})
            _in_channels *= stride

        for i in range(height - 2, -1, -1):
            for j in range(width // 2, width):
                self.upsample_module.update({'{}_{}'.format(i, j): UpSample(_in_channels)})
            _in_channels //= stride

    def forward(self, x):
        inp = self.conv_in(x)

        x_index = [[0 for _ in range(self.width)] for _ in range(self.height)]
        i, j = 0, 0

        x_index[0][0] = self.rdb_in(inp)

        for j in range(1, self.width // 2):
            x_index[0][j] = self.rdb_module['{}_{}'.format(0, j-1)](x_index[0][j-1])

        for i in range(1, self.height):
            x_index[i][0] = self.downsample_module['{}_{}'.format(i-1, 0)](x_index[i-1][0])

        for i in range(1, self.height):
            for j in range(1, self.width // 2):
                channel_num = int(2**(i-1)*self.stride*self.depth_rate)
                x_index[i][j] = self.coefficient[i, j, 0, :channel_num][None, :, None, None] * self.rdb_module['{}_{}'.format(i, j-1)](x_index[i][j-1]) + \
                                self.coefficient[i, j, 1, :channel_num][None, :, None, None] * self.downsample_module['{}_{}'.format(i-1, j)](x_index[i-1][j])

        x_index[i][j+1] = self.rdb_module['{}_{}'.format(i, j)](x_index[i][j])
        k = j

        for j in range(self.width // 2 + 1, self.width):
            x_index[i][j] = self.rdb_module['{}_{}'.format(i, j-1)](x_index[i][j-1])

        for i in range(self.height - 2, -1, -1):
            channel_num = int(2 ** (i-1) * self.stride * self.depth_rate)
            x_index[i][k+1] = self.coefficient[i, k+1, 0, :channel_num][None, :, None, None] * self.rdb_module['{}_{}'.format(i, k)](x_index[i][k]) + \
                              self.coefficient[i, k+1, 1, :channel_num][None, :, None, None] * self.upsample_module['{}_{}'.format(i, k+1)](x_index[i+1][k+1], x_index[i][k].size())

        for i in range(self.height - 2, -1, -1):
            for j in range(self.width // 2 + 1, self.width):
                channel_num = int(2 ** (i - 1) * self.stride * self.depth_rate)
                x_index[i][j] = self.coefficient[i, j, 0, :channel_num][None, :, None, None] * self.rdb_module['{}_{}'.format(i, j-1)](x_index[i][j-1]) + \
                                self.coefficient[i, j, 1, :channel_num][None, :, None, None] * self.upsample_module['{}_{}'.format(i, j)](x_index[i+1][j], x_index[i][j-1].size())

        out = self.rdb_out(x_index[i][j])
        out = F.relu(self.conv_out(out))

        return {'dehazed': out}


if __name__ == '__main__':
    pass