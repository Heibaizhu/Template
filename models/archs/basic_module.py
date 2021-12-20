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
