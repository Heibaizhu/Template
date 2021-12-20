import torch
import torch.nn as nn
import torch.nn.functional as F
from .convlstm import ConvLSTMCell
from .attention import CALayer, CPSPPSELayer, LearnedConvC
import pdb

class recursiveModule(nn.Module):  #52054
    def __init__(self, inplanes=16*2):
        super().__init__()

        hiddenChannels = inplanes // 4

        self.gate = nn.Sequential(
            nn.Conv2d(inplanes, hiddenChannels, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.attentionJ = CALayer(hiddenChannels, reduction=1)
        self.attentionD = CALayer(hiddenChannels, reduction=1)
        # self.attentionJ = LearnedConvC(hiddenChannels, reduction=4)
        # self.attentionD = LearnedConvC(hiddenChannels, reduction=4)
        # self.attentionJ = CPSPPSELayer(hiddenChannels, hiddenChannels, reduction=4)
        # self.attentionD = CPSPPSELayer(hiddenChannels, hiddenChannels, reduction=4)

        self.convJ = nn.Sequential(
            nn.Conv2d(hiddenChannels, hiddenChannels, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.convD = nn.Sequential(
            nn.Conv2d(hiddenChannels, hiddenChannels, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.lstmJ = ConvLSTMCell(inChannels=hiddenChannels, outChannels=hiddenChannels, kernel_size=3)
        self.lstmD = ConvLSTMCell(inChannels=hiddenChannels, outChannels=hiddenChannels, kernel_size=3)


        self.decoderJ = nn.Sequential(
            nn.Conv2d(hiddenChannels, hiddenChannels, kernel_size=3, padding=1),
            nn.Conv2d(hiddenChannels, 3, kernel_size=1))

        self.decoderD = nn.Sequential(
            nn.Conv2d(hiddenChannels, hiddenChannels, kernel_size=3, padding=1),
            nn.Conv2d(hiddenChannels, 1, kernel_size=1))

        # self.refresh = nn.Sequential(
        #     ResBlock(hiddenChannels*2, hiddenChannels*2)

        # )

    def forward(self, x, hJ=None, cJ=None, hD=None, cD=None):

        x = self.gate(x)

        # allocation features
        featsJ = self.convJ(self.attentionJ(x)) #
        featsD = self.convD(self.attentionD(x))

        #initization
        if hJ is None:
            hJ = featsJ
        if hD is None:
            hD = featsD
        if cJ is None:
            cJ = torch.zeros_like(hJ)
        if cD is None:
            cD = torch.zeros_like(hD)

        #lstm
        hJ, cJ = self.lstmJ(featsJ, hJ, cJ)
        hD, cD = self.lstmD(featsD, hD, cD)

        # Decoding
        J = self.decoderJ(hJ)
        D = self.decoderD(hD)

        # Aggregate
        x = torch.cat((featsJ, featsD), dim=1)
        # x = self.refresh(x)


        return J, D, x, hJ, cJ, hD, cD


class upconv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, scale=2, padding=1, activation=nn.ReLU(inplace=False)):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size, padding=padding, bias=False),
            activation
        )
        self.scale = scale

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode='nearest')
        x = self.conv(x)
        return x


class recursiveJModule(nn.Module):
    def __init__(self, inplanes=64, midplanes=None):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Conv2d(inplanes, midplanes, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.attentionJ = CALayer(midplanes, reduction=1)  #

        self.convJ = nn.Sequential(
            nn.Conv2d(midplanes, midplanes, kernel_size=3, padding=1),
            nn.PReLU()
        )

        self.lstmJ = ConvLSTMCell(inChannels=midplanes, outChannels=midplanes, kernel_size=3)

        self.up1 = upconv(midplanes, midplanes // 2)
        self.up2 = upconv(midplanes // 2, midplanes)

        self.decoderJ = nn.Sequential(nn.Conv2d(midplanes, 3, kernel_size=3, padding=1),
                                      nn.ReLU())

        self.tail = nn.Sequential(
            nn.Conv2d(midplanes, inplanes//2, kernel_size=3, padding=1),
            nn.ReLU())


    def forward(self, x, hJ=None, cJ=None):

        x = self.gate(x)

        # allocation features
        featsJ = self.convJ(self.attentionJ(x))

        #initization
        if hJ is None:
            hJ = featsJ
        if cJ is None:
            cJ = torch.zeros_like(hJ)

        #lstm
        hJ, cJ = self.lstmJ(featsJ, hJ, cJ)

        # Decoding
        J = self.up1(hJ)
        J = self.up2(J)
        J = self.decoderJ(J)

        # Aggregate
        x = self.tail(featsJ)

        return J, x, hJ, cJ

class recursiveDModule(nn.Module):  #52054
    def __init__(self, inChannels=16*2):
        super().__init__()

        hiddenChannels = inChannels // 16 * 10

        self.gate = nn.Sequential(
            nn.Conv2d(inChannels, hiddenChannels, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.attentionD = CALayer(hiddenChannels, reduction=1)  #

        self.convD = nn.Sequential(
            nn.Conv2d(hiddenChannels, hiddenChannels, kernel_size=3, padding=1),
            nn.PReLU()
        )


        self.lstmD = ConvLSTMCell(inChannels=hiddenChannels, outChannels=hiddenChannels, kernel_size=3)


        self.decoderD = nn.Sequential(
            nn.Conv2d(hiddenChannels, hiddenChannels, kernel_size=3, padding=1),
            nn.Conv2d(hiddenChannels, 1, kernel_size=1))


    def forward(self, x, hD=None, cD=None):

        x = self.gate(x)

        # allocation features
        featsD = self.convD(self.attentionD(x))

        #initization
        if hD is None:
            hD = featsD
        if cD is None:
            cD = torch.zeros_like(hD)

        #lstm
        hD, cD = self.lstmD(featsD, hD, cD)

        # Decoding
        D = self.decoderD(hD)

        # Aggregate
        x = featsD


        return D, x, hD, cD


class ResBlock(nn.Module):
    def __init__(self, inChannels=32, outChannels=32, stride=1, bias=False, expansion=4, activation=nn.ReLU(), *args, **kwargs):
        super().__init__()
        self.activation = activation
        self.block = nn.Sequential(
            nn.Conv2d(inChannels, inChannels * expansion, 3, padding=1, stride=stride, bias=bias),
            nn.BatchNorm2d(inChannels * expansion),
            activation,
            nn.Conv2d(inChannels * expansion, outChannels, 3, padding=1, bias=bias),
            nn.BatchNorm2d(outChannels)
        )
        if outChannels == inChannels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inChannels, outChannels, 3, padding=1, stride=stride, bias=bias),
                nn.BatchNorm2d(outChannels)
            )

    def forward(self, x):
        res = self.shortcut(x)
        x = self.block(x)
        x = x + res
        return self.activation(x)


class ConvHead(nn.Module):
    def __init__(self, inChannels=16*2, hiddenChannels=25):
        super().__init__()
        self.block = ResBlock(inChannels, hiddenChannels, expansion=2)

        self.decoderD = nn.Sequential(
            nn.Conv2d(hiddenChannels, hiddenChannels, 3, padding=1),
            nn.BatchNorm2d(hiddenChannels),
            nn.ReLU(),
            nn.Conv2d(hiddenChannels, 1, 3, padding=1),
        )

        self.decoderJ = nn.Sequential(
            nn.Conv2d(hiddenChannels, hiddenChannels, 3, padding=1),
            nn.BatchNorm2d(hiddenChannels),
            nn.ReLU(),
            nn.Conv2d(hiddenChannels, 3, 3, padding=1),
        )

    def forward(self, x):
        x = self.block(x)
        J = self.decoderJ(x)
        D = self.decoderD(x)
        return J, D


class ConvHeadJ(nn.Module):
    def __init__(self, inChannels=16*2, hiddenChannels=24):
        super().__init__()
        self.block = ResBlock(inChannels, hiddenChannels, expansion=2)

        hiddenChannels2 = hiddenChannels // 4 * 9
        self.decoderJ = nn.Sequential(
            nn.Conv2d(hiddenChannels, hiddenChannels2, 3, padding=1),
            nn.BatchNorm2d(hiddenChannels2),
            nn.ReLU(),
            nn.Conv2d(hiddenChannels2, 3, 3, padding=1),
        )
    
    def forward(self, x):
        x = self.block(x)
        J = self.decoderJ(x)
        return J

if __name__ == '__main__':
    model = ConvHeadJ()
    model.to('cuda')
    from torchsummary import summary
    summary(model, input_size=(32, 256, 256))