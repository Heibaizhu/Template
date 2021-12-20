import torch
import torch.nn as nn
from .RecursiveModule import recursiveModule, ConvHead, recursiveJModule, recursiveDModule, ConvHeadJ
from .resUnetPlus import ResUnetPlusPlusBackbone
import pdb

class JDRNN(nn.Module):
    def __init__(self, inChannels=3, hiddenChannels=16, N=1):
        super().__init__()
        self.N = N
        k = hiddenChannels
        self.hiddenChannels = hiddenChannels
        self.backbone = ResUnetPlusPlusBackbone(inChannels, filters=[k, 2 * k, 4*k, 8*k, 16*k])
        self.recurModule = recursiveModule(inChannels=hiddenChannels*4)
        self.s1 = nn.Parameter(torch.zeros(1)) # s1 = log2(sigma^2) for dehaze
        self.s2 = nn.Parameter(torch.zeros(1)) # s2 = log2(sigma^2) for depth

    def weight_loss(self, loss1, loss2):
        weighted_loss = torch.exp(-self.s1) * loss1 + torch.exp(-self.s2) * loss2 + (self.s1 + self.s2)
        return weighted_loss

    def forward(self, x):
        x = self.backbone(x) # 2 * hiddenchannels
        listJ = []
        listD = []
        init = x
        for i in range(self.N):
            x = torch.cat([x, init], dim=1)  # 4 * hiddenchannels
            if i == 0:
                J, D, x, hJ, cJ, hD, cD = self.recurModule(x)
            else:
                J, D, x, hJ, cJ, hD, cD = self.recurModule(x, hJ, cJ, hD, cD)
            listJ.append(J)
            listD.append(D)
        return listJ, listD


class JRNN(nn.Module):
    def __init__(self, inChannels=3, hiddenChannels=16, N=1):
        super().__init__()
        self.N = N
        self.hiddenChannels = hiddenChannels
        # self.backbone = ResUnetPlusPlusBackbone(inChannels, filters=[16, 32, 64, 128, 256])
        self.backbone = ResUnetPlusPlusBackbone(inChannels, filters=[16, 32, 64, 128, 256])
        self.recurJModule = recursiveJModule(inChannels=hiddenChannels*2)


    def forward(self, x):
        x = self.backbone(x)
        listJ = []
        listD = []
        init = x
        for i in range(self.N):
            x = torch.cat([x, init], dim=1)
            if i == 0:
                J, x, hJ, cJ = self.recurJModule(x)
            else:
                J, x, hJ, cJ = self.recurJModule(x, hJ, cJ)
            listJ.append(J)
        return listJ



class JDCNN(nn.Module):
    def __init__(self, inChannels=3, hiddenChannels=16):
        super().__init__()
        self.hiddenChannels = hiddenChannels
        self.backbone = ResUnetPlusPlusBackbone(inChannels, filters=[16, 32, 64, 128, 256])
        self.conv = ConvHead(hiddenChannels * 2)
        self.s1 = nn.Parameter(torch.zeros(1)) # s1 = log2(sigma^2) for dehaze
        self.s2 = nn.Parameter(torch.zeros(1)) # s2 = log2(sigma^2) for depth

    def weight_loss(self, loss1, loss2):
         # = torch.exp(-self.sigma1)  # log(sigma^2)
        weighted_loss = torch.exp(-self.s1) * loss1 + torch.exp(-self.s2) * loss2 + (self.s1 + self.s2)
        return weighted_loss

    def forward(self, x):
        x = self.backbone(x)
        J, D = self.conv(x)
        return [J], [D]

class JCNN(nn.Module):
    def __init__(self, inChannels=3, hiddenChannels=16):
        super().__init__()
        self.hiddenChannels = hiddenChannels
        self.backbone = ResUnetPlusPlusBackbone(inChannels, filters=[16, 32, 64, 128, 256])
        self.conv = ConvHeadJ(hiddenChannels * 2) 


    def forward(self, x):
        x = self.backbone(x)
        J = self.conv(x)
        return [J] 


if __name__ == '__main__':
    model = JDRNN() 
    model.to('cuda')
    from torchsummary import summary
    summary(model, input_size=(3, 256, 256))
    # x = torch.randn(1, 3, 255, 255).to('cuda')
    # y = model(x)
    # pdb.set_trace()
    # print(y.shape)