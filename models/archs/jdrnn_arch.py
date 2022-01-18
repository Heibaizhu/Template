import torch
import torch.nn as nn
from .RecursiveModule import recursiveModule, ConvHead, recursiveJModule, recursiveDModule, ConvHeadJ
from .resUnetPlus import ResUnetPlusPlusBackbone
import pdb
import torch.nn.functional as F

class LearnedConvC(nn.Module):
    def __init__(self, inplanes=64, reduction=4):
        super().__init__()
        self.metaConv = nn.Sequential(nn.Conv2d(inplanes, inplanes, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(inplanes),
                                      nn.ReLU())

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(inplanes, inplanes // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(inplanes // reduction, inplanes, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        y = self.metaConv(x)
        y = x * y
        y = self.avg_pool(y).view(B, -1)
        y = self.fc(y).view(B, C, 1, 1)
        x = y * x
        return x



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

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        res = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        wa = self.sigmoid(x)
        return wa * res

class pure_upconv(nn.Module):
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

class upconv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, scale=2, padding=1, activation=nn.ReLU(inplace=False)):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size, padding=padding, bias=False),
            activation
        )
        self.scale = scale

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=self.scale, mode='nearest')
        x = torch.cat([x, res], dim=1)
        x = self.conv(x)
        return x


class SCBlock(nn.Module):
    """
    spatial self attention block
    """
    def __init__(self, inplanes=32, midplanes=32, outplanes=32, scale=1, activation=nn.ReLU(inplace=False), reduction=2):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, midplanes, kernel_size=3, padding=1)
        self.act = activation
        if scale == 0:
            self.sconv = nn.Conv2d(midplanes, inplanes, kernel_size=1)
        else:
            self.sconv = nn.Conv2d(midplanes, inplanes, kernel_size=3, padding=scale, dilation=scale)
        self.conv2 = nn.Conv2d(inplanes, outplanes, kernel_size=3, padding=1)
        self.wa = SpatialAttention(kernel_size=3)
        # self.ca = LearnedConvC(inplanes=outplanes, reduction=reduction)
        self.ca = CALayer(outplanes, reduction=reduction)

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.act(x)
        x = self.sconv(x)
        x += res
        x = self.act(x)
        x = self.conv2(x)
        x = self.wa(x)
        x = self.ca(x)
        return x


class MSCBlock(nn.Module):
    """
    multiscale spatial channel self attention block
    """
    def __init__(self, inplanes, midplanes, outplanes, scales=(1, 2, 3), activation=nn.ReLU(inplace=False), reduction=2):
        super().__init__()
        self.scales = scales
        self.blocks = nn.ModuleDict()
        for scale in scales:
            self.blocks[str(scale)] = SCBlock(inplanes, midplanes, outplanes, scale=scale, activation=activation, reduction=reduction)

    def forward(self, x):
        res = x
        sum = 0
        for scale in self.scales:
            sum += self.blocks[str(scale)](x)
        return sum + res

class MSCGroup(nn.Module):
    def __init__(self, block_num, inplanes, scales=(1, 2, 3), activation=nn.ReLU(inplace=True), reduction=2, res=False):
        super().__init__()
        self.msc_blocks = nn.ModuleList()
        self.block_num = block_num
        self.res = res
        for i in range(block_num):
            self.msc_blocks.append(MSCBlock(inplanes, inplanes//2,
                                            inplanes, scales=scales,
                                            activation=activation,
                                            reduction=reduction))
        self.conv = nn.Conv2d(inplanes, inplanes, kernel_size=3, padding=1)

    def forward(self, x):
        res = x
        for i in range(self.block_num):
            x = self.msc_blocks[i](x)
        if self.res:
            x += res
            # x = self.conv(x)
        return x





class SMFA(nn.Module):
    def __init__(self, midplanes=64, block_num=(4, 4, 4), **args):
        super().__init__()

        #downsampling
        self.down1 = nn.Sequential(
            nn.Conv2d(3, midplanes//2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(midplanes//2, midplanes//2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(midplanes//2, midplanes, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(midplanes, midplanes, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        #feature extraction
        self.group1 = MSCGroup(block_num[0], midplanes, scales=(1, 2, 3), activation=nn.ReLU(inplace=True), reduction=2)
        self.group2 = MSCGroup(block_num[1], midplanes, scales=(1, 2, 3), activation=nn.ReLU(inplace=True), reduction=2)
        self.group3 = MSCGroup(block_num[2], midplanes, scales=(1, 2, 3), activation=nn.ReLU(inplace=True), reduction=2)

        #upsampling
        self.up1 = pure_upconv(midplanes, midplanes//2)
        self.up2 = pure_upconv(midplanes//2, midplanes)

        #decoder
        self.outconv = nn.Conv2d(midplanes, 3, kernel_size=3, padding=1)

    def forward(self, x):

        # downsampling
        x = self.down1(x)
        x = self.down2(x)

        # feature extraction
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        # for i in range(self.block_num):
        #     x = self.msc_blocks[i](x)
        # x = self.msc_blocks[0](x)

        #upsampling
        x = self.up1(x)
        x = self.up2(x)

        #decoder
        x = self.outconv(x)
        output = {'dehazed': [x]}


        return output

# import torchvision.models as models
# resnet18 = models.resnet18()


class JDRNN(nn.Module):
    def __init__(self, midplanes=64, block_num=(4, 4, 4), scales=(1, 2, 3), bridge_planes=(16, 16, 16, 16), iter_num=3, pos_dim=4):
        super().__init__()

        self.iter_num = iter_num
        self.bridge_planes = bridge_planes

        pi = 3.1415
        coeff = torch.tensor([3 ** i * pi for i in range(pos_dim)]).view(1, pos_dim, 1, 1, 1)
        self.register_buffer('coeff', coeff)

        #downsampling
        self.down1 = nn.Sequential(
            nn.Conv2d(3, midplanes//2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(midplanes//2, midplanes//2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(midplanes//2, midplanes, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(midplanes, midplanes, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        #feature extraction
        self.groups = nn.ModuleList()
        self.block_num = block_num

        for i, num in enumerate(block_num):
            if i == 0:
                res = True
            else:
                res = False
            self.groups.append(MSCGroup(num, midplanes * (2 * pos_dim), scales=scales, activation=nn.ReLU(inplace=True), reduction=2, res=res))

        self.conv = nn.Conv2d(midplanes * 2 * pos_dim, midplanes, kernel_size=1)
        # if bridge_planes:
        #     self.bridge = nn.ModuleList()
        #     for bridge_plane in bridge_planes:
        #         self.bridge.append(nn.Conv2d(midplanes, bridge_plane, kernel_size=1))
        #     feature_num = midplanes
        # else:
        #     feature_num = midplanes * len(block_num)
        feature_num = midplanes

        # self.recurJModule = recursiveJModule(inplanes=feature_num * 2, midplanes=feature_num)

        #decoder
        # upsampling
        self.up1 = pure_upconv(feature_num, feature_num)
        self.up2 = pure_upconv(feature_num, feature_num)
        self.outconv = nn.Sequential(nn.Conv2d(feature_num, 3, kernel_size=3, padding=1))

    def forward(self, x):
        res = x

        # downsampling
        x = self.down1(x)
        x = self.down2(x)

        # pos encoding
        B, C, H, W = x.shape
        x = x.view(B, 1, C, H, W)
        phase = x * self.coeff
        x = torch.cat((torch.sin(phase), torch.cos(phase)), dim=1).view(B, -1, H, W)

        # feature extraction
        for i in range(len(self.block_num)):
            x = self.groups[i](x)

            # if self.bridge_planes:
            #     features.append(self.bridge[i](x))
            # else:
            #     features.append(x)
        # x = torch.cat(features, dim=1)
        x = self.conv(x)

        # init = x
        # listJ = []
        # for i in range(self.iter_num):
        #     x = torch.cat((x, init), dim=1)
        #     if i == 0:
        #         J, x, hJ, cJ = self.recurJModule(x)
        #     else:
        #         J, x, hJ, cJ = self.recurJModule(x, hJ, cJ)
        #     listJ.append(J)

        x = self.up1(x)
        x = self.up2(x)
        x = self.outconv(x) + res

        output = {'dehazed': [x]}

        return output


# class JDRNN(nn.Module):
#     def __init__(self, inChannels=3, hiddenChannels=16, N=1):
#         super().__init__()
#         self.N = N
#         k = hiddenChannels
#         self.hiddenChannels = hiddenChannels
#         self.backbone = ResUnetPlusPlusBackbone(inChannels, filters=[k, 2 * k, 4*k, 8*k, 16*k])
#         self.recurModule = recursiveModule(inChannels=hiddenChannels*4)
#         self.s1 = nn.Parameter(torch.zeros(1)) # s1 = log2(sigma^2) for dehaze
#         self.s2 = nn.Parameter(torch.zeros(1)) # s2 = log2(sigma^2) for depth
#
#     def weight_loss(self, loss1, loss2):
#         weighted_loss = torch.exp(-self.s1) * loss1 + torch.exp(-self.s2) * loss2 + (self.s1 + self.s2)
#         return weighted_loss
#
#     def forward(self, x):
#         x = self.backbone(x) # 2 * hiddenchannels
#         listJ = []
#         listD = []
#         init = x
#         for i in range(self.N):
#             x = torch.cat([x, init], dim=1)  # 4 * hiddenchannels
#             if i == 0:
#                 J, D, x, hJ, cJ, hD, cD = self.recurModule(x)
#             else:
#                 J, D, x, hJ, cJ, hD, cD = self.recurModule(x, hJ, cJ, hD, cD)
#             listJ.append(J)
#             listD.append(D)
#         return listJ, listD


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
    # from torchsummary import summary
    # summary(model, input_size=(3, 256, 256))
    # x = torch.randn(1, 3, 255, 255).to('cuda')
    # y = model(x)
    # pdb.set_trace()
    # print(y.shape)