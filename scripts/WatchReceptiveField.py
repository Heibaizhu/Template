import torch
import torch.nn as nn 
import torchvision
import os


class GetERF:
    def __init__(self):
        pass

    def __call__(self, model, inShape, dataRange=(0, 1), iters=1000):
        """

        Args:
            model: the model has loaded the parameters, device=cpu
            inShape: the input shape of the model, only support single input
            dataRange: the datarange of the input data, [0, 1] and [-1, 1] are the normal choice
            iters: Compute the expected number of samples for the gradient
        Returns:
        """
        model.eval()
        grads = 0
        Max, Min = dataRange
        for i in range(iters):
            input = torch.rand(inShape)
            input = input * (Max-Min) + Min
            input.requires_grad = True
            output = model(input)['out']
            B, C, H, W = output.shape
            loss = output[:, :, H//2, W//2]
            loss = loss.mean()
            loss.backward()
            #求梯度
            grad = torch.sum(torch.abs(input.grad), dim=[0, 1], keepdim=True)
            grads += grad
        grads /= iters
        return grads

    def stretch(self, arr, c=100000.):
        arr = torch.log(1 + c * arr)
        return arr

    def normalize(self, grads):
        return (grads - torch.min(grads)) / (torch.max(grads) - torch.min(grads))

    def save(self, grads, centerCrop=(250, 250), stretchFactor=1e5, scaleFactor=5, outPath=''):
        dstH, dstW = centerCrop
        B, C, H, W = grads.shape
        grads = self.stretch(grads, c=stretchFactor)
        grads = self.normalize(grads)
        assert dstH <= H and dstW <= W
        grads = grads[:, :, H//2 - dstH//2: H// 2 + dstH//2, W//2 - dstW//2: W//2 + dstW//2]
        grads = torch.nn.functional.interpolate(grads, scale_factor=scaleFactor)
        if not outPath:
            outPath = 'ERF.png'
        torchvision.utils.save_image(grads, outPath)


if __name__ == '__main__':
    import torchvision.models as models
    import torch
    import os
    net = models.segmentation.fcn_resnet50()

    obj = GetERF()
    grads = obj(net, (1, 3, 224, 224), iters=100)
    obj.save(grads, outPath='temp.png', centerCrop=(150, 150))