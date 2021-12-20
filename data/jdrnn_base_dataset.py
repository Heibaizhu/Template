import torchvision.transforms

import torch.utils.data as data
from skimage import io
import random
import torch
import numpy as np
import os
from torchvision.transforms.functional import normalize
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import cv2
import numbers
from collections.abc import Sequence
from hdf5storage import loadmat
import time
from utils.logger import get_root_logger

def img2tensor(imgs, bgr2rgb=False, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=0)
            img = torch.from_numpy(img)
        else:
            if img.shape[2] == 3 and bgr2rgb:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)

def to_tensor(arr):
    """
    将numpy转为tensor，更改通道顺序
    :param arr:
    :return:
    """
    if arr.ndim == 2:
        return torch.from_numpy(arr[np.newaxis, :, :])
    elif arr.ndim == 1:
        return torch.from_numpy(arr)
    elif arr.ndim == 3:
        return torch.from_numpy(np.ascontiguousarray(np.transpose(arr, (2, 0, 1))))
    elif arr.ndim == 0:
        return torch.from_numpy(arr)
    else:
        raise NotImplementedError

class FunsCompose:
    def __init__(self, funs):
        self.funs = funs

    def __call__(self, *args):
        temp = random.random()
        results = []
        state = random.getstate()
        if len(args) == len(self.funs):
            for i, fun in enumerate(self.funs):
                random.setstate(state)
                results.append(fun(args[i]))
        else:
            for fun in self.funs:
                random.setstate(state)
                results.append(fun(*args))

        return results

def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size

class JDRNNDataset(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.phase = opt['phase']
        self.mean = opt.get('mean', None)
        self.std = opt.get('std', None)
        self.dataroot = opt['dataroot']
        self.datalist = self.get_datalist()
        self.center_size = opt.get('center_size', None)
        self.random_crop = opt.get('random_crop', None)
        if self.random_crop is not None:
            self.transformer_random_crop = torchvision.transforms.RandomCrop(self.random_crop)

    def get_datalist(self):
        pass

    def __len__(self):
        """
        当 phase 为 'val'时，repeats参数失去作用，dataset数据不重复
        :return:
        """
        return len(self.datalist)

    def __getitem__(self, index):
        haze_path, clear_path, depth_path = self.datalist[index]
        logger = get_root_logger()
        # IO
        io_time = time.time()
        # haze = cv2.imread(haze_path)
        # haze = cv2.cvtColor(haze, cv2.COLOR_BGR2RGB) / 255.
        haze = io.imread(haze_path) / 255.
        # clear = cv2.imread(clear_path)
        # clear = cv2.cvtColor(clear, cv2.COLOR_BGR2RGB) / 255.
        clear = io.imread(clear_path) / 255.
        depth = loadmat(depth_path)['depth'] / 10
        io_time = time.time() - io_time
        # logger.info("io time: {}".format(io_time))


        assert haze.ndim == 3 and clear.ndim == 3, "The ndim of the input image is not 3!"

        haze = img2tensor(haze)
        clear = img2tensor(clear)
        depth = img2tensor(depth)

        ### transformation piplines
        t_time = time.time()
        if self.mean is not None or self.std is not None:
            normalize(haze, self.mean, self.std, inplace=True)
            normalize(clear, self.mean, self.std, inplace=True)

        if self.center_size is not None:
            haze = TF.center_crop(haze, self.center_size)
            clear = TF.center_crop(clear, self.center_size)
            depth = TF.center_crop(depth, self.center_size)

        if self.random_crop:
            size = tuple(_setup_size(
            self.random_crop, error_msg="Please provide only two dimensions (h, w) for size."
        ))
            i, j, th, tw = transforms.RandomCrop.get_params(haze, size)
            haze = TF.crop(haze, i, j, th, tw)
            clear = TF.crop(clear, i, j, th, tw)
            depth = TF.crop(depth, i, j, th, tw)

        t_time = time.time() - t_time
        # logger.info("transformer time: {}".format(t_time))

        output = {
            'haze': haze,
            'clear': clear,
            'depth': depth,
            'name': os.path.basename(haze_path)
        }
        return output

