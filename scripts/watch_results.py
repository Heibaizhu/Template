import json
import glob
from make_grids import make_grids
import tqdm
import os
from skimage import io
import numpy as np


class StackImage():
    def __init__(self, outdir):
        self.outdir = outdir
        if not os.path.exists(outdir):
            os.makedirs(outdir)

    def satck(self, dataset_json, input_dir, *pred_dir, nogt=False):
        with open(dataset_json, 'r') as f:
            data_dict = json.load(f)
        if nogt:
            hazes = data_dict['haze']
        else:
            hazes, clears = data_dict['haze'], data_dict['clear']
        pb = tqdm.tqdm(range(len(hazes)))
        for i in pb:
            imgs = []
            name = os.path.basename(hazes[i])
            if not nogt:
                data_clear = io.imread(clears[i])
            data_haze = io.imread(hazes[i])
            imgs.append(data_haze)
            if not nogt:
                imgs.append(data_clear)
            for item_dir in pred_dir:
                pred_path = os.path.join(input_dir, item_dir, name)
                pred_path = os.path.splitext(pred_path)[0] + '.png'
                if not os.path.exists(pred_path):
                    print("{} doesn't exists!".format(pred_path))
                data_pred = io.imread(pred_path)
                imgs.append(data_pred)
            imgs = np.stack(imgs, axis=0)
            img_grid = make_grids(imgs, 5)
            io.imsave(os.path.join(self.outdir, name), img_grid)


if __name__ == '__main__':
    # input_json = r'/home/jwzha/workfs/Dataset/DEHAZEDataset/RESIDE/RESIDE_standard/SOTS/SOTS/indoor.json'
    # input_json = '/home/jwzha/workfs/Dataset/DEHAZEDataset/D-Hazy/D-HAZY_DATASET/MiddleburyResized1000.json'
    # input_json = '/home/jwzha/workfs/Dataset/DEHAZEDataset/O-Hazy/O-HAZE/# O-HAZY NTIRE 2018/Resized-Dataset.json'
    # input_json = '/home/jwzha/scratch/DATASETS/DEHAZEDATASET/I-Haze/I-HAZE/# I-HAZY NTIRE 2018/Resized-TEST-5.json'
    # input_json = '/home/jwzha/workfs/Dataset/DEHAZEDataset/RESIDE/RESIDE_standard/SOTS/SOTS/outdoor.json'
    # input_json = '/home/jwzha/scratch/DATASETS/DEHAZEDATASET/O-Hazy/O-HAZE/# O-HAZY NTIRE 2018/Resized-Test.json'
    # input_json = '/home/jwzha/workfs/Dataset/DEHAZEDataset/HazeRD/HazeRD/data/HazeRDResized1000.json'
    # output_dir = r'/scratch/jwzha/PREDICT/11-8-predict/GDN/SOTS_INDOOR_12_23_its/'
    output_dir = r'/home/jwzha/scratch/PREDICT/11-8-predict/AOD-Net/CONTRAST/NHHAZE-NHHAZE'

    obj = StackImage(output_dir)

    # OTS-SOTS_OUTDOOR
    option4 = ['/home/jwzha/scratch/DATASETS/DEHAZEDATASET/RESIDE/RESIDE_standard/SOTS/SOTS/outdoor.json',
               '/home/jwzha/scratch/PREDICT/11-8-predict/GDN/SOTS_OUTDOOR',
               '1-25-griddehazenet-OTS-TRAIN-VAL+1l1loss+0mseloss+0perceputal+0RCL+0GANLossD+0GANLossG+0ssim+0ms_ssim+0fft',
               '1-22-griddehazenet-OTS-TRAIN-VAL+1mseloss+0perceputal+0RCL-debug+0GANLossD+0GANLossG+0ssim+0ms_ssim+0fft',
               '11-11-griddehazenet-OTS1399-TRAIN-VAL+0perceptual+0RCL',
               '11-12-griddehazenet-OTS1399-TRAIN-VAL+0.04perceptual+0RCL',
               '12-17-griddehazenet-OTS-TRAIN-VAL+0perceputal+0RCL+0GANLossD+0GANLossG+0.01ssim+0ms_ssim',
               # '1-11-griddehazenet-OTS-TRAIN-VAL+0perceputal+0RCL+0GANLossD+0GANLossG+0ssim+0ms_ssim+1fftloss',
               '12-15-griddehazenet-OTS-TRAIN-VAL+0perceputal+0RCL+0.05GANLossD+0.05GANLossG+0ssim+0ms_ssim',
               '2-9-griddehazenet-OTS-TRAIN-VAL+1smoothl1loss+0.04perceptual+0.05GAN',
               '2-10-griddehazenet-OTS-TRAIN-VAL+1smoothl1loss+10SRL11',
               ]

    obj.satck(*option23, nogt=False)