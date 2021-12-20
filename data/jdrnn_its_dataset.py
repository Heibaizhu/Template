# from data.sihr_base_dataset import SIHRDataset
from data.jdrnn_base_dataset import JDRNNDataset
import json
import os


class JDRNNITSDataset(JDRNNDataset):

    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt


    def get_datalist(self):
        with open(self.opt['json'], 'r') as f:
            meta_list = json.load(f)

        hazes = meta_list['haze']
        clears = meta_list['clear']
        depths = meta_list['depth']
        if self.dataroot is not None:
            hazes = [os.path.join(self.dataroot, haze.partition('DEHAZEDATASET')[-1]) for haze in hazes]
            clears = [os.path.join(self.dataroot, clear.partition('DEHAZEDATASET')[-1]) for clear in clears]
        return list(zip(hazes, clears, depths))


