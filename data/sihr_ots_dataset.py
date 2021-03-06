from data.sihr_base_dataset import SIHRDataset
import json
import os


class SIHROTSDataset(SIHRDataset):

    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt


    def get_datalist(self):
        with open(self.opt['json'], 'r') as f:
            meta_list = json.load(f)

        hazes = meta_list['haze']
        clears = meta_list['clear']
        if self.dataroot is not None:
            hazes = [os.path.join(self.dataroot, haze) for haze in hazes]
            clears = [os.path.join(self.dataroot, clear) for clear in clears]
        return list(zip(hazes, clears))


