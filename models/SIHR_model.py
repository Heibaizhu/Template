# -*- coding: utf-8 -*-
import pdb
import importlib
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision
from models.archs import define_network
from models.base_model import BaseModel
from core_utils.losses import *
from core_utils.utils.logger import get_root_logger
from core_utils.utils.dist_utils import master_only
import time

"""
User module 
"""

loss_module = importlib.import_module('core_utils.losses')
metric_module = importlib.import_module('core_utils.metrics')


class SIHRModel(BaseModel):
    """Base interpolation model for novel view synthesis."""

    def __init__(self, opt):
        super().__init__(opt)

        # define network
        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        total_param = sum(p.numel() for p in self.net_g.parameters() if p.requires_grad) / 1024 / 1024

        logger = get_root_logger()
        logger.info("**************Module size (bumber)**************\n{}".format(total_param))
        self.print_network(self.net_g)

        self.output = None
        self.optimizer_g = None
        self.log_dict = None
        self.metric_results = None

        # discriminate the phase
        self.phase = 'train' if 'train' in self.opt['datasets'] else 'val'
        self.prior_metric = self.opt['val'].get('prior_metric', None)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        self.iter_num = self.opt['network_g'].get('iter_num', None)
        self.loss_iter_decay = self.opt['train'].get('loss_iter_decay', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True))



    def init_training_settings(self):
        pass

    def setup_optimizers(self):
        pass

    def feed_data(self, data):
        pass

    def optimize_parameters(self, current_iter):
        pass

    def loss_fn(self, output):
        pass

    def test(self):
        pass

    def save_best(self, current_iter):
        save_flag = True
        logger = get_root_logger()
        for metric in self.metric_results.keys():
            extremum_name = 'extremum' + '_' + metric
            if metric in ('psnr', 'ssim', 'PSNR', 'SSIM'):  # positive metric
                extremum = getattr(self, extremum_name, -float('inf'))
                if self.metric_results[metric] > extremum:
                    extremum = self.metric_results[metric]
                    setattr(self, extremum_name, extremum)
                    logger.info("Best {}: {} at iter: {}".format(metric, extremum, current_iter))
                elif self.metric_results[metric] == extremum:
                    pass
                else:
                    save_flag = False
                    break
            elif metric in ('ciede2000', 'CIEDE2000'):  # negative metric
                extremum = getattr(self, extremum_name, float('inf'))
                if self.metric_results[metric] < extremum:
                    extremum = self.metric_results[metric]
                    setattr(self, extremum_name, extremum)
                    logger.info("Best {}: {} at iter: {}".format(metric, extremum, current_iter))
                elif self.metric_results[metric] == extremum:
                    pass
                else:
                    save_flag = False
                    break
            else:
                pass

        if save_flag:
            self.save(-1, current_iter, best=True)

    def dist_validation(self, dataloader, current_iter, tb_logger,
                        save_img):
        with_metrics, metric_results, test_num = \
            self.core_validation(dataloader, current_iter, tb_logger,
                                 save_img)
        if with_metrics:
            self.metric_results = {}
            dataset_name = dataloader.dataset.opt['name']
            test_num_sum = torch.tensor(test_num).to(self.device)
            dist.all_reduce(test_num_sum,
                            op=dist.ReduceOp.SUM)
            for metric in metric_results.keys():
                result = torch.tensor(metric_results[metric]).to(self.device)
                dist.all_reduce(result,
                                op=dist.ReduceOp.SUM)
                self.metric_results[metric] = (result / test_num_sum).item()

            self.save_best(current_iter)

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)

    @master_only
    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img):
        with_metrics, metric_results, test_num = \
            self.core_validation(dataloader, current_iter, tb_logger,
                                 save_img)
        if with_metrics:
            self.metric_results = {}
            dataset_name = dataloader.dataset.opt['name']
            for metric in metric_results.keys():
                self.metric_results[metric] = metric_results[metric] / test_num

            self.save_best(current_iter)
            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)

    def core_validation(self, dataloader, current_iter,
                        tb_logger, save_img):
        with_metrics = {}
        metric_results = {}
        test_num = 0
        return with_metrics, metric_results, test_num

    @master_only
    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}\n'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        pass

    def save(self, epoch, current_iter, best=False):
        self.save_network(self.net_g, 'net_g', current_iter, best=best)
        self.save_training_state(epoch, current_iter, best=best)

