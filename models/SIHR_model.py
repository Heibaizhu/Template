# -*- coding: utf-8 -*-
import pdb
import importlib
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm
import numbers
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from models.archs import define_network
from models.base_model import BaseModel
from models.losses import *
from utils.logger import get_root_logger
from utils.img_utils import imwrite, tensor2img
from utils.dist_utils import master_only
import numpy as np


"""
User module
"""

loss_module = importlib.import_module('models.losses')
metric_module = importlib.import_module('metrics')


class SIHRModel(BaseModel):
    """Base interpolation model for novel view synthesis."""

    def __init__(self, opt):
        super().__init__(opt)

        # define network
        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        total_param = sum(p.numel() for p in self.net_g.parameters() if p.requires_grad) * 4 / 1024 / 1024
        print("Model size:", total_param)
        self.print_network(self.net_g)

        self.cri_pix = None
        self.cri_census = None
        self.cri_perceptual = None
        self.img_l = None
        self.img_r = None

        self.flow = None
        self.gt = None
        self.output = None
        self.optimizer_g = None
        self.log_dict = None
        self.metric_results = None
        # H, W = opt['datasets']['train']['resize_h'], opt['datasets']['train']['resize_w']

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True))

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        # define losses
        if train_opt.get('l1_loss'):
            loss_type = train_opt['l1_loss'].pop('type')
            self.l1_loss = getattr(loss_module, loss_type)(**train_opt['l1_loss']).to(self.device)

        if train_opt.get('l2_loss'):
            loss_type = train_opt['l2_loss'].pop('type')
            self.l2_loss = getattr(loss_module, loss_type)(**train_opt['l2_loss']).to(self.device)

        if train_opt.get('smooth_l1_loss'):
            loss_type = train_opt['smooth_l1_loss'].pop('type')
            self.smooth_l1_loss = getattr(loss_module, loss_type)(**train_opt['smooth_l1_loss']).to(self.device)

        if train_opt.get('ssim_loss'):
            loss_type = train_opt['ssim_loss'].pop('type')
            self.ssim_loss = getattr(loss_module, loss_type)(**train_opt['ssim_loss']).to(self.device)

        if train_opt.get('perceptual_loss'):
            loss_type = train_opt['perceptual_loss'].pop('type')
            self.perceputal_loss = getattr(loss_module, loss_type)(**train_opt['perceptual_loss']).to(self.device)

        if train_opt.get('mc_loss'):
            loss_type = train_opt['mc_loss'].pop('type')
            self.mc_loss = getattr(loss_module, loss_type)(**train_opt['mc_loss']).to(self.device)


        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')
        #
        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params,
                                                **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        # self.optimizer_g = torch.optim.Adam(
        #     [{'params': self.net_g.encoder.parameters(), 'lr': train_opt['optim_g'].pop('backbone_lr')},
        #     {'params': self.net_g.decoder.parameters(), 'lr': train_opt['optim_g'].pop('decoder_lr')}],
        #     **train_opt['optim_g']
        # )

        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.haze = data['haze'].to(self.device)
        self.clear = data['clear'].to(self.device)


    def optimize_parameters(self, current_iter):

        self.optimizer_g.zero_grad()

        self.output = self.net_g(self.haze)

        # loss_dict = OrderedDict()

        loss_exp_dict = self.loss_fn(self.output)
        loss_dict = loss_exp_dict

        l_total = loss_exp_dict['loss_total']
        l_total.backward()
        # torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.1)
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)
        return l_total

    def loss_fn(self, output):
        dehazed = output['dehazed']

        loss_l1 = self.l1_loss(dehazed, self.clear)
        loss_l2 = self.l2_loss(dehazed, self.clear)
        loss_smooth_l1 = self.smooth_l1_loss(dehazed, self.clear)
        loss_per = self.perceputal_loss(dehazed, self.clear)[0]
        loss_ssim = self.ssim_loss(dehazed, self.clear)
        loss_mc = self.mc_loss(dehazed, self.clear)

        # pdb.set_trace()
        if loss_per:
            loss_total = loss_l1 + loss_l2 + loss_smooth_l1 + loss_ssim + loss_mc + loss_per
            loss_exp_dict = {
                'loss_l1': loss_l1,
                'loss_mse': loss_l2,
                'loss_smooth_l1': loss_smooth_l1,
                'loss_per': loss_per,
                'loss_ssim': loss_ssim,
                'loss_mc': loss_mc,
                'loss_total': loss_total

            }
        else:
            loss_total = loss_l1 + loss_l2 + loss_smooth_l1 + loss_ssim + loss_mc
            loss_exp_dict = {
                'loss_l1': loss_l1,
                'loss_mse': loss_l2,
                'loss_smooth_l1': loss_smooth_l1,
                'loss_ssim': loss_ssim,
                'loss_mc': loss_mc,
                'loss_total': loss_total

            }

        return loss_exp_dict

    def test(self):
        self.net_g.eval()

        with torch.no_grad():
            self.output = self.net_g(self.haze)

        self.net_g.train()

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
            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)

    @master_only
    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img):
        pdb.set_trace()
        with_metrics, metric_results, test_num = \
            self.core_validation(dataloader, current_iter, tb_logger,
                                 save_img)
        if with_metrics:
            self.metric_results = {}
            dataset_name = dataloader.dataset.opt['name']
            for metric in metric_results.keys():
                self.metric_results[metric] = metric_results[metric] / test_num
            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)

    def core_validation(self, dataloader, current_iter,
                        tb_logger, save_img):
        test_num = 0
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        metric_results = {
            metric: 0
            for metric in self.opt['val']['metrics'].keys()
        }

        if self.opt['rank'] == 0:
            pbar = tqdm(total=len(dataloader), unit='image')

        set_ = set(range(0, len(dataloader), int(len(dataloader)/10)))
        for idx, val_data in enumerate(dataloader):
            img_names = [osp.splitext(osp.basename(name))[0]
                         for name in val_data['name']]
            test_num += len(img_names)
            self.feed_data(val_data)
            visuals = self.get_current_visuals()
            haze_imgs = []
            clear_imgs = []
            dehazed_imgs = []
            for batch_index in range(len(img_names)):
                haze_imgs.append(tensor2img([visuals['haze'][batch_index]]))
                clear_imgs.append(tensor2img([visuals['clear'][batch_index]]))
                dehazed_imgs.append(tensor2img([visuals['dehazed'][batch_index]]))

            # del self.haze
            # del self.clear
            torch.cuda.empty_cache()

            if save_img:
                for haze_img, clear_img, dehazed_img, img_name in zip(haze_imgs, clear_imgs, dehazed_imgs, img_names):
                    if self.opt['is_train']:
                        base_path = osp.join(self.opt['path']['visualization'], img_name)
                    elif self.opt['val']['suffix']:
                        base_path = osp.join(self.opt['path']['visualization'],
                                             dataset_name,
                                             img_name + '_' + self.opt["val"]["suffix"])
                    else:
                        base_path = osp.join(self.opt['path']['visualization'],
                                             dataset_name,
                                             img_name)
                    # haze_img_path = base_path + '_img_haze.png'
                    # clear_img_path = base_path + '_img_clear.png'
                    dehazed_img_path = base_path + '.png'

                    # imwrite(haze_img, haze_img_path)
                    # imwrite(clear_img, clear_img_path)
                    imwrite(dehazed_img, dehazed_img_path)

                    if idx in set_:
                        tb_img = np.concatenate((haze_img, dehazed_img, clear_img), axis=1)
                        tb_logger.add_image(base_path, tb_img, dataformats='HWC')


            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                for name, opt_ in opt_metric.items():
                    metric_type = opt_.pop('type')
                    for img_index in range(len(img_names)):
                        img_out = F.relu(self.output['dehazed'])
                        img_gt = self.clear
                        img_name = img_names[img_index]
                        metric_results[name] += getattr(
                            metric_module, metric_type)(**opt_)(img_out, img_gt)
                        if self.opt['rank'] == 0:
                            pbar.set_description(f'Test {img_name}')
            if self.opt['rank'] == 0:
                pbar.update(1)
        if self.opt['rank'] == 0:
            pbar.close()
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
        self.test()
        out_dict = OrderedDict()
        out_dict['haze'] = self.haze.detach().cpu()
        out_dict['clear'] = self.clear.detach().cpu()
        out_dict['dehazed'] = self.output['dehazed'].detach().cpu()

        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)

