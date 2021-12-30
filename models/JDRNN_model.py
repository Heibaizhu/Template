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
from models.SIHR_model import  SIHRModel
from core_utils.losses import *
from core_utils.utils.logger import get_root_logger
from core_utils.utils.dist_utils import master_only
import time

"""
User module 
"""

loss_module = importlib.import_module('core_utils.losses')
metric_module = importlib.import_module('core_utils.metrics')


class JDRNNModel(SIHRModel):
    """Base interpolation model for novel view synthesis."""

    def __init__(self, opt):
        super().__init__(opt)

        # define network
        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)


        self.output = None
        self.optimizer_g = None
        self.log_dict = None
        self.metric_results = None
        # H, W = opt['datasets']['train']['resize_h'], opt['datasets']['train']['resize_w']
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

        if train_opt.get('gradient_penalty_loss'):
            loss_type = train_opt['gradient_penalty_loss'].pop('type')
            self.gradient_penalty_loss = getattr(loss_module, loss_type)(**train_opt['gradient_penalty_loss']).to(self.device)

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
        #
        move_time = time.time()
        self.haze = data['haze'].to(self.device)
        self.clear = data['clear'].to(self.device)
        self.depth = data['depth'].to(self.device)
        move_time = time.time() - move_time
        logger = get_root_logger()
        # logger.info("move_time: {}".format(move_time))

    def optimize_parameters(self, current_iter):

        self.optimizer_g.zero_grad()

        with torch.autograd.set_detect_anomaly(False):
            self.haze.requires_grad_(True)
            # logger = get_root_logger()
            # self.tag = "conv7"
            # self.net_g.conv7[0].weight.register_hook(self.hook_fn)
            # self.net_g.log_grad(self.tb_logger)
            self.output = self.net_g(self.haze)

        # loss_dict = OrderedDict()
            loss_exp_dict = self.loss_fn(self.output)

            l_total = loss_exp_dict['loss_total']
            l_total.backward()
            # torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.1)
            self.optimizer_g.step()
            self.log_dict = self.reduce_loss_dict(loss_exp_dict)


        return l_total

    def loss_fn(self, output):

        loss_l1 = 0
        loss_l2 = 0
        loss_smooth_l1 = 0
        loss_per = 0
        loss_ssim = 0
        loss_mc = 0
        gradient_penalty_loss = 0

        loss_decays = [self.loss_iter_decay ** (self.iter_num - i - 1) for i in range(self.iter_num)]
        loss_decays = [i / sum(loss_decays) for i in loss_decays]


        for i in range(self.iter_num):
            loss_decay = loss_decays[i]
            dehazed = output['dehazed'][i]
            loss_l1 += self.l1_loss(dehazed, self.clear) * loss_decay
            loss_l2 += self.l2_loss(dehazed, self.clear) * loss_decay
            temp = self.smooth_l1_loss(dehazed, self.clear) * loss_decay
            loss_smooth_l1 += temp
            loss_per_temp = self.perceputal_loss(dehazed, self.clear)[0]
            gradient_penalty_loss += self.gradient_penalty_loss(temp, self.haze)
            if loss_per_temp:
                loss_per += loss_per_temp * loss_decay
            loss_ssim += self.ssim_loss(dehazed, self.clear) * loss_decay
            loss_mc += self.mc_loss(dehazed, self.clear) * loss_decay

        # pdb.set_trace()
        if loss_per:
            loss_total = loss_l1 + loss_l2 + loss_smooth_l1 + loss_ssim + loss_mc + loss_per + gradient_penalty_loss
            loss_exp_dict = {
                'loss_l1': loss_l1,
                'loss_mse': loss_l2,
                'loss_smooth_l1': loss_smooth_l1,
                'loss_per': loss_per,
                'loss_ssim': loss_ssim,
                'loss_mc': loss_mc,
                'loss_gradient_penalty': gradient_penalty_loss,
                'loss_total': loss_total

            }
        else:
            loss_total = loss_l1 + loss_l2 + loss_smooth_l1 + loss_ssim + loss_mc + gradient_penalty_loss
            loss_exp_dict = {
                'loss_l1': loss_l1,
                'loss_mse': loss_l2,
                'loss_smooth_l1': loss_smooth_l1,
                'loss_ssim': loss_ssim,
                'loss_mc': loss_mc,
                'loss_gradient_penalty': gradient_penalty_loss,
                'loss_total': loss_total

            }

        return loss_exp_dict

    def test(self):
        self.net_g.eval()

        with torch.no_grad():
            self.output = self.net_g(self.haze)

        self.net_g.train()


    def core_validation(self, dataloader, current_iter,
                        tb_logger, save_img):

        start_time = time.time()
        test_num = 0
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        metric_results = {
            metric: 0
            for metric in self.opt['val']['metrics'].keys()
        }
        N = len(dataloader)
        set_ = set(range(0, N, int(N/10) if N > 10 else 1))
        if self.opt['rank'] == 0:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_names = [osp.splitext(osp.basename(name))[0]
                         for name in val_data['name']]
            test_num += len(img_names)
            self.feed_data(val_data)

            #log time consume
            data_time = time.time() - start_time

            visuals = self.get_current_visuals()
            #log prediction time consume
            pred_time = time.time() - data_time - start_time


            img_name = img_names[0]
            haze_img = visuals['haze'][0]
            clear_img = visuals['clear'][0]
            dehazed_img = visuals['dehazed'][0]

            torch.cuda.empty_cache()

            if save_img:
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
                dehazed_img_path = base_path + '.png'

                if self.phase == 'train':
                    if idx in set_ and tb_logger:
                        tb_img = torch.cat((haze_img, dehazed_img, clear_img), dim=-1)
                        # tb_img = tb_img * 0.5 + 0.5
                        tb_img = torch.clamp(tb_img, 0, 1)
                        tb_logger.add_image(base_path, tb_img, global_step=current_iter, dataformats='CHW')
                else:
                    # dehazed_img = dehazed_img * 0.5 + 0.5
                    dehazed_img = torch.clamp(dehazed_img, 0, 1)
                    torchvision.utils.save_image(dehazed_img, dehazed_img_path)
                    # io.imsave(dehazed_img_path, dehazed_img)


            #log visualization consume
            vis_time = time.time() - start_time - pred_time - data_time


            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                for name, opt_ in opt_metric.items():
                    metric_type = opt_.pop('type')
                    for img_index in range(len(img_names)):
                        img_out = F.relu(self.output['dehazed'][-1])
                        img_gt = self.clear
                        img_name = img_names[img_index]
                        metric_results[name] += getattr(
                            metric_module, metric_type)(**opt_)(img_out, img_gt)
                        if self.opt['rank'] == 0:
                            pbar.set_description(f'Test {img_name}')

            #log metric consume
            metric_time = time.time() - start_time - data_time - pred_time - vis_time
            cost_time = {'data_time': data_time, 'pred_time': pred_time, 'vis_time': vis_time, 'metric_time': metric_time}
            logger = get_root_logger()
            logger.info(repr(cost_time))
            start_time = time.time()
            if self.opt['rank'] == 0:
                pbar.update(1)
        if self.opt['rank'] == 0:
            pbar.close()

        return with_metrics, metric_results, test_num


    def get_current_visuals(self):
        self.test()
        out_dict = OrderedDict()
        out_dict['haze'] = self.haze.detach().cpu()
        out_dict['clear'] = self.clear.detach().cpu()
        out_dict['dehazed'] = self.output['dehazed'][-1].detach().cpu()

        return out_dict

