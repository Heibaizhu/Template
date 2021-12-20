# -*- coding: utf-8 -*-

import datetime
import time
import pdb

import torch
from torch import autograd

from utils.options import Options
from utils.misc import load_resume_state, make_exp_dirs, check_resume
from utils.logger import Logger
from utils.visualizer import Visualizer
from data import create_train_val_dataloader
from data.prefetch_dataloader import create_prefetch
from models import create_model


def main():

    torch.backends.cudnn.benchmark = True

    opt = Options.parse_options()

    resume_state = load_resume_state(opt)

    if resume_state is None:
        make_exp_dirs(opt)

    # since text_logger will be used when creating dataloaders,
    # so, create loggers in advance
    # tb_logger  tensorboard logger
    text_logger, tb_logger = Logger.create_log(opt)

    # create train and validation dataloaders
    data_packet = create_train_val_dataloader(opt, text_logger)
    train_loader, train_sampler, val_loader_list, total_epochs, total_iters = data_packet

    # create model
    if resume_state:
        check_resume(opt, resume_state['iter'])
        model = create_model(opt)
        # handle optimizers and schedulers
        model.resume_training(resume_state)
        text_logger.info("Resuming test from epoch: {}, iter: {}.".format(resume_state['epoch'], resume_state['iter']))
        # text_logger.info(f"Resuming training from epoch: {resume_state['epoch']}, "
        #                  f"iter: {resume_state['iter']}.")
        current_iter = resume_state['iter']
    else:
        model = create_model(opt)
        current_iter = 0


    if opt.get('val') is not None:
        for val_loader in val_loader_list:
            start_time = time.time()
            model.validation(val_loader,
                             current_iter,
                             tb_logger,
                             opt['val']['save_img'])
            consumed_time = str(
                datetime.timedelta(seconds=int(time.time() - start_time)))
            dataset_name = val_loader.dataset.opt['name']
            text_logger.info('End of testing on {}. Time consumed: {}'.format(dataset_name, consumed_time))
    if tb_logger:
        tb_logger.close()


if __name__ == '__main__':
    main()
    # try:
    #     with autograd.detect_anomaly():
    #         main()
    # except:
    #     pdb.post_mortem()

