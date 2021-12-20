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

    # create visualizer for saving training data
    image_visualizer = Visualizer(opt)

    # create train and validation dataloaders
    data_packet = create_train_val_dataloader(opt, text_logger)
    train_loader, train_sampler, val_loader_list, total_epochs, total_iters = data_packet

    # create model
    if resume_state:
        check_resume(opt, resume_state['iter'])
        model = create_model(opt)

        # handle optimizers and schedulers
        model.resume_training(resume_state)
        text_logger.info("Resuming training from epoch: {}, iter: {}.".format(resume_state['epoch'], resume_state['iter']))
        # text_logger.info(f"Resuming training from epoch: {resume_state['epoch']}, "
        #                  f"iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
    else:
        model = create_model(opt)
        start_epoch = 0
        current_iter = 0
    # model.tb_logger = tb_logger
    if hasattr(model.net_g, 'log_grad'):
        model.net_g.log_grad(tb_logger)
    # create message logger (formatted outputs)
    msg_logger = Logger(opt, current_iter, tb_logger)

    # dataloader prefetcher
    prefetcher = create_prefetch(train_loader, opt, text_logger)

    # training
    text_logger.info(
        'Start training from epoch: {}, iter: {}'.format(start_epoch, current_iter))
    data_time, iter_time = time.time(), time.time()

    start_time = time.time()

    # for val_loader in val_loader_list:
    #     model.validation(val_loader,
    #                      current_iter,
    #                      tb_logger,
    #                      opt['val']['save_img'])
    #
    #
    #
    # for val_loader in val_loader_list:
    #     model.validation(val_loader,
    #                      current_iter,
    #                      tb_logger,
    #                      opt['val']['save_img'])

    for epoch in range(start_epoch, total_epochs + 1):
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()


        loss_total = 0

        while train_data is not None:
            data_time = time.time() - data_time

            current_iter += 1
            if current_iter > total_iters:
                break
            # update learning rate
            model.update_learning_rate(
                current_iter, warmup_iter=opt['train'].get('warmup_iter', -1), loss_total=loss_total)
            # training
            model.feed_data(train_data)
            loss_total = model.optimize_parameters(current_iter)
            torch.cuda.synchronize()
            iter_time = time.time() - iter_time
            # log
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update({'time': iter_time, 'data_time': data_time})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)
                image_visualizer(model.get_current_visuals(),
                                 epoch)

            # save models and training states
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                text_logger.info('Saving models and training states.')
                model.save(epoch, current_iter)

            if opt.get('val') is not None and (current_iter %
                                               opt['val']['val_freq'] == 0):
                for val_loader in val_loader_list:
                    model.validation(val_loader,
                                     current_iter,
                                     tb_logger,
                                     opt['val']['save_img'])

            data_time = time.time()
            iter_time = time.time()
            train_data = prefetcher.next()
        # end of iter

    # end of epoch

    consumed_time = str(
        datetime.timedelta(seconds=int(time.time() - start_time)))
    text_logger.info('End of training. Time consumed: {}'.format(consumed_time))
    text_logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest
    if opt.get('val') is not None:
        for val_loader in val_loader_list:
            model.validation(val_loader,
                             current_iter,
                             tb_logger,
                             opt['val']['save_img'])
    if tb_logger:
        tb_logger.close()


if __name__ == '__main__':
    main()
    # try:
    #     with autograd.detect_anomaly():
    #         main()
    # except:
    #     pdb.post_mortem()

