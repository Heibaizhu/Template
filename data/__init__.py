# -*- coding: utf-8 -*-

import importlib
import numpy as np
import random
import math
from functools import partial
import os.path as osp

import torch
import torch.utils.data

from data.data_sampler import EnlargedSampler
from utils.misc import scandir
from utils.logger import get_root_logger
from utils.dist_utils import get_dist_info
from data.prefetch_dataloader import PrefetchDataLoader
from torch.utils.data import distributed


__all__ = ['create_train_val_dataloader']

# automatically scan and import dataset modules
# scan all the files under the data folder with '_dataset' in file names
data_folder = osp.dirname(osp.abspath(__file__))
dataset_filenames = [
    osp.splitext(osp.basename(v))[0] for v in scandir(data_folder)
    if v.endswith('_dataset.py')
]

# import all the dataset modules
_dataset_modules = [
    importlib.import_module(f'data.{file_name}')
    for file_name in dataset_filenames
]


def create_dataloader(dataset,
                      dataset_opt,
                      num_gpu=1,
                      dist=False,
                      sampler=None,
                      seed=None):
    """Create dataloader.

    Args:
        dataset (torch.utils.data.Dataset): Dataset.
        dataset_opt (dict): Dataset options. It contains the following keys:
            phase (str): 'train' or 'val'.
            num_worker_per_gpu (int): Number of workers for each GPU.
            batch_size_per_gpu (int): Training batch size for each GPU.
        num_gpu (int): Number of GPUs. Used only in the train phase.
            Default: 1.
        dist (bool): Whether in distributed training. Used only in the train
            phase. Default: False.
        sampler (torch.utils.data.sampler): Data sampler. Default: None.
        seed (int | None): Seed. Default: None
    """
    phase = dataset_opt['phase']
    rank, _ = get_dist_info()
    if phase == 'train':
        if dist:  # distributed training
            batch_size = dataset_opt['batch_size_per_gpu']
            num_workers = dataset_opt['num_worker_per_gpu']
        else:  # non-distributed training
            multiplier = 1 if num_gpu == 0 else num_gpu
            batch_size = dataset_opt['batch_size_per_gpu'] * multiplier
            num_workers = dataset_opt['num_worker_per_gpu'] * multiplier
        dataloader_args = dict(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            sampler=sampler,
            drop_last=True)
        if sampler is None:
            dataloader_args['shuffle'] = True
        dataloader_args['worker_init_fn'] = partial(
            worker_init_fn, num_workers=num_workers, rank=rank,
            seed=seed) if seed is not None else None

    # TODO: change the default validation setting
    elif 'val' in phase or 'test' in phase:
    # elif phase in ['val', 'test']:  # validation
        dataloader_args = dict(
            dataset=dataset,
            batch_size=dataset_opt.get('batch_size_per_gpu', 1),
            shuffle=False,
            num_workers=dataset_opt.get('num_worker_per_gpu', 1),
            sampler=sampler
            )
    else:
        raise ValueError(f'Wrong dataset phase: {phase}. '
                         "Supported ones are 'train', 'val' and 'test'.")

    dataloader_args['pin_memory'] = dataset_opt.get('pin_memory', False)

    prefetch_mode = dataset_opt.get('prefetch_mode')
    if prefetch_mode == 'cpu':  # CPUPrefetcher
        num_prefetch_queue = dataset_opt.get('num_prefetch_queue', 1)
        logger = get_root_logger()
        logger.info(f'Use {prefetch_mode} prefetch dataloader: '
                    f'num_prefetch_queue = {num_prefetch_queue}')
        return PrefetchDataLoader(
            num_prefetch_queue=num_prefetch_queue, **dataloader_args)
    else:
        # prefetch_mode=None: Normal dataloader
        # prefetch_mode='cuda': dataloader for CUDAPrefetcher
        return torch.utils.data.DataLoader(**dataloader_args)


def create_dataset(dataset_opt):

    dataset_type = dataset_opt['type']

    # dynamic instantiation
    dataset_cls = None
    for module in _dataset_modules:
        dataset_cls = getattr(module, dataset_type, None)
        if dataset_cls is not None:
            break
    if dataset_cls is None:
        raise ValueError(f'Dataset {dataset_type} is not found.')

    dataset = dataset_cls(dataset_opt)

    logger = get_root_logger()
    logger.info(
        f'Dataset {dataset.__class__.__name__} - {dataset_opt["name"]} '
        'is created.')
    return dataset


def create_train_val_dataloader(opt, logger):
    """Create train validation dataloaders."""
    train_loader, val_loader = None, None
    train_sampler = None
    total_epochs = -1
    total_iters = -1

    # TODO: support multi validation dataloader
    val_loader_list = []
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = create_dataset(dataset_opt)
            train_sampler = EnlargedSampler(train_set, opt['world_size'],
                                            opt['rank'], dataset_enlarge_ratio)

            train_loader = create_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])

            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio /
                (dataset_opt['batch_size_per_gpu'] * opt['world_size']))

            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / num_iter_per_epoch)
            logger.info(
                'Training statistics:'
                f'\n\tNumber of train images: {len(train_set)}'
                f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                f'\n\tWorld size (gpu number): {opt["world_size"]}'
                f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')

        elif 'val' in phase:  # val1, val2...
            val_set = create_dataset(dataset_opt)
            if opt['dist']:
                val_sampler = distributed.DistributedSampler(val_set,
                                                             num_replicas=opt["world_size"],
                                                             rank=opt['rank'],
                                                             shuffle=False)
            else:
                val_sampler = None
            val_loader = create_dataloader(
                val_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=val_sampler,
                seed=opt['manual_seed'])
            logger.info(
                f'Number of val images/folders in {dataset_opt["name"]}: '
                f'{len(val_set)}')

            val_loader_list.append(val_loader)

        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, train_sampler, val_loader_list, total_epochs, total_iters


def create_eval_dataloader(dataset_opt, opt):
    eval_set = create_dataset(dataset_opt)
    if opt['dist']:
        val_sampler = distributed.DistributedSampler(eval_set,
                                                     num_replicas=opt["world_size"],
                                                     rank=opt['rank'],
                                                     shuffle=False)
    else:
        val_sampler = None

    eval_loader = create_dataloader(
        eval_set,
        dataset_opt,
        dist=opt['dist'],
        sampler=val_sampler,
        seed=opt['manual_seed']
    )
    len_eval = len(eval_set)

    return len_eval, eval_loader


def worker_init_fn(worker_id, num_workers, rank, seed):
    # Set the worker seed to num_workers * rank + worker_id + seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)