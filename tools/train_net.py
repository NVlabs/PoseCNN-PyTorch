#!/usr/bin/env python3

# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

"""Train a PoseCNN on an image database."""

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

import argparse
import pprint
import numpy as np
import sys
import os
import os.path as osp
import cv2

import _init_paths
import datasets
import networks
from fcn.config import cfg, cfg_from_file, get_output_dir
from fcn.train import train
from datasets.factory import get_dataset

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a PoseCNN network')
    parser.add_argument('--epochs', dest='epochs',
                        help='number of epochs to train',
                        default=40000, type=int)
    parser.add_argument('--startepoch', dest='startepoch',
                        help='the starting epoch',
                        default=0, type=int)
    parser.add_argument('--pretrained', dest='pretrained',
                        help='initialize with pretrained checkpoint',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--solver', dest='solver',
                        help='solver type',
                        default='sgd', type=str)
    parser.add_argument('--dataset', dest='dataset_name',
                        help='dataset to train on',
                        default='shapenet_scene_train', type=str)
    parser.add_argument('--dataset_background', dest='dataset_background_name',
                        help='background dataset to train on',
                        default='background_nvidia', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--network', dest='network_name',
                        help='name of the network',
                        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)

    # prepare dataset
    cfg.MODE = 'TRAIN'
    dataset = get_dataset(args.dataset_name)
    worker_init_fn = dataset.worker_init_fn if hasattr(dataset, 'worker_init_fn') else None
    if cfg.TRAIN.SYNTHESIZE:
        num_workers = 0
    else:
        num_workers = 4
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.TRAIN.IMS_PER_BATCH, shuffle=True, 
        num_workers=num_workers, worker_init_fn=worker_init_fn)
    print('Use dataset `{:s}` for training'.format(dataset.name))

    # background dataset
    if cfg.TRAIN.SYN_BACKGROUND_SPECIFIC:
        background_dataset = get_dataset(args.dataset_background_name)
    else:
        background_dataset = get_dataset('background_coco')
    background_loader = torch.utils.data.DataLoader(background_dataset, batch_size=cfg.TRAIN.IMS_PER_BATCH,
                                                    shuffle=True, num_workers=4)

    # overwrite intrinsics
    if len(cfg.INTRINSICS) > 0:
        K = np.array(cfg.INTRINSICS).reshape(3, 3)
        dataset._intrinsic_matrix = K
        background_dataset._intrinsic_matrix = K
        print(dataset._intrinsic_matrix)

    # output directory
    output_dir = get_output_dir(dataset, None)
    print('Output will be saved to `{:s}`'.format(output_dir))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # prepare network
    if args.pretrained:
        network_data = torch.load(args.pretrained)
        if isinstance(network_data, dict) and 'model' in network_data:
            network_data = network_data['model']
        print("=> using pre-trained network '{}'".format(args.network_name))
    else:
        network_data = None
        print("=> creating network '{}'".format(args.network_name))

    network = networks.__dict__[args.network_name](dataset.num_classes, cfg.TRAIN.NUM_UNITS, network_data).cuda()
    if torch.cuda.device_count() > 1:
        cfg.TRAIN.GPUNUM = torch.cuda.device_count()
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    network = torch.nn.DataParallel(network).cuda()
    cudnn.benchmark = True

    # renderer
    if cfg.TRAIN.SYNTHESIZE:
        from ycb_renderer import YCBRenderer
        print('loading 3D models')
        cfg.renderer = YCBRenderer(width=cfg.TRAIN.SYN_WIDTH, height=cfg.TRAIN.SYN_HEIGHT, render_marker=False)
        cfg.renderer.load_objects(dataset.model_mesh_paths, dataset.model_texture_paths, dataset.model_colors)
        cfg.renderer.set_camera_default()
        print(dataset.model_mesh_paths)

    # optimizer
    assert(args.solver in ['adam', 'sgd'])
    print('=> setting {} solver'.format(args.solver))
    param_groups = [{'params': network.module.bias_parameters(), 'weight_decay': cfg.TRAIN.WEIGHT_DECAY},
                    {'params': network.module.weight_parameters(), 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    if args.solver == 'adam':
        optimizer = torch.optim.Adam(param_groups, cfg.TRAIN.LEARNING_RATE,
                                     betas=(cfg.TRAIN.MOMENTUM, cfg.TRAIN.BETA))
    elif args.solver == 'sgd':
        optimizer = torch.optim.SGD(param_groups, cfg.TRAIN.LEARNING_RATE,
                                    momentum=cfg.TRAIN.MOMENTUM)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                        milestones=[m - args.startepoch for m in cfg.TRAIN.MILESTONES], gamma=cfg.TRAIN.GAMMA)
    cfg.epochs = args.epochs

    # main loop
    for epoch in range(args.startepoch, args.epochs):
        if args.solver == 'sgd':
            scheduler.step()

        train(dataloader, background_loader, network, optimizer, epoch)

        # save checkpoint
        if (epoch+1) % cfg.TRAIN.SNAPSHOT_EPOCHS == 0 or epoch == args.epochs - 1:
            state = network.module.state_dict()
            infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                     if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
            filename = (cfg.TRAIN.SNAPSHOT_PREFIX + infix + '_epoch_{:d}'.format(epoch+1) + '.checkpoint.pth')
            torch.save(state, os.path.join(output_dir, filename))
            print(filename)
