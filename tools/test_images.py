#!/usr/bin/env python3

# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

"""Test a PoseCNN on images"""

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data

import argparse
import pprint
import time, os, sys
import os.path as osp
import numpy as np
import cv2
import scipy.io
import glob

import _init_paths
from fcn.test_imageset import test_image
from fcn.config import cfg, cfg_from_file, yaml_from_file, get_output_dir
from datasets.factory import get_dataset
import networks
from ycb_renderer import YCBRenderer
from utils.blob import pad_im
from sdf.sdf_optimizer import sdf_optimizer

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a PoseCNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--pretrained', dest='pretrained',
                        help='initialize with pretrained checkpoint',
                        default=None, type=str)
    parser.add_argument('--pretrained_encoder', dest='pretrained_encoder',
                        help='initialize with pretrained encoder checkpoint',
                        default=None, type=str)
    parser.add_argument('--codebook', dest='codebook',
                        help='codebook',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--meta', dest='meta_file',
                        help='optional metadata file', default=None, type=str)
    parser.add_argument('--dataset', dest='dataset_name',
                        help='dataset to train on',
                        default='shapenet_scene_train', type=str)
    parser.add_argument('--depth', dest='depth_name',
                        help='depth image pattern',
                        default='*depth.png', type=str)
    parser.add_argument('--color', dest='color_name',
                        help='color image pattern',
                        default='*color.png', type=str)
    parser.add_argument('--imgdir', dest='imgdir',
                        help='path of the directory with the test images',
                        default='data/Images', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--network', dest='network_name',
                        help='name of the network',
                        default=None, type=str)
    parser.add_argument('--background', dest='background_name',
                        help='name of the background file',
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

    if len(cfg.TEST.CLASSES) == 0:
        cfg.TEST.CLASSES = cfg.TRAIN.CLASSES

    if args.meta_file is not None:
        meta = yaml_from_file(args.meta_file)
        # overwrite test classes
        print(meta)
        if 'ycb_ids' in meta:
            cfg.TEST.CLASSES = [0]
            for i in meta.ycb_ids:
                cfg.TEST.CLASSES.append(i)
            print('TEST CLASSES:', cfg.TEST.CLASSES)
        if 'INTRINSICS' in meta:
            cfg.INTRINSICS = meta['INTRINSICS']

    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)

    # device
    cfg.gpu_id = args.gpu_id
    cfg.device = torch.device('cuda:{:d}'.format(cfg.gpu_id))
    cfg.instance_id = 0
    print('GPU device {:d}'.format(args.gpu_id))

    # dataset
    cfg.MODE = 'TEST'
    cfg.TEST.SYNTHESIZE = False
    dataset = get_dataset(args.dataset_name)

    # overwrite intrinsics
    if len(cfg.INTRINSICS) > 0:
        K = np.array(cfg.INTRINSICS).reshape(3, 3)
        if cfg.TEST.SCALES_BASE[0] != 1:
            scale = cfg.TEST.SCALES_BASE[0]
            K[0, 0] *= scale
            K[0, 2] *= scale
            K[1, 1] *= scale
            K[1, 2] *= scale
        dataset._intrinsic_matrix = K
        print(dataset._intrinsic_matrix)

    # list images
    images_color = []
    filename = os.path.join(args.imgdir, args.color_name)
    files = glob.glob(filename)
    for i in range(len(files)):
        filename = files[i]
        images_color.append(filename)
    images_color.sort()

    images_depth = []
    filename = os.path.join(args.imgdir, args.depth_name)
    files = glob.glob(filename)
    for i in range(len(files)):
        filename = files[i]
        images_depth.append(filename)
    images_depth.sort()

    if cfg.TEST.VISUALIZE:
        index_images = np.random.permutation(len(images_color))
    else:
        index_images = range(len(images_color))
        resdir = args.imgdir + '_posecnn_results'
        if not os.path.exists(resdir):
            os.makedirs(resdir)

    # prepare network
    if args.pretrained:
        network_data = torch.load(args.pretrained)
        print("=> using pre-trained network '{}'".format(args.pretrained))
    else:
        network_data = None
        print("no pretrained network specified")
        sys.exit()

    network = networks.__dict__[args.network_name](dataset.num_classes, cfg.TRAIN.NUM_UNITS, network_data).cuda(device=cfg.device)
    network = torch.nn.DataParallel(network, device_ids=[cfg.gpu_id]).cuda(device=cfg.device)
    cudnn.benchmark = True
    network.eval()

    print('loading 3D models')
    cfg.renderer = YCBRenderer(width=cfg.TRAIN.SYN_WIDTH, height=cfg.TRAIN.SYN_HEIGHT, gpu_id=args.gpu_id, render_marker=False)
    if cfg.TEST.SYNTHESIZE:
        cfg.renderer.load_objects(dataset.model_mesh_paths, dataset.model_texture_paths, dataset.model_colors)
    else:
        model_mesh_paths = [dataset.model_mesh_paths[i-1] for i in cfg.TEST.CLASSES[1:]]
        model_texture_paths = [dataset.model_texture_paths[i-1] for i in cfg.TEST.CLASSES[1:]]
        model_colors = [dataset.model_colors[i-1] for i in cfg.TEST.CLASSES[1:]]
        cfg.renderer.load_objects(model_mesh_paths, model_texture_paths, model_colors)
    cfg.renderer.set_camera_default()
    print(dataset.model_mesh_paths)

    # load sdfs
    if cfg.TEST.POSE_REFINE:
        print('loading SDFs')
        sdf_files = []
        for i in cfg.TEST.CLASSES[1:]:
            sdf_files.append(dataset.model_sdf_paths[i-1])
        cfg.sdf_optimizer = sdf_optimizer(cfg.TEST.CLASSES[1:], sdf_files)

    # for each image
    for i in index_images:
        im = pad_im(cv2.imread(images_color[i], cv2.IMREAD_COLOR), 16)
        print(images_color[i])
        if len(images_depth) > 0 and osp.exists(images_depth[i]):
            depth = pad_im(cv2.imread(images_depth[i], cv2.IMREAD_UNCHANGED), 16)
            depth = depth.astype('float') / 1000.0
            print(images_depth[i])
        else:
            depth = None
            print('no depth image')

        # rescale image if necessary
        if cfg.TEST.SCALES_BASE[0] != 1:
            im_scale = cfg.TEST.SCALES_BASE[0]
            im = pad_im(cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR), 16)
            if depth is not None:
                depth = pad_im(cv2.resize(depth, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_NEAREST), 16)

        # run network
        im_pose, im_pose_refined, im_label, labels, rois, poses, poses_refined = test_image(network, dataset, im, depth)

        # save result
        if not cfg.TEST.VISUALIZE:

            # map the roi index
            for j in range(rois.shape[0]):
                rois[j, 1] = cfg.TRAIN.CLASSES.index(cfg.TEST.CLASSES[int(rois[j, 1])])

            result = {'labels': labels, 'rois': rois, 'poses': poses, 'poses_refined': poses_refined, 'intrinsic_matrix': dataset._intrinsic_matrix}
            head, tail = os.path.split(images_color[i])
            filename = os.path.join(resdir, tail + '.mat')
            scipy.io.savemat(filename, result, do_compression=True)
            # rendered image
            filename = os.path.join(resdir, tail + '_render.jpg')
            cv2.imwrite(filename, im_pose[:, :, (2, 1, 0)])
            filename = os.path.join(resdir, tail + '_render_refined.jpg')
            cv2.imwrite(filename, im_pose_refined[:, :, (2, 1, 0)])
