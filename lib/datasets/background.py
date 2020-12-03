# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import torch
import torchvision
import torch.utils.data as data
import os, math
import os.path as osp
from os.path import *
import numpy as np
import numpy.random as npr
import cv2
import datasets
from fcn.config import cfg
from utils.blob import chromatic_transform, add_noise, add_noise_depth

class BackgroundDataset(data.Dataset, datasets.imdb):

    def __init__(self, name):

        self._name = name
        self.files_color = []
        self.files_depth = []

        if name == 'coco':
            background_dir = os.path.join(self.cache_path, '../coco/train2014/train2014')
            for filename in os.listdir(background_dir):
                self.files_color.append(os.path.join(background_dir, filename))
            self.files_color.sort()

        elif name == 'texture':
            background_dir = os.path.join(self.cache_path, '../textures')
            for filename in os.listdir(background_dir):
                self.files_color.append(os.path.join(background_dir, filename))
            self.files_color.sort()

        elif name == 'nvidia':
            allencenter = os.path.join(self.cache_path, '../backgrounds/nvidia')
            subdirs = os.listdir(allencenter)
            for i in range(len(subdirs)):
                subdir = subdirs[i]
                files = os.listdir(os.path.join(allencenter, subdir))
                for j in range(len(files)):
                    filename = os.path.join(allencenter, subdir, files[j])
                    self.files_color.append(filename)
            self.files_color.sort()

        elif name == 'table':
            allencenter = os.path.join(self.cache_path, '../backgrounds/table')
            subdirs = os.listdir(allencenter)
            for i in range(len(subdirs)):
                subdir = subdirs[i]
                files = os.listdir(os.path.join(allencenter, subdir))
                for j in range(len(files)):
                    filename = os.path.join(allencenter, subdir, files[j])
                    self.files_color.append(filename)
            self.files_color.sort()

        elif name == 'isaac':
            allencenter = os.path.join(self.cache_path, '../backgrounds/isaac')
            subdirs = os.listdir(allencenter)
            for i in range(len(subdirs)):
                subdir = subdirs[i]
                files = os.listdir(os.path.join(allencenter, subdir))
                for j in range(len(files)):
                    filename = os.path.join(allencenter, subdir, files[j])
                    self.files_color.append(filename)
            self.files_color.sort()

        elif name == 'rgbd':
            comotion = os.path.join(self.cache_path, '../backgrounds/rgbd')
            subdirs = os.listdir(comotion)
            for i in range(len(subdirs)):
                subdir = subdirs[i]
                files = os.listdir(os.path.join(comotion, subdir))
                for j in range(len(files)):
                    filename = os.path.join(comotion, subdir, files[j])
                    if 'depth.png' in filename:
                        self.files_depth.append(filename)
                    else:
                        self.files_color.append(filename)

            self.files_color.sort()
            self.files_depth.sort()

        self._intrinsic_matrix = np.array([[524.7917885754071, 0, 332.5213232846151],
                                          [0, 489.3563960810721, 281.2339855172282],
                                          [0, 0, 1]])

        self.num = len(self.files_color)
        self.subtract_mean = cfg.TRAIN.SYN_BACKGROUND_SUBTRACT_MEAN
        if cfg.TRAIN.SYN_CROP:
            self._height = cfg.TRAIN.SYN_CROP_SIZE
            self._width = cfg.TRAIN.SYN_CROP_SIZE
        else:
            self._height = cfg.TRAIN.SYN_HEIGHT
            self._width = cfg.TRAIN.SYN_WIDTH
        self._pixel_mean = cfg.PIXEL_MEANS
        print('{} background images'.format(self.num))


    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        filename_color = self.files_color[idx]
        if self.name == 'rgbd':
            filename_depth = self.files_depth[idx]
        else:
            filename_depth = None
        return self.load(filename_color, filename_depth)

    def load(self, filename_color, filename_depth):
        if filename_depth is None:
            background_depth = np.zeros((3, self._height, self._width), dtype=np.float32)
            mask_depth = np.zeros((self._height, self._width), dtype=np.float32)

        if filename_depth is None and np.random.rand(1) < cfg.TRAIN.SYN_BACKGROUND_CONSTANT_PROB:  # only for rgb cases
            # constant background image
            background_color = np.ones((self._height, self._width, 3), dtype=np.uint8)
            color = np.random.randint(256, size=3)
            background_color[:, :, 0] = color[0]
            background_color[:, :, 1] = color[1]
            background_color[:, :, 2] = color[2]
        else:
            background_color = cv2.imread(filename_color, cv2.IMREAD_UNCHANGED)

            if filename_depth is not None:
                background_depth = cv2.imread(filename_depth, cv2.IMREAD_UNCHANGED)

            try:
                # randomly crop a region as background
                bw = background_color.shape[1]
                bh = background_color.shape[0]
                x1 = npr.randint(0, int(bw/3))
                y1 = npr.randint(0, int(bh/3))
                x2 = npr.randint(int(2*bw/3), bw)
                y2 = npr.randint(int(2*bh/3), bh)
                background_color = background_color[y1:y2, x1:x2]
                background_color = cv2.resize(background_color, (self._width, self._height), interpolation=cv2.INTER_LINEAR)
                if len(background_color.shape) != 3:
                    background_color = cv2.cvtColor(background_color, cv2.COLOR_GRAY2RGB)

                if filename_depth is not None:
                    background_depth = background_depth[y1:y2, x1:x2]
                    background_depth = cv2.resize(background_depth, (self._width, self._height), interpolation=cv2.INTER_NEAREST)
                    background_depth = self.backproject(background_depth, self._intrinsic_matrix, 1000.0)

            except:
                background_color = np.zeros((self._height, self._width, 3), dtype=np.uint8)
                print('bad background_color image', filename_color)
                if filename_depth is not None:
                    background_depth = np.zeros((self._height, self._width, 3), dtype=np.float32)
                    print('bad depth background image')

            if len(background_color.shape) != 3:
                background_color = np.zeros((self._height, self._width, 3), dtype=np.uint8)
                print('bad background_color image', filename_color)

            if filename_depth is not None:
                if len(background_depth.shape) != 3:
                    background_depth = np.zeros((self._height, self._width, 3), dtype=np.float32)
                    print('bad depth background image')

                z_im = background_depth[:, :, 2]
                mask_depth = z_im > 0.0
                mask_depth = mask_depth.astype(np.float32)

                if np.random.rand(1) > 0.1:
                    background_depth = add_noise_depth(background_depth)

                background_depth = background_depth.transpose(2, 0, 1).astype(np.float32)

            if np.random.rand(1) > 0.1:
                background_color = chromatic_transform(background_color)

        if np.random.rand(1) > 0.1:
            background_color = add_noise(background_color)

        background_color = background_color.astype(np.float32)

        if self.subtract_mean:
            background_color -= self._pixel_mean
        background_color = background_color.transpose(2, 0, 1) / 255.0

        sample = {'background_color': background_color,
                  'background_depth': background_depth,
                  'mask_depth': mask_depth}

        return sample
