# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

"""Factory method for easily getting imdbs by name."""

__sets = {}

import datasets.ycb_video
import datasets.ycb_object
import datasets.ycb_self_supervision
import datasets.dex_ycb
import datasets.background
import numpy as np

# ycb video dataset
for split in ['train', 'val', 'keyframe', 'trainval', 'debug']:
    name = 'ycb_video_{}'.format(split)
    print(name)
    __sets[name] = (lambda split=split:
            datasets.YCBVideo(split))

# ycb object dataset
for split in ['train', 'test']:
    name = 'ycb_object_{}'.format(split)
    print(name)
    __sets[name] = (lambda split=split:
            datasets.YCBObject(split))

# ycb self supervision dataset
for split in ['train_1', 'train_2', 'train_3', 'train_4', 'train_5', 'test', 'all', 'train_block_median', 'train_block_median_azure', 'train_block_median_demo', 'train_block_median_azure_demo', 'train_table',
              'debug', 'train_block', 'train_block_azure', 'train_block_big_sim', 'train_block_median_sim', 'train_block_small_sim']:
    name = 'ycb_self_supervision_{}'.format(split)
    print(name)
    __sets[name] = (lambda split=split:
            datasets.YCBSelfSupervision(split))

# background dataset
for split in ['coco', 'rgbd', 'nvidia', 'table', 'isaac', 'texture']:
    name = 'background_{}'.format(split)
    print(name)
    __sets[name] = (lambda split=split:
            datasets.BackgroundDataset(split))


# DEX YCB dataset
for setup in ('s0', 's1', 's2', 's3'):
    for split in ('train', 'val', 'test'):
        name = 'dex_ycb_{}_{}'.format(setup, split)
        __sets[name] = (lambda setup=setup, split=split: datasets.DexYCBDataset(setup, split))


def get_dataset(name):
    """Get an imdb (image database) by name."""
    if name not in __sets:
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_datasets():
    """List all registered imdbs."""
    return __sets.keys()
