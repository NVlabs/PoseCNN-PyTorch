# --------------------------------------------------------
# PoseCNN
# Copyright (c) 2020 NVIDIA
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

from .imdb import imdb
from .ycb_video import YCBVideo
from .ycb_self_supervision import YCBSelfSupervision
from .ycb_object import YCBObject
from .background import BackgroundDataset

import os.path as osp
ROOT_DIR = osp.join(osp.dirname(__file__), '..', '..')
