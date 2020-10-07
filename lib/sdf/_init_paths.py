# --------------------------------------------------------
# PoseCNN
# Copyright (c) 2020 NVIDIA
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# ---------------------------------------------------------

"""Set up paths for Fast R-CNN."""

import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)
lib_path = osp.join(this_dir, '..')
add_path(lib_path)
