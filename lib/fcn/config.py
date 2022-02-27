# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

"""PoseCNN config system.

This file specifies default config options for Fast R-CNN. You should not
change values in this file. Instead, you should write a config file (in yaml)
and use cfg_from_file(yaml_file) to load it and override the default options.

Most tools in $ROOT/tools take a --cfg option to specify an override file.
    - See tools/{train,test}_net.py for example code that uses cfg_from_file()
    - See experiments/cfgs/*.yml for example YAML config override files
"""

import os
import os.path as osp
import numpy as np
import math
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict
import logging

__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

__C.FLIP_X = False
__C.INPUT = 'RGBD'
__C.NETWORK = 'VGG16'
__C.RIG = ''
__C.CAD = ''
__C.POSE = ''
__C.BACKGROUND = ''
__C.USE_GPU_NMS = True
__C.MODE = 'TRAIN'
__C.INTRINSICS = ()
__C.DATA_PATH = ''

__C.FLOW_HEIGHT = 512
__C.FLOW_WIDTH = 640

# Anchor scales for RPN
__C.ANCHOR_SCALES = (8,16,32)

# Anchor ratios for RPN
__C.ANCHOR_RATIOS = (0.5,1,2)

__C.FEATURE_STRIDE = 16
__C.gpu_id = 0
__C.instance_id = 0

#
# Training options
#

__C.TRAIN = edict()

__C.TRAIN.WEIGHT_DECAY = 0.0001

__C.TRAIN.SEGMENTATION = True
__C.TRAIN.ITERNUM = 4
__C.TRAIN.HEATUP = 4
__C.TRAIN.GPUNUM = 1
__C.TRAIN.CLASSES = (0,1,2,3)
__C.TRAIN.SYMMETRY = (0,0,0,0)

__C.TRAIN.SLIM = False
__C.TRAIN.SINGLE_FRAME = False
__C.TRAIN.TRAINABLE = True
__C.TRAIN.VERTEX_REG = True
__C.TRAIN.VERTEX_REG_DELTA = False
__C.TRAIN.POSE_REG = True
__C.TRAIN.LABEL_W = 1.0
__C.TRAIN.VERTEX_W = 1.0
__C.TRAIN.VERTEX_W_INSIDE = 10.0
__C.TRAIN.POSE_W = 1.0
__C.TRAIN.BOX_W = 1.0
__C.TRAIN.HARD_LABEL_THRESHOLD = 1.0
__C.TRAIN.HARD_LABEL_SAMPLING = 1.0
__C.TRAIN.HARD_ANGLE = 15.0
__C.TRAIN.VISUALIZE = False
__C.TRAIN.GAN = False
__C.TRAIN.MATCHING = False
__C.TRAIN.NOISE_LEVEL = 0.05
__C.TRAIN.FREEZE_LAYERS = True
__C.TRAIN.MAX_ITERS_PER_EPOCH = 1000000
__C.TRAIN.UNIFORM_POSE_INTERVAL = 15
__C.TRAIN.AFFINE = False

# Hough voting
__C.TRAIN.HOUGH_LABEL_THRESHOLD = 100
__C.TRAIN.HOUGH_VOTING_THRESHOLD = -1
__C.TRAIN.HOUGH_SKIP_PIXELS = -1
__C.TRAIN.HOUGH_INLIER_THRESHOLD = 0.9

# synthetic training
__C.TRAIN.SYNTHESIZE = False
__C.TRAIN.SYN_ONLINE = False
__C.TRAIN.SYN_WIDTH = 640
__C.TRAIN.SYN_HEIGHT = 480
__C.TRAIN.SYNROOT = '/var/Projects/Deep_Pose/data/LOV/data_syn/'
if not os.path.exists(__C.TRAIN.SYNROOT):
    __C.TRAIN.SYNROOT = '/home/yuxiang/Projects/Deep_Pose/data/LOV/data_syn/'
__C.TRAIN.SYNITER = 0
__C.TRAIN.SYNNUM = 80000
__C.TRAIN.SYN_RATIO = 1
__C.TRAIN.SYN_CLASS_INDEX = 1
__C.TRAIN.SYN_TNEAR = 0.5
__C.TRAIN.SYN_TFAR = 2.0
__C.TRAIN.SYN_BACKGROUND_SPECIFIC = False
__C.TRAIN.SYN_BACKGROUND_SUBTRACT_MEAN = False
__C.TRAIN.SYN_BACKGROUND_CONSTANT_PROB = 0.1
__C.TRAIN.SYN_BACKGROUND_AFFINE = False
__C.TRAIN.SYN_SAMPLE_OBJECT = True
__C.TRAIN.SYN_SAMPLE_POSE = True
__C.TRAIN.SYN_STD_ROTATION = 15
__C.TRAIN.SYN_STD_TRANSLATION = 0.05
__C.TRAIN.SYN_MIN_OBJECT = 5
__C.TRAIN.SYN_MAX_OBJECT = 8
__C.TRAIN.SYN_TNEAR = 0.5
__C.TRAIN.SYN_TFAR = 2.0
__C.TRAIN.SYN_BOUND = 0.4
__C.TRAIN.SYN_SAMPLE_DISTRACTOR = True
__C.TRAIN.SYN_CROP = False
__C.TRAIN.SYN_CROP_SIZE = 224
__C.TRAIN.SYN_TABLE_PROB = 0.8

# autoencoder
__C.TRAIN.BOOSTRAP_PIXELS = 20

# domain adaptation
__C.TRAIN.ADAPT = False
__C.TRAIN.ADAPT_ROOT = ''
__C.TRAIN.ADAPT_NUM = 400
__C.TRAIN.ADAPT_RATIO = 1
__C.TRAIN.ADAPT_WEIGHT = 0.1

# learning rate
__C.TRAIN.OPTIMIZER = 'MOMENTUM'
__C.TRAIN.LEARNING_RATE = 0.0001
__C.TRAIN.MILESTONES = (100, 150, 200)
__C.TRAIN.MOMENTUM = 0.9
__C.TRAIN.BETA = 0.999
__C.TRAIN.GAMMA = 0.1

__C.TRAIN.SYMSIZE = 0

# voxel grid size
__C.TRAIN.GRID_SIZE = 256

# Scales to compute real features
__C.TRAIN.SCALES_BASE = (0.25, 0.5, 1.0, 2.0, 3.0)

# parameters for data augmentation
__C.TRAIN.CHROMATIC = True
__C.TRAIN.ADD_NOISE = False

# Images to use per minibatch
__C.TRAIN.IMS_PER_BATCH = 2
__C.TRAIN.NUM_STEPS = 5
__C.TRAIN.NUM_UNITS = 64

# Use horizontally-flipped images during training?
__C.TRAIN.USE_FLIPPED = True

# Iterations between snapshots
__C.TRAIN.SNAPSHOT_EPOCHS = 1

# solver.prototxt specifies the snapshot path prefix, this adds an optional
# infix to yield the path: <prefix>[_<infix>]_iters_XYZ.caffemodel
__C.TRAIN.SNAPSHOT_PREFIX = 'caffenet_fast_rcnn'
__C.TRAIN.SNAPSHOT_INFIX = ''

__C.TRAIN.DISPLAY = 20
__C.TRAIN.ITERS = 0

# Whether to add ground truth boxes to the pool when sampling regions
__C.TRAIN.USE_GT = False

# Minibatch size (number of regions of interest [ROIs])
__C.TRAIN.BATCH_SIZE = 128

# Fraction of minibatch that is labeled foreground (i.e. class > 0)
__C.TRAIN.FG_FRACTION = 0.25

# Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
__C.TRAIN.FG_THRESH = 0.5
__C.TRAIN.FG_THRESH_POSE = 0.2

# Overlap threshold for a ROI to be considered background (class = 0 if
# overlap in [LO, HI))
__C.TRAIN.BG_THRESH_HI = 0.5
__C.TRAIN.BG_THRESH_LO = 0.1


# Use RPN to detect objects
__C.TRAIN.HAS_RPN = True

# IOU >= thresh: positive example
__C.TRAIN.RPN_POSITIVE_OVERLAP = 0.7

# IOU < thresh: negative example
__C.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3

# If an anchor satisfied by positive and negative conditions set to negative
__C.TRAIN.RPN_CLOBBER_POSITIVES = False

# Max number of foreground examples
__C.TRAIN.RPN_FG_FRACTION = 0.5

# Total number of examples
__C.TRAIN.RPN_BATCHSIZE = 256

# NMS threshold used on RPN proposals
__C.TRAIN.RPN_NMS_THRESH = 0.7

# Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.TRAIN.RPN_PRE_NMS_TOP_N = 12000

# Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.TRAIN.RPN_POST_NMS_TOP_N = 2000

# Deprecated (outside weights)
__C.TRAIN.RPN_BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)

# Give the positive RPN examples weight of p * 1 / {num positives}
# and give negatives a weight of (1 - p)
# Set to -1.0 to use uniform example weighting
__C.TRAIN.RPN_POSITIVE_WEIGHT = -1.0

# Normalize the targets (subtract empirical mean, divide by empirical stddev)
__C.TRAIN.BBOX_NORMALIZE_TARGETS = True

# Deprecated (inside weights)
__C.TRAIN.BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)

# Normalize the targets using "precomputed" (or made up) means and stdevs
# (BBOX_NORMALIZE_TARGETS must also be True)
__C.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED = True

__C.TRAIN.BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)

__C.TRAIN.BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)


#
# Testing options
#

__C.TEST = edict()
__C.TEST.GLOBAL_SEARCH = False
__C.TEST.SEGMENTATION = True
__C.TEST.SINGLE_FRAME = False
__C.TEST.VERTEX_REG_2D = False
__C.TEST.VERTEX_REG_3D = False
__C.TEST.VISUALIZE = False
__C.TEST.RANSAC = False
__C.TEST.GAN = False
__C.TEST.POSE_REG = False
__C.TEST.POSE_REFINE = False
__C.TEST.POSE_SDF = True
__C.TEST.POSE_CODEBOOK = False
__C.TEST.SYNTHESIZE = False
__C.TEST.ROS_CAMERA = 'camera'
__C.TEST.DET_THRESHOLD = 0.5
__C.TEST.BUILD_CODEBOOK = False
__C.TEST.IMS_PER_BATCH = 1
__C.TEST.MEAN_SHIFT = False
__C.TEST.CHECK_SIZE = False
__C.TEST.NUM_SDF_ITERATIONS_INIT = 100
__C.TEST.NUM_SDF_ITERATIONS_TRACKING = 50
__C.TEST.SDF_TRANSLATION_REG = 10.0
__C.TEST.SDF_ROTATION_REG = 0.1
__C.TEST.NUM_LOST = 3
__C.TEST.ALIGN_Z_AXIS = False
__C.TEST.GEN_DATA = False

# Hough voting
__C.TEST.HOUGH_LABEL_THRESHOLD = 100
__C.TEST.HOUGH_VOTING_THRESHOLD = -1
__C.TEST.HOUGH_SKIP_PIXELS = -1
__C.TEST.HOUGH_INLIER_THRESHOLD = 0.9
__C.TEST.CLASSES = (0,1,2,3)
__C.TEST.SYMMETRY = (0,0,0,0)


__C.TEST.ITERNUM = 4

# Scales to compute real features
__C.TEST.SCALES_BASE = (0.25, 0.5, 1.0, 2.0, 3.0)

# voxel grid size
__C.TEST.GRID_SIZE = 256

## NMS threshold used on RPN proposals
__C.TEST.RPN_NMS_THRESH = 0.7

# Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.TEST.RPN_PRE_NMS_TOP_N = 6000

# Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.TEST.RPN_POST_NMS_TOP_N = 300

# Test using bounding-box regressors
__C.TEST.BBOX_REG = True

# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
__C.TEST.NMS = 0.3

# Pixel mean values (BGR order) as a (1, 1, 3) array
# These are the values originally used for training VGG16
__C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])

# For reproducibility
__C.RNG_SEED = 3

# A small number that's used many times
__C.EPS = 1e-14

# Root directory of project
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))

# Place outputs under an experiments directory
__C.EXP_DIR = 'default'

# Default GPU device id
__C.GPU_ID = 0


def get_output_dir(imdb, net):
    """Return the directory where experimental artifacts are placed.

    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    path = osp.abspath(osp.join(__C.ROOT_DIR, 'output', __C.EXP_DIR, imdb.name))
    if net is None:
        return path
    else:
        return osp.join(path, net)

def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        if type(b[k]) is not type(v):
            raise ValueError(('Type mismatch ({} vs. {}) '
                              'for config key: {}').format(type(b[k]),
                                                           type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v

def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml

    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f, Loader=yaml.FullLoader))

    _merge_a_into_b(yaml_cfg, __C)


def yaml_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f, Loader=yaml.FullLoader))
    return yaml_cfg
