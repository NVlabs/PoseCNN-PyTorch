# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import math
from torch import nn
from torch.autograd import Function
import torch
import posecnn_cuda


class HoughVotingFunction(Function):
    @staticmethod
    def forward(ctx, label, vertex, meta_data, extents, is_train, skip_pixels, \
            label_threshold, inlier_threshold, voting_threshold, per_threshold):

        outputs = posecnn_cuda.hough_voting_forward(label, vertex, meta_data, extents, is_train, skip_pixels, \
            label_threshold, inlier_threshold, voting_threshold, per_threshold)

        top_box = outputs[0]
        top_pose = outputs[1]
        return top_box, top_pose

    @staticmethod
    def backward(ctx, top_diff_box, top_diff_pose):
        return None, None, None, None, None, None, None, None, None, None


class HoughVoting(nn.Module):
    def __init__(self, is_train=0, skip_pixels=10, label_threshold=100, inlier_threshold=0.9, voting_threshold=-1, per_threshold=0.01):
        super(HoughVoting, self).__init__()
        self.is_train = is_train
        self.skip_pixels = skip_pixels
        self.label_threshold = label_threshold
        self.inlier_threshold = inlier_threshold
        self.voting_threshold = voting_threshold
        self.per_threshold = per_threshold

    def forward(self, label, vertex, meta_data, extents):
        return HoughVotingFunction.apply(label, vertex, meta_data, extents, self.is_train, self.skip_pixels, \
            self.label_threshold, self.inlier_threshold, self.voting_threshold, self.per_threshold)
