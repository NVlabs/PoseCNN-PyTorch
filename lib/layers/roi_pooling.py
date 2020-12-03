# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import math
from torch import nn
from torch.autograd import Function
import torch
import posecnn_cuda


class RoIPoolFunction(Function):
    @staticmethod
    def forward(ctx, features, rois, pool_height, pool_width, spatial_scale):
        outputs = posecnn_cuda.roi_pool_forward(pool_height, pool_width, spatial_scale, features, rois)
        top_data = outputs[0]
        variables = outputs[1:]
        variables.append(rois)
        ctx.feature_size = features.size()
        ctx.spatial_scale = spatial_scale
        ctx.save_for_backward(*variables)
        return top_data

    @staticmethod
    def backward(ctx, top_diff):
        argmax_data = ctx.saved_variables[0]
        rois = ctx.saved_variables[1]
        batch_size, num_channels, data_height, data_width = ctx.feature_size
        spatial_scale = ctx.spatial_scale
        outputs = posecnn_cuda.roi_pool_backward(batch_size, data_height, data_width, spatial_scale, top_diff, rois, argmax_data)
        d_features = outputs[0]
        return d_features, None, None, None, None   

class RoIPool(nn.Module):
    def __init__(self, pool_height, pool_width, spatial_scale):
        super(RoIPool, self).__init__()

        self.pool_width = int(pool_width)
        self.pool_height = int(pool_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        return RoIPoolFunction.apply(features, rois, self.pool_height, self.pool_width, self.spatial_scale)
