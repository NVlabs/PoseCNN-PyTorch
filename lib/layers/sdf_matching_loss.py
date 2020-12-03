# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import math
from torch import nn
from torch.autograd import Function
import torch
import posecnn_cuda

class SDFLossFunction(Function):
    @staticmethod
    def forward(ctx, pose_delta, pose_init, sdf_grids, sdf_limits, points, regularization):
        outputs = posecnn_cuda.sdf_loss_forward(pose_delta, pose_init, sdf_grids, sdf_limits, points, regularization)
        loss = outputs[0]
        sdf_values = outputs[1]
        se3 = outputs[2]
        dalpha = outputs[3]
        J = outputs[4]
        variables = outputs[4:]
        ctx.save_for_backward(*variables)

        return loss, sdf_values, se3, dalpha, J

    @staticmethod
    def backward(ctx, grad_loss, grad_sdf_values, grad_se3, grad_JTJ, grad_J):
        outputs = posecnn_cuda.sdf_loss_backward(grad_loss, *ctx.saved_variables)
        d_delta = outputs[0]

        return d_delta, None, None, None, None


class SDFLoss(nn.Module):
    def __init__(self):
        super(SDFLoss, self).__init__()

    def forward(self, pose_delta, pose_init, sdf_grids, sdf_limits, points, regularization):
        return SDFLossFunction.apply(pose_delta, pose_init, sdf_grids, sdf_limits, points, regularization)
