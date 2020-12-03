# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import math
from torch import nn
from torch.autograd import Function
import torch
import posecnn_cuda

class PMLossFunction(Function):
    @staticmethod
    def forward(ctx, prediction, target, weight, points, symmetry, hard_angle):
        outputs = posecnn_cuda.pml_forward(prediction, target, weight, points, symmetry, hard_angle)
        loss = outputs[0]
        variables = outputs[1:]
        ctx.save_for_backward(*variables)

        return loss

    @staticmethod
    def backward(ctx, grad_loss):
        outputs = posecnn_cuda.pml_backward(grad_loss, *ctx.saved_variables)
        d_rotation = outputs[0]

        return d_rotation, None, None, None, None, None


class PMLoss(nn.Module):
    def __init__(self, hard_angle=15):
        super(PMLoss, self).__init__()
        self.hard_angle = hard_angle

    def forward(self, prediction, target, weight, points, symmetry):
        return PMLossFunction.apply(prediction, target, weight, points, symmetry, self.hard_angle)
