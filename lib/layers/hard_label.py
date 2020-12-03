# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import math
from torch import nn
from torch.autograd import Function
import torch
import posecnn_cuda


class HardLabelFunction(Function):
    @staticmethod
    def forward(ctx, prob, label, rand, threshold, sample_percentage):
        outputs = posecnn_cuda.hard_label_forward(threshold, sample_percentage, prob, label, rand)
        top_data = outputs[0]
        return top_data

    @staticmethod
    def backward(ctx, top_diff):
        outputs = posecnn_cuda.hard_label_backward(top_diff)
        d_prob, d_label = outputs
        return d_prob, d_label, None, None, None


class HardLabel(nn.Module):
    def __init__(self, threshold, sample_percentage):
        super(HardLabel, self).__init__()
        self.threshold = threshold
        self.sample_percentage = sample_percentage

    def forward(self, prob, label, rand):
        return HardLabelFunction.apply(prob, label, rand, self.threshold, self.sample_percentage)
