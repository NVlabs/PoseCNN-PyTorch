# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import torch
import torch.nn as nn
import torchvision.models as models
import math
import sys
import copy
from torch.nn.init import kaiming_normal_
from layers.hard_label import HardLabel
from layers.hough_voting import HoughVoting
from layers.roi_pooling import RoIPool
from layers.point_matching_loss import PMLoss
from layers.roi_target_layer import roi_target_layer
from layers.pose_target_layer import pose_target_layer
from fcn.config import cfg

__all__ = [
    'posecnn',
]

vgg16 = models.vgg16(pretrained=False)

def log_softmax_high_dimension(input):
    num_classes = input.size()[1]
    m = torch.max(input, dim=1, keepdim=True)[0]
    if input.dim() == 4:
        d = input - m.repeat(1, num_classes, 1, 1)
    else:
        d = input - m.repeat(1, num_classes)
    e = torch.exp(d)
    s = torch.sum(e, dim=1, keepdim=True)
    if input.dim() == 4:
        output = d - torch.log(s.repeat(1, num_classes, 1, 1))
    else:
        output = d - torch.log(s.repeat(1, num_classes))
    return output


def softmax_high_dimension(input):
    num_classes = input.size()[1]
    m = torch.max(input, dim=1, keepdim=True)[0]
    if input.dim() == 4:
        e = torch.exp(input - m.repeat(1, num_classes, 1, 1))
    else:
        e = torch.exp(input - m.repeat(1, num_classes))
    s = torch.sum(e, dim=1, keepdim=True)
    if input.dim() == 4:
        output = torch.div(e, s.repeat(1, num_classes, 1, 1))
    else:
        output = torch.div(e, s.repeat(1, num_classes))
    return output

def conv(in_planes, out_planes, kernel_size=3, stride=1, relu=True):
    if relu:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.ReLU(inplace=True))
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True)


def fc(in_planes, out_planes, relu=True):
    if relu:
        return nn.Sequential(
            nn.Linear(in_planes, out_planes),
            nn.LeakyReLU(0.1, inplace=True))
    else:
        return nn.Linear(in_planes, out_planes)


def upsample(scale_factor):
    return nn.Upsample(scale_factor=scale_factor, mode='bilinear')


class PoseCNN(nn.Module):

    def __init__(self, num_classes, num_units):
        super(PoseCNN, self).__init__()
        self.num_classes = num_classes

        # conv features
        features = list(vgg16.features)[:30]
        
        # change the first conv layer for RGBD
        if cfg.INPUT == 'RGBD':
            conv0 = conv(6, 64, kernel_size=3, relu=False)
            conv0.weight.data[:, :3, :, :] = features[0].weight.data
            conv0.weight.data[:, 3:, :, :] = features[0].weight.data
            conv0.bias.data = features[0].bias.data
            features[0] = conv0

        self.features = nn.ModuleList(features)
        self.classifier = vgg16.classifier[:-1]
        if cfg.TRAIN.SLIM:
            dim_fc = 256
            self.classifier[0] = nn.Linear(512*7*7, 256)
            self.classifier[3] = nn.Linear(256, 256)
        else:
            dim_fc = 4096
            
        print(self.features)
        print(self.classifier)

        # freeze some layers
        if cfg.TRAIN.FREEZE_LAYERS:
            for i in [0, 2, 5, 7, 10, 12, 14]:
                self.features[i].weight.requires_grad = False
                self.features[i].bias.requires_grad = False

        # semantic labeling branch
        self.conv4_embed = conv(512, num_units, kernel_size=1)
        self.conv5_embed = conv(512, num_units, kernel_size=1)
        self.upsample_conv5_embed = upsample(2.0)
        self.upsample_embed = upsample(8.0)
        self.conv_score = conv(num_units, num_classes, kernel_size=1)
        self.hard_label = HardLabel(threshold=cfg.TRAIN.HARD_LABEL_THRESHOLD, sample_percentage=cfg.TRAIN.HARD_LABEL_SAMPLING)
        self.dropout = nn.Dropout()

        if cfg.TRAIN.VERTEX_REG:
            # center regression branch
            self.conv4_vertex_embed = conv(512, 2*num_units, kernel_size=1, relu=False)
            self.conv5_vertex_embed = conv(512, 2*num_units, kernel_size=1, relu=False)
            self.upsample_conv5_vertex_embed = upsample(2.0)
            self.upsample_vertex_embed = upsample(8.0)
            self.conv_vertex_score = conv(2*num_units, 3*num_classes, kernel_size=1, relu=False)
            # hough voting
            self.hough_voting = HoughVoting(is_train=0, skip_pixels=10, label_threshold=100, \
                                            inlier_threshold=0.9, voting_threshold=-1, per_threshold=0.01)

            self.roi_pool_conv4 = RoIPool(pool_height=7, pool_width=7, spatial_scale=1.0 / 8.0)
            self.roi_pool_conv5 = RoIPool(pool_height=7, pool_width=7, spatial_scale=1.0 / 16.0)
            self.fc8 = fc(dim_fc, num_classes)
            self.fc9 = fc(dim_fc, 4 * num_classes, relu=False)

            if cfg.TRAIN.POSE_REG:
                self.fc10 = fc(dim_fc, 4 * num_classes, relu=False)
                self.pml = PMLoss(hard_angle=cfg.TRAIN.HARD_ANGLE)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x, label_gt, meta_data, extents, gt_boxes, poses, points, symmetry):

        # conv features
        for i, model in enumerate(self.features):
            x = model(x)
            if i == 22:
                out_conv4_3 = x
            if i == 29:
                out_conv5_3 = x

        # semantic labeling branch
        out_conv4_embed = self.conv4_embed(out_conv4_3)
        out_conv5_embed = self.conv5_embed(out_conv5_3)
        out_conv5_embed_up = self.upsample_conv5_embed(out_conv5_embed)
        out_embed = self.dropout(out_conv4_embed + out_conv5_embed_up)
        out_embed_up = self.upsample_embed(out_embed)
        out_score = self.conv_score(out_embed_up)
        out_logsoftmax = log_softmax_high_dimension(out_score)
        out_prob = softmax_high_dimension(out_score)
        out_label = torch.max(out_prob, dim=1)[1].type(torch.IntTensor).cuda()
        out_weight = self.hard_label(out_prob, label_gt, torch.rand(out_prob.size()).cuda())

        if cfg.TRAIN.VERTEX_REG:
            # center regression branch
            out_conv4_vertex_embed = self.conv4_vertex_embed(out_conv4_3)
            out_conv5_vertex_embed = self.conv5_vertex_embed(out_conv5_3)
            out_conv5_vertex_embed_up = self.upsample_conv5_vertex_embed(out_conv5_vertex_embed)
            out_vertex_embed = self.dropout(out_conv4_vertex_embed + out_conv5_vertex_embed_up)
            out_vertex_embed_up = self.upsample_vertex_embed(out_vertex_embed)
            out_vertex = self.conv_vertex_score(out_vertex_embed_up)

            # hough voting
            if self.training:
                self.hough_voting.is_train = 1
                self.hough_voting.label_threshold = cfg.TRAIN.HOUGH_LABEL_THRESHOLD
                self.hough_voting.voting_threshold = cfg.TRAIN.HOUGH_VOTING_THRESHOLD
                self.hough_voting.skip_pixels = cfg.TRAIN.HOUGH_SKIP_PIXELS
                self.hough_voting.inlier_threshold = cfg.TRAIN.HOUGH_INLIER_THRESHOLD
            else:
                self.hough_voting.is_train = 0
                self.hough_voting.label_threshold = cfg.TEST.HOUGH_LABEL_THRESHOLD
                self.hough_voting.voting_threshold = cfg.TEST.HOUGH_VOTING_THRESHOLD
                self.hough_voting.skip_pixels = cfg.TEST.HOUGH_SKIP_PIXELS
                self.hough_voting.inlier_threshold = cfg.TEST.HOUGH_INLIER_THRESHOLD
            out_box, out_pose = self.hough_voting(out_label, out_vertex, meta_data, extents)

            # bounding box classification and regression branch
            bbox_labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = roi_target_layer(out_box, gt_boxes)
            out_roi_conv4 = self.roi_pool_conv4(out_conv4_3, out_box)
            out_roi_conv5 = self.roi_pool_conv5(out_conv5_3, out_box)
            out_roi = out_roi_conv4 + out_roi_conv5
            out_roi_flatten = out_roi.view(out_roi.size(0), -1)
            out_fc7 = self.classifier(out_roi_flatten)
            out_fc8 = self.fc8(out_fc7)
            out_logsoftmax_box = log_softmax_high_dimension(out_fc8)
            bbox_prob = softmax_high_dimension(out_fc8)
            bbox_label_weights = self.hard_label(bbox_prob, bbox_labels, torch.rand(bbox_prob.size()).cuda())
            bbox_pred = self.fc9(out_fc7)

            # rotation regression branch
            rois, poses_target, poses_weight = pose_target_layer(out_box, bbox_prob, bbox_pred, gt_boxes, poses, self.training)
            if cfg.TRAIN.POSE_REG:    
                out_qt_conv4 = self.roi_pool_conv4(out_conv4_3, rois)
                out_qt_conv5 = self.roi_pool_conv5(out_conv5_3, rois)
                out_qt = out_qt_conv4 + out_qt_conv5
                out_qt_flatten = out_qt.view(out_qt.size(0), -1)
                out_qt_fc7 = self.classifier(out_qt_flatten)
                out_quaternion = self.fc10(out_qt_fc7)
                # point matching loss
                poses_pred = nn.functional.normalize(torch.mul(out_quaternion, poses_weight))
                if self.training:
                    loss_pose = self.pml(poses_pred, poses_target, poses_weight, points, symmetry)

        if self.training:
            if cfg.TRAIN.VERTEX_REG:
                if cfg.TRAIN.POSE_REG:
                    return out_logsoftmax, out_weight, out_vertex, out_logsoftmax_box, bbox_label_weights, \
                           bbox_pred, bbox_targets, bbox_inside_weights, loss_pose, poses_weight
                else:
                    return out_logsoftmax, out_weight, out_vertex, out_logsoftmax_box, bbox_label_weights, \
                           bbox_pred, bbox_targets, bbox_inside_weights
            else:
                return out_logsoftmax, out_weight
        else:
            if cfg.TRAIN.VERTEX_REG:
                if cfg.TRAIN.POSE_REG:
                    return out_label, out_vertex, rois, out_pose, out_quaternion
                else:
                    return out_label, out_vertex, rois, out_pose
            else:
                return out_label

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


def posecnn(num_classes, num_units, data=None):

    model = PoseCNN(num_classes, num_units)

    if data is not None:
        model_dict = model.state_dict()
        print('model keys')
        print('=================================================')
        for k, v in model_dict.items():
            print(k)
        print('=================================================')

        print('data keys')
        print('=================================================')
        for k, v in data.items():
            print(k)
        print('=================================================')

        pretrained_dict = {k: v for k, v in data.items() if k in model_dict and v.size() == model_dict[k].size()}
        print('load the following keys from the pretrained model')
        print('=================================================')
        for k, v in pretrained_dict.items():
            print(k)
        print('=================================================')
        model_dict.update(pretrained_dict) 
        model.load_state_dict(model_dict)

    return model
