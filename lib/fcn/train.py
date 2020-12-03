# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import torch
import torch.nn as nn
import time
import sys, os
import numpy as np
import matplotlib.pyplot as plt

from fcn.config import cfg
from transforms3d.quaternions import mat2quat, quat2mat, qmult
from utils.se3 import *
from utils.nms import *
from utils.pose_error import re, te

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return '{:.3f} ({:.3f})'.format(self.val, self.avg)


def loss_cross_entropy(scores, labels):
    """
    scores: a tensor [batch_size, num_classes, height, width]
    labels: a tensor [batch_size, num_classes, height, width]
    """

    cross_entropy = -torch.sum(labels * scores, dim=1)
    loss = torch.div(torch.sum(cross_entropy), torch.sum(labels)+1e-10)

    return loss


def smooth_l1_loss(vertex_pred, vertex_targets, vertex_weights, sigma=1.0):
    sigma_2 = sigma ** 2
    vertex_diff = vertex_pred - vertex_targets
    diff = torch.mul(vertex_weights, vertex_diff)
    abs_diff = torch.abs(diff)
    smoothL1_sign = torch.lt(abs_diff, 1. / sigma_2).float().detach()
    in_loss = torch.pow(diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
            + (abs_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    loss = torch.div( torch.sum(in_loss), torch.sum(vertex_weights) + 1e-10 )
    return loss

#************************************#
#    train PoseCNN                   #
#************************************#

'''
sample = {'image_color': im_blob,
          'im_depth': im_depth,
          'label': label_blob,
          'mask': mask,
          'meta_data': meta_data_blob,
          'poses': pose_blob,
          'extents': self._extents,
          'points': self._point_blob,
          'symmetry': self._symmetry,
          'gt_boxes': gt_boxes,
          'im_info': im_info,
          'vertex_targets': vertex_targets,
          'vertex_weights': vertex_weights}
'''

def train(train_loader, background_loader, network, optimizer, epoch):

    batch_time = AverageMeter()
    losses = AverageMeter()

    epoch_size = len(train_loader)
    enum_background = enumerate(background_loader)
    pixel_mean = torch.from_numpy(cfg.PIXEL_MEANS.transpose(2, 0, 1) / 255.0).float()

    # switch to train mode
    network.train()

    for i, sample in enumerate(train_loader):

        end = time.time()

        # prepare data
        inputs = sample['image_color'].cuda()
        im_info = sample['im_info']
        mask = sample['mask'].cuda()
        labels = sample['label'].cuda()
        meta_data = sample['meta_data'].cuda()
        extents = sample['extents'][0, :, :].repeat(cfg.TRAIN.GPUNUM, 1, 1).cuda()
        gt_boxes = sample['gt_boxes'].cuda()
        poses = sample['poses'].cuda()
        points = sample['points'][0, :, :, :].repeat(cfg.TRAIN.GPUNUM, 1, 1, 1).cuda()
        symmetry = sample['symmetry'][0, :].repeat(cfg.TRAIN.GPUNUM, 1).cuda()

        if cfg.TRAIN.VERTEX_REG:
            vertex_targets = sample['vertex_targets'].cuda()
            vertex_weights = sample['vertex_weights'].cuda()
        else:
            vertex_targets = []
            vertex_weights = []

        # add background
        try:
            _, background = next(enum_background)
        except:
            enum_background = enumerate(background_loader)
            _, background = next(enum_background)

        if inputs.size(0) != background['background_color'].size(0):
            enum_background = enumerate(background_loader)
            _, background = next(enum_background)

        background_color = background['background_color'].cuda()
        for j in range(inputs.size(0)):
            is_syn = im_info[j, -1]
            if is_syn or np.random.rand(1) > 0.5:
                inputs[j] = mask[j] * inputs[j] + (1 - mask[j]) * background_color[j]

        # visualization
        if cfg.TRAIN.VISUALIZE:
            _vis_minibatch(inputs, background, labels, vertex_targets, sample, train_loader.dataset.class_colors)

        # compute output
        if cfg.TRAIN.VERTEX_REG:
            if cfg.TRAIN.POSE_REG:
                out_logsoftmax, out_weight, out_vertex, out_logsoftmax_box, \
                    bbox_labels, bbox_pred, bbox_targets, bbox_inside_weights, loss_pose_tensor, poses_weight \
                    = network(inputs, labels, meta_data, extents, gt_boxes, poses, points, symmetry)

                loss_label = loss_cross_entropy(out_logsoftmax, out_weight)
                loss_vertex = cfg.TRAIN.VERTEX_W * smooth_l1_loss(out_vertex, vertex_targets, vertex_weights)
                loss_box = loss_cross_entropy(out_logsoftmax_box, bbox_labels)
                loss_location = smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights)
                loss_pose = torch.mean(loss_pose_tensor)
                loss = loss_label + loss_vertex + loss_box + loss_location + loss_pose
            else:
                out_logsoftmax, out_weight, out_vertex, out_logsoftmax_box, \
                    bbox_labels, bbox_pred, bbox_targets, bbox_inside_weights \
                    = network(inputs, labels, meta_data, extents, gt_boxes, poses, points, symmetry)

                loss_label = loss_cross_entropy(out_logsoftmax, out_weight)
                loss_vertex = cfg.TRAIN.VERTEX_W * smooth_l1_loss(out_vertex, vertex_targets, vertex_weights)
                loss_box = loss_cross_entropy(out_logsoftmax_box, bbox_labels)
                loss_location = smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights)
                loss = loss_label + loss_vertex + loss_box + loss_location
        else:
            out_logsoftmax, out_weight = network(inputs, labels, meta_data, extents, gt_boxes, poses, points, symmetry)
            loss = loss_cross_entropy(out_logsoftmax, out_weight)

        # record loss
        losses.update(loss.data, inputs.size(0))

        # compute gradient and do optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)

        if cfg.TRAIN.VERTEX_REG:
            if cfg.TRAIN.POSE_REG:
                num_bg = torch.sum(bbox_labels[:, 0])
                num_fg = torch.sum(torch.sum(bbox_labels[:, 1:], dim=1))
                num_fg_pose = torch.sum(torch.sum(poses_weight[:, 4:], dim=1)) / 4
                print('[%d/%d][%d/%d], %.4f, label %.4f, center %.4f, box %.4f (%03d, %03d), loc %.4f, pose %.4f (%03d), lr %.6f, time %.2f' \
                   % (epoch, cfg.epochs, i, epoch_size, loss.data, loss_label.data, loss_vertex.data, loss_box.data, num_fg.data, num_bg.data, \
                      loss_location.data, loss_pose.data, num_fg_pose, optimizer.param_groups[0]['lr'], batch_time.val))
            else:
                num_bg = torch.sum(bbox_labels[:, 0])
                num_fg = torch.sum(torch.sum(bbox_labels[:, 1:], dim=1))
                print('[%d/%d][%d/%d], %.4f, label %.4f, center %.4f, box %.4f (%03d, %03d), loc %.4f, lr %.6f, time %.2f' \
                   % (epoch, cfg.epochs, i, epoch_size, loss.data, loss_label.data, loss_vertex.data, loss_box.data, num_fg.data, num_bg.data, \
                      loss_location.data, optimizer.param_groups[0]['lr'], batch_time.val))
        else:
            print('[%d/%d][%d/%d], loss %.4f, lr %.6f, time %.2f' \
               % (epoch, cfg.epochs, i, epoch_size, loss, optimizer.param_groups[0]['lr'], batch_time.val))
        cfg.TRAIN.ITERS += 1


def _get_bb3D(extent):
    bb = np.zeros((3, 8), dtype=np.float32)
    
    xHalf = extent[0] * 0.5
    yHalf = extent[1] * 0.5
    zHalf = extent[2] * 0.5
    
    bb[:, 0] = [xHalf, yHalf, zHalf]
    bb[:, 1] = [-xHalf, yHalf, zHalf]
    bb[:, 2] = [xHalf, -yHalf, zHalf]
    bb[:, 3] = [-xHalf, -yHalf, zHalf]
    bb[:, 4] = [xHalf, yHalf, -zHalf]
    bb[:, 5] = [-xHalf, yHalf, -zHalf]
    bb[:, 6] = [xHalf, -yHalf, -zHalf]
    bb[:, 7] = [-xHalf, -yHalf, -zHalf]
    
    return bb


def _vis_minibatch(inputs, background, labels, vertex_targets, sample, class_colors):

    """Visualize a mini-batch for debugging."""
    import matplotlib.pyplot as plt

    im_blob = inputs.cpu().numpy()
    label_blob = labels.cpu().numpy()
    gt_poses = sample['poses'].numpy()
    meta_data_blob = sample['meta_data'].numpy()
    gt_boxes = sample['gt_boxes'].numpy()
    extents = sample['extents'][0, :, :].numpy()
    background_color = background['background_color'].numpy()

    if cfg.TRAIN.VERTEX_REG:
        vertex_target_blob = vertex_targets.cpu().numpy()

    if cfg.INPUT == 'COLOR':
        m = 3
        n = 3
    else:
        m = 3
        n = 4
    
    for i in range(im_blob.shape[0]):
        fig = plt.figure()
        start = 1

        metadata = meta_data_blob[i, :]
        intrinsic_matrix = metadata[:9].reshape((3,3))

        # show image
        if cfg.INPUT == 'COLOR' or cfg.INPUT == 'RGBD':
            if cfg.INPUT == 'COLOR':
                im = im_blob[i, :, :, :].copy()
            else:
                im = im_blob[i, :3, :, :].copy()
            im = im.transpose((1, 2, 0)) * 255.0
            im += cfg.PIXEL_MEANS
            im = im[:, :, (2, 1, 0)]
            im = np.clip(im, 0, 255)
            im = im.astype(np.uint8)
            ax = fig.add_subplot(m, n, 1)
            plt.imshow(im)
            ax.set_title('color')
            start += 1

        if cfg.INPUT == 'DEPTH' or cfg.INPUT == 'RGBD':
            if cfg.INPUT == 'DEPTH':
                im_depth = im_blob[i, :, :, :].copy()
            else:
                im_depth = im_blob[i, 3:6, :, :].copy()

            ax = fig.add_subplot(m, n, start)
            plt.imshow(im_depth[0, :, :])
            ax.set_title('depth x')
            start += 1

            ax = fig.add_subplot(m, n, start)
            plt.imshow(im_depth[1, :, :])
            ax.set_title('depth y')
            start += 1

            ax = fig.add_subplot(m, n, start)
            plt.imshow(im_depth[2, :, :])
            ax.set_title('depth z')
            start += 1

            if cfg.INPUT == 'RGBD':
                ax = fig.add_subplot(m, n, start)
                mask = im_blob[i, 6, :, :].copy()
                plt.imshow(mask)
                ax.set_title('depth mask')
                start += 1

        # project the 3D box to image
        pose_blob = gt_poses[i]
        for j in range(pose_blob.shape[0]):
            if pose_blob[j, 0] == 0:
                continue

            class_id = int(pose_blob[j, 1])
            bb3d = _get_bb3D(extents[class_id, :])
            x3d = np.ones((4, 8), dtype=np.float32)
            x3d[0:3, :] = bb3d
            
            # projection
            RT = np.zeros((3, 4), dtype=np.float32)

            # allocentric to egocentric
            T = pose_blob[j, 6:]
            qt = allocentric2egocentric(pose_blob[j, 2:6], T)
            RT[:3, :3] = quat2mat(qt)

            # RT[:3, :3] = quat2mat(pose_blob[j, 2:6])
            RT[:, 3] = pose_blob[j, 6:]
            x2d = np.matmul(intrinsic_matrix, np.matmul(RT, x3d))
            x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
            x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])

            x1 = np.min(x2d[0, :])
            x2 = np.max(x2d[0, :])
            y1 = np.min(x2d[1, :])
            y2 = np.max(x2d[1, :])
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='g', linewidth=3, clip_on=False))

        if cfg.INPUT == 'COLOR' or cfg.INPUT == 'RGBD':
            im_background = background_color[i]
            im_background = im_background.transpose((1, 2, 0)) * 255.0
            im_background += cfg.PIXEL_MEANS
            im_background = im_background[:, :, (2, 1, 0)]
            im_background = np.clip(im_background, 0, 255)
            im_background = im_background.astype(np.uint8)
            ax = fig.add_subplot(m, n, start)
            plt.imshow(im_background)
            ax.set_title('background')
            start += 1

        # show gt boxes
        ax = fig.add_subplot(m, n, start)
        start += 1
        if cfg.INPUT == 'COLOR' or cfg.INPUT == 'RGBD':
            plt.imshow(im)
        else:
            plt.imshow(im_depth[2, :, :])
        ax.set_title('gt boxes')
        boxes = gt_boxes[i]
        for j in range(boxes.shape[0]):
            if boxes[j, 4] == 0:
                continue
            x1 = boxes[j, 0]
            y1 = boxes[j, 1]
            x2 = boxes[j, 2]
            y2 = boxes[j, 3]
            plt.gca().add_patch(
                plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='g', linewidth=3, clip_on=False))

        # show label
        label = label_blob[i, :, :, :]
        label = label.transpose((1, 2, 0))

        height = label.shape[0]
        width = label.shape[1]
        num_classes = label.shape[2]
        im_label = np.zeros((height, width, 3), dtype=np.uint8)
        for j in range(num_classes):
            I = np.where(label[:, :, j] > 0)
            im_label[I[0], I[1], :] = class_colors[j]

        ax = fig.add_subplot(m, n, start)
        start += 1
        plt.imshow(im_label)
        ax.set_title('label')

        # show vertex targets
        if cfg.TRAIN.VERTEX_REG:
            vertex_target = vertex_target_blob[i, :, :, :]
            center = np.zeros((3, height, width), dtype=np.float32)

            for j in range(1, num_classes):
                index = np.where(label[:, :, j] > 0)
                if len(index[0]) > 0:
                    center[0, index[0], index[1]] = vertex_target[3*j, index[0], index[1]]
                    center[1, index[0], index[1]] = vertex_target[3*j+1, index[0], index[1]]
                    center[2, index[0], index[1]] = np.exp(vertex_target[3*j+2, index[0], index[1]])

            ax = fig.add_subplot(m, n, start)
            start += 1
            plt.imshow(center[0,:,:])
            ax.set_title('center x') 

            ax = fig.add_subplot(m, n, start)
            start += 1
            plt.imshow(center[1,:,:])
            ax.set_title('center y')

            ax = fig.add_subplot(m, n, start)
            start += 1
            plt.imshow(center[2,:,:])
            ax.set_title('z')

        plt.show()
