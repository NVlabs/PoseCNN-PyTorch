# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
from fcn.config import cfg
from utils.bbox_transform import bbox_transform_inv
from utils.cython_bbox import bbox_overlaps

# rpn_rois: (batch_ids, cls, x1, y1, x2, y2, scores)
# gt_boxes: batch * num_classes * (x1, y1, x2, y2, cls)
def pose_target_layer(rois, bbox_prob, bbox_pred, gt_boxes, poses, is_training):

    rois = rois.detach().cpu().numpy()
    bbox_prob = bbox_prob.detach().cpu().numpy()
    bbox_pred = bbox_pred.detach().cpu().numpy()
    gt_boxes = gt_boxes.detach().cpu().numpy()
    num_classes = bbox_prob.shape[1]
  
    # process boxes
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        stds = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS), (num_classes))
        means = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS), (num_classes))
        bbox_pred *= stds
        bbox_pred += means

    boxes = rois[:, 2:6].copy()
    pred_boxes = bbox_transform_inv(boxes, bbox_pred)

    # assign boxes
    for i in range(rois.shape[0]):
        cls = int(rois[i, 1])
        rois[i, 2:6] = pred_boxes[i, cls*4:cls*4+4]
        rois[i, 6] = bbox_prob[i, cls]

    # convert boxes to (batch_ids, x1, y1, x2, y2, cls)
    roi_blob = rois[:, (0, 2, 3, 4, 5, 1)]
    gt_box_blob = np.zeros((0, 6), dtype=np.float32)
    pose_blob = np.zeros((0, 9), dtype=np.float32)
    for i in range(gt_boxes.shape[0]):
        for j in range(gt_boxes.shape[1]):
            if gt_boxes[i, j, -1] > 0:
                gt_box = np.zeros((1, 6), dtype=np.float32)
                gt_box[0, 0] = i
                gt_box[0, 1:5] = gt_boxes[i, j, :4]
                gt_box[0, 5] = gt_boxes[i, j, 4]
                gt_box_blob = np.concatenate((gt_box_blob, gt_box), axis=0)
                poses[i, j, 0] = i
                pose_blob = np.concatenate((pose_blob, poses[i, j, :].cpu().reshape(1, 9)), axis=0)

    if gt_box_blob.shape[0] == 0:
        num = rois.shape[0]
        poses_target = np.zeros((num, 4 * num_classes), dtype=np.float32)
        poses_weight = np.zeros((num, 4 * num_classes), dtype=np.float32)
    else:
        # overlaps: (rois x gt_boxes)
        overlaps = bbox_overlaps(
            np.ascontiguousarray(roi_blob[:, :5], dtype=np.float),
            np.ascontiguousarray(gt_box_blob[:, :5], dtype=np.float))

        gt_assignment = overlaps.argmax(axis=1)
        max_overlaps = overlaps.max(axis=1)
        labels = gt_box_blob[gt_assignment, 5]
        quaternions = pose_blob[gt_assignment, 2:6]

        # Select foreground RoIs as those with >= FG_THRESH overlap
        bg_inds = np.where(max_overlaps < cfg.TRAIN.FG_THRESH_POSE)[0]
        labels[bg_inds] = 0

        bg_inds = np.where(roi_blob[:, -1] != labels)[0]
        labels[bg_inds] = 0

        # in training, only use the positive boxes for pose regression
        if is_training:
            fg_inds = np.where(labels > 0)[0]
            if len(fg_inds) > 0:
                rois = rois[fg_inds, :]
                quaternions = quaternions[fg_inds, :]
                labels = labels[fg_inds]
    
        # pose regression targets and weights
        poses_target, poses_weight = _compute_pose_targets(quaternions, labels, num_classes)

    return torch.from_numpy(rois).cuda(), torch.from_numpy(poses_target).cuda(), torch.from_numpy(poses_weight).cuda()


def _compute_pose_targets(quaternions, labels, num_classes):
    """Compute pose regression targets for an image."""

    num = quaternions.shape[0]
    poses_target = np.zeros((num, 4 * num_classes), dtype=np.float32)
    poses_weight = np.zeros((num, 4 * num_classes), dtype=np.float32)

    for i in range(num):
        cls = labels[i]
        if cls > 0 and np.linalg.norm(quaternions[i, :]) > 0:
            start = int(4 * cls)
            end = start + 4
            poses_target[i, start:end] = quaternions[i, :]
            poses_weight[i, start:end] = 1.0

    return poses_target, poses_weight
