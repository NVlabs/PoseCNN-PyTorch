# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

from __future__ import absolute_import
from __future__ import division

import torch
import numpy as np
import numpy.random as npr
from fcn.config import cfg
from utils.bbox_transform import bbox_transform
from utils.cython_bbox import bbox_overlaps

# rpn_rois: (batch_ids, cls, x1, y1, x2, y2, scores)
# gt_boxes: batch * num_classes * (x1, y1, x2, y2, cls)
def roi_target_layer(rpn_rois, gt_boxes):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    rpn_rois = rpn_rois.detach().cpu().numpy()
    gt_boxes = gt_boxes.detach().cpu().numpy()
    num_classes = gt_boxes.shape[1]

    # convert boxes to (batch_ids, x1, y1, x2, y2, cls)
    roi_blob = rpn_rois[:, (0, 2, 3, 4, 5, 1)]
    gt_box_blob = np.zeros((0, 6), dtype=np.float32)
    for i in range(gt_boxes.shape[0]):
        for j in range(gt_boxes.shape[1]):
            if gt_boxes[i, j, -1] > 0:
                gt_box = np.zeros((1, 6), dtype=np.float32)
                gt_box[0, 0] = i
                gt_box[0, 1:5] = gt_boxes[i, j, :4]
                gt_box[0, 5] = gt_boxes[i, j, 4]
                gt_box_blob = np.concatenate((gt_box_blob, gt_box), axis=0)

    # sample rois with classification labels and bounding box regression targets
    labels, bbox_targets, bbox_inside_weights = _sample_rois(roi_blob, gt_box_blob, num_classes)
    bbox_outside_weights = np.array(bbox_inside_weights > 0).astype(np.float32)

    # convert labels
    num = labels.shape[0]
    label_blob = np.zeros((num, num_classes), dtype=np.float32)
    if np.any(roi_blob[:, -1] > 0):
        for i in range(num):
            label_blob[i, int(labels[i])] = 1.0

    return torch.from_numpy(label_blob).cuda(), torch.from_numpy(bbox_targets).cuda(), \
        torch.from_numpy(bbox_inside_weights).cuda(), torch.from_numpy(bbox_outside_weights).cuda()


def _get_bbox_regression_labels(bbox_target_data, num_classes):
  """Bounding-box regression targets (bbox_target_data) are stored in a
  compact form N x (class, tx, ty, tw, th)

  This function expands those targets into the 4-of-4*K representation used
  by the network (i.e. only one class has non-zero targets).

  Returns:
      bbox_target (ndarray): N x 4K blob of regression targets
      bbox_inside_weights (ndarray): N x 4K blob of loss weights
  """

  clss = bbox_target_data[:, 0]
  bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
  bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
  inds = np.where(clss > 0)[0]
  for ind in inds:
    cls = clss[ind]
    start = int(4 * cls)
    end = start + 4
    bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
    bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
  return bbox_targets, bbox_inside_weights


def _compute_targets(ex_rois, gt_rois, labels):
  """Compute bounding-box regression targets for an image."""

  assert ex_rois.shape[0] == gt_rois.shape[0]
  assert ex_rois.shape[1] == 4
  assert gt_rois.shape[1] == 4

  targets = bbox_transform(ex_rois, gt_rois)
  if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
    # Optionally normalize targets by a precomputed mean and stdev
    targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
               / np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))
  return np.hstack(
    (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)


def _sample_rois(all_rois, gt_boxes, num_classes):
  """Generate a random sample of RoIs comprising foreground and background
  examples.
  """
  # all_rois (batch_ids, x1, y1, x2, y2, cls)
  # gt_boxes (batch_ids, x1, y1, x2, y2, cls)
  # overlaps: (rois x gt_boxes)

  if gt_boxes.shape[0] == 0:
      num = all_rois.shape[0]
      labels = np.zeros((num, 1), dtype=np.float32)
      bbox_targets = np.zeros((num, 4 * num_classes), dtype=np.float32)
      bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
  else:
      overlaps = bbox_overlaps(
        np.ascontiguousarray(all_rois[:, :5], dtype=np.float),
        np.ascontiguousarray(gt_boxes[:, :5], dtype=np.float))

      gt_assignment = overlaps.argmax(axis=1)
      max_overlaps = overlaps.max(axis=1)
      labels = gt_boxes[gt_assignment, 5]

      # Select foreground RoIs as those with >= FG_THRESH overlap
      # fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
      bg_inds = np.where(max_overlaps < cfg.TRAIN.FG_THRESH)[0]
      labels[bg_inds] = 0

      # print '{:d} rois, {:d} fg, {:d} bg'.format(all_rois.shape[0], all_rois.shape[0]-len(bg_inds), len(bg_inds))
      # print all_rois

      bbox_target_data = _compute_targets(
        all_rois[:, 1:5], gt_boxes[gt_assignment, 1:5], labels)

      bbox_targets, bbox_inside_weights = \
        _get_bbox_regression_labels(bbox_target_data, num_classes)

  return labels, bbox_targets, bbox_inside_weights
