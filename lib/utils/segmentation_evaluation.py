# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import sys, os
import numpy as np
import cv2

# This function is modeled off of P/R/F measure as described by Dave et al. (arXiv19)
def multilabel_metrics(prediction, gt, num_classes):
    """ Computes F-Measure, Precision, Recall, IoU, #objects detected, #confident objects detected, #GT objects.
        It computes these measures only of objects, not background (0)/table (1).
        Uses the Hungarian algorithm to match predicted masks with ground truth masks.

        A "confident object" is an object that is predicted with more than 0.75 F-measure

        @param gt: a [H x W] numpy.ndarray with ground truth masks
        @param prediction: a [H x W] numpy.ndarray with predicted masks

        @return: a dictionary with the metrics
    """

    precisions = np.zeros((num_classes, ), dtype=np.float32)
    recalls = np.zeros((num_classes, ), dtype=np.float32)
    f1s = np.zeros((num_classes, ), dtype=np.float32)
    count = np.zeros((num_classes, ), dtype=np.float32)

    # for each class
    for cls in range(num_classes):

        gt_mask = (gt == cls)
        pred_mask = (prediction == cls)
        A = np.logical_and(pred_mask, gt_mask)

        count_true = np.count_nonzero(A)
        count_pred = np.count_nonzero(pred_mask)
        count_gt = np.count_nonzero(gt_mask)

        # precision
        if count_pred > 0:
            precisions[cls] = float(count_true) / float(count_pred)
            
        # recall
        if count_gt > 0:
            recalls[cls] = float(count_true) / float(count_gt)
            count[cls] = 1
            
        # F-measure
        if precisions[cls] + recalls[cls] != 0:
            f1s[cls] = (2 * precisions[cls] * recalls[cls]) / (precisions[cls] + recalls[cls])


    return {'F-measure' : f1s,
            'Precision' : precisions,
            'Recall' : recalls,
            'Count': count}
