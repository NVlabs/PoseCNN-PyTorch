# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import torch
import torch.nn.functional as F
import time
import sys, os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from fcn.config import cfg
from fcn.render_utils import render_image
from fcn.test_common import refine_pose
from transforms3d.quaternions import mat2quat, quat2mat, qmult
from utils.se3 import *
from utils.nms import *
from utils.pose_error import re, te


def test_image(network, dataset, im_color, im_depth=None, im_index=None):
    """test on a single image"""

    # compute image blob
    im = im_color.astype(np.float32, copy=True)
    im -= cfg.PIXEL_MEANS
    height = im.shape[0]
    width = im.shape[1]
    im = np.transpose(im / 255.0, (2, 0, 1))
    im = im[np.newaxis, :, :, :]

    K = dataset._intrinsic_matrix
    K[2, 2] = 1
    Kinv = np.linalg.pinv(K)
    meta_data = np.zeros((1, 18), dtype=np.float32)
    meta_data[0, 0:9] = K.flatten()
    meta_data[0, 9:18] = Kinv.flatten()
    meta_data = torch.from_numpy(meta_data).cuda()

    if im_depth is not None:
        depth_tensor = torch.from_numpy(im_depth).cuda().float()
    else:
        depth_tensor = None

    # transfer to GPU
    inputs = torch.from_numpy(im).cuda()

    # run network
    if cfg.TRAIN.VERTEX_REG:

        if cfg.TRAIN.POSE_REG:
            out_label, out_vertex, rois, out_pose, out_quaternion = network(inputs, dataset.input_labels, meta_data, \
                dataset.input_extents, dataset.input_gt_boxes, dataset.input_poses, dataset.input_points, dataset.input_symmetry)
            labels = out_label[0]

            # combine poses
            rois = rois.detach().cpu().numpy()
            out_pose = out_pose.detach().cpu().numpy()
            out_quaternion = out_quaternion.detach().cpu().numpy()
            num = rois.shape[0]
            poses = out_pose.copy()
            for j in range(num):
                cls = int(rois[j, 1])
                if cls >= 0:
                    qt = out_quaternion[j, 4*cls:4*cls+4]
                    qt = qt / np.linalg.norm(qt)
                    # allocentric to egocentric
                    poses[j, 4] *= poses[j, 6]
                    poses[j, 5] *= poses[j, 6]
                    T = poses[j, 4:]
                    poses[j, :4] = allocentric2egocentric(qt, T)

            # filter out detections
            index = np.where(rois[:, -1] > cfg.TEST.DET_THRESHOLD)[0]
            rois = rois[index, :]
            poses = poses[index, :]

            # non-maximum suppression within class
            index = nms(rois, 0.2)
            rois = rois[index, :]
            poses = poses[index, :]
           
            # optimize depths
            if cfg.TEST.POSE_REFINE and im_depth is not None:
                poses_refined = refine_pose(labels, depth_tensor, rois, poses, meta_data, dataset)
            else:
                poses_refined = None

        else:
            # no pose regression
            out_label, out_vertex, rois, out_pose = network(inputs, dataset.input_labels, meta_data, \
                dataset.input_extents, dataset.input_gt_boxes, dataset.input_poses, dataset.input_points, dataset.input_symmetry)

            labels = out_label[0]

            rois = rois.detach().cpu().numpy()
            out_pose = out_pose.detach().cpu().numpy()
            poses = out_pose.copy()

            # filter out detections
            index = np.where(rois[:, -1] > cfg.TEST.DET_THRESHOLD)[0]
            rois = rois[index, :]
            poses = poses[index, :]
            poses_refined = None

            # non-maximum suppression within class
            index = nms(rois, 0.2)
            rois = rois[index, :]
            poses = poses[index, :]
    else:
        # segmentation only
        out_label = network(inputs, dataset.input_labels, dataset.input_meta_data, \
            dataset.input_extents, dataset.input_gt_boxes, dataset.input_poses, dataset.input_points, dataset.input_symmetry)
        labels = out_label[0]
        rois = np.zeros((0, 7), dtype=np.float32)
        poses = np.zeros((0, 7), dtype=np.float32)
        poses_refined = None

    im_pose, im_pose_refined, im_label = render_image(dataset, im_color, rois, poses, poses_refined, labels.cpu().numpy())
    if cfg.TEST.VISUALIZE:
        vis_test(dataset, im, im_depth, labels.cpu().numpy(), rois, poses, poses_refined, im_pose, im_pose_refined, out_vertex)

    return im_pose, im_pose_refined, im_label, labels.cpu().numpy(), rois, poses, poses_refined


def vis_test(dataset, im, im_depth, label, rois, poses, poses_refined, im_pose, im_pose_refine, out_vertex=None, im_index=None):

    """Visualize a testing results."""
    import matplotlib.pyplot as plt

    num_classes = len(dataset._class_colors_test)
    classes = dataset._classes_test
    class_colors = dataset._class_colors_test
    points = dataset._points_all_test
    intrinsic_matrix = dataset._intrinsic_matrix
    height = label.shape[0]
    width = label.shape[1]

    if out_vertex is not None:
        vertex_pred = out_vertex.detach().cpu().numpy()

    fig = plt.figure()
    plot = 1
    m = 2
    n = 3
    # show image
    im = im[0, :, :, :].copy()
    im = im.transpose((1, 2, 0)) * 255.0
    im += cfg.PIXEL_MEANS
    im = im[:, :, (2, 1, 0)]
    im = im.astype(np.uint8)
    ax = fig.add_subplot(m, n, plot)
    plot += 1
    plt.imshow(im)
    plt.axis('off')
    ax.set_title('input image')

    # show predicted label
    im_label = dataset.labels_to_image(label)
    ax = fig.add_subplot(m, n, plot)
    plot += 1
    plt.imshow(im_label)
    plt.axis('off')
    ax.set_title('predicted labels')

    if cfg.TRAIN.VERTEX_REG or cfg.TRAIN.VERTEX_REG_DELTA:

        # show predicted boxes
        ax = fig.add_subplot(m, n, plot)
        plot += 1
        plt.imshow(im)
        plt.axis('off')
        ax.set_title('predicted boxes')
        for j in range(rois.shape[0]):
            cls = rois[j, 1]
            x1 = rois[j, 2]
            y1 = rois[j, 3]
            x2 = rois[j, 4]
            y2 = rois[j, 5]
            plt.gca().add_patch(
                plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor=np.array(class_colors[int(cls)])/255.0, linewidth=3))

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            plt.plot(cx, cy, 'yo')

        '''
        # show predicted poses
        if cfg.TRAIN.POSE_REG:
            ax = fig.add_subplot(m, n, plot)
            plot += 1
            ax.set_title('predicted poses')
            plt.imshow(im)
            for j in range(rois.shape[0]):
                cls = int(rois[j, 1])
                print(classes[cls], rois[j, -1])
                if cls > 0 and rois[j, -1] > cfg.TEST.DET_THRESHOLD:
                    # extract 3D points
                    x3d = np.ones((4, points.shape[1]), dtype=np.float32)
                    x3d[0, :] = points[cls,:,0]
                    x3d[1, :] = points[cls,:,1]
                    x3d[2, :] = points[cls,:,2]

                    # projection
                    RT = np.zeros((3, 4), dtype=np.float32)
                    RT[:3, :3] = quat2mat(poses[j, :4])
                    RT[:, 3] = poses[j, 4:7]
                    x2d = np.matmul(intrinsic_matrix, np.matmul(RT, x3d))
                    x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
                    x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])
                    plt.plot(x2d[0, :], x2d[1, :], '.', color=np.divide(class_colors[cls], 255.0), alpha=0.5)

        if out_vertex is not None:
            # show predicted vertex targets
            vertex_target = vertex_pred[0, :, :, :]
            center = np.zeros((3, height, width), dtype=np.float32)

            for j in range(1, dataset._num_classes):
                index = np.where(label == j)
                if len(index[0]) > 0:
                    center[0, index[0], index[1]] = vertex_target[3*j, index[0], index[1]]
                    center[1, index[0], index[1]] = vertex_target[3*j+1, index[0], index[1]]
                    center[2, index[0], index[1]] = np.exp(vertex_target[3*j+2, index[0], index[1]])

            ax = fig.add_subplot(m, n, plot)
            plot += 1
            plt.imshow(center[0,:,:])
            plt.axis('off')
            ax.set_title('predicted center x') 

            ax = fig.add_subplot(m, n, plot)
            plot += 1
            plt.imshow(center[1,:,:])
            plt.axis('off')
            ax.set_title('predicted center y')

            ax = fig.add_subplot(m, n, plot)
            plot += 1
            plt.imshow(center[2,:,:])
            plt.axis('off')
            ax.set_title('predicted z')
        '''

    # show depth
    if im_depth is not None:
        ax = fig.add_subplot(m, n, plot)
        plot += 1
        plt.imshow(im_depth)
        plt.axis('off')
        ax.set_title('input depth')

    ax = fig.add_subplot(m, n, plot)
    plot += 1
    plt.imshow(im_pose)
    plt.axis('off')
    ax.set_title('estimated poses')

    if cfg.TEST.POSE_REFINE and im_pose_refine is not None and im_depth is not None:
        ax = fig.add_subplot(m, n, plot)
        plot += 1
        plt.imshow(im_pose_refine)
        plt.axis('off')
        ax.set_title('estimated poses refined')

    if im_index is not None:
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
        plt.show(block=False)
        plt.pause(1)
        filename = 'output/images/%06d.png' % im_index
        fig.savefig(filename)
        plt.close()
    else:
        plt.show()
