# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import torch
import torch.nn as nn
import time
import sys, os
import numpy as np
import cv2
import scipy
import matplotlib.pyplot as plt

from fcn.config import cfg
from fcn.test_common import refine_pose
from transforms3d.quaternions import mat2quat, quat2mat, qmult
from utils.se3 import *
from utils.nms import nms
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


def test(test_loader, background_loader, network, output_dir):

    batch_time = AverageMeter()
    epoch_size = len(test_loader)
    enum_background = enumerate(background_loader)

    # switch to test mode
    network.eval()

    for i, sample in enumerate(test_loader):

        # if 'is_testing' in sample and sample['is_testing'] == 0:
        #    continue

        end = time.time()

        inputs = sample['image_color']
        im_info = sample['im_info']

        # add background
        mask = sample['mask']
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
            if is_syn:
                inputs[j] = mask[j] * inputs[j] + (1 - mask[j]) * background_color[j]

        labels = sample['label'].cuda()
        meta_data = sample['meta_data'].cuda()
        extents = sample['extents'][0, :, :].repeat(cfg.TRAIN.GPUNUM, 1, 1).cuda()
        gt_boxes = sample['gt_boxes'].cuda()
        poses = sample['poses'].cuda()
        points = sample['points'][0, :, :, :].repeat(cfg.TRAIN.GPUNUM, 1, 1, 1).cuda()
        symmetry = sample['symmetry'][0, :].repeat(cfg.TRAIN.GPUNUM, 1).cuda()
        
        # compute output
        if cfg.TRAIN.VERTEX_REG:
            if cfg.TRAIN.POSE_REG:
                out_label, out_vertex, rois, out_pose, out_quaternion = network(inputs, labels, meta_data, extents, gt_boxes, poses, points, symmetry)
                
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

                # non-maximum suppression within class
                index = nms(rois, 0.5)
                rois = rois[index, :]
                poses = poses[index, :]

                # refine pose
                if cfg.TEST.POSE_REFINE:
                    im_depth = sample['im_depth'].numpy()[0]
                    depth_tensor = torch.from_numpy(im_depth).cuda().float()
                    labels_out = out_label[0]
                    poses_refined = refine_pose(labels_out, depth_tensor, rois, poses, sample['meta_data'], test_loader.dataset)
                else:
                    poses_refined = []
            else:
                out_label, out_vertex, rois, out_pose = network(inputs, labels, meta_data, extents, gt_boxes, poses, points, symmetry)
                rois = rois.detach().cpu().numpy()
                out_pose = out_pose.detach().cpu().numpy()
                poses = out_pose.copy()
                poses_refined = []

                # non-maximum suppression within class
                index = nms(rois, 0.5)
                rois = rois[index, :]
                poses = poses[index, :]
        else:
            out_label = network(inputs, labels, meta_data, extents, gt_boxes, poses, points, symmetry)
            out_vertex = []
            rois = []
            poses = []
            poses_refined = []

        if cfg.TEST.VISUALIZE:
            _vis_test(inputs, labels, out_label, out_vertex, rois, poses, poses_refined, sample, \
                test_loader.dataset._points_all, test_loader.dataset.classes, test_loader.dataset.class_colors)

        # measure elapsed time
        batch_time.update(time.time() - end)

        if not cfg.TEST.VISUALIZE:
            result = {'labels': out_label[0].detach().cpu().numpy(), 'rois': rois, 'poses': poses, 'poses_refined': poses_refined}
            if 'video_id' in sample and 'image_id' in sample:
                filename = os.path.join(output_dir, sample['video_id'][0] + '_' + sample['image_id'][0] + '.mat')
            else:
                result['meta_data_path'] = sample['meta_data_path']
                print(result['meta_data_path'])
                filename = os.path.join(output_dir, '%06d.mat' % i)
            print(filename)
            scipy.io.savemat(filename, result, do_compression=True)

        print('[%d/%d], batch time %.2f' % (i, epoch_size, batch_time.val))

    filename = os.path.join(output_dir, 'results_posecnn.mat')
    if os.path.exists(filename):
        os.remove(filename)


def _vis_test(inputs, labels, out_label, out_vertex, rois, poses, poses_refined, sample, points, classes, class_colors):

    """Visualize a mini-batch for debugging."""
    import matplotlib.pyplot as plt

    im_blob = inputs.cpu().numpy()
    label_blob = labels.cpu().numpy()
    label_pred = out_label.cpu().numpy()
    gt_poses = sample['poses'].numpy()
    meta_data_blob = sample['meta_data'].numpy()
    metadata = meta_data_blob[0, :]
    intrinsic_matrix = metadata[:9].reshape((3,3))
    gt_boxes = sample['gt_boxes'].numpy()
    extents = sample['extents'][0, :, :].numpy()

    if cfg.TRAIN.VERTEX_REG or cfg.TRAIN.VERTEX_REG_DELTA:
        vertex_targets = sample['vertex_targets'].numpy()
        vertex_pred = out_vertex.detach().cpu().numpy()

    m = 4
    n = 4
    for i in range(im_blob.shape[0]):
        fig = plt.figure()
        start = 1

        # show image
        im = im_blob[i, :, :, :].copy()
        im = im.transpose((1, 2, 0)) * 255.0
        im += cfg.PIXEL_MEANS
        im = im[:, :, (2, 1, 0)]
        im = im.astype(np.uint8)
        ax = fig.add_subplot(m, n, 1)
        plt.imshow(im)
        ax.set_title('color')
        start += 1

        # show gt boxes
        boxes = gt_boxes[i]
        for j in range(boxes.shape[0]):
            if boxes[j, 4] == 0:
                continue
            x1 = boxes[j, 0]
            y1 = boxes[j, 1]
            x2 = boxes[j, 2]
            y2 = boxes[j, 3]
            plt.gca().add_patch(
                plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='g', linewidth=3))

        # show gt label
        label_gt = label_blob[i, :, :, :]
        label_gt = label_gt.transpose((1, 2, 0))
        height = label_gt.shape[0]
        width = label_gt.shape[1]
        num_classes = label_gt.shape[2]
        im_label_gt = np.zeros((height, width, 3), dtype=np.uint8)
        for j in range(num_classes):
            I = np.where(label_gt[:, :, j] > 0)
            im_label_gt[I[0], I[1], :] = class_colors[j]

        ax = fig.add_subplot(m, n, start)
        start += 1
        plt.imshow(im_label_gt)
        ax.set_title('gt labels') 

        # show predicted label
        label = label_pred[i, :, :]
        height = label.shape[0]
        width = label.shape[1]
        im_label = np.zeros((height, width, 3), dtype=np.uint8)
        for j in range(num_classes):
            I = np.where(label == j)
            im_label[I[0], I[1], :] = class_colors[j]

        ax = fig.add_subplot(m, n, start)
        start += 1
        plt.imshow(im_label)
        ax.set_title('predicted labels')

        if cfg.TRAIN.VERTEX_REG or cfg.TRAIN.VERTEX_REG_DELTA:

            # show predicted boxes
            ax = fig.add_subplot(m, n, start)
            start += 1
            plt.imshow(im)

            ax.set_title('predicted boxes')
            for j in range(rois.shape[0]):
                if rois[j, 0] != i or rois[j, -1] < cfg.TEST.DET_THRESHOLD:
                    continue
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

            # show gt poses
            ax = fig.add_subplot(m, n, start)
            start += 1
            ax.set_title('gt poses')
            plt.imshow(im)

            pose_blob = gt_poses[i]
            for j in range(pose_blob.shape[0]):
                if pose_blob[j, 0] == 0:
                    continue

                cls = int(pose_blob[j, 1])
                # extract 3D points
                x3d = np.ones((4, points.shape[1]), dtype=np.float32)
                x3d[0, :] = points[cls,:,0]
                x3d[1, :] = points[cls,:,1]
                x3d[2, :] = points[cls,:,2]
               
                # projection
                RT = np.zeros((3, 4), dtype=np.float32)
                qt = pose_blob[j, 2:6]
                T = pose_blob[j, 6:]
                qt_new = allocentric2egocentric(qt, T)
                RT[:3, :3] = quat2mat(qt_new)
                RT[:, 3] = T
                x2d = np.matmul(intrinsic_matrix, np.matmul(RT, x3d))
                x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
                x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])
                plt.plot(x2d[0, :], x2d[1, :], '.', color=np.divide(class_colors[cls], 255.0), alpha=0.1)                    

            # show predicted poses
            ax = fig.add_subplot(m, n, start)
            start += 1
            ax.set_title('predicted poses')
            plt.imshow(im)
            for j in range(rois.shape[0]):
                if rois[j, 0] != i:
                    continue
                cls = int(rois[j, 1])
                if cls > 0:
                    print('%s: detection score %s' % (classes[cls], rois[j, -1]))
                if rois[j, -1] > cfg.TEST.DET_THRESHOLD:
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
                    plt.plot(x2d[0, :], x2d[1, :], '.', color=np.divide(class_colors[cls], 255.0), alpha=0.1)

            # show predicted refined poses
            if cfg.TEST.POSE_REFINE:
                ax = fig.add_subplot(m, n, start)
                start += 1
                ax.set_title('predicted refined poses')
                plt.imshow(im)
                for j in range(rois.shape[0]):
                    if rois[j, 0] != i:
                        continue
                    cls = int(rois[j, 1])
                    if rois[j, -1] > cfg.TEST.DET_THRESHOLD:
                        # extract 3D points
                        x3d = np.ones((4, points.shape[1]), dtype=np.float32)
                        x3d[0, :] = points[cls,:,0]
                        x3d[1, :] = points[cls,:,1]
                        x3d[2, :] = points[cls,:,2]

                        # projection
                        RT = np.zeros((3, 4), dtype=np.float32)
                        RT[:3, :3] = quat2mat(poses_refined[j, :4])
                        RT[:, 3] = poses_refined[j, 4:7]
                        x2d = np.matmul(intrinsic_matrix, np.matmul(RT, x3d))
                        x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
                        x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])
                        plt.plot(x2d[0, :], x2d[1, :], '.', color=np.divide(class_colors[cls], 255.0), alpha=0.1)

            # show gt vertex targets
            vertex_target = vertex_targets[i, :, :, :]
            center = np.zeros((3, height, width), dtype=np.float32)

            for j in range(1, num_classes):
                index = np.where(label_gt[:, :, j] > 0)
                if len(index[0]) > 0:
                    center[:, index[0], index[1]] = vertex_target[3*j:3*j+3, index[0], index[1]]

            ax = fig.add_subplot(m, n, start)
            start += 1
            plt.imshow(center[0,:,:])
            ax.set_title('gt center x') 

            ax = fig.add_subplot(m, n, start)
            start += 1
            plt.imshow(center[1,:,:])
            ax.set_title('gt center y')

            ax = fig.add_subplot(m, n, start)
            start += 1
            plt.imshow(np.exp(center[2,:,:]))
            ax.set_title('gt z')

            # show predicted vertex targets
            vertex_target = vertex_pred[i, :, :, :]
            center = np.zeros((3, height, width), dtype=np.float32)

            for j in range(1, num_classes):
                index = np.where(label == j)
                if len(index[0]) > 0:
                    center[:, index[0], index[1]] = vertex_target[3*j:3*j+3, index[0], index[1]]

            ax = fig.add_subplot(m, n, start)
            start += 1
            plt.imshow(center[0,:,:])
            ax.set_title('predicted center x') 

            ax = fig.add_subplot(m, n, start)
            start += 1
            plt.imshow(center[1,:,:])
            ax.set_title('predicted center y')

            ax = fig.add_subplot(m, n, start)
            start += 1
            plt.imshow(np.exp(center[2,:,:]))
            ax.set_title('predicted z')

        plt.show()
