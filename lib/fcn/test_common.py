# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import torch
import time
import sys, os
import numpy as np
import posecnn_cuda
import matplotlib.pyplot as plt
from transforms3d.quaternions import mat2quat, quat2mat
from matplotlib.patches import Circle
from fcn.config import cfg

def compute_index_sdf(rois):
    num = rois.shape[0]
    index_sdf = []
    for i in range(num):
        cls = int(rois[i, 1])
        if cls == 0:
            continue
        if cfg.TRAIN.CLASSES[cls] not in cfg.TEST.CLASSES:
            continue
        if rois[i, -1] < cfg.TEST.DET_THRESHOLD:
            continue
        index_sdf.append(i)
    return index_sdf

# SDF refinement
def refine_pose(im_label, im_depth, rois, poses, meta_data, dataset, visualize=False):

    start_time = time.time()
    width = im_depth.shape[1]
    height = im_depth.shape[0]
    sdf_optim = cfg.sdf_optimizer
    steps = cfg.TEST.NUM_SDF_ITERATIONS_TRACKING
    index_sdf = compute_index_sdf(rois)

    # backproject depth
    intrinsic_matrix = meta_data[0, :9].cpu().numpy().reshape((3, 3))
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    px = intrinsic_matrix[0, 2]
    py = intrinsic_matrix[1, 2]
    zfar = 6.0
    znear = 0.01
    im_pcloud = posecnn_cuda.backproject_forward(fx, fy, px, py, im_depth)[0]
    dpoints = im_pcloud[:,:,:3].cpu().numpy().reshape((-1, 3))

    # rendering
    num = len(index_sdf)
    poses_all = []
    cls_indexes = []
    for i in range(num):
        ind = index_sdf[i]
        cls = int(rois[ind, 1])
        cls_render = cfg.TEST.CLASSES.index(cfg.TRAIN.CLASSES[cls]) - 1
        cls_indexes.append(cls_render)
        qt = np.zeros((7, ), dtype=np.float32)
        qt[3:] = poses[ind, :4]
        qt[:3] = poses[ind, 4:]
        poses_all.append(qt)
    cfg.renderer.set_poses(poses_all)
    cfg.renderer.set_projection_matrix(width, height, fx, fy, px, py, znear, zfar)
    image_tensor = torch.cuda.FloatTensor(height, width, 4).detach()
    seg_tensor = torch.cuda.FloatTensor(height, width, 4).detach()
    pcloud_tensor = torch.cuda.FloatTensor(height, width, 4).detach()
    cfg.renderer.render(cls_indexes, image_tensor, seg_tensor, pc2_tensor=pcloud_tensor)
    pcloud_tensor = pcloud_tensor.flip(0)
    pcloud = pcloud_tensor[:,:,:3].cpu().numpy().reshape((-1, 3))   

    # refine translation
    poses_t = poses.copy()
    for i in range(num):
        ind = index_sdf[i]
        cls = int(rois[ind, 1])
        cls_render = cfg.TEST.CLASSES.index(cfg.TRAIN.CLASSES[cls]) - 1
        x1 = max(int(rois[ind, 2]), 0)
        y1 = max(int(rois[ind, 3]), 0)
        x2 = min(int(rois[ind, 4]), width-1)
        y2 = min(int(rois[ind, 5]), height-1)
        labels = torch.zeros_like(im_label)
        labels[y1:y2, x1:x2] = im_label[y1:y2, x1:x2]
        labels = labels.cpu().numpy().reshape((width * height, ))
        index = np.where((labels == cls) & np.isfinite(dpoints[:, 0]) & (pcloud[:, 0] != 0) & (dpoints[:, 0] != 0))[0]
        if len(index) > 10:
            T = np.mean(dpoints[index, :] - pcloud[index, :], axis=0)
            z_new = poses[ind, 6] + T[2]
            poses_t[ind, 6] = z_new
            poses_t[ind, 4] = (poses[ind, 4] / poses[ind, 6]) * z_new
            poses_t[ind, 5] = (poses[ind, 5] / poses[ind, 6]) * z_new
            print('object {}, class {}, z {}, z new {}'.format(i, dataset._classes_test[cls_render+1], poses[ind, 6], z_new))

    if visualize:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        image_tensor = image_tensor.flip(0)
        im = image_tensor.cpu().numpy() * 255
        im = im.astype(np.uint8)
        plt.imshow(im)
        plt.show()

    # compare the depth
    depth_meas_roi = im_pcloud[:, :, 2]
    mask_depth_meas = depth_meas_roi > 0
    mask_depth_valid = torch.isfinite(depth_meas_roi)

    # prepare data
    T_oc_init = np.zeros((num, 4, 4), dtype=np.float32)
    cls_index = torch.cuda.FloatTensor(0, 1)
    obj_index = torch.cuda.FloatTensor(0, 1)
    pix_index = torch.cuda.LongTensor(0, 2)
    for i in range(num):

        # pose
        ind = index_sdf[i]
        pose = poses_t[ind, :].copy()
        T_co = np.eye(4, dtype=np.float32)
        T_co[:3, :3] = quat2mat(pose[:4])
        T_co[:3, 3] = pose[4:]
        T_oc_init[i] = np.linalg.inv(T_co)

        # filter out points very far away
        z = float(pose[6])
        roi = rois[ind, :].copy()
        cls = int(roi[1])
        extent = 1.0 * np.mean(dataset._extents[cls, :])
        mask_distance = torch.abs(depth_meas_roi - z) < extent
            
        # mask label
        cls_render = cfg.TEST.CLASSES.index(cfg.TRAIN.CLASSES[cls]) - 1
        w = roi[4] - roi[2]
        h = roi[5] - roi[3]
        x1 = max(int(roi[2] - w / 2), 0)
        y1 = max(int(roi[3] - h / 2), 0)
        x2 = min(int(roi[4] + w / 2), width - 1)
        y2 = min(int(roi[5] + h / 2), height - 1)
        if im_label is not None:
            labels = torch.zeros_like(im_label)
            labels[y1:y2, x1:x2] = im_label[y1:y2, x1:x2]
            mask_label = labels == cls
        else:
            mask_label = torch.zeros_like(mask_depth_meas)
            mask_label[y1:y2, x1:x2] = 1

        mask = mask_label * mask_depth_meas * mask_depth_valid * mask_distance
        index_p = torch.nonzero(mask)
        n = index_p.shape[0]

        if n > 100:
            pix_index = torch.cat((pix_index, index_p), dim=0)
            index = cls_render * torch.ones((n, 1), dtype=torch.float32, device=0)
            cls_index = torch.cat((cls_index, index), dim=0)
            index = i * torch.ones((n, 1), dtype=torch.float32, device=0)
            obj_index = torch.cat((obj_index, index), dim=0)
            print('sdf {} points for object {}, class {} {}'.format(n, i, cls_render, dataset._classes_test[cls_render+1]))
        else:
            print('sdf {} points for object {}, class {} {}, no refinement'.format(n, i, cls_render, dataset._classes_test[cls_render+1]))

        if visualize and n <= 100:
            fig = plt.figure()
            ax = fig.add_subplot(2, 3, 1)
            plt.imshow(mask_label.cpu().numpy())
            ax.set_title('mask label')
            ax = fig.add_subplot(2, 3, 2)
            plt.imshow(mask_depth_meas.cpu().numpy())
            ax.set_title('mask_depth_meas')
            ax = fig.add_subplot(2, 3, 3)
            plt.imshow(mask_depth_valid.cpu().numpy())
            ax.set_title('mask_depth_valid')
            ax = fig.add_subplot(2, 3, 4)
            plt.imshow(mask_distance.cpu().numpy())
            ax.set_title('mask_distance')
            print(extent, z)
            ax = fig.add_subplot(2, 3, 5)
            plt.imshow(depth_meas_roi.cpu().numpy())
            ax.set_title('depth')
            plt.show()

    # data
    n = pix_index.shape[0]
    print('sdf with {} points'.format(n))
    if n == 0:
        return poses_t.copy()
    points = im_pcloud[pix_index[:, 0], pix_index[:, 1], :]
    points = torch.cat((points, cls_index, obj_index), dim=1)
    T_oc_opt = sdf_optim.refine_pose_layer(T_oc_init, points, steps=steps)

    # collect poses
    poses_refined = poses_t.copy()
    for i in range(num):
        RT_opt = T_oc_opt[i]
        ind = index_sdf[i]
        if RT_opt[3, 3] > 0:
            RT_opt = np.linalg.inv(RT_opt)
            poses_refined[ind, :4] = mat2quat(RT_opt[:3, :3])
            poses_refined[ind, 4:] = RT_opt[:3, 3]

    if visualize:
        points = points.cpu().numpy()
        for i in range(num):
            ind = index_sdf[i]
            roi = rois[ind, :].copy()
            cls = int(roi[1])
            cls = cfg.TEST.CLASSES.index(cfg.TRAIN.CLASSES[cls])
            T_co_init = np.linalg.inv(T_oc_init[i])

            pose = poses_refined[ind, :].copy()
            T_co_opt = np.eye(4, dtype=np.float32)
            T_co_opt[:3, :3] = quat2mat(pose[:4])
            T_co_opt[:3, 3] = pose[4:]

            index = np.where(points[:, 4] == i)[0]
            if len(index) == 0:
                continue
            pts = points[index, :4].copy()
            pts[:, 3] = 1.0

            # show points
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            points_obj = dataset._points_all_test[cls, :, :]
            points_init = np.matmul(np.linalg.inv(T_co_init), pts.transpose()).transpose()
            points_opt = np.matmul(np.linalg.inv(T_co_opt), pts.transpose()).transpose()

            ax.scatter(points_obj[::5, 0], points_obj[::5, 1], points_obj[::5, 2], color='yellow')
            ax.scatter(points_init[::5, 0], points_init[::5, 1], points_init[::5, 2], color='red')
            ax.scatter(points_opt[::5, 0], points_opt[::5, 1], points_opt[::5, 2], color='blue')

            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            ax.set_xlim(sdf_optim.xmins[cls-1], sdf_optim.xmaxs[cls-1])
            ax.set_ylim(sdf_optim.ymins[cls-1], sdf_optim.ymaxs[cls-1])
            ax.set_zlim(sdf_optim.zmins[cls-1], sdf_optim.zmaxs[cls-1])
            ax.set_title(dataset._classes_test[cls])
            plt.show()

    print('pose refine time %.6f' % (time.time() - start_time))
    return poses_refined
