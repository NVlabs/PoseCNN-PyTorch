# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import torch
import torch.utils.data as data

import os, math
import sys
import os.path as osp
from os.path import *
import numpy as np
import numpy.random as npr
import cv2
import scipy.io
import copy
import glob
try:
    import cPickle  # Use cPickle on Python 2.7
except ImportError:
    import pickle as cPickle

import datasets
from fcn.config import cfg
from utils.blob import pad_im, chromatic_transform, add_noise, add_noise_cuda
from transforms3d.quaternions import mat2quat, quat2mat
from utils.se3 import *
from utils.pose_error import *
from utils.cython_bbox import bbox_overlaps
from utils.segmentation_evaluation import multilabel_metrics

def VOCap(rec, prec):
    index = np.where(np.isfinite(rec))[0]
    rec = rec[index]
    prec = prec[index]
    if len(rec) == 0 or len(prec) == 0:
        ap = 0
    else:
        mrec = np.insert(rec, 0, 0)
        mrec = np.append(mrec, 0.1)
        mpre = np.insert(prec, 0, 0)
        mpre = np.append(mpre, prec[-1])
        for i in range(1, len(mpre)):
            mpre[i] = max(mpre[i], mpre[i-1])
        i = np.where(mrec[1:] != mrec[:-1])[0] + 1
        ap = np.sum(np.multiply(mrec[i] - mrec[i-1], mpre[i])) * 10
    return ap

class YCBSelfSupervision(data.Dataset, datasets.imdb):
    def __init__(self, image_set, ycb_self_supervision_path = None):

        self._name = 'ycb_self_supervision_' + image_set
        self._image_set = image_set
        self._ycb_self_supervision_path = self._get_default_path() if ycb_self_supervision_path is None \
                            else ycb_self_supervision_path
        self._data_path = os.path.join(self._ycb_self_supervision_path, 'data')
        self._model_path = os.path.join(datasets.ROOT_DIR, 'data', 'models')

        # define all the classes
        self._classes_all = ('__background__', '002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle', \
                         '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana', '019_pitcher_base', \
                         '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '036_wood_block', '037_scissors', '040_large_marker', \
                         '051_large_clamp', '052_extra_large_clamp', '061_foam_brick', 'holiday_cup1', 'holiday_cup2', 'sanning_mug', \
                         '001_chips_can', 'block_red_big', 'block_green_big', 'block_blue_big', 'block_yellow_big', \
                         'block_red_small', 'block_green_small', 'block_blue_small', 'block_yellow_small', \
                         'block_red_median', 'block_green_median', 'block_blue_median', 'block_yellow_median', 'fusion_duplo_dude', 'cabinet_handle')
        self._num_classes_all = len(self._classes_all)
        self._class_colors_all = [(255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), \
                              (0, 0, 128), (0, 128, 0), (128, 0, 0), (128, 128, 0), (128, 0, 128), (0, 128, 128), \
                              (0, 64, 0), (64, 0, 0), (0, 0, 64), (64, 64, 0), (64, 0, 64), (0, 64, 64), \
                              (192, 0, 0), (0, 192, 0), (0, 0, 192), (192, 192, 0), (192, 0, 192), (0, 192, 192), (32, 0, 0), \
                              (150, 0, 0), (0, 150, 0), (0, 0, 150), (150, 150, 0), (75, 0, 0), (0, 75, 0), (0, 0, 75), (75, 75, 0), \
                              (200, 0, 0), (0, 200, 0), (0, 0, 200), (200, 200, 0), (16, 16, 0), (16, 16, 16)]
        self._extents_all = self._load_object_extents()

        self._width = cfg.TRAIN.SYN_WIDTH
        self._height = cfg.TRAIN.SYN_HEIGHT
        self._intrinsic_matrix = np.array([[616.3653,    0.,      310.25882],
                                           [  0.,      616.20294, 236.59981],
                                           [  0.,        0.,        1.     ]])

        if self._width == 1280:
            self._intrinsic_matrix = np.array([[599.48681641,   0.,         639.84338379],
                                               [  0.,         599.24389648, 366.09042358],
                                               [  0.,           0.,           1.        ]])


        # select a subset of classes
        self._classes = [self._classes_all[i] for i in cfg.TRAIN.CLASSES]
        self._classes_test = [self._classes_all[i] for i in cfg.TEST.CLASSES]
        self._num_classes = len(self._classes)
        self._class_colors = [self._class_colors_all[i] for i in cfg.TRAIN.CLASSES]
        self._class_colors_test = [self._class_colors_all[i] for i in cfg.TEST.CLASSES]
        self._symmetry = np.array(cfg.TRAIN.SYMMETRY).astype(np.float32)
        self._symmetry_test = np.array(cfg.TEST.SYMMETRY).astype(np.float32)
        self._extents = self._extents_all[cfg.TRAIN.CLASSES]
        self._extents_test = self._extents_all[cfg.TEST.CLASSES]
        self._points, self._points_all, self._point_blob = self._load_object_points(self._classes, self._extents, self._symmetry)
        self._points_test, self._points_all_test, self._point_blob_test = \
            self._load_object_points(self._classes_test, self._extents_test, self._symmetry_test)
        self._pixel_mean = torch.tensor(cfg.PIXEL_MEANS / 255.0).cuda().float()

        self._classes_other = []
        for i in range(self._num_classes_all):
            if i not in cfg.TRAIN.CLASSES:
                # do not use clamp
                if i == 19 and 20 in cfg.TRAIN.CLASSES:
                    continue
                if i == 20 and 19 in cfg.TRAIN.CLASSES:
                    continue
                self._classes_other.append(i)
        self._num_classes_other = len(self._classes_other)

        # 3D model paths
        self.model_sdf_paths = ['{}/{}/textured_simple_low_res.pth'.format(self._model_path, cls) for cls in self._classes_all[1:]]
        self.model_colors = [np.array(self._class_colors_all[i]) / 255.0 for i in range(1, len(self._classes_all))]

        self.model_mesh_paths = []
        for cls in self._classes_all[1:]:
            filename = '{}/{}/textured_simple.ply'.format(self._model_path, cls)
            if osp.exists(filename):
                self.model_mesh_paths.append(filename)
                continue
            filename = '{}/{}/textured_simple.obj'.format(self._model_path, cls)
            if osp.exists(filename):
                self.model_mesh_paths.append(filename)

        self.model_texture_paths = []
        for cls in self._classes_all[1:]:
            filename = '{}/{}/texture_map.png'.format(self._model_path, cls)
            if osp.exists(filename):
                self.model_texture_paths.append(filename)
            else:
                self.model_texture_paths.append('')

        # target meshes
        self.model_colors_target = [np.array(self._class_colors_all[i]) / 255.0 for i in cfg.TRAIN.CLASSES[1:]]
        self.model_mesh_paths_target = []
        for cls in self._classes[1:]:
            filename = '{}/{}/textured_simple.ply'.format(self._model_path, cls)
            if osp.exists(filename):
                self.model_mesh_paths_target.append(filename)
                continue
            filename = '{}/{}/textured_simple.obj'.format(self._model_path, cls)
            if osp.exists(filename):
                self.model_mesh_paths_target.append(filename)

        self.model_texture_paths_target = []
        for cls in self._classes[1:]:
            filename = '{}/{}/texture_map.png'.format(self._model_path, cls)
            if osp.exists(filename):
                self.model_texture_paths_target.append(filename)
            else:
                self.model_texture_paths_target.append('')

        self._class_to_ind = dict(zip(self._classes, range(self._num_classes)))
        self._image_ext = '.png'
        self._image_index = self._load_image_set_index(image_set)

        if (cfg.MODE == 'TRAIN' and cfg.TRAIN.SYNTHESIZE) or (cfg.MODE == 'TEST' and cfg.TEST.SYNTHESIZE):
            self._size = len(self._image_index) * (cfg.TRAIN.SYN_RATIO+1)
        else:
            self._size = len(self._image_index)

        if self._size > cfg.TRAIN.MAX_ITERS_PER_EPOCH * cfg.TRAIN.IMS_PER_BATCH:
            self._size = cfg.TRAIN.MAX_ITERS_PER_EPOCH * cfg.TRAIN.IMS_PER_BATCH
        self._roidb = self.gt_roidb()
        if cfg.MODE == 'TRAIN' or cfg.TEST.VISUALIZE:
            self._perm = np.random.permutation(np.arange(len(self._roidb)))
        else:
            self._perm = np.arange(len(self._roidb))
        self._cur = 0
        self._build_uniform_poses()
        self.lb_shift = -0.2
        self.ub_shift = 0.2
        self.lb_scale = 0.8
        self.ub_scale = 2.0

        assert os.path.exists(self._ycb_self_supervision_path), \
                'ycb_self_supervision path does not exist: {}'.format(self._ycb_self_supervision_path)
        assert os.path.exists(self._data_path), \
                'Data path does not exist: {}'.format(self._data_path)

        # construct fake inputs
        label_blob = np.zeros((1, self._num_classes, self._height, self._width), dtype=np.float32)
        pose_blob = np.zeros((1, self._num_classes, 9), dtype=np.float32)
        gt_boxes = np.zeros((1, self._num_classes, 5), dtype=np.float32)

        # construct the meta data
        K = self._intrinsic_matrix
        Kinv = np.linalg.pinv(K)
        meta_data_blob = np.zeros((1, 18), dtype=np.float32)
        meta_data_blob[0, 0:9] = K.flatten()
        meta_data_blob[0, 9:18] = Kinv.flatten()

        self.input_labels = torch.from_numpy(label_blob).cuda()
        self.input_meta_data = torch.from_numpy(meta_data_blob).cuda()
        self.input_extents = torch.from_numpy(self._extents).cuda()
        self.input_gt_boxes = torch.from_numpy(gt_boxes).cuda()
        self.input_poses = torch.from_numpy(pose_blob).cuda()
        self.input_points = torch.from_numpy(self._point_blob).cuda()
        self.input_symmetry = torch.from_numpy(self._symmetry).cuda()


    def _render_item(self):

        height = cfg.TRAIN.SYN_HEIGHT
        width = cfg.TRAIN.SYN_WIDTH
        fx = self._intrinsic_matrix[0, 0]
        fy = self._intrinsic_matrix[1, 1]
        px = self._intrinsic_matrix[0, 2]
        py = self._intrinsic_matrix[1, 2]
        zfar = 6.0
        znear = 0.01

        # sample target objects
        if cfg.TRAIN.SYN_SAMPLE_OBJECT:
            maxnum = np.minimum(self.num_classes-1, cfg.TRAIN.SYN_MAX_OBJECT)
            num = np.random.randint(cfg.TRAIN.SYN_MIN_OBJECT, maxnum+1)
            perm = np.random.permutation(np.arange(self.num_classes-1))
            indexes_target = perm[:num] + 1
        else:
            num = self.num_classes - 1
            indexes_target = np.arange(num) + 1
        num_target = num
        cls_indexes = [cfg.TRAIN.CLASSES[i]-1 for i in indexes_target]

        # sample other objects as distractors
        if cfg.TRAIN.SYN_SAMPLE_DISTRACTOR:
            num_other = min(5, self._num_classes_other)
            num_selected = np.random.randint(0, num_other+1)
            perm = np.random.permutation(np.arange(self._num_classes_other))
            indexes = perm[:num_selected]
            for i in range(num_selected):
                cls_indexes.append(self._classes_other[indexes[i]]-1)
        else:
            num_selected = 0

        # sample poses
        num = num_target + num_selected
        poses_all = []
        for i in range(num):
            qt = np.zeros((7, ), dtype=np.float32)
            # rotation
            cls = int(cls_indexes[i])
            if self.pose_indexes[cls] >= len(self.pose_lists[cls]):
                self.pose_indexes[cls] = 0
                self.pose_lists[cls] = np.random.permutation(np.arange(len(self.eulers)))
            yaw = self.eulers[self.pose_lists[cls][self.pose_indexes[cls]]][0] + 15 * np.random.randn()
            pitch = self.eulers[self.pose_lists[cls][self.pose_indexes[cls]]][1] + 15 * np.random.randn()
            pitch = np.clip(pitch, -90, 90)
            roll = self.eulers[self.pose_lists[cls][self.pose_indexes[cls]]][2] + 15 * np.random.randn()
            qt[3:] = euler2quat(yaw * math.pi / 180.0, pitch * math.pi / 180.0, roll * math.pi / 180.0, 'syxz')
            self.pose_indexes[cls] += 1

            # translation
            bound = cfg.TRAIN.SYN_BOUND
            if i == 0 or i >= num_target or np.random.rand(1) > 0.5:
                qt[0] = np.random.uniform(-bound, bound)
                qt[1] = np.random.uniform(-bound, bound)
                qt[2] = np.random.uniform(cfg.TRAIN.SYN_TNEAR, cfg.TRAIN.SYN_TFAR)
            else:
                # sample an object nearby
                object_id = np.random.randint(0, i, size=1)[0]
                extent = np.mean(self._extents_all[cls+1, :])

                flag = np.random.randint(0, 2)
                if flag == 0:
                    flag = -1
                qt[0] = poses_all[object_id][0] + flag * extent * np.random.uniform(1.0, 1.5)
                if np.absolute(qt[0]) > bound:
                    qt[0] = poses_all[object_id][0] - flag * extent * np.random.uniform(1.0, 1.5)
                if np.absolute(qt[0]) > bound:
                    qt[0] = np.random.uniform(-bound, bound)

                flag = np.random.randint(0, 2)
                if flag == 0:
                    flag = -1
                qt[1] = poses_all[object_id][1] + flag * extent * np.random.uniform(1.0, 1.5)
                if np.absolute(qt[1]) > bound:
                    qt[1] = poses_all[object_id][1] - flag * extent * np.random.uniform(1.0, 1.5)
                if np.absolute(qt[1]) > bound:
                    qt[1] = np.random.uniform(-bound, bound)

                qt[2] = poses_all[object_id][2] - extent * np.random.uniform(2.0, 4.0)
                if qt[2] < cfg.TRAIN.SYN_TNEAR:
                    qt[2] = poses_all[object_id][2] + extent * np.random.uniform(2.0, 4.0)

            poses_all.append(qt)
        cfg.renderer.set_poses(poses_all)

        # sample lighting
        cfg.renderer.set_light_pos(np.random.uniform(-0.5, 0.5, 3))

        intensity = np.random.uniform(0.8, 2)
        light_color = intensity * np.random.uniform(0.9, 1.1, 3)
        cfg.renderer.set_light_color(light_color)
            
        # rendering
        cfg.renderer.set_projection_matrix(width, height, fx, fy, px, py, znear, zfar)
        image_tensor = torch.cuda.FloatTensor(height, width, 4).detach()
        seg_tensor = torch.cuda.FloatTensor(height, width, 4).detach()
        pc_tensor = torch.cuda.FloatTensor(height, width, 4).detach()
        cfg.renderer.render(cls_indexes, image_tensor, seg_tensor, pc2_tensor=pc_tensor)
        image_tensor = image_tensor.flip(0)
        seg_tensor = seg_tensor.flip(0)
        pc_tensor = pc_tensor.flip(0)

        # foreground mask
        seg = seg_tensor[:,:,2] + 256*seg_tensor[:,:,1] + 256*256*seg_tensor[:,:,0]
        mask = (seg != 0).unsqueeze(0).repeat((3, 1, 1)).float()

        # RGB to BGR order
        im = image_tensor.cpu().numpy()
        im = np.clip(im, 0, 1)
        im = im[:, :, (2, 1, 0)] * 255
        im = im.astype(np.uint8)

        # XYZ coordinates in camera frame
        im_depth = pc_tensor.cpu().numpy()
        im_depth = im_depth[:, :, :3]
        im_depth_return = im_depth[:, :, 2].copy()

        im_label = seg_tensor.cpu().numpy()
        im_label = im_label[:, :, (2, 1, 0)] * 255
        im_label = np.round(im_label).astype(np.uint8)
        im_label = np.clip(im_label, 0, 255)
        im_label, im_label_all = self.process_label_image(im_label)

        centers = np.zeros((num, 2), dtype=np.float32)
        rcenters = cfg.renderer.get_centers()
        for i in range(num):
            centers[i, 0] = rcenters[i][1] * width
            centers[i, 1] = rcenters[i][0] * height
        centers = centers[:num_target, :]

        '''
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(3, 2, 1)
        plt.imshow(im[:, :, (2, 1, 0)])
        for i in range(num_target):
            plt.plot(centers[i, 0], centers[i, 1], 'yo')
        ax = fig.add_subplot(3, 2, 2)
        plt.imshow(im_label)
        ax = fig.add_subplot(3, 2, 3)
        plt.imshow(im_depth[:, :, 0])
        ax = fig.add_subplot(3, 2, 4)
        plt.imshow(im_depth[:, :, 1])
        ax = fig.add_subplot(3, 2, 5)
        plt.imshow(im_depth[:, :, 2])
        plt.show()
        #'''

        # chromatic transform
        if cfg.TRAIN.CHROMATIC and cfg.MODE == 'TRAIN' and np.random.rand(1) > 0.1:
            im = chromatic_transform(im)

        im_cuda = torch.from_numpy(im).cuda().float() / 255.0
        if cfg.TRAIN.ADD_NOISE and cfg.MODE == 'TRAIN' and np.random.rand(1) > 0.1:
            im_cuda = add_noise_cuda(im_cuda)
        im_cuda -= self._pixel_mean
        im_cuda = im_cuda.permute(2, 0, 1)

        if cfg.INPUT == 'DEPTH' or cfg.INPUT == 'RGBD':

            # depth mask
            z_im = im_depth[:, :, 2]
            mask_depth = z_im > 0.0
            mask_depth = mask_depth.astype('float')
            mask_depth_cuda = torch.from_numpy(mask_depth).cuda().float()
            mask_depth_cuda.unsqueeze_(0)

            im_cuda_depth = torch.from_numpy(im_depth).cuda().float()
            if cfg.TRAIN.ADD_NOISE and cfg.MODE == 'TRAIN' and np.random.rand(1) > 0.1:
                im_cuda_depth = add_noise_depth_cuda(im_cuda_depth)
            im_cuda_depth = im_cuda_depth.permute(2, 0, 1)
        else:
            im_cuda_depth = im_cuda.clone()
            mask_depth_cuda = torch.cuda.FloatTensor(1, height, width).fill_(0)

        # label blob
        classes = np.array(range(self.num_classes))
        label_blob = np.zeros((self.num_classes, self._height, self._width), dtype=np.float32)
        label_blob[0, :, :] = 1.0
        for i in range(1, self.num_classes):
            I = np.where(im_label == classes[i])
            if len(I[0]) > 0:
                label_blob[i, I[0], I[1]] = 1.0
                label_blob[0, I[0], I[1]] = 0.0

        # poses and boxes
        pose_blob = np.zeros((self.num_classes, 9), dtype=np.float32)
        gt_boxes = np.zeros((self.num_classes, 5), dtype=np.float32)
        count = 0
        for i in range(num_target):
            cls = int(indexes_target[i])
            T = poses_all[i][:3]
            qt = poses_all[i][3:]

            I = np.where(im_label == cls)
            if len(I[0]) == 0:
                continue

            # compute box
            x3d = np.ones((4, self._points_all.shape[1]), dtype=np.float32)
            x3d[0, :] = self._points_all[cls,:,0]
            x3d[1, :] = self._points_all[cls,:,1]
            x3d[2, :] = self._points_all[cls,:,2]
            RT = np.zeros((3, 4), dtype=np.float32)
            RT[:3, :3] = quat2mat(qt)
            RT[:, 3] = T
            x2d = np.matmul(self._intrinsic_matrix, np.matmul(RT, x3d))
            x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
            x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])

            x1 = np.min(x2d[0, :])
            y1 = np.min(x2d[1, :])
            x2 = np.max(x2d[0, :])
            y2 = np.max(x2d[1, :])
            if x1 > width or y1 > height or x2 < 0 or y2 < 0:
                continue

            gt_boxes[count, 0] = x1
            gt_boxes[count, 1] = y1
            gt_boxes[count, 2] = x2
            gt_boxes[count, 3] = y2
            gt_boxes[count, 4] = cls

            pose_blob[count, 0] = 1
            pose_blob[count, 1] = cls
            # egocentric to allocentric
            qt_allocentric = egocentric2allocentric(qt, T)
            if qt_allocentric[0] < 0:
                qt_allocentric = -1 * qt_allocentric
            pose_blob[count, 2:6] = qt_allocentric
            pose_blob[count, 6:] = T
            count += 1


        # construct the meta data
        """
        format of the meta_data
        intrinsic matrix: meta_data[0 ~ 8]
        inverse intrinsic matrix: meta_data[9 ~ 17]
        """
        K = self._intrinsic_matrix
        K[2, 2] = 1
        Kinv = np.linalg.pinv(K)
        meta_data_blob = np.zeros(18, dtype=np.float32)
        meta_data_blob[0:9] = K.flatten()
        meta_data_blob[9:18] = Kinv.flatten()

        # vertex regression target
        if cfg.TRAIN.VERTEX_REG:
            vertex_targets, vertex_weights = self._generate_vertex_targets(im_label, indexes_target, centers, poses_all, classes, self.num_classes)
        elif cfg.TRAIN.VERTEX_REG_DELTA and cfg.INPUT == 'DEPTH' or cfg.INPUT == 'RGBD':
            vertex_targets, vertex_weights = self._generate_vertex_deltas(im_label, indexes_target, centers, poses_all,
                                                                           classes, self.num_classes, im_depth)
        else:
            vertex_targets = []
            vertex_weights = []

        im_info = np.array([im.shape[1], im.shape[2], cfg.TRAIN.SCALES_BASE[0], 1], dtype=np.float32)

        sample = {'image_color': im_cuda,
                  'im_depth': im_depth_return,
                  'label': label_blob,
                  'mask': mask,
                  'meta_data': meta_data_blob,
                  'poses': pose_blob,
                  'extents': self._extents,
                  'points': self._point_blob,
                  'symmetry': self._symmetry,
                  'gt_boxes': gt_boxes,
                  'im_info': im_info,
                  'meta_data_path': ''}

        if cfg.TRAIN.VERTEX_REG or cfg.TRAIN.VERTEX_REG_DELTA:
            sample['vertex_targets'] = vertex_targets
            sample['vertex_weights'] = vertex_weights

        # affine transformation
        if cfg.TRAIN.AFFINE:
            shift = np.float32([np.random.uniform(self.lb_shift, self.ub_shift), np.random.uniform(self.lb_shift, self.ub_shift)])
            scale = np.random.uniform(self.lb_scale, self.ub_scale)
            affine_matrix = np.float32([[scale, 0, shift[0]], [0, scale, shift[1]]])

            affine_1 = np.eye(3, dtype=np.float32)
            affine_1[0, 2] = -0.5 * self._width
            affine_1[1, 2] = -0.5 * self._height

            affine_2 = np.eye(3, dtype=np.float32)
            affine_2[0, 0] = 1.0 / scale
            affine_2[0, 2] = -shift[0] * 0.5 * self._width / scale
            affine_2[1, 1] = 1.0 / scale
            affine_2[1, 2] = -shift[1] * 0.5 * self._height / scale

            affine_3 = np.matmul(affine_2, affine_1)
            affine_4 = np.matmul(np.linalg.inv(affine_1), affine_3)
            affine_matrix_coordinate = affine_4[:3, :]

            sample['affine_matrix'] = torch.from_numpy(affine_matrix).cuda()
            sample['affine_matrix_coordinate'] = torch.from_numpy(affine_matrix_coordinate).cuda()

        return sample


    def __getitem__(self, index):

        is_syn = 0
        if ((cfg.MODE == 'TRAIN' and cfg.TRAIN.SYNTHESIZE) or (cfg.MODE == 'TEST' and cfg.TEST.SYNTHESIZE)) and (index % (cfg.TRAIN.SYN_RATIO+1) != 0):
            is_syn = 1

        if is_syn:
            return self._render_item()

        if self._cur >= len(self._roidb):
            self._perm = np.random.permutation(np.arange(len(self._roidb)))
            self._cur = 0
        db_ind = self._perm[self._cur]
        roidb = self._roidb[db_ind]
        self._cur += 1

        # Get the input image blob
        random_scale_ind = npr.randint(0, high=len(cfg.TRAIN.SCALES_BASE))
        im_blob, im_depth, im_scale, height, width = self._get_image_blob(roidb, random_scale_ind)

        # build the label blob
        label_blob, mask, meta_data_blob, pose_blob, gt_boxes, vertex_targets, vertex_weights \
            = self._get_label_blob(roidb, self._num_classes, im_scale, height, width)

        im_info = np.array([im_blob.shape[1], im_blob.shape[2], im_scale, is_syn], dtype=np.float32)
        mask_depth_cuda = torch.cuda.FloatTensor(1, height, width).fill_(0)

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
                  'meta_data_path': roidb['meta_data']}

        if cfg.TRAIN.VERTEX_REG:
            sample['vertex_targets'] = vertex_targets
            sample['vertex_weights'] = vertex_weights

        return sample


    def _get_image_blob(self, roidb, scale_ind):    

        # rgba
        rgba = pad_im(cv2.imread(roidb['image'], cv2.IMREAD_UNCHANGED), 16)
        if rgba.shape[2] == 4:
            im = np.copy(rgba[:,:,:3])
            alpha = rgba[:,:,3]
            I = np.where(alpha == 0)
            im[I[0], I[1], :] = 0
        else:
            im = rgba

        im_scale = cfg.TRAIN.SCALES_BASE[scale_ind]
        if im_scale != 1.0:
            im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        height = im.shape[0]
        width = im.shape[1]

        if roidb['flipped']:
            im = im[:, ::-1, :]

        # chromatic transform
        if cfg.TRAIN.CHROMATIC and cfg.MODE == 'TRAIN' and np.random.rand(1) > 0.1:
            im = chromatic_transform(im)

        im_cuda = torch.from_numpy(im).cuda().float() / 255.0
        if cfg.TRAIN.ADD_NOISE and cfg.MODE == 'TRAIN' and np.random.rand(1) > 0.1:
            im_cuda = add_noise_cuda(im_cuda)
        im_cuda -= self._pixel_mean
        im_cuda = im_cuda.permute(2, 0, 1)

        # depth image
        im_depth = pad_im(cv2.imread(roidb['depth'], cv2.IMREAD_UNCHANGED), 16)
        if im_scale != 1.0:
            im_depth = cv2.resize(im_depth, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_NEAREST)
        im_depth = im_depth.astype(np.float32) / 1000.0

        return im_cuda, im_depth, im_scale, height, width


    def _get_label_blob(self, roidb, num_classes, im_scale, height, width):
        """ build the label blob """

        meta_data = scipy.io.loadmat(roidb['meta_data'])
        meta_data['cls_indexes'] = meta_data['cls_indexes'].flatten()
        classes = np.array(cfg.TRAIN.CLASSES)
        classes_test = np.array(cfg.TEST.CLASSES).flatten()

        intrinsic_matrix = np.matrix(meta_data['intrinsic_matrix'])
        fx = intrinsic_matrix[0, 0]
        fy = intrinsic_matrix[1, 1]
        px = intrinsic_matrix[0, 2]
        py = intrinsic_matrix[1, 2]
        zfar = 6.0
        znear = 0.01

        # poses
        poses = meta_data['poses']
        if len(poses.shape) == 2:
            poses = np.reshape(poses, (3, 4, 1))
        if roidb['flipped']:
            poses = _flip_poses(poses, meta_data['intrinsic_matrix'], width)
        num = poses.shape[2]

        # render poses to get the label image
        cls_indexes = []
        poses_all = []
        qt = np.zeros((7, ), dtype=np.float32)
        for i in range(num):
            RT = poses[:, :, i]
            qt[:3] = RT[:, 3]
            qt[3:] = mat2quat(RT[:, :3])
            if cfg.MODE == 'TEST':
                index = np.where(classes_test == meta_data['cls_indexes'][i])[0]
                cls_indexes.append(index[0])
            else:
                cls_indexes.append(meta_data['cls_indexes'][i] - 1)
            poses_all.append(qt.copy())
            
        # rendering
        cfg.renderer.set_poses(poses_all)
        cfg.renderer.set_light_pos([0, 0, 0])
        cfg.renderer.set_light_color([1, 1, 1])
        cfg.renderer.set_projection_matrix(width, height, fx, fy, px, py, znear, zfar)
        image_tensor = torch.cuda.FloatTensor(height, width, 4).detach()
        seg_tensor = torch.cuda.FloatTensor(height, width, 4).detach()
        cfg.renderer.render(cls_indexes, image_tensor, seg_tensor)
        image_tensor = image_tensor.flip(0)
        seg_tensor = seg_tensor.flip(0)

        # semantic labels
        im_label = seg_tensor.cpu().numpy()
        im_label = im_label[:, :, (2, 1, 0)] * 255
        im_label = np.round(im_label).astype(np.uint8)
        im_label = np.clip(im_label, 0, 255)
        im_label, im_label_all = self.process_label_image(im_label)

        centers = np.zeros((num, 2), dtype=np.float32)
        rcenters = cfg.renderer.get_centers()
        for i in range(num):
            centers[i, 0] = rcenters[i][1] * width
            centers[i, 1] = rcenters[i][0] * height

        # label blob
        label_blob = np.zeros((num_classes, height, width), dtype=np.float32)
        label_blob[0, :, :] = 1.0
        for i in range(1, num_classes):
            I = np.where(im_label_all == classes[i])
            if len(I[0]) > 0:
                label_blob[i, I[0], I[1]] = 1.0
                label_blob[0, I[0], I[1]] = 0.0

        '''
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        plt.imshow(im_label)
        for i in range(num):
            plt.plot(centers[i, 0], centers[i, 1], 'yo')
        ax = fig.add_subplot(1, 2, 2)
        plt.imshow(im_label_all)
        plt.show()
        #'''

        # foreground mask
        seg = torch.from_numpy((im_label != 0).astype(np.float32))
        mask = seg.unsqueeze(0).repeat((3, 1, 1)).float().cuda()

        # gt poses
        pose_blob = np.zeros((num_classes, 9), dtype=np.float32)
        gt_boxes = np.zeros((num_classes, 5), dtype=np.float32)
        count = 0
        for i in range(num):
            cls = int(meta_data['cls_indexes'][i])
            ind = np.where(classes == cls)[0]
            if len(ind) > 0:

                I = np.where(im_label == ind[0])
                if len(I[0]) == 0:
                    continue

                R = poses[:, :3, i]
                T = poses[:, 3, i]

                # compute box
                x3d = np.ones((4, self._points_all.shape[1]), dtype=np.float32)
                x3d[0, :] = self._points_all[ind,:,0]
                x3d[1, :] = self._points_all[ind,:,1]
                x3d[2, :] = self._points_all[ind,:,2]
                RT = np.zeros((3, 4), dtype=np.float32)
                RT[:3, :3] = R
                RT[:, 3] = T
                x2d = np.matmul(meta_data['intrinsic_matrix'], np.matmul(RT, x3d))
                x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
                x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])

                x1 = np.min(x2d[0, :]) * im_scale
                y1 = np.min(x2d[1, :]) * im_scale
                x2 = np.max(x2d[0, :]) * im_scale
                y2 = np.max(x2d[1, :]) * im_scale
                if x1 > width or y1 > height or x2 < 0 or y2 < 0:
                    continue
                gt_boxes[count, 0] = x1
                gt_boxes[count, 1] = y1
                gt_boxes[count, 2] = x2
                gt_boxes[count, 3] = y2
                gt_boxes[count, 4] = ind

                # pose
                pose_blob[count, 0] = 1
                pose_blob[count, 1] = ind
                qt = mat2quat(R)

                # egocentric to allocentric
                qt_allocentric = egocentric2allocentric(qt, T)
                if qt_allocentric[0] < 0:
                   qt_allocentric = -1 * qt_allocentric
                pose_blob[count, 2:6] = qt_allocentric
                pose_blob[count, 6:] = T
                count += 1

        # construct the meta data
        """
        format of the meta_data
        intrinsic matrix: meta_data[0 ~ 8]
        inverse intrinsic matrix: meta_data[9 ~ 17]
        """
        K = np.matrix(meta_data['intrinsic_matrix']) * im_scale
        K[2, 2] = 1
        Kinv = np.linalg.pinv(K)
        meta_data_blob = np.zeros(18, dtype=np.float32)
        meta_data_blob[0:9] = K.flatten()
        meta_data_blob[9:18] = Kinv.flatten()

        # vertex regression target
        if cfg.TRAIN.VERTEX_REG:
            if roidb['flipped']:
                centers[:, 0] = width - centers[:, 0]
            vertex_targets, vertex_weights = self._generate_vertex_targets(im_label_all, meta_data['cls_indexes'], \
                centers, poses_all, classes, num_classes)
        else:
            vertex_targets = []
            vertex_weights = []

        return label_blob, mask, meta_data_blob, pose_blob, gt_boxes, vertex_targets, vertex_weights


    # compute the voting label image in 2D
    def _generate_vertex_targets(self, im_label, cls_indexes, center, poses, classes, num_classes):

        width = im_label.shape[1]
        height = im_label.shape[0]
        vertex_targets = np.zeros((3 * num_classes, height, width), dtype=np.float32)
        vertex_weights = np.zeros((3 * num_classes, height, width), dtype=np.float32)

        c = np.zeros((2, 1), dtype=np.float32)
        for i in range(1, num_classes):
            y, x = np.where(im_label == classes[i])
            I = np.where(im_label == classes[i])
            ind = np.where(cls_indexes == classes[i])[0]
            if len(x) > 0 and len(ind) > 0:
                c[0] = center[ind, 0]
                c[1] = center[ind, 1]
                z = poses[int(ind)][2]
                R = np.tile(c, (1, len(x))) - np.vstack((x, y))
                # compute the norm
                N = np.linalg.norm(R, axis=0) + 1e-10
                # normalization
                R = np.divide(R, np.tile(N, (2,1)))
                # assignment
                vertex_targets[3*i+0, y, x] = R[0,:]
                vertex_targets[3*i+1, y, x] = R[1,:]
                vertex_targets[3*i+2, y, x] = math.log(z)

                vertex_weights[3*i+0, y, x] = cfg.TRAIN.VERTEX_W_INSIDE
                vertex_weights[3*i+1, y, x] = cfg.TRAIN.VERTEX_W_INSIDE
                vertex_weights[3*i+2, y, x] = cfg.TRAIN.VERTEX_W_INSIDE

        return vertex_targets, vertex_weights


    def __len__(self):
        return self._size


    def _get_default_path(self):
        """
        Return the default path where ycb_self_supervision is expected to be installed.
        """
        return os.path.join(datasets.ROOT_DIR, 'data', 'YCB_Self_Supervision')


    def _load_image_set_index(self, image_set):
        """
        Load the indexes of images in the data folder
        """

        image_set_file = os.path.join(self._ycb_self_supervision_path, image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)

        subdirs = []
        with open(image_set_file) as f:
            for x in f.readlines():
                subdirs.append(x.rstrip('\n'))

        image_index = []
        for i in range(len(subdirs)):
            subdir = subdirs[i]
            folder = osp.join(self._data_path, subdir)
            filename = os.path.join(folder, '*.mat')
            files = glob.glob(filename)
            print(subdir, len(files))
            for k in range(len(files)):
                filename = files[k]
                head, name = os.path.split(filename)
                index = subdir + '/' + name[:-9]
                image_index.append(index)

        print('=======================================================')
        print('%d image in %s' % (len(image_index), self._data_path))
        print('=======================================================')
        return image_index


    def _load_object_points(self, classes, extents, symmetry):

        points = [[] for _ in range(len(classes))]
        num = np.inf
        num_classes = len(classes)
        for i in range(1, num_classes):
            point_file = os.path.join(self._model_path, classes[i], 'points.xyz')
            print(point_file)
            assert os.path.exists(point_file), 'Path does not exist: {}'.format(point_file)
            points[i] = np.loadtxt(point_file)
            if points[i].shape[0] < num:
                num = points[i].shape[0]

        points_all = np.zeros((num_classes, num, 3), dtype=np.float32)
        for i in range(1, num_classes):
            points_all[i, :, :] = points[i][:num, :]

        # rescale the points
        point_blob = points_all.copy()
        for i in range(1, num_classes):
            # compute the rescaling factor for the points
            weight = 10.0 / np.amax(extents[i, :])
            if weight < 10:
                weight = 10
            if symmetry[i] > 0:
                point_blob[i, :, :] = 4 * weight * point_blob[i, :, :]
            else:
                point_blob[i, :, :] = weight * point_blob[i, :, :]

        return points, points_all, point_blob


    def _load_object_extents(self):

        extents = np.zeros((self._num_classes_all, 3), dtype=np.float32)
        for i in range(1, self._num_classes_all):
            point_file = os.path.join(self._model_path, self._classes_all[i], 'points.xyz')
            print(point_file)
            assert os.path.exists(point_file), 'Path does not exist: {}'.format(point_file)
            points = np.loadtxt(point_file)
            extents[i, :] = 2 * np.max(np.absolute(points), axis=0)

        return extents


    # image
    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self.image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """

        image_path_jpg = os.path.join(self._data_path, index + '_color.jpg')
        image_path_png = os.path.join(self._data_path, index + '_color.png')
        if os.path.exists(image_path_jpg):
            return image_path_jpg
        elif os.path.exists(image_path_png):
            return image_path_png

        assert os.path.exists(image_path_jpg) or os.path.exists(image_path_png), \
                'Path does not exist: {} or {}'.format(image_path_jpg, image_path_png)

    # depth
    def depth_path_at(self, i):
        """
        Return the absolute path to depth i in the image sequence.
        """
        return self.depth_path_from_index(self.image_index[i])

    def depth_path_from_index(self, index):
        """
        Construct an depth path from the image's "index" identifier.
        """
        depth_path = os.path.join(self._data_path, index + '_depth' + self._image_ext)
        assert os.path.exists(depth_path), \
                'Path does not exist: {}'.format(depth_path)
        return depth_path

    # camera pose
    def metadata_path_at(self, i):
        """
        Return the absolute path to metadata i in the image sequence.
        """
        return self.metadata_path_from_index(self.image_index[i])

    def metadata_path_from_index(self, index):
        """
        Construct an metadata path from the image's "index" identifier.
        """
        metadata_path = os.path.join(self._data_path, index + '_meta.mat')
        assert os.path.exists(metadata_path), \
                'Path does not exist: {}'.format(metadata_path)
        return metadata_path

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """

        gt_roidb = [self._load_ycb_self_supervision_annotation(index)
                    for index in self._image_index]

        return gt_roidb


    def _load_ycb_self_supervision_annotation(self, index):
        """
        Load class name and meta data
        """
        # image path
        image_path = self.image_path_from_index(index)

        # depth path
        depth_path = self.depth_path_from_index(index)

        # metadata path
        metadata_path = self.metadata_path_from_index(index)
        
        return {'image': image_path,
                'depth': depth_path,
                'meta_data': metadata_path,
                'flipped': False}


    def labels_to_image(self, labels):

        height = labels.shape[0]
        width = labels.shape[1]
        im_label = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(self.num_classes):
            I = np.where(labels == i)
            im_label[I[0], I[1], :] = self._class_colors[i]

        return im_label


    def process_label_image(self, label_image):
        """
        change label image to label index
        """
        height = label_image.shape[0]
        width = label_image.shape[1]
        labels = np.zeros((height, width), dtype=np.int32)
        labels_all = np.zeros((height, width), dtype=np.int32)

        # label image is in BGR order
        index = label_image[:,:,2] + 256*label_image[:,:,1] + 256*256*label_image[:,:,0]
        for i in range(1, len(self._class_colors_all)):
            color = self._class_colors_all[i]
            ind = color[0] + 256*color[1] + 256*256*color[2]
            I = np.where(index == ind)
            labels_all[I[0], I[1]] = i

            ind = np.where(np.array(cfg.TRAIN.CLASSES) == i)[0]
            if len(ind) > 0:
                labels[I[0], I[1]] = ind

        return labels, labels_all


    def render_gt_pose(self, meta_data):

        width = self._width
        height = self._height
        meta_data['cls_indexes'] = meta_data['cls_indexes'].flatten()
        classes = np.array(cfg.TRAIN.CLASSES)
        classes_test = np.array(cfg.TEST.CLASSES).flatten()

        intrinsic_matrix = np.matrix(meta_data['intrinsic_matrix'])
        fx = intrinsic_matrix[0, 0]
        fy = intrinsic_matrix[1, 1]
        px = intrinsic_matrix[0, 2]
        py = intrinsic_matrix[1, 2]
        zfar = 6.0
        znear = 0.01

        # poses
        poses = meta_data['poses']
        if len(poses.shape) == 2:
            poses = np.reshape(poses, (3, 4, 1))
        num = poses.shape[2]

        # render poses to get the label image
        cls_indexes = []
        poses_all = []
        qt = np.zeros((7, ), dtype=np.float32)
        for i in range(num):
            RT = poses[:, :, i]
            qt[:3] = RT[:, 3]
            qt[3:] = mat2quat(RT[:, :3])
            if cfg.MODE == 'TEST':
                index = np.where(classes_test == meta_data['cls_indexes'][i])[0]
                cls_indexes.append(index[0])
            else:
                cls_indexes.append(meta_data['cls_indexes'][i] - 1)
            poses_all.append(qt.copy())
            
        # rendering
        cfg.renderer.set_poses(poses_all)
        cfg.renderer.set_light_pos([0, 0, 0])
        cfg.renderer.set_light_color([1, 1, 1])
        cfg.renderer.set_projection_matrix(width, height, fx, fy, px, py, znear, zfar)
        image_tensor = torch.cuda.FloatTensor(height, width, 4).detach()
        seg_tensor = torch.cuda.FloatTensor(height, width, 4).detach()
        cfg.renderer.render(cls_indexes, image_tensor, seg_tensor)
        image_tensor = image_tensor.flip(0)
        seg_tensor = seg_tensor.flip(0)

        # semantic labels
        im_label = seg_tensor.cpu().numpy()
        im_label = im_label[:, :, (2, 1, 0)] * 255
        im_label = np.round(im_label).astype(np.uint8)
        im_label = np.clip(im_label, 0, 255)
        im_label, im_label_all = self.process_label_image(im_label)

        # foreground mask
        seg = torch.from_numpy((im_label != 0).astype(np.float32))
        mask = seg.unsqueeze(0).repeat((3, 1, 1)).float().cuda()

        # gt boxes
        gt_boxes = np.zeros((self._num_classes, 5), dtype=np.float32)
        count = 0
        selected = []
        for i in range(num):
            cls = int(meta_data['cls_indexes'][i])
            ind = np.where(classes == cls)[0]
            if len(ind) > 0:
                R = poses[:, :3, i]
                T = poses[:, 3, i]

                # compute box
                x3d = np.ones((4, self._points_all.shape[1]), dtype=np.float32)
                x3d[0, :] = self._points_all[ind,:,0]
                x3d[1, :] = self._points_all[ind,:,1]
                x3d[2, :] = self._points_all[ind,:,2]
                RT = np.zeros((3, 4), dtype=np.float32)
                RT[:3, :3] = R
                RT[:, 3] = T
                x2d = np.matmul(meta_data['intrinsic_matrix'], np.matmul(RT, x3d))
                x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
                x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])

                x1 = np.min(x2d[0, :])
                y1 = np.min(x2d[1, :])
                x2 = np.max(x2d[0, :])
                y2 = np.max(x2d[1, :])
                if x1 > width or y1 > height or x2 < 0 or y2 < 0:
                    continue
                gt_boxes[count, 0] = x1
                gt_boxes[count, 1] = y1
                gt_boxes[count, 2] = x2
                gt_boxes[count, 3] = y2
                selected.append(i)
                count += 1

        meta_data['cls_indexes'] = meta_data['cls_indexes'][selected]
        meta_data['poses'] = poses[:, :, selected]
        meta_data['im_label'] = im_label
        meta_data['box'] = gt_boxes[:count, :4]
        return meta_data


    def evaluation(self, output_dir):

        filename = os.path.join(output_dir, 'results_posecnn.mat')
        if os.path.exists(filename):
            results_all = scipy.io.loadmat(filename)
            print('load results from file')
            print(filename)
            distances_sys = results_all['distances_sys']
            distances_non = results_all['distances_non']
            errors_rotation = results_all['errors_rotation']
            errors_translation = results_all['errors_translation']
            results_frame_id = results_all['results_frame_id'].flatten()
            results_object_id = results_all['results_object_id'].flatten()
            results_cls_id = results_all['results_cls_id'].flatten()
            segmentation_precision = results_all['segmentation_precision']
            segmentation_recall = results_all['segmentation_recall']
            segmentation_f1 = results_all['segmentation_f1']
            segmentation_count = results_all['segmentation_count']
        else:
            # save results
            num_max = 100000
            num_results = 2
            distances_sys = np.zeros((num_max, num_results), dtype=np.float32)
            distances_non = np.zeros((num_max, num_results), dtype=np.float32)
            errors_rotation = np.zeros((num_max, num_results), dtype=np.float32)
            errors_translation = np.zeros((num_max, num_results), dtype=np.float32)
            results_frame_id = np.zeros((num_max, ), dtype=np.float32)
            results_object_id = np.zeros((num_max, ), dtype=np.float32)
            results_cls_id = np.zeros((num_max, ), dtype=np.float32)
            segmentation_precision = np.zeros((num_max, self._num_classes), dtype=np.float32)
            segmentation_recall = np.zeros((num_max, self._num_classes), dtype=np.float32)
            segmentation_f1 = np.zeros((num_max, self._num_classes), dtype=np.float32)
            segmentation_count = np.zeros((num_max, self._num_classes), dtype=np.float32)

            # for each image
            count = -1
            count_file = -1
            filename = os.path.join(output_dir, '*.mat')
            files = glob.glob(filename)
            for i in range(len(files)):

                # load result
                filename = files[i]
                print(filename)
                result_posecnn = scipy.io.loadmat(filename)

                # load gt poses
                filename = result_posecnn['meta_data_path'][0]
                print(filename)
                gt = scipy.io.loadmat(filename)

                # render gt poses
                gt = self.render_gt_pose(gt)

                # compute segmentation metrics
                metrics_dict = multilabel_metrics(result_posecnn['labels'].astype(np.int32), gt['im_label'].astype(np.int32), self._num_classes)
                count_file += 1
                segmentation_precision[count_file, :] = metrics_dict['Precision']
                segmentation_recall[count_file, :] = metrics_dict['Recall']
                segmentation_f1[count_file, :] = metrics_dict['F-measure']
                segmentation_count[count_file, :] = metrics_dict['Count']

                '''
                import matplotlib.pyplot as plt
                fig = plt.figure()
                im_file = filename.replace('_meta.mat', '_color.png')
                im = cv2.imread(im_file)
                ax = fig.add_subplot(2, 2, 1)
                plt.imshow(im[:, :, (2, 1, 0)])
                for i in range(gt['box'].shape[0]):
                    x1 = gt['box'][i, 0]
                    y1 = gt['box'][i, 1]
                    x2 = gt['box'][i, 2]
                    y2 = gt['box'][i, 3]
                    plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='g', linewidth=3))
                ax = fig.add_subplot(2, 2, 2)
                plt.imshow(gt['im_label'])
                ax = fig.add_subplot(2, 2, 3)
                plt.imshow(result_posecnn['labels'].astype(np.int32))
                plt.show()
                #'''

                # for each gt poses
                cls_indexes = gt['cls_indexes'].flatten()
                for j in range(len(cls_indexes)):
                    count += 1
                    cls_index = cls_indexes[j]
                    RT_gt = gt['poses'][:, :, j]

                    results_frame_id[count] = i
                    results_object_id[count] = j
                    results_cls_id[count] = cls_index

                    # network result
                    result = result_posecnn
                    roi_index = []
                    if len(result['rois']) > 0:     
                        for k in range(result['rois'].shape[0]):
                            ind = int(result['rois'][k, 1])
                            cls = cfg.TRAIN.CLASSES[ind]
                            if cls == cls_index:
                                roi_index.append(k)                   

                    # select the roi
                    if len(roi_index) > 1:
                        # overlaps: (rois x gt_boxes)
                        roi_blob = result['rois'][roi_index, :]
                        roi_blob = roi_blob[:, (0, 2, 3, 4, 5, 1)]
                        gt_box_blob = np.zeros((1, 5), dtype=np.float32)
                        gt_box_blob[0, 1:] = gt['box'][j, :]
                        overlaps = bbox_overlaps(
                            np.ascontiguousarray(roi_blob[:, :5], dtype=np.float),
                            np.ascontiguousarray(gt_box_blob, dtype=np.float)).flatten()
                        assignment = overlaps.argmax()
                        roi_index = [roi_index[assignment]]

                    if len(roi_index) > 0:
                        RT = np.zeros((3, 4), dtype=np.float32)
                        ind = int(result['rois'][roi_index, 1])
                        if ind == -1:
                            points = self._points_clamp
                        else:
                            points = self._points[ind]

                        # pose from network
                        RT[:3, :3] = quat2mat(result['poses'][roi_index, :4].flatten())
                        RT[:, 3] = result['poses'][roi_index, 4:]
                        distances_sys[count, 0] = adi(RT[:3, :3], RT[:, 3],  RT_gt[:3, :3], RT_gt[:, 3], points)
                        distances_non[count, 0] = add(RT[:3, :3], RT[:, 3],  RT_gt[:3, :3], RT_gt[:, 3], points)
                        errors_rotation[count, 0] = re(RT[:3, :3], RT_gt[:3, :3])
                        errors_translation[count, 0] = te(RT[:, 3], RT_gt[:, 3])

                        # pose after depth refinement
                        if cfg.TEST.POSE_REFINE:
                            RT[:3, :3] = quat2mat(result['poses_refined'][roi_index, :4].flatten())
                            RT[:, 3] = result['poses_refined'][roi_index, 4:]
                            distances_sys[count, 1] = adi(RT[:3, :3], RT[:, 3],  RT_gt[:3, :3], RT_gt[:, 3], points)
                            distances_non[count, 1] = add(RT[:3, :3], RT[:, 3],  RT_gt[:3, :3], RT_gt[:, 3], points)
                            errors_rotation[count, 1] = re(RT[:3, :3], RT_gt[:3, :3])
                            errors_translation[count, 1] = te(RT[:, 3], RT_gt[:, 3])
                        else:
                            distances_sys[count, 1] = np.inf
                            distances_non[count, 1] = np.inf
                            errors_rotation[count, 1] = np.inf
                            errors_translation[count, 1] = np.inf
                    else:
                        distances_sys[count, :] = np.inf
                        distances_non[count, :] = np.inf
                        errors_rotation[count, :] = np.inf
                        errors_translation[count, :] = np.inf

            distances_sys = distances_sys[:count+1, :]
            distances_non = distances_non[:count+1, :]
            errors_rotation = errors_rotation[:count+1, :]
            errors_translation = errors_translation[:count+1, :]
            results_frame_id = results_frame_id[:count+1]
            results_object_id = results_object_id[:count+1]
            results_cls_id = results_cls_id[:count+1]
            segmentation_precision = segmentation_precision[:count_file+1, :]
            segmentation_recall = segmentation_recall[:count_file+1, :]
            segmentation_f1 = segmentation_f1[:count_file+1, :]
            segmentation_count = segmentation_count[:count_file+1, :]

            results_all = {'distances_sys': distances_sys,
                       'distances_non': distances_non,
                       'errors_rotation': errors_rotation,
                       'errors_translation': errors_translation,
                       'results_frame_id': results_frame_id,
                       'results_object_id': results_object_id,
                       'results_cls_id': results_cls_id,
                       'segmentation_precision': segmentation_precision, 
                       'segmentation_recall': segmentation_recall,
                       'segmentation_f1': segmentation_f1,
                       'segmentation_count': segmentation_count}

            filename = os.path.join(output_dir, 'results_posecnn.mat')
            scipy.io.savemat(filename, results_all)

        # for each class
        import matplotlib.pyplot as plt
        max_distance = 0.1
        index_plot = [0, 1]
        color = ['r', 'b']
        leng = ['PoseCNN', 'refined']
        num = len(leng)
        ADD = np.zeros((self._num_classes_all, num), dtype=np.float32)
        ADDS = np.zeros((self._num_classes_all, num), dtype=np.float32)
        TS = np.zeros((self._num_classes_all, num), dtype=np.float32)
        classes = list(copy.copy(self._classes_all))
        classes[0] = 'all'
        for k in range(self._num_classes_all):
            fig = plt.figure()
            if k == 0:
                index = range(len(results_cls_id))
            else:
                index = np.where(results_cls_id == k)[0]

            if len(index) == 0:
                continue
            print('%s: %d objects' % (classes[k], len(index)))

            # distance symmetry
            ax = fig.add_subplot(2, 3, 1)
            lengs = []
            for i in index_plot:
                D = distances_sys[index, i]
                ind = np.where(D > max_distance)[0]
                D[ind] = np.inf
                d = np.sort(D)
                n = len(d)
                accuracy = np.cumsum(np.ones((n, ), np.float32)) / n
                plt.plot(d, accuracy, color[i], linewidth=2)
                ADDS[k, i] = VOCap(d, accuracy)
                lengs.append('%s (%.2f)' % (leng[i], ADDS[k, i] * 100))
                print('%s, %s: %d objects missed' % (classes[k], leng[i], np.sum(np.isinf(D))))

            ax.legend(lengs)
            plt.xlabel('Average distance threshold in meter (symmetry)')
            plt.ylabel('accuracy')
            ax.set_title(classes[k])

            # distance non-symmetry
            ax = fig.add_subplot(2, 3, 2)
            lengs = []
            for i in index_plot:
                D = distances_non[index, i]
                ind = np.where(D > max_distance)[0]
                D[ind] = np.inf
                d = np.sort(D)
                n = len(d)
                accuracy = np.cumsum(np.ones((n, ), np.float32)) / n
                plt.plot(d, accuracy, color[i], linewidth=2)
                ADD[k, i] = VOCap(d, accuracy)
                lengs.append('%s (%.2f)' % (leng[i], ADD[k, i] * 100))
                print('%s, %s: %d objects missed' % (classes[k], leng[i], np.sum(np.isinf(D))))

            ax.legend(lengs)
            plt.xlabel('Average distance threshold in meter (non-symmetry)')
            plt.ylabel('accuracy')
            ax.set_title(classes[k])

            # translation
            ax = fig.add_subplot(2, 3, 3)
            lengs = []
            for i in index_plot:
                D = errors_translation[index, i]
                ind = np.where(D > max_distance)[0]
                D[ind] = np.inf
                d = np.sort(D)
                n = len(d)
                accuracy = np.cumsum(np.ones((n, ), np.float32)) / n
                plt.plot(d, accuracy, color[i], linewidth=2)
                TS[k, i] = VOCap(d, accuracy)
                lengs.append('%s (%.2f)' % (leng[i], TS[k, i] * 100))
                print('%s, %s: %d objects missed' % (classes[k], leng[i], np.sum(np.isinf(D))))

            ax.legend(lengs)
            plt.xlabel('Translation threshold in meter')
            plt.ylabel('accuracy')
            ax.set_title(classes[k])

            # rotation histogram
            count = 4
            for i in index_plot:
                ax = fig.add_subplot(2, 3, count)
                D = errors_rotation[index, i]
                ind = np.where(np.isfinite(D))[0]
                D = D[ind]
                ax.hist(D, bins=range(0, 190, 10), range=(0, 180))
                plt.xlabel('Rotation angle error')
                plt.ylabel('count')
                ax.set_title(leng[i])
                count += 1

            # mng = plt.get_current_fig_manager()
            # mng.full_screen_toggle()
            filename = output_dir + '/' + classes[k] + '.png'
            plt.savefig(filename)
            # plt.show()

        # print ADD
        print('==================ADD======================')
        for k in range(len(classes)):
            print('%s: %f' % (classes[k], ADD[k, 0]))
        for k in range(len(classes)-1):
            print('%f' % (ADD[k+1, 0]))
        print('%f' % (ADD[0, 0]))
        print(cfg.TRAIN.SNAPSHOT_INFIX)
        print('===========================================')

        # print ADD-S
        print('==================ADD-S====================')
        for k in range(len(classes)):
            print('%s: %f' % (classes[k], ADDS[k, 0]))
        for k in range(len(classes)-1):
            print('%f' % (ADDS[k+1, 0]))
        print('%f' % (ADDS[0, 0]))
        print(cfg.TRAIN.SNAPSHOT_INFIX)
        print('===========================================')

        # print ADD
        print('==================ADD refined======================')
        for k in range(len(classes)):
            print('%s: %f' % (classes[k], ADD[k, 1]))
        for k in range(len(classes)-1):
            print('%f' % (ADD[k+1, 1]))
        print('%f' % (ADD[0, 1]))
        print(cfg.TRAIN.SNAPSHOT_INFIX)
        print('===========================================')

        # print ADD-S
        print('==================ADD-S refined====================')
        for k in range(len(classes)):
            print('%s: %f' % (classes[k], ADDS[k, 1]))
        for k in range(len(classes)-1):
            print('%f' % (ADDS[k+1, 1]))
        print('%f' % (ADDS[0, 1]))
        print(cfg.TRAIN.SNAPSHOT_INFIX)
        print('===========================================')

        # print segmentation precision
        print('==================segmentation precision====================')
        for i in range(self._num_classes):
            count = np.sum(segmentation_count[:, i])
            if count > 0:
                precision = np.sum(segmentation_precision[:, i]) / count
            else:
                precision = 0
            print('%s: %d objects, %f' % (self._classes[i], count, precision))
        for i in range(self._num_classes):
            count = np.sum(segmentation_count[:, i])
            if count > 0:
                precision = np.sum(segmentation_precision[:, i]) / count
            else:
                precision = 0
            print('%f' % (precision))
        print('===========================================')

        # print segmentation recall
        print('==================segmentation recall====================')
        for i in range(self._num_classes):
            count = np.sum(segmentation_count[:, i])
            if count > 0:
                recall = np.sum(segmentation_recall[:, i]) / count
            else:
                recall = 0
            print('%s: %d objects, %f' % (self._classes[i], count, recall))
        for i in range(self._num_classes):
            count = np.sum(segmentation_count[:, i])
            if count > 0:
                recall = np.sum(segmentation_recall[:, i]) / count
            else:
                recall = 0
            print('%f' % (recall))
        print('===========================================')

        # print segmentation f1
        print('==================segmentation f1====================')
        for i in range(self._num_classes):
            count = np.sum(segmentation_count[:, i])
            if count > 0:
                f1 = np.sum(segmentation_f1[:, i]) / count
            else:
                f1 = 0
            print('%s: %d objects, %f' % (self._classes[i], count, f1))
        for i in range(self._num_classes):
            count = np.sum(segmentation_count[:, i])
            if count > 0:
                f1 = np.sum(segmentation_f1[:, i]) / count
            else:
                f1 = 0
            print('%f' % (f1))
        print('===========================================')
