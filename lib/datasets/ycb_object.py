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
try:
    import cPickle  # Use cPickle on Python 2.7
except ImportError:
    import pickle as cPickle
import scipy.io
import glob

import datasets
from fcn.config import cfg
from utils.blob import pad_im, chromatic_transform, add_noise, add_noise_cuda, add_noise_depth_cuda
from transforms3d.quaternions import mat2quat, quat2mat
from transforms3d.euler import euler2quat
from utils.se3 import *
from scipy.optimize import minimize
import matplotlib.pyplot as plt


class YCBObject(data.Dataset, datasets.imdb):
    def __init__(self, image_set, ycb_object_path = None):

        self._name = 'ycb_object_' + image_set
        self._image_set = image_set
        self._ycb_object_path = self._get_default_path() if ycb_object_path is None \
                            else ycb_object_path
        self._data_path = os.path.join(self._ycb_object_path, 'data')
        self._model_path = os.path.join(datasets.ROOT_DIR, 'data', 'models')
        self.root_path = self._ycb_object_path

        # define all the classes
        self._classes_all = ('__background__', '002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle', \
                         '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana', '019_pitcher_base', \
                         '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '036_wood_block', '037_scissors', '040_large_marker', \
                         '051_large_clamp', '052_extra_large_clamp', '061_foam_brick', 'holiday_cup1', 'holiday_cup2', 'sanning_mug', \
                         '001_chips_can', 'block_red_big', 'block_green_big', 'block_blue_big', 'block_yellow_big', \
                         'block_red_small', 'block_green_small', 'block_blue_small', 'block_yellow_small', \
                         'block_red_median', 'block_green_median', 'block_blue_median', 'block_yellow_median',
                         'fusion_duplo_dude', 'cabinet_handle')
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
        self._intrinsic_matrix = np.array([[524.7917885754071, 0, 332.5213232846151],
                                          [0, 489.3563960810721, 281.2339855172282],
                                          [0, 0, 1]])

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

        # train classes
        self._points, self._points_all, self._point_blob = \
            self._load_object_points(self._classes, self._extents, self._symmetry)

        # test classes
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
                continue

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
            filename = '{}/{}/textured_simple.obj'.format(self._model_path, cls)
            if osp.exists(filename):
                self.model_mesh_paths_target.append(filename)
                continue
            filename = '{}/{}/textured_simple.ply'.format(self._model_path, cls)
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
        self._size = cfg.TRAIN.SYNNUM
        self._build_uniform_poses()

        # sample indexes real for ycb object
        num_poses = 600
        num_classes = len(self._classes_all) - 1 # no background
        self.pose_indexes_real = np.zeros((num_classes, ), dtype=np.int32)
        self.pose_lists_real = []
        self.pose_images = []
        for i in range(num_classes):
            self.pose_lists_real.append(np.random.permutation(np.arange(num_poses)))
            dirname = osp.join(self._data_path, self._classes_all[i+1], '*.jpg')
            files = glob.glob(dirname)
            self.pose_images.append(files)

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
                extent = 2 * np.mean(self._extents_all[cls+1, :])

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
        for i in range(num_target):
            cls = int(indexes_target[i])
            pose_blob[i, 0] = 1
            pose_blob[i, 1] = cls
            T = poses_all[i][:3]
            qt = poses_all[i][3:]

            # egocentric to allocentric
            qt_allocentric = egocentric2allocentric(qt, T)
            if qt_allocentric[0] < 0:
                qt_allocentric = -1 * qt_allocentric
            pose_blob[i, 2:6] = qt_allocentric
            pose_blob[i, 6:] = T

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
        
            gt_boxes[i, 0] = np.min(x2d[0, :])
            gt_boxes[i, 1] = np.min(x2d[1, :])
            gt_boxes[i, 2] = np.max(x2d[0, :])
            gt_boxes[i, 3] = np.max(x2d[1, :])
            gt_boxes[i, 4] = cls


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
                  'image_depth': im_cuda_depth,
                  'im_depth': im_depth_return,
                  'label': label_blob,
                  'mask': mask,
                  'mask_depth': mask_depth_cuda,
                  'meta_data': meta_data_blob,
                  'poses': pose_blob,
                  'extents': self._extents,
                  'points': self._point_blob,
                  'symmetry': self._symmetry,
                  'gt_boxes': gt_boxes,
                  'im_info': im_info}

        if cfg.TRAIN.VERTEX_REG or cfg.TRAIN.VERTEX_REG_DELTA:
            sample['vertex_targets'] = vertex_targets
            sample['vertex_weights'] = vertex_weights

        return sample


    def __getitem__(self, index):
        return self._render_item()


    def __len__(self):
        return self._size


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


    def _generate_vertex_deltas(self, im_label, cls_indexes, center, poses, classes, num_classes, im_depth):

        x_image = im_depth[:, :, 0]
        y_image = im_depth[:, :, 1]
        z_image = im_depth[:, :, 2]

        width = im_label.shape[1]
        height = im_label.shape[0]
        vertex_targets = np.zeros((3 * num_classes, height, width), dtype=np.float32)
        vertex_weights = np.zeros((3 * num_classes, height, width), dtype=np.float32)

        c = np.zeros((2, 1), dtype=np.float32)
        for i in range(1, num_classes):

            valid_mask = (z_image != 0.0)
            label_mask = (im_label == classes[i])
            fin_mask = valid_mask * label_mask

            y, x = np.where(fin_mask)
            ind = np.where(cls_indexes == classes[i])[0]
            if len(x) > 0 and len(ind) > 0:

                extents_here = self._extents[i, :]
                largest_dim = np.sqrt(np.sum(extents_here * extents_here))
                half_diameter = largest_dim / 2.0

                c[0] = center[ind, 0]
                c[1] = center[ind, 1]

                if isinstance(poses, list):
                    x_center_coord = poses[int(ind)][0]
                    y_center_coord = poses[int(ind)][1]
                    z_center_coord = poses[int(ind)][2]
                else:
                    if len(poses.shape) == 3:
                        x_center_coord = poses[0, 3, ind]
                        y_center_coord = poses[1, 3, ind]
                        z_center_coord = poses[2, 3, ind]
                    else:
                        x_center_coord = poses[ind, -3]
                        y_center_coord = poses[ind, -2]
                        z_center_coord = poses[ind, -1]

                targets_x = (x_image[y, x] - x_center_coord) / half_diameter
                targets_y = (y_image[y, x] - y_center_coord) / half_diameter
                targets_z = (z_image[y, x] - z_center_coord) / half_diameter

                vertex_targets[3 * i + 0, y, x] = targets_x
                vertex_targets[3 * i + 1, y, x] = targets_y
                vertex_targets[3 * i + 2, y, x] = targets_z

                vertex_weights[3 * i + 0, y, x] = cfg.TRAIN.VERTEX_W_INSIDE
                vertex_weights[3 * i + 1, y, x] = cfg.TRAIN.VERTEX_W_INSIDE
                vertex_weights[3 * i + 2, y, x] = cfg.TRAIN.VERTEX_W_INSIDE
        
        return vertex_targets, vertex_weights


    def _get_default_path(self):
        """
        Return the default path where ycb_object is expected to be installed.
        """
        return os.path.join(datasets.ROOT_DIR, 'data', 'YCB_Object')


    def _load_object_extents(self):

        extents = np.zeros((self._num_classes_all, 3), dtype=np.float32)
        for i in range(1, self._num_classes_all):
            point_file = os.path.join(self._model_path, self._classes_all[i], 'points.xyz')
            print(point_file)
            assert os.path.exists(point_file), 'Path does not exist: {}'.format(point_file)
            points = np.loadtxt(point_file)
            extents[i, :] = 2 * np.max(np.absolute(points), axis=0)

        return extents


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
