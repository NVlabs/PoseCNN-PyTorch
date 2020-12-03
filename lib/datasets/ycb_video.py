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


class YCBVideo(data.Dataset, datasets.imdb):
    def __init__(self, image_set, ycb_video_path = None):

        self._name = 'ycb_video_' + image_set
        self._image_set = image_set
        self._ycb_video_path = self._get_default_path() if ycb_video_path is None \
                            else ycb_video_path

        path = os.path.join(self._ycb_video_path, 'data')
        if not os.path.exists(path):
            path = os.path.join(self._ycb_video_path, 'YCB_Video_Dataset/YCB_Video_Dataset/YCB_Video_Dataset/data')
        self._data_path = path

        self._model_path = os.path.join(datasets.ROOT_DIR, 'data', 'models')

        # define all the classes
        self._classes_all = ('__background__', '002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle', \
                         '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana', '019_pitcher_base', \
                         '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '036_wood_block', '037_scissors', '040_large_marker', \
                         '051_large_clamp', '052_extra_large_clamp', '061_foam_brick')
        self._num_classes_all = len(self._classes_all)
        self._class_colors_all = [(255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), \
                              (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128), \
                              (64, 0, 0), (0, 64, 0), (0, 0, 64), (64, 64, 0), (64, 0, 64), (0, 64, 64), 
                              (192, 0, 0), (0, 192, 0), (0, 0, 192)]
        self._symmetry_all = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]).astype(np.float32)
        self._extents_all = self._load_object_extents()

        self._width = 640
        self._height = 480
        self._intrinsic_matrix = np.array([[1.066778e+03, 0.000000e+00, 3.129869e+02],
                                          [0.000000e+00, 1.067487e+03, 2.413109e+02],
                                          [0.000000e+00, 0.000000e+00, 1.000000e+00]])

        # select a subset of classes
        self._classes = [self._classes_all[i] for i in cfg.TRAIN.CLASSES]
        self._classes_test = [self._classes_all[i] for i in cfg.TEST.CLASSES]
        self._num_classes = len(self._classes)
        self._class_colors = [self._class_colors_all[i] for i in cfg.TRAIN.CLASSES]
        self._symmetry = self._symmetry_all[cfg.TRAIN.CLASSES]
        self._symmetry_test = self._symmetry_all[cfg.TEST.CLASSES]
        self._extents = self._extents_all[cfg.TRAIN.CLASSES]
        self._extents_test = self._extents_all[cfg.TEST.CLASSES]
        self._pixel_mean = cfg.PIXEL_MEANS / 255.0

        # train classes
        self._points, self._points_all, self._point_blob = \
            self._load_object_points(self._classes, self._extents, self._symmetry)

        # test classes
        self._points_test, self._points_all_test, self._point_blob_test = \
            self._load_object_points(self._classes_test, self._extents_test, self._symmetry_test)

        # 3D model paths
        self.model_mesh_paths = ['{}/{}/textured_simple.obj'.format(self._model_path, cls) for cls in self._classes_all[1:]]
        self.model_sdf_paths = ['{}/{}/textured_simple_low_res.pth'.format(self._model_path, cls) for cls in self._classes_all[1:]]
        self.model_texture_paths = ['{}/{}/texture_map.png'.format(self._model_path, cls) for cls in self._classes_all[1:]]
        self.model_colors = [np.array(self._class_colors_all[i]) / 255.0 for i in range(1, len(self._classes_all))]

        self.model_mesh_paths_target = ['{}/{}/textured_simple.obj'.format(self._model_path, cls) for cls in self._classes[1:]]
        self.model_sdf_paths_target = ['{}/{}/textured_simple.sdf'.format(self._model_path, cls) for cls in self._classes[1:]]
        self.model_texture_paths_target = ['{}/{}/texture_map.png'.format(self._model_path, cls) for cls in self._classes[1:]]
        self.model_colors_target = [np.array(self._class_colors_all[i]) / 255.0 for i in cfg.TRAIN.CLASSES[1:]]

        self._class_to_ind = dict(zip(self._classes, range(self._num_classes)))
        self._image_index = self._load_image_set_index(image_set)

        self._size = len(self._image_index)
        if self._size > cfg.TRAIN.MAX_ITERS_PER_EPOCH * cfg.TRAIN.IMS_PER_BATCH:
            self._size = cfg.TRAIN.MAX_ITERS_PER_EPOCH * cfg.TRAIN.IMS_PER_BATCH
        self._roidb = self.gt_roidb()

        assert os.path.exists(self._ycb_video_path), \
                'ycb_video path does not exist: {}'.format(self._ycb_video_path)
        assert os.path.exists(self._data_path), \
                'Data path does not exist: {}'.format(self._data_path)


    def __getitem__(self, index):

        is_syn = 0
        roidb = self._roidb[index]

        # Get the input image blob
        random_scale_ind = npr.randint(0, high=len(cfg.TRAIN.SCALES_BASE))
        im_blob, im_depth, im_scale, height, width = self._get_image_blob(roidb, random_scale_ind)

        # build the label blob
        label_blob, mask, meta_data_blob, pose_blob, gt_boxes, vertex_targets, vertex_weights \
            = self._get_label_blob(roidb, self._num_classes, im_scale, height, width)

        is_syn = roidb['is_syn']
        im_info = np.array([im_blob.shape[1], im_blob.shape[2], im_scale, is_syn], dtype=np.float32)

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
                  'video_id': roidb['video_id'],
                  'image_id': roidb['image_id']}

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
        if cfg.TRAIN.ADD_NOISE and cfg.MODE == 'TRAIN' and np.random.rand(1) > 0.1:
            im = add_noise(im)
        im_tensor = torch.from_numpy(im) / 255.0
        im_tensor -= self._pixel_mean
        image_blob = im_tensor.permute(2, 0, 1).float()

        # depth image
        im_depth = pad_im(cv2.imread(roidb['depth'], cv2.IMREAD_UNCHANGED), 16)
        if im_scale != 1.0:
            im_depth = cv2.resize(im_depth, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_NEAREST)
        im_depth = im_depth.astype('float') / 10000.0

        return image_blob, im_depth, im_scale, height, width


    def _get_label_blob(self, roidb, num_classes, im_scale, height, width):
        """ build the label blob """

        meta_data = scipy.io.loadmat(roidb['meta_data'])
        meta_data['cls_indexes'] = meta_data['cls_indexes'].flatten()
        classes = np.array(cfg.TRAIN.CLASSES)

        # read label image
        im_label = pad_im(cv2.imread(roidb['label'], cv2.IMREAD_UNCHANGED), 16)
        if roidb['flipped']:
            if len(im_label.shape) == 2:
                im_label = im_label[:, ::-1]
            else:
                im_label = im_label[:, ::-1, :]
        if im_scale != 1.0:
            im_label = cv2.resize(im_label, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_NEAREST)

        label_blob = np.zeros((num_classes, height, width), dtype=np.float32)
        label_blob[0, :, :] = 1.0
        for i in range(1, num_classes):
            I = np.where(im_label == classes[i])
            if len(I[0]) > 0:
                label_blob[i, I[0], I[1]] = 1.0
                label_blob[0, I[0], I[1]] = 0.0

        # foreground mask
        seg = torch.from_numpy((im_label != 0).astype(np.float32))
        mask = seg.unsqueeze(0).repeat((3, 1, 1)).float()

        # poses
        poses = meta_data['poses']
        if len(poses.shape) == 2:
            poses = np.reshape(poses, (3, 4, 1))
        if roidb['flipped']:
            poses = _flip_poses(poses, meta_data['intrinsic_matrix'], width)

        num = poses.shape[2]
        pose_blob = np.zeros((num_classes, 9), dtype=np.float32)
        gt_boxes = np.zeros((num_classes, 5), dtype=np.float32)
        count = 0
        for i in range(num):
            cls = int(meta_data['cls_indexes'][i])
            ind = np.where(classes == cls)[0]
            if len(ind) > 0:
                R = poses[:, :3, i]
                T = poses[:, 3, i]
                pose_blob[count, 0] = 1
                pose_blob[count, 1] = ind
                qt = mat2quat(R)

                # egocentric to allocentric
                qt_allocentric = egocentric2allocentric(qt, T)
                if qt_allocentric[0] < 0:
                   qt_allocentric = -1 * qt_allocentric
                pose_blob[count, 2:6] = qt_allocentric
                pose_blob[count, 6:] = T

                # compute box
                x3d = np.ones((4, self._points_all.shape[1]), dtype=np.float32)
                x3d[0, :] = self._points_all[ind,:,0]
                x3d[1, :] = self._points_all[ind,:,1]
                x3d[2, :] = self._points_all[ind,:,2]
                RT = np.zeros((3, 4), dtype=np.float32)
                RT[:3, :3] = quat2mat(qt)
                RT[:, 3] = T
                x2d = np.matmul(meta_data['intrinsic_matrix'], np.matmul(RT, x3d))
                x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
                x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])
        
                gt_boxes[count, 0] = np.min(x2d[0, :]) * im_scale
                gt_boxes[count, 1] = np.min(x2d[1, :]) * im_scale
                gt_boxes[count, 2] = np.max(x2d[0, :]) * im_scale
                gt_boxes[count, 3] = np.max(x2d[1, :]) * im_scale
                gt_boxes[count, 4] = ind
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
            center = meta_data['center']
            if roidb['flipped']:
                center[:, 0] = width - center[:, 0]
            vertex_targets, vertex_weights = self._generate_vertex_targets(im_label,
                meta_data['cls_indexes'], center, poses, classes, num_classes)
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
                if isinstance(poses, list):
                    z = poses[int(ind)][2]
                else:
                    if len(poses.shape) == 3:
                        z = poses[2, 3, ind]
                    else:
                        z = poses[ind, -1]
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
        Return the default path where YCB_Video is expected to be installed.
        """
        return os.path.join(datasets.ROOT_DIR, 'data', 'YCB_Video')


    def _load_image_set_index(self, image_set):
        """
        Load the indexes listed in this dataset's image set file.
        """
        image_set_file = os.path.join(self._ycb_video_path, image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)

        image_index = []
        video_ids_selected = set([])
        video_ids_not = set([])
        count = np.zeros((self.num_classes, ), dtype=np.int32)

        with open(image_set_file) as f:
            for x in f.readlines():
                index = x.rstrip('\n')
                pos = index.find('/')
                video_id = index[:pos]

                if not video_id in video_ids_selected and not video_id in video_ids_not:
                    filename = os.path.join(self._data_path, video_id, '000001-meta.mat')
                    meta_data = scipy.io.loadmat(filename)
                    cls_indexes = meta_data['cls_indexes'].flatten()
                    flag = 0
                    for i in range(len(cls_indexes)):
                        cls_index = int(cls_indexes[i])
                        ind = np.where(np.array(cfg.TRAIN.CLASSES) == cls_index)[0]
                        if len(ind) > 0:
                            count[ind] += 1
                            flag = 1
                    if flag:
                        video_ids_selected.add(video_id)
                    else:
                        video_ids_not.add(video_id)

                if video_id in video_ids_selected:
                    image_index.append(index)

        for i in range(1, self.num_classes):
            print('%d %s [%d/%d]' % (i, self.classes[i], count[i], len(list(video_ids_selected))))

        # sample a subset for training
        if image_set == 'train':
            image_index = image_index[::5]

            # add synthetic data
            filename = os.path.join(self._data_path + '_syn', '*.mat')
            files = glob.glob(filename)
            print('adding synthetic %d data' % (len(files)))
            for i in range(len(files)):
                filename = files[i].replace(self._data_path, '../data')[:-9]
                image_index.append(filename)

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

        image_path = os.path.join(self._data_path, index + '-color.jpg')
        if not os.path.exists(image_path):
            image_path = os.path.join(self._data_path, index + '-color.png')

        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

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
        depth_path = os.path.join(self._data_path, index + '-depth.png')
        assert os.path.exists(depth_path), \
                'Path does not exist: {}'.format(depth_path)
        return depth_path

    # label
    def label_path_at(self, i):
        """
        Return the absolute path to metadata i in the image sequence.
        """
        return self.label_path_from_index(self.image_index[i])

    def label_path_from_index(self, index):
        """
        Construct an metadata path from the image's "index" identifier.
        """
        label_path = os.path.join(self._data_path, index + '-label.png')
        assert os.path.exists(label_path), \
                'Path does not exist: {}'.format(label_path)
        return label_path

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
        metadata_path = os.path.join(self._data_path, index + '-meta.mat')
        assert os.path.exists(metadata_path), \
                'Path does not exist: {}'.format(metadata_path)
        return metadata_path

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """

        prefix = '_class'
        for i in range(len(cfg.TRAIN.CLASSES)):
            prefix += '_%d' % cfg.TRAIN.CLASSES[i]
        cache_file = os.path.join(self.cache_path, self.name + prefix + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        print('loading gt...')
        gt_roidb = [self._load_ycb_video_annotation(index)
                    for index in self._image_index]

        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb


    def _load_ycb_video_annotation(self, index):
        """
        Load class name and meta data
        """
        # image path
        image_path = self.image_path_from_index(index)

        # depth path
        depth_path = self.depth_path_from_index(index)

        # label path
        label_path = self.label_path_from_index(index)

        # metadata path
        metadata_path = self.metadata_path_from_index(index)

        # is synthetic image or not
        if 'data_syn' in image_path:
            is_syn = 1
            video_id = ''
            image_id = ''
        else:
            is_syn = 0
            # parse image name
            pos = index.find('/')
            video_id = index[:pos]
            image_id = index[pos+1:]
        
        return {'image': image_path,
                'depth': depth_path,
                'label': label_path,
                'meta_data': metadata_path,
                'video_id': video_id,
                'image_id': image_id,
                'is_syn': is_syn,
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
            results_seq_id = results_all['results_seq_id'].flatten()
            results_frame_id = results_all['results_frame_id'].flatten()
            results_object_id = results_all['results_object_id'].flatten()
            results_cls_id = results_all['results_cls_id'].flatten()
        else:
            # save results
            num_max = 100000
            num_results = 2
            distances_sys = np.zeros((num_max, num_results), dtype=np.float32)
            distances_non = np.zeros((num_max, num_results), dtype=np.float32)
            errors_rotation = np.zeros((num_max, num_results), dtype=np.float32)
            errors_translation = np.zeros((num_max, num_results), dtype=np.float32)
            results_seq_id = np.zeros((num_max, ), dtype=np.float32)
            results_frame_id = np.zeros((num_max, ), dtype=np.float32)
            results_object_id = np.zeros((num_max, ), dtype=np.float32)
            results_cls_id = np.zeros((num_max, ), dtype=np.float32)

            # for each image
            count = -1
            for i in range(len(self._roidb)):
    
                # parse keyframe name
                seq_id = int(self._roidb[i]['video_id'])
                frame_id = int(self._roidb[i]['image_id'])

                # load result
                filename = os.path.join(output_dir, '%04d_%06d.mat' % (seq_id, frame_id))
                print(filename)
                result_posecnn = scipy.io.loadmat(filename)

                # load gt poses
                filename = osp.join(self._data_path, '%04d/%06d-meta.mat' % (seq_id, frame_id))
                print(filename)
                gt = scipy.io.loadmat(filename)

                # for each gt poses
                cls_indexes = gt['cls_indexes'].flatten()
                for j in range(len(cls_indexes)):
                    count += 1
                    cls_index = cls_indexes[j]
                    RT_gt = gt['poses'][:, :, j]

                    results_seq_id[count] = seq_id
                    results_frame_id[count] = frame_id
                    results_object_id[count] = j
                    results_cls_id[count] = cls_index

                    # network result
                    result = result_posecnn
                    roi_index = []
                    if len(result['rois']) > 0:     
                        for k in range(result['rois'].shape[0]):
                            ind = int(result['rois'][k, 1])
                            if ind == -1:
                                cls = 19
                            else:
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
            results_seq_id = results_seq_id[:count+1]
            results_frame_id = results_frame_id[:count+1]
            results_object_id = results_object_id[:count+1]
            results_cls_id = results_cls_id[:count+1]

            results_all = {'distances_sys': distances_sys,
                       'distances_non': distances_non,
                       'errors_rotation': errors_rotation,
                       'errors_translation': errors_translation,
                       'results_seq_id': results_seq_id,
                       'results_frame_id': results_frame_id,
                       'results_object_id': results_object_id,
                       'results_cls_id': results_cls_id }

            filename = os.path.join(output_dir, 'results_posecnn.mat')
            scipy.io.savemat(filename, results_all)

        # print the results
        # for each class
        import matplotlib.pyplot as plt
        max_distance = 0.1
        index_plot = [0, 1]
        color = ['r', 'b']
        leng = ['PoseCNN', 'PoseCNN refined']
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
