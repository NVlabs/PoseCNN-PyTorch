import os
import sys
import yaml
import numpy as np
import torch
import torch.utils.data as data
import numpy as np
import numpy.random as npr
import cv2

import datasets
from fcn.config import cfg
from utils.blob import pad_im, chromatic_transform, add_noise
from transforms3d.quaternions import mat2quat, quat2mat
from utils.se3 import *
from utils.pose_error import *
from utils.cython_bbox import bbox_overlaps

_SUBJECTS = [
    '20200709-weiy',
    '20200813-ceppner',
    '20200820-amousavian',
    '20200903-ynarang',
    '20200908-yux',
    '20200918-ftozetoramos',
]

_SERIALS = [
    '836212060125',
    '839512060362',
    '840412060917',
    '841412060263',
    '932122060857',
    '932122060861',
    '932122061900',
    '932122062010',
]

_YCB_CLASSES = [
    '__background__',
    '002_master_chef_can',
    '003_cracker_box',
    '004_sugar_box',
    '005_tomato_soup_can',
    '006_mustard_bottle',
    '007_tuna_fish_can',
    '008_pudding_box',
    '009_gelatin_box',
    '010_potted_meat_can',
    '011_banana',
    '019_pitcher_base',
    '021_bleach_cleanser',
    '024_bowl',
    '025_mug',
    '035_power_drill',
    '036_wood_block',
    '037_scissors',
    '040_large_marker',
    '051_large_clamp',
    '052_extra_large_clamp',
    '061_foam_brick',
]


class DexYCBDataset(data.Dataset, datasets.imdb):

    def __init__(self, setup, split):
        self._setup = setup
        self._split = split
        self._color_format = "color_{:06d}.jpg"
        self._depth_format = "aligned_depth_to_color_{:06d}.png"
        self._label_format = "labels_{:06d}.npz"
        self._height = 480
        self._width = 640

        # paths
        self._name = 'dex_ycb_' + setup + '_' + split
        self._image_set = split
        self._dex_ycb_path = self._get_default_path()
        path = os.path.join(self._dex_ycb_path, 'data')
        self._data_dir = path
        self._calib_dir = os.path.join(self._data_dir, "calibration")
        self._model_dir = os.path.join(self._data_dir, "models")
        self._obj_file = [
            os.path.join(self._model_dir, x, "textured_simple.obj")
            for x in _YCB_CLASSES
        ]

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
        self.model_mesh_paths = ['{}/{}/textured_simple.obj'.format(self._model_dir, cls) for cls in self._classes_all[1:]]
        self.model_sdf_paths = ['{}/{}/textured_simple_low_res.pth'.format(self._model_dir, cls) for cls in self._classes_all[1:]]
        self.model_texture_paths = ['{}/{}/texture_map.png'.format(self._model_dir, cls) for cls in self._classes_all[1:]]
        self.model_colors = [np.array(self._class_colors_all[i]) / 255.0 for i in range(1, len(self._classes_all))]

        self.model_mesh_paths_target = ['{}/{}/textured_simple.obj'.format(self._model_dir, cls) for cls in self._classes[1:]]
        self.model_sdf_paths_target = ['{}/{}/textured_simple.sdf'.format(self._model_dir, cls) for cls in self._classes[1:]]
        self.model_texture_paths_target = ['{}/{}/texture_map.png'.format(self._model_dir, cls) for cls in self._classes[1:]]
        self.model_colors_target = [np.array(self._class_colors_all[i]) / 255.0 for i in cfg.TRAIN.CLASSES[1:]]

        # Seen subjects, camera views, grasped objects.
        if self._setup == 's0':
            if self._split == 'train':
                subject_ind = [0, 1, 2, 3, 4, 5]
                serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
                sequence_ind = [i for i in range(100) if i % 5 != 4]
            if self._split == 'val':
                subject_ind = [5]
                serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
                sequence_ind = [i for i in range(100) if i % 5 == 4]
            if self._split == 'test':
                subject_ind = [0, 1, 2, 3, 4]
                serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
                sequence_ind = [i for i in range(100) if i % 5 == 4]

        # Unseen subjects.
        if self._setup == 's1':
            if self._split == 'train':
                subject_ind = [0, 1, 2, 3, 4]
                serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
                sequence_ind = list(range(100))
            if self._split == 'val':
                subject_ind = [5]
                serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
                sequence_ind = list(range(100))
            if self._split == 'test':
                raise NotImplementedError

        # Unseen camera views.
        if self._setup == 's2':
            if self._split == 'train':
                subject_ind = [0, 1, 2, 3, 4, 5]
                serial_ind = [0, 1, 2, 3, 4, 5]
                sequence_ind = list(range(100))
            if self._split == 'val':
                subject_ind = [0, 1, 2, 3, 4, 5]
                serial_ind = [6]
                sequence_ind = list(range(100))
            if self._split == 'test':
                subject_ind = [0, 1, 2, 3, 4, 5]
                serial_ind = [7]
                sequence_ind = list(range(100))

        # Unseen grasped objects.
        if self._setup == 's3':
            if self._split == 'train':
                subject_ind = [0, 1, 2, 3, 4, 5]
                serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
                sequence_ind = [i for i in range(100) if i // 5 not in (3, 7, 11, 15, 19)]
            if self._split == 'val':
                subject_ind = [0, 1, 2, 3, 4, 5]
                serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
                sequence_ind = [i for i in range(100) if i // 5 in (3, 19)]
            if self._split == 'test':
                subject_ind = [0, 1, 2, 3, 4, 5]
                serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
                sequence_ind = [i for i in range(100) if i // 5 in (7, 11, 15)]

        self._subjects = [_SUBJECTS[i] for i in subject_ind]
        self._serials = [_SERIALS[i] for i in serial_ind]
        self._intrinsics = []
        for s in self._serials:
            intr_file = os.path.join(self._calib_dir, "intrinsics", "{}_{}x{}.yml".format(s, self._width, self._height))
            with open(intr_file, 'r') as f:
                intr = yaml.load(f, Loader=yaml.FullLoader)
            intr = intr['color']
            self._intrinsics.append(intr)

        # build mapping
        self._sequences = []
        self._mapping = []
        self._ycb_ids = []
        offset = 0
        for n in self._subjects:
            seq = sorted(os.listdir(os.path.join(self._data_dir, n)))
            seq = [os.path.join(n, s) for s in seq]
            assert len(seq) == 100
            seq = [seq[i] for i in sequence_ind]
            self._sequences += seq
            for i, q in enumerate(seq):
                meta_file = os.path.join(self._data_dir, q, "meta.yml")
                with open(meta_file, 'r') as f:
                    meta = yaml.load(f, Loader=yaml.FullLoader)
                c = np.arange(len(self._serials))
                f = np.arange(meta['num_frames'])
                f, c = np.meshgrid(f, c)
                c = c.ravel()
                f = f.ravel()
                s = (offset + i) * np.ones_like(c)
                m = np.vstack((s, c, f)).T
                self._mapping.append(m)
                self._ycb_ids.append(meta['ycb_ids'])
            offset += len(seq)
        self._mapping = np.vstack(self._mapping)

        # dataset size
        self._size = len(self._mapping)
        print('dataset %s with images %d' % (self._name, self._size))
        if self._size > cfg.TRAIN.MAX_ITERS_PER_EPOCH * cfg.TRAIN.IMS_PER_BATCH:
            self._size = cfg.TRAIN.MAX_ITERS_PER_EPOCH * cfg.TRAIN.IMS_PER_BATCH


    def __len__(self):
        return self._size


    def __getitem__(self, idx):
        s, c, f = self._mapping[idx]
        d = os.path.join(self._data_dir, self._sequences[s], self._serials[c])
        roidb = {
            'color_file': os.path.join(d, self._color_format.format(f)),
            'depth_file': os.path.join(d, self._depth_format.format(f)),
            'label_file': os.path.join(d, self._label_format.format(f)),
            'intrinsics': self._intrinsics[c],
            'ycb_ids': self._ycb_ids[s],
        }

        # Get the input image blob
        random_scale_ind = npr.randint(0, high=len(cfg.TRAIN.SCALES_BASE))
        im_blob, im_depth, im_scale, height, width = self._get_image_blob(roidb['color_file'], roidb['depth_file'], random_scale_ind)

        # build the label blob
        label_blob, mask, meta_data_blob, pose_blob, gt_boxes, vertex_targets, vertex_weights \
            = self._get_label_blob(roidb, self._num_classes, im_scale, height, width)

        is_syn = 0
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
                  'im_info': im_info}

        if cfg.TRAIN.VERTEX_REG:
            sample['vertex_targets'] = vertex_targets
            sample['vertex_weights'] = vertex_weights

        return sample


    def _get_image_blob(self, color_file, depth_file, scale_ind):    

        # rgba
        rgba = pad_im(cv2.imread(color_file, cv2.IMREAD_UNCHANGED), 16)
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

        # chromatic transform
        if cfg.TRAIN.CHROMATIC and cfg.MODE == 'TRAIN' and np.random.rand(1) > 0.1:
            im = chromatic_transform(im)
        if cfg.TRAIN.ADD_NOISE and cfg.MODE == 'TRAIN' and np.random.rand(1) > 0.1:
            im = add_noise(im)
        im_tensor = torch.from_numpy(im) / 255.0
        im_tensor -= self._pixel_mean
        image_blob = im_tensor.permute(2, 0, 1).float()

        # depth image
        im_depth = pad_im(cv2.imread(depth_file, cv2.IMREAD_UNCHANGED), 16)
        if im_scale != 1.0:
            im_depth = cv2.resize(im_depth, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_NEAREST)
        im_depth = im_depth.astype('float') / 1000.0

        return image_blob, im_depth, im_scale, height, width


    def _get_label_blob(self, roidb, num_classes, im_scale, height, width):
        """ build the label blob """

        # parse data
        cls_indexes = roidb['ycb_ids']
        classes = np.array(cfg.TRAIN.CLASSES)
        fx = roidb['intrinsics']['fx']
        fy = roidb['intrinsics']['fy']
        px = roidb['intrinsics']['ppx']
        py = roidb['intrinsics']['ppy']
        intrinsic_matrix = np.eye(3, dtype=np.float32)
        intrinsic_matrix[0, 0] = fx
        intrinsic_matrix[1, 1] = fy
        intrinsic_matrix[0, 2] = px
        intrinsic_matrix[1, 2] = py
        label = np.load(roidb['label_file'])

        # read label image
        im_label = label['seg']
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
        poses = label['pose_y']
        if len(poses.shape) == 2:
            poses = np.reshape(poses, (1, 3, 4))
        num = poses.shape[0]
        assert num == len(cls_indexes), 'number of poses not equal to number of objects'

        pose_blob = np.zeros((num_classes, 9), dtype=np.float32)
        gt_boxes = np.zeros((num_classes, 5), dtype=np.float32)
        center = np.zeros((num, 2), dtype=np.float32)
        count = 0
        for i in range(num):
            cls = int(cls_indexes[i])
            ind = np.where(classes == cls)[0]
            if len(ind) > 0:
                R = poses[i, :, :3]
                T = poses[i, :, 3]
                pose_blob[count, 0] = 1
                pose_blob[count, 1] = ind
                qt = mat2quat(R)

                # egocentric to allocentric
                qt_allocentric = egocentric2allocentric(qt, T)
                if qt_allocentric[0] < 0:
                   qt_allocentric = -1 * qt_allocentric
                pose_blob[count, 2:6] = qt_allocentric
                pose_blob[count, 6:] = T

                # compute center
                center[i, 0] = fx * T[0] / T[2] + px
                center[i, 1] = fy * T[1] / T[2] + py

                # compute box
                x3d = np.ones((4, self._points_all.shape[1]), dtype=np.float32)
                x3d[0, :] = self._points_all[ind,:,0]
                x3d[1, :] = self._points_all[ind,:,1]
                x3d[2, :] = self._points_all[ind,:,2]
                RT = np.zeros((3, 4), dtype=np.float32)
                RT[:3, :3] = quat2mat(qt)
                RT[:, 3] = T
                x2d = np.matmul(intrinsic_matrix, np.matmul(RT, x3d))
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
        K = np.matrix(intrinsic_matrix) * im_scale
        K[2, 2] = 1
        Kinv = np.linalg.pinv(K)
        meta_data_blob = np.zeros(18, dtype=np.float32)
        meta_data_blob[0:9] = K.flatten()
        meta_data_blob[9:18] = Kinv.flatten()

        # vertex regression target
        if cfg.TRAIN.VERTEX_REG:
            vertex_targets, vertex_weights = self._generate_vertex_targets(im_label,
                cls_indexes, center, poses, classes, num_classes)
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
                        z = poses[ind, 2, 3]
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


    def _get_default_path(self):
        """
        Return the default path where YCB_Video is expected to be installed.
        """
        return os.path.join(datasets.ROOT_DIR, 'data', 'DEX_YCB')


    def _load_object_extents(self):
        extents = np.zeros((self._num_classes_all, 3), dtype=np.float32)
        for i in range(1, self._num_classes_all):
            point_file = os.path.join(self._model_dir, self._classes_all[i], 'points.xyz')
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
            point_file = os.path.join(self._model_dir, classes[i], 'points.xyz')
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
