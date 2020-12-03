# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import sys
import cv2
import time
from .sdf_utils import *
import _init_paths
from fcn.config import cfg
from layers.sdf_matching_loss import SDFLoss

class sdf_optimizer():
    def __init__(self, classes, sdf_files, lr=0.01, optimizer='Adam', use_gpu=True):

        self.classes = classes
        self.sdf_files = sdf_files
        self.use_gpu = use_gpu
        num = len(sdf_files)
        self.xmins = np.zeros((num, ), dtype=np.float32)
        self.ymins = np.zeros((num, ), dtype=np.float32)
        self.zmins = np.zeros((num, ), dtype=np.float32)
        self.xmaxs = np.zeros((num, ), dtype=np.float32)
        self.ymaxs = np.zeros((num, ), dtype=np.float32)
        self.zmaxs = np.zeros((num, ), dtype=np.float32)

        sdf_torch_list = []
        for i in range(len(sdf_files)):
            sdf_file = sdf_files[i]
            print(' start loading sdf from {} ... '.format(sdf_file))

            if sdf_file[-3:] == 'sdf':
                sdf_info = read_sdf(sdf_file)
                sdf = sdf_info[0]
                min_coords = sdf_info[1]
                delta = sdf_info[2]
                max_coords = min_coords + delta * np.array(sdf.shape)
                self.xmins[i], self.ymins[i], self.zmins[i] = min_coords
                self.xmaxs[i], self.ymaxs[i], self.zmaxs[i] = max_coords
                sdf_torch_list.append(torch.from_numpy(sdf).float())
            elif sdf_file[-3:] == 'pth':
                sdf_info = torch.load(sdf_file)
                min_coords = sdf_info['min_coords']
                max_coords = sdf_info['max_coords']
                self.xmins[i], self.ymins[i], self.zmins[i] = min_coords
                self.xmaxs[i], self.ymaxs[i], self.zmaxs[i] = max_coords
                sdf_torch_list.append(sdf_info['sdf_torch'][0, 0].permute(1, 0, 2))

            print('     minimal coordinate = ({:.4f}, {:.4f}, {:.4f}) cm'.format(self.xmins[i] * 100, self.ymins[i] * 100, self.zmins[i] * 100))
            print('     maximal coordinate = ({:.4f}, {:.4f}, {:.4f}) cm'.format(self.xmaxs[i] * 100, self.ymaxs[i] * 100, self.zmaxs[i] * 100))
            print(sdf_torch_list[-1].shape)
            print(' finished loading sdf ! ')

        # combine sdfs
        max_shape = np.array([sdf.shape for sdf in sdf_torch_list]).max(axis=0)
        self.sdf_torch = torch.ones((num, max_shape[0], max_shape[1], max_shape[2]), dtype=torch.float32)
        self.sdf_limits = np.zeros((num, 9), dtype=np.float32)
        for i in range(num):
            size = sdf_torch_list[i].shape
            self.sdf_torch[i, :size[0], :size[1], :size[2]] = sdf_torch_list[i]
            self.sdf_limits[i, 0] = self.xmins[i]
            self.sdf_limits[i, 1] = self.ymins[i]
            self.sdf_limits[i, 2] = self.zmins[i]
            self.sdf_limits[i, 3] = self.xmins[i] + (self.xmaxs[i] - self.xmins[i]) * max_shape[0] / size[0]
            self.sdf_limits[i, 4] = self.ymins[i] + (self.ymaxs[i] - self.ymins[i]) * max_shape[1] / size[1]
            self.sdf_limits[i, 5] = self.zmins[i] + (self.zmaxs[i] - self.zmins[i]) * max_shape[2] / size[2]
            self.sdf_limits[i, 6] = max_shape[0]
            self.sdf_limits[i, 7] = max_shape[1]
            self.sdf_limits[i, 8] = max_shape[2]
        self.sdf_limits = torch.from_numpy(self.sdf_limits)

        if self.use_gpu:
            self.sdf_torch = self.sdf_torch.cuda()
            self.sdf_limits = self.sdf_limits.cuda()

        self.sdf_loss = SDFLoss()


    def look_up(self, samples_x, samples_y, samples_z):
        samples_x = torch.clamp(samples_x, self.xmin, self.xmax)
        samples_y = torch.clamp(samples_y, self.ymin, self.ymax)
        samples_z = torch.clamp(samples_z, self.zmin, self.zmax)

        samples_x = (samples_x - self.xmin) / (self.xmax - self.xmin)
        samples_y = (samples_y - self.ymin) / (self.ymax - self.ymin)
        samples_z = (samples_z - self.zmin) / (self.zmax - self.zmin)

        samples = torch.cat((samples_z.unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4),
                             samples_x.unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4),
                             samples_y.unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4)),
                            dim=4)

        samples = samples * 2 - 1

        return F.grid_sample(self.sdf_torch, samples, padding_mode="border")

    def compute_dist(self, d_pose, T_oc_0, ps_c):

        ps_o = torch.mm(Oplus(T_oc_0, d_pose, self.use_gpu), ps_c.permute(1, 0)).permute(1, 0)[:, :3]

        dist = self.look_up(ps_o[:, 0], ps_o[:, 1], ps_o[:, 2])

        return torch.abs(dist)

    def refine_pose(self, T_co_0, ps_c, steps=100):
        # input T_co_0: 4x4
        #       ps_c:   nx4

        if self.use_gpu:
            T_oc_0 = torch.from_numpy(np.linalg.inv(T_co_0)).cuda()
        else:
            T_oc_0 = torch.from_numpy(np.linalg.inv(T_co_0))

        self.dpose.data[:3] *= 0
        self.dpose.data[3:] = self.dpose.data[3:] * 0 + 1e-12

        self.dist = torch.zeros((ps_c.size(0),))
        if self.use_gpu:
            self.dist = self.dist.cuda()

        for i in range(steps):

            if self.optimizer_type == 'LBFGS':
                def closure():
                    self.optimizer.zero_grad()

                    dist = self.compute_dist(self.dpose, T_oc_0, ps_c)

                    self.dist = dist.detach()

                    dist_target = torch.zeros_like(dist)
                    if self.use_gpu:
                        dist_target = dist_target.cuda()

                    loss = self.loss(dist, dist_target)
                    loss.backward()

                    return loss

                self.optimizer.step(closure)

            elif self.optimizer_type == 'Adam':
                self.optimizer.zero_grad()

                dist = self.compute_dist(self.dpose, T_oc_0, ps_c)
                self.dist = dist.detach()
                dist_target = torch.zeros_like(dist)
                if self.use_gpu:
                    dist_target = dist_target.cuda()

                loss = self.loss(dist, dist_target)
                loss.backward()

                self.optimizer.step()

            # print('step: {}, loss = {}'.format(i + 1, loss.data.cpu().item()))

        T_oc_opt = Oplus(T_oc_0, self.dpose, self.use_gpu)
        T_co_opt = np.linalg.inv(T_oc_opt.cpu().detach().numpy())

        dist = torch.mean(torch.abs(self.dist)).detach().cpu().numpy()

        return T_co_opt, dist


    def refine_pose_layer(self, T_oc_0, points, steps=100):
        # input T_co_0: mx4x4, m is the number of objects
        #       points: nx3 in camera

        # construct initial pose
        pose_init = torch.from_numpy(T_oc_0).cuda()

        m = T_oc_0.shape[0]
        dpose = torch.zeros((m, 6), dtype=torch.float32, requires_grad=True, device=0)
        dpose.data[:, :3] *= 0
        dpose.data[:, 3:] = dpose.data[:, 3:] * 0 + 1e-12
        treg = cfg.TEST.SDF_TRANSLATION_REG
        rreg = cfg.TEST.SDF_ROTATION_REG
        regularization = torch.tensor([treg, treg, treg, rreg, rreg, rreg], dtype=torch.float32, requires_grad=False, device=0)

        start = time.time()
        for i in range(steps):

            # self.optimizer.zero_grad()
            loss, sdf_values, T_oc_opt, dalpha, J = self.sdf_loss(dpose, pose_init, self.sdf_torch, self.sdf_limits, points, regularization)
            # print(loss)
            # loss.backward()
            # self.optimizer.step()

            # JTJ = JTJ.cpu().detach().numpy() + np.diag([100, 100, 100, 0.001, 0.001, 0.001]).astype(np.float32)
            # J = J.cpu().detach().numpy()
            # dalpha = torch.from_numpy(np.matmul(np.linalg.inv(JTJ), J)).cuda()
            dpose = dpose - dalpha

            # self.dpose = self.dpose - 0.001 * J

        end = time.time()
        print('sdf refinement iterations %d, time %f' % (steps, end - start))
        return T_oc_opt.cpu().detach().numpy()
