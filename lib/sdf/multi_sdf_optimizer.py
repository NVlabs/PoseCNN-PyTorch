# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

from sdf_utils import *

class multi_sdf_optimizer():
    def __init__(self, sdf_file, lr=0.01, online_calib=True, use_gpu=False):

        self.use_gpu = use_gpu
        print(' start loading sdf ... ')
        sdf_info = read_sdf(sdf_file)
        sdf = sdf_info[0]
        min_coords = sdf_info[1]
        delta = sdf_info[2]
        max_coords = min_coords + delta * np.array(sdf.shape)
        self.xmin, self.ymin, self.zmin = min_coords
        self.xmax, self.ymax, self.zmax = max_coords
        self.sdf_torch = torch.from_numpy(sdf).float().permute(1, 0, 2).unsqueeze(0).unsqueeze(1)
        if self.use_gpu:
            self.sdf_torch = self.sdf_torch.cuda()
        print('     sdf size = {}x{}x{}'.format(self.sdf_torch.size(2), self.sdf_torch.size(3), self.sdf_torch.size(4)))
        print('     minimal coordinate = ({:.4f}, {:.4f}, {:.4f}) cm'.format(self.xmin * 100, self.ymin * 100, self.zmin * 100))
        print('     maximal coordinate = ({:.4f}, {:.4f}, {:.4f}) cm'.format(self.xmax * 100, self.ymax * 100, self.zmax * 100))
        print(' finished loading sdf ! ')

        if use_gpu:
            self.dpose = torch.tensor([0, 0, 0, 1e-12, 1e-12, 1e-12], dtype=torch.float32, requires_grad=True, device=0)
        else:
            self.dpose = torch.tensor([0, 0, 0, 1e-12, 1e-12, 1e-12], dtype=torch.float32, requires_grad=True)

        self.online_calib = online_calib
        if online_calib:
            if use_gpu:
                self.dpose_ext = torch.tensor([0, 0, 0, 1e-12, 1e-12, 1e-12], dtype=torch.float32, requires_grad=True,
                                              device=0)
            else:
                self.dpose_ext = torch.tensor([0, 0, 0, 1e-12, 1e-12, 1e-12], dtype=torch.float32, requires_grad=True)
        else:
            self.dpose_ext = torch.tensor([0, 0, 0, 1e-12, 1e-12, 1e-12], dtype=torch.float32, requires_grad=False)
            if use_gpu:
                self.dpose_ext = self.dpose_ext.cuda()

        if online_calib:
            self.optimizer = optim.Adam([self.dpose,
                                         self.dpose_ext],
                                        lr=lr)
        else:
            self.optimizer = optim.Adam([self.dpose],
                                        lr=lr)

        self.loss = nn.L1Loss(reduction='sum')
        self.loss_extrinsics = nn.L1Loss(reduction='mean')

        if use_gpu:
            self.loss = self.loss.cuda()
            self.loss_ext_t = self.loss_ext_t.cuda()
            self.loss_extrinsics = self.loss_extrinsics.cuda()

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

        return F.grid_sample(self.sdf_torch, samples)

    def compute_dist(self, d_pose, T_oc_0, ps_c):

        ps_o = torch.mm(Oplus(T_oc_0, d_pose, self.use_gpu), ps_c.permute(1, 0)).permute(1, 0)[:, :3]

        dist = self.look_up(ps_o[:, 0], ps_o[:, 1], ps_o[:, 2])

        return dist

    def compute_dist_multiview(self, d_pose, d_pose_ext, T_oc_0, T_rc_0, ps_c, T_r0rv):
        # convert points to referece camera frame
        T_rc = Oplus(T_rc_0, d_pose_ext, self.use_gpu)
        T_cr = torch.inverse(T_rc)
        T_c0cv = torch.matmul(T_cr.unsqueeze(0), torch.matmul(T_r0rv, T_rc.unsqueeze(0)))

        ps_c0_all = []
        for i in range(len(ps_c)):
            ps_c0_all.append(torch.mm(T_c0cv[i], ps_c[i].permute(1, 0)).permute(1, 0).view(-1, 4))

        ps_c0 = torch.cat(ps_c0_all, dim=0)

        dist = self.compute_dist(d_pose, T_oc_0, ps_c0)

        return dist

    def refine_pose_singleview(self, T_co_0, ps_c, steps=100):
        # input T_co_0: 4x4
        #       ps_c:   nx4

        if self.use_gpu:
            T_oc_0 = torch.from_numpy(np.linalg.inv(T_co_0)).cuda()
        else:
            T_oc_0 = torch.from_numpy(np.linalg.inv(T_co_0))

        self.dpose.data[:3] *= 0
        self.dpose.data[3:] = self.dpose.data[3:] * 0 + 1e-12

        for i in range(steps):

            self.optimizer.zero_grad()

            dist = self.compute_dist(self.dpose, T_oc_0, ps_c)
            dist_target = torch.zeros_like(dist)
            if self.use_gpu:
                dist_target = dist_target.cuda()

            loss = self.loss(dist, dist_target)
            loss.backward()

            self.optimizer.step()

            # print('step: {}, loss = {}'.format(i + 1, loss.data.cpu().item()))

        T_oc_opt = Oplus(T_oc_0, self.dpose, self.use_gpu)
        T_co_opt = np.linalg.inv(T_oc_opt.cpu().detach().numpy())

        dist = torch.mean(torch.abs(dist)).detach().cpu().numpy()

        return T_co_opt, dist

    def refine_pose_multiview(self, T_co_0, T_rc_0, ps_c, T_r0rv, steps=100):
        # input T_co_0: 4x4 Relative pose of
        #       T_cr_0: 4x4 Extrinsics
        #       T_r0rv: vx4x4 Relative pose between robot frame at time 0 and time v
        #       ps_c: tuple   vXn_ix4, point cloud in each frame

        if self.use_gpu:
            T_oc_0 = torch.from_numpy(np.linalg.inv(T_co_0)).cuda()
            T_rc_0 = torch.from_numpy(T_rc_0).cuda()
            T_r0rv = torch.from_numpy(T_r0rv).cuda()
        else:
            T_oc_0 = torch.from_numpy(np.linalg.inv(T_co_0))
            T_rc_0 = torch.from_numpy(T_rc_0)
            T_r0rv = torch.from_numpy(T_r0rv)

        self.dpose.data[:3] *= 0
        self.dpose.data[3:] = self.dpose.data[3:] * 0 + 1e-12
        self.dpose_ext.data[:3] *= 0
        self.dpose_ext.data[3:] = self.dpose.data[3:] * 0 + 1e-12

        for i in range(steps):

            self.optimizer.zero_grad()

            dist = self.compute_dist_multiview(self.dpose, self.dpose_ext, T_oc_0, T_rc_0, ps_c, T_r0rv)

            dist_target = torch.zeros_like(dist)
            if self.use_gpu:
                dist_target = dist_target.cuda()


            loss = self.loss(dist, dist_target) + \
                   self.loss_extrinsics(self.dpose_ext[[0, 1, 2, 3, 5]],
                                        torch.zeros_like(self.dpose_ext[[0, 1, 2, 3, 5]])) * 1e8

            loss.backward()

            self.optimizer.step()

            # print('step: {}, loss = {}'.format(i + 1, loss.data.cpu().item()))

        T_oc_opt = Oplus(T_oc_0, self.dpose, self.use_gpu)
        T_co_opt = np.linalg.inv(T_oc_opt.cpu().detach().numpy())

        T_rc_opt = Oplus(T_rc_0, self.dpose_ext, self.use_gpu)
        T_rc_opt = T_rc_opt.cpu().detach().numpy()

        dist = torch.mean(torch.abs(dist)).detach().cpu().numpy()

        return T_co_opt, T_rc_opt, dist
