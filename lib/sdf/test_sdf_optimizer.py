# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

from sdf_optimizer import *
from transforms3d.quaternions import mat2quat, quat2mat
import cv2

def Twc_np(pose):

    Twc = np.zeros((4, 4), dtype=np.float32)

    Twc[:3, :3] = quat2mat(pose[3:])
    Twc[:3, 3] = pose[:3]
    Twc[3, 3] = 1

    return Twc

try:
    import cPickle as pickle
except:
    import pickle
import numpy as np
from itertools import izip_longest as izip

class SignedDensityField(object):
    """ Data is stored in the following way
            data[x, y, z]
    """

    def __init__(self, data, origin, delta):
        self.data = data
        self.nx, self.ny, self.nz = data.shape
        self.origin = origin
        self.delta = delta
        self.max_coords = self.origin + delta * np.array(data.shape)

    def _rel_pos_to_idxes(self, rel_pos):
        i_min = np.array([0, 0, 0], dtype=np.int)
        i_max = np.array([self.nx - 1, self.ny - 1, self.nz - 1], dtype=np.int)
        return np.clip(((rel_pos - self.origin) / self.delta).astype(int), i_min, i_max)

    def get_distance(self, rel_pos):
        idxes = self._rel_pos_to_idxes(rel_pos)
        assert idxes.shape[0] == rel_pos.shape[0]
        return self.data[idxes[:, 0], idxes[:, 1], idxes[:, 2]]

    def dump(self, pkl_file):
        data = {}
        data['data'] = self.data
        data['origin'] = self.origin
        data['delta'] = self.delta
        pickle.dump(data, open(pkl_file, "wb"), protocol=2)

    def visualize(self, max_dist=0.1):
        try:
            from mayavi import mlab
        except:
            print("mayavi is not installed!")

        figure = mlab.figure('Signed Density Field')
        SCALE = 100  # The dimensions will be expressed in cm for better visualization.
        data = np.copy(self.data)
        data = np.minimum(max_dist, data)
        xmin, ymin, zmin = SCALE * self.origin
        xmax, ymax, zmax = SCALE * self.max_coords
        delta = SCALE * self.delta
        xi, yi, zi = np.mgrid[xmin:xmax:delta, ymin:ymax:delta, zmin:zmax:delta]
        data[data <= 0] -= 0.2
        data = -data
        grid = mlab.pipeline.scalar_field(xi, yi, zi, data)
        vmin = np.min(data)
        vmax = np.max(data)
        mlab.pipeline.volume(grid, vmin=vmin, vmax=(vmax + vmin) / 2)
        mlab.axes()
        mlab.show()

    @classmethod
    def from_sdf(cls, sdf_file):
        with open(sdf_file, "r") as file:
            axis = 2
            lines = file.readlines()
            nx, ny, nz = map(int, lines[0].split(' '))
            print(nx, ny, nz)
            x0, y0, z0 = map(float, lines[1].split(' '))
            print(x0, y0, z0)
            delta = float(lines[2].strip())
            print(delta)
            data = np.zeros([nx, ny, nz])
            for i, line in enumerate(lines[3:]):
                idx = i % nx
                idy = int(i / nx) % ny
                idz = int(i / (nx * ny))
                val = float(line.strip())
                data[idx, idy, idz] = val

        return cls(data, np.array([x0, y0, z0]), delta)

    @classmethod
    def from_pkl(cls, pkl_file):
        data = pickle.load(open(pkl_file, "r"))

        return cls(data['data'], data['origin'], data['delta'])

if __name__ == '__main__':

    # object_name = '002_master_chef_can'
    # object_name = '037_scissors'
    # object_name = '061_foam_brick'
    object_name = '007_tuna_fish_can'

    visualize_sdf = False

    sdf_file = '../../data/YCB_Object/models/{}/textured_simple_low_res.pth'.format(object_name)

    sdf_optim = sdf_optimizer(sdf_file, lr=0.01, use_gpu=True, optimizer='Adam')
    print(torch.max(sdf_optim.sdf_torch))
    print(torch.min(sdf_optim.sdf_torch))

    if visualize_sdf:
        sdf_show = SignedDensityField.from_sdf(sdf_file)
        sdf_show.visualize()

    # load points of the same object
    point_file = '../../data/YCB_Object/models/{}/points.xyz'.format(object_name)
    points = torch.from_numpy(np.loadtxt(point_file)).float()
    points = torch.cat((points, torch.ones((points.size(0), 1), dtype=torch.float32)), dim=1)
    points_np = points.numpy()
    print(points_np.shape)

    # set ground truth pose
    pose_gt = np.zeros((7,), dtype=np.float32)
    pose_gt[:3] = np.array([1, 1, 1], dtype=np.float32)

    R = np.array([[-1, 0, 0],
                 [0, np.sqrt(0.5), -np.sqrt(0.5)],
                 [0, -np.sqrt(0.5), -np.sqrt(0.5)]], dtype=np.float32)

    pose_gt[3:] = mat2quat(R)

    # get measurements
    Twc_gt = Twc_np(pose_gt)
    points_c = np.matmul(np.linalg.inv(Twc_gt), np.transpose(points_np)).transpose()
    points_c = torch.from_numpy(points_c)

    index = np.random.permutation(np.arange(points_c.shape[0]))[:2000]
    points_c = points_c[index, :]
    print(points_c.shape)

    T_co_init = np.linalg.inv(Twc_gt)
    R_perturb = axangle2mat(np.random.rand(3,), 40 * np.random.rand() / 57.3, is_normalized=False)
    T_co_init[:3, :3] = np.matmul(T_co_init[:3, :3], R_perturb)
    T_co_init[:3, 3] += 0.01

    points_init = np.matmul(np.linalg.inv(T_co_init), points_c.numpy().transpose()).transpose()

    # optimization
    points_input = points_c[:, :3].clone().cuda()
    T_co_opt, sdf_values = sdf_optim.refine_pose_layer(T_co_init, points_input, steps=100)

    print(T_co_opt)
    print(np.linalg.inv(Twc_gt))
    np.set_printoptions(threshold=sys.maxsize)
    # print(sdf_values.detach().cpu().numpy())

    # visualization for debugging
    points_opt = np.matmul(np.linalg.inv(T_co_opt), points_c.numpy().transpose()).transpose()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(points_np[::5, 0], points_np[::5, 1], points_np[::5, 2], color='green')
    ax.scatter(points_init[::5, 0], points_init[::5, 1], points_init[::5, 2], color='red')
    ax.scatter(points_opt[::5, 0], points_opt[::5, 1], points_opt[::5, 2], color='blue')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    min_coor = np.min(np.array([sdf_optim.xmin, sdf_optim.ymin, sdf_optim.zmin]))
    max_coor = np.max(np.array([sdf_optim.xmax, sdf_optim.ymax, sdf_optim.zmax]))

    ax.set_xlim(min_coor, max_coor)
    ax.set_ylim(min_coor, max_coor)
    ax.set_zlim(min_coor, max_coor)

    plt.show()
