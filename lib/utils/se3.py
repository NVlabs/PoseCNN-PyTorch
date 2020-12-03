# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import numpy as np
from transforms3d.quaternions import mat2quat, quat2mat, qmult, qinverse
from transforms3d.euler import quat2euler, mat2euler, euler2quat

# RT is a 3x4 matrix
def se3_inverse(RT):
    R = RT[0:3, 0:3]
    T = RT[0:3, 3].reshape((3,1))
    RT_new = np.zeros((3, 4), dtype=np.float32)
    RT_new[0:3, 0:3] = R.transpose()
    RT_new[0:3, 3] = -1 * np.dot(R.transpose(), T).reshape((3))
    return RT_new

def se3_mul(RT1, RT2):
    R1 = RT1[0:3, 0:3]
    T1 = RT1[0:3, 3].reshape((3,1))

    R2 = RT2[0:3, 0:3]
    T2 = RT2[0:3, 3].reshape((3,1))

    RT_new = np.zeros((3, 4), dtype=np.float32)
    RT_new[0:3, 0:3] = np.dot(R1, R2)
    T_new = np.dot(R1, T2) + T1
    RT_new[0:3, 3] = T_new.reshape((3))
    return RT_new


def egocentric2allocentric(qt, T):
    dx = np.arctan2(T[0], -T[2])
    dy = np.arctan2(T[1], -T[2])
    quat = euler2quat(-dy, -dx, 0, axes='sxyz')
    quat = qmult(qinverse(quat), qt)
    return quat


def allocentric2egocentric(qt, T):
    dx = np.arctan2(T[0], -T[2])
    dy = np.arctan2(T[1], -T[2])
    quat = euler2quat(-dy, -dx, 0, axes='sxyz')
    quat = qmult(quat, qt)
    return quat


def T_inv_transform(T_src, T_tgt):
    '''
    :param T_src: 
    :param T_tgt:
    :return: T_delta: delta in pixel 
    '''
    T_delta = np.zeros((3, ), dtype=np.float32)

    T_delta[0] = T_tgt[0] / T_tgt[2] - T_src[0] / T_src[2]
    T_delta[1] = T_tgt[1] / T_tgt[2] - T_src[1] / T_src[2]
    T_delta[2] = np.log(T_src[2] / T_tgt[2])

    return T_delta


def rotation_x(theta):
    t = theta * np.pi / 180.0
    R = np.zeros((3, 3), dtype=np.float32)
    R[0, 0] = 1
    R[1, 1] = np.cos(t)
    R[1, 2] = -np.sin(t)
    R[2, 1] = np.sin(t)
    R[2, 2] = np.cos(t)
    return R

def rotation_y(theta):
    t = theta * np.pi / 180.0
    R = np.zeros((3, 3), dtype=np.float32)
    R[0, 0] = np.cos(t)
    R[0, 2] = np.sin(t)
    R[1, 1] = 1
    R[2, 0] = -np.sin(t)
    R[2, 2] = np.cos(t)
    return R

def rotation_z(theta):
    t = theta * np.pi / 180.0
    R = np.zeros((3, 3), dtype=np.float32)
    R[0, 0] = np.cos(t)
    R[0, 1] = -np.sin(t)
    R[1, 0] = np.sin(t)
    R[1, 1] = np.cos(t)
    R[2, 2] = 1
    return R
