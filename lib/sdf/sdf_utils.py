# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.init as init
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from transforms3d.quaternions import *
from transforms3d.axangles import *
import torch.nn.functional as F
import time


def read_sdf(sdf_file):
    with open(sdf_file, "r") as file:
        lines = file.readlines()
        nx, ny, nz = map(int, lines[0].split(' '))
        x0, y0, z0 = map(float, lines[1].split(' '))
        delta = float(lines[2].strip())
        data = np.zeros([nx, ny, nz])
        for i, line in enumerate(lines[3:]):
            idx = i % nx
            idy = int(i / nx) % ny
            idz = int(i / (nx * ny))
            val = float(line.strip())
            data[idx, idy, idz] = val
    return (data, np.array([x0, y0, z0]), delta)


def skew(w, gpu=False):
    if gpu:
        wc = torch.stack((torch.tensor(0, dtype=torch.float32).cuda(), -w[2], w[1],
                          w[2], torch.tensor(0, dtype=torch.float32).cuda(), -w[0],
                          -w[1], w[0], torch.tensor(0, dtype=torch.float32).cuda()
                          )).view(3, 3)
    else:
        wc = torch.stack((torch.tensor(0, dtype=torch.float32), -w[2], w[1],
                          w[2], torch.tensor(0, dtype=torch.float32), -w[0],
                          -w[1], w[0], torch.tensor(0, dtype=torch.float32)
                         )).view(3, 3)

    return wc


def Exp(dq, gpu):

    if gpu:
        I = torch.eye(3, dtype=torch.float32).cuda()
    else:
        I = torch.eye(3, dtype=torch.float32)

    dphi = torch.norm(dq, p=2, dim=0)

    if dphi > 0.05:
        u = 1/dphi * dq

        ux = skew(u, gpu)

        if gpu:
            dR = I + torch.sin(dphi) * ux + (torch.tensor(1, dtype=torch.float32).cuda() - torch.cos(dphi)) * torch.mm(ux, ux)
        else:
            dR = I + torch.sin(dphi) * ux + (torch.tensor(1, dtype=torch.float32) - torch.cos(dphi)) * torch.mm(ux, ux)
    else:
        dR = I + skew(dq, gpu)

    return dR


def Oplus(T, v, gpu=False):

    dR = Exp(v[3:], gpu)
    dt = v[:3]

    if gpu:
        dT = torch.stack((dR[0, 0], dR[0, 1], dR[0, 2], dt[0],
                          dR[1, 0], dR[1, 1], dR[1, 2], dt[1],
                          dR[2, 0], dR[2, 1], dR[2, 2], dt[2],
                          torch.tensor(0, dtype=torch.float32).cuda(),
                          torch.tensor(0, dtype=torch.float32).cuda(),
                          torch.tensor(0, dtype=torch.float32).cuda(),
                          torch.tensor(1, dtype=torch.float32).cuda())).view(4, 4)
    else:
        dT = torch.stack((dR[0, 0], dR[0, 1], dR[0, 2], dt[0],
                          dR[1, 0], dR[1, 1], dR[1, 2], dt[1],
                          dR[2, 0], dR[2, 1], dR[2, 2], dt[2],
                          torch.tensor(0, dtype=torch.float32),
                          torch.tensor(0, dtype=torch.float32),
                          torch.tensor(0, dtype=torch.float32),
                          torch.tensor(1, dtype=torch.float32))).view(4, 4)

    return torch.mm(T, dT)
