# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='posecnn',
    ext_modules=[
        CUDAExtension(
            name='posecnn_cuda', 
            sources = [
            'backproject_kernel.cu',
            'sdf_matching_loss_kernel.cu',
            'posecnn_layers.cpp',
            'hard_label_kernel.cu',
            'hough_voting_kernel.cu',
            'roi_pooling_kernel.cu',
            'ROIAlign_cuda.cu',
            'point_matching_loss_kernel.cu'],
            include_dirs = ['/usr/local/include/eigen3', '/usr/local/include'])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
