# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

"""Blob helper functions."""

import torch
import torch.nn as nn
import numpy as np
import cv2
import random

def im_list_to_blob(ims, num_channels):
    """Convert a list of images into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], num_channels),
                    dtype=np.float32)
    for i in xrange(num_images):
        im = ims[i]
        if num_channels == 1:
            blob[i, 0:im.shape[0], 0:im.shape[1], :] = im[:,:,np.newaxis]
        else:
            blob[i, 0:im.shape[0], 0:im.shape[1], :] = im

    return blob

def prep_im_for_blob(im, pixel_means, target_size, max_size):
    """Mean subtract and scale an image for use in a blob."""
    im = im.astype(np.float32, copy=False)
    im -= pixel_means
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)

    return im, im_scale


def pad_im(im, factor, value=0):
    height = im.shape[0]
    width = im.shape[1]

    pad_height = int(np.ceil(height / float(factor)) * factor - height)
    pad_width = int(np.ceil(width / float(factor)) * factor - width)

    if len(im.shape) == 3:
        return np.lib.pad(im, ((0, pad_height), (0, pad_width), (0,0)), 'constant', constant_values=value)
    elif len(im.shape) == 2:
        return np.lib.pad(im, ((0, pad_height), (0, pad_width)), 'constant', constant_values=value)


def unpad_im(im, factor):
    height = im.shape[0]
    width = im.shape[1]

    pad_height = int(np.ceil(height / float(factor)) * factor - height)
    pad_width = int(np.ceil(width / float(factor)) * factor - width)

    if len(im.shape) == 3:
        return im[0:height-pad_height, 0:width-pad_width, :]
    elif len(im.shape) == 2:
        return im[0:height-pad_height, 0:width-pad_width]


def chromatic_transform(im, label=None, d_h=None, d_s=None, d_l=None):
    """
    Given an image array, add the hue, saturation and luminosity to the image
    """
    # Set random hue, luminosity and saturation which ranges from -0.1 to 0.1
    if d_h is None:
        d_h = (np.random.rand(1) - 0.5) * 0.1 * 180
    if d_l is None:
        d_l = (np.random.rand(1) - 0.5) * 0.2 * 256
    if d_s is None:
        d_s = (np.random.rand(1) - 0.5) * 0.2 * 256
    # Convert the BGR to HLS
    hls = cv2.cvtColor(im, cv2.COLOR_BGR2HLS)
    h, l, s = cv2.split(hls)
    # Add the values to the image H, L, S
    new_h = (h + d_h) % 180
    new_l = np.clip(l + d_l, 0, 255)
    new_s = np.clip(s + d_s, 0, 255)
    # Convert the HLS to BGR
    new_hls = cv2.merge((new_h, new_l, new_s)).astype('uint8')
    new_im = cv2.cvtColor(new_hls, cv2.COLOR_HLS2BGR)

    if label is not None:
        I = np.where(label > 0)
        new_im[I[0], I[1], :] = im[I[0], I[1], :]
    return new_im


def add_noise(image, level = 0.1):

    # random number
    r = np.random.rand(1)

    # gaussian noise
    if r < 0.9:
        row,col,ch= image.shape
        mean = 0
        noise_level = random.uniform(0, level)
        sigma = np.random.rand(1) * noise_level * 256
        gauss = sigma * np.random.randn(row,col) + mean
        gauss = np.repeat(gauss[:, :, np.newaxis], ch, axis=2)
        noisy = image + gauss
        noisy = np.clip(noisy, 0, 255)
    else:
        # motion blur
        sizes = [3, 5, 7, 9, 11, 15]
        size = sizes[int(np.random.randint(len(sizes), size=1))]
        kernel_motion_blur = np.zeros((size, size))
        if np.random.rand(1) < 0.5:
            kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
        else:
            kernel_motion_blur[:, int((size-1)/2)] = np.ones(size)
        kernel_motion_blur = kernel_motion_blur / size
        noisy = cv2.filter2D(image, -1, kernel_motion_blur)

    return noisy.astype('uint8')


def add_noise_depth(image, level = 0.1):
    row,col,ch= image.shape
    noise_level = random.uniform(0, level)
    gauss = noise_level * np.random.randn(row,col)
    gauss = np.repeat(gauss[:, :, np.newaxis], ch, axis=2)
    noisy = image + gauss
    return noisy


def add_noise_depth_cuda(image, level = 0.1):
    noise_level = random.uniform(0, level)
    gauss = torch.randn_like(image) * noise_level
    noisy = image + gauss
    return noisy

def add_gaussian_noise_cuda(image, level = 0.1):

    # gaussian noise
    noise_level = random.uniform(0, level)
    gauss = torch.randn_like(image) * noise_level
    noisy = image + gauss
    noisy = torch.clamp(noisy, 0, 1.0)
    return noisy


def add_noise_cuda(image, level = 0.1):
    # random number
    r = np.random.rand(1)

    # gaussian noise
    if r < 0.8:
        noise_level = random.uniform(0, level)
        gauss = torch.randn_like(image) * noise_level
        noisy = image + gauss
        noisy = torch.clamp(noisy, 0, 1.0)
    else:
        # motion blur
        sizes = [3, 5, 7, 9, 11, 15]
        size = sizes[int(np.random.randint(len(sizes), size=1))]
        kernel_motion_blur = torch.zeros((size, size))
        if np.random.rand(1) < 0.5:
            kernel_motion_blur[int((size-1)/2), :] = torch.ones(size)
        else:
            kernel_motion_blur[:, int((size-1)/2)] = torch.ones(size)
        kernel_motion_blur = kernel_motion_blur.cuda() / size
        kernel_motion_blur = kernel_motion_blur.view(1, 1, size, size)
        kernel_motion_blur = kernel_motion_blur.repeat(image.size(2), 1,  1, 1)

        motion_blur_filter = nn.Conv2d(in_channels=image.size(2),
                                       out_channels=image.size(2),
                                       kernel_size=size,
                                       groups=image.size(2),
                                       bias=False,
                                       padding=int(size/2))

        motion_blur_filter.weight.data = kernel_motion_blur
        motion_blur_filter.weight.requires_grad = False
        noisy = motion_blur_filter(image.permute(2, 0, 1).unsqueeze(0))
        noisy = noisy.squeeze(0).permute(1, 2, 0)

    return noisy
