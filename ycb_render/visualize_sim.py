# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import torch
import cv2
import numpy as np
import glob
from transforms3d.quaternions import mat2quat, quat2mat
from ycb_renderer_sim import YCBRenderer

if __name__ == '__main__':

    model_path = '.'
    width = 640
    height = 480
    files = glob.glob('data/*.npy')

    renderer = YCBRenderer(width=width, height=height, render_marker=True)
    models = ['003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle', '010_potted_meat_can']
    colors = [[0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0.5, 0.5, 0]]

    # models = ['003_cracker_box']
    # colors = [[0, 1, 0]]

    obj_paths = [
        '{}/models_sim/{}/meshes/{}.obj'.format(model_path, item, item) for item in models]
    texture_paths = [
        '{}/models_sim/{}/meshes/texture_map.png'.format(model_path, item) for item in models]
    renderer.load_objects(obj_paths, texture_paths, colors)

    renderer.set_fov(60)
    renderer.set_light_pos([0, 0, 0])
    renderer.set_camera([0, 0, 0], [1, 0, 0], [0, 0, 1])

    image_tensor = torch.cuda.FloatTensor(height, width, 4).detach()
    seg_tensor = torch.cuda.FloatTensor(height, width, 4).detach()
    RT_object = np.zeros((3, 4), dtype=np.float32)
    RT_camera = np.zeros((3, 4), dtype=np.float32)

    for file_path in files[1:]:

        print file_path
      
        data = np.load(file_path).item()
        cls_indexes = []
        poses = []

        print('object_labels', data['object_labels'])
        print('fov', data['horizontal_fov'])

        for i, object_name in enumerate(data['object_labels']):

            cls_index = -1
            for j in range(len(models)):
                if object_name in models[j]:
                    cls_index = j
                    break

            if cls_index >= 0:
                cls_indexes.append(cls_index)

                RT = np.zeros((3, 4), dtype=np.float32)

                w = data['relative_poses'][i][0]
                x = data['relative_poses'][i][1]
                y = data['relative_poses'][i][2]
                z = data['relative_poses'][i][3]
                RT[:3, :3] = quat2mat([w, x, y, z])

                x = data['relative_poses'][i][4]
                y = data['relative_poses'][i][5]
                z = data['relative_poses'][i][6]
                RT[:, 3] = [x, y, z]
                print RT

                qt = np.zeros((7, ), dtype=np.float32)
                qt[3:] = mat2quat(RT[:3, :3])
                qt[:3] = RT[:, 3]
                print qt

                poses.append(qt)

            print('object_name: {}, relative_qt = {}, absolute_qt = {}'.format(data['object_labels'][i], data['relative_poses'][i], data['absolute_poses'][i]))

        renderer.set_poses(poses)

        renderer.render(cls_indexes, image_tensor, seg_tensor)
        image_tensor = image_tensor.flip(0)
        seg_tensor = seg_tensor.flip(0)

        # RGB to BGR order
        im = image_tensor.cpu().numpy()
        im = np.clip(im, 0, 1)
        im = im[:, :, (2, 1, 0)] * 255
        im = im.astype(np.uint8)

        im_label = seg_tensor.cpu().numpy()
        im_label = im_label[:, :, (2, 1, 0)] * 255
        im_label = np.round(im_label).astype(np.uint8)
        im_label = np.clip(im_label, 0, 255)
    
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(2, 2, 1)
        plt.imshow(data['rgb'][:, :, (2, 1, 0)])

        ax = fig.add_subplot(2, 2, 2)
        mask = np.squeeze(data['segmentation'], -1).astype(np.uint8)
        mask *= 40
        plt.imshow(mask)

        ax = fig.add_subplot(2, 2, 3)
        plt.imshow(im[:, :, (2, 1, 0)])

        ax = fig.add_subplot(2, 2, 4)
        plt.imshow(im_label[:, :, (2, 1, 0)])
        plt.show()
