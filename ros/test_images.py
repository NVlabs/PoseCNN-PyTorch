#!/usr/bin/env python

# --------------------------------------------------------
# PoseCNN
# Copyright (c) 2018 NVIDIA
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""Test a PoseCNN on images"""

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data

import tf
import rosnode
import message_filters
import cv2
import torch.nn as nn
import threading

import argparse
import pprint
import time, os, sys
import os.path as osp
import numpy as np
import time
import rospy
import _init_paths
import networks

from fcn.test_imageset import test_image
from cv_bridge import CvBridge, CvBridgeError
from fcn.config import cfg, cfg_from_file, get_output_dir
from datasets.factory import get_dataset
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from transforms3d.quaternions import mat2quat, quat2mat, qmult
from scipy.optimize import minimize
from utils.blob import pad_im, chromatic_transform, add_noise
from geometry_msgs.msg import PoseStamped
from ycb_renderer import YCBRenderer
from utils.se3 import *
from utils.nms import *
from Queue import Queue
from sdf.sdf_optimizer import sdf_optimizer

lock = threading.Lock()

class ImageListener:

    def __init__(self, network, dataset):

        self.net = network
        self.dataset = dataset
        self.cv_bridge = CvBridge()
        self.renders = dict()

        self.im = None
        self.depth = None
        self.rgb_frame_id = None

        suffix = '_%02d' % (cfg.instance_id)
        prefix = '%02d_' % (cfg.instance_id)
        self.suffix = suffix
        self.prefix = prefix
        fusion_type = ''

        # initialize a node
        rospy.init_node("posecnn_rgb")
        self.br = tf.TransformBroadcaster()
        self.label_pub = rospy.Publisher('posecnn_label' + fusion_type + suffix, Image, queue_size=10)
        self.pose_pub = rospy.Publisher('posecnn_pose' + fusion_type + suffix, Image, queue_size=10)
        self.pose_refined_pub = rospy.Publisher('posecnn_pose_refined' + fusion_type + suffix, Image, queue_size=10)

        # create pose publisher for each known object class
        self.pubs = []
        for i in range(1, self.dataset.num_classes):
            if self.dataset.classes[i][3] == '_':
                cls = prefix + self.dataset.classes[i][4:]
            else:
                cls = prefix + self.dataset.classes[i]
            cls = cls + fusion_type
            self.pubs.append(rospy.Publisher('/objects/prior_pose/' + cls, PoseStamped, queue_size=10))

        if cfg.TEST.ROS_CAMERA == 'D435':
            # use RealSense D435
            rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image, queue_size=10)
            depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, queue_size=10)
            msg = rospy.wait_for_message('/camera/color/camera_info', CameraInfo)
        elif cfg.TEST.ROS_CAMERA == 'Azure':             
            rgb_sub = message_filters.Subscriber('/rgb/image_raw', Image, queue_size=10)
            depth_sub = message_filters.Subscriber('/depth_to_rgb/image_raw', Image, queue_size=10)
            msg = rospy.wait_for_message('/rgb/camera_info', CameraInfo)
        else:
            # use kinect
            rgb_sub = message_filters.Subscriber('/%s/rgb/image_color' % (cfg.TEST.ROS_CAMERA), Image, queue_size=2)
            depth_sub = message_filters.Subscriber('/%s/depth_registered/image' % (cfg.TEST.ROS_CAMERA), Image, queue_size=2)
            msg = rospy.wait_for_message('/%s/rgb/camera_info' % (cfg.TEST.ROS_CAMERA), CameraInfo)

        # update camera intrinsics
        K = np.array(msg.K).reshape(3, 3)
        self.dataset._intrinsic_matrix = K
        print(self.dataset._intrinsic_matrix)

        queue_size = 1
        slop_seconds = 0.1
        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size, slop_seconds)
        ts.registerCallback(self.callback_rgbd)

    def callback_rgbd(self, rgb, depth):

        if depth.encoding == '32FC1':
            depth_cv = self.cv_bridge.imgmsg_to_cv2(depth)
        elif depth.encoding == '16UC1':
            depth_cv = self.cv_bridge.imgmsg_to_cv2(depth).copy().astype(np.float32)
            depth_cv /= 1000.0
        else:
            rospy.logerr_throttle(
                1, 'Unsupported depth type. Expected 16UC1 or 32FC1, got {}'.format(
                    depth.encoding))
            return

        im = self.cv_bridge.imgmsg_to_cv2(rgb, 'bgr8')

        with lock:
            self.im = im.copy()
            self.depth = depth_cv.copy()
            self.rgb_frame_id = rgb.header.frame_id
        
    def run_network(self):

        with lock:
            if listener.im is None:
                return
            im = self.im.copy()
            depth_cv = self.depth.copy()
            rgb_frame_id = self.rgb_frame_id

        fusion_type = ''
        start_time = time.time()
        im_pose, im_pose_refined, im_label, labels, rois, poses, poses_refined = test_image(self.net, self.dataset, im, depth_cv)
        print("--- %s seconds ---" % (time.time() - start_time))

        # publish label image
        label_msg = self.cv_bridge.cv2_to_imgmsg(im_label)
        label_msg.header.stamp = rospy.Time.now()
        label_msg.header.frame_id = rgb_frame_id
        label_msg.encoding = 'rgb8'
        self.label_pub.publish(label_msg)

        # publish pose image
        pose_msg = self.cv_bridge.cv2_to_imgmsg(im_pose)
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = rgb_frame_id
        pose_msg.encoding = 'rgb8'
        self.pose_pub.publish(pose_msg)

        # publish pose refined image
        pose_msg = self.cv_bridge.cv2_to_imgmsg(im_pose_refined)
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = rgb_frame_id
        pose_msg.encoding = 'rgb8'
        self.pose_refined_pub.publish(pose_msg)

        # poses
        if cfg.TEST.ROS_CAMERA == 'D435':
            frame = 'camera_color_optical_frame'
        elif cfg.TEST.ROS_CAMERA == 'Azure':
            frame = 'rgb_camera_link'
        else:
            frame = '%s_depth_optical_frame' % (cfg.TEST.ROS_CAMERA)

        indexes = np.zeros((self.dataset.num_classes, ), dtype=np.int32)
        if not rois.shape[0]:
            return

        index = np.argsort(rois[:, 2])
        rois = rois[index, :]
        poses = poses[index, :]
        for i in range(rois.shape[0]):
            cls = int(rois[i, 1])
            if cls > 0 and rois[i, -1] > cfg.TEST.DET_THRESHOLD:
                if not np.any(poses[i, 4:]):
                    continue

                if self.dataset.classes[cls][3] == '_':
                    name = self.prefix + self.dataset.classes[cls][4:]
                else:
                    name = self.prefix + self.dataset.classes[cls]
                name = name + fusion_type
                indexes[cls] += 1
                name = name + '_%02d' % (indexes[cls])
                tf_name = os.path.join("posecnn", name)

                # send transformation as bounding box (mis-used)
                n = np.linalg.norm(rois[i, 2:6])
                x1 = rois[i, 2] / n
                y1 = rois[i, 3] / n
                x2 = rois[i, 4] / n
                y2 = rois[i, 5] / n
                now = rospy.Time.now()
                self.br.sendTransform([n, now.secs, 0], [x1, y1, x2, y2], now, tf_name + '_roi', frame)

                quat = [poses[i, 1], poses[i, 2], poses[i, 3], poses[i, 0]]
                self.br.sendTransform(poses[i, 4:7], quat, rospy.Time.now(), tf_name, frame)

                # create pose msg
                msg = PoseStamped()
                msg.header.stamp = rospy.Time.now()
                msg.header.frame_id = frame
                msg.pose.orientation.x = poses[i, 1]
                msg.pose.orientation.y = poses[i, 2]
                msg.pose.orientation.z = poses[i, 3]
                msg.pose.orientation.w = poses[i, 0]
                msg.pose.position.x = poses[i, 4]
                msg.pose.position.y = poses[i, 5]
                msg.pose.position.z = poses[i, 6]
                pub = self.pubs[cls - 1]
                pub.publish(msg)


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a PoseCNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--instance', dest='instance_id', help='PoseCNN instance id to use',
                        default=0, type=int)
    parser.add_argument('--pretrained', dest='pretrained',
                        help='initialize with pretrained checkpoint',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--dataset', dest='dataset_name',
                        help='dataset to train on',
                        default='shapenet_scene_train', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--network', dest='network_name',
                        help='name of the network',
                        default=None, type=str)
    parser.add_argument('--cad', dest='cad_name',
                        help='name of the CAD file',
                        default=None, type=str)
    parser.add_argument('--pose', dest='pose_name',
                        help='name of the pose files',
                        default=None, type=str)
    parser.add_argument('--background', dest='background_name',
                        help='name of the background file',
                        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)

    # device
    cfg.device = torch.device('cuda:{:d}'.format(0))
    print('GPU device {:d}'.format(args.gpu_id))
    cfg.gpu_id = args.gpu_id
    cfg.instance_id = args.instance_id

    # dataset
    cfg.MODE = 'TEST'
    dataset = get_dataset(args.dataset_name)

    # prepare network
    if args.pretrained:
        network_data = torch.load(args.pretrained)
        print("=> using pre-trained network '{}'".format(args.pretrained))
    else:
        network_data = None
        print("no pretrained network specified")
        sys.exit()

    network = networks.__dict__[args.network_name](dataset.num_classes, cfg.TRAIN.NUM_UNITS, network_data).cuda(device=cfg.device)
    network = torch.nn.DataParallel(network, device_ids=[0]).cuda(device=cfg.device)
    cudnn.benchmark = True

    #'''
    print('loading 3D models')
    cfg.renderer = YCBRenderer(width=cfg.TRAIN.SYN_WIDTH, height=cfg.TRAIN.SYN_HEIGHT, gpu_id=args.gpu_id, render_marker=False)
    model_mesh_paths = [dataset.model_mesh_paths[i-1] for i in cfg.TEST.CLASSES[1:]]
    model_texture_paths = [dataset.model_texture_paths[i-1] for i in cfg.TEST.CLASSES[1:]]
    model_colors = [dataset.model_colors[i-1] for i in cfg.TEST.CLASSES[1:]]
    cfg.renderer.load_objects(model_mesh_paths, model_texture_paths, model_colors)
    cfg.renderer.set_camera_default()
    print(dataset.model_mesh_paths)
    #'''

    # load sdfs
    if cfg.TEST.POSE_REFINE:
        print('loading SDFs')
        sdf_files = []
        for i in cfg.TEST.CLASSES[1:]:
            sdf_files.append(dataset.model_sdf_paths[i-1])
        cfg.sdf_optimizer = sdf_optimizer(cfg.TEST.CLASSES[1:], sdf_files)

    # image listener
    network.eval()
    listener = ImageListener(network, dataset)

    while not rospy.is_shutdown():       
       listener.run_network()
