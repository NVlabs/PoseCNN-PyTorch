#!/usr/bin/env python

# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

"""collect images from Intel RealSense D435"""

import rospy
import message_filters
import cv2
import argparse
import pprint
import time, os, sys
import os.path as osp
import numpy as np
import yaml
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo

class ImageListener:

    def __init__(self):

        self.cv_bridge = CvBridge()
        self.count = 0

        # output dir
        this_dir = osp.dirname(__file__)
        self.outdir = osp.join(this_dir, '..', 'data', 'images')
        if not os.path.exists(self.outdir):
            os.mkdir(self.outdir)

        # initialize a node
        rospy.init_node("image_listener")
        rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image, queue_size=2)
        depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, queue_size=2)
        msg = rospy.wait_for_message('/camera/color/camera_info', CameraInfo)

        # save camera intrinsics
        intrinsic_matrix = np.array(msg.K).reshape(3, 3)
        print(intrinsic_matrix)

        dict_file = {'INTRINSICS' : intrinsic_matrix.flatten().tolist()}
        filename = os.path.join(self.outdir, 'meta.yml')
        with open(filename, 'w') as file:
            yaml.dump(dict_file, file)

        queue_size = 1
        slop_seconds = 0.025
        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size, slop_seconds)
        ts.registerCallback(self.callback)

    def callback(self, rgb, depth):
        if depth.encoding == '32FC1':
            depth_32 = self.cv_bridge.imgmsg_to_cv2(depth) * 1000
            depth_cv = np.array(depth_32, dtype=np.uint16)
        elif depth.encoding == '16UC1':
            depth_cv = self.cv_bridge.imgmsg_to_cv2(depth)
        else:
            rospy.logerr_throttle(
                1, 'Unsupported depth type. Expected 16UC1 or 32FC1, got {}'.format(
                    depth.encoding))
            return

        # write images
        im = self.cv_bridge.imgmsg_to_cv2(rgb, 'bgr8')
        filename = self.outdir + '/%06d-color.png' % self.count
        cv2.imwrite(filename, im)
        print filename

        filename = self.outdir + '/%06d-depth.png' % self.count
        cv2.imwrite(filename, depth_cv)
        print(filename)

        self.count += 1


if __name__ == '__main__':

    # image listener
    listener = ImageListener()
    try:  
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down"
