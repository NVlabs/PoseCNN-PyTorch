#!/bin/bash
	
set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

time ./ros/test_images.py --gpu $1 \
  --instance 0 \
  --network posecnn \
  --pretrained data/checkpoints/vgg16_ycb_object_detection_self_supervision_all_epoch_8.checkpoint.pth \
  --dataset ycb_object_test \
  --cfg experiments/cfgs/ycb_object_detection.yml
