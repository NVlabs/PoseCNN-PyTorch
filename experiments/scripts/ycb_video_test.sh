#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

time ./tools/test_net.py --gpu $1 \
  --network posecnn \
  --pretrained output/ycb_video/ycb_video_train/vgg16_ycb_video_epoch_$2.checkpoint.pth \
  --dataset ycb_video_keyframe \
  --cfg experiments/cfgs/ycb_video.yml
