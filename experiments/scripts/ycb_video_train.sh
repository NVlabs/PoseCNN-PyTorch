#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0,1

time ./tools/train_net.py \
  --network posecnn \
  --pretrained data/checkpoints/vgg16-397923af.pth \
  --dataset ycb_video_train \
  --cfg experiments/cfgs/ycb_video.yml \
  --solver sgd \
  --epochs 8
