#!/bin/bash

set -x
set -e

./tools/train_net.py \
  --network posecnn \
  --pretrained output/ycb_object/ycb_object_train/vgg16_ycb_object_epoch_16.checkpoint.pth \
  --dataset ycb_self_supervision_all \
  --cfg experiments/cfgs/ycb_object_self_supervision.yml \
  --solver sgd \
  --epochs 8
