#!/bin/bash

set -x
set -e

./tools/train_net.py \
  --network posecnn \
  --pretrained data/checkpoints/vgg16-397923af.pth \
  --dataset ycb_object_train \
  --cfg experiments/cfgs/ycb_object.yml \
  --solver sgd \
  --epochs 16
