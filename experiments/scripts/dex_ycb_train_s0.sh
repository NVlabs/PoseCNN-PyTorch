#!/bin/bash

set -x
set -e

time ./tools/train_net.py \
  --network posecnn \
  --pretrained data/checkpoints/vgg16-397923af.pth \
  --dataset dex_ycb_s0_train \
  --cfg experiments/cfgs/dex_ycb.yml \
  --solver sgd \
  --epochs 16
