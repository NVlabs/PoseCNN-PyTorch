#!/bin/bash

set -x
set -e
export CUDA_VISIBLE_DEVICES=$1

time ./tools/test_net.py --gpu $1 \
  --network posecnn \
  --pretrained output/dex_ycb/dex_ycb_s1_train/vgg16_dex_ycb_epoch_$2.checkpoint.pth \
  --dataset dex_ycb_s1_test \
  --cfg experiments/cfgs/dex_ycb.yml
