#!/bin/bash
	
set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

time ./tools/test_images.py --gpu 0 \
  --imgdir data/images/kitchen \
  --meta data/images/kitchen/meta.yml \
  --color *color.png \
  --network posecnn \
  --pretrained output/ycb_object/ycb_object_train/vgg16_ycb_object_epoch_13.checkpoint.pth \
  --dataset ycb_object_test \
  --cfg experiments/cfgs/ycb_object.yml
