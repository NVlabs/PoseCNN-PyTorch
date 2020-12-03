#!/bin/bash
	
set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

time ./tools/test_images.py --gpu 0 \
  --imgdir data/demo/ \
  --meta data/demo/meta.yml \
  --color *color.png \
  --network posecnn \
  --pretrained data/checkpoints/ycb_object/vgg16_ycb_object_self_supervision_epoch_8.checkpoint.pth \
  --dataset ycb_object_test \
  --cfg experiments/cfgs/ycb_object.yml
