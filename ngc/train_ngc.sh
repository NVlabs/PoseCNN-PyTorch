#!/bin/bash

# Usage: ./train_ngc.sh [run-name] [script-path] [num-gpus]

set -e
set -v

NAME=$1
SCRIPT=$2
NGPU=$3

ngc batch run \
    --instance "dgx1v.16g.$NGPU.norm" \
    --name "$NAME" \
    --image "nvcr.io/nvidian/robotics/posecnn-pytorch:latest" \
    --result /result \
    --datasetid "58777:/posecnn/data/models" \
    --datasetid "58774:/posecnn/data/backgrounds" \
    --datasetid "8187:/posecnn/data/coco" \
    --datasetid "11888:/posecnn-release/data/YCB_Video/YCB_Video_Dataset" \
    --workspace posecnn-release:/posecnn-release \
    --commandline "cd /posecnn-release; bash $SCRIPT" \
    --total-runtime 7D \
    --port 6006
