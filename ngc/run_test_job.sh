#!/bin/bash

# Usage: ./train_ngc.sh [run-name] [script-path]

set -e
set -v

ngc batch run \
    --instance "dgx1v.16g.2.norm" \
    --name "test_job" \
    --image "nvcr.io/nvidian/robotics/posecnn-pytorch:latest" \
    --result /result \
    --datasetid "58777:/posecnn-release/data/models" \
    --datasetid "58774:/posecnn-release/data/backgrounds" \
    --datasetid "8187:/posecnn-release/data/coco" \
    --datasetid "12874:/posecnn-release/data/YCB_Video/YCB_Video_Dataset" \
    --workspace posecnn-release:/posecnn-release \
    --commandline "sleep 168h"
