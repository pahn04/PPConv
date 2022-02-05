#!/bin/bash

python train.py configs/s3dis/ppcnnpp.py --devices $1 \
                                         --configs.train.save_path runs/s3dis.ppcnnpp.conv_pillar_caf

