#!/bin/bash

python train.py configs/s3dis/ppcnnpp.py --devices $1 \
                                         --evaluate \
                                         --configs.evaluate.best_checkpoint_path ./pretrained/ppcnnpp_pv_s3dis_caf.pth.tar

