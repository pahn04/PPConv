#!/bin/bash

python train.py configs/s3dis/ppcnnpp.py --devices $1 \
                                         --evaluate \
                                         --configs.evaluate.batch_size 7 \
                                         --configs.evaluate.best_checkpoint_path ./ppcnnpp_s3dis_xyz_pillar_caf.pth.tar

