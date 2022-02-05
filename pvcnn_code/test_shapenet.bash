#!/bin/bash

python train_dml.py configs/shapenet/ppcnnpp.py --devices $1 \
                                                --evaluate \
                                                --configs.evaluate.batch_size 16 \
                                                --configs.evaluate.best_checkpoint_path runs/shapenet.ppcnnpp.iwf.run$2/best.pth.tar

