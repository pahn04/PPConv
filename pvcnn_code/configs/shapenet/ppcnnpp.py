import torch.optim as optim

from models.shapenet import PPCNN2
from utils.config import Config, configs

# model
configs.model = Config(PPCNN2)
configs.model.num_classes = configs.data.num_classes
configs.model.num_shapes = configs.data.num_shapes
configs.model.extra_feature_channels = 3

configs.train.num_epochs = 250
configs.train.scheduler = Config(optim.lr_scheduler.CosineAnnealingLR)
configs.train.scheduler.T_max = configs.train.num_epochs

configs.model.sa_blocks = [((32, 1, 64), (512, 0.1, 32, (32, 64))),
                           ((64, 1, 32), (128, 0.2, 32, (64, 128))),
                           ((128, 1, 16), (32, 0.4, 32, (128, 256))),
                           (None, (16, 0.8, 16, (256, 512)))]
configs.model.fp_blocks = [((256, 256), (256, 1, 8)),
                           ((256, 256), (256, 1, 16)),
                           ((256, 128), (128, 1, 32)),
                           ((128, 64), (64, 1, 64))]

configs.model.proj_axes = [1,2,3]

configs.train.augment = True

