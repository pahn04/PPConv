import torch.optim as optim

from models.s3dis import PPCNN2
from utils.config import Config, configs

# model
configs.model = Config(PPCNN2)
configs.model.num_classes = configs.data.num_classes
configs.model.extra_feature_channels = 6
configs.dataset.num_points = 8192

configs.train.optimizer.weight_decay = 0.00001
# train: scheduler
configs.train.scheduler = Config(optim.lr_scheduler.CosineAnnealingLR)
configs.train.scheduler.T_max = configs.train.num_epochs

configs.dataset.holdout_area = 5

configs.model.sa_blocks = [((32, 2, 64), (1024, 0.1, 32, (32, 64))),
                           ((64, 3, 32), (256, 0.2, 32, (64, 128))),
                           ((128, 3, 16), (64, 0.4, 32, (128, 256))),
                           (None, (16, 0.8, 32, (256, 256, 512)))]
configs.model.fp_blocks = [((256, 256), (256, 1, 8)),
                           ((256, 256), (256, 1, 16)),
                           ((256, 128), (128, 2, 32)),
                           ((128, 128, 64), (64, 1, 64))]

configs.model.proj_axes = [1,2,3]

configs.train.augment = True
