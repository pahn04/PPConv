import torch.nn as nn
import torch

from models.utils import create_pointnet2_sa_components, create_pointnet2_fp_modules, create_mlp_components

__all__ = ['PPCNN2']

class PPCNN2(nn.Module):

    def __init__(self, num_classes, num_shapes, extra_feature_channels=3,
                 width_multiplier=1, voxel_resolution_multiplier=1,
                 sa_blocks = [((32, 2, 32), (1024, 0.1, 32, (32, 64))),
                              ((64, 3, 16), (256, 0.2, 32, (64, 128))),
                              ((128, 3, 8), (64, 0.4, 32, (128, 256))),
                              (None, (16, 0.8, 32, (256, 256, 512))),],
                 fp_blocks = [((256, 256), (256, 1, 8)),
                              ((256, 256), (256, 1, 8)),
                              ((256, 128), (128, 2, 16)),
                              ((128, 128, 64), (64, 1, 32)),],
                 with_se=True, proj_axes=[1,2,3]):
        super().__init__()
        self.in_channels = extra_feature_channels + 3
        self.num_shapes = num_shapes
        self.sa_blocks = sa_blocks
        self.fp_blocks = fp_blocks
        self.with_se = with_se
        self.proj_axes = proj_axes

        sa_layers, sa_in_channels, channels_sa_features, _ = create_pointnet2_sa_components(
            sa_blocks=self.sa_blocks, extra_feature_channels=extra_feature_channels, with_se=self.with_se,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier,
            eps=1e-4, proj_axes=self.proj_axes)
        self.sa_layers = nn.ModuleList(sa_layers)

        sa_in_channels[0] = extra_feature_channels
        fp_layers, channels_fp_features = create_pointnet2_fp_modules(
            fp_blocks=self.fp_blocks, in_channels=channels_sa_features, sa_in_channels=sa_in_channels,
            with_se=self.with_se, width_multiplier=width_multiplier,
            voxel_resolution_multiplier=voxel_resolution_multiplier, eps=1e-4, proj_axes=self.proj_axes)
        self.fp_layers = nn.ModuleList(fp_layers)

        layers, _ = create_mlp_components(in_channels=channels_fp_features + num_shapes,
                                          out_channels=[256, 0.2, 256, 0.2, 128, num_classes],
                                          classifier=True, dim=2, width_multiplier=width_multiplier)
        self.classifier = nn.Sequential(*layers)

    def forward(self, inputs):
        features = inputs[:, :self.in_channels, :]
        one_hot_vectors = inputs[:, -self.num_shapes:, :]
        num_points = features.size(-1)

        coords = features[:, :3, :]
        init_features = features[:, 3:, :]

        coords_list, in_features_list = [], []
        for sa_blocks in self.sa_layers:
            in_features_list.append(features)
            coords_list.append(coords)
            features, coords = sa_blocks((features, coords))
        in_features_list[0] = init_features

        for fp_idx, fp_blocks in enumerate(self.fp_layers):
            features, coords = fp_blocks((coords_list[-1-fp_idx], coords, features, in_features_list[-1-fp_idx]))

        features = torch.cat((features, one_hot_vectors), 1)

        return self.classifier(features)
