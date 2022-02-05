import torch.nn as nn
import torch

from ppcnn.utils import create_pointnet2_sa_components, create_pointnet2_fp_modules, create_mlp_components

def get_model(num_class, input_channels, num_pts=8192):
    return PPCNN2(num_class)

class PPCNN2(nn.Module):
    def __init__(self, num_classes, extra_feature_channels=6,
                 width_multiplier=1, voxel_resolution_multiplier=1,
                 sa_blocks = [
                     ((32, 2, 64), (4096, 0.1, 32, (32, 64))),
                     ((64, 2, 32), (1024, 0.2, 32, (64, 128))),
                     ((128, 2, 16), (256, 0.4, 32, (128, 256))),
                     ((256, 2, 8), (64, 0.8, 32, (256, 512))),
                 ],
                 fp_blocks = [
                     ((512, 512), (512, 1, 8)),
                     ((512, 256), (256, 1, 16)),
                     ((256, 128), (128, 1, 32)),
                     ((128, 64), (64, 1, 64)),
                 ]):
        super().__init__()
        self.sa_blocks = sa_blocks
        self.fp_blocks = fp_blocks
        self.in_channels = extra_feature_channels + 3

        sa_layers, sa_in_channels, channels_sa_features, _ = create_pointnet2_sa_components(
            sa_blocks=self.sa_blocks, extra_feature_channels=extra_feature_channels, with_se=True,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier
        )
        self.sa_layers = nn.ModuleList(sa_layers)

        sa_in_channels[0] = extra_feature_channels
        fp_layers, channels_fp_features = create_pointnet2_fp_modules(
            fp_blocks=self.fp_blocks, in_channels=channels_sa_features, sa_in_channels=sa_in_channels, with_se=True,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier
        )
        self.fp_layers = nn.ModuleList(fp_layers)

        layers, _ = create_mlp_components(in_channels=channels_fp_features, out_channels=[128, 0.5, num_classes],
                                          classifier=True, dim=2, width_multiplier=width_multiplier)
        self.classifier = nn.Sequential(*layers)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, inputs):
        coords, features_ = self._break_up_pc(inputs)
        coords = coords.transpose(1,2)
        features = torch.cat((coords, features_), 1)

        coords_list, in_features_list = [], []
        for sa_blocks in self.sa_layers:
            in_features_list.append(features)
            coords_list.append(coords)
            features, coords = sa_blocks((features, coords))
        in_features_list[0] = features_

        for fp_idx, fp_blocks in enumerate(self.fp_layers):
            features, coords = fp_blocks((coords_list[-1-fp_idx], coords, features, in_features_list[-1-fp_idx]))

        return self.classifier(features).transpose(1,2).contiguous()

