import torch
import torch.nn as nn

import torch_scatter

from modules.shared_mlp import SharedMLP

class Projection(nn.Module):
    def __init__(self, resolution, in_channels, out_channels, eps=1e-4):
        super().__init__()
        self.resolution = int(resolution)
        self.eps = eps
        mlp = [SharedMLP(in_channels+5, out_channels)]
        self.mlp = nn.Sequential(*mlp)
        self.out_channels = out_channels

    def forward(self, features, norm_coords, coords_int, p_v_dist, proj_axis):
        B, C, Np = features.shape
        R = self.resolution
        dev = features.device

        projections = []
        axes_all = [0,1,2,3]
        axes = axes_all[:proj_axis] + axes_all[proj_axis+1:]

        x_p_y_p = p_v_dist[:, axes[1:]]

        pillar_mean = torch.zeros([B * R * R, 3], device=dev)
        coords_int = coords_int[:,axes]
        index = (coords_int[:,0] * R * R) + (coords_int[:,1] * R) + coords_int[:,2]
        index = index.unsqueeze(1).expand(-1, 3)
        torch_scatter.scatter(norm_coords, index, dim=0, out=pillar_mean, reduce="mean")
        pillar_mean = torch.gather(pillar_mean, 0, index)
        x_c_y_c_z_c = norm_coords - pillar_mean

        features = torch.cat((features.transpose(1,2).reshape(B*Np,C),x_p_y_p,x_c_y_c_z_c),1).contiguous()

        features = self.mlp(features.reshape(B, Np, -1).transpose(1,2)).transpose(1,2).reshape(B * Np, -1)
        pillar_features = torch.zeros([B * R * R, self.out_channels], device=dev)
        index = index[:,0].unsqueeze(1).expand(-1, self.out_channels)
        torch_scatter.scatter(features, index, dim=0, out=pillar_features, reduce="max")

        return pillar_features.reshape(B, R, R, self.out_channels)

