import torch.nn as nn
import torch

import modules.functional as F
from modules.shared_mlp import SharedMLP
from modules.se import SE

from modules.projection import Projection
from modules.backprojection import BackProjection

class Conv_pillar_caf(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, resolution, with_se=False, normalize=True, eps=1e-4, proj_axes=[1,2,3]):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.resolution = resolution
        self.eps = eps
        self.proj_axes = proj_axes

        mid_channels = out_channels // 2
        self.mid_channels = mid_channels

        self.projection = Projection(resolution, in_channels, mid_channels, eps=eps)
        if 1 in proj_axes:
            x_layers = [
                nn.Conv2d(mid_channels, mid_channels, kernel_size, stride=1, padding=kernel_size // 2),
                nn.BatchNorm2d(mid_channels, eps=1e-4),
                nn.LeakyReLU(0.1, True),
                nn.Conv2d(mid_channels, mid_channels, kernel_size, stride=1, padding=kernel_size // 2),
                nn.BatchNorm2d(mid_channels, eps=1e-4),
                nn.LeakyReLU(0.1, True),
            ]
            if with_se: x_layers.append(SE(mid_channels, reduction=4))
            self.x_layers = nn.Sequential(*x_layers)
        if 2 in proj_axes:
            y_layers = [
                nn.Conv2d(mid_channels, mid_channels, kernel_size, stride=1, padding=kernel_size // 2),
                nn.BatchNorm2d(mid_channels, eps=1e-4),
                nn.LeakyReLU(0.1, True),
                nn.Conv2d(mid_channels, mid_channels, kernel_size, stride=1, padding=kernel_size // 2),
                nn.BatchNorm2d(mid_channels, eps=1e-4),
                nn.LeakyReLU(0.1, True),
            ]
            if with_se: y_layers.append(SE(mid_channels, reduction=4))
            self.y_layers = nn.Sequential(*y_layers)
        if 3 in proj_axes:
            z_layers = [
                nn.Conv2d(mid_channels, mid_channels, kernel_size, stride=1, padding=kernel_size // 2),
                nn.BatchNorm2d(mid_channels, eps=1e-4),
                nn.LeakyReLU(0.1, True),
                nn.Conv2d(mid_channels, mid_channels, kernel_size, stride=1, padding=kernel_size // 2),
                nn.BatchNorm2d(mid_channels, eps=1e-4),
                nn.LeakyReLU(0.1, True),
            ]
            if with_se: z_layers.append(SE(mid_channels, reduction=4))
            self.z_layers = nn.Sequential(*z_layers)

        self.backprojection = BackProjection(proj_axes=proj_axes, eps=eps)

        self.point_layers = SharedMLP(in_channels, mid_channels)

        self.att_mlp1 = SharedMLP(6, (mid_channels, mid_channels), last_relu=False)
        self.att_mlp2 = SharedMLP(mid_channels*2, (mid_channels, 1+len(proj_axes)), last_relu=False)
        self.softmax = nn.Softmax(dim=1)

        self.last_mlp = SharedMLP(mid_channels, out_channels)

    def forward(self, inputs):
        features, coords = inputs
        B, C, Np = features.shape
        dev = features.get_device()
        R = self.resolution

        #### for projection & backprojection
        norm_coords = coords - coords.mean(dim=2, keepdim=True)
        norm_coords = norm_coords/(norm_coords.norm(dim=1, keepdim=True).max(dim=2, keepdim=True)[0] * 2. + self.eps)  +0.5
        norm_coords = torch.clamp(norm_coords * (R-1), 0, R - 1 - self.eps)
        sample_idx = torch.arange(B, dtype=torch.int64, device=dev)
        sample_idx = sample_idx.unsqueeze(-1).expand(-1, Np).reshape(-1).unsqueeze(1)
        norm_coords = norm_coords.transpose(1,2).reshape(B * Np, 3)
        coords_int = torch.round(norm_coords).to(torch.int64)
        coords_int = torch.cat((sample_idx, coords_int), 1)
        p_v_dist = torch.cat((sample_idx, torch.abs(norm_coords-coords_int[:,1:])), 1)

        # projection & 2D Convolution layers
        proj_feat = []
        if 1 in self.proj_axes:
            proj_x = self.projection(features, norm_coords, coords_int, p_v_dist, 1).permute(0,3,1,2)
            proj_feat.append(proj_x + self.x_layers(proj_x))
        if 2 in self.proj_axes:
            proj_y = self.projection(features, norm_coords, coords_int, p_v_dist, 2).permute(0,3,1,2)
            proj_feat.append(proj_y + self.y_layers(proj_y))
        if 3 in self.proj_axes:
            proj_z = self.projection(features, norm_coords, coords_int, p_v_dist, 3).permute(0,3,1,2)
            proj_feat.append(proj_z + self.z_layers(proj_z))

        # backprojection
        backproj_feat = self.backprojection(proj_feat, coords_int, p_v_dist)

        # pointwise MLP
        point_feat = self.point_layers(features)

        # Context-Aware Fusion module
        fusion_feat = self.att_mlp1(torch.cat((coords, norm_coords.reshape(B, Np, 3).transpose(1,2)), 1))
        att_w = self.att_mlp2(torch.cat((fusion_feat, torch.max(fusion_feat, 2, keepdim=True).values.repeat(1, 1, Np)), 1))
        att_w = self.softmax(att_w)
        last_feat = att_w[:,[0],:] * point_feat + att_w[:,[1],:] * backproj_feat[0] + att_w[:,[2],:] * backproj_feat[1] + att_w[:,[3],:] * backproj_feat[2]
        last_feat = self.last_mlp(last_feat)
        
        return last_feat, coords

