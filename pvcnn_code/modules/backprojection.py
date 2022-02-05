import torch
import torch.nn as nn

class BackProjection(nn.Module):
    def __init__(self, proj_axes=[1,2,3], eps=1e-4):
        super().__init__()
        self.proj_axes = proj_axes
        self.eps = eps

    def forward(self, proj_feat, coords_int, p_v_dist):
        N, _ = coords_int.shape
        B, C, R, _ = proj_feat[0].shape
        Np = N // B
        dev = coords_int.device
        eps = torch.Tensor([self.eps]).to(dev)

        backprojections = []
        axes = [0,1,2,3]
        for i, a in enumerate(self.proj_axes):
            axis = axes[0:a] + axes[a+1:]
            coords_curr = coords_int[:,axis] # (B x Np) x 3
            proj = proj_feat[i].permute(0,2,3,1).reshape(-1, C).contiguous() # (B x R x R) x C
            index = (coords_curr[:,0] * R * R) + (coords_curr[:,1] * R) + coords_curr[:,2] # (B x Np)
            interp_w = (.5 - p_v_dist[:,axis[1:]] + eps).prod(dim=1).unsqueeze(1) # (B x Np)
            index = index.unsqueeze_(1).expand(-1, C) # (B x Np) x C
            backproj_feat = torch.gather(proj, 0, index) # (B x Np) x C
            backprojections.append((backproj_feat*interp_w).reshape(B, Np, C).transpose(1,2)) # B x C x Np

        return backprojections

