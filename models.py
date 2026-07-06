"""
PersNet:
----------------
Input:  (B, 8, max_points_per_dim, point_dim) 
    For each of the 8 directions:
        flatten the last two dims
        -> per-direction branch: stack of Linear + ReLU layers
        -> per-direction embedding vector

    Concatenate the 8 branch embeddings -> shared Linear + ReLU stack
        -> logits over {0: Normal, 1: Hole, 2: Dent}
Conv3DNet:
-------------------
Input:  (B, 1, resolution, resolution, resolution) -- a native 3D volumetric binary voxel grid
    Pass through a stack of 3D Convolutions
    -> Stride 2 does our downsampling
    -> nn.AdaptiveMaxPool3d((1, 1, 1)) collapses remaining spatial structures into the peak activation points
    -> Flatten down to a 128-dimensional configuration vector

Classification Head:
    Pass the global configuration vector -> shared Linear + ReLU + Dropout stack
        -> Logits over {0: Normal, 1: Hole, 2: Dent}
"""

import torch
import torch.nn as nn
from config import Config


class DirectionBranch(nn.Module):
    """Linear-ReLU stack for a single filtration direction."""

    def __init__(self, in_dim: int, hidden_dims=(128, 64), out_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, out_dim), nn.ReLU(inplace=True)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.flatten(start_dim=1) 
        return self.net(x)


class PersNet(nn.Module):
    def __init__(self,
                 cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.num_directions = cfg.num_directions
        branch_in = cfg.max_points_per_dir * cfg.point_dim

        # 8 independent branches, one per filtration direction
        self.branches = nn.ModuleList([
            DirectionBranch(branch_in, cfg.branch_hidden, cfg.branch_out, cfg.dropout)
            for _ in range(cfg.num_directions)
        ])

        # shared stack after concatenating the 8 branch embeddings
        head_in = cfg.branch_out * cfg.num_directions
        head_layers = []
        prev = head_in
        for h in cfg.head_hidden:
            head_layers += [nn.Linear(prev, h), nn.ReLU(inplace=True), nn.Dropout(cfg.dropout)]
            prev = h
        head_layers += [nn.Linear(prev, cfg.num_classes)]  # raw logits
        self.head = nn.Sequential(*head_layers)

    def forward(self, x):
        # x: (B, num_directions, max_points_per_dir, point_dim)
        assert x.shape[1] == self.num_directions, \
            f"expected {self.num_directions} directions, got {x.shape[1]}"

        branch_outs = []
        for i, branch in enumerate(self.branches):
            branch_outs.append(branch(x[:, i]))  # (B, branch_out)

        fused = torch.cat(branch_outs, dim=1)  # (B, branch_out * num_directions)
        return self.head(fused)  # (B, num_classes) logits

class Conv3DNet(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        # conv feature extractor with a stride 2 downsample. Leaky ReLU is less hard-edge than ReLU since this dataset is less processed than our topological one
        self.feature_extractor = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=2, padding=1),   # -> (32, Res/2, Res/2, Res/2)
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.1, inplace=True),
            
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),  # -> (64, Res/4, Res/4, Res/4)
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.1, inplace=True),
            
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1), # -> (128, Res/8, Res/8, Res/8)
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.1, inplace=True),
            
            # condense to single feature vector
            nn.AdaptiveMaxPool3d((1, 1, 1))                         # -> (128, 1, 1, 1)
        )
        
        # classification head
        head_layers = []
        prev = 128
        for h in cfg.head_hidden:
            head_layers += [nn.Linear(prev, h), nn.ReLU(inplace=True), nn.Dropout(cfg.dropout)]
            prev = h
        head_layers += [nn.Linear(prev, cfg.num_classes)]
        self.head = nn.Sequential(*head_layers)

    def forward(self, x):
        # x shape: (B, 1, Depth, Height, Width)
        feats = self.feature_extractor(x)
        feats = feats.flatten(start_dim=1)  # -> (B, 128)
        return self.head(feats)


if __name__ == "__main__":
    # quick sanity check
    model = PersNet()
    dummy = torch.zeros(4, 8, 32, 4)
    logits = model(dummy)
    print("logits shape:", logits.shape)  # (4, 3)