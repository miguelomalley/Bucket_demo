from dataclasses import dataclass
import torch

@dataclass
class Config:
    num_directions = 8
    num_classes = 3
    train_dir = "train/Screws and bolts with hexagonal head/"
    test_dir = "test/Screws and bolts with hexagonal head/"

    batch_size = 8
    epochs = 30
    lr = 1e-3
    dropout = 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    branch_hidden=(128, 64)
    branch_out: int = 32
    head_hidden=(128, 64)


    target_jitter = 0.05
    dent_depth_fraction = 0.4

    max_points_per_dir = 32   # cap on (H0 + H1) points kept per direction
    inf_fill = 10.0           # stand in for +inf death times
    point_dim = 4              # [birth, death, is_H0, is_H1]

    resolution = 32         # just for conv model, voxel dim
