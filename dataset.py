import glob
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from Mesh import Mesh
from mesh_deformations import generate_random_ray, ray_dent_mesh, ray_hole_mesh, mesh_to_voxel_grid
from topology import mesh_to_feature_tensor
from config import Config

LABEL_NORMAL = 0
LABEL_HOLE = 1
LABEL_DENT = 2


class ScrewMeshDataset(Dataset):
    def __init__(self, root_dir: str, cfg: Config, split: str = "train",
                 dent_depth_fraction: float = 0.4, target_jitter: float = 0.05,
                 cache_dir: str = "cache"):
        self.root_dir = root_dir
        self.cfg = cfg
        self.dent_depth_fraction = dent_depth_fraction
        self.target_jitter = target_jitter
        
        # make unique path name for the cache file based on the dataset root directory
        os.makedirs(cache_dir, exist_ok=True)
        folder_slug = os.path.basename(os.path.normpath(root_dir)).replace(" ", "_")
        cache_file = os.path.join(cache_dir, f"topo_cache_{folder_slug}.npz")

        # check if cached already
        if os.path.exists(cache_file):
            print(f"--> Found cached topological features at '{cache_file}'. Loading directly from disk...")
            with np.load(cache_file) as data:
                all_features = data["features"]
                all_labels = data["labels"]
            num_triplets = len(all_features) // 3
        else:
            # cache miss so need to get them
            base_files = sorted(glob.glob(os.path.join(root_dir, "*.obj")))
            if len(base_files) == 0:
                raise FileNotFoundError(f"No .obj files found in {root_dir}")

            all_features = []
            all_labels = []

            print(f"--> Cache miss. Loading and pre-computing topological features from {root_dir}...")
            for file_path in tqdm(base_files, desc="Pre-processing meshes"):
                # Load and repair baseline
                base_mesh = Mesh(file_path)
                
                # normal
                feat_normal = mesh_to_feature_tensor(base_mesh, cfg.max_points_per_dir, cfg.inf_fill)
                all_features.append(feat_normal)
                all_labels.append(LABEL_NORMAL)

                # defect rays
                ray_o_dent, ray_d_dent = generate_random_ray(base_mesh, target_jitter=self.target_jitter)
                ray_o_hole, ray_d_hole = generate_random_ray(base_mesh, target_jitter=self.target_jitter)

                # make dented mesh
                dent_mesh = None
                counter = 0
                while dent_mesh is None and counter < 5:
                    dent_mesh = ray_dent_mesh(base_mesh, ray_o_dent, ray_d_dent, depth_fraction=self.dent_depth_fraction)
                    counter += 1
                if dent_mesh is None:
                    dent_mesh = base_mesh  # fallback 
                
                feat_dent = mesh_to_feature_tensor(dent_mesh, cfg.max_points_per_dir, cfg.inf_fill)
                all_features.append(feat_dent)
                all_labels.append(LABEL_DENT)

                # make hol(e)y mesh
                hole_mesh = None
                counter = 0
                while hole_mesh is None and counter < 5:
                    hole_mesh = ray_hole_mesh(base_mesh, ray_o_hole, ray_d_hole)
                    counter += 1
                if hole_mesh is None:
                    hole_mesh = base_mesh  # fallback
                    
                feat_hole = mesh_to_feature_tensor(hole_mesh, cfg.max_points_per_dir, cfg.inf_fill)
                all_features.append(feat_hole)
                all_labels.append(LABEL_HOLE)

            # numpy
            all_features = np.array(all_features, dtype=np.float32)
            all_labels = np.array(all_labels, dtype=np.int64)

            # save to disk to save time
            np.savez_compressed(cache_file, features=all_features, labels=all_labels)
            print(f"--> Successfully saved processed features to disk at '{cache_file}'.")
            num_triplets = len(base_files)

        # 8-1-1 split
        indices = np.arange(num_triplets)
        
        # fixed seed for reprod
        rng = np.random.default_rng(seed=42)
        rng.shuffle(indices)

        n_train = int(0.8 * num_triplets)
        n_val = int(0.1 * num_triplets)

        if split == "train":
            split_triplets = indices[:n_train]
        elif split == "test":
            split_triplets = indices[n_train + n_val:]
        else:
            raise ValueError(f"Unknown split type: {split}. Choose from 'train', 'test'.")

        flat_indices = []
        for t_idx in split_triplets:
            flat_indices.extend([t_idx * 3, t_idx * 3 + 1, t_idx * 3 + 2])

        # final split
        self.features = torch.tensor(all_features[flat_indices], dtype=torch.float32)
        self.labels = torch.tensor(all_labels[flat_indices], dtype=torch.long)
        
        print(f"Split '{split}' initialized in memory with {len(self.labels)} total samples.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    


class ScrewMeshVoxelDataset(Dataset):
    def __init__(self, root_dir: str, cfg, split: str = "train",
                 dent_depth_fraction: float = 0.4, target_jitter: float = 0.05,
                 resolution: int = 32, cache_dir: str = "cache"):
        self.root_dir = root_dir
        self.cfg = cfg
        self.resolution = resolution
        
        folder_slug = os.path.basename(os.path.normpath(root_dir)).replace(" ", "_")
        self.split_cache_dir = os.path.join(cache_dir, f"voxels_{folder_slug}_{resolution}")
        os.makedirs(self.split_cache_dir, exist_ok=True)

        base_files = sorted(glob.glob(os.path.join(root_dir, "*.obj")))
        if len(base_files) == 0:
            raise FileNotFoundError(f"No .obj files found in {root_dir}")

        num_triplets = len(base_files)
        sentinel_file = os.path.join(self.split_cache_dir, f"sample_{num_triplets-1}_dent.npy")
        
        # cache miss
        if not os.path.exists(sentinel_file):
            print(f"--> Cache missing. Voxelizing to disk...")
            for idx, file_path in enumerate(tqdm(base_files, desc="Voxelizing to 3D")):
                norm_p = os.path.join(self.split_cache_dir, f"sample_{idx}_normal.npy")
                dent_p = os.path.join(self.split_cache_dir, f"sample_{idx}_dent.npy")
                hole_p = os.path.join(self.split_cache_dir, f"sample_{idx}_hole.npy")
                
                if os.path.exists(norm_p) and os.path.exists(dent_p) and os.path.exists(hole_p):
                    continue

                base_mesh = Mesh(file_path)
                
                # Normal 
                grid_normal = mesh_to_voxel_grid(base_mesh, self.resolution)
                np.save(norm_p, grid_normal)

                # Dent
                ray_o_dent, ray_d_dent = generate_random_ray(base_mesh, target_jitter=target_jitter)
                dent_mesh = None; counter = 0
                while dent_mesh is None and counter < 5:
                    dent_mesh = ray_dent_mesh(base_mesh, ray_o_dent, ray_d_dent, depth_fraction=dent_depth_fraction)
                    counter += 1
                if dent_mesh is None: dent_mesh = base_mesh
                grid_dent = mesh_to_voxel_grid(dent_mesh, self.resolution)
                np.save(dent_p, grid_dent)

                # Hole
                ray_o_hole, ray_d_hole = generate_random_ray(base_mesh, target_jitter=target_jitter)
                hole_mesh = None; counter = 0
                while hole_mesh is None and counter < 5:
                    hole_mesh = ray_hole_mesh(base_mesh, ray_o_hole, ray_d_hole)
                    counter += 1
                if hole_mesh is None: hole_mesh = base_mesh
                grid_hole = mesh_to_voxel_grid(hole_mesh, self.resolution)
                np.save(hole_p, grid_hole)

        # 8-1-1 split here too
        indices = np.arange(num_triplets)
        rng = np.random.default_rng(seed=42)
        rng.shuffle(indices)

        n_train = int(0.8 * num_triplets)
        n_val = int(0.1 * num_triplets)

        if split == "train":
            split_triplets = indices[:n_train]
        elif split == "test":
            split_triplets = indices[n_train + n_val:]
        else:
            raise ValueError(f"Unknown split type: {split}")

        self.sample_records = []
        for t_idx in split_triplets:
            self.sample_records.append((os.path.join(self.split_cache_dir, f"sample_{t_idx}_normal.npy"), LABEL_NORMAL))
            self.sample_records.append((os.path.join(self.split_cache_dir, f"sample_{t_idx}_dent.npy"), LABEL_DENT))
            self.sample_records.append((os.path.join(self.split_cache_dir, f"sample_{t_idx}_hole.npy"), LABEL_HOLE))

        print(f"3D Voxel Split '{split}' initialized.")

    def __len__(self):
        return len(self.sample_records)

    def __getitem__(self, idx):
        file_path, label = self.sample_records[idx]
        grid_tensor = np.load(file_path)
        return torch.from_numpy(grid_tensor), torch.tensor(label, dtype=torch.long)