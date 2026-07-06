import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from models import PersNet, Conv3DNet
from dataset import ScrewMeshDataset, ScrewMeshVoxelDataset

CLASS_NAMES = {0: "Normal", 1: "Hole", 2: "Dent"}

def evaluate_model(model, loader, device):
    """
    Evaluates Conv3D and PersNet models
    """
    model.eval()
    model.to(device)
    
    total_correct = 0
    total_samples = 0
    
    # Trackers for per-class accuracy
    class_correct = {0: 0, 1: 0, 2: 0}
    class_total = {0: 0, 1: 0, 2: 0}
    
    with torch.no_grad():
        for batch_data, labels in tqdm(loader, desc=f"Evaluating {model.__class__.__name__}", leave=False):
            batch_data = batch_data.to(device)
            labels = labels.to(device)
            
            logits = model(batch_data)
            preds = logits.argmax(dim=1)
            
            # Global tracking
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            
            # Per-class breakdown tracking
            for p, l in zip(preds.cpu().numpy(), labels.cpu().numpy()):
                class_total[l] += 1
                if p == l:
                    class_correct[l] += 1
                    
    overall_acc = total_correct / total_samples if total_samples > 0 else 0.0
    per_class_acc = {}
    for c in [0, 1, 2]:
        per_class_acc[c] = class_correct[c] / class_total[c] if class_total[c] > 0 else 0.0
        
    return overall_acc, per_class_acc

def main():
    # 1. Initialize configuration
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using execution device: {device}\n")
    
    # Ensure resolution fallback
    if not hasattr(cfg, "resolution"):
        cfg.resolution = 32
        
    persnet_weights = "best_persnet.pt"
    conv3d_weights = "best_conv3d_net.pt" 
    if not os.path.exists(conv3d_weights) and os.path.exists("best_conv2d_ablation_net.pt"):
        conv3d_weights = "best_conv2d_ablation_net.pt"

    print("=" * 60)
    print(" 1. LOADING DATASETS & MODELS ")
    print("=" * 60)
    
    # --- PersNet Pipeline Setup ---
    print("-> Getting Topological Test Dataset...")
    topo_test_ds = ScrewMeshDataset(cfg.train_dir, cfg, split="test")
    topo_loader = DataLoader(topo_test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    
    print("-> Building PersNet Architecture and mapping weights...")
    persnet = PersNet(cfg)
    if os.path.exists(persnet_weights):
        persnet.load_state_dict(torch.load(persnet_weights, map_location=device))
        print(f"   [SUCCESS] Loaded PersNet weights from: {persnet_weights}")
    else:
        print(f"   [WARNING] {persnet_weights} missing. Evaluation will use randomly initialized weights.")

    print("-" * 40)

    # --- Conv3DNet Pipeline Setup ---
    print(f"-> Getting Voxel 3D Test Dataset ({cfg.resolution}^3)...")
    voxel_test_ds = ScrewMeshVoxelDataset(cfg.train_dir, cfg, split="test", resolution=cfg.resolution)
    voxel_loader = DataLoader(voxel_test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    
    print("-> Building Conv3DNet Architecture and mapping weights...")
    conv3d_net = Conv3DNet(cfg)
    if os.path.exists(conv3d_weights):
        conv3d_net.load_state_dict(torch.load(conv3d_weights, map_location=device))
        print(f"   [SUCCESS] Loaded 3D Voxel weights from: {conv3d_weights}")
    else:
        print(f"   [WARNING] {conv3d_weights} missing. Evaluation will use randomly initialized weights.")

    print("\n" + "=" * 60)
    print(" 2. RUNNING PERFORMANCE EVALUATIONS ")
    print("=" * 60)
    
    pers_acc, pers_class = evaluate_model(persnet, topo_loader, device)
    c3d_acc, c3d_class = evaluate_model(conv3d_net, voxel_loader, device)
    
    print("\n" + "=" * 60)
    print(" 3. COMPARATIVE BENCHMARK ANALYSIS REPORT ")
    print("=" * 60)
    
    print(f"{'Metric / Class':<25} | {'PersNet (TDA Approach)':<22} | {'Conv3DNet (Volumetric)':<22}")
    print("-" * 75)
    print(f"{'Overall Accuracy':<25} | {pers_acc*100:20.2f}% | {c3d_acc*100:20.2f}%")
    print("-" * 75)
    for c, name in CLASS_NAMES.items():
        print(f"{f'Accuracy [{name}]':<25} | {pers_class[c]*100:20.2f}% | {c3d_class[c]*100:20.2f}%")
    print("=" * 75)
    

if __name__ == "__main__":
    main()