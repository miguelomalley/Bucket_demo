"""
Trains a Conv3DNet to classify a screw mesh's directional
persistence barcodes as Normal (0), Hole (1), or Dent (2).

Usage:
    python train_conv.py
Higher resolution (storage intensive):
    python train_conv.py --resolution=64
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

from dataset import ScrewMeshVoxelDataset
from models import Conv3DNet
from config import Config


def evaluate(model, loader, criterion):
    model.eval()
    cfg = model.cfg
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for voxels, labels in loader:
            voxels, labels = voxels.to(cfg.device), labels.to(cfg.device)
            logits = model(voxels)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def train(model: Conv3DNet):
    cfg = model.cfg
    resolution = cfg.resolution
    model.to(cfg.device)

    print(f"Initializing 3D Datasets (Resolution: {resolution}^3)")
    train_ds = ScrewMeshVoxelDataset(cfg.train_dir, cfg, split="train", resolution=resolution)
    test_ds  = ScrewMeshVoxelDataset(cfg.train_dir, cfg, split="test", resolution=resolution)

    # torch loader
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    # boilerplate
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    best_test_acc = 0.0
    epoch_bar = tqdm(range(1, cfg.epochs + 1), desc="Training Conv3D Native Net")
    
    for epoch in epoch_bar:
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for voxels, labels in train_loader:
            voxels, labels = voxels.to(cfg.device), labels.to(cfg.device)

            optimizer.zero_grad()
            logits = model(voxels)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        test_loss, test_acc = evaluate(model, test_loader, criterion)

        tqdm.write(f"Epoch {epoch:3d} | Train Loss {train_loss:.4f} Acc {train_acc:.3f} | Test Loss {test_loss:.4f} Acc {test_acc:.3f}")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), "best_conv3d_net.pt")

    print(f"\nDone. Conv3D Best Test Accuracy: {best_test_acc:.3f}")


if __name__ == "__main__":
    cfg = Config()
    parser = argparse.ArgumentParser(description="Train Conv3D Net.")
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--batch_size", type=int, default=cfg.batch_size)
    parser.add_argument("--lr", type=float, default=cfg.lr, help="Learning rate")
    parser.add_argument("--device", type=str, default=cfg.device, help="(e.g., 'cuda' or 'cpu')")
    parser.add_argument("--train_dir", type=str, default=cfg.train_dir, help="Directory for train data")
    parser.add_argument("--test_dir", type=str, default=cfg.test_dir, help="Directory for test data")
    parser.add_argument("--dent_depth_fraction", type=float, default=cfg.dent_depth_fraction, help="Mesh dent depth factor")
    parser.add_argument("--target_jitter", type=float, default=cfg.target_jitter, help="Ray caster center-of-mass jitter factor")
    parser.add_argument("--resolution", type=int, default=32, help="3D Voxel dimension")

    args = parser.parse_args()

    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.device = args.device
    cfg.train_dir = args.train_dir
    cfg.test_dir = args.test_dir
    cfg.dent_depth_fraction = args.dent_depth_fraction
    cfg.target_jitter = args.target_jitter
    cfg.resolution = args.resolution
    args = parser.parse_args()

    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size

    model = Conv3DNet(cfg)
    train(model, resolution=args.resolution)