"""
Trains a PersNet to classify a screw mesh's directional
persistence barcodes as Normal (0), Hole (1), or Dent (2).

Usage:
    python train.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import ScrewMeshDataset
from models import PersNet
from config import Config

import argparse

from tqdm.notebook import tqdm


def evaluate(model, loader, criterion):
    model.eval()
    cfg = model.cfg
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for feats, labels in loader:
            feats, labels = feats.to(cfg.device), labels.to(cfg.device)
            logits = model(feats)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def train(model: PersNet):
    cfg = model.cfg

    model.to(cfg.device)

    print(f"Fetching barcodes...")
    train_ds = ScrewMeshDataset(cfg.train_dir, cfg, split="train")
    test_ds   = ScrewMeshDataset(cfg.train_dir, cfg, split="test")

    # torch loader (not really needed because barcodes are tiny but hey)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    # boilerplate
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    best_test_acc = 0.0
    
    epoch_bar = tqdm(range(1, cfg.epochs + 1), desc="Training Model", unit="epoch")
    
    for epoch in epoch_bar:
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch:2d}", leave=False, unit="batch")
        
        for feats, labels in train_bar:
            feats, labels = feats.to(cfg.device), labels.to(cfg.device)

            optimizer.zero_grad()
            logits = model(feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            current_loss = running_loss / total
            current_acc = correct / total
            train_bar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.3f}")

        train_loss = running_loss / total
        train_acc = correct / total
        test_loss, test_acc = evaluate(model, test_loader, criterion)

        tqdm.write(f"Epoch {epoch:3d} | train loss {train_loss:.4f} acc {train_acc:.3f} "
                   f"| test loss {test_loss:.4f} acc {test_acc:.3f}")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), "best_persnet.pt")
            tqdm.write(f" -> Saved new best model checkpoint! (Test Acc: {best_test_acc:.3f})")

        epoch_bar.set_postfix(best_test_acc=f"{best_test_acc:.3f}")

    print(f"\nDone. PersNet best test accuracy achieved: {best_test_acc:.3f}")

if __name__ == "__main__":
    cfg = Config()
    parser = argparse.ArgumentParser(description="Train PersNet on barcodes.")
    
    parser.add_argument("--epochs", type=int, default=cfg.epochs, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=cfg.batch_size, help="Input batch size for training")
    parser.add_argument("--lr", type=float, default=cfg.lr, help="Learning rate")
    parser.add_argument("--device", type=str, default=cfg.device, help="(e.g., 'cuda' or 'cpu')")
    parser.add_argument("--train_dir", type=str, default=cfg.train_dir, help="Directory for train data")
    parser.add_argument("--test_dir", type=str, default=cfg.test_dir, help="Directory for test data")
    parser.add_argument("--dent_depth_fraction", type=float, default=cfg.dent_depth_fraction, help="Mesh dent depth factor")
    parser.add_argument("--target_jitter", type=float, default=cfg.target_jitter, help="Ray caster center-of-mass jitter factor")

    args = parser.parse_args()

    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.device = args.device
    cfg.train_dir = args.train_dir
    cfg.test_dir = args.test_dir
    cfg.dent_depth_fraction = args.dent_depth_fraction
    cfg.target_jitter = args.target_jitter

    model = PersNet(cfg)

    train(model)