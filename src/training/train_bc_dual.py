"""
train_bc_dual.py
-----------------
BC Training cho Dual-Timeframe Transformer (M5 + H1).

Cách dùng:
  python src/training/train_bc_dual.py --h5 data/processed/XAUUSD_DUAL_M5w256_H1w64.h5
  python src/training/train_bc_dual.py --h5 data/processed/XAUUSD_DUAL_M5w256_H1w64.h5 --epochs 100
"""

import argparse
import logging
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, classification_report

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.model.dual_transformer import DualTimeframeTransformer
from src.training.focal_loss import FocalLoss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

CHECKPOINT_DIR = Path("checkpoints")


class DualDataset(Dataset):
    """Dataset cho Dual-TF HDF5."""
    def __init__(self, X_m5, X_h1, y):
        self.X_m5 = torch.tensor(X_m5, dtype=torch.float32)
        self.X_h1 = torch.tensor(X_h1, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_m5[idx], self.X_h1[idx], self.y[idx]


def make_dual_split(h5_path, batch_size=256, val_ratio=0.2):
    """Load & split dual HDF5 dataset."""
    with h5py.File(h5_path, "r") as f:
        X_m5 = f["X_m5"][:].astype(np.float32)
        X_h1 = f["X_h1"][:].astype(np.float32)
        y = f["y"][:].astype(np.int64)

    n = len(y)
    split = int(n * (1 - val_ratio))

    train_ds = DualDataset(X_m5[:split], X_h1[:split], y[:split])
    val_ds = DualDataset(X_m5[split:], X_h1[split:], y[split:])

    # Class weights
    counts = np.bincount(y[:split], minlength=3).astype(np.float32)
    weights = 1.0 / (counts + 1e-5)
    weights = weights / weights.sum() * 3.0
    class_weights = torch.tensor(weights)

    log.info(f"Train: {len(train_ds):,} | Val: {len(val_ds):,}")
    log.info(f"Class weights: {weights}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    return train_loader, val_loader, class_weights


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_targets, total_loss = [], [], 0.0
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for X_m5, X_h1, y in loader:
            X_m5, X_h1, y = X_m5.to(device), X_h1.to(device), y.to(device)
            logits, _ = model(X_m5, X_h1)
            loss = loss_fn(logits, y)
            total_loss += loss.item() * len(y)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    f1_per = f1_score(all_targets, all_preds, average=None, labels=[0, 1, 2], zero_division=0)
    report = classification_report(all_targets, all_preds,
                                   target_names=["Hold", "Buy", "Sell"], zero_division=0)
    return total_loss / len(loader.dataset), f1_per[1], f1_per[2], report


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    train_loader, val_loader, class_weights = make_dual_split(
        args.h5, batch_size=args.batch_size
    )

    # Get dimensions from first batch
    sample_m5, sample_h1, _ = next(iter(train_loader))
    n_feat_m5 = sample_m5.shape[2]
    n_feat_h1 = sample_h1.shape[2]
    window_m5 = sample_m5.shape[1]
    window_h1 = sample_h1.shape[1]

    log.info(f"M5: {window_m5}×{n_feat_m5} | H1: {window_h1}×{n_feat_h1}")

    model = DualTimeframeTransformer(
        n_features_m5=n_feat_m5,
        n_features_h1=n_feat_h1,
        window_m5=window_m5,
        window_h1=window_h1,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers_m5=args.n_layers_m5,
        n_layers_h1=args.n_layers_h1,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model params: {n_params:,}")

    loss_fn = FocalLoss(gamma=args.focal_gamma, weight=class_weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1 = 0.0
    patience_counter = 0
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_samples = 0

        for X_m5, X_h1, y in train_loader:
            X_m5, X_h1, y = X_m5.to(device), X_h1.to(device), y.to(device)

            optimizer.zero_grad()
            logits, _ = model(X_m5, X_h1)
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * len(y)
            n_samples += len(y)

        scheduler.step()
        train_loss = total_loss / n_samples

        # Validate
        val_loss, f1_buy, f1_sell, report = evaluate(model, val_loader, device)
        f1_avg = (f1_buy + f1_sell) / 2

        log.info(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
            f"F1(B)={f1_buy:.3f} F1(S)={f1_sell:.3f} Avg={f1_avg:.3f}"
        )

        if f1_avg > best_f1:
            best_f1 = f1_avg
            patience_counter = 0
            torch.save(model.state_dict(), CHECKPOINT_DIR / "best_model_dual_bc.pt")
            log.info(f"   ★ Saved best model (F1={best_f1:.3f})")
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            log.info(f"Early stopping at epoch {epoch} (patience={args.patience})")
            break

    log.info(f"\n{'='*50}")
    log.info(f"Best F1(Buy+Sell): {best_f1:.3f}")
    log.info(f"Checkpoint: {CHECKPOINT_DIR / 'best_model_dual_bc.pt'}")

    # Final report
    best_model = DualTimeframeTransformer(
        n_features_m5=n_feat_m5, n_features_h1=n_feat_h1,
        window_m5=window_m5, window_h1=window_h1,
        d_model=args.d_model, n_heads=args.n_heads,
        n_layers_m5=args.n_layers_m5, n_layers_h1=args.n_layers_h1,
    ).to(device)
    best_model.load_state_dict(torch.load(CHECKPOINT_DIR / "best_model_dual_bc.pt", map_location=device))
    _, _, _, report = evaluate(best_model, val_loader, device)
    print(report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers-m5", type=int, default=6)
    parser.add_argument("--n-layers-h1", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
