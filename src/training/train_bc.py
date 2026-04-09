"""
train_bc.py
-----------
Phase 1: Behavioral Cloning Training Loop.

Mục tiêu: F1(Buy) > 0.4, F1(Sell) > 0.4, Win Rate val > 55%.
Dùng Focal Loss + class weights để chống class imbalance.
Early stopping dựa trên F1-macro của Buy+Sell (không phải accuracy).

Cách dùng:
  python src/training/train_bc.py --h5 data/processed/XAUUSD_M15_w128.h5
  python src/training/train_bc.py --h5 data/processed/XAUUSD_M15_w128.h5 --epochs 50 --lr 0.0003
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.optim as optim
from sklearn.metrics import f1_score, classification_report

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.model.transformer import XAUTransformer
from src.training.focal_loss import FocalLoss
from src.training.dataset_loader import make_purged_split

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

CHECKPOINT_DIR = Path("checkpoints")

# Tham số mặc định (cân chỉnh theo GPU server)
DEFAULTS = dict(
    window_size = 128,
    d_model     = 256,
    n_heads     = 8,
    n_layers    = 6,
    dropout     = 0.1,
    epochs      = 100,
    lr          = 3e-4,
    batch_size  = 512,
    patience    = 10,     # early stopping
    focal_gamma = 2.0,
)


def evaluate(model, loader, device):
    """Đánh giá model trên một DataLoader, trả về (loss, f1_buy, f1_sell, report)."""
    model.eval()
    all_preds, all_targets, total_loss = [], [], 0.0
    loss_fn = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits, _ = model(X)
            loss = loss_fn(logits, y)
            total_loss += loss.item() * len(y)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    f1_per_class = f1_score(all_targets, all_preds, average=None,
                            labels=[0, 1, 2], zero_division=0)
    report = classification_report(all_targets, all_preds,
                                   target_names=["Hold", "Buy", "Sell"],
                                   zero_division=0)
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, f1_per_class[1], f1_per_class[2], report


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    log.info(f"H5 file: {args.h5}")

    # ── Data ──────────────────────────────────────────────────────────
    train_loader, val_loader, class_weights = make_purged_split(
        h5_path    = args.h5,
        val_ratio  = 0.2,
        gap_bars   = 200,
        batch_size = args.batch_size,
    )
    log.info(f"Train: {len(train_loader.dataset):,} samples")
    log.info(f"Val:   {len(val_loader.dataset):,} samples")
    log.info(f"Class weights: Hold={class_weights[0]:.3f}, "
             f"Buy={class_weights[1]:.3f}, Sell={class_weights[2]:.3f}")

    # ── Model ─────────────────────────────────────────────────────────
    # Đọc n_features & window_size trực tiếp từ dataset để chống Lỗi Size Mismatch
    sample_X, _ = next(iter(train_loader))
    window_size = sample_X.shape[1]
    n_features  = sample_X.shape[2]

    model = XAUTransformer(
        n_features  = n_features,
        window_size = window_size,
        d_model     = args.d_model,
        n_heads     = args.n_heads,
        n_layers    = args.n_layers,
        dropout     = args.dropout,
        n_actions   = 3,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model params: {total_params:,}")

    # ── Loss & Optimizer ──────────────────────────────────────────────
    loss_fn   = FocalLoss(gamma=args.focal_gamma,
                          class_weights=class_weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Training Loop ─────────────────────────────────────────────────
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    best_f1    = 0.0
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(X)
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(y)

        scheduler.step()
        avg_train_loss = epoch_loss / len(train_loader.dataset)

        # Đánh giá validation mỗi 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            val_loss, f1_buy, f1_sell, report = evaluate(model, val_loader, device)
            f1_trade = (f1_buy + f1_sell) / 2.0

            log.info(
                f"Epoch {epoch:3d}/{args.epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"F1(Buy): {f1_buy:.3f} | F1(Sell): {f1_sell:.3f}"
            )

            if f1_trade > best_f1:
                best_f1    = f1_trade
                no_improve = 0
                ckpt_path  = CHECKPOINT_DIR / "best_model_bc.pt"
                torch.save({
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "f1_buy":      f1_buy,
                    "f1_sell":     f1_sell,
                    "f1_trade":    f1_trade,
                }, ckpt_path)
                log.info(f"  ✅ New best F1(trade)={best_f1:.3f} → saved {ckpt_path}")
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    log.info(f"Early stopping tại epoch {epoch}")
                    break

    # ── Kết quả cuối ─────────────────────────────────────────────────
    log.info(f"\nBest F1(Buy+Sell)/2 = {best_f1:.3f}")
    if best_f1 >= 0.4:
        log.info("✅ PASS: Đủ điều kiện chuyển sang Phase 2 (RL)")
    else:
        log.warning("❌ FAIL: F1 chưa đủ, cần điều chỉnh hyperparameter")

    # In classification report lần cuối
    _, _, _, report = evaluate(model, val_loader, device)
    log.info(f"\n{report}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--h5",          required=True)
    p.add_argument("--window-size", type=int,   default=DEFAULTS["window_size"], dest="window_size")
    p.add_argument("--d-model",     type=int,   default=DEFAULTS["d_model"],     dest="d_model")
    p.add_argument("--n-heads",     type=int,   default=DEFAULTS["n_heads"],     dest="n_heads")
    p.add_argument("--n-layers",    type=int,   default=DEFAULTS["n_layers"],    dest="n_layers")
    p.add_argument("--dropout",     type=float, default=DEFAULTS["dropout"])
    p.add_argument("--epochs",      type=int,   default=DEFAULTS["epochs"])
    p.add_argument("--lr",          type=float, default=DEFAULTS["lr"])
    p.add_argument("--batch-size",  type=int,   default=DEFAULTS["batch_size"],  dest="batch_size")
    p.add_argument("--patience",    type=int,   default=DEFAULTS["patience"])
    p.add_argument("--focal-gamma", type=float, default=DEFAULTS["focal_gamma"], dest="focal_gamma")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())