"""
train_cme_transfer.py
---------------------
Thực thi Phẫu thuật Mạng (Network Surgery) trên `best_model_bc.pt`.
1. Khởi tạo XAUTransformer(15 features).
2. Sắp xếp lại trọng số từ checkpoint 13 features.
3. Đóng băng (Freeze) các lớp Encoder.
4. Fine-Tune lớp Linear(15, d_model) và Heads với Focal Loss.
"""

import sys
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

# Add root to pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model.transformer import XAUTransformer
from src.training.dataset_loader import H5Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

def run_transfer_learning(
    h5_path: str = "data/processed/XAUUSD_M5_w256.h5",
    old_ckpt_path: str = "checkpoints/best_model_bc.pt",
    new_ckpt_path: str = "models/best_model_cme_sniper.pt",
    batch_size: int = 256,
    epochs: int = 15,
    lr: float = 1e-4,
):
    Path(new_ckpt_path).parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # ==================================================
    # 1. Khởi tạo Mô Hình (15 Features)
    # ==================================================
    model = XAUTransformer(n_features=15, window_size=256, d_model=256, n_heads=8, n_layers=6)

    # ==================================================
    # 2. Network Surgery (Phẫu thuật trọng số)
    # ==================================================
    log.info(f"Loading weights from {old_ckpt_path}...")
    checkpoint = torch.load(old_ckpt_path, map_location="cpu", weights_only=False)
    
    if isinstance(checkpoint, dict):
        if "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Xử lý lớp input_projection.weight (Cũ: 256x13, Mới: 256x15)
    old_proj_weight = state_dict["input_projection.weight"]
    new_proj_weight = model.input_projection.weight.data.clone()

    if old_proj_weight.shape[1] == 13:
        # Bê đắp 13 tính năng cũ vào đúng vị trí
        new_proj_weight[:, :13] = old_proj_weight
        # 2 tính năng mới (VWAP, Surge) sẽ dùng Random Init (đã có ở new_proj_weight)
        state_dict["input_projection.weight"] = new_proj_weight
        log.info("Network Surgery: Expanded input_projection from 13 to 15 features.")
    else:
        log.warning("Checkpoint input was not 13 features! Pls check.")

    # Load vào mô hình (strict=True)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)

    # ==================================================
    # 3. Freeze & Unfreeze
    # ==================================================
    # Đóng băng Encoder để chống Overfitting trên data nhỏ
    for param in model.encoder.parameters():
        param.requires_grad = False
    
    # Chỉ mở khóa Input Projection và Heads
    for param in model.input_projection.parameters():
        param.requires_grad = True
    for param in model.policy_head.parameters():
        param.requires_grad = True
    for param in model.value_head.parameters():
        param.requires_grad = True

    # ==================================================
    # 4. DataLoader & Focal Loss + WeightedRandomSampler
    # ==================================================
    dataset = H5Dataset(h5_path)
    
    import h5py
    with h5py.File(h5_path, "r") as f:
        labels = f["y"][:]
        
    counts = {0: 0, 1: 0, 2: 0}
    for l in labels: counts[l] += 1
    total = sum(counts.values())
    
    log.info(f"Label dist: HOLD={counts[0]/total:.1%}, BUY={counts[1]/total:.1%}, SELL={counts[2]/total:.1%}")
    
    # Trọng số lấy mẫu để ưu tiên lấy BUY/SELL (nếu mất cân bằng)
    class_weights = {0: 1.0/counts[0] if counts[0] > 0 else 0, 
                     1: 1.0/counts[1] if counts[1] > 0 else 0, 
                     2: 1.0/counts[2] if counts[2] > 0 else 0}
    sample_weights = [class_weights[l] for l in labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(labels), replacement=True)
    
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # ==================================================
    # 5. Training Loop
    # ==================================================
    log.info("Starting Transfer Learning...")
    best_loss = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device).long()
            
            optimizer.zero_grad()
            logits, value = model(x)
            loss = criterion(logits, y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)
        log.info(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            try:
                # Đảm bảo không lưu cả optimizer state để file gọn
                torch.save({"model_state_dict": model.state_dict()}, new_ckpt_path)
                log.info(f"   [+] Saved Best Sniper Checkpoint (Loss {best_loss:.4f})")
            except Exception as e:
                log.error(f"Khong the luu model: {e}")

    log.info("Xong Transfer Learning! Sẵn sàng gỡ gRPC.")

if __name__ == "__main__":
    run_transfer_learning()
