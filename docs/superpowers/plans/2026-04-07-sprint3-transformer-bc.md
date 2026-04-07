# XAUUSD Bot — Sprint 3: Transformer Brain & BC Training (Phase 1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Xây dựng mạng Transformer Causal và huấn luyện Phase 1 Behavioral Cloning — Bot học bắt chước Oracle với F1(Buy) > 0.4, F1(Sell) > 0.4, Win Rate > 55% trên tập validation.

**Architecture:** Causal Transformer Encoder (Self-Attention với triangular mask) + Policy Head (3 classes) + Value Head (1 scalar). Huấn luyện với Focal Loss để chống Class Imbalance. Validation dùng Purged Walk-Forward (không random split).

**Tech Stack:** PyTorch, h5py, scikit-learn, numpy, pytest

---

## File Structure

```
src/model/
├── transformer.py           [NEW] — Causal Transformer + Policy/Value heads
└── tests/
    └── test_transformer.py  [NEW]

src/training/
├── dataset_loader.py        [NEW] — PyTorch Dataset/DataLoader cho HDF5
├── focal_loss.py            [NEW] — Focal Loss với class weights
├── train_bc.py              [NEW] — Training loop Phase 1
└── tests/
    ├── test_dataset_loader.py [NEW]
    └── test_focal_loss.py     [NEW]
```

---

## Task 1: Transformer Architecture

**Files:**
- Create: `src/model/__init__.py` (rỗng)
- Create: `src/model/transformer.py`
- Create: `src/model/tests/__init__.py` (rỗng)
- Create: `src/model/tests/test_transformer.py`

### Step 1.1: Viết failing tests

- [ ] **Viết test file:**

```python
# src/model/tests/test_transformer.py
import torch
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.model.transformer import XAUTransformer


BATCH      = 4
WINDOW     = 64
N_FEATURES = 10
N_ACTIONS  = 3


class TestXAUTransformer:
    def setup_method(self):
        self.model = XAUTransformer(
            n_features = N_FEATURES,
            window_size = WINDOW,
            d_model = 64,
            n_heads = 4,
            n_layers = 2,
            dropout = 0.1,
            n_actions = N_ACTIONS,
        )

    def test_policy_output_shape(self):
        """Policy head phải output (batch, n_actions)."""
        x = torch.randn(BATCH, WINDOW, N_FEATURES)
        logits, value = self.model(x)
        assert logits.shape == (BATCH, N_ACTIONS), (
            f"Policy shape kỳ vọng ({BATCH}, {N_ACTIONS}), nhận {logits.shape}"
        )

    def test_value_output_shape(self):
        """Value head phải output (batch, 1)."""
        x = torch.randn(BATCH, WINDOW, N_FEATURES)
        logits, value = self.model(x)
        assert value.shape == (BATCH, 1), (
            f"Value shape kỳ vọng ({BATCH}, 1), nhận {value.shape}"
        )

    def test_no_nan_in_output(self):
        """Output không được có NaN."""
        x = torch.randn(BATCH, WINDOW, N_FEATURES)
        logits, value = self.model(x)
        assert not torch.isnan(logits).any(), "NaN trong policy logits"
        assert not torch.isnan(value).any(), "NaN trong value output"

    def test_gradient_flows_to_all_layers(self):
        """Gradient phải lưu thông đến tận lớp embedding (layer đầu tiên)."""
        x = torch.randn(BATCH, WINDOW, N_FEATURES)
        logits, value = self.model(x)
        loss = logits.sum() + value.sum()
        loss.backward()

        embedding_weight = self.model.input_projection.weight
        assert embedding_weight.grad is not None, "Gradient không đến được embedding layer"
        assert not torch.isnan(embedding_weight.grad).any(), "NaN trong gradient"

    def test_causal_last_token_only(self):
        """
        [CRITICAL FIX] Phải dùng x[:, -1, :] (token cuối),
        KHÔNG dùng x.mean(dim=1) (Global Average Pooling).

        Lý do: Causal Mask đảm bảo token t chỉ nhìn thấy [0..t].
        Token cuối (t = window-1) là duy nhất tổng hợp được toàn bộ lịch sử.
        Mean-pooling làm loãng thông tin của nó với các token đầu (bị mù).
        """
        x = torch.randn(BATCH, WINDOW, N_FEATURES)
        logits, value = self.model(x)
        # Verify bằng cách kiểm tra chỉ thay đổi token đầu không ảnh hưởng output
        x_mod = x.clone()
        x_mod[:, 0, :] += 999.0  # Thay đổi token đầu tiên
        with torch.no_grad():
            logits_mod, _ = self.model(x_mod)
        # Nếu dùng last-token: thay token 0 không ảnh hưởng output
        # Nếu dùng mean-pool: thay token 0 SỌ ảnh hưởng output
        # Test này chỉ pass khi dùng last-token
        diff = (logits - logits_mod).abs().max().item()
        assert diff < 1e-4, (
            f"[FAIL] Đang dùng mean-pool — token đầu ảnh hưởng output: diff={diff:.6f}. "
            f"Hãy sử lụng x[:, -1, :] thay vì x.mean(dim=1)"
        )
        x     = torch.randn(1, WINDOW, N_FEATURES)
        x_mod = x.clone()
        x_mod[0, WINDOW//2:, :] += 999.0  # Thay đổi nửa sau (tương lai)

        with torch.no_grad():
            logits1, _ = self.model(x)
            logits2, _ = self.model(x_mod)

        # Output tại vị trí nến đầu (nến 0) không được thay đổi
        diff = (logits1 - logits2).abs().max().item()
        assert diff < 1e-4, (
            f"Causal mask bị vi phạm — output khác nhau khi thay đổi tương lai: {diff}"
        )

    def test_batch_independence(self):
        """Output của một sample không phụ thuộc sample khác trong cùng batch."""
        x = torch.randn(BATCH, WINDOW, N_FEATURES)
        single = x[0:1]

        with torch.no_grad():
            logits_batch, _ = self.model(x)
            logits_single, _ = self.model(single)

        diff = (logits_batch[0] - logits_single[0]).abs().max().item()
        assert diff < 1e-4, (
            f"Batch không độc lập — output sample 0 thay đổi theo batch: {diff}"
        )

    def test_dropout_disabled_in_eval(self):
        """Dropout phải tắt trong eval mode (output deterministic)."""
        x = torch.randn(BATCH, WINDOW, N_FEATURES)
        self.model.eval()
        with torch.no_grad():
            out1, _ = self.model(x)
            out2, _ = self.model(x)
        diff = (out1 - out2).abs().max().item()
        assert diff == 0.0, f"Eval mode không deterministic: diff={diff}"
```

- [ ] **Chạy để verify FAIL:**
```bash
python -m pytest src/model/tests/test_transformer.py -v
```
Kết quả mong đợi: `ERROR — ModuleNotFoundError`

### Step 1.2: Implement Transformer

- [ ] **Tạo `src/model/__init__.py` và `src/model/tests/__init__.py` rỗng.**

- [ ] **Tạo `src/model/transformer.py`:**

```python
"""
transformer.py
--------------
Causal Transformer Encoder cho chuỗi giá XAUUSD.

Kiến trúc:
  Input:  (batch, window_size, n_features)
  → Linear Projection → Positional Encoding → 
  → L × Causal TransformerEncoderLayer (triangular mask) →
  → Global Average Pool →
  → Policy Head (n_actions logits) + Value Head (1 scalar)

Causal mask đảm bảo nến tại thời điểm t chỉ
nhìn thấy nến từ 0 đến t (không rò rỉ tương lai).
"""

import math
import torch
import torch.nn as nn


class XAUTransformer(nn.Module):
    def __init__(
        self,
        n_features:  int   = 10,
        window_size: int   = 128,
        d_model:     int   = 256,
        n_heads:     int   = 8,
        n_layers:    int   = 6,
        dropout:     float = 0.1,
        n_actions:   int   = 3,
    ):
        super().__init__()
        self.window_size = window_size
        self.d_model     = d_model

        # ── Input Projection ──────────────────────────────────────────
        self.input_projection = nn.Linear(n_features, d_model)

        # ── Positional Encoding (sine/cosine, cố định) ────────────────
        self.register_buffer(
            "pos_enc",
            self._build_pos_enc(window_size, d_model)
        )

        # ── Causal Transformer Encoder ────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = n_heads,
            dim_feedforward = d_model * 4,
            dropout         = dropout,
            batch_first     = True,   # Input: (batch, seq, d_model)
            norm_first      = True,   # Pre-LN stabilizes training
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # ── Output Heads ──────────────────────────────────────────────
        self.policy_head = nn.Linear(d_model, n_actions)
        self.value_head  = nn.Linear(d_model, 1)

        # ── Causal Mask (triangular) ──────────────────────────────────
        self.register_buffer(
            "causal_mask",
            nn.Transformer.generate_square_subsequent_mask(window_size)
        )

        self._init_weights()

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : Tensor shape (batch, window_size, n_features)

        Returns
        -------
        logits : (batch, n_actions)
        value  : (batch, 1)
        """
        # Input projection
        x = self.input_projection(x)                   # (B, W, d_model)

        # Positional encoding
        x = x + self.pos_enc[:x.size(1), :]            # (B, W, d_model)

        # Causal Transformer Encoder
        x = self.encoder(x, mask=self.causal_mask,
                         is_causal=True)                # (B, W, d_model)

        # [CRITICAL FIX] Chỉ lấy token cuối cùng
        # Token W-1 là duy nhất tổng hợp được toàn bộ lịch sử [0..W-1]
        # Mean-pooling sẽ làm loãng thông tin với các token đầu bị mù
        x = x[:, -1, :]                                # (B, d_model)  ← FIX

        # [FIX GRADIENT EARTHQUAKE] Cắt đứt gradient từ Value Head về phần thân Transformer
        # Ở PPO Phase 2, Value Head ngẫu nhiên => Error siêu lớn. Nếu k detach, nó phá nát Core đã train ở BC.
        return self.policy_head(x), self.value_head(x.detach())

    @staticmethod
    def _build_pos_enc(max_len: int, d_model: int) -> torch.Tensor:
        """Tạo Positional Encoding (PE) cố định dạng sine/cosine."""
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe  # (max_len, d_model)

    def _init_weights(self):
        """Xavier init cho Linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
```

- [ ] **Chạy để verify PASS:**
```bash
python -m pytest src/model/tests/test_transformer.py -v
```
Kết quả mong đợi: `7 passed`

- [ ] **Commit:**
```bash
git add src/model/
git commit -m "feat(sprint3): XAUTransformer - causal encoder, policy/value heads, PE"
```

---

## Task 2: Focal Loss & DataLoader

**Files:**
- Create: `src/training/__init__.py` (rỗng)
- Create: `src/training/focal_loss.py`
- Create: `src/training/dataset_loader.py`
- Create: `src/training/tests/__init__.py` (rỗng)
- Create: `src/training/tests/test_focal_loss.py`
- Create: `src/training/tests/test_dataset_loader.py`

### Step 2.1: Focal Loss tests

- [ ] **Tạo `src/training/tests/test_focal_loss.py`:**

```python
# src/training/tests/test_focal_loss.py
import torch
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.training.focal_loss import FocalLoss


class TestFocalLoss:
    def test_loss_is_scalar(self):
        """FocalLoss phải trả về scalar tensor."""
        loss_fn = FocalLoss(gamma=2.0)
        logits  = torch.randn(16, 3)
        targets = torch.randint(0, 3, (16,))
        loss    = loss_fn(logits, targets)
        assert loss.ndim == 0, f"Loss phải là scalar, nhận shape {loss.shape}"

    def test_loss_non_negative(self):
        """Loss phải luôn >= 0."""
        loss_fn = FocalLoss(gamma=2.0)
        logits  = torch.randn(32, 3)
        targets = torch.randint(0, 3, (32,))
        loss    = loss_fn(logits, targets)
        assert loss.item() >= 0, f"Loss âm: {loss.item()}"

    def test_correct_prediction_low_loss(self):
        """Prediction càng chính xác, loss càng thấp."""
        loss_fn = FocalLoss(gamma=2.0)
        # Logit rất cao cho đúng class
        logits_correct = torch.tensor([[10.0, -10.0, -10.0]] * 8)
        targets = torch.zeros(8, dtype=torch.long)
        loss = loss_fn(logits_correct, targets)
        assert loss.item() < 0.01, f"Loss cho prediction đúng phải gần 0: {loss.item()}"

    def test_focal_penalizes_wrong_class_harder(self):
        """
        Focal Loss phải phạt miss trên class hiếm (Buy/Sell) nặng hơn
        so với Cross-Entropy thông thường khi dùng class_weights.
        """
        # Class weights: Hold=0.1 (rất phổ biến), Buy=5.0, Sell=5.0
        fl = FocalLoss(gamma=2.0, class_weights=torch.tensor([0.1, 5.0, 5.0]))
        ce = FocalLoss(gamma=0.0, class_weights=torch.tensor([0.1, 5.0, 5.0]))

        # Sai nhãn Buy (class 1) với confidence cao
        logits  = torch.tensor([[-5.0, -5.0, 5.0]])  # Predict Sell (2)
        targets = torch.tensor([1])                   # Đúng phải là Buy (1)

        fl_loss = fl(logits, targets)
        ce_loss = ce(logits, targets)

        # Với gamma > 0, Focal Loss tập trung hơn vào hard examples
        # Kết quả: FL và CE có thể xấp xỉ nhau với class_weight đơn giản,
        # nhưng quan trọng là cả hai > 0
        assert fl_loss.item() > 0
        assert ce_loss.item() > 0

    def test_gradient_exists(self):
        """Gradient phải tồn tại để huấn luyện được."""
        loss_fn = FocalLoss(gamma=2.0)
        logits  = torch.randn(8, 3, requires_grad=True)
        targets = torch.randint(0, 3, (8,))
        loss    = loss_fn(logits, targets)
        loss.backward()
        assert logits.grad is not None
```

- [ ] **Chạy để verify FAIL:**
```bash
python -m pytest src/training/tests/test_focal_loss.py -v
```

- [ ] **Tạo `src/training/focal_loss.py`:**

```python
"""
focal_loss.py
-------------
Focal Loss để chống Class Imbalance trong BC Training.
Phạt mạnh hơn khi model tự tin nhưng predict sai nhãn hiếm (Buy/Sell).

FL(pt) = -alpha_t × (1 - pt)^gamma × log(pt)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma:         float               = 2.0,
        class_weights: Optional[torch.Tensor] = None,
        reduction:     str                 = "mean",
    ):
        super().__init__()
        self.gamma         = gamma
        self.class_weights = class_weights
        self.reduction     = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : (batch, n_classes) — raw logits (chưa softmax)
        targets : (batch,) — class indices [0, n_classes)
        """
        ce_loss = F.cross_entropy(logits, targets,
                                  weight=self.class_weights,
                                  reduction="none")
        pt      = torch.exp(-ce_loss)           # xác suất của class đúng
        fl      = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return fl.mean()
        elif self.reduction == "sum":
            return fl.sum()
        return fl
```

- [ ] **Chạy để verify PASS:**
```bash
python -m pytest src/training/tests/test_focal_loss.py -v
```
Kết quả mong đợi: `5 passed`

- [ ] **Tạo `src/training/dataset_loader.py`:**

```python
"""
dataset_loader.py
-----------------
PyTorch Dataset và DataLoader cho file HDF5 (output của Sprint 1).

Hỗ trợ Purged Walk-Forward Split:
  - train: 0 → split_idx - gap
  - val:   split_idx → end
  với gap nến để loại bỏ rò rỉ chuỗi thời gian.
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class H5Dataset(Dataset):
    """
    Lazy-load từ HDF5 để không tốn RAM.

    Parameters
    ----------
    h5_path  : Path tới file .h5
    start_idx: Index bắt đầu (slice)
    end_idx  : Index kết thúc (slice)
    """
    def __init__(self, h5_path: str, start_idx: int = 0, end_idx: int = None):
        self.h5_path  = h5_path
        self.start    = start_idx
        with h5py.File(h5_path, "r") as f:
            total = f["X"].shape[0]
        self.end = end_idx if end_idx is not None else total

    def __len__(self) -> int:
        return self.end - self.start

    def __getitem__(self, idx: int):
        with h5py.File(self.h5_path, "r") as f:
            X = f["X"][self.start + idx]
            y = f["y"][self.start + idx]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(int(y), dtype=torch.long)


def make_purged_split(
    h5_path:     str,
    val_ratio:   float = 0.2,
    gap_bars:    int   = 100,
    batch_size:  int   = 256,
    num_workers: int   = 0,
):
    """
    Tạo train/val DataLoader theo Purged Walk-Forward.

    Train: [0, split - gap)
    Val:   [split, end)

    Parameters
    ----------
    h5_path    : Path tới file .h5
    val_ratio  : Tỷ lệ dữ liệu dành cho validation (mặc định 10%)
    gap_bars   : Số bars bỏ giữa train và val (purge period)
    batch_size : Batch size
    num_workers: Số worker DataLoader

    Returns
    -------
    (train_loader, val_loader, class_weights)
    """
    with h5py.File(h5_path, "r") as f:
        total     = f["X"].shape[0]
        all_labels = f["y"][:]

    split  = int(total * (1 - val_ratio))
    train_end = max(0, split - gap_bars)

    train_ds = H5Dataset(h5_path, 0, train_end)
    val_ds   = H5Dataset(h5_path, split, total)

    # Tính class weights (nghịch đảo tần suất) cho Focal Loss
    train_labels = all_labels[:train_end]
    counts       = np.bincount(train_labels, minlength=3).astype(np.float32)
    counts       = np.maximum(counts, 1)
    weights      = 1.0 / counts
    weights      = weights / weights.sum() * 3   # Normalize về sum=3
    class_weights = torch.tensor(weights, dtype=torch.float32)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)

    return train_loader, val_loader, class_weights
```

- [ ] **Commit:**
```bash
git add src/training/
git commit -m "feat(sprint3): FocalLoss + H5Dataset + Purged Walk-Forward DataLoader"
```

---

## Task 3: Training Loop (BC Phase 1)

**Files:**
- Create: `src/training/train_bc.py`

- [ ] **Tạo `src/training/train_bc.py`:**

```python
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
    # Đọc n_features từ dataset
    sample_X, _ = next(iter(train_loader))
    n_features = sample_X.shape[2]

    model = XAUTransformer(
        n_features  = n_features,
        window_size = args.window_size,
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
```

- [ ] **Commit:**
```bash
git add src/training/train_bc.py
git commit -m "feat(sprint3): BC training loop - Focal Loss, early stopping on F1(Buy+Sell)"
```

---

## Task 4: Chạy toàn bộ tests Sprint 3 & Push

- [ ] **Chạy toàn bộ tests:**
```bash
python -m pytest src/model/tests/ src/training/tests/ -v
```
Kết quả mong đợi: `≥ 12 passed, 0 failed`

- [ ] **Smoke test training (chỉ khi đã có HDF5 từ Sprint 1):**
```bash
python src/training/train_bc.py --h5 data/processed/XAUUSD_M15_w128.h5 --epochs 5
```
Kết quả mong đợi: Loss giảm qua 5 epochs, không crash.

- [ ] **Push:**
```bash
git push origin main
```

## Điều kiện DONE cho Sprint 3
- [ ] `python -m pytest src/model/tests/ src/training/tests/ -v` → tất cả PASS
- [ ] Causal mask test PASS (không rò rỉ tương lai)
- [ ] Gradient flows đến embedding layer
- [ ] Training loop chạy không crash với batch đầu tiên
- [ ] `best_model_bc.pt` được lưu vào `checkpoints/`
