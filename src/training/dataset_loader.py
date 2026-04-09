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