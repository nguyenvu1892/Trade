"""
dataset_builder.py
------------------
Đóng gói Feature Tensor (output DataProcessor) và Labels (output Oracle)
thành file HDF5 dạng Sliding Window sẵn sàng cho PyTorch DataLoader.

Format HDF5 (3 datasets):
  X     : float32 array, shape (N, window_size, n_features)  — features đưa vào model
  y     : int8    array, shape (N,)                           — nhãn Oracle (0/1/2)
  close : float32 array, shape (N,)                           — giá đóng gốc (USD)
                                                                 cho Simulator tính PnL
"""

import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class DatasetBuilder:
    """
    Parameters
    ----------
    window_size : int — Số nến trong mỗi cửa sổ trượt (mặc định 128)
    """

    def __init__(self, window_size: int = 128):
        self.window_size = window_size

    def build(
        self,
        features:         pd.DataFrame,
        labels:           pd.Series,
        out_path:         Path,
        close_prices:     pd.Series,
        open_next_prices: pd.Series = None,   # [CRITICAL FIX]
    ) -> None:
        """
        Tạo file HDF5 từ features, labels, và close_prices.

        Parameters
        ----------
        features          : DataFrame output của DataProcessor
        labels            : Series output của Oracle
        out_path          : Path lưu file .h5
        close_prices      : Series giá đóng gốc (chưa normalize) — vị trí đóng lệnh
        open_next_prices  : Series giá Mở của nến KẸ TIẼP (shift -1) —
                            [FIX LOOKAHEAD] Env dùng làm giá khớp lệnh thực tế.
        """
        # Align tất cả theo index
        features, labels = features.align(labels, join="inner", axis=0)
        close_prices     = close_prices.reindex(features.index)
        if open_next_prices is not None:
            open_next_prices = open_next_prices.reindex(features.index)
        else:
            open_next_prices = close_prices.copy() # fallback for tests that don't pass it

        feat_array      = features.to_numpy(dtype=np.float32)
        label_array     = labels.to_numpy(dtype=np.int8)
        close_array     = close_prices.to_numpy(dtype=np.float32)
        open_next_array = open_next_prices.to_numpy(dtype=np.float32)
        n_rows, n_features = feat_array.shape
        win = self.window_size

        n_windows = n_rows - win
        if n_windows <= 0:
            raise ValueError(
                f"Không đủ dữ liệu để tạo window. "
                f"n_rows={n_rows}, window_size={win}"
            )

        log.info(f"Building dataset: {n_windows} windows \u00d7 {win} bars \u00d7 {n_features} features")

        # ── Tạo tensor Sliding Window ─────────────────────────────────
        X = np.lib.stride_tricks.sliding_window_view(
            feat_array, window_shape=(win, n_features)
        )
        X = X[1:, 0, :, :]                 # [FIX DIMENSION CLASH] Bỏ cửa sổ đầu tiên để X có độ dài khớp với y
        y = label_array[win:]              # nhãn của nến cuối mỗi cửa sổ

        # [FIX] Giá đóng của nến cuối (exit price)
        close_out     = close_array[win:]      # shape: (n_windows,)
        # [FIX LOOKAHEAD] Giá open của nến kế tiếp (entry price thực tế)
        # Khi model quan sát nến t, lệnh sẽ vào tại open(t+1)
        open_next_out = open_next_array[win:]  # shape: (n_windows,)

        # ── Ghi HDF5 (4 datasets) ─────────────────────────────────
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(out_path, "w") as f:
            f.create_dataset("X",         data=X,            compression="gzip", compression_opts=4)
            f.create_dataset("y",         data=y,            compression="gzip", compression_opts=4)
            f.create_dataset("close",     data=close_out,    compression="gzip", compression_opts=4)
            f.create_dataset("open_next", data=open_next_out,compression="gzip", compression_opts=4)

        size_mb = out_path.stat().st_size / (1024 * 1024)
        log.info(f"\u2705 Saved: {out_path}  (X={X.shape}, y={y.shape}, {size_mb:.1f} MB)")
