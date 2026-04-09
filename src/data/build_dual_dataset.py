"""
build_dual_dataset.py
---------------------
Tạo Dual-Timeframe HDF5 dataset chứa cả M5 và H1 features.

HDF5 format:
  X_m5     : (N, 256, 13)  — M5 features windows
  X_h1     : (N, 64, 13)   — H1 features windows (aligned với M5)
  y        : (N,)          — Labels
  close    : (N,)          — Close prices (M5)

Cách dùng:
  python src/data/build_dual_dataset.py --m5 data/raw/XAUUSD_M5_*.csv --h1 data/raw/XAUUSD_H1_*.csv
"""

import argparse
import logging
import sys
import glob
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.data_processor import DataProcessor
from src.data.oracle import Oracle

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/processed")


def load_csv(path_pattern: str) -> pd.DataFrame:
    files = glob.glob(path_pattern)
    if not files:
        raise FileNotFoundError(f"Không tìm thấy file: {path_pattern}")
    df = pd.concat([pd.read_csv(f, index_col="datetime", parse_dates=True) for f in files])
    df = df.sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    log.info(f"  Loaded: {len(df):,} bars từ {df.index[0]} → {df.index[-1]}")
    return df


def build_dual_dataset(
    m5_csv: str,
    h1_csv: str,
    window_m5: int = 256,
    window_h1: int = 64,
    atr_period: int = 14,
):
    """Tạo dual-TF dataset: M5 cho entry, H1 cho trend context."""

    log.info("=" * 60)
    log.info("  DUAL TIMEFRAME DATASET BUILDER")
    log.info("=" * 60)

    # ── 1. Load & process M5 ──
    log.info("\n📊 1. Processing M5...")
    df_m5 = load_csv(m5_csv)
    proc = DataProcessor(atr_period=atr_period)
    feat_m5 = proc.compute_features(df_m5)
    log.info(f"   M5 features: {feat_m5.shape}")

    # ── 2. Load & process H1 ──
    log.info("\n📊 2. Processing H1...")
    df_h1 = load_csv(h1_csv)
    feat_h1 = proc.compute_features(df_h1)
    log.info(f"   H1 features: {feat_h1.shape}")

    # ── 3. Oracle labels (M5 level) ──
    log.info("\n📊 3. Oracle labeling (M5)...")
    atr_abs = (feat_m5["atr_norm"] / 1000.0) * df_m5.loc[feat_m5.index, "close"]
    oracle_df = df_m5.loc[feat_m5.index, ["close", "high", "low"]]
    oracle = Oracle(tp_atr_mult=1.5, sl_atr_mult=1.0, max_hold_bars=144)
    labels = oracle.label(oracle_df, atr_abs)

    counts = labels.value_counts().sort_index()
    total = len(labels)
    log.info(f"   Labels: Hold={counts.get(0,0)/total:.1%}, Buy={counts.get(1,0)/total:.1%}, Sell={counts.get(2,0)/total:.1%}")

    # ── 4. Align H1 to M5 ──
    # Cho mỗi nến M5, tìm nến H1 gần nhất trước đó (floor)
    log.info("\n📊 4. Aligning H1 → M5...")
    h1_arr = feat_h1.values.astype(np.float32)
    h1_index = feat_h1.index

    m5_arr = feat_m5.values.astype(np.float32)
    m5_index = feat_m5.index
    close_arr = df_m5.loc[m5_index, "close"].values.astype(np.float32)
    label_arr = labels.values.astype(np.int8)

    # Tạo mapping: mỗi M5 index → H1 index tương ứng
    h1_idx_map = np.searchsorted(h1_index, m5_index, side="right") - 1
    h1_idx_map = np.clip(h1_idx_map, 0, len(h1_arr) - 1)

    log.info(f"   H1 aligned to M5: {len(h1_idx_map)} mappings")

    # ── 5. Build sliding windows ──
    log.info("\n📊 5. Building sliding windows...")

    n_rows = len(m5_arr)
    n_windows = n_rows - window_m5
    if n_windows <= 0:
        raise ValueError(f"Không đủ data: {n_rows} rows, window={window_m5}")

    # M5 windows (standard sliding window)
    X_m5 = np.lib.stride_tricks.sliding_window_view(
        m5_arr, window_shape=(window_m5, m5_arr.shape[1])
    )
    X_m5 = X_m5[1:, 0, :, :]  # (n_windows, window_m5, n_features)

    # H1 windows (aligned to M5)
    # Cho mỗi M5 window kết thúc tại bar i, lấy 64 bars H1 kết thúc trước bar i
    X_h1_list = []
    valid_mask = []

    for i in range(window_m5, n_rows):
        h1_end_idx = h1_idx_map[i]  # H1 bar tương ứng với M5 bar i
        h1_start_idx = h1_end_idx - window_h1 + 1

        if h1_start_idx < 0:
            # Không đủ H1 bars → zero-pad
            pad_size = -h1_start_idx
            h1_window = np.zeros((window_h1, h1_arr.shape[1]), dtype=np.float32)
            h1_window[pad_size:] = h1_arr[0:h1_end_idx + 1]
            valid_mask.append(h1_end_idx >= 0)
        else:
            h1_window = h1_arr[h1_start_idx:h1_end_idx + 1]
            if len(h1_window) < window_h1:
                pad = np.zeros((window_h1 - len(h1_window), h1_arr.shape[1]), dtype=np.float32)
                h1_window = np.vstack([pad, h1_window])
            valid_mask.append(True)

        X_h1_list.append(h1_window)

    X_h1 = np.array(X_h1_list, dtype=np.float32)  # (n_windows, window_h1, n_features_h1)
    y = label_arr[window_m5:]
    close_out = close_arr[window_m5:]

    log.info(f"   X_m5: {X_m5.shape}")
    log.info(f"   X_h1: {X_h1.shape}")
    log.info(f"   y:    {y.shape}")
    log.info(f"   Valid H1 windows: {sum(valid_mask)}/{len(valid_mask)}")

    # ── 6. Save HDF5 ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"XAUUSD_DUAL_M5w{window_m5}_H1w{window_h1}.h5"

    with h5py.File(out_path, "w") as f:
        f.create_dataset("X_m5",  data=X_m5,      compression="gzip", compression_opts=4)
        f.create_dataset("X_h1",  data=X_h1,      compression="gzip", compression_opts=4)
        f.create_dataset("y",     data=y,          compression="gzip", compression_opts=4)
        f.create_dataset("close", data=close_out,  compression="gzip", compression_opts=4)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    log.info(f"\n✅ Saved: {out_path}  ({size_mb:.1f} MB)")
    log.info(f"   X_m5={X_m5.shape}, X_h1={X_h1.shape}, y={y.shape}")


def main():
    parser = argparse.ArgumentParser(description="Build Dual-TF dataset (M5 + H1)")
    parser.add_argument("--m5", required=True, help="Glob path tới CSV M5")
    parser.add_argument("--h1", required=True, help="Glob path tới CSV H1")
    parser.add_argument("--window-m5", type=int, default=256)
    parser.add_argument("--window-h1", type=int, default=64)
    args = parser.parse_args()

    build_dual_dataset(args.m5, args.h1, args.window_m5, args.window_h1)


if __name__ == "__main__":
    main()
