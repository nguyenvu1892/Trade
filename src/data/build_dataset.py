"""
build_dataset.py
----------------
Pipeline runner cho Sprint 1.
Đọc CSV thô → DataProcessor → Oracle → DatasetBuilder → HDF5.

Cách dùng:
  python src/data/build_dataset.py --m15 data/raw/XAUUSD_M15_*.csv
  python src/data/build_dataset.py --m15 data/raw/XAUUSD_M15_*.csv --h1 data/raw/XAUUSD_H1_*.csv
"""

import argparse
import logging
import sys
from pathlib import Path
import glob

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.data_processor import DataProcessor
from src.data.oracle import Oracle
from src.data.dataset_builder import DatasetBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/processed")


def load_csv(path_pattern: str) -> pd.DataFrame:
    """Nạp CSV (hỗ trợ glob pattern), parse datetime, sort theo thời gian."""
    files = glob.glob(path_pattern)
    if not files:
        raise FileNotFoundError(f"Không tìm thấy file: {path_pattern}")
    df = pd.concat([pd.read_csv(f, index_col="datetime", parse_dates=True) for f in files])
    df = df.sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    log.info(f"  Loaded: {len(df):,} bars từ {df.index[0]} → {df.index[-1]}")
    return df


def run_pipeline(
    csv_path:    str,
    tf_name:     str,
    window_size: int  = 128,
    atr_period:  int  = 14,
    tp_mult:     float = 1.5,
    sl_mult:     float = 1.0,
    max_hold:    int  = 48,
) -> None:
    log.info(f"\n{'='*60}")
    log.info(f"  PIPELINE: {tf_name}")
    log.info(f"{'='*60}")

    # 1. Load
    log.info("Step 1: Load CSV...")
    df = load_csv(csv_path)

    # 2. Feature Engineering
    log.info("Step 2: DataProcessor (Log Returns + ATR + Sine/Cosine)...")
    proc     = DataProcessor(atr_period=atr_period)
    features = proc.compute_features(df)
    log.info(f"  Features shape: {features.shape}")

    # 3. ATR cho Oracle (dùng cột atr_norm * close để ra ATR tuyệt đối USD)
    # LƯU Ý: atr_norm đã được nhân 1000 lúc tạo, nên phải chia lại 1000 để Oracle có ATR chuẩn vị trí giá!
    atr_abs = (features["atr_norm"] / 1000.0) * df.loc[features.index, "close"]

    # 4. Oracle labeling
    log.info("Step 3: Oracle (Triple Barrier Method)...")
    oracle_df = df.loc[features.index, ["close", "high", "low"]]
    oracle    = Oracle(tp_atr_mult=tp_mult, sl_atr_mult=sl_mult, max_hold_bars=max_hold)
    labels    = oracle.label(oracle_df, atr_abs)

    # Log phân phối nhãn
    counts = labels.value_counts().sort_index()
    total  = len(labels)
    log.info(f"  Label distribution: "
             f"Hold={counts.get(0,0)/total:.1%}, "
             f"Buy={counts.get(1,0)/total:.1%}, "
             f"Sell={counts.get(2,0)/total:.1%}")

    # 5. Build HDF5
    log.info("Step 4: DatasetBuilder → HDF5...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"XAUUSD_{tf_name}_w{window_size}.h5"
    builder  = DatasetBuilder(window_size=window_size)

    # Lấy giá gốc để truyền vào simulator
    close_prices = df.loc[features.index, "close"]
    open_next    = df.loc[features.index, "open"].shift(-1).bfill()

    builder.build(
        features=features,
        labels=labels,
        out_path=out_path,
        close_prices=close_prices,
        open_next_prices=open_next
    )
    log.info(f"  ✅ Dataset saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Build XAUUSD training dataset")
    parser.add_argument("--m5",  help="Glob path tới file CSV M5")
    parser.add_argument("--m15", help="Glob path tới file CSV M15")
    parser.add_argument("--h1",  help="Glob path tới file CSV H1")
    parser.add_argument("--window-size", type=int, default=128)
    args = parser.parse_args()

    if not args.m5 and not args.m15 and not args.h1:
        parser.error("Cần ít nhất --m5, --m15 hoặc --h1")

    if args.m5:
        run_pipeline(args.m5, "M5", window_size=args.window_size,
                     max_hold=24)  # 24 nến M5 = 2 giờ
    if args.m15:
        run_pipeline(args.m15, "M15", window_size=args.window_size,
                     max_hold=48)   # 48 nến M15 = 12 giờ
    if args.h1:
        run_pipeline(args.h1,  "H1",  window_size=args.window_size,
                     max_hold=12)   # 12 nến H1 = 12 giờ

    log.info("\n🎉 Sprint 1 hoàn tất! Dataset sẵn sàng cho Sprint 3 (BC Training).")


if __name__ == "__main__":
    main()
