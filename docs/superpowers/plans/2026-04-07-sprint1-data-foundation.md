# XAUUSD Bot — Sprint 1: Data Foundation & Oracle

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Xây dựng pipeline dữ liệu hoàn chỉnh: từ CSV thô → Feature Tensor đã chuẩn hóa → nhãn Triple Barrier → file HDF5 sẵn sàng cho Phase 1 BC training.

**Architecture:** Data Processor chuyển OHLCV sang Log Returns + Sine/Cosine time encoding. Oracle dùng Triple Barrier Method để gắn nhãn chỉ khi giá chạm TP trước SL. Dataset Builder đóng gói sliding window thành HDF5. Toàn bộ pipeline áp dụng TDD — test trước, code sau.

**Tech Stack:** Python 3.10+, pandas, numpy, h5py, ta (technical analysis), pytest

---

## File Structure

```
src/data/
├── download_mt5.py          ✅ DONE (Sprint 0)
├── data_processor.py        [NEW] — Log returns, ATR, Sine/Cosine
├── oracle.py                [NEW] — Triple Barrier Method labeler
├── dataset_builder.py       [NEW] — Sliding window → HDF5
├── tests/
│   ├── test_download_mt5.py ✅ DONE
│   ├── test_data_processor.py [NEW]
│   ├── test_oracle.py         [NEW]
│   └── test_dataset_builder.py [NEW]
data/
└── raw/                     — CSV files từ MT5 (input)
data/
└── processed/               — HDF5 dataset (output)
```

---

## Task 1: DataProcessor — Khử phi dừng & Feature Engineering

**Files:**
- Create: `src/data/data_processor.py`
- Create: `src/data/tests/test_data_processor.py`

### Step 1.1: Viết failing tests cho DataProcessor

- [ ] **Viết test file**

```python
# src/data/tests/test_data_processor.py
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.data.data_processor import DataProcessor


def _make_ohlcv(n: int = 200) -> pd.DataFrame:
    """Tạo DataFrame OHLCV giả với index datetime UTC."""
    idx = pd.date_range("2020-01-01", periods=n, freq="15min", tz="UTC")
    np.random.seed(42)
    close = 1800.0 + np.cumsum(np.random.randn(n))
    df = pd.DataFrame({
        "open":        close + np.random.uniform(-1, 1, n),
        "high":        close + np.random.uniform(0, 2, n),
        "low":         close - np.random.uniform(0, 2, n),
        "close":       close,
        "tick_volume": np.random.randint(100, 5000, n),
    }, index=idx)
    return df


class TestLogReturns:
    def test_log_returns_no_nan_after_warmup(self):
        """Log returns không có NaN sau các nến đầu (warmup period)."""
        df = _make_ohlcv(200)
        proc = DataProcessor(atr_period=14)
        result = proc.compute_features(df)
        # Sau warmup (14 nến), không có NaN trong log_close
        assert result["log_close"].iloc[14:].isnull().sum() == 0

    def test_log_returns_stationary(self):
        """Log returns phải nằm trong khoảng hợp lý [-0.1, 0.1] cho XAUUSD."""
        df = _make_ohlcv(500)
        proc = DataProcessor(atr_period=14)
        result = proc.compute_features(df)
        log_ret = result["log_close"].dropna()
        assert log_ret.abs().max() < 0.1, (
            f"Log return bất thường: {log_ret.abs().max():.4f}"
        )

    def test_output_has_required_columns(self):
        """Output DataFrame phải có đủ 12 cột feature bắt buộc."""
        df = _make_ohlcv(200)
        proc = DataProcessor(atr_period=14)
        result = proc.compute_features(df)
        required = {
            "log_open", "log_high", "log_low", "log_close", "log_volume",
            "atr_norm",  # scaled ×1000
            "hour_sin", "hour_cos", "dow_sin", "dow_cos",
            "is_us_session",  # [NEW] binary flag
            "is_weekend",     # [NEW] binary flag
        }
        missing = required - set(result.columns)
        assert not missing, f"Thiếu các cột: {missing}"

    def test_atr_norm_scaled(self):
        """ATR phải được scale ×1000 — giá trị nằm quanh 0.5–5.0, không phải 0.0005."""
        df = _make_ohlcv(200)
        proc = DataProcessor(atr_period=14)
        result = proc.compute_features(df)
        atr = result["atr_norm"].dropna()
        # ATR/price ≈ 0.001 → ×1000 ≈ 1.0
        assert atr.mean() > 0.1, (
            f"ATR_norm quá nhỏ ({atr.mean():.6f}) — quên nhân ×1000?"
        )

    def test_is_us_session_binary(self):
        """is_us_session phải là 0 hoặc 1."""
        df = _make_ohlcv(200)
        proc = DataProcessor(atr_period=14)
        result = proc.compute_features(df)
        vals = result["is_us_session"].unique()
        assert set(vals).issubset({0, 1}), f"is_us_session không phải binary: {vals}"

    def test_sine_cosine_range(self):
        """Sine/Cosine encoding phải nằm trong [-1, 1]."""
        df = _make_ohlcv(200)
        proc = DataProcessor(atr_period=14)
        result = proc.compute_features(df)
        for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]:
            assert result[col].min() >= -1.0 - 1e-9
            assert result[col].max() <= 1.0 + 1e-9, f"{col} vượt quá [-1,1]"

    def test_atr_norm_positive(self):
        """ATR chuẩn hóa phải luôn dương."""
        df = _make_ohlcv(200)
        proc = DataProcessor(atr_period=14)
        result = proc.compute_features(df)
        atr = result["atr_norm"].dropna()
        assert (atr > 0).all(), "ATR phải luôn > 0"

    def test_weekend_gap_capped(self):
        """
        Gap cuối tuần (XAUUSD tăng +5% sau weekend) phải bị cap
        — log return không được vượt ngưỡng [-0.05, 0.05].
        """
        # Giả lập gap: nến T2 mở cửa tăng 5% so với nến đóng T6
        idx = pd.date_range("2020-01-03 21:00", periods=5, freq="15min", tz="UTC")
        # Thêm nến T2 sau 60 giờ ngắt quãng
        idx_weekend = pd.DatetimeIndex(
            list(idx) + [pd.Timestamp("2020-01-06 21:00", tz="UTC")]
        )
        close = np.array([1800.0, 1801, 1802, 1803, 1804, 1894.0])  # +5% gap
        df = pd.DataFrame({
            "open": close + 1, "high": close + 2,
            "low": close - 1, "close": close,
            "tick_volume": [1000] * 6,
        }, index=idx_weekend)
        proc = DataProcessor(atr_period=3)
        result = proc.compute_features(df)
        # Log return của nến gap không được vượt 0.05
        assert result["log_close"].abs().max() <= 0.051, (
            f"Gap log return chưa được cap: {result['log_close'].abs().max():.4f}"
        )

    def test_is_gap_flag_set_after_weekend(self):
        """
        is_gap phải bằng 1.0 cho nến ngay sau gap thời gian > 15 phút.
        """
        idx = pd.date_range("2020-01-03 21:00", periods=5, freq="15min", tz="UTC")
        idx_with_gap = pd.DatetimeIndex(
            list(idx) + [pd.Timestamp("2020-01-06 21:00", tz="UTC")]
        )
        close = np.array([1800.0, 1801, 1802, 1803, 1804, 1850.0])
        df = pd.DataFrame({
            "open": close + 1, "high": close + 2,
            "low": close - 1, "close": close,
            "tick_volume": [1000] * 6,
        }, index=idx_with_gap)
        proc = DataProcessor(atr_period=3)
        result = proc.compute_features(df)
        # Nến cuối (sau gap 60h) phải có is_gap=1
        assert result["is_gap"].iloc[-1] == 1.0, (
            f"is_gap phải là 1 sau weekend gap, nhận: {result['is_gap'].iloc[-1]}"
        )

    def test_warmup_rows_dropped(self):
        """DataProcessor phải tự động bỏ các hàng warmup (NaN) khỏi output."""
        df = _make_ohlcv(200)
        proc = DataProcessor(atr_period=14)
        result = proc.compute_features(df)
        # Không được có bất kỳ NaN nào trong output cuối cùng
        assert result.isnull().sum().sum() == 0, (
            f"Còn NaN trong output:\n{result.isnull().sum()}"
        )

    def test_reproducible_output(self):
        """Cùng input phải cho cùng output (deterministic)."""
        df = _make_ohlcv(200)
        proc = DataProcessor(atr_period=14)
        r1 = proc.compute_features(df)
        r2 = proc.compute_features(df)
        pd.testing.assert_frame_equal(r1, r2)
```

- [ ] **Chạy để verify FAIL:**
```bash
python -m pytest src/data/tests/test_data_processor.py -v
```
Kết quả mong đợi: `ERROR — ModuleNotFoundError: No module named 'src.data.data_processor'`

### Step 1.2: Implement DataProcessor

- [ ] **Tạo file `src/data/data_processor.py`:**

```python
"""
data_processor.py
-----------------
Chuyển đổi dữ liệu OHLCV thô thành Feature Tensor đã chuẩn hóa.

Transformations:
  - Log Returns (khử phi dừng)
  - ATR chuẩn hóa (đo volatility)
  - Sine/Cosine time encoding (nhúng ngữ cảnh thời gian)
"""

import numpy as np
import pandas as pd


class DataProcessor:
    """
    Parameters
    ----------
    atr_period : int
        Số nến để tính ATR (mặc định 14).
    """

    def __init__(self, atr_period: int = 14):
        self.atr_period = atr_period

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Nhận DataFrame OHLCV với DatetimeIndex UTC.
        Trả về DataFrame features đã chuẩn hóa, không có NaN.

        Parameters
        ----------
        df : pd.DataFrame
            Cột bắt buộc: open, high, low, close, tick_volume
            Index: DatetimeIndex với timezone UTC

        Returns
        -------
        pd.DataFrame với các cột feature (xem REQUIRED_COLS)
        """
        df = df.copy()
        result = pd.DataFrame(index=df.index)

        # ── Log Returns với Gap Detection & Capping ──────────────────
        # Phát hiện gap thời gian: nến T2 nối nến T6 → khoảng cách > 15 phút
        time_delta       = df.index.to_series().diff().dt.total_seconds() / 60  # phút
        gap_mask         = (time_delta > 15).fillna(False)  # True nến ngay sau gap
        result["is_gap"] = gap_mask.astype(np.float32)

        for col in ["open", "high", "low", "close"]:
            raw_log = np.log(df[col] / df["close"].shift(1))
            # Cap [-0.05, 0.05] — tương đương 5% mỗi nến, đủ bắt GAP mà không gây outlier
            result[f"log_{col}"] = raw_log.clip(-0.05, 0.05)

        # Volume: log(vol_t / vol_{t-1}), clip để tránh log(0)
        vol = df["tick_volume"].clip(lower=1)
        result["log_volume"] = np.log(vol / vol.shift(1)).clip(-3.0, 3.0)

        # ── ATR chuẩn hóa ×1000 (tránh gradient vanishing) ──────────
        # ATR/price ≈ 0.001 → quá nhỏ cho Transformer → scale ×1000 → ~1.0
        high_low   = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close  = (df["low"]  - df["close"].shift(1)).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr        = true_range.rolling(self.atr_period).mean()
        result["atr_norm"] = (atr / df["close"]) * 1000  # ×1000 để scale về ~1.0

        # ── Sine/Cosine Time Encoding ─────────────────────────────────
        idx = df.index
        if idx.tz is None:
            raise ValueError("DataFrame index phải có timezone UTC")

        hour = idx.hour
        dow  = idx.dayofweek  # 0=Monday, 6=Sunday

        result["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        result["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        result["dow_sin"]  = np.sin(2 * np.pi * dow  / 7)
        result["dow_cos"]  = np.cos(2 * np.pi * dow  / 7)

        # ── [NEW] US Session Flag (13:00–21:00 UTC = 08:00–16:00 EST) ─
        # Vàng biến động mạnh nhất lúc mở cửa Mỹ (13:30 UTC) và Âu (07:00 UTC)
        # Flag này giúp Transformer nhận ra vùng rủi ro thanh khoản cao
        result["is_us_session"] = ((hour >= 13) & (hour < 21)).astype(np.float32)

        # ── [NEW] Weekend Flag ────────────────────────────────────────
        result["is_weekend"] = (dow >= 5).astype(np.float32)

        # ── Bỏ warmup rows (NaN do rolling/shift) ─────────────────────
        result = result.dropna()

        return result
```

- [ ] **Chạy để verify PASS:**
```bash
python -m pytest src/data/tests/test_data_processor.py -v
```
Kết quả mong đợi: `7 passed`

- [ ] **Commit:**
```bash
git add src/data/data_processor.py src/data/tests/test_data_processor.py
git commit -m "feat(sprint1): DataProcessor - log returns, ATR, sine/cosine time encoding"
```

---

## Task 2: Oracle — Triple Barrier Method

**Files:**
- Create: `src/data/oracle.py`
- Create: `src/data/tests/test_oracle.py`

### Step 2.1: Viết failing tests cho Oracle

- [ ] **Viết test file:**

```python
# src/data/tests/test_oracle.py
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.data.oracle import Oracle


def _make_price_series(values: list, freq: str = "15min") -> pd.Series:
    """Tạo Series giá với index DatetimeIndex UTC."""
    idx = pd.date_range("2020-01-01", periods=len(values), freq=freq, tz="UTC")
    return pd.Series(values, index=idx, name="close")


class TestTripleBarrier:
    def test_clear_uptrend_labels_buy(self):
        """
        Giá tăng thẳng không kéo dài → Oracle gắn nhãn BUY=1.
        Thiết lập: TP=+2%, SL=-1%, giá tăng đều 0.5% mỗi nến.
        Giá sẽ chạm TP (rào 1) trước SL (rào 2).
        """
        prices = [1800.0 * (1.005 ** i) for i in range(30)]
        close = _make_price_series(prices)
        df = pd.DataFrame({"close": close, "high": close * 1.001, "low": close * 0.999})

        oracle = Oracle(tp_atr_mult=2.0, sl_atr_mult=1.0, max_hold_bars=20)
        atr = pd.Series([18.0] * len(df), index=df.index)  # ATR cố định = 1%
        labels = oracle.label(df, atr)

        # Điểm đầu (index 0) phải là BUY
        assert labels.iloc[0] == 1, f"Kỳ vọng BUY=1, nhận được {labels.iloc[0]}"

    def test_clear_downtrend_labels_sell(self):
        """
        Giá giảm thẳng → Oracle gắn nhãn SELL=2.
        """
        prices = [1800.0 * (0.995 ** i) for i in range(30)]
        close = _make_price_series(prices)
        df = pd.DataFrame({"close": close, "high": close * 1.001, "low": close * 0.999})

        oracle = Oracle(tp_atr_mult=2.0, sl_atr_mult=1.0, max_hold_bars=20)
        atr = pd.Series([18.0] * len(df), index=df.index)
        labels = oracle.label(df, atr)

        assert labels.iloc[0] == 2, f"Kỳ vọng SELL=2, nhận được {labels.iloc[0]}"

    def test_sideways_labels_hold(self):
        """
        Giá đi ngang → không chạm TP hay SL trước max_hold_bars → HOLD=0.
        """
        prices = [1800.0 + np.sin(i * 0.1) * 2 for i in range(50)]
        close = _make_price_series(prices)
        df = pd.DataFrame({"close": close, "high": close + 1, "low": close - 1})

        oracle = Oracle(tp_atr_mult=5.0, sl_atr_mult=5.0, max_hold_bars=5)
        atr = pd.Series([200.0] * len(df), index=df.index)  # ATR rất lớn → khó chạm
        labels = oracle.label(df, atr)

        # Phần lớn phải là HOLD
        hold_ratio = (labels == 0).mean()
        assert hold_ratio > 0.5, f"HOLD ratio quá thấp: {hold_ratio:.2f}"

    def test_sl_hit_before_tp_returns_hold(self):
        """
        Giá tăng nhưng trước tiên nhúng sâu chạm SL → bị lọc thành HOLD.
        Đây là bài test quan trọng nhất — kiểm tra chống 'nhãn giả'.
        """
        # Giá: tăng một chút, rồi nhúng mạnh (chạm SL), rồi mới tăng lên (TP)
        prices = [1800, 1802, 1804, 1770, 1830, 1850, 1870]  # nhúng xuống 1770
        close = _make_price_series(prices)
        # Low bar thứ 3 xuống 1760 — đủ để chạm SL nếu SL = 1800 - 1*ATR(=30) = 1770
        low   = pd.Series([1798, 1800, 1802, 1760, 1828, 1848, 1868], index=close.index)
        high  = close + 2
        df    = pd.DataFrame({"close": close, "high": high, "low": low})

        oracle = Oracle(tp_atr_mult=2.0, sl_atr_mult=1.0, max_hold_bars=10)
        atr = pd.Series([30.0] * len(df), index=df.index)  # SL = 1800 - 30 = 1770
        labels = oracle.label(df, atr)

        # Điểm entry 0 → SL bị chạm trước TP → phải là HOLD=0
        assert labels.iloc[0] == 0, (
            f"Giá nhúng chạm SL trước TP, kỳ vọng HOLD=0 nhưng nhận {labels.iloc[0]}"
        )

    def test_labels_only_valid_values(self):
        """Tất cả nhãn phải là 0, 1, hoặc 2."""
        prices = [1800 + np.random.randn() * 5 for _ in range(100)]
        close = _make_price_series(prices)
        df = pd.DataFrame({"close": close, "high": close + 5, "low": close - 5})

        oracle = Oracle(tp_atr_mult=2.0, sl_atr_mult=1.0, max_hold_bars=20)
        atr = pd.Series([18.0] * len(df), index=df.index)
        labels = oracle.label(df, atr)

        assert set(labels.unique()).issubset({0, 1, 2}), (
            f"Nhãn bất hợp lệ: {labels.unique()}"
        )

    def test_output_length_matches_input(self):
        """Số nhãn output phải bằng số hàng input."""
        prices = [1800.0 + i for i in range(50)]
        close = _make_price_series(prices)
        df = pd.DataFrame({"close": close, "high": close + 2, "low": close - 2})

        oracle = Oracle(tp_atr_mult=2.0, sl_atr_mult=1.0, max_hold_bars=10)
        atr = pd.Series([18.0] * len(df), index=df.index)
        labels = oracle.label(df, atr)

        assert len(labels) == len(df), (
            f"len(labels)={len(labels)} != len(df)={len(df)}"
        )
```

- [ ] **Chạy để verify FAIL:**
```bash
python -m pytest src/data/tests/test_oracle.py -v
```
Kết quả mong đợi: `ERROR — ModuleNotFoundError: No module named 'src.data.oracle'`

### Step 2.2: Implement Oracle

- [ ] **Tạo file `src/data/oracle.py`:**

```python
"""
oracle.py
---------
Triple Barrier Method (Marcos Lopez de Prado).

Tại mỗi điểm thời gian t trong quá khứ, Oracle "nhìn về tương lai"
và gắn nhãn:
  - BUY  (1): Giá chạm rào TP TRÊN trước rào SL DƯỚI
  - SELL (2): Giá chạm rào TP DƯỚI trước rào SL TRÊN
  - HOLD (0): Hết max_hold_bars mà chưa chạm rào nào (hoặc SL bị chạm trước TP)

Label HOLD bao gồm cả trường hợp SL bị chạm — đây là điểm mấu chốt
để lọc 'nhãn giả' khỏi dataset.
"""

import numpy as np
import pandas as pd


class Oracle:
    """
    Parameters
    ----------
    tp_atr_mult  : float — TP = entry_price ± (tp_atr_mult × ATR)
    sl_atr_mult  : float — SL = entry_price ∓ (sl_atr_mult × ATR)
    max_hold_bars: int   — Tối đa bao nhiêu nến giữ lệnh
    """

    def __init__(
        self,
        tp_atr_mult:   float = 1.5,
        sl_atr_mult:   float = 1.0,
        max_hold_bars: int   = 48,   # 12 giờ với M15
    ):
        self.tp_atr_mult   = tp_atr_mult
        self.sl_atr_mult   = sl_atr_mult
        self.max_hold_bars = max_hold_bars

    def _label_one(
        self,
        i:     int,
        close: np.ndarray,
        high:  np.ndarray,
        low:   np.ndarray,
        atr:   np.ndarray,
    ) -> int:
        """
        Gắn nhãn cho một điểm entry tại vị trí i.
        Nhìn về phía trước tối đa max_hold_bars nến.
        """
        entry = close[i]
        atr_i = atr[i]

        tp_long  = entry + self.tp_atr_mult  * atr_i  # Rào TP cho Long
        sl_long  = entry - self.sl_atr_mult   * atr_i  # Rào SL cho Long
        tp_short = entry - self.tp_atr_mult  * atr_i  # Rào TP cho Short
        sl_short = entry + self.sl_atr_mult  * atr_i  # Rào SL cho Short

        end = min(i + self.max_hold_bars + 1, len(close))

        for j in range(i + 1, end):
            h = high[j]
            l = low[j]

            # Kiểm tra Long: TP trên, SL dưới
            long_tp_hit = h >= tp_long
            long_sl_hit = l <= sl_long

            if long_tp_hit and not long_sl_hit:
                return 1  # BUY — TP chạm trước SL
            if long_sl_hit:
                # SL bị chạm (dù có TP hay không) → nhãn không hợp lệ
                break

        # Reset và kiểm tra Short
        for j in range(i + 1, end):
            h = high[j]
            l = low[j]

            short_tp_hit = l <= tp_short
            short_sl_hit = h >= sl_short

            if short_tp_hit and not short_sl_hit:
                return 2  # SELL — TP chạm trước SL
            if short_sl_hit:
                break

        return 0  # HOLD — không có tín hiệu hợp lệ

    def label(
        self,
        df:  pd.DataFrame,
        atr: pd.Series,
    ) -> pd.Series:
        """
        Gắn nhãn toàn bộ DataFrame.

        Parameters
        ----------
        df  : DataFrame với cột close, high, low (index = DatetimeIndex UTC)
        atr : Series ATR cùng index với df

        Returns
        -------
        pd.Series nhãn [0=Hold, 1=Buy, 2=Sell], cùng index với df
        """
        close = df["close"].to_numpy(dtype=np.float64)
        high  = df["high"].to_numpy(dtype=np.float64)
        low   = df["low"].to_numpy(dtype=np.float64)
        atr_v = atr.to_numpy(dtype=np.float64)

        n      = len(df)
        labels = np.zeros(n, dtype=np.int8)

        for i in range(n - self.max_hold_bars):
            labels[i] = self._label_one(i, close, high, low, atr_v)

        # Các nến cuối không đủ lookforward → HOLD (đã là 0 mặc định)
        return pd.Series(labels, index=df.index, name="label", dtype=np.int8)
```

- [ ] **Chạy để verify PASS:**
```bash
python -m pytest src/data/tests/test_oracle.py -v
```
Kết quả mong đợi: `6 passed`

- [ ] **Commit:**
```bash
git add src/data/oracle.py src/data/tests/test_oracle.py
git commit -m "feat(sprint1): Oracle - Triple Barrier Method labeler, chống nhãn giả"
```

---

## Task 3: DatasetBuilder — Sliding Window → HDF5

**Files:**
- Create: `src/data/dataset_builder.py`
- Create: `src/data/tests/test_dataset_builder.py`

### Step 3.1: Viết failing tests cho DatasetBuilder

- [ ] **Viết test file:**

```python
# src/data/tests/test_dataset_builder.py
import numpy as np
import pandas as pd
import h5py
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.data.dataset_builder import DatasetBuilder


NUM_FEATURES = 13  # log_OHLCV(5) + atr_norm + hour_sin/cos + dow_sin/cos + is_us_session + is_weekend + is_gap


def _make_feature_df(n: int = 300) -> pd.DataFrame:
    """Tạo DataFrame features giả (output của DataProcessor)."""
    idx = pd.date_range("2020-01-01", periods=n, freq="15min", tz="UTC")
    np.random.seed(0)
    data = {
        "log_open":   np.random.randn(n) * 0.001,
        "log_high":   np.random.randn(n) * 0.001,
        "log_low":    np.random.randn(n) * 0.001,
        "log_close":  np.random.randn(n) * 0.001,
        "log_volume": np.random.randn(n) * 0.1,
        "atr_norm":   np.abs(np.random.randn(n)) * 0.001 + 0.001,
        "hour_sin":   np.sin(np.arange(n) * 2 * np.pi / 24),
        "hour_cos":   np.cos(np.arange(n) * 2 * np.pi / 24),
        "dow_sin":    np.sin(np.arange(n) * 2 * np.pi / 7),
        "dow_cos":    np.cos(np.arange(n) * 2 * np.pi / 7),
    }
    return pd.DataFrame(data, index=idx)


def _make_labels(n: int = 300) -> pd.Series:
    """Tạo Series nhãn giả [0, 1, 2]."""
    np.random.seed(1)
    idx = pd.date_range("2020-01-01", periods=n, freq="15min", tz="UTC")
    return pd.Series(np.random.choice([0, 1, 2], size=n, p=[0.9, 0.05, 0.05]),
                     index=idx, dtype=np.int8)


class TestDatasetBuilder:
    def test_h5_file_created(self, tmp_path):
        """File HDF5 phải được tạo ra sau khi build."""
        features = _make_feature_df(300)
        labels   = _make_labels(300)
        builder  = DatasetBuilder(window_size=64)
        out_path = tmp_path / "test_dataset.h5"

        builder.build(features, labels, out_path)

        assert out_path.exists(), "File HDF5 không được tạo ra"

    def test_h5_has_correct_datasets(self, tmp_path):
        """HDF5 phải có 2 dataset: 'X' (features) và 'y' (labels)."""
        features = _make_feature_df(300)
        labels   = _make_labels(300)
        builder  = DatasetBuilder(window_size=64)
        out_path = tmp_path / "test_dataset.h5"
        builder.build(features, labels, out_path)

        with h5py.File(out_path, "r") as f:
            assert "X" in f, "Thiếu dataset 'X' trong HDF5"
            assert "y" in f, "Thiếu dataset 'y' trong HDF5"

    def test_x_shape_correct(self, tmp_path):
        """Shape của X phải là (N_windows, window_size, n_features)."""
        n, win = 300, 64
        features = _make_feature_df(n)
        labels   = _make_labels(n)
        builder  = DatasetBuilder(window_size=win)
        out_path = tmp_path / "test_dataset.h5"
        builder.build(features, labels, out_path)

        expected_windows = n - win  # Số cửa sổ có thể tạo ra
        with h5py.File(out_path, "r") as f:
            X = f["X"]
            assert X.shape == (expected_windows, win, NUM_FEATURES), (
                f"Shape kỳ vọng ({expected_windows}, {win}, {NUM_FEATURES}), "
                f"nhận được {X.shape}"
            )

    def test_y_shape_matches_x(self, tmp_path):
        """Số nhãn y phải bằng số cửa sổ X."""
        n, win = 300, 64
        features = _make_feature_df(n)
        labels   = _make_labels(n)
        builder  = DatasetBuilder(window_size=win)
        out_path = tmp_path / "test_dataset.h5"
        builder.build(features, labels, out_path)

        with h5py.File(out_path, "r") as f:
            assert f["X"].shape[0] == f["y"].shape[0], (
                f"Số cửa sổ X ({f['X'].shape[0]}) != số nhãn y ({f['y'].shape[0]})"
            )

    def test_no_nan_in_x(self, tmp_path):
        """Không được có NaN trong tensor X."""
        features = _make_feature_df(300)
        labels   = _make_labels(300)
        builder  = DatasetBuilder(window_size=64)
        out_path = tmp_path / "test_dataset.h5"
        builder.build(features, labels, out_path)

        with h5py.File(out_path, "r") as f:
            X = f["X"][:]
            assert not np.isnan(X).any(), "Phát hiện NaN trong tensor X"

    def test_y_only_valid_labels(self, tmp_path):
        """Tất cả nhãn y phải là 0, 1, hoặc 2."""
        features = _make_feature_df(300)
        labels   = _make_labels(300)
        builder  = DatasetBuilder(window_size=64)
        out_path = tmp_path / "test_dataset.h5"
        builder.build(features, labels, out_path)

        with h5py.File(out_path, "r") as f:
            y = f["y"][:]
            assert set(np.unique(y)).issubset({0, 1, 2}), (
                f"Nhãn bất hợp lệ trong y: {np.unique(y)}"
            )
```

- [ ] **Chạy để verify FAIL:**
```bash
python -m pytest src/data/tests/test_dataset_builder.py -v
```
Kết quả mong đợi: `ERROR — ModuleNotFoundError: No module named 'src.data.dataset_builder'`

### Step 3.2: Implement DatasetBuilder

- [ ] **Tạo file `src/data/dataset_builder.py`:**

```python
"""
dataset_builder.py
------------------
Đóng gói Feature Tensor (output DataProcessor) và Labels (output Oracle)
thành file HDF5 dạng Sliding Window sẵn sàng cho PyTorch DataLoader.

Format HDF5:
  X: float32 array, shape (N, window_size, n_features)
  y: int8 array,    shape (N,)
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
        features: pd.DataFrame,
        labels:   pd.Series,
        out_path: Path,
    ) -> None:
        """
        Tạo file HDF5 từ features và labels.

        Parameters
        ----------
        features : DataFrame output của DataProcessor (index phải khớp labels)
        labels   : Series output của Oracle (index phải khớp features)
        out_path : Path lưu file .h5
        """
        # Align theo index (phòng trường hợp lệch index)
        features, labels = features.align(labels, join="inner", axis=0)

        feat_array  = features.to_numpy(dtype=np.float32)
        label_array = labels.to_numpy(dtype=np.int8)
        n_rows, n_features = feat_array.shape
        win = self.window_size

        n_windows = n_rows - win
        if n_windows <= 0:
            raise ValueError(
                f"Không đủ dữ liệu để tạo window. "
                f"n_rows={n_rows}, window_size={win}"
            )

        log.info(f"Building dataset: {n_windows} windows × {win} bars × {n_features} features")

        # ── Tạo tensor Sliding Window ─────────────────────────────────
        X = np.lib.stride_tricks.sliding_window_view(
            feat_array, window_shape=(win, n_features)
        )
        # sliding_window_view cho shape (n_windows, 1, win, n_features) — squeeze dim 1
        X = X[:, 0, :, :]                      # shape: (n_windows, win, n_features)
        y = label_array[win:]                  # nhãn tương ứng nến cuối cửa sổ

        # ── Ghi HDF5 ─────────────────────────────────────────────────
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(out_path, "w") as f:
            f.create_dataset("X", data=X, compression="gzip", compression_opts=4)
            f.create_dataset("y", data=y, compression="gzip", compression_opts=4)

        size_mb = out_path.stat().st_size / (1024 * 1024)
        log.info(f"✅ Saved: {out_path}  (X={X.shape}, y={y.shape}, {size_mb:.1f} MB)")
```

- [ ] **Chạy để verify PASS:**
```bash
python -m pytest src/data/tests/test_dataset_builder.py -v
```
Kết quả mong đợi: `6 passed`

- [ ] **Commit:**
```bash
git add src/data/dataset_builder.py src/data/tests/test_dataset_builder.py
git commit -m "feat(sprint1): DatasetBuilder - sliding window → HDF5 dataset"
```

---

## Task 4: Pipeline Runner — Ghép toàn bộ Sprint 1

**Files:**
- Create: `src/data/build_dataset.py`

### Step 4.1: Viết pipeline script

- [ ] **Tạo file `src/data/build_dataset.py`:**

```python
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
    atr_abs = features["atr_norm"] * df.loc[features.index, "close"]

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
    builder.build(features, labels, out_path)
    log.info(f"  ✅ Dataset saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Build XAUUSD training dataset")
    parser.add_argument("--m15", help="Glob path tới file CSV M15")
    parser.add_argument("--h1",  help="Glob path tới file CSV H1")
    parser.add_argument("--window-size", type=int, default=128)
    args = parser.parse_args()

    if not args.m15 and not args.h1:
        parser.error("Cần ít nhất --m15 hoặc --h1")

    if args.m15:
        run_pipeline(args.m15, "M15", window_size=args.window_size,
                     max_hold=48)   # 48 nến M15 = 12 giờ
    if args.h1:
        run_pipeline(args.h1,  "H1",  window_size=args.window_size,
                     max_hold=12)   # 12 nến H1 = 12 giờ

    log.info("\n🎉 Sprint 1 hoàn tất! Dataset sẵn sàng cho Sprint 3 (BC Training).")


if __name__ == "__main__":
    main()
```

- [ ] **Commit:**
```bash
git add src/data/build_dataset.py
git commit -m "feat(sprint1): pipeline runner - CSV → features → Oracle → HDF5"
```

---

## Task 5: Chạy toàn bộ tests Sprint 1 & Push

- [ ] **Chạy toàn bộ test suite:**
```bash
python -m pytest src/data/tests/ -v
```
Kết quả mong đợi: `≥ 25 passed, 0 failed`

- [ ] **Kiểm tra phân phối nhãn (chỉ khi đã có data CSV từ MT5):**
```bash
python src/data/build_dataset.py --m15 "data/raw/XAUUSD_M15_*.csv"
```
Kết quả mong đợi: Hold ≈ 85-95%, Buy ≈ 3-8%, Sell ≈ 3-8%

- [ ] **Push lên GitHub:**
```bash
git push origin main
```

---

## Điều kiện DONE cho Sprint 1
- [ ] `python -m pytest src/data/tests/ -v` → tất cả PASS
- [ ] File `.h5` trong `data/processed/` được tạo ra với shape đúng
- [ ] Label distribution: Hold < 95%, Buy ≥ 2%, Sell ≥ 2%
- [ ] Không có NaN trong tensor X
