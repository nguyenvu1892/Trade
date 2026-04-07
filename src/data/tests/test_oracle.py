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
