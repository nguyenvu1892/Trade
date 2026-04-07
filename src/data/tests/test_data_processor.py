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
