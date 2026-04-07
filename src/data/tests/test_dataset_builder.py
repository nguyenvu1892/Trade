import numpy as np
import pandas as pd
import h5py
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.data.dataset_builder import DatasetBuilder


NUM_FEATURES = 13  # log_OHLCV(5) + atr_norm + hour_sin/cos + dow_sin/cos + is_us_session + is_weekend + is_gap


def _make_feature_df(n: int = 300) -> tuple[pd.DataFrame, pd.Series]:
    """Tạo DataFrame features giả (output của DataProcessor)."""
    idx = pd.date_range("2020-01-01", periods=n, freq="15min", tz="UTC")
    np.random.seed(0)
    close_raw = 1800.0 + np.cumsum(np.random.randn(n) * 0.5)  # giá đóng thực
    data = {
        "log_open":     np.random.randn(n) * 0.001,
        "log_high":     np.random.randn(n) * 0.001,
        "log_low":      np.random.randn(n) * 0.001,
        "log_close":    np.random.randn(n) * 0.001,
        "log_volume":   np.random.randn(n) * 0.1,
        "atr_norm":     np.abs(np.random.randn(n)) * 0.001 + 0.001,
        "hour_sin":     np.sin(np.arange(n) * 2 * np.pi / 24),
        "hour_cos":     np.cos(np.arange(n) * 2 * np.pi / 24),
        "dow_sin":      np.sin(np.arange(n) * 2 * np.pi / 7),
        "dow_cos":      np.cos(np.arange(n) * 2 * np.pi / 7),
        "is_us_session":np.zeros(n),
        "is_weekend":   np.zeros(n),
        "is_gap":       np.zeros(n),
    }
    return pd.DataFrame(data, index=idx), pd.Series(close_raw, index=idx, name="close")


def _make_labels(n: int = 300) -> pd.Series:
    """Tạo Series nhãn giả [0, 1, 2]."""
    np.random.seed(1)
    idx = pd.date_range("2020-01-01", periods=n, freq="15min", tz="UTC")
    return pd.Series(np.random.choice([0, 1, 2], size=n, p=[0.9, 0.05, 0.05]),
                     index=idx, dtype=np.int8)


class TestDatasetBuilder:
    def test_h5_file_created(self, tmp_path):
        """File HDF5 phải được tạo ra sau khi build."""
        features, close_s = _make_feature_df(300)
        labels   = _make_labels(300)
        builder  = DatasetBuilder(window_size=64)
        out_path = tmp_path / "test_dataset.h5"

        builder.build(features, labels, out_path, close_prices=close_s)

        assert out_path.exists(), "File HDF5 không được tạo ra"

    def test_h5_has_correct_datasets(self, tmp_path):
        """HDF5 phải có 3 dataset: 'X', 'y', và 'close' (giá gốc cho Simulator)."""
        features, close_s = _make_feature_df(300)
        labels   = _make_labels(300)
        builder  = DatasetBuilder(window_size=64)
        out_path = tmp_path / "test_dataset.h5"
        builder.build(features, labels, out_path, close_prices=close_s)

        with h5py.File(out_path, "r") as f:
            assert "X"     in f, "Thiếu dataset 'X' trong HDF5"
            assert "y"     in f, "Thiếu dataset 'y' trong HDF5"
            assert "close" in f, (
                "[CRITICAL] Thiếu dataset 'close' — Simulator sẽ không thể tính PnL!"
            )

    def test_x_shape_correct(self, tmp_path):
        """Shape của X phải là (N_windows, window_size, n_features)."""
        n, win = 300, 64
        features, close_s = _make_feature_df(n)
        labels   = _make_labels(n)
        builder  = DatasetBuilder(window_size=win)
        out_path = tmp_path / "test_dataset.h5"
        builder.build(features, labels, out_path, close_prices=close_s)

        expected_windows = n - win
        with h5py.File(out_path, "r") as f:
            X = f["X"]
            assert X.shape == (expected_windows, win, NUM_FEATURES), (
                f"Shape kỳ vọng ({expected_windows}, {win}, {NUM_FEATURES}), "
                f"nhận được {X.shape}"
            )

    def test_close_shape_matches_windows(self, tmp_path):
        """'close' phải có shape (N_windows,) — một giá đóng cho mỗi cửa sổ."""
        n, win = 300, 64
        features, close_s = _make_feature_df(n)
        labels   = _make_labels(n)
        builder  = DatasetBuilder(window_size=win)
        out_path = tmp_path / "test_dataset.h5"
        builder.build(features, labels, out_path, close_prices=close_s)

        with h5py.File(out_path, "r") as f:
            assert f["close"].shape[0] == f["X"].shape[0], (
                f"'close' shape không khớp với số cửa sổ X: "
                f"{f['close'].shape[0]} vs {f['X'].shape[0]}"
            )

    def test_close_prices_positive(self, tmp_path):
        """Giá đóng phải luôn dương (giá XAUUSD thực tế)."""
        n, win = 300, 64
        features, close_s = _make_feature_df(n)
        labels   = _make_labels(n)
        builder  = DatasetBuilder(window_size=win)
        out_path = tmp_path / "test_dataset.h5"
        builder.build(features, labels, out_path, close_prices=close_s)

        with h5py.File(out_path, "r") as f:
            close = f["close"][:]
            assert (close > 0).all(), f"Giá đóng phải dương, min={close.min():.2f}"

    def test_y_shape_matches_x(self, tmp_path):
        """Số nhãn y phải bằng số cửa sổ X."""
        n, win = 300, 64
        features, close_s = _make_feature_df(n)
        labels   = _make_labels(n)
        builder  = DatasetBuilder(window_size=win)
        out_path = tmp_path / "test_dataset.h5"
        builder.build(features, labels, out_path, close_prices=close_s)

        with h5py.File(out_path, "r") as f:
            assert f["X"].shape[0] == f["y"].shape[0], (
                f"Số cửa sổ X ({f['X'].shape[0]}) != số nhãn y ({f['y'].shape[0]})"
            )

    def test_no_nan_in_x(self, tmp_path):
        """Không được có NaN trong tensor X."""
        features, close_s = _make_feature_df(300)
        labels   = _make_labels(300)
        builder  = DatasetBuilder(window_size=64)
        out_path = tmp_path / "test_dataset.h5"
        builder.build(features, labels, out_path, close_prices=close_s)

        with h5py.File(out_path, "r") as f:
            X = f["X"][:]
            assert not np.isnan(X).any(), "Phát hiện NaN trong tensor X"

    def test_y_only_valid_labels(self, tmp_path):
        """Tất cả nhãn y phải là 0, 1, hoặc 2."""
        features, close_s = _make_feature_df(300)
        labels   = _make_labels(300)
        builder  = DatasetBuilder(window_size=64)
        out_path = tmp_path / "test_dataset.h5"
        builder.build(features, labels, out_path, close_prices=close_s)

        with h5py.File(out_path, "r") as f:
            y = f["y"][:]
            assert set(np.unique(y)).issubset({0, 1, 2}), (
                f"Nhãn bất hợp lệ trong y: {np.unique(y)}"
            )
