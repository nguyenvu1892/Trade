"""
test_download_mt5.py
--------------------
Unit tests cho download_mt5.py — KHÔNG cần MT5 terminal thật.
Dùng mock để kiểm tra logic xử lý data mà không phụ thuộc network.

Chạy: pytest src/data/tests/test_download_mt5.py -v
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

# ─── Import module cần test ────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.download_mt5 import download_bars, validate_symbol


# ─── Helpers ──────────────────────────────────────────────────────────

def _make_fake_rates(n: int = 1000, start_price: float = 1800.0) -> np.ndarray:
    """
    Tạo mảng rates giả dạng numpy structured array — giống hệt format MT5 trả về.
    """
    dtype = np.dtype([
        ("time",        np.int64),
        ("open",        np.float64),
        ("high",        np.float64),
        ("low",         np.float64),
        ("close",       np.float64),
        ("tick_volume", np.int64),
        ("spread",      np.int32),
        ("real_volume", np.int64),
    ])
    rates = np.zeros(n, dtype=dtype)

    base_time = int(datetime(2015, 1, 5, 0, 0, tzinfo=timezone.utc).timestamp())
    bar_seconds = 900  # M15 = 15 phút

    for i in range(n):
        close_price = start_price + np.random.uniform(-5, 5)
        rates[i]["time"]        = base_time + i * bar_seconds
        rates[i]["open"]        = close_price + np.random.uniform(-1, 1)
        rates[i]["high"]        = close_price + np.random.uniform(0, 3)
        rates[i]["low"]         = close_price - np.random.uniform(0, 3)
        rates[i]["close"]       = close_price
        rates[i]["tick_volume"] = np.random.randint(100, 5000)
        rates[i]["spread"]      = 20
        rates[i]["real_volume"] = 0

    return rates


# ─── Tests ────────────────────────────────────────────────────────────

class TestDownloadBars:
    """Kiểm tra hàm download_bars() với MT5 được mock."""

    @patch("src.data.download_mt5.mt5")
    def test_returns_csv_path_on_success(self, mock_mt5, tmp_path):
        """download_bars() phải trả về Path của file CSV khi thành công."""
        mock_mt5.copy_rates_range.return_value = _make_fake_rates(1000)

        import MetaTrader5 as mt5_real
        result = download_bars(
            symbol    = "XAUUSD",
            tf_name   = "M15",
            tf_const  = 16385,  # mt5.TIMEFRAME_M15
            years     = 1,
            output_dir= tmp_path,
        )

        assert result is not None, "Phải trả về Path, không phải None"
        assert result.exists(), "File CSV phải tồn tại trên disk"
        assert result.suffix == ".csv", "File phải có đuôi .csv"

    @patch("src.data.download_mt5.mt5")
    def test_csv_has_correct_columns(self, mock_mt5, tmp_path):
        """CSV output phải có đúng 7 cột chuẩn."""
        mock_mt5.copy_rates_range.return_value = _make_fake_rates(500)

        result = download_bars("XAUUSD", "H1", 16388, 1, tmp_path)
        df = pd.read_csv(result, index_col="datetime", parse_dates=True)

        expected_cols = {"open", "high", "low", "close", "tick_volume", "spread", "real_volume"}
        assert expected_cols.issubset(set(df.columns)), (
            f"Thiếu cột. Có: {set(df.columns)}. Cần: {expected_cols}"
        )

    @patch("src.data.download_mt5.mt5")
    def test_datetime_index_is_utc(self, mock_mt5, tmp_path):
        """Index của CSV phải là datetime UTC, không phải Unix timestamp."""
        mock_mt5.copy_rates_range.return_value = _make_fake_rates(200)

        result = download_bars("XAUUSD", "M15", 16385, 1, tmp_path)
        df = pd.read_csv(result, index_col="datetime", parse_dates=True)

        assert pd.api.types.is_datetime64_any_dtype(df.index), (
            "Index phải là kiểu datetime64"
        )
        # Kiểm tra giá trị năm hợp lý (không phải Unix timestamp số lớn)
        assert df.index[0].year >= 2010, (
            f"Năm đầu tiên không hợp lý: {df.index[0].year}"
        )

    @patch("src.data.download_mt5.mt5")
    def test_data_sorted_ascending(self, mock_mt5, tmp_path):
        """Dữ liệu phải được sắp xếp tăng dần theo thời gian."""
        mock_mt5.copy_rates_range.return_value = _make_fake_rates(300)

        result = download_bars("XAUUSD", "M15", 16385, 1, tmp_path)
        df = pd.read_csv(result, index_col="datetime", parse_dates=True)

        assert df.index.is_monotonic_increasing, (
            "Dữ liệu PHẢI được sắp xếp tăng dần theo thời gian"
        )

    @patch("src.data.download_mt5.mt5")
    def test_no_null_values_in_ohlc(self, mock_mt5, tmp_path):
        """OHLC không được có giá trị NULL."""
        mock_mt5.copy_rates_range.return_value = _make_fake_rates(500)

        result = download_bars("XAUUSD", "M15", 16385, 1, tmp_path)
        df = pd.read_csv(result, index_col="datetime", parse_dates=True)

        null_counts = df[["open", "high", "low", "close"]].isnull().sum()
        assert null_counts.sum() == 0, (
            f"OHLC có NULL: {null_counts[null_counts > 0].to_dict()}"
        )

    @patch("src.data.download_mt5.mt5")
    def test_returns_none_when_mt5_fails(self, mock_mt5, tmp_path):
        """Phải trả về None (không crash) khi MT5 không trả về data."""
        mock_mt5.copy_rates_range.return_value = None
        mock_mt5.last_error.return_value = (10001, "Simulated MT5 error")

        result = download_bars("XAUUSD", "M15", 16385, 1, tmp_path)

        assert result is None, "Phải trả về None khi MT5 thất bại"

    @patch("src.data.download_mt5.mt5")
    def test_returns_none_when_rates_empty(self, mock_mt5, tmp_path):
        """Phải trả về None khi MT5 trả về mảng rỗng."""
        mock_mt5.copy_rates_range.return_value = np.array([])
        mock_mt5.last_error.return_value = (0, "")

        result = download_bars("XAUUSD", "M15", 16385, 1, tmp_path)

        assert result is None, "Phải trả về None khi mảng data rỗng"

    @patch("src.data.download_mt5.mt5")
    def test_high_always_gte_low(self, mock_mt5, tmp_path):
        """
        Kiểm tra tính nhất quán: high >= low cho mọi nến.
        Script phải LOG WARNING nếu phát hiện, nhưng vẫn lưu file.
        """
        rates = _make_fake_rates(100)
        # Đảm bảo dữ liệu của chúng ta nhất quán (high >= low)
        rates["high"] = np.maximum(rates["high"], rates["low"])

        mock_mt5.copy_rates_range.return_value = rates

        result = download_bars("XAUUSD", "M15", 16385, 1, tmp_path)
        df = pd.read_csv(result, index_col="datetime", parse_dates=True)

        violations = (df["high"] < df["low"]).sum()
        assert violations == 0, (
            f"Có {violations} nến với high < low trong dữ liệu giả sạch"
        )


class TestValidateSymbol:
    """Kiểm tra hàm validate_symbol() với MT5 được mock."""

    @patch("src.data.download_mt5.mt5")
    def test_returns_true_for_valid_visible_symbol(self, mock_mt5):
        """Trả về True khi symbol tồn tại và đã visible."""
        symbol_info = MagicMock()
        symbol_info.visible = True
        symbol_info.digits = 2
        symbol_info.spread = 20
        mock_mt5.symbol_info.return_value = symbol_info

        assert validate_symbol("XAUUSD") is True

    @patch("src.data.download_mt5.mt5")
    def test_returns_false_for_missing_symbol(self, mock_mt5):
        """Trả về False khi symbol không tồn tại trong MT5."""
        mock_mt5.symbol_info.return_value = None

        assert validate_symbol("FAKESYMBOL") is False

    @patch("src.data.download_mt5.mt5")
    def test_enables_hidden_symbol(self, mock_mt5):
        """
        Nếu symbol tồn tại nhưng chưa visible, phải gọi symbol_select(True)
        để enable nó trước khi tiếp tục.
        """
        symbol_info = MagicMock()
        symbol_info.visible = False
        symbol_info.digits = 2
        symbol_info.spread = 20
        mock_mt5.symbol_info.return_value = symbol_info
        mock_mt5.symbol_select.return_value = True

        result = validate_symbol("XAUUSD")

        mock_mt5.symbol_select.assert_called_once_with("XAUUSD", True)
        assert result is True

    @patch("src.data.download_mt5.mt5")
    def test_returns_false_when_enable_fails(self, mock_mt5):
        """Nếu symbol_select() thất bại, phải trả về False."""
        symbol_info = MagicMock()
        symbol_info.visible = False
        mock_mt5.symbol_info.return_value = symbol_info
        mock_mt5.symbol_select.return_value = False
        mock_mt5.last_error.return_value = (10002, "Select failed")

        assert validate_symbol("XAUUSD") is False
