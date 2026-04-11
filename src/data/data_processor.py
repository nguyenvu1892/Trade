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
        pd.DataFrame với các cột feature
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

        # --- Giữ nguyên log_tick_volume cũ để duy trì 13 features gốc ---
        result["log_tick_volume"] = np.log(df["tick_volume"] + 1.0).astype(np.float32)

        # --- Bypass tính VWAP/Surge nếu chạy Live (đã nhận từ C# gRPC) ---
        if "vwap_distance" in df.columns and "volume_surge" in df.columns:
            result["volume_surge"] = df["volume_surge"].astype(np.float32)
            result["vwap_distance"] = df["vwap_distance"].astype(np.float32)
        else:
            # ── Volume Surge (Thay thế log_volume cũ) ─────────────────────
            epsilon = 1e-8
            vol = df["tick_volume"]
            vol_mean = vol.rolling(20, min_periods=1).mean()
            result["volume_surge"] = np.log((vol + epsilon) / (vol_mean + epsilon)).clip(-5.0, 5.0).astype(np.float32)

            # ── CME VWAP Distance (Reset theo phiên Globex US/Eastern) ────
            # Chuyển index sang múi giờ New York để tính đúng giờ mùa Đông/Hè (DST)
            df_est = df.copy()
            df_est.index = df_est.index.tz_convert('US/Eastern')
            
            # Phiên Globex mới bắt đầu từ 18:00 EST. 
            # Nến từ 18:00 trở đi sẽ được tính vào Session Date của ngày hôm sau.
            session_date = df_est.index.date
            mask_after_18 = df_est.index.hour >= 18
            session_date = session_date + pd.to_timedelta(mask_after_18.astype(int), unit='D')
            
            typical_price = (df_est["high"] + df_est["low"] + df_est["close"]) / 3.0
            cv = df_est["tick_volume"].groupby(session_date).cumsum()
            ctpv = (typical_price * df_est["tick_volume"]).groupby(session_date).cumsum()
            
            vwap = ctpv / cv
            vwap = vwap.fillna(df_est["close"]) # Tránh chia 0
            
            # Gắn lại đúng index UTC cũ cho result
            result["vwap_distance"] = ((df_est["close"] - vwap) / vwap).values.astype(np.float32)

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

        result["hour_sin"] = np.sin(2 * np.pi * hour / 24).astype(np.float32)
        result["hour_cos"] = np.cos(2 * np.pi * hour / 24).astype(np.float32)
        result["dow_sin"]  = np.sin(2 * np.pi * dow  / 7).astype(np.float32)
        result["dow_cos"]  = np.cos(2 * np.pi * dow  / 7).astype(np.float32)

        # ── [NEW] US Session Flag (13:00–21:00 UTC = 08:00–16:00 EST) ─
        # Vàng biến động mạnh nhất lúc mở cửa Mỹ (13:30 UTC) và Âu (07:00 UTC)
        # Flag này giúp Transformer nhận ra vùng rủi ro thanh khoản cao
        result["is_us_session"] = ((hour >= 13) & (hour < 21)).astype(np.float32)

        # ── [NEW] Weekend Flag ────────────────────────────────────────
        result["is_weekend"] = (dow >= 5).astype(np.float32)

        # ── Bỏ warmup rows (NaN do rolling/shift) ─────────────────────
        result = result.dropna()

        # ── Sắp xếp lại cột TUYỆT ĐỐI TUÂN THỦ KIẾN TRÚC GỐC ──────────
        # Đảm bảo 13 features gốc đứng đầu tiên để Network Surgery map đúng Weight
        legacy_columns = [
            "is_gap", "log_open", "log_high", "log_low", "log_close", "log_tick_volume",
            "atr_norm", "hour_sin", "hour_cos", "dow_sin", "dow_cos", 
            "is_us_session", "is_weekend"
        ]
        new_columns = ["volume_surge", "vwap_distance"]
        
        # Bê nguyên xi output với 15 cột chuẩn
        result = result[legacy_columns + new_columns]

        return result
