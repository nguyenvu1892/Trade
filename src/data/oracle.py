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
        max_hold_bars: int   = 24,   # 2 giờ với M5
        spread_pips:   int   = 25,
        commission_usd: float = 0.07,
        lot_size:      float = 0.01,
    ):
        self.tp_atr_mult   = tp_atr_mult
        self.sl_atr_mult   = sl_atr_mult
        self.max_hold_bars = max_hold_bars
        self.cost_equivalent = (spread_pips * 0.01) + (commission_usd / (lot_size * 100))

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

        # [FIX] Đẩy rào cản giãn ra một khoảng chi phí (Spread + Commission)
        tp_long  = entry + self.cost_equivalent + self.tp_atr_mult  * atr_i
        sl_long  = entry - self.cost_equivalent - self.sl_atr_mult  * atr_i
        tp_short = entry - self.cost_equivalent - self.tp_atr_mult  * atr_i
        sl_short = entry + self.cost_equivalent + self.sl_atr_mult  * atr_i

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

        # In diagnostic tracking metric
        total = len(labels)
        holds = (labels == 0).sum()
        buys = (labels == 1).sum()
        sells = (labels == 2).sum()
        print(f"--- ORACLE LABEL DISTRIBUTION ---")
        print(f"Total: {total}")
        print(f"HOLD: {holds} ({holds/total*100:.2f}%)")
        print(f"BUY: {buys} ({buys/total*100:.2f}%)")
        print(f"SELL: {sells} ({sells/total*100:.2f}%)")
        buy_sell_ratio = (buys + sells) / total * 100
        print(f"Valid Action Ratio: {buy_sell_ratio:.2f}%")
        
        if buy_sell_ratio < 2.0:
            print("WARNING: Buy/Sell ratio is below 2%. Tuning Oracle params advised.")

        # Các nến cuối không đủ lookforward → HOLD (đã là 0 mặc định)
        return pd.Series(labels, index=df.index, name="label", dtype=np.int8)
