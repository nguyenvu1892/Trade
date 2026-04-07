import re

file_path = 'src/data/oracle.py'
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace(
'''    def __init__(
        self,
        tp_atr_mult:   float = 1.5,
        sl_atr_mult:   float = 1.0,
        max_hold_bars: int   = 48,   # 12 giờ với M15
    ):
        self.tp_atr_mult   = tp_atr_mult
        self.sl_atr_mult   = sl_atr_mult
        self.max_hold_bars = max_hold_bars''',
'''    def __init__(
        self,
        tp_atr_mult:   float = 1.5,
        sl_atr_mult:   float = 1.0,
        max_hold_bars: int   = 48,   # 12 giờ với M15
        spread_pips:   int   = 25,
        commission_usd: float = 0.07,
        lot_size:      float = 0.01,
    ):
        self.tp_atr_mult   = tp_atr_mult
        self.sl_atr_mult   = sl_atr_mult
        self.max_hold_bars = max_hold_bars
        self.cost_equivalent = (spread_pips * 0.01) + (commission_usd / (lot_size * 100))''')

text = text.replace(
'''        tp_long  = entry + self.tp_atr_mult  * atr_i  # Rào TP cho Long
        sl_long  = entry - self.sl_atr_mult  * atr_i  # Rào SL cho Long
        tp_short = entry - self.tp_atr_mult  * atr_i  # Rào TP cho Short
        sl_short = entry + self.sl_atr_mult  * atr_i  # Rào SL cho Short''',
'''        # [FIX] Đẩy rào cản giãn ra một khoảng chi phí (Spread + Commission)
        tp_long  = entry + self.cost_equivalent + self.tp_atr_mult  * atr_i
        sl_long  = entry - self.cost_equivalent - self.sl_atr_mult  * atr_i
        tp_short = entry - self.cost_equivalent - self.tp_atr_mult  * atr_i
        sl_short = entry + self.cost_equivalent + self.sl_atr_mult  * atr_i''')

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(text)
