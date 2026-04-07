import re

# --- 1. Sprint 2: Environment Spread Fix and Rollover Expansion ---
file_path = 'docs/superpowers/plans/2026-04-07-sprint2-market-env.md'
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

# Fix self._spread_usd
text = text.replace(
'''        self._spread_usd          = spread_pips * lot_size / 100.0  # VD: 25 pips * 0.01 / 100 = 0.0025$ (Quá nhỏ?)''',
'''        # [FIX SPREAD BUG] 1 pip = 0.01 USD. Độ lệch của spread không phụ thuộc Lot Size.
        self._spread_price_shift  = spread_pips * 0.01''')
text = text.replace('''self._spread_usd''', '''self._spread_price_shift''')

# Add rollover logic
text = text.replace(
'''        current_close = float(self._close[self._cursor])       # dùng để đóng lệnh
        entry_price   = float(self._open_next[self._cursor])   # dùng để mở lệnh (nến kế)''',
'''        current_close = float(self._close[self._cursor])       # dùng để đóng lệnh
        entry_price   = float(self._open_next[self._cursor])   # dùng để mở lệnh (nến kế)
        
        # [FIX SPREAD ROLLOVER] Giãn spread x5 vào 2 tiếng giao phiên (21:00 - 22:59 UTC)
        # Vì M15 có 96 nến/ngày, nến 21:00 bắt đầu từ index 84 (21*4) trong ngày
        # Trick đơn giản: trong features chứa giờ hiện tại, nhưng Env nhận numpy array. 
        # (Chỉ áp dụng giả lập giãn spread nếu cần độ chính xác cao)
        current_spread = self._spread_price_shift
        hour = (self._cursor % 96) / 4  # Gần đúng với chu kỳ 24h trên M15 gốc
        if 21 <= hour < 23:
            current_spread *= 5.0''')

text = text.replace(
'''                self._entry_price = entry_price + self._spread_price_shift if action == 1 else entry_price - self._spread_price_shift''',
'''                self._entry_price = entry_price + current_spread if action == 1 else entry_price - current_spread''')

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(text)

# --- 2. Sprint 3: Value Head Zero Initialization ---
file_path = 'docs/superpowers/plans/2026-04-07-sprint3-transformer-bc.md'
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace(
'''        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)''',
'''        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        # [FIX] Khởi động lạnh Value Head = 0 để tránh PPO phá nát trọng số ở đầu Phase 2
        nn.init.zeros_(self.value_head.weight)
        if self.value_head.bias is not None:
            nn.init.zeros_(self.value_head.bias)''')

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(text)

# --- 3. Sprint 1: Oracle Optimistic Spread ---
file_path = 'docs/superpowers/plans/2026-04-07-sprint1-data-foundation.md'
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
        sl_long  = entry - self.sl_atr_mult   * atr_i  # Rào SL cho Long
        tp_short = entry - self.tp_atr_mult  * atr_i  # Rào TP cho Short
        sl_short = entry + self.sl_atr_mult  * atr_i  # Rào SL cho Short''',
'''        # [FIX] Đẩy rào cản giãn ra một khoảng chi phí (Spread + Commission)
        tp_long  = entry + self.cost_equivalent + self.tp_atr_mult  * atr_i
        sl_long  = entry - self.cost_equivalent - self.sl_atr_mult  * atr_i
        tp_short = entry - self.cost_equivalent - self.tp_atr_mult  * atr_i
        sl_short = entry + self.cost_equivalent + self.sl_atr_mult  * atr_i''')

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(text)

# --- 4. Specs Document Synchronization ---
file_path = 'docs/superpowers/specs/2026-04-07-xauusd-bot-design.md'
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('''Global Average Pooling''', '''Last Token Extraction''')
text = text.replace('''Stable-Baselines3''', '''CleanRL PPO''')
text = text.replace('''Timeframe đầu vào: M5, M15, H1 (nạp song song)''', '''Timeframe đầu vào: Single Timeframe M15''')

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(text)

print("Review 15 Doc Patch Completed")
