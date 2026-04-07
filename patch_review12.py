import re

file_path = 'docs/superpowers/plans/2026-04-07-sprint3-transformer-bc.md'
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

# Replace val_ratio = 0.1 with 0.2
text = text.replace('val_ratio  = 0.1', 'val_ratio  = 0.2')
text = text.replace('val_ratio:   float = 0.1', 'val_ratio:   float = 0.2')

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(text)

file_path = 'docs/superpowers/plans/2026-04-07-sprint2-market-env.md'
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

# Replace current_close in PnL
text = text.replace(
'''                pnl = self._calc_pnl(
                    self._entry_price, current_close,
                    self._position_dir, self._lot
                )''',
'''                # [FIX EXIT LOOKAHEAD] Đóng lệnh ở giá open nhánh tiếp theo, không dùng current_close!
                pnl = self._calc_pnl(
                    self._entry_price, entry_price, # entry_price chính là open_next
                    self._position_dir, self._lot
                )''')

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(text)

file_path = 'docs/superpowers/plans/2026-04-07-sprint4-rl-cloud.md'
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

# Replace max_drawdown_usd in evaluate_oos
text = text.replace(
'''        max_drawdown_usd=999999.0  # ChÃÂºÂy hÃÂºÂt OOS''',
'''        max_drawdown_usd=20.0  # [FIX EQUITY HOLE] Dừng sớm nếu âm 10% vốn Prop Firm''')
# fallback for unicode issues
text = re.sub(r'max_drawdown_usd=999999\.0.*?#.*', 'max_drawdown_usd=20.0  # [FIX EQUITY HOLE] Dung thiet lap Prop Firm DD () tranh chia cho so am', text)


with open(file_path, 'w', encoding='utf-8') as f:
    f.write(text)

print("Review 12 patched")
