import re

file_path = 'docs/superpowers/plans/2026-04-07-sprint2-market-env.md'
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

# Replace step init
text = text.replace('        reward        = 0.0\n        terminated    = False',
'''        reward        = 0.0
        terminated    = False
        is_trade      = False''')

# Replace on buy/sell from flat
text = text.replace('''            if action == 1:   # Buy
                commission             = self._reward_calc.on_open_commission()''',
'''            if action == 1:   # Buy
                is_trade               = True
                commission             = self._reward_calc.on_open_commission()''')

text = text.replace('''            elif action == 2:  # Sell
                commission             = self._reward_calc.on_open_commission()''',
'''            elif action == 2:  # Sell
                is_trade               = True
                commission             = self._reward_calc.on_open_commission()''')

# Replace on reversal
text = text.replace('''                if is_reversal:
                    #  Mở ngay lệnh ngược chiều trong cùng 1 step 
                    commission         = self._reward_calc.on_open_commission()''',
'''                if is_reversal:
                    is_trade           = True
                    #  Mở ngay lệnh ngược chiều trong cùng 1 step 
                    commission         = self._reward_calc.on_open_commission()''')

# Replace info assignment
text = text.replace('''            "balance":  self._balance,
            "equity":   equity,  # Dùng để tính Sharpe ratio thực sự
            "drawdown": self._peak_balance - equity,''',
'''            "balance":  self._balance,
            "equity":   equity,  # Dùng để tính Sharpe ratio thực sự
            "drawdown": self._peak_balance - equity,
            "is_trade": is_trade,''')

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(text)
print("Added is_trade to Sprint 2 step")
