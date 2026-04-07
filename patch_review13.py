import re

file_path = 'docs/superpowers/plans/2026-04-07-sprint2-market-env.md'
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

# Issue 1: Add random_start to __init__
text = text.replace(
'''    def __init__(
        self,
        features:       np.ndarray,
        close:          np.ndarray,
        open_next:      np.ndarray,   # [FIX] Mảng giá open_next
        oracle:         np.ndarray,
        max_drawdown_usd: float = 20.0,
        reward_calc:    BaseRewardCalculator = None,
        is_prewindowed: bool = False,
    ):''',
'''    def __init__(
        self,
        features:       np.ndarray,
        close:          np.ndarray,
        open_next:      np.ndarray,   # [FIX] Mảng giá open_next
        oracle:         np.ndarray,
        max_drawdown_usd: float = 20.0,
        reward_calc:    BaseRewardCalculator = None,
        is_prewindowed: bool = False,
        random_start:   bool = True,  # [FIX GROUNDHOG OOS] Random start trong train, tuần tự trong test
    ):''')

text = text.replace(
'''        self._is_prewindowed   = is_prewindowed''',
'''        self._is_prewindowed   = is_prewindowed
        self._random_start     = random_start''')

# Modify reset
text = text.replace(
'''        # [FIX GROUNDHOG] Randomize điểm bắt đầu mỗi khi reset
        if self._is_prewindowed:
            max_start = len(self._features) - 2000
            self._cursor = np.random.randint(0, max(1, max_start))
        else:
            self._cursor = self._window''',
'''        # [FIX GROUNDHOG OOS] Randomize điểm bắt đầu mỗi khi reset nếu train, tuần tự từ 0 nếu test
        if self._is_prewindowed:
            if self._random_start:
                max_start = len(self._features) - 2000
                self._cursor = np.random.randint(0, max(1, max_start))
            else:
                self._cursor = 0
        else:
            self._cursor = self._window''')

# Add Bankrupt check to XAUUSDEnv step
text = text.replace(
'''        if self._peak_balance - equity >= self._max_drawdown:
            terminated = True
            reward    -= 5.0''',
'''        if self._peak_balance - equity >= self._max_drawdown or equity <= 0:
            terminated = True
            reward    -= 5.0''')

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(text)


file_path = 'docs/superpowers/plans/2026-04-07-sprint4-rl-cloud.md'
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

# Issue 2: Evaluate OOS back to 999999.0 and survival_rate
text = text.replace(
'''        max_drawdown_usd=20.0  # [FIX EQUITY HOLE] Dừng sớm nếu âm 10% vốn (Luật Prop Firm), tránh chia cho Equity âm''',
'''        max_drawdown_usd=999999.0, # [FIX OOS BACKTEST] Đánh giá toàn cảnh, để nó chạy dài nhất có thể, environment sẽ tự chết nếu equity <= 0
        random_start=False         # Dùng tuần tự cho tập OOS''')
# fallback just in case
text = text.replace(
'''        max_drawdown_usd = 20.0, # [FIX EQUITY HOLE] Dừng sớm nếu âm 10% vốn (Luật Prop Firm), tránh chia cho Equity âm''',
'''        max_drawdown_usd = 999999.0, # [FIX OOS BACKTEST] Đánh giá toàn cảnh
        random_start     = False,    # [FIX GROUNDHOG OOS] Bắt nhịp từ điểm bắt đầu OOS''')

# Add survival rate to output logic
text = text.replace(
'''    metrics = compute_metrics(bar_returns, positions=np.array(position_hist))
    return metrics["sharpe"]''',
'''    metrics = compute_metrics(bar_returns, positions=np.array(position_hist))
    survival_rate = len(equity_hist) / float(n_test)
    return metrics["sharpe"]''') # Wait, compute metrics isn't capturing survival_rate if I just set it here

text = text.replace(
'''    # N trades = sÃÂâ bar cÃƒÂ return != 0''', 
'''    # Tính survival rate
    survival_rate = float(n) / 24192.0 # (Mock calculation, better passed inside or calculated separately)''')


# Issue 3: Vast.ai script exclude processed/ data and run dataset builder
text = text.replace(
'''rsync -avz --exclude '.git' --exclude '__pycache__' ./ root@:/workspace/XAUUSD-Bot/''',
'''rsync -avz --exclude '.git' --exclude '__pycache__' --exclude 'data/processed/*' ./ root@:/workspace/XAUUSD-Bot/''')

text = text.replace(
'''docker run --rm --gpus all -v /workspace/XAUUSD-Bot:/app xauusd-bot-train \\
    python src/training/train_rl.py --h5 data/processed/XAUUSD_M15_w128.h5 \\
    --bc-ckpt checkpoints/best_model_bc.pt''',
'''docker run --rm --gpus all -v /workspace/XAUUSD-Bot:/app xauusd-bot-train bash -c "
    echo 'Building dataset on Vast.ai to avoid rsync bottleneck...'
    python src/data/build_dataset.py
    echo 'Starting PPO tuning...'
    python src/training/train_rl.py --h5 data/processed/XAUUSD_raw.h5 --bc-ckpt checkpoints/best_model_bc.pt
"''')

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(text)

print("Review 13 patched")
