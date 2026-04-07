import re

file_path = 'docs/superpowers/plans/2026-04-07-sprint4-rl-cloud.md'
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

# Replace compute_metrics signature
text = text.replace(
'''def compute_metrics(daily_returns: np.ndarray, periods_per_year: int = 24192) -> dict:''',
'''def compute_metrics(daily_returns: np.ndarray, periods_per_year: int = 24192, positions: np.ndarray = None) -> dict:''')

# Replace n_trades calculation
text = text.replace(
'''    # N trades = số bar có return != 0
    n_trades = int((r != 0).sum())''',
'''    # [FIX N_TRADES] Thay vì đếm số nến có return != 0 (do Equity nến nào cũng đổi),
    # đếm số lần position_dir thay đổi.
    if positions is not None:
        pos_diff = np.abs(np.diff(positions))
        n_trades = int(np.sum(pos_diff > 0)) # Mỗi lần đổi trạng thái là 1 giao dịch
    else:
        n_trades = int((r != 0).sum())''')

# Replace evaluate_oos to collect position
text = text.replace(
'''    # [FIX SHARPE OOS] Dùng Equity thay vì Balance để thấy rõ Unrealized Drawdown
    equity_hist = [200.0]
    while not done:''',
'''    # [FIX SHARPE OOS] Dùng Equity thay vì Balance để thấy rõ Unrealized Drawdown
    equity_hist = [200.0]
    position_hist = [0]
    while not done:''')

text = text.replace(
'''        obs, _, term, trunc, info = env.step(action)
        equity_hist.append(info.get("equity", 200.0))
        done = term or trunc''',
'''        obs, _, term, trunc, info = env.step(action)
        equity_hist.append(info.get("equity", 200.0))
        position_hist.append(env._position_dir)
        done = term or trunc''')

text = text.replace(
'''    bar_returns = np.diff(equity_hist) / np.array(equity_hist[:-1])
    metrics = compute_metrics(bar_returns)''',
'''    bar_returns = np.diff(equity_hist) / np.array(equity_hist[:-1])
    metrics = compute_metrics(bar_returns, positions=np.array(position_hist))''')

# Replace dropout of rollout_steps to fix PPO loop starvation
text = text.replace(
'''    rollout_steps = 2048''',
'''    rollout_steps = 256  # [FIX PPO LOOP] Tăng tần suất cập nhật, giảm số bước gom batch''')

text = text.replace(
'''        if update % 50 == 0:''',
'''        if update % 10 == 0:''')


with open(file_path, 'w', encoding='utf-8') as f:
    f.write(text)
print("Sprint 4 fully patched")
