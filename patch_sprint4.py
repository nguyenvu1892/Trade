import re

file_path = 'docs/superpowers/plans/2026-04-07-sprint4-rl-cloud.md'
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

# Replace MSE loss with Huber Loss
text = text.replace('vf_loss = F.mse_loss(value.squeeze(-1), flat_ret[b])',
'''# [FIX GRADIENT EXPLOSION] value head là random => error đầu tiên rất cao
            # MSE Loss sẽ bình phương error này, tạo ra gradient khổng lồ phá vỡ Policy head
            # Dùng smooth_l1_loss (Huber Loss) để giới hạn penalty
            vf_loss = F.smooth_l1_loss(value.squeeze(-1), flat_ret[b])''')

# Replace exact signature of compute_metrics
old_sig = 'def compute_metrics(daily_returns: np.ndarray) -> dict:'
new_sig = 'def compute_metrics(daily_returns: np.ndarray, periods_per_year: int = 24192) -> dict:'
text = text.replace(old_sig, new_sig)

# Replace the 252 in Sharpe calculation
old_sharpe = 'sharpe = (mean_r / std_r * np.sqrt(252)) if std_r > 0 else 0.0'
new_sharpe = 'sharpe = (mean_r / std_r * np.sqrt(periods_per_year)) if std_r > 0 else 0.0'
text = text.replace(old_sharpe, new_sharpe)

# Replace the 252 in Sortino calculation
old_sortino = 'sortino = (mean_r / down_std * np.sqrt(252)) if down_std > 0 else 0.0'
new_sortino = 'sortino = (mean_r / down_std * np.sqrt(periods_per_year)) if down_std > 0 else 0.0'
text = text.replace(old_sortino, new_sortino)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(text)

print("Patched sprint 4")
