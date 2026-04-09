"""
backtest.py
-----------
Out-of-sample backtest: chÃ¡ÂºÂ¡y model trained qua dÃ¡Â»Â¯ liÃ¡Â»â€¡u chÃ†Â°a thÃ¡ÂºÂ¥y,
tÃƒÂ­nh cÃƒÂ¡c chÃ¡Â»Â‰ sÃ¡Â»Â‘ tÃƒÂ i chÃƒÂ­nh chuÃ¡ÂºÂ©n.

CÃƒÂ¡ch dÃƒÂ¹ng:
  python src/training/backtest.py \\
      --h5      data/processed/XAUUSD_M15_w128.h5 \\
      --ckpt    checkpoints/ppo_xauusd.zip \\
      --mode    ppo                         # hoÃ¡ÂºÂ·c bc

ChÃ¡Â»Â‰ sÃ¡Â»Â‘ output:
  Sharpe Ratio, Sortino Ratio, Max Drawdown, Win Rate, Total Return
"""

import argparse
import logging
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def compute_metrics(daily_returns: np.ndarray, periods_per_year: int = 24192, positions: np.ndarray = None) -> dict:
    """
    TÃƒÂ­nh cÃƒÂ¡c chÃ¡Â»Â‰ sÃ¡Â»Â‘ tÃƒÂ i chÃƒÂ­nh tÃ¡Â»Â« mÃ¡ÂºÂ£ng daily returns.

    Parameters
    ----------
    daily_returns : np.ndarray Ã¢â‚¬â€ mÃ¡ÂºÂ£ng % return mÃ¡Â»â€”i bar/ngÃƒÂ y

    Returns
    -------
    dict vÃ¡Â»â€ºi keys: sharpe, sortino, max_drawdown, win_rate,
                   total_return, n_trades
    """
    r = np.array(daily_returns, dtype=np.float64)
    n = len(r)

    if n == 0:
        return dict(sharpe=0.0, sortino=0.0, max_drawdown=0.0,
                    win_rate=0.0, total_return=0.0, n_trades=0)

    mean_r = r.mean()
    std_r  = r.std()

    # Sharpe Ratio (annualized Ãƒâ€”Ã¢Ë†Å¡252 nÃ¡ÂºÂ¿u daily, Ãƒâ€”Ã¢Ë†Å¡(252Ãƒâ€”24Ãƒâ€”4) nÃ¡ÂºÂ¿u M15)
    sharpe = (mean_r / std_r * np.sqrt(periods_per_year)) if std_r > 0 else 0.0

    # Sortino Ratio (chÃ¡Â»â€° dÃƒÂ¹ng downside std)
    negative_r  = r[r < 0]
    down_std    = negative_r.std() if len(negative_r) > 0 else 0.0
    sortino = (mean_r / down_std * np.sqrt(periods_per_year)) if down_std > 0 else 0.0

    # Max Drawdown
    cumulative = np.cumprod(1 + r)
    peak       = np.maximum.accumulate(cumulative)
    drawdown   = (cumulative - peak) / peak
    max_dd     = float(drawdown.min())

    # Win Rate
    win_rate = float((r > 0).mean())

    # Total Return
    total_return = float(cumulative[-1] - 1.0) if n > 0 else 0.0

    # [FIX N_TRADES] Thay vì đếm số nến có return != 0 (do Equity nến nào cũng đổi),
    # đếm số lần position_dir thay đổi để tránh ảo tưởng trade.
    if positions is not None:
        pos_diff = np.abs(np.diff(positions))
        n_trades = int(np.sum(pos_diff > 0)) # Mỗi lần đổi trạng thái là 1 giao dịch
    else:
        n_trades = int((r != 0).sum())

    return dict(
        sharpe       = round(sharpe,       4),
        sortino      = round(sortino,      4),
        max_drawdown = round(max_dd,       4),
        win_rate     = round(win_rate,     4),
        total_return = round(total_return, 4),
        n_trades     = n_trades,
    )


def print_report(metrics: dict, label: str = "OUT-OF-SAMPLE"):
    log.info(f"\n{'='*55}")
    log.info(f"  BACKTEST REPORT Ã¢â‚¬â€ {label}")
    log.info(f"{'='*55}")
    log.info(f"  Sharpe Ratio   : {metrics['sharpe']:>8.4f}  (Target: >1.0)")
    log.info(f"  Sortino Ratio  : {metrics['sortino']:>8.4f}")
    log.info(f"  Max Drawdown   : {metrics['max_drawdown']:>8.2%}  (Target: >-10%)")
    log.info(f"  Win Rate       : {metrics['win_rate']:>8.2%}  (Target: >55%)")
    log.info(f"  Total Return   : {metrics['total_return']:>8.2%}")
    log.info(f"  N Trades       : {metrics['n_trades']:>8,}")

    # PASS/FAIL
    passed = (
        metrics["sharpe"]       >= 1.0 and
        metrics["win_rate"]     >= 0.55 and
        metrics["max_drawdown"] >= -0.10
    )
    status = "Ã¢Å“â€¦ PASS Ã¢â‚¬â€ Ã„ÂÃ¡Â»Â§ tiÃƒÂªu chuÃ¡ÂºÂ©n deploy" if passed else "Ã¢ÂÅ’ FAIL Ã¢â‚¬â€ CÃ¡ÂºÂ§n cÃ¡ÂºÂ£i thiÃ¡Â»â€¡n thÃƒÂªm"
    log.info(f"\n  {status}")
    log.info(f"{'='*55}\n")
    return passed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5",    required=True)
    p.add_argument("--ckpt",  required=True)
    p.add_argument("--mode",  choices=["bc", "ppo"], default="ppo")
    args = p.parse_args()

    log.info(f"Backtest mode: {args.mode.upper()}")
    log.info(f"Checkpoint: {args.ckpt}")

    from src.model.transformer import XAUTransformer
    from src.env.xauusd_env import XAUUSDEnv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with h5py.File(args.h5, "r") as f:
        n_total, window_size, n_features = f["X"].shape

    GAP_BARS = 200
    split_idx = int(n_total * 0.8) - GAP_BARS
    oos_start = split_idx + GAP_BARS

    model = XAUTransformer(
        n_features=n_features, window_size=window_size,
        d_model=256, n_heads=8, n_layers=6
    ).to(device)

    # Load model
    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
    # Check if checkpoint is an optimizer dictionary or raw state dict
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()

    env = XAUUSDEnv(
        h5_path          = args.h5,
        start_idx        = oos_start,
        end_idx          = n_total,
        window_size      = window_size,
        spread_pips      = 25,
        lot_size         = 0.01,
        initial_balance  = 200.0,
        max_drawdown_usd = 999999.0, # Cho phép thả nổi để xem tổng rủi ro
        random_start     = False, 
    )

    obs, _ = env.reset()
    done = False
    
    equity_hist = [200.0]
    position_hist = [0]
    
    log.info(f"Bắt đầu Backtest V22 từ index {oos_start} đến {n_total}...")
    
    while not done:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = model(obs_t)
            action = logits.argmax(-1).item()  # Deterministic (greedy) strategy for testing
            
        obs, _, term, trunc, info = env.step(action)
        equity_hist.append(info.get("equity", 200.0))
        position_hist.append(env._position_dir)
        done = term or trunc

    # Calculate bar returns
    bar_returns = np.diff(equity_hist) / np.array(equity_hist[:-1])
    
    # Generate report
    metrics = compute_metrics(bar_returns, positions=np.array(position_hist))
    
    # Print it
    print_report(metrics, label=f"OOS BACKTEST V22 | {args.ckpt}")


if __name__ == "__main__":
    main()