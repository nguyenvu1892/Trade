"""
backtest_ppo_m5.py — Backtest PPO Model + So sánh BC vs PPO
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import h5py
import numpy as np
from src.model.transformer import XAUTransformer
from src.env.xauusd_env import XAUUSDEnv
from src.training.backtest import compute_metrics

def run_backtest(model, h5_path, oos_start, n_total, window_size, device, label="Model"):
    env = XAUUSDEnv(
        h5_path=h5_path, start_idx=oos_start, end_idx=n_total,
        window_size=window_size, spread_pips=25, lot_size=0.01,
        initial_balance=200.0, max_drawdown_usd=999999.0, random_start=False,
    )
    
    obs, _ = env.reset()
    done = False
    equity_hist = [200.0]
    position_hist = [0]
    action_counts = {0: 0, 1: 0, 2: 0}
    trade_count = 0
    step = 0
    
    while not done:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = model(obs_t)
            action = logits.argmax(-1).item()
        prev_pos = env._position_dir
        obs, _, term, trunc, info = env.step(action)
        equity_hist.append(info.get("equity", equity_hist[-1]))
        position_hist.append(env._position_dir)
        action_counts[action] += 1
        if env._position_dir != prev_pos and env._position_dir != 0:
            trade_count += 1
        done = term or trunc
        step += 1
    
    bar_returns = np.diff(equity_hist) / np.array(equity_hist[:-1])
    metrics = compute_metrics(bar_returns, positions=np.array(position_hist))
    final_eq = equity_hist[-1]
    pnl = final_eq - 200.0
    
    return {
        "label": label,
        "final_equity": final_eq,
        "pnl": pnl,
        "pnl_pct": pnl / 200 * 100,
        "sharpe": metrics["sharpe"],
        "sortino": metrics["sortino"],
        "max_dd": metrics["max_drawdown"] * 100,
        "win_rate": metrics["win_rate"] * 100,
        "trades": trade_count,
        "hold_pct": action_counts[0] / step * 100,
        "buy_pct": action_counts[1] / step * 100,
        "sell_pct": action_counts[2] / step * 100,
    }

def main():
    device = torch.device('cpu')
    h5_path = "data/processed/XAUUSD_M5_w256.h5"
    
    with h5py.File(h5_path, "r") as f:
        n_total, window_size, n_features = f["X"].shape
    
    oos_start = int(n_total * 0.8)
    n_test = n_total - oos_start
    
    print(f"📊 OOS: {n_test:,} bars (~{n_test*5/60/24:.0f} ngày)")
    print(f"   Window: {window_size} | Features: {n_features}\n")
    
    results = []
    
    for ckpt_name, label in [
        ("checkpoints/best_model_bc.pt", "BC (Phase 1)"),
        ("checkpoints/ppo_best.pt", "PPO Best (Phase 2)"),
        ("checkpoints/ppo_final.pt", "PPO Final (Phase 2)"),
    ]:
        print(f"🔄 Backtest: {label}...")
        model = XAUTransformer(
            n_features=n_features, window_size=window_size,
            d_model=256, n_heads=8, n_layers=6
        ).to(device)
        
        ckpt = torch.load(ckpt_name, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
        else:
            # PPO saves raw state_dict (OrderedDict)
            model.load_state_dict(ckpt)
        model.eval()
        
        r = run_backtest(model, h5_path, oos_start, n_total, window_size, device, label)
        results.append(r)
        print(f"   ✅ Equity: ${r['final_equity']:.2f} | Sharpe: {r['sharpe']:.4f}")
    
    # Report
    print(f"\n{'='*70}")
    print(f"📋 SO SÁNH BC vs PPO — M5 SNIPER (w256)")
    print(f"{'='*70}")
    print(f"{'Chỉ số':<20} {'BC Phase 1':>15} {'PPO Best':>15} {'PPO Final':>15}")
    print(f"{'-'*70}")
    
    for key, fmt, name in [
        ("final_equity", "${:.2f}", "Final Equity"),
        ("pnl", "${:+.2f}", "PnL"),
        ("pnl_pct", "{:+.1f}%", "PnL %"),
        ("sharpe", "{:.4f}", "Sharpe Ratio"),
        ("sortino", "{:.4f}", "Sortino Ratio"),
        ("max_dd", "{:.2f}%", "Max Drawdown"),
        ("win_rate", "{:.1f}%", "Win Rate"),
        ("trades", "{}", "Trades"),
        ("hold_pct", "{:.1f}%", "Hold %"),
        ("buy_pct", "{:.1f}%", "Buy %"),
        ("sell_pct", "{:.1f}%", "Sell %"),
    ]:
        vals = [fmt.format(r[key]) for r in results]
        print(f"{name:<20} {vals[0]:>15} {vals[1]:>15} {vals[2]:>15}")
    
    print(f"{'='*70}")
    
    # Lưu report
    best = max(results, key=lambda x: x["sharpe"])
    print(f"\n🏆 WINNER: {best['label']} (Sharpe={best['sharpe']:.4f})")
    
    with open("logs/backtest_comparison_m5.txt", "w", encoding="utf-8") as f:
        f.write(f"BC vs PPO Comparison — M5 Sniper w256\n")
        for r in results:
            f.write(f"\n{r['label']}: Equity=${r['final_equity']:.2f}, Sharpe={r['sharpe']:.4f}, MaxDD={r['max_dd']:.2f}%, Trades={r['trades']}\n")
    print("📄 Saved → logs/backtest_comparison_m5.txt")

if __name__ == "__main__":
    main()
