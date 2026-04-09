"""
test_confidence_thresholds.py — So sánh Win Rate ở các ngưỡng Confidence khác nhau
==================================================================================
Bot chỉ được vào lệnh khi xác suất dự đoán > threshold, nếu không thì Hold.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import h5py
import numpy as np
from src.model.transformer import XAUTransformer
from src.env.xauusd_env import XAUUSDEnv

def run_with_threshold(model, h5_path, oos_start, n_total, window_size, device, threshold):
    """Chạy backtest với ngưỡng confidence tối thiểu."""
    env = XAUUSDEnv(
        h5_path=h5_path, start_idx=oos_start, end_idx=n_total,
        window_size=window_size, spread_pips=25, lot_size=0.01,
        initial_balance=200.0, max_drawdown_usd=999999.0, random_start=False,
    )
    
    obs, _ = env.reset()
    done = False
    equity_hist = [200.0]
    trades = []
    current_trade = None
    step = 0
    filtered_count = 0  # Số lệnh bị lọc ra
    
    with h5py.File(h5_path, "r") as f:
        close_prices = f["close"][oos_start:n_total].astype(np.float32)
    
    while not done:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = model(obs_t)
            probs = torch.softmax(logits, dim=-1).squeeze().numpy()
            raw_action = logits.argmax(-1).item()
        
        # ÁP DỤNG BỘ LỌC CONFIDENCE
        if raw_action in (1, 2):  # Buy hoặc Sell
            conf = probs[raw_action]
            if conf < threshold:
                action = 0  # HOLD — không đủ tự tin
                filtered_count += 1
            else:
                action = raw_action
        else:
            action = raw_action
        
        prev_pos = env._position_dir
        prev_balance = env._balance
        
        obs, reward, term, trunc, info = env.step(action)
        equity_hist.append(info.get("equity", equity_hist[-1]))
        
        # Tracking trades
        if prev_pos == 0 and env._position_dir != 0:
            current_trade = {
                "direction": "BUY" if env._position_dir == 1 else "SELL",
                "entry_bar": step,
                "equity_at_entry": prev_balance,
                "confidence": float(probs[1] if env._position_dir == 1 else probs[2]),
            }
        
        if prev_pos != 0 and env._position_dir != prev_pos:
            if current_trade is not None:
                pnl = info["balance"] - current_trade["equity_at_entry"]
                current_trade["pnl"] = pnl
                current_trade["duration"] = (step - current_trade["entry_bar"]) * 5
                current_trade["result"] = "WIN" if pnl > 0 else "LOSS"
                trades.append(current_trade)
                
                if env._position_dir != 0:
                    current_trade = {
                        "direction": "BUY" if env._position_dir == 1 else "SELL",
                        "entry_bar": step,
                        "equity_at_entry": info["balance"],
                        "confidence": float(probs[1] if env._position_dir == 1 else probs[2]),
                    }
                else:
                    current_trade = None
        
        done = term or trunc
        step += 1
    
    # Tính metrics
    wins = [t for t in trades if t["result"] == "WIN"]
    losses = [t for t in trades if t["result"] == "LOSS"]
    total = len(trades)
    win_rate = len(wins) / total * 100 if total > 0 else 0
    total_pnl = sum(t["pnl"] for t in trades)
    final_eq = equity_hist[-1]
    
    avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
    avg_loss = np.mean([t["pnl"] for t in losses]) if losses else 0
    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Max Drawdown
    peak = 200.0
    max_dd = 0.0
    for eq in equity_hist:
        peak = max(peak, eq)
        dd = (peak - eq) / peak
        max_dd = max(max_dd, dd)
    
    # Sharpe
    bar_returns = np.diff(equity_hist) / np.array(equity_hist[:-1])
    sharpe = np.mean(bar_returns) / (np.std(bar_returns) + 1e-8) * np.sqrt(252 * 288)
    
    n_days = step * 5 / 60 / 24
    trades_per_day = total / n_days if n_days > 0 else 0
    
    avg_conf_win = np.mean([t["confidence"] for t in wins]) if wins else 0
    avg_conf_loss = np.mean([t["confidence"] for t in losses]) if losses else 0
    
    # Chuỗi thua
    max_consec = 0
    cur = 0
    for t in trades:
        if t["result"] == "LOSS":
            cur += 1
            max_consec = max(max_consec, cur)
        else:
            cur = 0
    
    return {
        "threshold": threshold,
        "total_trades": total,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "final_equity": final_eq,
        "profit_factor": pf,
        "sharpe": sharpe,
        "max_dd": max_dd * 100,
        "trades_per_day": trades_per_day,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "max_consec_loss": max_consec,
        "filtered": filtered_count,
        "avg_conf_win": avg_conf_win,
        "avg_conf_loss": avg_conf_loss,
    }

def main():
    device = torch.device('cpu')
    h5_path = "data/processed/XAUUSD_M5_w256.h5"
    
    with h5py.File(h5_path, "r") as f:
        n_total, window_size, n_features = f["X"].shape
    
    oos_start = int(n_total * 0.8)
    
    model = XAUTransformer(
        n_features=n_features, window_size=window_size,
        d_model=256, n_heads=8, n_layers=6
    ).to(device)
    ckpt = torch.load("checkpoints/ppo_best.pt", map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    
    # Test nhiều ngưỡng
    thresholds = [0.0, 0.35, 0.38, 0.40, 0.42, 0.45, 0.48, 0.50]
    results = []
    
    print("🔬 KIỂM TRA ẢNH HƯỞNG NGƯỠNG CONFIDENCE")
    print("=" * 90)
    
    for th in thresholds:
        label = "Không lọc" if th == 0 else f"{th*100:.0f}%"
        print(f"  Testing threshold {label}...", end=" ", flush=True)
        r = run_with_threshold(model, h5_path, oos_start, n_total, window_size, device, th)
        results.append(r)
        print(f"✅ {r['total_trades']} trades, WR={r['win_rate']:.1f}%, PnL=${r['total_pnl']:+.2f}")
    
    # Bảng so sánh
    print(f"\n{'='*110}")
    print(f"📋 BẢNG SO SÁNH CÁC NGƯỠNG CONFIDENCE")
    print(f"{'='*110}")
    print(f"  {'Ngưỡng':>8s} {'Trades':>7s} {'Win':>5s} {'Loss':>5s} {'WinRate':>8s} {'PnL':>12s} "
          f"{'Equity':>10s} {'PF':>6s} {'Sharpe':>8s} {'MaxDD':>8s} {'T/Day':>6s} {'AvgW':>8s} {'AvgL':>8s} {'Streak':>7s}")
    print(f"  {'─'*8} {'─'*7} {'─'*5} {'─'*5} {'─'*8} {'─'*12} "
          f"{'─'*10} {'─'*6} {'─'*8} {'─'*8} {'─'*6} {'─'*8} {'─'*8} {'─'*7}")
    
    for r in results:
        label = "Không lọc" if r["threshold"] == 0 else f"{r['threshold']*100:.0f}%"
        print(
            f"  {label:>8s} {r['total_trades']:>7d} {r['wins']:>5d} {r['losses']:>5d} "
            f"{r['win_rate']:>7.1f}% ${r['total_pnl']:>+10.2f} "
            f"${r['final_equity']:>9.2f} {r['profit_factor']:>6.2f} {r['sharpe']:>8.2f} "
            f"{r['max_dd']:>7.1f}% {r['trades_per_day']:>5.1f} "
            f"${r['avg_win']:>+6.2f} ${r['avg_loss']:>+6.2f} {r['max_consec_loss']:>5d}L"
        )
    
    # Khuyến nghị
    print(f"\n{'='*110}")
    print(f"🎯 PHÂN TÍCH & KHUYẾN NGHỊ")
    print(f"{'='*110}")
    
    best_sharpe = max(results, key=lambda x: x["sharpe"])
    best_wr = max(results, key=lambda x: x["win_rate"])
    best_pnl = max(results, key=lambda x: x["total_pnl"])
    
    bs_label = "Không lọc" if best_sharpe["threshold"] == 0 else f"{best_sharpe['threshold']*100:.0f}%"
    bw_label = "Không lọc" if best_wr["threshold"] == 0 else f"{best_wr['threshold']*100:.0f}%"
    bp_label = "Không lọc" if best_pnl["threshold"] == 0 else f"{best_pnl['threshold']*100:.0f}%"
    
    print(f"  🏆 Sharpe cao nhất:   Ngưỡng {bs_label} (Sharpe={best_sharpe['sharpe']:.2f})")
    print(f"  🏆 Win Rate cao nhất: Ngưỡng {bw_label} (WR={best_wr['win_rate']:.1f}%)")
    print(f"  🏆 PnL cao nhất:     Ngưỡng {bp_label} (PnL=${best_pnl['total_pnl']:+.2f})")
    
    # Lưu
    with open("logs/confidence_threshold_test.txt", "w", encoding="utf-8") as f:
        f.write("Confidence Threshold Test Results\n")
        for r in results:
            f.write(f"Threshold={r['threshold']:.2f}: Trades={r['total_trades']}, WR={r['win_rate']:.1f}%, "
                    f"PnL=${r['total_pnl']:+.2f}, Sharpe={r['sharpe']:.2f}, MaxDD={r['max_dd']:.1f}%\n")
    print(f"\n📄 Saved → logs/confidence_threshold_test.txt")

if __name__ == "__main__":
    main()
