"""
backtest_bc_m5.py — Backtest BC Model trên dữ liệu M5 OOS
==========================================================
Chạy simulator xuyên suốt vùng Out-of-Sample, xuất báo cáo chi tiết.
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

def main():
    device = torch.device('cpu')  # Backtest nhẹ, chạy CPU cho an toàn
    h5_path = "data/processed/XAUUSD_M5_w256.h5"
    
    # 1. Đọc metadata
    with h5py.File(h5_path, "r") as f:
        n_total, window_size, n_features = f["X"].shape
    
    GAP_BARS = 200
    split_idx = int(n_total * 0.8) - GAP_BARS
    oos_start = int(n_total * 0.8)
    n_test = n_total - oos_start
    
    print(f"📊 Dataset: {h5_path}")
    print(f"   Tổng samples: {n_total:,}")
    print(f"   Window size: {window_size}")
    print(f"   Features: {n_features}")
    print(f"   OOS range: [{oos_start} → {n_total}] ({n_test:,} bars)")
    print(f"   Tương đương: ~{n_test * 5 / 60:.0f} giờ = ~{n_test * 5 / 60 / 24:.0f} ngày trading\n")
    
    # 2. Load Model
    model = XAUTransformer(
        n_features=n_features, window_size=window_size,
        d_model=256, n_heads=8, n_layers=6
    ).to(device)
    
    ckpt = torch.load("checkpoints/best_model_bc.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"✅ Model loaded (Epoch {ckpt.get('epoch', '?')})\n")
    
    # 3. Chạy Simulator OOS
    env = XAUUSDEnv(
        h5_path=h5_path,
        start_idx=oos_start,
        end_idx=n_total,
        window_size=window_size,
        spread_pips=25,
        lot_size=0.01,
        initial_balance=200.0,
        max_drawdown_usd=999999.0,
        random_start=False,
    )
    
    obs, _ = env.reset()
    done = False
    equity_hist = [200.0]
    position_hist = [0]
    action_counts = {0: 0, 1: 0, 2: 0}  # Hold, Buy, Sell
    trade_count = 0
    
    print("🚀 Bắt đầu Backtest OOS...")
    step = 0
    while not done:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = model(obs_t)
            action = logits.argmax(-1).item()
        
        prev_pos = env._position_dir
        obs, reward, term, trunc, info = env.step(action)
        
        equity_hist.append(info.get("equity", equity_hist[-1]))
        position_hist.append(env._position_dir)
        action_counts[action] += 1
        
        # Đếm trade (khi position thay đổi)
        if env._position_dir != prev_pos and env._position_dir != 0:
            trade_count += 1
        
        done = term or trunc
        step += 1
        
        if step % 500 == 0:
            print(f"   Step {step:,}/{n_test:,} | Equity: ${equity_hist[-1]:.2f} | Trades: {trade_count}")
    
    # 4. Tính toán Metrics
    bar_returns = np.diff(equity_hist) / np.array(equity_hist[:-1])
    metrics = compute_metrics(bar_returns, positions=np.array(position_hist))
    survival_rate = len(equity_hist) / float(n_test) * 100
    
    final_equity = equity_hist[-1]
    total_pnl = final_equity - 200.0
    
    # 5. Báo cáo
    report = f"""
{'='*60}
📋 BÁO CÁO BACKTEST BC MODEL — M5 SNIPER (w256)
{'='*60}

🔧 Cấu hình:
   Dataset:      XAUUSD_M5_w256.h5
   OOS Period:   {n_test:,} bars (~{n_test * 5 / 60 / 24:.0f} ngày)
   Initial:      $200.00
   Lot Size:     0.01
   Spread:       25 pips

💰 Kết quả:
   Final Equity:   ${final_equity:.2f}
   Total PnL:      ${total_pnl:+.2f} ({total_pnl/200*100:+.1f}%)
   Sharpe Ratio:   {metrics['sharpe']:.4f}
   Sortino Ratio:  {metrics['sortino']:.4f}
   Max Drawdown:   {metrics['max_drawdown']*100:.2f}%
   Win Rate:       {metrics['win_rate']*100:.1f}%
   Total Return:   {metrics['total_return']*100:.2f}%
   Survival Rate:  {survival_rate:.1f}%

📊 Phân bổ lệnh:
   Hold:  {action_counts[0]:,} ({action_counts[0]/(step)*100:.1f}%)
   Buy:   {action_counts[1]:,} ({action_counts[1]/(step)*100:.1f}%)
   Sell:  {action_counts[2]:,} ({action_counts[2]/(step)*100:.1f}%)
   Trades thực tế: {trade_count}

{'='*60}
"""
    print(report)
    
    # Lưu report
    with open("logs/backtest_bc_m5_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print("📄 Report saved → logs/backtest_bc_m5_report.txt")

if __name__ == "__main__":
    main()
