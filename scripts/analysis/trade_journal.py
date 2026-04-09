"""
trade_journal.py — Sổ Nhật Ký Giao Dịch Chi Tiết
==================================================
Chạy Backtest và ghi lại TỪNG LỆNH bot vào/ra,
xuất bảng thống kê để anh phân tích bot sai ở đâu.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import h5py
import numpy as np
from datetime import datetime, timedelta
from src.model.transformer import XAUTransformer
from src.env.xauusd_env import XAUUSDEnv

def main():
    device = torch.device('cpu')
    h5_path = "data/processed/XAUUSD_M5_w256.h5"
    ckpt_path = "checkpoints/ppo_best.pt"
    
    with h5py.File(h5_path, "r") as f:
        n_total, window_size, n_features = f["X"].shape
    
    oos_start = int(n_total * 0.8)
    n_test = n_total - oos_start
    
    # Load Model
    model = XAUTransformer(
        n_features=n_features, window_size=window_size,
        d_model=256, n_heads=8, n_layers=6
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    
    # Load close prices để ghi nhận giá
    with h5py.File(h5_path, "r") as f:
        close_prices = f["close"][oos_start:n_total].astype(np.float32)
        open_next = f["open_next"][oos_start:n_total].astype(np.float32)
    
    # Chạy Backtest & ghi nhật ký
    env = XAUUSDEnv(
        h5_path=h5_path, start_idx=oos_start, end_idx=n_total,
        window_size=window_size, spread_pips=25, lot_size=0.01,
        initial_balance=200.0, max_drawdown_usd=999999.0, random_start=False,
    )
    
    obs, _ = env.reset()
    done = False
    
    trades = []         # Danh sách lệnh đã đóng
    current_trade = None  # Lệnh đang mở
    equity_hist = [200.0]
    step = 0
    
    while not done:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = model(obs_t)
            probs = torch.softmax(logits, dim=-1).squeeze().numpy()
            action = logits.argmax(-1).item()
        
        prev_pos = env._position_dir
        prev_entry = env._entry_price
        prev_balance = env._balance
        
        obs, reward, term, trunc, info = env.step(action)
        
        cur_price = float(close_prices[min(step, len(close_prices)-1)])
        equity = info.get("equity", equity_hist[-1])
        equity_hist.append(equity)
        
        # Phát hiện MỞ LỆNH
        if prev_pos == 0 and env._position_dir != 0:
            current_trade = {
                "id": len(trades) + 1,
                "direction": "BUY" if env._position_dir == 1 else "SELL",
                "entry_bar": step,
                "entry_price": env._entry_price,
                "confidence": {
                    "hold": float(probs[0]),
                    "buy": float(probs[1]),
                    "sell": float(probs[2]),
                },
                "equity_at_entry": prev_balance,
            }
        
        # Phát hiện ĐÓNG LỆNH
        if prev_pos != 0 and env._position_dir != prev_pos:
            if current_trade is not None:
                exit_price = float(open_next[min(step, len(open_next)-1)])
                pnl = info["balance"] - current_trade["equity_at_entry"]
                
                # Nếu là reversal, PnL tính từ balance change
                if env._position_dir != 0:
                    # Reversal: đóng lệnh cũ + mở lệnh mới
                    current_trade["exit_type"] = "REVERSAL"
                else:
                    current_trade["exit_type"] = "CLOSE"
                
                current_trade["exit_bar"] = step
                current_trade["exit_price"] = exit_price
                current_trade["pnl"] = pnl
                current_trade["duration_bars"] = step - current_trade["entry_bar"]
                current_trade["duration_mins"] = current_trade["duration_bars"] * 5
                current_trade["result"] = "WIN" if pnl > 0 else "LOSS"
                
                trades.append(current_trade)
                
                # Nếu reversal, bắt đầu trade mới
                if env._position_dir != 0:
                    current_trade = {
                        "id": len(trades) + 1,
                        "direction": "BUY" if env._position_dir == 1 else "SELL",
                        "entry_bar": step,
                        "entry_price": env._entry_price,
                        "confidence": {
                            "hold": float(probs[0]),
                            "buy": float(probs[1]),
                            "sell": float(probs[2]),
                        },
                        "equity_at_entry": info["balance"],
                    }
                else:
                    current_trade = None
        
        done = term or trunc
        step += 1
    
    # ═══════════════════════════════════════════════════════════
    # BÁO CÁO
    # ═══════════════════════════════════════════════════════════
    
    wins = [t for t in trades if t["result"] == "WIN"]
    losses = [t for t in trades if t["result"] == "LOSS"]
    buys = [t for t in trades if t["direction"] == "BUY"]
    sells = [t for t in trades if t["direction"] == "SELL"]
    buy_wins = [t for t in buys if t["result"] == "WIN"]
    sell_wins = [t for t in sells if t["result"] == "WIN"]
    
    avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
    avg_loss = np.mean([t["pnl"] for t in losses]) if losses else 0
    avg_dur_win = np.mean([t["duration_mins"] for t in wins]) if wins else 0
    avg_dur_loss = np.mean([t["duration_mins"] for t in losses]) if losses else 0
    
    # Chuỗi thua liên tiếp
    max_consec_loss = 0
    cur_consec = 0
    for t in trades:
        if t["result"] == "LOSS":
            cur_consec += 1
            max_consec_loss = max(max_consec_loss, cur_consec)
        else:
            cur_consec = 0
    
    # Profit Factor
    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    report = []
    report.append("=" * 80)
    report.append("📋 SỔ NHẬT KÝ GIAO DỊCH — PPO Best M5 Sniper (w256)")
    report.append("=" * 80)
    
    # TỔNG QUAN
    report.append(f"\n{'─'*80}")
    report.append(f"📊 TỔNG QUAN")
    report.append(f"{'─'*80}")
    report.append(f"  Tổng số lệnh:        {len(trades)}")
    report.append(f"  Lệnh Win:            {len(wins)} ({len(wins)/len(trades)*100:.1f}%)")
    report.append(f"  Lệnh Loss:           {len(losses)} ({len(losses)/len(trades)*100:.1f}%)")
    report.append(f"  Profit Factor:       {profit_factor:.2f}")
    report.append(f"  Chuỗi thua dài nhất: {max_consec_loss} lệnh liên tiếp")
    report.append(f"  PnL ròng:            ${sum(t['pnl'] for t in trades):+.2f}")
    
    # PHÂN TÍCH THEO CHIỀU
    report.append(f"\n{'─'*80}")
    report.append(f"📈 PHÂN TÍCH THEO CHIỀU GIAO DỊCH")
    report.append(f"{'─'*80}")
    report.append(f"  {'':20s} {'BUY':>12s} {'SELL':>12s}")
    report.append(f"  {'Tổng lệnh':20s} {len(buys):>12d} {len(sells):>12d}")
    report.append(f"  {'Win':20s} {len(buy_wins):>12d} {len(sell_wins):>12d}")
    report.append(f"  {'Win Rate':20s} {len(buy_wins)/max(len(buys),1)*100:>11.1f}% {len(sell_wins)/max(len(sells),1)*100:>11.1f}%")
    buy_pnl = sum(t["pnl"] for t in buys)
    sell_pnl = sum(t["pnl"] for t in sells)
    report.append(f"  {'PnL':20s} ${buy_pnl:>+11.2f} ${sell_pnl:>+11.2f}")
    
    # THỐNG KÊ PNL
    report.append(f"\n{'─'*80}")
    report.append(f"💰 THỐNG KÊ PNL")
    report.append(f"{'─'*80}")
    report.append(f"  TB lệnh Win:         ${avg_win:+.2f}")
    report.append(f"  TB lệnh Loss:        ${avg_loss:+.2f}")
    report.append(f"  Win lớn nhất:        ${max(t['pnl'] for t in wins):+.2f}" if wins else "")
    report.append(f"  Loss lớn nhất:       ${min(t['pnl'] for t in losses):+.2f}" if losses else "")
    report.append(f"  TB thời gian Win:    {avg_dur_win:.0f} phút ({avg_dur_win/5:.0f} nến)")
    report.append(f"  TB thời gian Loss:   {avg_dur_loss:.0f} phút ({avg_dur_loss/5:.0f} nến)")
    
    # BẢNG CHI TIẾT 20 LỆNH LỖ NẶNG NHẤT
    report.append(f"\n{'─'*80}")
    report.append(f"🔴 TOP 20 LỆNH LỖ NẶNG NHẤT (Để phân tích bot sai ở đâu)")
    report.append(f"{'─'*80}")
    report.append(f"  {'#':>4s} {'Dir':>5s} {'Entry':>10s} {'Exit':>10s} {'PnL':>10s} {'Dur':>8s} {'Conf%':>8s} {'Exit':>8s}")
    report.append(f"  {'─'*4} {'─'*5} {'─'*10} {'─'*10} {'─'*10} {'─'*8} {'─'*8} {'─'*8}")
    
    worst = sorted(trades, key=lambda t: t["pnl"])[:20]
    for t in worst:
        dir_conf = t["confidence"]["buy"] if t["direction"] == "BUY" else t["confidence"]["sell"]
        report.append(
            f"  {t['id']:>4d} {t['direction']:>5s} "
            f"{t['entry_price']:>10.2f} {t['exit_price']:>10.2f} "
            f"${t['pnl']:>+9.2f} {t['duration_mins']:>6d}m "
            f"{dir_conf*100:>7.1f}% {t['exit_type']:>8s}"
        )
    
    # BẢNG CHI TIẾT 20 LỆNH LÃI LỚN NHẤT
    report.append(f"\n{'─'*80}")
    report.append(f"🟢 TOP 20 LỆNH LÃI LỚN NHẤT")
    report.append(f"{'─'*80}")
    report.append(f"  {'#':>4s} {'Dir':>5s} {'Entry':>10s} {'Exit':>10s} {'PnL':>10s} {'Dur':>8s} {'Conf%':>8s} {'Exit':>8s}")
    report.append(f"  {'─'*4} {'─'*5} {'─'*10} {'─'*10} {'─'*10} {'─'*8} {'─'*8} {'─'*8}")
    
    best = sorted(trades, key=lambda t: t["pnl"], reverse=True)[:20]
    for t in best:
        dir_conf = t["confidence"]["buy"] if t["direction"] == "BUY" else t["confidence"]["sell"]
        report.append(
            f"  {t['id']:>4d} {t['direction']:>5s} "
            f"{t['entry_price']:>10.2f} {t['exit_price']:>10.2f} "
            f"${t['pnl']:>+9.2f} {t['duration_mins']:>6d}m "
            f"{dir_conf*100:>7.1f}% {t['exit_type']:>8s}"
        )
    
    # ĐƯỜNG EQUITY
    report.append(f"\n{'─'*80}")
    report.append(f"📉 ĐƯỜNG EQUITY (Mini-chart)")
    report.append(f"{'─'*80}")
    
    # Chia thành 50 điểm
    chart_points = 50
    step_size = max(len(equity_hist) // chart_points, 1)
    sampled = [equity_hist[i] for i in range(0, len(equity_hist), step_size)]
    eq_min, eq_max = min(sampled), max(sampled)
    chart_height = 15
    
    for row in range(chart_height, -1, -1):
        threshold = eq_min + (eq_max - eq_min) * row / chart_height
        line = "  "
        if row == chart_height:
            line += f"${eq_max:>7.0f} │"
        elif row == 0:
            line += f"${eq_min:>7.0f} │"
        elif row == chart_height // 2:
            mid = (eq_max + eq_min) / 2
            line += f"${mid:>7.0f} │"
        else:
            line += f"{'':>8s}│"
        
        for eq in sampled:
            if eq >= threshold:
                line += "█"
            else:
                line += " "
        report.append(line)
    report.append(f"  {'':>8s}└{'─' * len(sampled)}")
    
    # TOÀN BỘ LỆNH
    report.append(f"\n{'─'*80}")
    report.append(f"📝 DANH SÁCH TOÀN BỘ LỆNH ({len(trades)} lệnh)")
    report.append(f"{'─'*80}")
    report.append(f"  {'#':>4s} {'Dir':>5s} {'Entry$':>10s} {'Exit$':>10s} {'PnL':>10s} {'Dur':>8s} {'Result':>7s} {'Conf%':>7s}")
    report.append(f"  {'─'*4} {'─'*5} {'─'*10} {'─'*10} {'─'*10} {'─'*8} {'─'*7} {'─'*7}")
    
    for t in trades:
        dir_conf = t["confidence"]["buy"] if t["direction"] == "BUY" else t["confidence"]["sell"]
        marker = "✅" if t["result"] == "WIN" else "❌"
        report.append(
            f"  {t['id']:>4d} {t['direction']:>5s} "
            f"{t['entry_price']:>10.2f} {t['exit_price']:>10.2f} "
            f"${t['pnl']:>+9.2f} {t['duration_mins']:>6d}m "
            f"{marker:>5s} {dir_conf*100:>6.1f}%"
        )
    
    report.append(f"\n{'='*80}")
    
    # In & Lưu
    full_report = "\n".join(report)
    print(full_report)
    
    with open("logs/trade_journal.txt", "w", encoding="utf-8") as f:
        f.write(full_report)
    print(f"\n📄 Saved → logs/trade_journal.txt")

if __name__ == "__main__":
    main()
