import torch
import h5py
import numpy as np
import sys
from pathlib import Path

# Thêm đường dẫn gốc để import module
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.model.transformer import XAUTransformer
from src.env.xauusd_env import XAUUSDEnv

def run_backtest(ckpt_path, h5_path, label):
    with h5py.File(h5_path, "r") as f:
        n_total = f["X"].shape[0]
        n_features = f["X"].shape[2]
        window_size = f["X"].shape[1]
    
    # Backtest trên 20% data cuối cùng (Out-Of-Sample)
    oos_start = int(n_total * 0.8)
    
    print(f"\n{'-'*50}")
    print(f"Bắt đầu Backtest: {label}")
    print(f"Data: {oos_start} -> {n_total} ({n_total - oos_start} mẫu)")
    
    model = XAUTransformer(n_features=n_features, window_size=window_size, d_model=256, n_heads=8, n_layers=6)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    env = XAUUSDEnv(h5_path=h5_path, start_idx=oos_start, end_idx=n_total,
                    window_size=window_size, spread_pips=30, lot_size=0.01,
                    initial_balance=200.0, max_drawdown_usd=999999.0, random_start=False)
    
    obs, _ = env.reset()
    done = False
    equity_hist = [200.0]
    trades_won, trades_lost = 0, 0
    prev_pos = 0
    prev_balance = 200.0
    action_counts = {0:0, 1:0, 2:0}
    
    while not done:
        tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, _ = model(tensor)
            probs = torch.softmax(logits, dim=-1).squeeze().numpy()
            
        action = int(np.argmax(probs))
        conf = float(probs[action])
        
        # Chỉ vào lệnh khi confidence > 45%
        if action in (1, 2) and conf < 0.45:
            action = 0
            
        action_counts[action] += 1
            
        prev_pos = env._position_dir
        prev_balance = env._balance
        obs, _, term, trunc, info = env.step(action)
        equity_hist.append(info.get("equity", equity_hist[-1]))

        if prev_pos != 0 and env._position_dir != prev_pos:
            pnl = env._balance - prev_balance
            if pnl > 0: trades_won += 1
            elif pnl < 0: trades_lost += 1

        done = term or trunc

    total_trades = trades_won + trades_lost
    win_rate = trades_won / total_trades if total_trades > 0 else 0

    eq = np.array(equity_hist)
    returns = np.diff(eq) / eq[:-1]
    sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 288)

    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / peak
    max_dd = float(np.max(dd))
    
    # Tính lại tổng Action (trừ HOLD là đóng lệnh tự động của ENV)
    c_hold = action_counts[0]
    c_buy = action_counts[1]
    c_sell = action_counts[2]

    print(f"📊 Kết quả: {label}")
    print(f" + PnL:      ${eq[-1] - 200:.2f} (Total Eq: ${eq[-1]:.2f})")
    print(f" + Trades:   {total_trades} (Thắng {trades_won} - Thua {trades_lost})")
    print(f" + Win Rate: {win_rate*100:.1f}%")
    print(f" + Max DD:   {max_dd*100:.1f}%")
    print(f" + Sharpe:   {sharpe:.2f}")
    print(f" + Actions:  BUY={c_buy} | SELL={c_sell} | HOLD={c_hold}")
    print(f" + Tỉ lệ B/S: {c_buy/(c_sell+1e-8):.2f}")

h5_file = "data/processed/XAUUSD_M5_w256.h5"

print("Đang chạy Backtest...")
run_backtest("checkpoints/best_model_bc.pt", h5_file, "OLD BC MODEL (Bị lệch bán)")
run_backtest("checkpoints/fresh_6month_best_model_bc.pt", h5_file, "NEW BC MODEL (Train 6 tháng)")

