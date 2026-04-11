import torch
import numpy as np
import h5py
from src.model.transformer import XAUTransformer
from src.env.xauusd_env import XAUUSDEnv
from torch.utils.data import TensorDataset, DataLoader

def fast_batched_sim():
    print("Loading Dataset & Model for Batched Simulation...")
    device = torch.device("cpu")
    model = XAUTransformer(n_features=15, window_size=256, d_model=256, n_heads=8, n_layers=6).to(device)
    
    ckpt = torch.load("checkpoints/best_model_cme_sniper_v2.pt", map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    
    h5_path = "data/processed/XAUUSD_M5_w256.h5"
    with h5py.File(h5_path, "r") as f:
        n_total = f["X"].shape[0]
        
    GAP_BARS = 200
    split_idx = int(n_total * 0.8) - GAP_BARS
    oos_start = split_idx + GAP_BARS
    # Grab 25,000 steps OOS for fast eval (~ 4 tháng dữ liệu M5)
    max_steps = min(25000, n_total - oos_start - 1)
    end_idx = oos_start + max_steps
    
    print(f"Running Inference locally over {max_steps} OOS windows...")
    
    with h5py.File(h5_path, "r") as f:
        features = f["X"][oos_start:end_idx].astype(np.float32)
        
    dataset = TensorDataset(torch.tensor(features))
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    
    all_actions = []
    
    with torch.no_grad():
        for (x_batch,) in loader:
            logits, _ = model(x_batch.to(device))
            probs = torch.softmax(logits, dim=-1)
            
            for j in range(len(probs)):
                p = probs[j]
                action_idx = p.argmax().item()
                confidence = p[action_idx].item()
                if confidence >= 0.40:
                    all_actions.append(action_idx)
                else:
                    all_actions.append(0)
                    
    print("Inference Complete! Simulating trades through XAUUSD Env...")
    
    env = XAUUSDEnv(
        h5_path = h5_path,
        start_idx = oos_start,
        end_idx = end_idx + 1,
        window_size = 256,
        spread_pips = 25,
        lot_size = 0.01,
        initial_balance = 50000.0,
        max_drawdown_usd = 999999.0,
        random_start = False
    )
    
    env.reset()
    
    trade_pnls = []
    current_position_open_balance = env._balance
    current_position_dir = 0
    
    for i, action in enumerate(all_actions):
        pre_pos = current_position_dir
        
        _, _, term, trunc, info = env.step(action)
        cur_pos = env._position_dir
        
        # Determine if a trade was closed!
        if pre_pos != 0 and cur_pos != pre_pos:
            # Pos changed from non-zero to zero (or reversed)
            # The exact PnL of this trade is current balance - balance before trade
            pnl = env._balance - current_position_open_balance 
            trade_pnls.append(pnl)
            
            # If reversal, log the new trade's starting balance
            if cur_pos != 0:
                current_position_open_balance = env._balance
        elif pre_pos == 0 and cur_pos != 0:
            current_position_open_balance = env._balance
            
        current_position_dir = cur_pos
        
        if term or trunc:
            break

    wins = [x for x in trade_pnls if x > 0]
    losses = [x for x in trade_pnls if x <= 0]
    
    print("\n" + "="*50)
    print(" OOS TRADE-LEVEL SIMULATION (SNIPER V2) ")
    print("="*50)
    print(f" Total Trades : {len(trade_pnls)}")
    if len(trade_pnls) > 0:
        win_rate = len(wins) / len(trade_pnls) * 100
        avg_win = sum(wins) / len(wins) if len(wins) > 0 else 0
        avg_loss = sum(losses) / len(losses) if len(losses) > 0 else 0
        rr = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        total_profit = sum(trade_pnls)
        
        print(f" Win Rate     : {win_rate:.2f}%")
        print(f" Average RR   : {rr:.2f} (Reward/Risk)")
        print(f" Average Win  : ${avg_win:.2f}")
        print(f" Average Loss : ${avg_loss:.2f}")
        print(f" Net Profit   : ${total_profit:.2f}")
    print("="*50)

if __name__ == "__main__":
    fast_batched_sim()
