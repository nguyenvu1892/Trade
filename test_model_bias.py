import torch, numpy as np, sys
sys.path.insert(0, '.')
from src.model.transformer import XAUTransformer
from src.data.data_processor import DataProcessor
import MetaTrader5 as mt5

mt5.initialize()
rates = mt5.copy_rates_from_pos('XAUUSD', mt5.TIMEFRAME_M5, 0, 280)
mt5.shutdown()

import pandas as pd
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
df.rename(columns={'time':'datetime'}, inplace=True)
df.set_index('datetime', inplace=True)

proc = DataProcessor(atr_period=14)
feat = proc.compute_features(df).values.astype(np.float32)
n_feat = feat.shape[1]
actions = ["HOLD", "BUY", "SELL"]
X = torch.tensor(feat[-256:], dtype=torch.float32).unsqueeze(0)

print(f"Current price: ${df['close'].iloc[-1]:.2f}")
print(f"Features: {n_feat}, Window: 256\n")

# Old BC
model_old = XAUTransformer(n_features=n_feat, d_model=256, n_heads=8, n_layers=6, window_size=256)
ckpt_old = torch.load('checkpoints/best_model_bc.pt', map_location='cpu', weights_only=False)
model_old.load_state_dict(ckpt_old['model_state'] if 'model_state' in ckpt_old else ckpt_old)
model_old.eval()
with torch.no_grad():
    logits, _ = model_old(X)
    probs_old = torch.softmax(logits, dim=1).squeeze().numpy()
print(f"OLD BC (1 month)   -> {actions[probs_old.argmax()]:4s} | H:{probs_old[0]*100:.1f}% B:{probs_old[1]*100:.1f}% S:{probs_old[2]*100:.1f}%")

# New BC
model_new = XAUTransformer(n_features=n_feat, d_model=256, n_heads=8, n_layers=6, window_size=256)
ckpt_new = torch.load('checkpoints/fresh_6month_best_model_bc.pt', map_location='cpu', weights_only=False)
model_new.load_state_dict(ckpt_new['model_state'] if 'model_state' in ckpt_new else ckpt_new)
model_new.eval()
with torch.no_grad():
    logits, _ = model_new(X)
    probs_new = torch.softmax(logits, dim=1).squeeze().numpy()
print(f"NEW BC (6 months)  -> {actions[probs_new.argmax()]:4s} | H:{probs_new[0]*100:.1f}% B:{probs_new[1]*100:.1f}% S:{probs_new[2]*100:.1f}%")
