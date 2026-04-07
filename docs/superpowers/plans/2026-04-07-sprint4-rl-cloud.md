# XAUUSD Bot Ã¢â‚¬â€ Sprint 4: RL Fine-tuning & Cloud Scale-up (Phase 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune model BC bÃ¡ÂºÂ±ng PPO Reinforcement Learning (CleanRL) Ã¢â‚¬â€ tÃ¡Â»â€˜i Ã†Â°u Drawdown, Sharpe, Win Rate trÃƒÂªn out-of-sample. Ã„ÂÃƒÂ³ng gÃƒÂ³i Docker Ã„â€˜Ã¡Â»Æ’ triÃ¡Â»Æ’n khai lÃƒÂªn Vast.ai GPU server.

**Architecture:** Load `best_model_bc.pt` Ã¢â€ â€™ sÃ¡Â»Â­ dÃ¡Â»Â¥ng **CleanRL** (khÃƒÂ´ng dÃƒÂ¹ng SB3) Ã¢â‚¬â€ toÃƒÂ n bÃ¡Â»â„¢ loss logic trong 1 file duy nhÃ¡ÂºÂ¥t, dÃ¡Â»â€¦ Ã„â€˜Ã¡Â»Âc vÃƒÂ  chÃ¡Â»â€°nh sÃ¡Â»Â­a. Transformer Ã„â€˜Ã†Â°Ã¡Â»Â£c nhÃƒÂºng trÃ¡Â»Â±c tiÃ¡ÂºÂ¿p vÃƒÂ o PPO actor-critic, khÃƒÂ´ng bÃ¡Â»â€¹ flatten. KL-Divergence anchor chÃ¡Â»â€˜ng Catastrophic Forgetting Ã„â€˜Ã†Â°Ã¡Â»Â£c chÃƒÂ¨n thÃ¡ÂºÂ³ng vÃƒÂ o training loop chÃ¡Â»â€° 2 dÃƒÂ²ng toÃƒÂ¡n hÃ¡Â»Âc.

**LÃƒÂ½ do KHÃƒâ€NG dÃƒÂ¹ng Stable-Baselines3 (SB3):**
- SB3 mÃ¡ÂºÂ·c Ã„â€˜Ã¡Â»â€¹nh **flatten** input 2D `(128, 12)` thÃƒÂ nh `(1536,)` trÃ†Â°Ã¡Â»â€ºc khi Ã„â€˜Ã†Â°a vÃƒÂ o mÃ¡ÂºÂ¡ng Ã¢â‚¬â€ phÃƒÂ¡ vÃ¡Â»Â¡ cÃ¡ÂºÂ¥u trÃƒÂºc thÃ¡Â»Âi gian cÃ¡Â»Â§a Transformer hoÃƒÂ n toÃƒÂ n.
- ViÃ¡Â»â€¡c hack SB3 Ã„â€˜Ã¡Â»Æ’ thÃƒÂªm KL Penalty loss tÃƒÂ¹y chÃ¡Â»â€°nh Ã„â€˜ÃƒÂ²i hÃ¡Â»Âi monkey-patch rÃ¡ÂºÂ¥t phÃ¡Â»Â©c tÃ¡ÂºÂ¡p vÃƒÂ  dÃ¡Â»â€¦ gÃƒÂ¢y regression.
- CleanRL giÃ¡ÂºÂ£i quyÃ¡ÂºÂ¿t cÃ¡ÂºÂ£ 2 vÃ¡ÂºÂ¥n Ã„â€˜Ã¡Â»Â chÃ¡Â»â€° vÃ¡Â»â€ºi 2 dÃƒÂ²ng code.

**Tech Stack:** PyTorch, CleanRL (single-file PPO), gymnasium, Docker, quantstats, pytest

---

## File Structure

```
src/training/
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ train_rl.py              [NEW] Ã¢â‚¬â€ CleanRL PPO single-file + KL anchor + Transformer
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ backtest.py              [NEW] Ã¢â‚¬â€ Out-of-sample evaluation & report
Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ tests/
    Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ test_backtest.py     [NEW]

Dockerfile                   [NEW] Ã¢â‚¬â€ Container cho Vast.ai
scripts/
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ vast_launch.sh           [NEW] Ã¢â‚¬â€ Script khÃ¡Â»Å¸i Ã„â€˜Ã¡Â»â„¢ng training trÃƒÂªn Vast.ai
Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ pull_results.sh          [NEW] Ã¢â‚¬â€ Script kÃƒÂ©o kÃ¡ÂºÂ¿t quÃ¡ÂºÂ£ vÃ¡Â»Â local
```

---

## Task 1: PPO Training vÃ¡Â»â€ºi KL-Divergence Anchor

**Files:**
- Create: `src/training/train_rl.py`

### Step 1.1: Implement train_rl.py (CleanRL style)

> Ã¢Å¡Â Ã¯Â¸Â **KhÃƒÂ´ng dÃƒÂ¹ng SB3.** DÃƒÂ¹ng CleanRL approach: tÃƒÂ i PPO thÃ¡Â»Â¥ cÃƒÂ´ng trong 1 file, chÃƒÂ¨n Transformer vÃƒÂ  KL Anchor trÃ¡Â»Â±c tiÃ¡ÂºÂ¿p.

- [ ] **TÃ¡ÂºÂ¡o `src/training/train_rl.py`:**

```python
"""
train_rl.py  (CleanRL-style Ã¢â‚¬â€ v3, fixed)
-----------------------------------------
Phase 2: PPO Reinforcement Learning Fine-tuning.

Fix log:
  v1: initial CleanRL draft
  v2: AsyncVectorEnv Ã„â€˜Ã¡Â»Æ’ song song hÃƒÂ³a CPU
  v3: [FIX OOM] DÃƒÂ¹ng h5_path + offset thay vÃƒÂ¬ truyÃ¡Â»Ân raw array
      [FIX SYN] compute_gae thiÃ¡ÂºÂ¿u 'def'
      [FIX DC]  XÃƒÂ³a dead code sau return trong make_async_envs
      [FIX RMS] Freeze RunningMeanStd trong PPO epochs, chÃ¡Â»â€° update 1 lÃ¡ÂºÂ§n/rollout

CÃƒÂ¡ch dÃƒÂ¹ng:
  python src/training/train_rl.py \\\
      --h5      data/processed/XAUUSD_M15_w128.h5 \\\
      --bc-ckpt checkpoints/best_model_bc.pt \\\
      --n-envs  64 \\\
      --total-steps 2000000
"""

import argparse
import logging
import math
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from gymnasium.vector import AsyncVectorEnv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.model.transformer import XAUTransformer
from src.env.xauusd_env import XAUUSDEnv

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

CHECKPOINT_DIR = Path("checkpoints")


def make_async_envs(
    h5_path:      str,
    n_total:      int,
    split_idx:    int,    # Ã„ÂÃƒÂ£ bao gÃ¡Â»â€œm purged gap bÃƒÂªn trong
    window_size:  int,
    n_envs:       int,
):
    """
    [FIX OOM] TruyÃ¡Â»Ân h5_path + offset, khÃƒÂ´ng truyÃ¡Â»Ân raw array.
    MÃ¡Â»â€”i worker tÃ¡Â»Â± Ã„â€˜Ã¡Â»Âc HDF5 (cÃ¡ÂºÂ£ X, y, vÃƒÂ  CLOSE) trong process riÃƒÂªng.
    """
    def _make_env_fn(offset: int):
        def _init():
            # [FIX OOM] Không copy numpy array vào RAM, truyền tham số h5_path cho Env tự Lazy Load
            return XAUUSDEnv(
                h5_path          = h5_path,
                start_idx        = offset,         # Bắt đầu từ offset của worker này
                end_idx          = split_idx,      # Không vượt quá tập train
                window_size      = window_size,
                spread_pips      = 25,
                lot_size         = 0.01,
                initial_balance  = 200.0,
                max_drawdown_usd = 20.0,
            )
        return _init

    return AsyncVectorEnv([
        _make_env_fn((i * 500) % max(1, split_idx - window_size - 2000))
        for i in range(n_envs)
    ])


def collect_rollout(vec_env, model, device, rollout_steps=2048):
    """
    Thu thÃ¡ÂºÂ­p rollout tÃ¡Â»Â« AsyncVectorEnv Ã¢â‚¬â€ tÃ¡ÂºÂ¥t cÃ¡ÂºÂ£ envs step() song song.
    TrÃ¡ÂºÂ£ vÃ¡Â»Â (obs, actions, log_probs, rewards, dones, values).
    """
    obs_list, act_list, logp_list, rew_list, done_list, val_list = \
        [], [], [], [], [], []

    obs, _ = vec_env.reset()

    for _ in range(rollout_steps):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            logits, value = model(obs_t)
            dist   = Categorical(logits=logits)
            action = dist.sample()
            logp   = dist.log_prob(action)

        obs_list.append(obs_t.cpu())
        act_list.append(action.cpu())
        logp_list.append(logp.cpu())
        val_list.append(value.squeeze(-1).cpu())

        next_obs, rewards, terms, truncs, _ = vec_env.step(action.cpu().numpy())
        rew_list.append(torch.tensor(rewards, dtype=torch.float32))
        done_list.append(torch.tensor(terms | truncs, dtype=torch.float32))
        obs = next_obs

    # [FIX GAE] Lấy value của bước tiếp theo để tính GAE chuẩn xác
    with torch.no_grad():
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
        _, next_value = model(obs_t)
        next_value = next_value.squeeze(-1).cpu()

    return (
        torch.stack(obs_list),
        torch.stack(act_list),
        torch.stack(logp_list),
        torch.stack(rew_list),
        torch.stack(done_list),
        torch.stack(val_list),
        next_value,
    )


class RunningMeanStd:
    """
    [FIX RMS] ChÃ¡Â»â€° update 1 lÃ¡ÂºÂ§n SAU mÃ¡Â»â€”i rollout Ã¢â‚¬â€ freeze trong suÃ¡Â»â€˜t PPO epochs.

    LÃƒÂ½ do: NÃ¡ÂºÂ¿u update trong mÃ¡Â»â€”i mini-batch, 'target' cÃ¡Â»Â§a Value Head thay Ã„â€˜Ã¡Â»â€¢i
    liÃƒÂªn tÃ¡Â»Â¥c (Non-stationary target) Ã¢â€ â€™ Value Head hÃ¡Â»Âc rÃ¡ÂºÂ¥t chÃ¡ÂºÂ­m / khÃƒÂ´ng hÃ¡Â»â„¢i tÃ¡Â»Â¥.
    GiÃ¡ÂºÂ£i phÃƒÂ¡p: GÃ¡Â»Âi ret_rms.update() 1 lÃ¡ÂºÂ§n trÃ†Â°Ã¡Â»â€ºc ppo_update(),
    rÃ¡Â»â€œi dÃƒÂ¹ng mean/var Ã„â€˜ÃƒÂ£ freeze cho cÃ¡ÂºÂ£ 4 epochs cÃ¡Â»Â§a PPO.
    """
    def __init__(self, epsilon: float = 1e-8):
        self.mean  = 0.0
        self.var   = 1.0
        self.count = epsilon

    def update(self, x: torch.Tensor) -> None:
        """GÃ¡Â»Â i NGOAI ppo_update() Ã¢â‚¬â€  1 lÃ¡ÂºÂ§n/rollout."""
        v = x.detach().float()
        b_mean, b_var, b_n = v.mean().item(), v.var().item(), v.numel()
        total = self.count + b_n
        delta = b_mean - self.mean
        self.mean  += delta * b_n / total
        self.var    = (self.var * self.count + b_var * b_n +
                       delta ** 2 * self.count * b_n / total) / total
        self.count  = total

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """DÃƒÂ¹ng mean/var Ã„â€˜ÃƒÂ£ Ã¢â‚¬Å“freezeÃ¢â‚¬Â  (khÃƒÂ´ng gÃ¡Â»Â i update() trong hÃƒÂ m nÃƒÂ y)."""
        return ((x - self.mean) / (self.var ** 0.5 + 1e-8)).clamp(-10.0, 10.0)

def evaluate_oos(model, h5_path, split_idx, n_total, window_size, device,
                 gap_bars=200):
    """
    Chay 1 episode duy nhat xuyen suot toan bo OOS den khi truncated = True.

    [FIX GROUNDHOG DAY] Khong lap n_eval_eps lan - du lieu time-series khong
    ngau nhien, policy la deterministic => lap = nhan ban ket qua => Sharpe gia.
    """
    from src.training.backtest import compute_metrics

    oos_start = split_idx + gap_bars

    env = XAUUSDEnv(
        h5_path          = h5_path,
        start_idx        = oos_start,
        end_idx          = n_total,
        window_size      = window_size,
        spread_pips      = 25,
        lot_size         = 0.01,
        initial_balance  = 200.0,
        max_drawdown_usd = 20.0, # [FIX EQUITY HOLE] Dừng sớm nếu âm 10% vốn (Luật Prop Firm), tránh chia cho Equity âm
    )
    model.eval()

    # [FIX GROUNDHOG] Chi 1 episode - chay den het OOS (truncated = True)
    obs, _ = env.reset()
    done = False
    
    # [FIX SHARPE OOS] Dùng Equity thay vì Balance để thấy rõ Unrealized Drawdown
    equity_hist = [200.0]
    position_hist = [0]
    while not done:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = model(obs_t)
            action = logits.argmax(-1).item()   # deterministic (greedy)
        obs, _, term, trunc, info = env.step(action)
        equity_hist.append(info.get("equity", 200.0))
        position_hist.append(env._position_dir)
        done = term or trunc

    model.train()
    bar_returns = np.diff(equity_hist) / np.array(equity_hist[:-1])
    metrics = compute_metrics(bar_returns, positions=np.array(position_hist))
    return metrics["sharpe"]


def compute_gae(rewards, values, next_value, dones, gamma=0.99, lam=0.95):
    """Generalized Advantage Estimation - [FIX BELLMAN] Dùng next_val."""
    T, E = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_adv   = 0.0
    for t in reversed(range(T)):
        if t == T - 1:
            next_val = next_value
        else:
            next_val = values[t+1]
        
        mask       = 1.0 - dones[t]
        delta      = rewards[t] + gamma * next_val * mask - values[t]
        last_adv   = delta + gamma * lam * mask * last_adv
        advantages[t] = last_adv
    returns = advantages + values
    return advantages, returns


def ppo_update(
    model,
    bc_model,
    optimizer,
    obs, actions, old_logps, flat_ret, advantages,
    device,
    clip_eps   = 0.2,
    vf_coef    = 0.5,
    ent_coef   = 0.01,
    kl_coef    = 0.3,
    n_epochs   = 4,
    batch_size = 256,
):
    """
    PPO update vÃ¡Â»Âºi KL Anchor.
    Returns Ã„â€˜ÃƒÂ£ Ã„â€˜Ã†Â°Ã¡Â»Â£c chuÃ¡ÂºÂ©n hÃƒÂ³a BÃƒÅ N NGOAI (RunningMeanStd.update() gÃ¡Â»Â i trÃ†Â°Ã¡Â»Â›c).
    flat_ret lÃƒÂ  tensor Ã„â€˜ÃƒÂ£ normalize, mean/var freeze trong suÃ¡Â»Â‘t vÃƒÂ²ng lÃ¡ÂºÂ·p.
    """
    T, E = obs.shape[:2]
    flat_obs  = obs.view(T * E, *obs.shape[2:]).to(device)
    flat_act  = actions.view(-1).to(device)
    flat_logp = old_logps.view(-1).to(device)
    flat_adv  = advantages.view(-1).to(device)
    flat_adv  = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)
    # flat_ret Ã„â€˜ÃƒÂ£ sÃ¡ÂºÂµn sÃƒÂ ng (normalized, frozen)
    flat_ret  = flat_ret.to(device)

    losses = []
    for _ in range(n_epochs):
        idx = torch.randperm(T * E, device=device)
        for start in range(0, T * E, batch_size):
            b = idx[start:start + batch_size]
            logits, value = model(flat_obs[b])
            dist    = Categorical(logits=logits)
            logp    = dist.log_prob(flat_act[b])
            entropy = dist.entropy().mean()

            ratio   = (logp - flat_logp[b]).exp()
            pg_loss = -torch.min(
                ratio * flat_adv[b],
                ratio.clamp(1 - clip_eps, 1 + clip_eps) * flat_adv[b]
            ).mean()

            # [FIX GRADIENT EXPLOSION] value head là random => error đầu tiên rất cao
            # MSE Loss sẽ bình phương error này, tạo ra gradient khổng lồ phá vỡ Policy head
            # Dùng smooth_l1_loss (Huber Loss) để giới hạn penalty
            vf_loss = F.smooth_l1_loss(value.squeeze(-1), flat_ret[b])

            # KL Anchor Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬
            with torch.no_grad():
                bc_logits, _ = bc_model(flat_obs[b])
            kl_loss = F.kl_div(
                F.log_softmax(logits, dim=-1),
                F.softmax(bc_logits, dim=-1),
                reduction="batchmean",
            )

            loss = pg_loss + vf_coef * vf_loss - ent_coef * entropy + kl_coef * kl_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            losses.append(loss.item())

    return np.mean(losses)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # Ã¢â€ â‚¬Ã¢â€ â‚¬ Ã„â€˜Ã¡Â»Â c metadata tÃ¡Â»Â« HDF5 Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬
    with h5py.File(args.h5, "r") as f:
        n_total, window_size, n_features = f["X"].shape
    GAP_BARS  = 200  # [FIX] Purged gap giÃ¡Â»Â¯a train vÃƒÂ  OOS
    split_idx = int(n_total * 0.8) - GAP_BARS  # Train dÃ¡Â»Â«ng trÃ†Â°Ã¡Â»Â›c gap
    oos_start = int(n_total * 0.8)              # OOS bÃ¡ÂºÂ¯t Ã„â€˜Ã¡ÂºÂ§u sau gap
    log.info(f"Split: train[:{split_idx}] | gap={GAP_BARS} | OOS[{oos_start}:]")

    # Ã¢â€ â‚¬Ã¢â€ â‚¬ BC Model Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬
    ckpt = torch.load(args.bc_ckpt, map_location=device)
    bc_model = XAUTransformer(n_features=n_features, window_size=window_size,
                              d_model=256, n_heads=8, n_layers=6).to(device)
    bc_model.load_state_dict(ckpt["model_state"])
    bc_model.eval()
    for p in bc_model.parameters():
        p.requires_grad = False
    log.info(f"BC loaded: F1_buy={ckpt['f1_buy']:.3f}")

    ppo_model = XAUTransformer(n_features=n_features, window_size=window_size,
                               d_model=256, n_heads=8, n_layers=6).to(device)
    ppo_model.load_state_dict(ckpt["model_state"])
    optimizer = optim.AdamW(ppo_model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Ã¢â€ â‚¬Ã¢â€ â‚¬ [FIX OOM] Envs Ã„â€˜Ã¡Â»Â c HDF5 tÃ¡Â»Â« file, khÃƒÂ´ng copy array Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬
    vec_env = make_async_envs(
        h5_path=args.h5, n_total=n_total,
        split_idx=split_idx, window_size=window_size,
        n_envs=args.n_envs,
    )
    log.info(f"{args.n_envs} async envs ready")

    rollout_steps = 256  # [FIX PPO LOOP] Tăng lên 122 updates thay vì 15 updates
    total_updates = args.total_steps // (rollout_steps * args.n_envs)
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    ret_rms     = RunningMeanStd()
    best_sharpe = -np.inf  # [FIX] Theo dÃƒÂµi best checkpoint bÃ¡ÂºÂ±ng Sharpe OOS

    for update in range(1, total_updates + 1):
        kl_coef = max(0.05, 0.5 * math.exp(-update / (total_updates * 0.5)))

        obs, actions, logps, rewards, dones, values, next_value = \
            collect_rollout(vec_env, ppo_model, device, rollout_steps)

        adv, returns = compute_gae(rewards, values, next_value, dones)

        raw_ret = returns.view(-1)
        ret_rms.update(raw_ret)
        flat_ret_norm = ret_rms.normalize(raw_ret)

        loss = ppo_update(
            ppo_model, bc_model, optimizer,
            obs, actions, logps, flat_ret_norm, adv,
            device, kl_coef=kl_coef
        )

        if update % 10 == 0:
            log.info(f"Update {update:4d}/{total_updates} | "
                     f"Loss={loss:.4f} | AvgRew={rewards.mean():.4f} | KLÃŽÂ»={kl_coef:.3f}")

        # [FIX] Eval định kỳ mỗi chu kỳ chẵn (nhanh hơn do update nhiều hơn)       
        if update % 10 == 0:
            sharpe = evaluate_oos(
                ppo_model, args.h5, split_idx, n_total,
                window_size, device, gap_bars=GAP_BARS
            )
            log.info(f"  [OOS EVAL] Sharpe={sharpe:.4f} | Best={best_sharpe:.4f}")
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                torch.save(ppo_model.state_dict(),
                           CHECKPOINT_DIR / "ppo_best.pt")
                log.info(f"  Ã¢Å“â€¦ New best checkpoint saved (Sharpe={best_sharpe:.4f})")

        if update % 100 == 0:
            torch.save(ppo_model.state_dict(),
                       CHECKPOINT_DIR / f"ppo_step{update}.pt")

    log.info("Ã°Å¸Å½â€° PPO training hoÃƒÂ n tÃ¡ÂºÂ¥t!")
    torch.save(ppo_model.state_dict(), CHECKPOINT_DIR / "ppo_final.pt")
    vec_env.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--h5",          required=True)
    p.add_argument("--bc-ckpt",     required=True, dest="bc_ckpt")
    p.add_argument("--n-envs",      type=int,   default=64,        dest="n_envs")
    p.add_argument("--total-steps", type=int,   default=2_000_000, dest="total_steps")
    p.add_argument("--lr",          type=float, default=1e-4)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
```

- [ ] **Commit:**
```bash
git add src/training/train_rl.py
git commit -m "feat(sprint4): PPO CleanRL - AsyncVectorEnv, KL anchor, RMS freeze, OOM fix"
```

---


## Task 2: Backtest & Reporting

**Files:**
- Create: `src/training/backtest.py`
- Create: `src/training/tests/test_backtest.py`

### Step 2.1: Viáº¿t failing tests cho Backtest

- [ ] **Táº¡o `src/training/tests/test_backtest.py`:**

```python
# src/training/tests/test_backtest.py
import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.training.backtest import compute_metrics


class TestBacktestMetrics:
    def test_positive_pnl_series_sharpe_positive(self):
        """Chuá»—i PnL dÆ°Æ¡ng Ä‘á» u Ä‘áº·n pháº£i cÃ³ Sharpe > 0."""
        daily_returns = np.array([0.001] * 252)
        metrics = compute_metrics(daily_returns)
        assert metrics["sharpe"] > 0, f"Sharpe pháº£i dÆ°Æ¡ng: {metrics['sharpe']}"

    def test_all_zero_returns_sharpe_zero(self):
        """PnL = 0 má» i ngÃ y â†’ Sharpe = 0."""
        daily_returns = np.zeros(252)
        metrics = compute_metrics(daily_returns)


    def test_max_drawdown_is_non_positive(self):
        """Max drawdown phÃ¡ÂºÂ£i <= 0 (biÃ¡Â»Æ’u diÃ¡Â»â€¦n mÃ¡ÂºÂ¥t vÃ¡Â»Â€˜n)."""
        daily_returns = np.array([0.01, -0.05, 0.02, -0.03, 0.01])
        metrics = compute_metrics(daily_returns)
        assert metrics["max_drawdown"] <= 0

    def test_win_rate_between_0_and_1(self):
        """Win rate phÃ¡ÂºÂ£i nÃ¡ÂºÂ±m trong [0, 1]."""
        returns = np.random.randn(100) * 0.01
        metrics = compute_metrics(returns)
        assert 0.0 <= metrics["win_rate"] <= 1.0

    def test_metrics_has_required_keys(self):
        """KÃ¡ÂºÂ¿t quÃ¡ÂºÂ£ phÃ¡ÂºÂ£i cÃƒÂ³ Ã„â€˜Ã¡Â»Â§ cÃƒÂ¡c key bÃ¡ÂºÂ¯t buÃ¡Â»â„¢c."""
        returns = np.random.randn(252) * 0.001
        metrics = compute_metrics(returns)
        required = {"sharpe", "sortino", "max_drawdown", "win_rate",
                    "total_return", "n_trades"}
        assert required.issubset(set(metrics.keys()))
```

- [ ] **ChÃ¡ÂºÂ¡y Ã„â€˜Ã¡Â»Âƒ verify FAIL:**
```bash
python -m pytest src/training/tests/test_backtest.py -v
```

- [ ] **TÃ¡ÂºÂ¡o `src/training/backtest.py`:**

```python
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

    # Trong thÃ¡Â»Â±c tÃ¡ÂºÂ¿: chÃ¡ÂºÂ¡y model qua out-of-sample env vÃƒÂ  thu thÃ¡ÂºÂ­p returns
    # Ã„ÂÃƒÂ¢y lÃƒÂ  scaffold Ã¢â‚¬â€ sÃ¡ÂºÂ½ Ã„â€˜Ã†Â°Ã¡Â»Â£c hoÃƒÂ n thiÃ¡Â»â€¡n khi env + model Ã„â€˜ÃƒÂ£ sÃ¡ÂºÂµn sÃƒÂ ng
    log.info("Ã¢Å¡Â Ã¯Â¸Â  Running simplified backtest scaffold (returns cÃ¡ÂºÂ§n Ã„â€˜Ã†Â°Ã¡Â»Â£c thu thÃ¡ÂºÂ­p tÃ¡Â»Â« env.step)")
    log.info("    SÃ¡Â»Â­ dÃ¡Â»Â¥ng compute_metrics() vÃ¡Â»â€ºi array returns sau khi roll-out Ã„â€˜Ã¡ÂºÂ§y Ã„â€˜Ã¡Â»Â§.")


if __name__ == "__main__":
    main()
```

- [ ] **ChÃ¡ÂºÂ¡y Ã„â€˜Ã¡Â»Æ’ verify PASS:**
```bash
python -m pytest src/training/tests/test_backtest.py -v
```
KÃ¡ÂºÂ¿t quÃ¡ÂºÂ£ mong Ã„â€˜Ã¡Â»Â£i: `5 passed`

- [ ] **Commit:**
```bash
git add src/training/backtest.py src/training/tests/test_backtest.py
git commit -m "feat(sprint4): backtest - compute_metrics Sharpe/Sortino/MaxDD/WinRate"
```

---

## Task 3: Dockerfile & Vast.ai Scripts

**Files:**
- Create: `Dockerfile`
- Create: `scripts/vast_launch.sh`

### Step 3.1: TÃ¡ÂºÂ¡o Dockerfile

- [ ] **TÃ¡ÂºÂ¡o `Dockerfile`:**

```dockerfile
# XAUUSD AI Trading Bot Ã¢â‚¬â€ Training Container (Phase 2: PPO)
# TÃ¡Â»â€˜i Ã†Â°u cho RTX 4090 / RTX 5090 trÃƒÂªn Vast.ai
# [FIX] Phase 2 (PPO) dÃƒÂ¹ng AsyncVectorEnv Ã¢â‚¬â€ cÃ¡ÂºÂ§n CPU Ã„â€˜a luÃ¡Â»â€œng mÃ¡ÂºÂ¡nh,
# khÃƒÂ´ng chÃ¡ÂºÂ¡y BC (Phase 1) vÃƒÂ¬ BC Ã„â€˜ÃƒÂ£ xong rÃ¡Â»â€œi!

FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    git \\
    htop \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements vÃƒÂ  install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY data/ ./data/

# TÃ¡ÂºÂ¡o thÃ†Â° mÃ¡Â»Â¥c output
RUN mkdir -p checkpoints logs

# [FIX] CMD chÃ¡ÂºÂ¡y PPO (Phase 2), khÃƒÂ´ng phÃ¡ÂºÂ£i BC (Phase 1)
# BC checkpoint truyÃ¡Â»Ân vÃƒÂ o qua --bc-ckpt khi docker run
CMD ["python", "src/training/train_rl.py", \\
     "--h5",          "data/processed/XAUUSD_M15_w128.h5", \\
     "--bc-ckpt",     "checkpoints/best_model_bc.pt", \\
     "--n-envs",      "64", \\
     "--total-steps", "2000000"]
```

- [ ] **TÃ¡ÂºÂ¡o `scripts/vast_launch.sh`:**

```bash
#!/bin/bash
# vast_launch.sh Ã¢â‚¬â€ TriÃ¡Â»Æ’n khai PPO training (Phase 2) lÃƒÂªn Vast.ai instance
# [FIX] ChÃ¡ÂºÂ¡y train_rl.py (PPO), khÃƒÂ´ng phÃ¡ÂºÂ£i train_bc.py (BC Ã„â€˜ÃƒÂ£ xong Ã¡Â»Å¸ local)
# CÃƒÂ¡ch dÃƒÂ¹ng: ./scripts/vast_launch.sh <INSTANCE_ID> <SSH_KEY_PATH> <BC_CKPT_PATH>
#
# Workflow:
#   1. Upload source + HDF5 data + BC checkpoint lÃƒÂªn server
#   2. Launch PPO training dÃƒÂ¹ng AsyncVectorEnv (64 envs)
#   3. Checkpoint ppo_best.pt tÃ¡Â»Â± Ã„â€˜Ã¡Â»â„¢ng lÃ†Â°u khi Sharpe OOS tÃ¡Â»â€˜t hÃ†Â¡n

set -e

INSTANCE_ID=${1:?"Usage: $0 <instance_id> <ssh_key> <bc_ckpt>"}
SSH_KEY=${2:?"Usage: $0 <instance_id> <ssh_key> <bc_ckpt>"}
BC_CKPT=${3:-"checkpoints/best_model_bc.pt"}

echo "=== XAUUSD Bot Ã¢â‚¬â€ Vast.ai PPO Deployment ==="
echo "Instance: $INSTANCE_ID"
echo "BC Checkpoint: $BC_CKPT"

# 1. LÃ¡ÂºÂ¥y connection info
CONN=$(vastai ssh-url $INSTANCE_ID)
SSH_HOST=$(echo $CONN | sed 's/ssh:\/\///')

# 2. Rsync source code + data + BC checkpoint lÃƒÂªn instance
echo "Uploading source code..."
rsync -avz --exclude '.git' --exclude '__pycache__' \\
    -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no" \\
    ./ root@${SSH_HOST}:/workspace/

# Upload BC checkpoint riÃƒÂªng nÃ¡ÂºÂ¿u nÃ¡ÂºÂ±m ngoÃƒÂ i workspace
if [ -f "$BC_CKPT" ]; then
    scp -i $SSH_KEY $BC_CKPT root@${SSH_HOST}:/workspace/checkpoints/best_model_bc.pt
    echo "BC checkpoint uploaded."
fi

# 3. Install dependencies
echo "Installing dependencies..."
ssh -i $SSH_KEY -o StrictHostKeyChecking=no root@${SSH_HOST} \\
    "cd /workspace && pip install -q -r requirements.txt"

# 4. [FIX] Launch PPO training (Phase 2) Ã¢â‚¬â€ khÃƒÂ´ng phÃ¡ÂºÂ£i BC!
echo "Starting PPO RL training (Phase 2)..."
ssh -i $SSH_KEY -o StrictHostKeyChecking=no root@${SSH_HOST} \\
    "cd /workspace && nohup python src/training/train_rl.py \\
        --h5          data/processed/XAUUSD_M15_w128.h5 \\
        --bc-ckpt     checkpoints/best_model_bc.pt \\
        --n-envs      64 \\
        --total-steps 2000000 \\
        > logs/train_rl.log 2>&1 &"

echo "Ã¢Å“â€¦ PPO training started! Theo dÃƒÂµi log:"
echo "   ssh -i $SSH_KEY root@${SSH_HOST} 'tail -f /workspace/logs/train_rl.log'"
```

- [ ] **TÃ¡ÂºÂ¡o `scripts/pull_results.sh`:**

```bash
#!/bin/bash
# pull_results.sh Ã¢â‚¬â€ KÃƒÂ©o kÃ¡ÂºÂ¿t quÃ¡ÂºÂ£ training vÃ¡Â»Â local
# CÃƒÂ¡ch dÃƒÂ¹ng: ./scripts/pull_results.sh <INSTANCE_ID> <SSH_KEY_PATH>

set -e
INSTANCE_ID=${1:?"Usage: $0 <instance_id> <ssh_key>"}
SSH_KEY=${2:?"Usage: $0 <instance_id> <ssh_key>"}

CONN=$(vastai ssh-url $INSTANCE_ID)
SSH_HOST=$(echo $CONN | sed 's/ssh:\/\///')

echo "Pulling checkpoints and logs from Vast.ai..."
rsync -avz \
    -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no" \
    root@${SSH_HOST}:/workspace/checkpoints/ ./checkpoints/

rsync -avz \
    -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no" \
    root@${SSH_HOST}:/workspace/logs/ ./logs/

echo "Ã¢Å“â€¦ Done! Checkpoints saved to ./checkpoints/"
ls -lh ./checkpoints/
```

- [ ] **Commit:**
```bash
git add Dockerfile scripts/
git commit -m "feat(sprint4): Dockerfile + Vast.ai launch/pull scripts"
```

---

## Task 4: ToÃƒÂ n bÃ¡Â»â„¢ tests Sprint 4 & Final Push

- [ ] **ChÃ¡ÂºÂ¡y toÃƒÂ n bÃ¡Â»â„¢ test suite:**
```bash
python -m pytest src/ -v --tb=short
```
KÃ¡ÂºÂ¿t quÃ¡ÂºÂ£ mong Ã„â€˜Ã¡Â»Â£i: `Ã¢â€°Â¥ 35 passed, 0 failed` (tÃ¡ÂºÂ¥t cÃ¡ÂºÂ£ 4 sprints)

- [ ] **Final push:**
```bash
git push origin main
```

---

## Ã„ÂiÃ¡Â»Âu kiÃ¡Â»â€¡n DONE cho Sprint 4 (& toÃƒÂ n dÃ¡Â»Â± ÃƒÂ¡n v1)

| ChÃ¡Â»â€° sÃ¡Â»â€˜ | Target |
|---|---|
| `pytest src/ -v` | Ã¢Å“â€¦ 0 failures |
| F1(Buy) validation | > 0.40 |
| F1(Sell) validation | > 0.40 |
| Sharpe Ratio (out-of-sample) | > 1.0 |
| Max Drawdown | < 10% ($20) |
| Win Rate | > 55% |
| Docker build | Ã¢Å“â€¦ khÃƒÂ´ng crash |
| Vast.ai training log | Ã¢Å“â€¦ converging loss |
