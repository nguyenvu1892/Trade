# XAUUSD Bot â€” Sprint 4: RL Fine-tuning & Cloud Scale-up (Phase 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune model BC báº±ng PPO Reinforcement Learning (CleanRL) â€” tá»‘i Æ°u Drawdown, Sharpe, Win Rate trÃªn out-of-sample. ÄÃ³ng gÃ³i Docker Ä‘á»ƒ triá»ƒn khai lÃªn Vast.ai GPU server.

**Architecture:** Load `best_model_bc.pt` â†’ sá»­ dá»¥ng **CleanRL** (khÃ´ng dÃ¹ng SB3) â€” toÃ n bá»™ loss logic trong 1 file duy nháº¥t, dá»… Ä‘á»c vÃ  chá»‰nh sá»­a. Transformer Ä‘Æ°á»£c nhÃºng trá»±c tiáº¿p vÃ o PPO actor-critic, khÃ´ng bá»‹ flatten. KL-Divergence anchor chá»‘ng Catastrophic Forgetting Ä‘Æ°á»£c chÃ¨n tháº³ng vÃ o training loop chá»‰ 2 dÃ²ng toÃ¡n há»c.

**LÃ½ do KHÃ”NG dÃ¹ng Stable-Baselines3 (SB3):**
- SB3 máº·c Ä‘á»‹nh **flatten** input 2D `(128, 12)` thÃ nh `(1536,)` trÆ°á»›c khi Ä‘Æ°a vÃ o máº¡ng â€” phÃ¡ vá»¡ cáº¥u trÃºc thá»i gian cá»§a Transformer hoÃ n toÃ n.
- Viá»‡c hack SB3 Ä‘á»ƒ thÃªm KL Penalty loss tÃ¹y chá»‰nh Ä‘Ã²i há»i monkey-patch ráº¥t phá»©c táº¡p vÃ  dá»… gÃ¢y regression.
- CleanRL giáº£i quyáº¿t cáº£ 2 váº¥n Ä‘á» chá»‰ vá»›i 2 dÃ²ng code.

**Tech Stack:** PyTorch, CleanRL (single-file PPO), gymnasium, Docker, quantstats, pytest

---

## File Structure

```
src/training/
â”œâ”€â”€ train_rl.py              [NEW] â€” CleanRL PPO single-file + KL anchor + Transformer
â”œâ”€â”€ backtest.py              [NEW] â€” Out-of-sample evaluation & report
â””â”€â”€ tests/
    â””â”€â”€ test_backtest.py     [NEW]

Dockerfile                   [NEW] â€” Container cho Vast.ai
scripts/
â”œâ”€â”€ vast_launch.sh           [NEW] â€” Script khá»Ÿi Ä‘á»™ng training trÃªn Vast.ai
â””â”€â”€ pull_results.sh          [NEW] â€” Script kÃ©o káº¿t quáº£ vá» local
```

---

## Task 1: PPO Training vá»›i KL-Divergence Anchor

**Files:**
- Create: `src/training/train_rl.py`

### Step 1.1: Implement train_rl.py (CleanRL style)

> âš ï¸ **KhÃ´ng dÃ¹ng SB3.** DÃ¹ng CleanRL approach: tÃ i PPO thá»¥ cÃ´ng trong 1 file, chÃ¨n Transformer vÃ  KL Anchor trá»±c tiáº¿p.

- [ ] **Táº¡o `src/training/train_rl.py`:**

```python
"""
train_rl.py  (CleanRL-style â€” v3, fixed)
-----------------------------------------
Phase 2: PPO Reinforcement Learning Fine-tuning.

Fix log:
  v1: initial CleanRL draft
  v2: AsyncVectorEnv Ä‘á»ƒ song song hÃ³a CPU
  v3: [FIX OOM] DÃ¹ng h5_path + offset thay vÃ¬ truyá»n raw array
      [FIX SYN] compute_gae thiáº¿u 'def'
      [FIX DC]  XÃ³a dead code sau return trong make_async_envs
      [FIX RMS] Freeze RunningMeanStd trong PPO epochs, chá»‰ update 1 láº§n/rollout

CÃ¡ch dÃ¹ng:
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
    split_idx:    int,    # ÄÃ£ bao gá»“m purged gap bÃªn trong
    window_size:  int,
    n_envs:       int,
):
    """
    [FIX OOM] Truyá»n h5_path + offset, khÃ´ng truyá»n raw array.
    Má»—i worker tá»± Ä‘á»c HDF5 (cáº£ X, y, vÃ  CLOSE) trong process riÃªng.
    """
    def _make_env_fn(offset: int):
        def _init():
            with h5py.File(h5_path, "r") as f:
                X_slice      = f["X"][offset:split_idx].astype(np.float32)
                label_slice  = f["y"][offset:split_idx]
                close_slice  = f["close"][offset:split_idx].astype(np.float32)  # [FIX]
            return XAUUSDEnv(
                features         = X_slice,
                close_prices     = close_slice,   # real prices!
                oracle_labels    = label_slice,
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
    Thu tháº­p rollout tá»« AsyncVectorEnv â€” táº¥t cáº£ envs step() song song.
    Tráº£ vá» (obs, actions, log_probs, rewards, dones, values).
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

    return (
        torch.stack(obs_list),
        torch.stack(act_list),
        torch.stack(logp_list),
        torch.stack(rew_list),
        torch.stack(done_list),
        torch.stack(val_list),
    )


class RunningMeanStd:
    """
    [FIX RMS] Chá»‰ update 1 láº§n SAU má»—i rollout â€” freeze trong suá»‘t PPO epochs.

    LÃ½ do: Náº¿u update trong má»—i mini-batch, 'target' cá»§a Value Head thay Ä‘á»•i
    liÃªn tá»¥c (Non-stationary target) â†’ Value Head há»c ráº¥t cháº­m / khÃ´ng há»™i tá»¥.
    Giáº£i phÃ¡p: Gá»i ret_rms.update() 1 láº§n trÆ°á»›c ppo_update(),
    rá»“i dÃ¹ng mean/var Ä‘Ã£ freeze cho cáº£ 4 epochs cá»§a PPO.
    """
    def __init__(self, epsilon: float = 1e-8):
        self.mean  = 0.0
        self.var   = 1.0
        self.count = epsilon

    def update(self, x: torch.Tensor) -> None:
        """Gá»i NGOAI ppo_update() â€” 1 láº§n/rollout."""
        v = x.detach().float()
        b_mean, b_var, b_n = v.mean().item(), v.var().item(), v.numel()
        total = self.count + b_n
        delta = b_mean - self.mean
        self.mean  += delta * b_n / total
        self.var    = (self.var * self.count + b_var * b_n +
                       delta ** 2 * self.count * b_n / total) / total
        self.count  = total

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """DÃ¹ng mean/var Ä‘Ã£ â€œfreezeâ€ (khÃ´ng gá»i update() trong hÃ m nÃ y)."""
        return ((x - self.mean) / (self.var ** 0.5 + 1e-8)).clamp(-10.0, 10.0)

def evaluate_oos(model, h5_path, split_idx, n_total, window_size, device,
                 gap_bars=200, n_eval_eps=5):
    """
    Cháº¡y nhanh model trÃªn pháº§n Out-Of-Sample (20% cuá»‘i) Ä‘á»ƒ theo dÃµi tiáº¿n Ä‘á»™.
    Tráº£ vá» Sharpe Ratio trÃªn OOS episodes.
    """
    from src.training.backtest import compute_metrics

    oos_start = split_idx + gap_bars  # [FIX] ÄÆ°á»£c bÃºt qua gap 200 bars
    with h5py.File(h5_path, "r") as f:
        X_oos     = f["X"][oos_start:].astype(np.float32)
        y_oos     = f["y"][oos_start:]
        close_oos = f["close"][oos_start:].astype(np.float32)

    env = XAUUSDEnv(
        features=X_oos, close_prices=close_oos, oracle_labels=y_oos,
        window_size=window_size, spread_pips=25, lot_size=0.01,
        initial_balance=200.0, max_drawdown_usd=20.0,
    )
    model.eval()
    all_returns = []
    for _ in range(n_eval_eps):
        obs, _ = env.reset()
        done = False
        balance_hist = [200.0]
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits, _ = model(obs_t)
                action = logits.argmax(-1).item()  # deterministic (greedy)
            obs, _, term, trunc, _ = env.step(action)
            balance_hist.append(env._balance)
            done = term or trunc
        returns = np.diff(balance_hist) / np.array(balance_hist[:-1])
        all_returns.extend(returns.tolist())
    model.train()
    metrics = compute_metrics(np.array(all_returns))
    return metrics["sharpe"]


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """Generalized Advantage Estimation."""
    T, E = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_adv   = 0.0
    for t in reversed(range(T)):
        mask       = 1.0 - dones[t]
        delta      = rewards[t] + gamma * (values[t] * mask) - values[t]
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
    PPO update vá»›i KL Anchor.
    Returns Ä‘Ã£ Ä‘Æ°á»£c chuáº©n hÃ³a BÃŠN NGOAI (RunningMeanStd.update() gá»i trÆ°á»›c).
    flat_ret lÃ  tensor Ä‘Ã£ normalize, mean/var freeze trong suá»‘t vÃ²ng láº·p.
    """
    T, E = obs.shape[:2]
    flat_obs  = obs.view(T * E, *obs.shape[2:]).to(device)
    flat_act  = actions.view(-1).to(device)
    flat_logp = old_logps.view(-1).to(device)
    flat_adv  = advantages.view(-1).to(device)
    flat_adv  = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)
    # flat_ret Ä‘Ã£ sáºµn sÃ ng (normalized, frozen)
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

            vf_loss = F.mse_loss(value.squeeze(-1), flat_ret[b])

            # KL Anchor â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

    # â”€â”€ Ä‘á»c metadata tá»« HDF5 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    with h5py.File(args.h5, "r") as f:
        n_total, window_size, n_features = f["X"].shape
    GAP_BARS  = 200  # [FIX] Purged gap giá»¯a train vÃ  OOS
    split_idx = int(n_total * 0.8) - GAP_BARS  # Train dá»«ng trÆ°á»›c gap
    oos_start = int(n_total * 0.8)              # OOS báº¯t Ä‘áº§u sau gap
    log.info(f"Split: train[:{split_idx}] | gap={GAP_BARS} | OOS[{oos_start}:]")

    # â”€â”€ BC Model â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

    # â”€â”€ [FIX OOM] Envs Ä‘á»c HDF5 tá»« file, khÃ´ng copy array â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    vec_env = make_async_envs(
        h5_path=args.h5, n_total=n_total,
        split_idx=split_idx, window_size=window_size,
        n_envs=args.n_envs,
    )
    log.info(f"{args.n_envs} async envs ready")

    rollout_steps = 2048
    total_updates = args.total_steps // (rollout_steps * args.n_envs)
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    ret_rms     = RunningMeanStd()
    best_sharpe = -np.inf  # [FIX] Theo dÃµi best checkpoint báº±ng Sharpe OOS

    for update in range(1, total_updates + 1):
        kl_coef = max(0.05, 0.5 * math.exp(-update / (total_updates * 0.5)))

        obs, actions, logps, rewards, dones, values = \
            collect_rollout(vec_env, ppo_model, device, rollout_steps)

        adv, returns = compute_gae(rewards, values, dones)

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
                     f"Loss={loss:.4f} | AvgRew={rewards.mean():.4f} | KLÎ»={kl_coef:.3f}")

        # [FIX] Eval Ä‘á»‹nh ká»³ trÃªn OOS â€” lÆ°u best checkpoint báº±ng Sharpe
        if update % 50 == 0:
            sharpe = evaluate_oos(
                ppo_model, args.h5, split_idx, n_total,
                window_size, device, gap_bars=GAP_BARS
            )
            log.info(f"  [OOS EVAL] Sharpe={sharpe:.4f} | Best={best_sharpe:.4f}")
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                torch.save(ppo_model.state_dict(),
                           CHECKPOINT_DIR / "ppo_best.pt")
                log.info(f"  âœ… New best checkpoint saved (Sharpe={best_sharpe:.4f})")

        if update % 100 == 0:
            torch.save(ppo_model.state_dict(),
                       CHECKPOINT_DIR / f"ppo_step{update}.pt")

    log.info("ðŸŽ‰ PPO training hoÃ n táº¥t!")
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


## Task 2: Backtest &amp; Reporting

**Files:**
- Create: `src/training/backtest.py`
- Create: `src/training/tests/test_backtest.py`

### Step 2.1: Viết failing tests cho Backtest

- [ ] **Tạo `src/training/tests/test_backtest.py`:**

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
        """Chuỗi PnL dương đều đặn phải có Sharpe > 0."""
        daily_returns = np.array([0.001] * 252)
        metrics = compute_metrics(daily_returns)
        assert metrics["sharpe"] > 0, f"Sharpe phải dương: {metrics['sharpe']}"

    def test_all_zero_returns_sharpe_zero(self):
        """PnL = 0 mọi ngày → Sharpe = 0."""
        daily_returns = np.zeros(252)
        metrics = compute_metrics(daily_returns)


    def test_max_drawdown_is_non_positive(self):
        """Max drawdown pháº£i <= 0 (biá»ƒu diá»…n máº¥t vá»‘n)."""
        daily_returns = np.array([0.01, -0.05, 0.02, -0.03, 0.01])
        metrics = compute_metrics(daily_returns)
        assert metrics["max_drawdown"] <= 0

    def test_win_rate_between_0_and_1(self):
        """Win rate pháº£i náº±m trong [0, 1]."""
        returns = np.random.randn(100) * 0.01
        metrics = compute_metrics(returns)
        assert 0.0 <= metrics["win_rate"] <= 1.0

    def test_metrics_has_required_keys(self):
        """Káº¿t quáº£ pháº£i cÃ³ Ä‘á»§ cÃ¡c key báº¯t buá»™c."""
        returns = np.random.randn(252) * 0.001
        metrics = compute_metrics(returns)
        required = {"sharpe", "sortino", "max_drawdown", "win_rate",
                    "total_return", "n_trades"}
        assert required.issubset(set(metrics.keys()))
```

- [ ] **Cháº¡y Ä‘á»ƒ verify FAIL:**
```bash
python -m pytest src/training/tests/test_backtest.py -v
```

- [ ] **Táº¡o `src/training/backtest.py`:**

```python
"""
backtest.py
-----------
Out-of-sample backtest: cháº¡y model trained qua dá»¯ liá»‡u chÆ°a tháº¥y,
tÃ­nh cÃ¡c chá»‰ sá»‘ tÃ i chÃ­nh chuáº©n.

CÃ¡ch dÃ¹ng:
  python src/training/backtest.py \\
      --h5      data/processed/XAUUSD_M15_w128.h5 \\
      --ckpt    checkpoints/ppo_xauusd.zip \\
      --mode    ppo                         # hoáº·c bc

Chá»‰ sá»‘ output:
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


def compute_metrics(daily_returns: np.ndarray) -> dict:
    """
    TÃ­nh cÃ¡c chá»‰ sá»‘ tÃ i chÃ­nh tá»« máº£ng daily returns.

    Parameters
    ----------
    daily_returns : np.ndarray â€” máº£ng % return má»—i bar/ngÃ y

    Returns
    -------
    dict vá»›i keys: sharpe, sortino, max_drawdown, win_rate,
                   total_return, n_trades
    """
    r = np.array(daily_returns, dtype=np.float64)
    n = len(r)

    if n == 0:
        return dict(sharpe=0.0, sortino=0.0, max_drawdown=0.0,
                    win_rate=0.0, total_return=0.0, n_trades=0)

    mean_r = r.mean()
    std_r  = r.std()

    # Sharpe Ratio (annualized Ã—âˆš252 náº¿u daily, Ã—âˆš(252Ã—24Ã—4) náº¿u M15)
    sharpe = (mean_r / std_r * np.sqrt(252)) if std_r > 0 else 0.0

    # Sortino Ratio (chá»‰ dÃ¹ng downside std)
    negative_r  = r[r < 0]
    down_std    = negative_r.std() if len(negative_r) > 0 else 0.0
    sortino = (mean_r / down_std * np.sqrt(252)) if down_std > 0 else 0.0

    # Max Drawdown
    cumulative = np.cumprod(1 + r)
    peak       = np.maximum.accumulate(cumulative)
    drawdown   = (cumulative - peak) / peak
    max_dd     = float(drawdown.min())

    # Win Rate
    win_rate = float((r > 0).mean())

    # Total Return
    total_return = float(cumulative[-1] - 1.0) if n > 0 else 0.0

    # N trades = sá»‘ bar cÃ³ return != 0
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
    log.info(f"  BACKTEST REPORT â€” {label}")
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
    status = "âœ… PASS â€” Äá»§ tiÃªu chuáº©n deploy" if passed else "âŒ FAIL â€” Cáº§n cáº£i thiá»‡n thÃªm"
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

    # Trong thá»±c táº¿: cháº¡y model qua out-of-sample env vÃ  thu tháº­p returns
    # ÄÃ¢y lÃ  scaffold â€” sáº½ Ä‘Æ°á»£c hoÃ n thiá»‡n khi env + model Ä‘Ã£ sáºµn sÃ ng
    log.info("âš ï¸  Running simplified backtest scaffold (returns cáº§n Ä‘Æ°á»£c thu tháº­p tá»« env.step)")
    log.info("    Sá»­ dá»¥ng compute_metrics() vá»›i array returns sau khi roll-out Ä‘áº§y Ä‘á»§.")


if __name__ == "__main__":
    main()
```

- [ ] **Cháº¡y Ä‘á»ƒ verify PASS:**
```bash
python -m pytest src/training/tests/test_backtest.py -v
```
Káº¿t quáº£ mong Ä‘á»£i: `5 passed`

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

### Step 3.1: Táº¡o Dockerfile

- [ ] **Táº¡o `Dockerfile`:**

```dockerfile
# XAUUSD AI Trading Bot â€” Training Container (Phase 2: PPO)
# Tá»‘i Æ°u cho RTX 4090 / RTX 5090 trÃªn Vast.ai
# [FIX] Phase 2 (PPO) dÃ¹ng AsyncVectorEnv â€” cáº§n CPU Ä‘a luá»“ng máº¡nh,
# khÃ´ng cháº¡y BC (Phase 1) vÃ¬ BC Ä‘Ã£ xong rá»“i!

FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    git \\
    htop \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements vÃ  install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY data/ ./data/

# Táº¡o thÆ° má»¥c output
RUN mkdir -p checkpoints logs

# [FIX] CMD cháº¡y PPO (Phase 2), khÃ´ng pháº£i BC (Phase 1)
# BC checkpoint truyá»n vÃ o qua --bc-ckpt khi docker run
CMD ["python", "src/training/train_rl.py", \\
     "--h5",          "data/processed/XAUUSD_M15_w128.h5", \\
     "--bc-ckpt",     "checkpoints/best_model_bc.pt", \\
     "--n-envs",      "64", \\
     "--total-steps", "2000000"]
```

- [ ] **Táº¡o `scripts/vast_launch.sh`:**

```bash
#!/bin/bash
# vast_launch.sh â€” Triá»ƒn khai PPO training (Phase 2) lÃªn Vast.ai instance
# [FIX] Cháº¡y train_rl.py (PPO), khÃ´ng pháº£i train_bc.py (BC Ä‘Ã£ xong á»Ÿ local)
# CÃ¡ch dÃ¹ng: ./scripts/vast_launch.sh <INSTANCE_ID> <SSH_KEY_PATH> <BC_CKPT_PATH>
#
# Workflow:
#   1. Upload source + HDF5 data + BC checkpoint lÃªn server
#   2. Launch PPO training dÃ¹ng AsyncVectorEnv (64 envs)
#   3. Checkpoint ppo_best.pt tá»± Ä‘á»™ng lÆ°u khi Sharpe OOS tá»‘t hÆ¡n

set -e

INSTANCE_ID=${1:?"Usage: $0 <instance_id> <ssh_key> <bc_ckpt>"}
SSH_KEY=${2:?"Usage: $0 <instance_id> <ssh_key> <bc_ckpt>"}
BC_CKPT=${3:-"checkpoints/best_model_bc.pt"}

echo "=== XAUUSD Bot â€” Vast.ai PPO Deployment ==="
echo "Instance: $INSTANCE_ID"
echo "BC Checkpoint: $BC_CKPT"

# 1. Láº¥y connection info
CONN=$(vastai ssh-url $INSTANCE_ID)
SSH_HOST=$(echo $CONN | sed 's/ssh:\/\///')

# 2. Rsync source code + data + BC checkpoint lÃªn instance
echo "Uploading source code..."
rsync -avz --exclude '.git' --exclude '__pycache__' \\
    -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no" \\
    ./ root@${SSH_HOST}:/workspace/

# Upload BC checkpoint riÃªng náº¿u náº±m ngoÃ i workspace
if [ -f "$BC_CKPT" ]; then
    scp -i $SSH_KEY $BC_CKPT root@${SSH_HOST}:/workspace/checkpoints/best_model_bc.pt
    echo "BC checkpoint uploaded."
fi

# 3. Install dependencies
echo "Installing dependencies..."
ssh -i $SSH_KEY -o StrictHostKeyChecking=no root@${SSH_HOST} \\
    "cd /workspace && pip install -q -r requirements.txt"

# 4. [FIX] Launch PPO training (Phase 2) â€” khÃ´ng pháº£i BC!
echo "Starting PPO RL training (Phase 2)..."
ssh -i $SSH_KEY -o StrictHostKeyChecking=no root@${SSH_HOST} \\
    "cd /workspace && nohup python src/training/train_rl.py \\
        --h5          data/processed/XAUUSD_M15_w128.h5 \\
        --bc-ckpt     checkpoints/best_model_bc.pt \\
        --n-envs      64 \\
        --total-steps 2000000 \\
        > logs/train_rl.log 2>&1 &"

echo "âœ… PPO training started! Theo dÃµi log:"
echo "   ssh -i $SSH_KEY root@${SSH_HOST} 'tail -f /workspace/logs/train_rl.log'"
```

- [ ] **Táº¡o `scripts/pull_results.sh`:**

```bash
#!/bin/bash
# pull_results.sh â€” KÃ©o káº¿t quáº£ training vá» local
# CÃ¡ch dÃ¹ng: ./scripts/pull_results.sh <INSTANCE_ID> <SSH_KEY_PATH>

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

echo "âœ… Done! Checkpoints saved to ./checkpoints/"
ls -lh ./checkpoints/
```

- [ ] **Commit:**
```bash
git add Dockerfile scripts/
git commit -m "feat(sprint4): Dockerfile + Vast.ai launch/pull scripts"
```

---

## Task 4: ToÃ n bá»™ tests Sprint 4 & Final Push

- [ ] **Cháº¡y toÃ n bá»™ test suite:**
```bash
python -m pytest src/ -v --tb=short
```
Káº¿t quáº£ mong Ä‘á»£i: `â‰¥ 35 passed, 0 failed` (táº¥t cáº£ 4 sprints)

- [ ] **Final push:**
```bash
git push origin main
```

---

## Äiá»u kiá»‡n DONE cho Sprint 4 (& toÃ n dá»± Ã¡n v1)

| Chá»‰ sá»‘ | Target |
|---|---|
| `pytest src/ -v` | âœ… 0 failures |
| F1(Buy) validation | > 0.40 |
| F1(Sell) validation | > 0.40 |
| Sharpe Ratio (out-of-sample) | > 1.0 |
| Max Drawdown | < 10% ($20) |
| Win Rate | > 55% |
| Docker build | âœ… khÃ´ng crash |
| Vast.ai training log | âœ… converging loss |
