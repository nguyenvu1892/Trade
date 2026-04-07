# XAUUSD Bot — Sprint 4: RL Fine-tuning & Cloud Scale-up (Phase 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune model BC bằng PPO Reinforcement Learning (CleanRL) — tối ưu Drawdown, Sharpe, Win Rate trên out-of-sample. Đóng gói Docker để triển khai lên Vast.ai GPU server.

**Architecture:** Load `best_model_bc.pt` → sử dụng **CleanRL** (không dùng SB3) — toàn bộ loss logic trong 1 file duy nhất, dễ đọc và chỉnh sửa. Transformer được nhúng trực tiếp vào PPO actor-critic, không bị flatten. KL-Divergence anchor chống Catastrophic Forgetting được chèn thẳng vào training loop chỉ 2 dòng toán học.

**Lý do KHÔNG dùng Stable-Baselines3 (SB3):**
- SB3 mặc định **flatten** input 2D `(128, 12)` thành `(1536,)` trước khi đưa vào mạng — phá vỡ cấu trúc thời gian của Transformer hoàn toàn.
- Việc hack SB3 để thêm KL Penalty loss tùy chỉnh đòi hỏi monkey-patch rất phức tạp và dễ gây regression.
- CleanRL giải quyết cả 2 vấn đề chỉ với 2 dòng code.

**Tech Stack:** PyTorch, CleanRL (single-file PPO), gymnasium, Docker, quantstats, pytest

---

## File Structure

```
src/training/
├── train_rl.py              [NEW] — CleanRL PPO single-file + KL anchor + Transformer
├── backtest.py              [NEW] — Out-of-sample evaluation & report
└── tests/
    └── test_backtest.py     [NEW]

Dockerfile                   [NEW] — Container cho Vast.ai
scripts/
├── vast_launch.sh           [NEW] — Script khởi động training trên Vast.ai
└── pull_results.sh          [NEW] — Script kéo kết quả về local
```

---

## Task 1: PPO Training với KL-Divergence Anchor

**Files:**
- Create: `src/training/train_rl.py`

### Step 1.1: Implement train_rl.py (CleanRL style)

> ⚠️ **Không dùng SB3.** Dùng CleanRL approach: tài PPO thụ công trong 1 file, chèn Transformer và KL Anchor trực tiếp.

- [ ] **Tạo `src/training/train_rl.py`:**

```python
"""
train_rl.py  (CleanRL-style — v3, fixed)
-----------------------------------------
Phase 2: PPO Reinforcement Learning Fine-tuning.

Fix log:
  v1: initial CleanRL draft
  v2: AsyncVectorEnv để song song hóa CPU
  v3: [FIX OOM] Dùng h5_path + offset thay vì truyền raw array
      [FIX SYN] compute_gae thiếu 'def'
      [FIX DC]  Xóa dead code sau return trong make_async_envs
      [FIX RMS] Freeze RunningMeanStd trong PPO epochs, chỉ update 1 lần/rollout

Cách dùng:
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
    h5_path:      str,   # [FIX OOM] Chỉ truyền đường dẫn, không truyền array
    n_total:      int,
    split_idx:    int,
    window_size:  int,
    n_envs:       int,
):
    """
    [FIX OOM] Truyền h5_path + offset vào closure thay vì raw numpy array.

    Vấn đề: multiprocessing pickle toàn bộ X (vài GB) thành 64 bản sao.
    Giải pháp: Mỗi worker tự đọc slice dữ liệu từ file HDF5 của chính nó.
    HDF5 hỗ trợ concurrent read, không cần lock.
    """
    def _make_env_fn(offset: int):
        def _init():
            # Mỗi worker mở file HDF5 trong process riêng của nó
            with h5py.File(h5_path, "r") as f:
                X_slice      = f["X"][offset:split_idx].astype(np.float32)
                label_slice  = f["y"][offset:split_idx]
            # close_prices: placeholder — thay bằng real close khi có pipeline
            close_slice = np.ones(len(X_slice), dtype=np.float32) * 1900.0
            return XAUUSDEnv(
                features         = X_slice,
                close_prices     = close_slice,
                oracle_labels    = label_slice,
                window_size      = window_size,
                spread_pips      = 25,
                lot_size         = 0.01,
                initial_balance  = 200.0,
                max_drawdown_usd = 20.0,
            )
        return _init

    # [FIX DC] Không có dead code sau return
    return AsyncVectorEnv([
        _make_env_fn((i * 500) % max(1, split_idx - window_size - 2000))
        for i in range(n_envs)
    ])


def collect_rollout(vec_env, model, device, rollout_steps=2048):
    """
    Thu thập rollout từ AsyncVectorEnv — tất cả envs step() song song.
    Trả về (obs, actions, log_probs, rewards, dones, values).
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
    [FIX RMS] Chỉ update 1 lần SAU mỗi rollout — freeze trong suốt PPO epochs.

    Lý do: Nếu update trong mỗi mini-batch, 'target' của Value Head thay đổi
    liên tục (Non-stationary target) → Value Head học rất chậm / không hội tụ.
    Giải pháp: Gọi ret_rms.update() 1 lần trước ppo_update(),
    rồi dùng mean/var đã freeze cho cả 4 epochs của PPO.
    """
    def __init__(self, epsilon: float = 1e-8):
        self.mean  = 0.0
        self.var   = 1.0
        self.count = epsilon

    def update(self, x: torch.Tensor) -> None:
        """Gọi NGOAI ppo_update() — 1 lần/rollout."""
        v = x.detach().float()
        b_mean, b_var, b_n = v.mean().item(), v.var().item(), v.numel()
        total = self.count + b_n
        delta = b_mean - self.mean
        self.mean  += delta * b_n / total
        self.var    = (self.var * self.count + b_var * b_n +
                       delta ** 2 * self.count * b_n / total) / total
        self.count  = total

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Dùng mean/var đã “freeze” (không gọi update() trong hàm này)."""
        return ((x - self.mean) / (self.var ** 0.5 + 1e-8)).clamp(-10.0, 10.0)


# [FIX SYN] Thêm 'def' bị thiếu do copy/paste lỗi
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
    PPO update với KL Anchor.
    Returns đã được chuẩn hóa BÊN NGOAI (RunningMeanStd.update() gọi trước).
    flat_ret là tensor đã normalize, mean/var freeze trong suốt vòng lặp.
    """
    T, E = obs.shape[:2]
    flat_obs  = obs.view(T * E, *obs.shape[2:]).to(device)
    flat_act  = actions.view(-1).to(device)
    flat_logp = old_logps.view(-1).to(device)
    flat_adv  = advantages.view(-1).to(device)
    flat_adv  = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)
    # flat_ret đã sẵn sàng (normalized, frozen)
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

            # KL Anchor ─────────────────────────────────────
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

    # ── đọc metadata từ HDF5 (không load toàn bộ vào RAM main process) ──
    with h5py.File(args.h5, "r") as f:
        n_total, window_size, n_features = f["X"].shape
    split_idx = int(n_total * 0.8)
    log.info(f"Dataset: N={n_total}, W={window_size}, F={n_features} | split={split_idx}")

    # ── BC Model ───────────────────────────────────────────────────
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

    # ── [FIX OOM] Envs đọc HDF5 từ file, không copy array ──────────────
    vec_env = make_async_envs(
        h5_path=args.h5, n_total=n_total,
        split_idx=split_idx, window_size=window_size,
        n_envs=args.n_envs,
    )
    log.info(f"{args.n_envs} async envs ready")

    rollout_steps = 2048
    total_updates = args.total_steps // (rollout_steps * args.n_envs)
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    ret_rms = RunningMeanStd()  # [FIX RMS] khởi tạo ngoài vòng lặp

    for update in range(1, total_updates + 1):
        kl_coef = max(0.05, 0.5 * math.exp(-update / (total_updates * 0.5)))

        obs, actions, logps, rewards, dones, values = \
            collect_rollout(vec_env, ppo_model, device, rollout_steps)

        adv, returns = compute_gae(rewards, values, dones)

        # [FIX RMS] Update RMS 1 lần sau rollout, FREEZE khi ppo_update()
        raw_ret = returns.view(-1)
        ret_rms.update(raw_ret)
        flat_ret_norm = ret_rms.normalize(raw_ret)  # tensor, freeze từ đây

        loss = ppo_update(
            ppo_model, bc_model, optimizer,
            obs, actions, logps, flat_ret_norm, adv,
            device, kl_coef=kl_coef
        )

        if update % 10 == 0:
            log.info(f"Update {update:4d}/{total_updates} | "
                     f"Loss={loss:.4f} | AvgRew={rewards.mean():.4f} | KLλ={kl_coef:.3f}")

        if update % 100 == 0:
            torch.save(ppo_model.state_dict(),
                       CHECKPOINT_DIR / f"ppo_step{update}.pt")

    log.info("🎉 PPO training hoàn tất!")
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
----------------------------
Phase 2: PPO Reinforcement Learning Fine-tuning.

KHÔNG dùng Stable-Baselines3 vì:
  - SB3 flatten input (128, 12) → (1536,) phá Transformer
  - Khó chèn custom KL Loss vào SB3 PPO internals

Dung CleanRL approach: toàn bộ PPO logic trong 1 file, trong suốt.
Transformer được nhúng trực tiếp vào actor-critic — không qua wrapper.
KL anchor chỉ cần 2 dòng toán học trong vòng lặp loss.

Cách dùng:
  python src/training/train_rl.py \\
      --h5      data/processed/XAUUSD_M15_w128.h5 \\
      --bc-ckpt checkpoints/best_model_bc.pt \\
      --n-envs  64 \\
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.model.transformer import XAUTransformer
from src.env.xauusd_env import XAUUSDEnv

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

CHECKPOINT_DIR = Path("checkpoints")


def make_async_envs(
    X,
    close_prices,
    oracle_labels,
    window_size:  int,
    n_envs:       int,
):
    """
    Tạo AsyncVectorEnv — 64 processes độc lập chạy song song trên CPU.
    Không dùng list comprehension (sequential) vì sẽ block GPU.

    gymnasium.vector.AsyncVectorEnv đẩy mỗi env sang 1 process riêng,
    tất cả step() chạy đồng thời → GPU luôn có batch mới để xử lý.
    """
    from gymnasium.vector import AsyncVectorEnv
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.env.xauusd_env import XAUUSDEnv

    n = len(X)
    def _make_env_fn(i):
        def _init():
            offset = (i * 500) % max(1, n - window_size - 2000)
            return XAUUSDEnv(
                features         = X[offset:],
                close_prices     = close_prices[offset:],
                oracle_labels    = oracle_labels[offset:],
                window_size      = window_size,
                spread_pips      = 25,
                lot_size         = 0.01,
                initial_balance  = 200.0,
                max_drawdown_usd = 20.0,
            )
        return _init

    return AsyncVectorEnv([_make_env_fn(i) for i in range(n_envs)])
    envs = []
    n = len(X)
    for i in range(n_envs):
        offset = (i * 500) % max(1, n - window_size - 2000)
        env = XAUUSDEnv(
            features         = X[offset:],
            close_prices     = close_prices[offset:],
            oracle_labels    = oracle_labels[offset:],
            window_size      = window_size,
            spread_pips      = 25,
            lot_size         = 0.01,
            initial_balance  = 200.0,
            max_drawdown_usd = 20.0,
        )
        envs.append(env)
    return envs


def collect_rollout(vec_env, model, device, rollout_steps=2048):
    """
    Thu thập rollout từ AsyncVectorEnv — tất cả envs step() song song.
    Trả về (obs, actions, log_probs, rewards, dones, values).
    """
    obs_list, act_list, logp_list, rew_list, done_list, val_list = \
        [], [], [], [], [], []

    obs, _ = vec_env.reset()

    for _ in range(rollout_steps):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device)  # (E, W, F)
        with torch.no_grad():
            logits, value = model(obs_t)
            dist   = Categorical(logits=logits)
            action = dist.sample()
            logp   = dist.log_prob(action)

        obs_list.append(obs_t.cpu())
        act_list.append(action.cpu())
        logp_list.append(logp.cpu())
        val_list.append(value.squeeze(-1).cpu())

        # [FIX] AsyncVectorEnv.step() chạy song song trên CPU — không sequential!
        next_obs, rewards, terms, truncs, _ = vec_env.step(action.cpu().numpy())
        rew_list.append(torch.tensor(rewards, dtype=torch.float32))
        dones = torch.tensor(terms | truncs, dtype=torch.float32)
        done_list.append(dones)
        obs = next_obs

    return (
        torch.stack(obs_list),      # (T, E, W, F)
        torch.stack(act_list),      # (T, E)
        torch.stack(logp_list),     # (T, E)
        torch.stack(rew_list),      # (T, E)
        torch.stack(done_list),     # (T, E)
        torch.stack(val_list),      # (T, E)
    )

class RunningMeanStd:
    """
    Nhớ Mean và Std của Returns theo thời gian — duy trì qua các updates.
    Chuẩn hóa Returns về N(0,1) trước khi tính MSE cho Value Head.

    Lý do cần thiết: Returns USD biến động -$20 → +$50.
    Value Head cần dự đoán số không có scale context → Gradient nổ / hội tụ chậm.
    """
    def __init__(self, epsilon=1e-8):
        self.mean  = 0.0
        self.var   = 1.0
        self.count = epsilon

    def update(self, x: torch.Tensor):
        batch_mean = x.mean().item()
        batch_var  = x.var().item()
        batch_count = x.numel()

        total  = self.count + batch_count
        delta  = batch_mean - self.mean
        self.mean  += delta * batch_count / total
        self.var    = ((self.var * self.count + batch_var * batch_count +
                       delta ** 2 * self.count * batch_count / total) / total)
        self.count  = total

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Trả về (x - mean) / std, clip [-10, 10] để tránh giá trị cực trị."""
        return ((x - self.mean) / (self.var ** 0.5 + 1e-8)).clamp(-10, 10)


(rewards, values, dones, gamma=0.99, lam=0.95):
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
    obs, actions, old_logps, returns, advantages,
    device,
    ret_normalizer,    # [NEW] RunningMeanStd instance
    clip_eps  = 0.2,
    vf_coef   = 0.5,
    ent_coef  = 0.01,
    kl_coef   = 0.3,
    n_epochs  = 4,
    batch_size = 256,
):
    """
    PPO update với KL Anchor + [NEW] Return Normalization cho Value Head.

    Total_Loss = PPO_clip + vf_coef * MSE(V(s), normalized_returns)
               - ent_coef * Entropy
               + kl_coef * KL(pi_PPO || pi_BC)
    """
    T, E = obs.shape[:2]
    flat_obs  = obs.view(T * E, *obs.shape[2:]).to(device)
    flat_act  = actions.view(-1).to(device)
    flat_logp = old_logps.view(-1).to(device)
    flat_adv  = advantages.view(-1).to(device)
    flat_adv  = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)

    # [NEW] Chuẩn hóa Returns trước khi dùng cho Value Head
    raw_returns  = returns.view(-1)
    ret_normalizer.update(raw_returns)
    flat_ret = ret_normalizer.normalize(raw_returns).to(device)

    n_samples = T * E
    losses    = []

    for _ in range(n_epochs):
        idx = torch.randperm(n_samples, device=device)
        for start in range(0, n_samples, batch_size):
            b = idx[start:start + batch_size]
            b_obs = flat_obs[b]
            b_act = flat_act[b]

            logits, value = model(b_obs)
            dist     = Categorical(logits=logits)
            logp     = dist.log_prob(b_act)
            entropy  = dist.entropy().mean()

            # PPO Clip Loss
            ratio    = (logp - flat_logp[b]).exp()
            pg_loss  = -torch.min(
                ratio * flat_adv[b],
                ratio.clamp(1 - clip_eps, 1 + clip_eps) * flat_adv[b]
            ).mean()

            # Value Loss
            vf_loss = F.mse_loss(value.squeeze(-1), flat_ret[b])

            # KL Anchor (2 dòng) ──────────────────────────────────
            with torch.no_grad():
                bc_logits, _ = bc_model(b_obs)
            kl_loss = F.kl_div(          # Giữ pi_PPO gần với pi_BC
                F.log_softmax(logits, dim=-1),
                F.softmax(bc_logits, dim=-1),
                reduction="batchmean",
            )
            # ─────────────────────────────────────────────────

            loss = (pg_loss
                    + vf_coef * vf_loss
                    - ent_coef * entropy
                    + kl_coef * kl_loss)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            losses.append(loss.item())

    return np.mean(losses)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ── Load Dataset ─────────────────────────────────────────────
    with h5py.File(args.h5, "r") as f:
        X      = f["X"][:]   # (N, W, F)
        labels = f["y"][:]
    _, window_size, n_features = X.shape
    split     = int(len(X) * 0.8)
    close     = np.ones(len(X)) * 1900.0   # Placeholder — thay bằng real close
    X_train   = X[:split]
    lbl_train = labels[:split]
    close_tr  = close[:split]

    # ── Load BC Model (frozen anchor) ────────────────────────────
    ckpt = torch.load(args.bc_ckpt, map_location=device)
    bc_model = XAUTransformer(n_features=n_features, window_size=window_size,
                              d_model=256, n_heads=8, n_layers=6).to(device)
    bc_model.load_state_dict(ckpt["model_state"])
    bc_model.eval()
    for p in bc_model.parameters():
        p.requires_grad = False
    log.info(f"BC checkpoint loaded (F1_buy={ckpt['f1_buy']:.3f})")

    # PPO model khởi tạo từ BC weights (không random!)
    ppo_model = XAUTransformer(n_features=n_features, window_size=window_size,
                               d_model=256, n_heads=8, n_layers=6).to(device)
    ppo_model.load_state_dict(ckpt["model_state"])  # Muờn weights từ BC
    optimizer = optim.AdamW(ppo_model.parameters(), lr=args.lr, weight_decay=1e-4)

    # ── Envs ────────────────────────────────────────────────────
    envs = make_envs(X_train, close_tr, lbl_train, window_size, args.n_envs)
    log.info(f"{args.n_envs} envs tạo xong")

    rollout_steps = 2048
    total_updates = args.total_steps // (rollout_steps * args.n_envs)
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    for update in range(1, total_updates + 1):
        # KL lambda giảm dần 0.5 → 0.05 qua các updates
        kl_coef = max(0.05, 0.5 * math.exp(-update / (total_updates * 0.5)))

        obs, actions, logps, rewards, dones, values = \
            collect_rollout(envs, ppo_model, device, rollout_steps)

        adv, returns = compute_gae(rewards, values, dones)

        loss = ppo_update(
            ppo_model, bc_model, optimizer,
            obs, actions, logps, returns, adv,
            device, kl_coef=kl_coef
        )

        if update % 10 == 0:
            avg_rew = rewards.mean().item()
            log.info(f"Update {update:4d}/{total_updates} | "
                     f"Loss={loss:.4f} | AvgReward={avg_rew:.4f} | KL_lambda={kl_coef:.3f}")

        if update % 100 == 0:
            ckpt_path = CHECKPOINT_DIR / f"ppo_step{update}.pt"
            torch.save(ppo_model.state_dict(), ckpt_path)
            log.info(f"  ✅ Checkpoint saved: {ckpt_path}")

    log.info("🎉 PPO training hoàn tất!")
    torch.save(ppo_model.state_dict(), CHECKPOINT_DIR / "ppo_final.pt")


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

```python
"""
train_rl.py
-----------
Phase 2: PPO Reinforcement Learning Fine-tuning.

Load model BC đã train → wrap vào SB3 CustomActorCriticPolicy →
Train PPO trong XAUUSDEnv với Vectorized Environments.

Anti-Catastrophic Forgetting: KL-Divergence Anchor
  Total Loss = PPO_Loss + λ(t) × KL(π_PPO || π_BC)
  λ giảm dần từ 0.5 → 0.05 theo số training steps.

Cách dùng:
  python src/training/train_rl.py \\
      --h5      data/processed/XAUUSD_M15_w128.h5 \\
      --bc-ckpt checkpoints/best_model_bc.pt \\
      --n-envs  64 \\
      --steps   2000000
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Callable

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
import gymnasium as gym

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.model.transformer import XAUTransformer
from src.env.xauusd_env import XAUUSDEnv

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

CHECKPOINT_DIR = Path("checkpoints")


def load_h5_arrays(h5_path: str):
    """Nạp toàn bộ X (features) và y (labels) từ HDF5 vào RAM."""
    with h5py.File(h5_path, "r") as f:
        X = f["X"][:]   # (N, window, n_features)
        y = f["y"][:]   # (N,)
    return X, y


def make_env_fn(
    features:      np.ndarray,
    close_prices:  np.ndarray,
    oracle_labels: np.ndarray,
    window_size:   int,
    start_offset:  int = 0,
) -> Callable:
    """Factory function tạo môi trường — cần thiết cho SubprocVecEnv."""
    def _init():
        env = XAUUSDEnv(
            features      = features[start_offset:],
            close_prices  = close_prices[start_offset:],
            oracle_labels = oracle_labels[start_offset:],
            window_size   = window_size,
            spread_pips   = 25,
            lot_size      = 0.01,
            initial_balance  = 200.0,
            max_drawdown_usd = 20.0,
        )
        return env
    return _init


class KLAnchorCallback:
    """
    Callback tính KL penalty giữa π_PPO và π_BC (frozen).
    Tích hợp vào training thông qua monkey-patch loss.
    
    Lưu ý: SB3 không hỗ trợ custom loss trực tiếp.
    Giải pháp thực tế: chạy evaluate_actions() + KL thủ công
    và cộng vào entropy loss.
    λ giảm dần: lambda_kl = 0.5 × exp(-step / decay_steps)
    """
    def __init__(self, bc_model: XAUTransformer, decay_steps: int = 500_000):
        self.bc_model    = bc_model
        self.decay_steps = decay_steps

    def get_lambda(self, current_step: int) -> float:
        import math
        return 0.5 * math.exp(-current_step / self.decay_steps)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ── 1. Load Dataset ───────────────────────────────────────────────
    log.info("Loading HDF5 dataset...")
    X, y = load_h5_arrays(args.h5)
    n_total, window_size, n_features = X.shape

    # Train/Test split: 80% train, 20% held-out out-of-sample
    split_idx = int(n_total * 0.8)

    # Giả lập close prices từ log returns (reverse log returns)
    # Thực tế: cần raw close từ CSV — đây là placeholder nếu chưa có
    close_prices  = np.ones(n_total) * 1900.0  # Sẽ thay bằng dữ liệu thật
    oracle_labels = y

    # ── 2. Load BC Model (frozen — làm anchor) ───────────────────────
    log.info(f"Loading BC checkpoint: {args.bc_ckpt}")
    ckpt = torch.load(args.bc_ckpt, map_location=device)
    bc_model = XAUTransformer(
        n_features  = n_features,
        window_size = window_size,
        d_model     = 256,
        n_heads     = 8,
        n_layers    = 6,
        n_actions   = 3,
    ).to(device)
    bc_model.load_state_dict(ckpt["model_state"])
    bc_model.eval()
    for p in bc_model.parameters():
        p.requires_grad = False  # Đóng băng BC model

    log.info(f"  BC checkpoint loaded — F1(Buy)={ckpt['f1_buy']:.3f}, "
             f"F1(Sell)={ckpt['f1_sell']:.3f}")

    # ── 3. Tạo Vectorized Environments ───────────────────────────────
    log.info(f"Creating {args.n_envs} vectorized environments...")
    # Mỗi env bắt đầu từ một offset khác nhau để đa dạng hóa experience
    env_fns = [
        make_env_fn(
            features      = X[:split_idx],
            close_prices  = close_prices[:split_idx],
            oracle_labels = oracle_labels[:split_idx],
            window_size   = window_size,
            start_offset  = (i * 500) % max(1, split_idx - window_size - 1000),
        )
        for i in range(args.n_envs)
    ]

    vec_env = SubprocVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)

    # ── 4. PPO Agent ──────────────────────────────────────────────────
    log.info("Initializing PPO Agent...")
    ppo = PPO(
        policy             = "MlpPolicy",  # Sẽ được override bởi BC weights
        env                = vec_env,
        n_steps            = 2048,
        batch_size         = 512,
        n_epochs           = 10,
        gamma              = 0.99,
        gae_lambda         = 0.95,
        clip_range         = 0.2,
        ent_coef           = 0.01,        # Khuyến khích exploration
        vf_coef            = 0.5,
        max_grad_norm      = 0.5,
        learning_rate      = 1e-4,        # Nhỏ hơn BC để không phá weights
        verbose            = 1,
        device             = device,
        tensorboard_log    = "logs/rl_training",
    )

    # ── 5. Training ───────────────────────────────────────────────────
    log.info(f"Starting PPO training for {args.steps:,} steps...")
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    ppo.learn(
        total_timesteps = args.steps,
        progress_bar    = True,
    )

    # Lưu PPO model
    ppo_path = CHECKPOINT_DIR / "ppo_xauusd"
    ppo.save(str(ppo_path))
    log.info(f"✅ PPO model saved: {ppo_path}")

    vec_env.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--h5",      required=True, help="Path tới HDF5 dataset")
    p.add_argument("--bc-ckpt", required=True, help="Path tới best_model_bc.pt",
                   dest="bc_ckpt")
    p.add_argument("--n-envs",  type=int, default=64,       dest="n_envs")
    p.add_argument("--steps",   type=int, default=2_000_000)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
```

- [ ] **Commit:**
```bash
git add src/training/train_rl.py
git commit -m "feat(sprint4): PPO training loop - vectorized envs, KL anchor, BC checkpoint loading"
```

---

## Task 2: Backtest & Reporting

**Files:**
- Create: `src/training/backtest.py`
- Create: `src/training/tests/test_backtest.py`

### Step 2.1: Test backtest metrics

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
        daily_returns = np.array([0.001] * 252)  # +0.1%/ngày
        metrics = compute_metrics(daily_returns)
        assert metrics["sharpe"] > 0, f"Sharpe phải dương: {metrics['sharpe']}"

    def test_all_zero_returns_sharpe_zero(self):
        """PnL = 0 mọi ngày → Sharpe = 0."""
        daily_returns = np.zeros(252)
        metrics = compute_metrics(daily_returns)
        assert metrics["sharpe"] == 0.0

    def test_max_drawdown_is_non_positive(self):
        """Max drawdown phải <= 0 (biểu diễn mất vốn)."""
        daily_returns = np.array([0.01, -0.05, 0.02, -0.03, 0.01])
        metrics = compute_metrics(daily_returns)
        assert metrics["max_drawdown"] <= 0

    def test_win_rate_between_0_and_1(self):
        """Win rate phải nằm trong [0, 1]."""
        returns = np.random.randn(100) * 0.01
        metrics = compute_metrics(returns)
        assert 0.0 <= metrics["win_rate"] <= 1.0

    def test_metrics_has_required_keys(self):
        """Kết quả phải có đủ các key bắt buộc."""
        returns = np.random.randn(252) * 0.001
        metrics = compute_metrics(returns)
        required = {"sharpe", "sortino", "max_drawdown", "win_rate",
                    "total_return", "n_trades"}
        assert required.issubset(set(metrics.keys()))
```

- [ ] **Chạy để verify FAIL:**
```bash
python -m pytest src/training/tests/test_backtest.py -v
```

- [ ] **Tạo `src/training/backtest.py`:**

```python
"""
backtest.py
-----------
Out-of-sample backtest: chạy model trained qua dữ liệu chưa thấy,
tính các chỉ số tài chính chuẩn.

Cách dùng:
  python src/training/backtest.py \\
      --h5      data/processed/XAUUSD_M15_w128.h5 \\
      --ckpt    checkpoints/ppo_xauusd.zip \\
      --mode    ppo                         # hoặc bc

Chỉ số output:
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
    Tính các chỉ số tài chính từ mảng daily returns.

    Parameters
    ----------
    daily_returns : np.ndarray — mảng % return mỗi bar/ngày

    Returns
    -------
    dict với keys: sharpe, sortino, max_drawdown, win_rate,
                   total_return, n_trades
    """
    r = np.array(daily_returns, dtype=np.float64)
    n = len(r)

    if n == 0:
        return dict(sharpe=0.0, sortino=0.0, max_drawdown=0.0,
                    win_rate=0.0, total_return=0.0, n_trades=0)

    mean_r = r.mean()
    std_r  = r.std()

    # Sharpe Ratio (annualized ×√252 nếu daily, ×√(252×24×4) nếu M15)
    sharpe = (mean_r / std_r * np.sqrt(252)) if std_r > 0 else 0.0

    # Sortino Ratio (chỉ dùng downside std)
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

    # N trades = số bar có return != 0
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
    log.info(f"  BACKTEST REPORT — {label}")
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
    status = "✅ PASS — Đủ tiêu chuẩn deploy" if passed else "❌ FAIL — Cần cải thiện thêm"
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

    # Trong thực tế: chạy model qua out-of-sample env và thu thập returns
    # Đây là scaffold — sẽ được hoàn thiện khi env + model đã sẵn sàng
    log.info("⚠️  Running simplified backtest scaffold (returns cần được thu thập từ env.step)")
    log.info("    Sử dụng compute_metrics() với array returns sau khi roll-out đầy đủ.")


if __name__ == "__main__":
    main()
```

- [ ] **Chạy để verify PASS:**
```bash
python -m pytest src/training/tests/test_backtest.py -v
```
Kết quả mong đợi: `5 passed`

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

### Step 3.1: Tạo Dockerfile

- [ ] **Tạo `Dockerfile`:**

```dockerfile
# XAUUSD AI Trading Bot — Training Container
# Tối ưu cho RTX 4090 / RTX 5090 trên Vast.ai

FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements và install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY data/ ./data/

# Tạo thư mục output
RUN mkdir -p checkpoints logs

# Default command: BC training
CMD ["python", "src/training/train_bc.py", \
     "--h5", "data/processed/XAUUSD_M15_w128.h5", \
     "--epochs", "100", \
     "--batch-size", "512"]
```

- [ ] **Tạo `scripts/vast_launch.sh`:**

```bash
#!/bin/bash
# vast_launch.sh — Triển khai training lên Vast.ai instance
# Cách dùng: ./scripts/vast_launch.sh <INSTANCE_ID> <SSH_KEY_PATH>
#
# Yêu cầu: vast CLI đã cài (pip install vastai)
# Trước khi chạy: vastai set api-key <YOUR_API_KEY>

set -e

INSTANCE_ID=${1:?"Usage: $0 <instance_id> <ssh_key>"}
SSH_KEY=${2:?"Usage: $0 <instance_id> <ssh_key>"}

echo "=== XAUUSD Bot — Vast.ai Deployment ==="
echo "Instance: $INSTANCE_ID"

# 1. Lấy connection info
CONN=$(vastai ssh-url $INSTANCE_ID)
SSH_HOST=$(echo $CONN | sed 's/ssh:\/\///')

# 2. Rsync source code lên instance
echo "Uploading source code..."
rsync -avz --exclude '.git' --exclude '__pycache__' \
    -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no" \
    ./ root@${SSH_HOST}:/workspace/

# 3. Install dependencies
echo "Installing dependencies..."
ssh -i $SSH_KEY -o StrictHostKeyChecking=no root@${SSH_HOST} \
    "cd /workspace && pip install -q -r requirements.txt"

# 4. Launch BC training (background với nohup)
echo "Starting BC training..."
ssh -i $SSH_KEY -o StrictHostKeyChecking=no root@${SSH_HOST} \
    "cd /workspace && nohup python src/training/train_bc.py \
        --h5 data/processed/XAUUSD_M15_w128.h5 \
        --epochs 100 \
        --batch-size 512 \
        > logs/train_bc.log 2>&1 &"

echo "✅ Training started! Theo dõi log:"
echo "   ssh -i $SSH_KEY root@${SSH_HOST} 'tail -f /workspace/logs/train_bc.log'"
```

- [ ] **Tạo `scripts/pull_results.sh`:**

```bash
#!/bin/bash
# pull_results.sh — Kéo kết quả training về local
# Cách dùng: ./scripts/pull_results.sh <INSTANCE_ID> <SSH_KEY_PATH>

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

echo "✅ Done! Checkpoints saved to ./checkpoints/"
ls -lh ./checkpoints/
```

- [ ] **Commit:**
```bash
git add Dockerfile scripts/
git commit -m "feat(sprint4): Dockerfile + Vast.ai launch/pull scripts"
```

---

## Task 4: Toàn bộ tests Sprint 4 & Final Push

- [ ] **Chạy toàn bộ test suite:**
```bash
python -m pytest src/ -v --tb=short
```
Kết quả mong đợi: `≥ 35 passed, 0 failed` (tất cả 4 sprints)

- [ ] **Final push:**
```bash
git push origin main
```

---

## Điều kiện DONE cho Sprint 4 (& toàn dự án v1)

| Chỉ số | Target |
|---|---|
| `pytest src/ -v` | ✅ 0 failures |
| F1(Buy) validation | > 0.40 |
| F1(Sell) validation | > 0.40 |
| Sharpe Ratio (out-of-sample) | > 1.0 |
| Max Drawdown | < 10% ($20) |
| Win Rate | > 55% |
| Docker build | ✅ không crash |
| Vast.ai training log | ✅ converging loss |
