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

def evaluate_oos(model, h5_path, split_idx, n_total, window_size, device, gap_bars=200):
    model.eval()
    oos_start = split_idx + gap_bars
    n_test    = n_total - oos_start  # [FIX NAME ERROR] Khai báo n_test tính từ tổng trừ điểm bắt đầu
    
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
        max_drawdown_usd = 999999.0, # [FIX OOS BACKTEST] Đánh giá toàn cảnh
        random_start     = False,    # [FIX GROUNDHOG OOS] Bắt nhịp từ điểm bắt đầu OOS
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
    survival_rate = len(equity_hist) / float(n_test)
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
    # 1. Load Pre-trained BC Model
    log.info(f"Loading BC checkpoint: {args.bc_ckpt}")
    
    # Nạp weights_only=False để tránh lỗi khác phiên bản numpy giữa Local và Cloud
    bc_model = XAUTransformer(n_features=n_features, window_size=window_size,
                              d_model=256, n_heads=8, n_layers=6).to(device)
    bc_model.load_state_dict(torch.load(args.bc_ckpt, map_location=device, weights_only=False)["model_state"])
    bc_model.eval()
    for p in bc_model.parameters():
        p.requires_grad = False
    log.info(f"BC loaded")

    ppo_model = XAUTransformer(n_features=n_features, window_size=window_size,
                               d_model=256, n_heads=8, n_layers=6).to(device)
    ppo_model.load_state_dict(torch.load(args.bc_ckpt, map_location=device, weights_only=False)["model_state"])
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