"""
train_rl_dual.py
-----------------
PPO Reinforcement Learning cho Dual-Timeframe Transformer (M5 + H1).

Cách dùng:
  python src/training/train_rl_dual.py \
    --h5 data/processed/XAUUSD_DUAL_M5w256_H1w64.h5 \
    --bc-ckpt checkpoints/best_model_dual_bc.pt \
    --n-envs 64 --total-steps 2000000
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.model.dual_transformer import DualTimeframeTransformer
from src.env.xauusd_dual_env import XAUUSDDualEnv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

CHECKPOINT_DIR = Path("checkpoints")


def make_envs(h5_path, n_envs, start_idx, end_idx, random_start=True):
    """Tạo n_envs song song."""
    envs = []
    for _ in range(n_envs):
        env = XAUUSDDualEnv(
            h5_path=h5_path,
            start_idx=start_idx,
            end_idx=end_idx,
            random_start=random_start,
        )
        envs.append(env)
    return envs


def collect_rollout(envs, model, device, n_steps=128):
    """Thu thập rollout từ nhiều env song song."""
    n_envs = len(envs)

    # Storage
    all_obs_m5 = []
    all_obs_h1 = []
    all_actions = []
    all_rewards = []
    all_dones = []
    all_log_probs = []
    all_values = []

    for step in range(n_steps):
        obs_m5 = np.stack([env._get_obs()["m5"] for env in envs])
        obs_h1 = np.stack([env._get_obs()["h1"] for env in envs])

        m5_t = torch.tensor(obs_m5, dtype=torch.float32, device=device)
        h1_t = torch.tensor(obs_h1, dtype=torch.float32, device=device)

        with torch.no_grad():
            logits, values = model(m5_t, h1_t)
            dist = torch.distributions.Categorical(logits=logits)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)

        actions_np = actions.cpu().numpy()
        values_np = values.squeeze(-1).cpu().numpy()
        log_probs_np = log_probs.cpu().numpy()

        all_obs_m5.append(obs_m5)
        all_obs_h1.append(obs_h1)
        all_actions.append(actions_np)
        all_log_probs.append(log_probs_np)
        all_values.append(values_np)

        rewards = np.zeros(n_envs)
        dones = np.zeros(n_envs, dtype=bool)

        for i, env in enumerate(envs):
            obs, rew, term, trunc, info = env.step(int(actions_np[i]))
            rewards[i] = rew
            dones[i] = term or trunc
            if dones[i]:
                env.reset()

        all_rewards.append(rewards)
        all_dones.append(dones)

    # Last values for GAE
    obs_m5 = np.stack([env._get_obs()["m5"] for env in envs])
    obs_h1 = np.stack([env._get_obs()["h1"] for env in envs])
    m5_t = torch.tensor(obs_m5, dtype=torch.float32, device=device)
    h1_t = torch.tensor(obs_h1, dtype=torch.float32, device=device)
    with torch.no_grad():
        _, last_values = model(m5_t, h1_t)
    last_values = last_values.squeeze(-1).cpu().numpy()

    return {
        "obs_m5": np.array(all_obs_m5),       # (n_steps, n_envs, 256, 13)
        "obs_h1": np.array(all_obs_h1),       # (n_steps, n_envs, 64, 13)
        "actions": np.array(all_actions),      # (n_steps, n_envs)
        "rewards": np.array(all_rewards),      # (n_steps, n_envs)
        "dones": np.array(all_dones),          # (n_steps, n_envs)
        "log_probs": np.array(all_log_probs),  # (n_steps, n_envs)
        "values": np.array(all_values),        # (n_steps, n_envs)
        "last_values": last_values,            # (n_envs,)
    }


def compute_gae(rewards, values, dones, last_values, gamma=0.99, lam=0.95):
    """Generalized Advantage Estimation."""
    n_steps, n_envs = rewards.shape
    advantages = np.zeros_like(rewards)
    last_gae = np.zeros(n_envs)

    for t in reversed(range(n_steps)):
        if t == n_steps - 1:
            next_values = last_values
        else:
            next_values = values[t + 1]
        next_nonterminal = 1.0 - dones[t].astype(np.float32)
        delta = rewards[t] + gamma * next_values * next_nonterminal - values[t]
        last_gae = delta + gamma * lam * next_nonterminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


def ppo_update(model, optimizer, rollout, device, epochs=4, clip_eps=0.2, vf_coef=0.5, ent_coef=0.01):
    """PPO clipped update."""
    advantages, returns = compute_gae(
        rollout["rewards"], rollout["values"],
        rollout["dones"], rollout["last_values"]
    )

    # Flatten
    n_steps, n_envs = rollout["actions"].shape
    flat_m5 = rollout["obs_m5"].reshape(-1, *rollout["obs_m5"].shape[2:])
    flat_h1 = rollout["obs_h1"].reshape(-1, *rollout["obs_h1"].shape[2:])
    flat_actions = rollout["actions"].reshape(-1)
    flat_old_lp = rollout["log_probs"].reshape(-1)
    flat_adv = advantages.reshape(-1)
    flat_ret = returns.reshape(-1)

    # Normalize advantages
    flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)

    total = len(flat_actions)
    batch_size = min(512, total)
    indices = np.arange(total)

    total_loss_sum = 0.0
    n_updates = 0

    for _ in range(epochs):
        np.random.shuffle(indices)
        for start in range(0, total, batch_size):
            end = start + batch_size
            idx = indices[start:end]

            m5_b = torch.tensor(flat_m5[idx], dtype=torch.float32, device=device)
            h1_b = torch.tensor(flat_h1[idx], dtype=torch.float32, device=device)
            act_b = torch.tensor(flat_actions[idx], dtype=torch.long, device=device)
            old_lp_b = torch.tensor(flat_old_lp[idx], dtype=torch.float32, device=device)
            adv_b = torch.tensor(flat_adv[idx], dtype=torch.float32, device=device)
            ret_b = torch.tensor(flat_ret[idx], dtype=torch.float32, device=device)

            logits, values = model(m5_b, h1_b)
            dist = torch.distributions.Categorical(logits=logits)
            new_lp = dist.log_prob(act_b)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_lp - old_lp_b)
            surr1 = ratio * adv_b
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_b
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = nn.functional.mse_loss(values.squeeze(-1), ret_b)

            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss_sum += loss.item()
            n_updates += 1

    return total_loss_sum / max(n_updates, 1)


def evaluate_model(model, h5_path, device, n_episodes=5):
    """Quick eval: chạy vài episodes, trả về avg reward và equity."""
    import h5py
    with h5py.File(h5_path, "r") as f:
        n_total = f["X_m5"].shape[0]
    oos_start = int(n_total * 0.8)

    total_reward = 0.0
    total_equity = 0.0

    for _ in range(n_episodes):
        env = XAUUSDDualEnv(h5_path=h5_path, start_idx=oos_start, end_idx=n_total, random_start=False)
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            m5_t = torch.tensor(obs["m5"], dtype=torch.float32, device=device).unsqueeze(0)
            h1_t = torch.tensor(obs["h1"], dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits, _ = model(m5_t, h1_t)
                action = logits.argmax(-1).item()
            obs, rew, term, trunc, info = env.step(action)
            ep_reward += rew
            done = term or trunc

        total_reward += ep_reward
        total_equity += info.get("equity", 200.0)

    return total_reward / n_episodes, total_equity / n_episodes


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    log.info(f"H5: {args.h5}")

    import h5py
    with h5py.File(args.h5, "r") as f:
        n_total = f["X_m5"].shape[0]
        n_feat_m5 = f["X_m5"].shape[2]
        n_feat_h1 = f["X_h1"].shape[2]
        window_m5 = f["X_m5"].shape[1]
        window_h1 = f["X_h1"].shape[1]

    train_end = int(n_total * 0.8)
    log.info(f"Train: 0-{train_end} | OOS: {train_end}-{n_total}")
    log.info(f"M5: {window_m5}×{n_feat_m5} | H1: {window_h1}×{n_feat_h1}")

    # Model
    model = DualTimeframeTransformer(
        n_features_m5=n_feat_m5, n_features_h1=n_feat_h1,
        window_m5=window_m5, window_h1=window_h1,
        d_model=args.d_model, n_heads=8,
        n_layers_m5=6, n_layers_h1=3,
    ).to(device)

    # Load BC weights if available
    if args.bc_ckpt and Path(args.bc_ckpt).exists():
        bc_state = torch.load(args.bc_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(bc_state, strict=False)
        log.info(f"✅ BC weights loaded: {args.bc_ckpt}")

    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model: {n_params:,} params")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

    # Create envs
    envs = make_envs(args.h5, args.n_envs, 0, train_end, random_start=True)
    for env in envs:
        env.reset()

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    n_steps_per_rollout = 128
    steps_per_update = n_steps_per_rollout * args.n_envs
    total_updates = args.total_steps // steps_per_update

    best_equity = 0.0
    log.info(f"Total updates: {total_updates} ({args.total_steps:,} steps)")

    for update in range(1, total_updates + 1):
        t0 = time.time()

        rollout = collect_rollout(envs, model, device, n_steps_per_rollout)
        avg_loss = ppo_update(model, optimizer, rollout, device)

        avg_reward = rollout["rewards"].mean()
        elapsed = time.time() - t0
        fps = steps_per_update / elapsed

        if update % 10 == 0:
            log.info(
                f"Update {update}/{total_updates} | "
                f"Loss={avg_loss:.4f} | AvgR={avg_reward:.4f} | "
                f"FPS={fps:.0f}"
            )

        if update % 30 == 0:
            avg_rew, avg_eq = evaluate_model(model, args.h5, device)
            log.info(f"   📊 Eval: AvgReward={avg_rew:.2f}, AvgEquity=${avg_eq:.2f}")

            if avg_eq > best_equity:
                best_equity = avg_eq
                torch.save(model.state_dict(), CHECKPOINT_DIR / "ppo_best_dual.pt")
                log.info(f"   ★ Best dual model saved (Equity=${best_equity:.2f})")

    # Save final
    torch.save(model.state_dict(), CHECKPOINT_DIR / "ppo_final_dual.pt")
    log.info(f"\n✅ Training complete! Best equity: ${best_equity:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", required=True)
    parser.add_argument("--bc-ckpt", default=None)
    parser.add_argument("--n-envs", type=int, default=64)
    parser.add_argument("--total-steps", type=int, default=2_000_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=256)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
