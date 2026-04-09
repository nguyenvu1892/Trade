"""
xauusd_dual_env.py
-------------------
Dual-Timeframe Environment: M5 + H1 observation space.

State: Dict observation:
  "m5" : (256, 13)  — M5 features window
  "h1" : (64,  13)  — H1 features window (aligned)

Action: Discrete(3) — 0=Hold, 1=Buy, 2=Sell
Kế thừa logic PnL/Reward từ XAUUSDEnv gốc.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import h5py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.env.reward import RewardCalculator


class XAUUSDDualEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        h5_path:          str,
        start_idx:        int   = 0,
        end_idx:          int   = -1,
        window_m5:        int   = 256,
        window_h1:        int   = 64,
        spread_pips:      int   = 25,
        lot_size:         float = 0.01,
        initial_balance:  float = 200.0,
        max_drawdown_usd: float = 20.0,
        random_start:     bool  = True,
    ):
        super().__init__()

        self._window_m5 = window_m5
        self._window_h1 = window_h1
        self._random_start = random_start
        self._start_idx = start_idx

        # Load dual data from HDF5
        with h5py.File(h5_path, "r") as f:
            total_len = f["X_m5"].shape[0] if end_idx == -1 else end_idx
            self._feat_m5   = f["X_m5"][start_idx:total_len].astype(np.float32)
            self._feat_h1   = f["X_h1"][start_idx:total_len].astype(np.float32)
            self._close     = f["close"][start_idx:total_len].astype(np.float32)
            self._oracle    = f["y"][start_idx:total_len]

            n_feat_m5 = f["X_m5"].shape[2]
            n_feat_h1 = f["X_h1"].shape[2]

        self._n_samples = len(self._close)
        self._spread_price_shift = spread_pips * lot_size / 100.0
        self._lot = lot_size
        self._init_balance = initial_balance
        self._max_dd = max_drawdown_usd

        # Dual observation space
        self.observation_space = spaces.Dict({
            "m5": spaces.Box(-np.inf, np.inf, shape=(window_m5, n_feat_m5), dtype=np.float32),
            "h1": spaces.Box(-np.inf, np.inf, shape=(window_h1, n_feat_h1), dtype=np.float32),
        })
        self.action_space = spaces.Discrete(3)

        self._reward_calc = RewardCalculator(
            initial_balance=initial_balance,
            max_drawdown_usd=max_drawdown_usd,
        )

        # State
        self._cursor = 0
        self._balance = initial_balance
        self._peak_balance = initial_balance
        self._position_dir = 0
        self._entry_price = 0.0
        self._consecutive_hold = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self._random_start:
            max_s = self._n_samples - 2000
            self._cursor = np.random.randint(0, max(1, max_s))
        else:
            self._cursor = 0
        self._balance = self._init_balance
        self._peak_balance = self._init_balance
        self._position_dir = 0
        self._entry_price = 0.0
        self._consecutive_hold = 0
        return self._get_obs(), {}

    def step(self, action: int):
        close_price = float(self._close[self._cursor])
        oracle_action = int(self._oracle[self._cursor])
        reward = 0.0
        terminated = False

        # Use close as proxy for entry (pre-windowed data doesn't have open_next)
        entry_price = close_price

        if self._position_dir == 0:
            if action == 1:  # Buy
                commission = self._reward_calc.on_open_commission()
                self._balance += commission
                self._position_dir = 1
                self._entry_price = entry_price + self._spread_price_shift
                self._consecutive_hold = 0
                reward = commission
            elif action == 2:  # Sell
                commission = self._reward_calc.on_open_commission()
                self._balance += commission
                self._position_dir = -1
                self._entry_price = entry_price - self._spread_price_shift
                self._consecutive_hold = 0
                reward = commission
            else:
                self._consecutive_hold += 1
                reward = self._reward_calc.on_hold(
                    self._consecutive_hold, has_position=False,
                    oracle_action=oracle_action
                )
        else:
            is_reversal = (
                (self._position_dir == 1 and action == 2) or
                (self._position_dir == -1 and action == 1)
            )
            is_close = (action == 0)

            if is_close or is_reversal:
                pnl = self._calc_pnl(self._entry_price, close_price,
                                     self._position_dir, self._lot)
                self._balance += pnl
                self._peak_balance = max(self._peak_balance, self._balance)
                reward = self._reward_calc.on_close(
                    pnl, self._peak_balance, self._balance
                )
                self._position_dir = 0
                self._consecutive_hold = 0

                if is_reversal:
                    commission = self._reward_calc.on_open_commission()
                    self._balance += commission
                    reward += commission
                    new_dir = 1 if action == 1 else -1
                    spread_adj = self._spread_price_shift if new_dir == 1 else -self._spread_price_shift
                    self._position_dir = new_dir
                    self._entry_price = close_price + spread_adj
            else:
                self._consecutive_hold += 1
                reward = self._reward_calc.on_hold(
                    self._consecutive_hold, has_position=True,
                    oracle_action=oracle_action
                )

        # Equity
        unrealized = 0.0
        if self._position_dir != 0:
            unrealized = self._calc_pnl(self._entry_price, close_price,
                                        self._position_dir, self._lot)
        equity = self._balance + unrealized

        self._cursor += 1

        if self._cursor >= self._n_samples - 1:
            truncated = True
        else:
            truncated = False

        if equity <= self._init_balance - self._max_dd:
            terminated = True
            reward -= 5.0

        return self._get_obs(), float(reward), terminated, truncated, {
            "balance": self._balance,
            "equity": equity,
            "drawdown": self._peak_balance - equity,
        }

    def _get_obs(self):
        return {
            "m5": self._feat_m5[self._cursor].copy(),
            "h1": self._feat_h1[self._cursor].copy(),
        }

    def _calc_pnl(self, entry, exit, direction, lot):
        return (exit - entry) * direction * lot * 100.0
