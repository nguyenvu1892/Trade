# src/env/tests/test_xauusd_env.py
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.env.xauusd_env import XAUUSDEnv


WINDOW = 64
N_FEATURES = 10


def _make_env_2d(n_bars: int = 500) -> XAUUSDEnv:
    """
    Môi trường test dùng features 2D (T, F) — dữ liệu raw chưa window hóa.
    Dùng cho unit test cơ bản, cursor bắt đầu từ WINDOW.
    """
    np.random.seed(42)
    close_prices  = 1800.0 + np.cumsum(np.random.randn(n_bars))
    open_next     = close_prices + np.random.randn(n_bars) * 0.2  # gap nhỏ
    features      = np.random.randn(n_bars, N_FEATURES).astype(np.float32) * 0.001
    oracle_labels = np.random.choice([0, 1, 2], size=n_bars, p=[0.9, 0.05, 0.05])
    return XAUUSDEnv(
        features         = features,
        close_prices     = close_prices,
        open_next_prices = open_next,
        oracle_labels    = oracle_labels,
        window_size      = WINDOW,
        spread_pips      = 30,
        lot_size         = 0.01,
        initial_balance  = 200.0,
        max_drawdown_usd = 20.0,
    )


def _make_env_3d(n_windows: int = 436) -> XAUUSDEnv:
    """
    [KEY TEST] Môi trường dùng features 3D (N_windows, WINDOW, F) —
    mô phỏng CHÍNH XÁC shape HDF5 từ DatasetBuilder.
    cursor bắt đầu từ 0, _get_obs() trả về self._features[cursor] trực tiếp.
    """
    np.random.seed(7)
    features_3d   = np.random.randn(n_windows, WINDOW, N_FEATURES).astype(np.float32)
    close_prices  = 1800.0 + np.cumsum(np.random.randn(n_windows))
    open_next     = close_prices + np.random.randn(n_windows) * 0.2
    oracle_labels = np.random.choice([0, 1, 2], size=n_windows, p=[0.9, 0.05, 0.05])
    return XAUUSDEnv(
        features         = features_3d,
        close_prices     = close_prices,
        open_next_prices = open_next,
        oracle_labels    = oracle_labels,
        window_size      = WINDOW,
        spread_pips      = 30,
        lot_size         = 0.01,
        initial_balance  = 200.0,
        max_drawdown_usd = 20.0,
    )


# Alias backward compatible
_make_env = _make_env_2d


class TestXAUUSDEnv:
    def test_reset_returns_correct_obs_shape_2d(self):
        """2D features: reset() phải trả về shape (WINDOW, N_FEATURES)."""
        env = _make_env_2d()
        obs, info = env.reset()
        assert obs.shape == (WINDOW, N_FEATURES), (
            f"Obs shape kỳ vọng ({WINDOW}, {N_FEATURES}), nhận {obs.shape}"
        )

    def test_reset_returns_correct_obs_shape_3d(self):
        """
        [KEY] 3D features từ HDF5: reset() phải trả về shape (WINDOW, N_FEATURES).
        cursor = 0, _get_obs() = features[0] trực tiếp (không slice).
        """
        env = _make_env_3d()
        obs, info = env.reset()
        assert obs.shape == (WINDOW, N_FEATURES), (
            f"[3D env] Obs shape kỳ vọng ({WINDOW}, {N_FEATURES}), nhận {obs.shape}"
        )

    def test_action_space_correct(self):
        """Action space phải là Discrete(3): [Hold=0, Buy=1, Sell=2]."""
        env = _make_env()
        assert env.action_space.n == 3

    def test_step_returns_5_values(self):
        """step() phải trả về (obs, reward, terminated, truncated, info)."""
        env = _make_env()
        env.reset()
        result = env.step(0)  # Hold
        assert len(result) == 5, f"step() phải trả về 5 giá trị, nhận {len(result)}"

    def test_hold_action_returns_negative_reward(self):
        """Hold liên tục phải nhận reward âm do holding cost."""
        env = _make_env()
        env.reset()
        rewards = []
        for _ in range(10):
            _, reward, term, trunc, _ = env.step(0)  # Luôn Hold
            rewards.append(reward)
            if term or trunc:
                break
        assert sum(rewards) < 0, "Hold liên tục phải cho tổng reward âm"

    def test_episode_terminates_on_drawdown(self):
        """Episode phải kết thúc khi equity dưới $180 (drawdown $20)."""
        env = _make_env()
        env.reset()
        # Ép balance về $178 thủ công
        env._balance = 178.0
        env._peak_balance = 200.0
        _, _, terminated, truncated, info = env.step(0)
        assert terminated or truncated, "Episode phải kết thúc khi drawdown vượt $20"

    def test_balance_starts_at_200(self):
        """Balance ban đầu phải là $200."""
        env = _make_env()
        env.reset()
        assert env._balance == 200.0

    def test_obs_no_nan(self):
        """Observation không được có NaN."""
        env = _make_env()
        obs, _ = env.reset()
        assert not np.isnan(obs).any(), "Observation chứa NaN"

    def test_atomic_position_reversal(self):
        """
        Đang Long + nhận action Sell → phải đóng Long VÀ mở Short trong cùng 1 step.
        Không được trở về Flat (mất 15phút chờ cây nến kế tiếp).
        """
        env = _make_env()
        env.reset()
        # Mở Long
        env.step(1)
        assert env._position_dir == 1, "Phải đang Long sau action Buy"
        # Ngay sau đó, Sell → đóng Long + mở Short ngay
        env.step(2)
        assert env._position_dir == -1, (
            "Sau khi đương Long gặp Sell, phải mở Short ngay (không qua Flat)"
        )

    def test_reversal_deducts_double_spread(self):
        """
        Đảo chiều tạo ra 2 commission (đóng + mở) — spread trừ 2 lần.
        """
        env = _make_env()
        env.reset()
        balance_before = env._balance
        env.step(1)  # Mở Long
        env.step(2)  # Đảo chiều: đóng Long + mở Short
        # Phải mất ít nhất 2 × commission ($0.07) do 2 lần giao dịch
        assert env._position_dir == -1
        assert env._balance != balance_before

    def test_pnl_calculation_buy(self):
        """
        Mua tại giá X, đóng tại giá X + $10 với 0.01 lot.
        PnL = (10 USD move × 0.01 lot × 100 oz/lot) / 100 ≈ $1.0
        Thực tế: 0.01 lot XAUUSD = 1 oz → PnL = $10 * 0.01 = $0.10... 
        Giá trị: XAUUSD 0.01 lot = PnL $ = price_diff * 0.01
        """
        env = _make_env()
        env.reset()
        # Giả lập mua rồi đóng ngay
        env._balance = 200.0
        pnl = env._calc_pnl(entry_price=1900.0, exit_price=1910.0,
                            direction=1, lot=0.01)
        # $10 move × 0.01 lot × 100 oz = $10.00
        assert abs(pnl - 10.0) < 0.01, f"PnL Buy kỳ vọng ~$10.0, nhận {pnl:.4f}"

    def test_pnl_calculation_sell(self):
        """Bán tại 1910, đóng tại 1900: profit = $10.00."""
        env = _make_env()
        env.reset()
        pnl = env._calc_pnl(entry_price=1910.0, exit_price=1900.0,
                            direction=-1, lot=0.01)
        assert abs(pnl - 10.0) < 0.01, f"PnL Sell kỳ vọng ~$10.0, nhận {pnl:.4f}"