# XAUUSD Bot — Sprint 2: Market Simulator (Gymnasium Environment)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Xây dựng môi trường giao dịch mô phỏng XAUUSD chuẩn xác như Exness thật — bao gồm Spread, Commission, Slippage — tuân thủ Gymnasium API để PPO Agent có thể tương tác trực tiếp.

**Architecture:** `XAUUSDEnv` kế thừa `gymnasium.Env`. State là Feature Tensor cửa sổ W nến cuối. Action space là Discrete(3): [Hold, Buy, Sell]. Reward trả về ngay khi đóng lệnh. Episode kết thúc khi hết data hoặc drawdown vượt $20.

**Tech Stack:** Python 3.10+, gymnasium, numpy, pandas, h5py, pytest

---

## File Structure

```
src/env/
├── xauusd_env.py             [NEW] — Gymnasium environment chính
├── reward.py                 [NEW] — Hàm reward tách biệt để test độc lập
└── tests/
    ├── test_xauusd_env.py    [NEW]
    └── test_reward.py        [NEW]
```

---

## Task 1: Reward Function

**Files:**
- Create: `src/env/reward.py`
- Create: `src/env/tests/test_reward.py`

### Step 1.1: Viết failing tests cho Reward

- [ ] **Viết test file:**

```python
# src/env/tests/test_reward.py
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.env.reward import RewardCalculator


class TestRewardCalculator:
    def setup_method(self):
        self.calc = RewardCalculator(
            initial_balance    = 200.0,
            max_drawdown_usd   = 20.0,
            holding_cost_per_bar = 0.001,
        )

    def test_commission_deducted_on_open(self):
        """$0.07 commission phải bị trừ ngay khi mở lệnh."""
        r = self.calc.on_open_commission()
        assert r == -0.07, f"Commission phải là -$0.07, nhận: {r}"

    def test_swap_deducted_at_midnight(self):
        """Swap -$0.05 phải bị trừ khi giử lệnh qua 00:00 server."""
        r = self.calc.on_midnight_swap(is_friday=False)
        assert r == -0.05, f"Swap phải là -$0.05, nhận: {r}"

    def test_swap_triple_on_friday(self):
        """Thứ 6 tính swap x3 (bù cho cả weekend)."""
        r = self.calc.on_midnight_swap(is_friday=True)
        assert r == -0.15, f"Swap thứ 6 phải là -$0.15, nhận: {r}"

    def test_winning_trade_positive_reward(self):
        """Lệnh thắng → reward dương."""
        r = self.calc.on_close(pnl=2.5, peak_balance=200.0, current_balance=202.5)
        assert r > 0, f"Lệnh thắng phải có reward dương, nhận: {r}"

    def test_losing_trade_negative_reward(self):
        """Lệnh thua → reward âm."""
        r = self.calc.on_close(pnl=-1.0, peak_balance=200.0, current_balance=199.0)
        assert r < 0, f"Lệnh thua phải có reward âm, nhận: {r}"

    def test_holding_cost_only_when_flat(self):
        """Phạt đứng im CHỈ khi Flat (không có vị thế). Khi đang giữ lệnh = 0."""
        r_flat    = self.calc.on_hold(consecutive_hold=5, has_position=False,
                                      oracle_action=0)
        r_holding = self.calc.on_hold(consecutive_hold=5, has_position=True,
                                      oracle_action=0)
        assert r_flat < 0, "Phải phạt khi Flat"
        assert r_holding == 0.0, (
            f"Không được phạt khi đang giữ lệnh có lời, nhận: {r_holding}"
        )

    def test_holding_cost_accumulates(self):
        """Phạt Flat cộng dồn theo số nến."""
        r0 = self.calc.on_hold(consecutive_hold=1, has_position=False, oracle_action=0)
        r5 = self.calc.on_hold(consecutive_hold=5, has_position=False, oracle_action=0)
        assert r5 < r0, "Phạt Flat phải tăng theo thời gian"

    def test_drawdown_penalty_triggers(self):
        """Phạt nặng khi drawdown vượt $20."""
        r = self.calc.on_close(pnl=-22.0, peak_balance=200.0, current_balance=178.0)
        assert r < -10, f"Drawdown penalty phải rất nặng, nhận: {r}"

    def test_no_penalty_within_drawdown_limit(self):
        """Không có drawdown penalty khi balance vẫn trên ngưỡng."""
        r = self.calc.on_close(pnl=-5.0, peak_balance=200.0, current_balance=195.0)
        assert r > -10, f"Không nên có penalty nặng khi trong giới hạn: {r}"

    def test_opportunity_cost_when_oracle_says_trade(self):
        """Phạt cơ hội khi Oracle muốn vào lệnh mà Bot Hold."""
        r_no_opp  = self.calc.on_hold(consecutive_hold=1, oracle_action=0)   # Oracle cũng Hold
        r_with_opp = self.calc.on_hold(consecutive_hold=1, oracle_action=1)  # Oracle muốn Buy
        assert r_with_opp < r_no_opp, "Phạt cơ hội phải khiến reward thấp hơn"
```

- [ ] **Chạy để verify FAIL:**
```bash
python -m pytest src/env/tests/test_reward.py -v
```
Kết quả mong đợi: `ERROR — ModuleNotFoundError`

### Step 1.2: Implement RewardCalculator

- [ ] **Tạo `src/env/__init__.py` rỗng:**
```bash
type nul > src\env\__init__.py
type nul > src\env\tests\__init__.py
```

- [ ] **Tạo `src/env/reward.py`:**

```python
"""
reward.py
---------
Hàm reward được tách riêng để test độc lập.
Gồm 4 thành phần:
  1. PnL thực khi đóng lệnh
  2. Opportunity Cost (phạt bỏ lỡ cơ hội Oracle)
  3. Holding Cost (phạt đứng im cộng dồn)
  4. Drawdown Penalty (phạt khi equity < peak - max_dd)
"""


class RewardCalculator:
    def __init__(
        self,
        initial_balance:      float = 200.0,
        max_drawdown_usd:     float = 20.0,
        holding_cost_per_bar: float = 0.001,   # Chỉ áp khi FLAT
        opportunity_cost_usd: float = 0.5,
        commission_usd:       float = 0.07,    # [NEW] Exness Raw $7/lot → $0.07 cho 0.01 lot
        swap_per_night:       float = 0.05,    # [NEW] Swap âm ≈ -$0.05/đêm cho 0.01 lot Long
    ):
        self.initial_balance      = initial_balance
        self.max_drawdown_usd     = max_drawdown_usd
        self.holding_cost_per_bar = holding_cost_per_bar
        self.opportunity_cost_usd = opportunity_cost_usd
        self.commission_usd       = commission_usd
        self.swap_per_night       = swap_per_night

    def on_open_commission(self) -> float:
        """[NEW] Trừ commission cố định khi mở lệnh."""
        return -self.commission_usd

    def on_midnight_swap(self, is_friday: bool = False) -> float:
        """[NEW] Trừ swap khi lệnh giữ qua 00:00 server time.
        Thứ 6 tính x3 (bù cho 2 ngày weekend không có swap rênh).
        """
        multiplier = 3 if is_friday else 1
        return -(self.swap_per_night * multiplier)

    def on_close(self, pnl: float, peak_balance: float, current_balance: float) -> float:
        """Reward khi đóng lệnh. Commission đã bị trừ khi mở lệnh."""
        reward = pnl
        drawdown = peak_balance - current_balance
        if drawdown > self.max_drawdown_usd:
            excess = drawdown - self.max_drawdown_usd
            reward -= excess * 2.0
        return reward

    def on_hold(
        self,
        consecutive_hold: int,
        has_position:     bool = False,   # [FIX] chỉ phạt khi Flat
        oracle_action:    int  = 0,
    ) -> float:
        """Reward mỗi timestep.
        - Flat + đứng im: phạt (bot không chịu tham gia thị trường)
        - Có vị thế + kiên nhẫn giữ: không phạt (thị trường cần thời gian chạy tới TP)
        """
        if has_position:
            return 0.0  # [FIX] Không phạt patience khi giữ lệnh

        reward = -self.holding_cost_per_bar * consecutive_hold
        if oracle_action in (1, 2):
            reward -= self.opportunity_cost_usd
        return reward
```

- [ ] **Chạy để verify PASS:**
```bash
python -m pytest src/env/tests/test_reward.py -v
```
Kết quả mong đợi: `6 passed`

- [ ] **Commit:**
```bash
git add src/env/ 
git commit -m "feat(sprint2): RewardCalculator - PnL, holding cost, drawdown penalty, opportunity cost"
```

---

## Task 2: XAUUSDEnv — Gymnasium Environment

**Files:**
- Create: `src/env/xauusd_env.py`
- Create: `src/env/tests/test_xauusd_env.py`

### Step 2.1: Viết failing tests

- [ ] **Viết test file:**

```python
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


def _make_env(n_bars: int = 500) -> XAUUSDEnv:
    """Tạo môi trường test với dữ liệu giả."""
    np.random.seed(42)
    idx = pd.date_range("2020-01-01", periods=n_bars, freq="15min", tz="UTC")

    # Feature matrix (normalized)
    features = np.random.randn(n_bars, N_FEATURES).astype(np.float32) * 0.001

    # Giá close thực (để tính PnL)
    close_prices = 1800.0 + np.cumsum(np.random.randn(n_bars))

    # Oracle labels
    oracle_labels = np.random.choice([0, 1, 2], size=n_bars, p=[0.9, 0.05, 0.05])

    return XAUUSDEnv(
        features      = features,
        close_prices  = close_prices,
        oracle_labels = oracle_labels,
        window_size   = WINDOW,
        spread_pips   = 30,       # 30 pips spread ≈ $0.30 với 0.01 lot
        lot_size      = 0.01,
        initial_balance = 200.0,
        max_drawdown_usd = 20.0,
    )


class TestXAUUSDEnv:
    def test_reset_returns_correct_obs_shape(self):
        """reset() phải trả về observation shape đúng (window, n_features)."""
        env = _make_env()
        obs, info = env.reset()
        assert obs.shape == (WINDOW, N_FEATURES), (
            f"Obs shape kỳ vọng ({WINDOW}, {N_FEATURES}), nhận {obs.shape}"
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
        # $10 move × 0.01 lot = $0.10
        assert abs(pnl - 0.10) < 0.01, f"PnL Buy kỳ vọng ~$0.10, nhận {pnl:.4f}"

    def test_pnl_calculation_sell(self):
        """Bán tại 1910, đóng tại 1900: profit = $0.10."""
        env = _make_env()
        env.reset()
        pnl = env._calc_pnl(entry_price=1910.0, exit_price=1900.0,
                            direction=-1, lot=0.01)
        assert abs(pnl - 0.10) < 0.01, f"PnL Sell kỳ vọng ~$0.10, nhận {pnl:.4f}"
```

- [ ] **Chạy để verify FAIL:**
```bash
python -m pytest src/env/tests/test_xauusd_env.py -v
```
Kết quả mong đợi: `ERROR — ModuleNotFoundError`

### Step 2.2: Implement XAUUSDEnv

- [ ] **Tạo `src/env/xauusd_env.py`:**

```python
"""
xauusd_env.py
-------------
Môi trường giao dịch XAUUSD chuẩn Gymnasium.

State  : Feature window (window_size × n_features) float32
Action : Discrete(3) — 0=Hold, 1=Buy, 2=Sell
Reward : RewardCalculator (PnL + holding_cost + drawdown_penalty + opportunity_cost)

Constraints (Exness Raw, $200 vốn, 0.01 lot cố định):
  - Chỉ 1 vị thế tại 1 thời điểm
  - Lot cố định 0.01
  - Spread tính như Exness Raw (~20-30 pips)
  - Max drawdown $20 → episode kết thúc sớm
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.env.reward import RewardCalculator


class XAUUSDEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        features:         np.ndarray,   # shape (T, n_features) float32
        close_prices:     np.ndarray,   # shape (T,)  — giá Close tuyệt đối USD
        oracle_labels:    np.ndarray,   # shape (T,)  — nhãn Oracle [0,1,2]
        window_size:      int   = 128,
        spread_pips:      int   = 25,   # 25 = $0.25 cho 0.01 lot
        lot_size:         float = 0.01,
        initial_balance:  float = 200.0,
        max_drawdown_usd: float = 20.0,
    ):
        super().__init__()
        self._features      = features
        self._close         = close_prices
        self._oracle        = oracle_labels
        self._window        = window_size
        self._spread_usd    = spread_pips * lot_size / 100.0  # đơn giản hóa
        self._lot           = lot_size
        self._init_balance  = initial_balance
        self._max_dd        = max_drawdown_usd

        n_features = features.shape[1]

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(window_size, n_features),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)  # 0=Hold, 1=Buy, 2=Sell

        self._reward_calc = RewardCalculator(
            initial_balance      = initial_balance,
            max_drawdown_usd     = max_drawdown_usd,
            holding_cost_per_bar = 0.001,
            opportunity_cost_usd = 0.5,
        )

        # State variables (khởi tạo trong reset())
        self._cursor:           int   = window_size
        self._balance:          float = initial_balance
        self._peak_balance:     float = initial_balance
        self._position_dir:     int   = 0     # 0=flat, 1=long, -1=short
        self._entry_price:      float = 0.0
        self._consecutive_hold: int   = 0

    # ── Gymnasium API ──────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._cursor           = self._window
        self._balance          = self._init_balance
        self._peak_balance     = self._init_balance
        self._position_dir     = 0
        self._entry_price      = 0.0
        self._consecutive_hold = 0
        return self._get_obs(), {}

    def step(self, action: int):
        oracle_action = int(self._oracle[self._cursor])
        current_price = float(self._close[self._cursor])
        reward        = 0.0
        terminated    = False

        # ── Xử lý Action ──────────────────────────────────────────────
        if self._position_dir == 0:
            # Flat: có thể mở lệnh mới
            if action == 1:   # Buy
                self._position_dir = 1
                self._entry_price  = current_price + self._spread_usd
                self._consecutive_hold = 0
            elif action == 2:  # Sell
                self._position_dir = -1
                self._entry_price  = current_price - self._spread_usd
                self._consecutive_hold = 0
            else:  # Hold
                self._consecutive_hold += 1
                reward = self._reward_calc.on_hold(self._consecutive_hold, oracle_action)
        else:
            # Có vị thế → đóng khi action = ngược chiều hoặc Hold → tiếp tục giữ
            if (action == 0) or (action == self._position_dir + 1):
                # Đóng lệnh
                pnl = self._calc_pnl(self._entry_price, current_price,
                                     self._position_dir, self._lot)
                self._balance         += pnl
                self._peak_balance     = max(self._peak_balance, self._balance)
                reward                 = self._reward_calc.on_close(
                    pnl, self._consecutive_hold,
                    self._peak_balance, self._balance
                )
                self._position_dir     = 0
                self._entry_price      = 0.0
                self._consecutive_hold = 0
            else:
                # Giữ lệnh tiếp
                self._consecutive_hold += 1
                reward = self._reward_calc.on_hold(self._consecutive_hold, oracle_action)

        # ── Kiểm tra điều kiện kết thúc ───────────────────────────────
        drawdown = self._peak_balance - self._balance
        if drawdown >= self._max_dd:
            terminated = True  # Drawdown vượt $20 → game over

        self._cursor += 1
        truncated = self._cursor >= len(self._close)

        return self._get_obs(), reward, terminated, truncated, {
            "balance":  self._balance,
            "drawdown": drawdown,
        }

    # ── Helpers ────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        start = self._cursor - self._window
        end   = self._cursor
        return self._features[start:end].copy()

    def _calc_pnl(
        self,
        entry_price: float,
        exit_price:  float,
        direction:   int,    # 1=long, -1=short
        lot:         float,
    ) -> float:
        """
        XAUUSD 0.01 lot = 0.01 oz vàng.
        PnL = price_diff_USD × lot_size
        Ví dụ: entry 1900, exit 1910, long → $10 × 0.01 = $0.10
        """
        return (exit_price - entry_price) * direction * lot
```

- [ ] **Chạy để verify PASS:**
```bash
python -m pytest src/env/tests/test_xauusd_env.py -v
```
Kết quả mong đợi: `9 passed`

- [ ] **Commit:**
```bash
git add src/env/
git commit -m "feat(sprint2): XAUUSDEnv - Gymnasium env, spread, PnL, drawdown termination"
```

---

## Task 3: Chạy toàn bộ tests Sprint 2 & Push

- [ ] **Chạy toàn bộ test suite:**
```bash
python -m pytest src/env/tests/ -v
```
Kết quả mong đợi: `15 passed, 0 failed`

- [ ] **Push:**
```bash
git push origin main
```

## Điều kiện DONE cho Sprint 2
- [ ] `python -m pytest src/env/tests/ -v` → tất cả PASS
- [ ] `env.step()` trả về đúng 5 giá trị theo chuẩn Gymnasium
- [ ] PnL formula đúng với 0.01 lot XAUUSD
- [ ] Episode terminate khi drawdown vượt $20
