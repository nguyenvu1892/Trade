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
        """Phạt đứng im CHỈ khi Flat. Khi đang giữ lệnh = 0."""
        r_flat    = self.calc.on_hold(consecutive_hold=5, has_position=False,
                                      oracle_action=0)
        r_holding = self.calc.on_hold(consecutive_hold=5, has_position=True,
                                      oracle_action=0)
        assert r_flat < 0, "Phải phạt khi Flat"
        assert r_holding == 0.0, (
            f"Không được phạt khi đang giữ lệnh có lời, nhận: {r_holding}"
        )

    def test_holding_cost_flat_rate_not_accumulating(self):
        """
        [FIX] Phạt phải là FLAT RATE, không cộng dồn theo thời gian.
        Lý do: Nếu cộng dồn (-0.001 × 50 = -0.05/nến), bot sẽ 'lách
        luật' bằng cách mở + đóng ngay lệnh chỉ để reset counter → Overtrading.
        Phạt flat rate thì không tạo 'mortgage effect' hấp dẫn hơn commission.
        """
        r_bar1  = self.calc.on_hold(consecutive_hold=1,  has_position=False, oracle_action=0)
        r_bar50 = self.calc.on_hold(consecutive_hold=50, has_position=False, oracle_action=0)
        assert r_bar1 == r_bar50, (
            f"Phạt phải cố định: bar1={r_bar1}, bar50={r_bar50}. Không cộng dồn!"
        )

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
        has_position:     bool = False,
        oracle_action:    int  = 0,
    ) -> float:
        """Reward mỗi timestep.
        - Flat + đứng im: phạt CỐ ĐỊNH (không cộng dồn — tránh Overtrading)
        - Có vị thế + kiên nhẫn giữ: không phạt
        """
        if has_position:
            return 0.0

        # [FIX] Flat rate — không nhân với consecutive_hold
        # Nếu cộng dồn: bot mở + đóng lệnh ngay lập tức để reset counter → Overtrading
        reward = -self.holding_cost_per_bar
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
        assert env._balance <= balance_before - 0.10, (
            f"Phải trừ phí đảo chiều, balance: {env._balance:.4f}"
        )

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
         Lấy từ features 2D shape (T, n_features) bằng cách slice:
         features[cursor - window : cursor]

Action : Discrete(3) — 0=Hold, 1=Buy, 2=Sell
Reward : RewardCalculator (PnL + holding_cost + drawdown_penalty)

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
        h5_path:           str = None,           # [FIX OOM] Đường dẫn nạp dữ liệu lazy
        start_idx:         int = 0,               # Index bắt đầu (danh cho rollout)
        end_idx:           int = -1,              # Index kết thúc
        features:          np.ndarray = None,   # 2D (T, F) (Dùng cho Unit Test)
        close_prices:      np.ndarray = None,   
        open_next_prices:  np.ndarray = None,   
        oracle_labels:     np.ndarray = None,   
        window_size:       int   = 128,
        spread_pips:       int   = 25,
        lot_size:          float = 0.01,
        initial_balance:   float = 200.0,
        max_drawdown_usd:  float = 20.0,
    ):
        super().__init__()
        
        self.h5_path       = h5_path
        self._start_idx    = start_idx
        self._end_idx      = end_idx

        import h5py
        if h5_path is not None:
            self._is_prewindowed = True
            # [FIX DISK I/O] Đọc toàn bộ chunk của worker này vào RAM (Khoảng 15-20MB/worker)
            # Tránh mở/đóng file HDF5 131,000 lần mỗi epoch!
            with h5py.File(h5_path, "r") as f:
                n_features = f["X"].shape[2]
                total_len  = f["X"].shape[0] if end_idx == -1 else end_idx
                self._features  = f["X"][start_idx:total_len].astype(np.float32)
                self._close     = f["close"][start_idx:total_len].astype(np.float32)
                self._open_next = f["open_next"][start_idx:total_len].astype(np.float32)
                self._oracle    = f["y"][start_idx:total_len]
        else:
            # Tham số fallback dành cho Unit Test 2D mode
            assert features is not None and features.ndim == 2
            self._is_prewindowed = False
            self._features       = features
            self._close          = close_prices
            self._open_next      = open_next_prices       
            self._oracle         = oracle_labels
            n_features           = features.shape[1]
        self._window         = window_size
        self._spread_usd     = spread_pips * lot_size / 100.0
        self._lot            = lot_size
        self._init_balance   = initial_balance
        self._max_dd         = max_drawdown_usd

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

        # State variables — reset trong reset()
        self._cursor:           int   = 0
        self._balance:          float = initial_balance
        self._peak_balance:     float = initial_balance
        self._position_dir:     int   = 0      # 0=flat, 1=long, -1=short
        self._entry_price:      float = 0.0
        self._consecutive_hold: int   = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # [FIX] 3D pre-windowed: cursor bắt đầu từ 0 (cửa sổ đầu tiên)
        # 2D raw:             cursor bắt đầu từ window_size (đủ dữ liệu lịch sử)
        self._cursor           = 0 if self._is_prewindowed else self._window
        self._balance          = self._init_balance
        self._peak_balance     = self._init_balance
        self._position_dir     = 0
        self._entry_price      = 0.0
        self._consecutive_hold = 0
        return self._get_obs(), {}

    def step(self, action: int):
        # [FIX LOOKAHEAD] Dùng giá Close nến hiện tại chỉ để tham khảo.
        # Lệnh thực tế khớp tại giá OPEN của nến KẾ TIẾP (open_next).
        current_close = float(self._close[self._cursor])       # dùng để đóng lệnh
        entry_price   = float(self._open_next[self._cursor])   # [FIX] khớp tại open+1
        oracle_action = int(self._oracle[self._cursor])
        reward        = 0.0
        terminated    = False

        # ── Xử lý Action ─────────────────────────────────────────────
        if self._position_dir == 0:
            # Flat: có thể mở lệnh mới
            if action == 1:   # Buy
                commission             = self._reward_calc.on_open_commission()
                self._balance         += commission
                self._position_dir     = 1
                self._entry_price      = entry_price + self._spread_usd
                self._consecutive_hold = 0
                reward                 = commission
            elif action == 2:  # Sell
                commission             = self._reward_calc.on_open_commission()
                self._balance         += commission
                self._position_dir     = -1
                self._entry_price      = entry_price - self._spread_usd
                self._consecutive_hold = 0
                reward                 = commission
            else:  # Hold khi Flat — phạt đứng im
                self._consecutive_hold += 1
                reward = self._reward_calc.on_hold(
                    self._consecutive_hold,
                    has_position=False,
                    oracle_action=oracle_action,
                )
        else:
            # Có vị thế: kiểm tra đảo chiều hay giữ tiếp
            is_reversal = (
                (self._position_dir == 1  and action == 2) or  # Long + Sell
                (self._position_dir == -1 and action == 1)     # Short + Buy
            )
            is_close = (action == 0)  # Hold = đóng lệnh

            if is_close or is_reversal:
                # ── Đóng lệnh hiện tại ───────────────────────────────
                pnl = self._calc_pnl(
                    self._entry_price, current_close,
                    self._position_dir, self._lot
                )
                self._balance         += pnl
                self._peak_balance     = max(self._peak_balance, self._balance)
                reward                 = self._reward_calc.on_close(
                    pnl, self._peak_balance, self._balance
                )
                self._position_dir     = 0
                self._consecutive_hold = 0

                if is_reversal:
                    # ── Mở ngay lệnh ngược chiều trong cùng 1 step ──
                    commission         = self._reward_calc.on_open_commission()
                    self._balance     += commission
                    reward            += commission
                    new_dir            = 1 if action == 1 else -1
                    spread_adj         = self._spread_usd if new_dir == 1 else -self._spread_usd
                    self._position_dir = new_dir
                    self._entry_price  = entry_price + spread_adj
            else:
                # Giữ lệnh tiếp — kiểm tra swap qua đêm
                self._consecutive_hold += 1
                is_midnight = (self._cursor % 96 == 0)  # M15: 96 nến = 24h
                is_friday   = (self._get_current_dow() == 4 and is_midnight)
                swap_r      = self._reward_calc.on_midnight_swap(is_friday) if is_midnight else 0.0
                self._balance += swap_r
                reward = self._reward_calc.on_hold(
                    self._consecutive_hold,
                    has_position=True,
                    oracle_action=oracle_action,
                ) + swap_r

        # ── Kiểm tra điều kiện kết thúc & Tính Equity ─────────────────────
        # [FIX SHARPE] Tính Equity = Balance + Unrealized PnL để report chính xác
        unrealized_pnl = 0.0
        if self._position_dir != 0:
            unrealized_pnl = self._calc_pnl(
                self._entry_price, current_close,
                self._position_dir, self._lot
            )
        equity = self._balance + unrealized_pnl

        self._cursor += 1
        
        # Nếu dùng mode test (end_idx != -1) -> dừng khi hết chunk
        if self._end_idx != -1 and self._cursor >= (self._end_idx - self._start_idx) - 1:
            truncated = True
        elif self._is_prewindowed and self._cursor >= len(self._close) - 1:
            truncated = True
        # Nếu dùng 2D raw -> dừng khi hết features
        elif not self._is_prewindowed and self._cursor >= len(self._features) - 1:
            truncated = True
        else:
            truncated = False

        # Phạt cháy tài khoản (Drawdown lố)
        terminated = False
        if equity <= self._init_balance - self._max_dd:
            terminated = True
            reward    -= 5.0  

        return self._get_obs(), float(reward), terminated, truncated, {
            "balance":  self._balance,
            "equity":   equity,  # Dùng để tính Sharpe ratio thực sự
            "drawdown": self._peak_balance - equity,
        }

    # ── Helpers ────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """
        [FIX DUAL-MODE & OOM]
        - 3D pre-windowed (HDF5 array cache): trả là features[cursor] cực nhanh
        - 2D raw            : slice features[cursor-W : cursor]
        """
        if self._is_prewindowed:
            return self._features[self._cursor].copy()
        else:
            start = self._cursor - self._window
            return self._features[start:self._cursor].copy()

    def _get_current_dow(self) -> int:
        """Ngày trong tuần (0=Mon, 4=Fri) — ước tính từ cursor."""
        return (self._cursor // 96) % 7   # M15: 96 nến = 1 ngày

    def _calc_pnl(
        self,
        entry_price: float,
        exit_price:  float,
        direction:   int,    # 1=long, -1=short
        lot:         float,
    ) -> float:
        """
        XAUUSD 1 lot chuẩn = 100 oz vàng.
        PnL = price_diff_USD × lot_size × 100
        Ví dụ: entry 1900, exit 1910, long → $10 × 0.01 × 100 = $10.00
        """
        contract_size = 100.0
        return (exit_price - entry_price) * direction * lot * contract_size
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
