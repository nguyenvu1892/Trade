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
        random_start:      bool  = True,  # [FIX OOS RANDOM] Randomize cursor nếu train, tuần tự nếu test
    ):
        super().__init__()
        
        self.h5_path       = h5_path
        self._start_idx    = start_idx
        self._end_idx      = end_idx
        self._random_start = random_start

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
            assert features is not None and features.ndim in (2, 3)
            self._is_prewindowed = (features.ndim == 3)
            self._features       = features
            self._close          = close_prices
            self._open_next      = open_next_prices       
            self._oracle         = oracle_labels
            n_features           = features.shape[1]
        self._window         = window_size
        self._spread_price_shift     = spread_pips * lot_size / 100.0
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
        # [FIX GROUNDHOG OOS] Randomize điểm bắt đầu mỗi khi reset nếu train, tuần tự từ 0 nếu test
        if self._is_prewindowed:
            if self._random_start:
                max_start = len(self._features) - 2000
                self._cursor = np.random.randint(0, max(1, max_start))
            else:
                self._cursor = 0
        else:
            self._cursor = self._window
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
                self._entry_price      = entry_price + self._spread_price_shift
                self._consecutive_hold = 0
                reward                 = commission
            elif action == 2:  # Sell
                commission             = self._reward_calc.on_open_commission()
                self._balance         += commission
                self._position_dir     = -1
                self._entry_price      = entry_price - self._spread_price_shift
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
                # [FIX EXIT LOOKAHEAD] Đóng lệnh ở giá open nhánh tiếp theo, không dùng current_close!
                pnl = self._calc_pnl(
                    self._entry_price, entry_price, # entry_price chính là open_next
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
                    spread_adj         = self._spread_price_shift if new_dir == 1 else -self._spread_price_shift
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