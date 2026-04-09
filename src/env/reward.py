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