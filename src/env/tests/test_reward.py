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
        assert abs(r - (-0.15)) < 1e-6, f"Swap thứ 6 phải là -$0.15, nhận: {r}"

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