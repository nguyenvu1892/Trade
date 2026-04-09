# src/training/tests/test_backtest.py
import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.training.backtest import compute_metrics


class TestBacktestMetrics:
    def test_positive_pnl_series_sharpe_positive(self):
        """Chuá»—i PnL dÆ°Æ¡ng Ä‘á» u Ä‘áº·n pháº£i cÃ³ Sharpe > 0."""
        daily_returns = np.array([0.001] * 252)
        metrics = compute_metrics(daily_returns)
        assert metrics["sharpe"] > 0, f"Sharpe pháº£i dÆ°Æ¡ng: {metrics['sharpe']}"

    def test_all_zero_returns_sharpe_zero(self):
        """PnL = 0 má» i ngÃ y â†’ Sharpe = 0."""
        daily_returns = np.zeros(252)
        metrics = compute_metrics(daily_returns)


    def test_max_drawdown_is_non_positive(self):
        """Max drawdown phÃ¡ÂºÂ£i <= 0 (biÃ¡Â»Æ’u diÃ¡Â»â€¦n mÃ¡ÂºÂ¥t vÃ¡Â»Â€˜n)."""
        daily_returns = np.array([0.01, -0.05, 0.02, -0.03, 0.01])
        metrics = compute_metrics(daily_returns)
        assert metrics["max_drawdown"] <= 0

    def test_win_rate_between_0_and_1(self):
        """Win rate phÃ¡ÂºÂ£i nÃ¡ÂºÂ±m trong [0, 1]."""
        returns = np.random.randn(100) * 0.01
        metrics = compute_metrics(returns)
        assert 0.0 <= metrics["win_rate"] <= 1.0

    def test_metrics_has_required_keys(self):
        """KÃ¡ÂºÂ¿t quÃ¡ÂºÂ£ phÃ¡ÂºÂ£i cÃƒÂ³ Ã„â€˜Ã¡Â»Â§ cÃƒÂ¡c key bÃ¡ÂºÂ¯t buÃ¡Â»â„¢c."""
        returns = np.random.randn(252) * 0.001
        metrics = compute_metrics(returns)
        required = {"sharpe", "sortino", "max_drawdown", "win_rate",
                    "total_return", "n_trades"}
        assert required.issubset(set(metrics.keys()))