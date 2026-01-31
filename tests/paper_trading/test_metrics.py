"""Tests for paper trading metrics."""

import pytest
from datetime import datetime

from src.paper_trading.metrics import PaperTradingMetrics, TradeRecord


class TestPaperTradingMetrics:
    @pytest.fixture
    def sample_trades(self):
        return [
            TradeRecord(
                timestamp=datetime(2026, 1, 1, 10, 0),
                edge_theoretical=0.10,
                edge_realized=0.08,
                pnl=40.0,
                size=500,
            ),
            TradeRecord(
                timestamp=datetime(2026, 1, 1, 11, 0),
                edge_theoretical=0.12,
                edge_realized=-0.02,
                pnl=-10.0,
                size=500,
            ),
            TradeRecord(
                timestamp=datetime(2026, 1, 1, 12, 0),
                edge_theoretical=0.08,
                edge_realized=0.06,
                pnl=30.0,
                size=500,
            ),
        ]

    def test_total_pnl(self, sample_trades):
        metrics = PaperTradingMetrics(sample_trades)
        assert metrics.total_pnl == 60.0

    def test_win_rate(self, sample_trades):
        metrics = PaperTradingMetrics(sample_trades)
        assert metrics.win_rate == pytest.approx(2 / 3, rel=0.01)

    def test_avg_edge_theoretical(self, sample_trades):
        metrics = PaperTradingMetrics(sample_trades)
        assert metrics.avg_edge_theoretical == pytest.approx(0.10, rel=0.01)

    def test_avg_edge_realized(self, sample_trades):
        metrics = PaperTradingMetrics(sample_trades)
        expected = (0.08 + (-0.02) + 0.06) / 3
        assert metrics.avg_edge_realized == pytest.approx(expected, rel=0.01)

    def test_sharpe_ratio(self, sample_trades):
        metrics = PaperTradingMetrics(sample_trades)
        # Should be positive given overall profit
        assert metrics.sharpe_ratio > 0

    def test_max_drawdown(self, sample_trades):
        metrics = PaperTradingMetrics(sample_trades)
        # Max drawdown happened at trade 2: from 40 to 30
        assert metrics.max_drawdown == 10.0

    def test_empty_trades_handle_gracefully(self):
        metrics = PaperTradingMetrics([])
        assert metrics.total_pnl == 0
        assert metrics.win_rate == 0
        assert metrics.sharpe_ratio == 0

    def test_as_dict(self, sample_trades):
        metrics = PaperTradingMetrics(sample_trades)
        d = metrics.as_dict()

        assert "total_pnl" in d
        assert "win_rate" in d
        assert "n_trades" in d
        assert d["n_trades"] == 3
