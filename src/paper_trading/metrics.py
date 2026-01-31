"""Metrics calculation for paper trading."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np


@dataclass
class TradeRecord:
    """Record of a single paper trade."""

    timestamp: datetime
    edge_theoretical: float
    edge_realized: float
    pnl: float
    size: float


class PaperTradingMetrics:
    """Calculate performance metrics from paper trades."""

    def __init__(self, trades: list[TradeRecord]):
        """Initialize with list of trades."""
        self.trades = trades

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def total_pnl(self) -> float:
        if not self.trades:
            return 0.0
        return sum(t.pnl for t in self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t.pnl > 0)
        return wins / len(self.trades)

    @property
    def avg_edge_theoretical(self) -> float:
        if not self.trades:
            return 0.0
        return np.mean([t.edge_theoretical for t in self.trades])

    @property
    def avg_edge_realized(self) -> float:
        if not self.trades:
            return 0.0
        return np.mean([t.edge_realized for t in self.trades])

    @property
    def sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio (simplified, no risk-free rate)."""
        if len(self.trades) < 2:
            return 0.0

        returns = [t.pnl / t.size for t in self.trades if t.size > 0]
        if not returns or np.std(returns) == 0:
            return 0.0

        return np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized

    @property
    def max_drawdown(self) -> float:
        """Calculate maximum drawdown in absolute terms."""
        if not self.trades:
            return 0.0

        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0

        for t in self.trades:
            cumulative += t.pnl
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd

        return max_dd

    @property
    def profit_factor(self) -> float:
        """Gross profit / gross loss."""
        if not self.trades:
            return 0.0

        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))

        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0

        return gross_profit / gross_loss

    def as_dict(self) -> dict[str, Any]:
        """Return all metrics as dictionary."""
        return {
            "n_trades": self.n_trades,
            "total_pnl": self.total_pnl,
            "win_rate": self.win_rate,
            "avg_edge_theoretical": self.avg_edge_theoretical,
            "avg_edge_realized": self.avg_edge_realized,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "profit_factor": self.profit_factor,
        }
