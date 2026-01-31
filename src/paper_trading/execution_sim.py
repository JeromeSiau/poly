# src/paper_trading/execution_sim.py
"""Simulate trade execution with slippage."""

from dataclasses import dataclass


@dataclass
class ExecutionSimulator:
    """Simulate realistic trade execution with slippage.

    Models slippage as proportional to size / depth.
    """

    default_depth: float = 10000  # Default orderbook depth in dollars
    max_slippage: float = 0.05   # Max 5% slippage

    def simulate_fill(
        self,
        target_price: float,
        size: float,
        side: str,
        orderbook_depth: float | None = None,
    ) -> float:
        """Simulate order fill with slippage.

        Args:
            target_price: Target fill price (best bid/ask)
            size: Order size in dollars
            side: "BUY" or "SELL"
            orderbook_depth: Available liquidity (uses default if None)

        Returns:
            Simulated fill price including slippage
        """
        depth = orderbook_depth or self.default_depth

        # Slippage proportional to size / depth
        # Using square root for more realistic impact
        impact_ratio = (size / depth) ** 0.5
        slippage_pct = min(impact_ratio * 0.02, self.max_slippage)

        if side.upper() == "BUY":
            fill_price = target_price * (1 + slippage_pct)
        else:
            fill_price = target_price * (1 - slippage_pct)

        # Clamp to valid price range
        return max(0.01, min(0.99, fill_price))

    def estimate_slippage_pct(
        self,
        size: float,
        orderbook_depth: float | None = None,
    ) -> float:
        """Estimate slippage percentage for an order."""
        depth = orderbook_depth or self.default_depth
        impact_ratio = (size / depth) ** 0.5
        return min(impact_ratio * 0.02, self.max_slippage)
