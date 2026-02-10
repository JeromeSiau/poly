"""Unified Risk Manager for all arbitrage strategies.

This module provides a shared risk management layer across all strategies
(Reality Arb, Cross-Market Arb, Crypto Arb, NO Bet). It enforces:
- Global capital limits and per-strategy allocation
- Position sizing using fractional Kelly criterion
- Daily loss limit tracking and trading halts
"""

from typing import Literal

import structlog

logger = structlog.get_logger()

StrategyType = Literal["reality", "crossmarket", "crypto", "nobet", "fear"]


class UnifiedRiskManager:
    """Unified Risk Manager for arbitrage strategies.

    Manages capital allocation, position limits, and daily loss tracking
    across multiple arbitrage strategies (Reality Arb and Cross-Market Arb).
    """

    def __init__(
        self,
        global_capital: float,
        reality_allocation_pct: float,
        crossmarket_allocation_pct: float,
        max_position_pct: float,
        daily_loss_limit_pct: float,
        crypto_allocation_pct: float = 0.0,
        nobet_allocation_pct: float = 0.0,
        fear_allocation_pct: float = 0.0,
    ) -> None:
        """Initialize the Unified Risk Manager.

        Args:
            global_capital: Total capital available for trading
            reality_allocation_pct: Percentage of capital allocated to Reality Arb (0-100)
            crossmarket_allocation_pct: Percentage of capital allocated to Cross-Market Arb (0-100)
            max_position_pct: Maximum position size as a fraction of allocated capital (0-1)
            daily_loss_limit_pct: Daily loss limit as a fraction of global capital (0-1)
            crypto_allocation_pct: Percentage of capital allocated to Crypto Arb (0-100)
            nobet_allocation_pct: Percentage of capital allocated to NO Bet strategy (0-100)
            fear_allocation_pct: Percentage of capital allocated to Fear Selling strategy (0-100)
        """
        self._global_capital = global_capital
        self._reality_allocation_pct = reality_allocation_pct
        self._crossmarket_allocation_pct = crossmarket_allocation_pct
        self._crypto_allocation_pct = crypto_allocation_pct
        self._nobet_allocation_pct = nobet_allocation_pct
        self._fear_allocation_pct = fear_allocation_pct
        self._max_position_pct = max_position_pct
        self._daily_loss_limit_pct = daily_loss_limit_pct

        # Daily P&L tracking
        self._daily_pnl: float = 0.0
        self._daily_pnl_by_strategy: dict[str, float] = {
            "reality": 0.0,
            "crossmarket": 0.0,
            "crypto": 0.0,
            "nobet": 0.0,
            "fear": 0.0,
        }

        # Trading halt flag
        self._is_halted: bool = False

        logger.info(
            "risk_manager_initialized",
            global_capital=global_capital,
            reality_allocation_pct=reality_allocation_pct,
            crossmarket_allocation_pct=crossmarket_allocation_pct,
            max_position_pct=max_position_pct,
            daily_loss_limit_pct=daily_loss_limit_pct,
            fear_allocation_pct=fear_allocation_pct,
        )

    @property
    def daily_pnl(self) -> float:
        """Get the current daily P&L across all strategies."""
        return self._daily_pnl

    @property
    def is_halted(self) -> bool:
        """Check if trading is halted due to risk limits."""
        return self._is_halted

    def get_available_capital(self, strategy: StrategyType) -> float:
        """Get the allocated capital for a specific strategy.

        Args:
            strategy: The strategy type ("reality" or "crossmarket")

        Returns:
            The allocated capital amount for the strategy
        """
        if strategy == "reality":
            allocation_pct = self._reality_allocation_pct
        elif strategy == "crossmarket":
            allocation_pct = self._crossmarket_allocation_pct
        elif strategy == "crypto":
            allocation_pct = self._crypto_allocation_pct
        elif strategy == "nobet":
            allocation_pct = self._nobet_allocation_pct
        elif strategy == "fear":
            allocation_pct = self._fear_allocation_pct
        else:
            logger.warning("unknown_strategy", strategy=strategy)
            return 0.0

        capital = self._global_capital * (allocation_pct / 100.0)

        logger.debug(
            "available_capital_calculated",
            strategy=strategy,
            allocation_pct=allocation_pct,
            capital=capital,
        )

        return capital

    def check_position_limit(self, size: float, strategy: StrategyType) -> bool:
        """Check if a position size is within the allowed limit.

        Args:
            size: The proposed position size in dollars
            strategy: The strategy type ("reality" or "crossmarket")

        Returns:
            True if the position is within limits, False otherwise
        """
        available_capital = self.get_available_capital(strategy)
        max_position = available_capital * self._max_position_pct

        within_limit = size <= max_position

        logger.debug(
            "position_limit_check",
            strategy=strategy,
            size=size,
            max_position=max_position,
            within_limit=within_limit,
        )

        return within_limit

    def check_daily_loss_limit(self) -> bool:
        """Check if the daily loss limit has been exceeded.

        Returns:
            True if trading is allowed (not exceeded), False if limit hit
        """
        loss_limit = self._global_capital * self._daily_loss_limit_pct

        # Check if loss (negative P&L) exceeds limit
        current_loss = -self._daily_pnl if self._daily_pnl < 0 else 0.0
        within_limit = current_loss <= loss_limit

        if not within_limit and not self._is_halted:
            self._is_halted = True
            logger.warning(
                "daily_loss_limit_exceeded",
                daily_pnl=self._daily_pnl,
                loss_limit=loss_limit,
            )

        return within_limit

    def record_pnl(self, amount: float, strategy: StrategyType) -> None:
        """Record a P&L amount for a strategy.

        Args:
            amount: The P&L amount (positive for profit, negative for loss)
            strategy: The strategy type ("reality" or "crossmarket")
        """
        self._daily_pnl += amount
        self._daily_pnl_by_strategy[strategy] = (
            self._daily_pnl_by_strategy.get(strategy, 0.0) + amount
        )

        logger.info(
            "pnl_recorded",
            amount=amount,
            strategy=strategy,
            daily_pnl=self._daily_pnl,
            strategy_pnl=self._daily_pnl_by_strategy[strategy],
        )

        # Check if loss limit is now exceeded
        self.check_daily_loss_limit()

    def calculate_position_size(
        self,
        strategy: StrategyType,
        available_liquidity: float,
        edge_pct: float,
    ) -> float:
        """Calculate optimal position size using fractional Kelly criterion.

        Uses 1/4 Kelly for reduced variance. Position is capped by:
        - Maximum position limit (max_position_pct of allocated capital)
        - Available liquidity (50% of liquidity to avoid market impact)

        Args:
            strategy: The strategy type ("reality" or "crossmarket")
            available_liquidity: Available liquidity in the market
            edge_pct: Expected edge as a decimal (e.g., 0.05 for 5%)

        Returns:
            Optimal position size in dollars
        """
        if self._is_halted:
            logger.warning("trading_halted_no_position")
            return 0.0

        available_capital = self.get_available_capital(strategy)
        max_position = available_capital * self._max_position_pct

        # Kelly criterion: f* = edge / variance
        # For binary outcomes with ~50% probability, variance ~ 0.25
        # Simplified approximation for position sizing
        variance = 0.25  # Assume ~50% probability events

        if variance <= 0 or edge_pct <= 0:
            return 0.0

        # Full Kelly fraction
        kelly_fraction = edge_pct / variance

        # Use fractional Kelly (1/4) for safety
        fractional_kelly = kelly_fraction * 0.25

        # Calculate base position size
        kelly_position = available_capital * fractional_kelly

        # Liquidity constraint: don't take more than 50% of available liquidity
        liquidity_limit = available_liquidity * 0.5

        # Final position size is minimum of all constraints
        position_size = min(max_position, kelly_position, liquidity_limit)
        position_size = max(0.0, position_size)

        logger.debug(
            "position_size_calculated",
            strategy=strategy,
            edge_pct=edge_pct,
            kelly_fraction=kelly_fraction,
            fractional_kelly=fractional_kelly,
            max_position=max_position,
            kelly_position=kelly_position,
            liquidity_limit=liquidity_limit,
            final_position=position_size,
        )

        return position_size

    def reset_daily_stats(self) -> None:
        """Reset daily P&L statistics.

        Should be called at the start of each trading day.
        """
        self._daily_pnl = 0.0
        self._daily_pnl_by_strategy = {
            "reality": 0.0,
            "crossmarket": 0.0,
            "crypto": 0.0,
            "nobet": 0.0,
            "fear": 0.0,
        }
        self._is_halted = False

        logger.info("daily_stats_reset")

    def get_strategy_pnl(self, strategy: StrategyType) -> float:
        """Get the daily P&L for a specific strategy.

        Args:
            strategy: The strategy type ("reality" or "crossmarket")

        Returns:
            The daily P&L for the strategy
        """
        return self._daily_pnl_by_strategy.get(strategy, 0.0)

    def get_risk_status(self) -> dict:
        """Get a summary of the current risk status.

        Returns:
            Dictionary with risk metrics and status
        """
        loss_limit = self._global_capital * self._daily_loss_limit_pct
        current_loss = -self._daily_pnl if self._daily_pnl < 0 else 0.0

        return {
            "global_capital": self._global_capital,
            "daily_pnl": self._daily_pnl,
            "daily_loss_limit": loss_limit,
            "current_loss": current_loss,
            "loss_limit_pct_used": (current_loss / loss_limit * 100) if loss_limit > 0 else 0.0,
            "is_halted": self._is_halted,
            "strategy_pnl": self._daily_pnl_by_strategy.copy(),
            "reality_capital": self.get_available_capital("reality"),
            "crossmarket_capital": self.get_available_capital("crossmarket"),
            "crypto_capital": self.get_available_capital("crypto"),
            "nobet_capital": self.get_available_capital("nobet"),
            "fear_capital": self.get_available_capital("fear"),
        }
