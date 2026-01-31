"""Position sizing and risk management for paper trading."""

from dataclasses import dataclass, field
from datetime import date


def kelly_fraction(
    our_prob: float,
    market_price: float,
    max_fraction: float = 0.25,
) -> float:
    """Calculate Kelly criterion fraction.

    Kelly: f* = (p * b - q) / b
    where p = our prob, q = 1-p, b = decimal odds - 1

    Args:
        our_prob: Our predicted probability of winning
        market_price: Current market price (implied prob)
        max_fraction: Maximum fraction to bet (default 25%)

    Returns:
        Optimal fraction of bankroll to bet (0 to max_fraction)
    """
    if our_prob <= market_price:
        return 0.0

    p = our_prob
    q = 1 - p
    b = (1 / market_price) - 1  # Decimal odds - 1

    if b <= 0:
        return 0.0

    kelly = (p * b - q) / b

    return max(0.0, min(kelly, max_fraction))


@dataclass
class PositionManager:
    """Manage positions and risk for paper trading."""

    capital: float
    max_position_pct: float = 0.25
    min_edge: float = 0.05
    max_daily_loss_pct: float = 0.10

    # State
    daily_pnl: float = field(default=0.0, init=False)
    daily_trades: int = field(default=0, init=False)
    current_date: date = field(default_factory=date.today, init=False)

    def calculate_position_size(
        self,
        our_prob: float,
        market_price: float,
    ) -> float:
        """Calculate position size for a trade."""
        self._check_new_day()

        edge = our_prob - market_price
        if edge < self.min_edge:
            return 0.0

        if not self.can_trade():
            return 0.0

        # Kelly fraction
        fraction = kelly_fraction(our_prob, market_price, self.max_position_pct)

        # Base position size
        size = self.capital * fraction

        # Cap by remaining daily loss allowance
        remaining_loss_allowance = (
            self.capital * self.max_daily_loss_pct + self.daily_pnl
        )
        if remaining_loss_allowance <= 0:
            return 0.0

        # Size can't exceed what we can afford to lose
        size = min(size, remaining_loss_allowance)

        return max(0.0, size)

    def can_trade(self) -> bool:
        """Check if we can place more trades today."""
        self._check_new_day()

        # Check daily loss limit
        if self.daily_pnl <= -self.capital * self.max_daily_loss_pct:
            return False

        return True

    def record_win(self, amount: float) -> None:
        """Record a winning trade."""
        self._check_new_day()
        self.daily_pnl += amount
        self.capital += amount
        self.daily_trades += 1

    def record_loss(self, amount: float) -> None:
        """Record a losing trade."""
        self._check_new_day()
        self.daily_pnl -= amount
        self.capital -= amount
        self.daily_trades += 1

    def _check_new_day(self) -> None:
        """Reset daily counters if new day."""
        today = date.today()
        if today != self.current_date:
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.current_date = today
