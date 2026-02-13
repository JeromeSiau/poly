"""Cross-Market Arbitrage Engine.

This is the core engine that detects price discrepancies between prediction market
platforms (Polymarket, Azuro, Overtime), evaluates opportunities, and triggers
alerts or execution.

The engine:
1. Scans matched events across platforms for price discrepancies
2. Calculates edge after fees and gas
3. Evaluates opportunities using risk management
4. Triggers alerts for manual approval or executes in autopilot mode
"""

from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Any, Optional

import structlog

from config.settings import settings
from src.risk.guard import RiskGuard
from src.risk.sizing import kelly_position_size

logger = structlog.get_logger()


@dataclass
class CrossMarketOpportunity:
    """Represents a cross-market arbitrage opportunity.

    An opportunity exists when the same event has different prices on
    different platforms, allowing for a buy-low-sell-high strategy.

    Attributes:
        event: The matched event across platforms
        source_platform: Platform to buy from (lower price)
        source_price: Price on source platform
        source_liquidity: Available liquidity on source platform
        target_platform: Platform to sell on (higher price)
        target_price: Price on target platform
        target_liquidity: Available liquidity on target platform
        fees_pct: Total fees as a percentage (default 0)
        gas_estimate: Estimated gas cost in USD (default 0)
        timestamp: When the opportunity was detected
    """

    event: Any
    source_platform: str
    source_price: float
    source_liquidity: float
    target_platform: str
    target_price: float
    target_liquidity: float
    fees_pct: float = 0.0
    gas_estimate: float = 0.0
    timestamp: float = field(default_factory=lambda: datetime.now(UTC).timestamp())

    @property
    def gross_edge_pct(self) -> float:
        """Calculate the gross edge percentage before fees.

        Returns:
            The price difference as a percentage (target - source)
        """
        return self.target_price - self.source_price

    @property
    def net_edge_pct(self) -> float:
        """Calculate the net edge percentage after fees.

        Net edge accounts for:
        - Platform fees (both sides)
        - Gas costs (converted to percentage of min liquidity)

        Returns:
            The net edge after fees as a percentage
        """
        # Calculate gas as percentage of minimum tradeable amount
        min_liquidity = min(self.source_liquidity, self.target_liquidity)
        gas_pct = 0.0
        if min_liquidity > 0:
            gas_pct = self.gas_estimate / min_liquidity

        return self.gross_edge_pct - self.fees_pct - gas_pct

    @property
    def is_valid(self) -> bool:
        """Check if the opportunity is valid for trading.

        An opportunity is valid if:
        - Net edge is above the minimum threshold
        - Net edge is below the anomaly threshold (too good = suspicious)
        - Prices are valid (between 0 and 1)

        Returns:
            True if the opportunity is valid for trading
        """
        min_edge = settings.CROSSMARKET_MIN_EDGE_PCT
        max_edge = settings.ANOMALY_THRESHOLD_PCT

        return (
            self.net_edge_pct >= min_edge
            and self.net_edge_pct < max_edge
            and 0 < self.source_price < 1
            and 0 < self.target_price < 1
        )

    @property
    def min_liquidity(self) -> float:
        """Get the minimum available liquidity across both platforms.

        Returns:
            The minimum of source and target liquidity
        """
        return min(self.source_liquidity, self.target_liquidity)

    def to_dict(self) -> dict[str, Any]:
        """Convert opportunity to dictionary for serialization.

        Returns:
            Dictionary representation of the opportunity
        """
        return {
            "event_name": getattr(self.event, "name", str(self.event)),
            "source_platform": self.source_platform,
            "source_price": self.source_price,
            "source_liquidity": self.source_liquidity,
            "target_platform": self.target_platform,
            "target_price": self.target_price,
            "target_liquidity": self.target_liquidity,
            "gross_edge_pct": self.gross_edge_pct,
            "net_edge_pct": self.net_edge_pct,
            "fees_pct": self.fees_pct,
            "gas_estimate": self.gas_estimate,
            "is_valid": self.is_valid,
            "timestamp": self.timestamp,
        }


@dataclass
class EvaluationResult:
    """Result of evaluating a cross-market opportunity.

    Attributes:
        opportunity: The evaluated opportunity
        position_size: Recommended position size in USD
        approved: Whether the opportunity passed all checks
        rejection_reason: Reason for rejection if not approved
    """

    opportunity: CrossMarketOpportunity
    position_size: float
    approved: bool = True
    rejection_reason: Optional[str] = None


class CrossMarketArbEngine:
    """Cross-Market Arbitrage Engine.

    Detects and evaluates arbitrage opportunities across prediction market
    platforms. Uses RiskGuard for circuit-breaker gating and
    ``kelly_position_size`` for sizing.
    """

    def __init__(
        self,
        guard: Optional[RiskGuard] = None,
        allocated_capital: float = 0.0,
        max_position_pct: float = 0.0,
        min_edge_pct: Optional[float] = None,
    ):
        """Initialize the Cross-Market Arbitrage Engine.

        Args:
            guard: Optional RiskGuard for circuit-breaker gating
            allocated_capital: Capital allocated to this strategy (USD)
            max_position_pct: Max position as fraction of capital (0-1)
            min_edge_pct: Minimum edge percentage to consider (default from settings)
        """
        self.guard = guard
        self.allocated_capital = (
            allocated_capital
            if allocated_capital > 0
            else settings.GLOBAL_CAPITAL
            * (settings.CAPITAL_ALLOCATION_CROSSMARKET_PCT / 100.0)
        )
        self.max_position_pct = max_position_pct or settings.MAX_POSITION_PCT
        self.min_edge_pct = min_edge_pct or settings.CROSSMARKET_MIN_EDGE_PCT

        # Tracking state
        self._is_running: bool = False
        self._opportunities_found: int = 0
        self._pending_opportunities: dict[str, CrossMarketOpportunity] = {}

        logger.info(
            "cross_market_arb_engine_initialized",
            min_edge_pct=self.min_edge_pct,
        )

    @property
    def is_running(self) -> bool:
        """Check if the engine is currently running.

        Returns:
            True if the engine is running
        """
        return self._is_running

    @property
    def opportunities_found(self) -> int:
        """Get the total number of opportunities found.

        Returns:
            Count of opportunities found since engine start
        """
        return self._opportunities_found

    def find_opportunities_for_event(
        self,
        event: Any,
        prices: dict[str, dict[str, float]],
        liquidity: Optional[dict[str, dict[str, float]]] = None,
    ) -> list[CrossMarketOpportunity]:
        """Find arbitrage opportunities for a matched event.

        Compares prices across platforms to find buy-low-sell-high opportunities.

        Args:
            event: MatchedEvent with platform IDs
            prices: Price data by platform and outcome, e.g.,
                    {"polymarket": {"YES": 0.42, "NO": 0.58}, ...}
            liquidity: Optional liquidity data by platform and outcome

        Returns:
            List of CrossMarketOpportunity instances found
        """
        opportunities = []

        # Default liquidity if not provided
        if liquidity is None:
            liquidity = {platform: {"YES": 10000.0, "NO": 10000.0} for platform in prices}

        # Get all platforms with prices
        platforms = list(prices.keys())

        # Compare each pair of platforms
        for i, source_platform in enumerate(platforms):
            for target_platform in platforms[i + 1:]:
                source_prices = prices.get(source_platform, {})
                target_prices = prices.get(target_platform, {})
                source_liq = liquidity.get(source_platform, {"YES": 10000.0, "NO": 10000.0})
                target_liq = liquidity.get(target_platform, {"YES": 10000.0, "NO": 10000.0})

                # Check YES outcome
                if "YES" in source_prices and "YES" in target_prices:
                    opps = self._compare_prices(
                        event=event,
                        outcome="YES",
                        platform1=source_platform,
                        price1=source_prices["YES"],
                        liquidity1=source_liq.get("YES", 10000.0),
                        platform2=target_platform,
                        price2=target_prices["YES"],
                        liquidity2=target_liq.get("YES", 10000.0),
                    )
                    opportunities.extend(opps)

                # Check NO outcome
                if "NO" in source_prices and "NO" in target_prices:
                    opps = self._compare_prices(
                        event=event,
                        outcome="NO",
                        platform1=source_platform,
                        price1=source_prices["NO"],
                        liquidity1=source_liq.get("NO", 10000.0),
                        platform2=target_platform,
                        price2=target_prices["NO"],
                        liquidity2=target_liq.get("NO", 10000.0),
                    )
                    opportunities.extend(opps)

        # Log findings
        for opp in opportunities:
            if opp.is_valid:
                self._opportunities_found += 1
                logger.info(
                    "opportunity_found",
                    event_name=getattr(event, "name", str(event)),
                    source=opp.source_platform,
                    target=opp.target_platform,
                    gross_edge=opp.gross_edge_pct,
                    net_edge=opp.net_edge_pct,
                )

        return opportunities

    def _compare_prices(
        self,
        event: Any,
        outcome: str,
        platform1: str,
        price1: float,
        liquidity1: float,
        platform2: str,
        price2: float,
        liquidity2: float,
    ) -> list[CrossMarketOpportunity]:
        """Compare prices between two platforms for an outcome.

        Creates opportunities for both directions if edge exists.

        Args:
            event: The matched event
            outcome: The outcome being compared (YES/NO)
            platform1: First platform name
            price1: Price on first platform
            liquidity1: Liquidity on first platform
            platform2: Second platform name
            price2: Price on second platform
            liquidity2: Liquidity on second platform

        Returns:
            List of opportunities (0-2 depending on price differences)
        """
        opportunities = []

        # Check if platform1 is cheaper (buy on 1, sell on 2)
        if price2 > price1:
            edge = price2 - price1
            if edge >= self.min_edge_pct:
                opp = CrossMarketOpportunity(
                    event=event,
                    source_platform=platform1,
                    source_price=price1,
                    source_liquidity=liquidity1,
                    target_platform=platform2,
                    target_price=price2,
                    target_liquidity=liquidity2,
                )
                opportunities.append(opp)

        # Check if platform2 is cheaper (buy on 2, sell on 1)
        if price1 > price2:
            edge = price1 - price2
            if edge >= self.min_edge_pct:
                opp = CrossMarketOpportunity(
                    event=event,
                    source_platform=platform2,
                    source_price=price2,
                    source_liquidity=liquidity2,
                    target_platform=platform1,
                    target_price=price1,
                    target_liquidity=liquidity1,
                )
                opportunities.append(opp)

        return opportunities

    async def evaluate_opportunity(
        self,
        opportunity: CrossMarketOpportunity,
    ) -> Optional[EvaluationResult]:
        """Evaluate an opportunity for trading.

        Performs pre-flight checks:
        - Validates the opportunity is still valid
        - Checks daily loss limits
        - Checks position limits
        - Calculates optimal position size

        Args:
            opportunity: The opportunity to evaluate

        Returns:
            EvaluationResult with position size and approval status,
            or None if opportunity is rejected
        """
        # Check if opportunity is valid
        if not opportunity.is_valid:
            logger.debug(
                "opportunity_invalid",
                net_edge=opportunity.net_edge_pct,
                min_edge=self.min_edge_pct,
            )
            return EvaluationResult(
                opportunity=opportunity,
                position_size=0.0,
                approved=False,
                rejection_reason="Opportunity net edge below minimum threshold",
            )

        # Circuit-breaker check
        if self.guard and self.guard.circuit_broken:
            logger.warning("circuit_breaker_active")
            return EvaluationResult(
                opportunity=opportunity,
                position_size=0.0,
                approved=False,
                rejection_reason="Circuit breaker active",
            )

        # Position sizing via Kelly
        position_size = kelly_position_size(
            capital=self.allocated_capital,
            edge=opportunity.net_edge_pct,
            max_pct=self.max_position_pct,
            liquidity=opportunity.min_liquidity,
        )

        # Inline position limit check
        max_position = self.allocated_capital * self.max_position_pct
        if position_size > max_position:
            logger.warning(
                "position_limit_exceeded",
                position_size=position_size,
                max_position=max_position,
            )
            return EvaluationResult(
                opportunity=opportunity,
                position_size=0.0,
                approved=False,
                rejection_reason="Position limit exceeded",
            )

        # Store pending opportunity
        event_key = self._make_opportunity_key(opportunity)
        self._pending_opportunities[event_key] = opportunity

        logger.info(
            "opportunity_evaluated",
            event_name=getattr(opportunity.event, "name", str(opportunity.event)),
            position_size=position_size,
            net_edge=opportunity.net_edge_pct,
            approved=True,
        )

        return EvaluationResult(
            opportunity=opportunity,
            position_size=position_size,
            approved=True,
        )

    async def scan_all_events(
        self,
        matched_events: list[Any],
        price_feeds: dict[str, dict[str, dict[str, float]]],
        liquidity_feeds: Optional[dict[str, dict[str, dict[str, float]]]] = None,
    ) -> list[EvaluationResult]:
        """Scan all matched events for arbitrage opportunities.

        This is the main scanning loop that:
        1. Iterates through all matched events
        2. Finds opportunities for each event
        3. Evaluates valid opportunities

        Args:
            matched_events: List of MatchedEvent objects
            price_feeds: Price data by event_id, platform, and outcome
            liquidity_feeds: Optional liquidity data by event_id, platform, and outcome

        Returns:
            List of EvaluationResult for all valid opportunities
        """
        self._is_running = True
        results = []

        try:
            for event in matched_events:
                # Get event identifier
                event_id = self._get_event_id(event)

                # Get prices for this event
                event_prices = price_feeds.get(event_id, {})
                if not event_prices:
                    logger.debug("no_prices_for_event", event_id=event_id)
                    continue

                # Get liquidity if available
                event_liquidity = None
                if liquidity_feeds:
                    event_liquidity = liquidity_feeds.get(event_id)

                # Find opportunities
                opportunities = self.find_opportunities_for_event(
                    event=event,
                    prices=event_prices,
                    liquidity=event_liquidity,
                )

                # Evaluate each valid opportunity
                for opp in opportunities:
                    if opp.is_valid:
                        result = await self.evaluate_opportunity(opp)
                        if result and result.approved:
                            results.append(result)

            logger.info(
                "scan_complete",
                events_scanned=len(matched_events),
                opportunities_found=len(results),
            )

        finally:
            self._is_running = False

        return results

    def _get_event_id(self, event: Any) -> str:
        """Get a unique identifier for an event.

        Args:
            event: The event object

        Returns:
            String identifier for the event
        """
        # Try common ID attributes
        if hasattr(event, "polymarket_id") and event.polymarket_id:
            return event.polymarket_id
        if hasattr(event, "azuro_condition_id") and event.azuro_condition_id:
            return event.azuro_condition_id
        if hasattr(event, "overtime_game_id") and event.overtime_game_id:
            return event.overtime_game_id
        if hasattr(event, "name"):
            return event.name
        return str(id(event))

    def _make_opportunity_key(self, opportunity: CrossMarketOpportunity) -> str:
        """Create a unique key for an opportunity.

        Args:
            opportunity: The opportunity

        Returns:
            Unique string key
        """
        event_id = self._get_event_id(opportunity.event)
        return f"{event_id}:{opportunity.source_platform}:{opportunity.target_platform}"

    def get_pending_opportunities(self) -> list[CrossMarketOpportunity]:
        """Get all pending opportunities awaiting action.

        Returns:
            List of pending CrossMarketOpportunity instances
        """
        return list(self._pending_opportunities.values())

    def clear_opportunity(self, key: str) -> None:
        """Clear a pending opportunity.

        Args:
            key: The opportunity key to clear
        """
        self._pending_opportunities.pop(key, None)

    def clear_all_opportunities(self) -> None:
        """Clear all pending opportunities."""
        self._pending_opportunities.clear()
        logger.debug("all_opportunities_cleared")
