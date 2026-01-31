# tests/arb/test_reality_arb.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, UTC

from src.arb.reality_arb import RealityArbEngine, ArbOpportunity
from src.realtime.event_detector import SignificantEvent, EventDetector
from src.realtime.market_mapper import MarketMapping


class TestArbOpportunityDetection:
    @pytest.fixture
    def engine(self):
        engine = RealityArbEngine()
        engine.polymarket_feed = MagicMock()
        # Use real EventDetector for price impact estimation
        engine.event_detector = EventDetector()
        engine.market_mapper = MagicMock()
        return engine

    def test_detect_opportunity_from_baron_kill(self, engine):
        """Baron kill should create arb opportunity."""
        engine.polymarket_feed.get_best_prices.return_value = (0.49, 0.51)

        event = SignificantEvent(
            original_event=None,
            is_significant=True,
            impact_score=0.85,
            favored_team="T1",
            event_description="Baron kill by T1"
        )

        mapping = MarketMapping(
            polymarket_id="0x123",
            game="lol",
            event_identifier="T1_vs_GenG",
            team_to_outcome={"T1": "YES", "Gen.G": "NO"}
        )

        opportunity = engine.evaluate_opportunity(event, mapping)

        assert opportunity is not None
        assert opportunity.edge_pct >= 0.10
        assert opportunity.side == "BUY"
        assert opportunity.outcome == "YES"

    def test_no_opportunity_if_market_already_moved(self, engine):
        """If market already reflects event, no opportunity."""
        # Market already at 0.88 - very little room to move up
        # With diminishing returns near extremes, edge should be minimal
        engine.polymarket_feed.get_best_prices.return_value = (0.87, 0.89)

        event = SignificantEvent(
            original_event=None,
            is_significant=True,
            impact_score=0.30,  # Low impact event
            favored_team="T1",
            event_description="Tower destroyed"
        )

        mapping = MarketMapping(
            polymarket_id="0x123",
            game="lol",
            event_identifier="T1_vs_GenG",
            team_to_outcome={"T1": "YES", "Gen.G": "NO"}
        )

        opportunity = engine.evaluate_opportunity(event, mapping)

        # With price at 0.88 and low impact, edge should be below min threshold
        assert opportunity is None or opportunity.edge_pct < 0.02

    def test_calculates_position_size_correctly(self, engine):
        """Position size respects risk limits."""
        engine.capital = 10000
        engine.max_position_pct = 0.10

        opportunity = ArbOpportunity(
            market_id="0x123",
            side="BUY",
            outcome="YES",
            current_price=0.50,
            estimated_fair_price=0.65,
            edge_pct=0.10,  # Below anomaly threshold (0.15)
            trigger_event="baron_kill",
            timestamp=datetime.now(UTC).timestamp()
        )

        size = engine.calculate_position_size(opportunity)

        assert size <= 1000
        assert size > 0


class TestArbExecution:
    @pytest.fixture
    def engine(self):
        engine = RealityArbEngine()
        engine.executor = AsyncMock()
        return engine

    @pytest.mark.asyncio
    async def test_execute_opportunity(self, engine):
        """Executes trade via executor."""
        engine.executor.place_order.return_value = {
            "order_id": "order_123",
            "status": "FILLED",
            "fill_price": 0.51
        }

        # Use edge_pct=0.10 which is below anomaly threshold (0.15)
        # but above min_edge (0.02) so is_valid returns True
        opportunity = ArbOpportunity(
            market_id="0x123",
            side="BUY",
            outcome="YES",
            current_price=0.50,
            estimated_fair_price=0.60,
            edge_pct=0.10,
            trigger_event="baron_kill",
            timestamp=datetime.now(UTC).timestamp()
        )

        result = await engine.execute(opportunity, size=100)

        assert result["status"] == "FILLED"
        engine.executor.place_order.assert_called_once()
