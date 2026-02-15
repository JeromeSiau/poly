# tests/integration/test_reality_arb_flow.py
"""Integration test for full Reality Arb flow with mock data."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, UTC

from src.feeds.base import FeedEvent
from src.feeds.pandascore import PandaScoreFeed, PandaScoreEvent
from src.feeds.polymarket import PolymarketFeed
from src.realtime.event_detector import EventDetector
from src.realtime.market_mapper import MarketMapper
from src.arb.reality_arb import RealityArbEngine


class TestFullArbFlow:
    """Test complete flow from game event to trade opportunity."""

    @pytest.fixture
    def setup_components(self):
        """Setup all components with mocks."""
        # Mock Polymarket feed
        polymarket = MagicMock(spec=PolymarketFeed)
        polymarket.get_best_prices.return_value = (0.48, 0.52)  # 50% market

        # Real detector and mapper
        detector = EventDetector()
        mapper = MarketMapper()

        # Add test mapping
        mapper.add_mapping(
            game="lol",
            event_identifier="T1_vs_GenG_LCK_Finals",
            polymarket_id="0xtest123",
            outcomes={"T1": "YES", "Gen.G": "NO"}
        )

        # Engine
        engine = RealityArbEngine(
            polymarket_feed=polymarket,
            event_detector=detector,
            market_mapper=mapper
        )
        engine.capital = 10000
        engine.anomaly_threshold = 0.25  # CS:GO match points produce legit high edges

        return {
            "polymarket": polymarket,
            "detector": detector,
            "mapper": mapper,
            "engine": engine
        }

    @pytest.mark.asyncio
    async def test_baron_kill_creates_opportunity(self, setup_components):
        """Baron kill should create high-edge opportunity."""
        engine = setup_components["engine"]
        mapper = setup_components["mapper"]

        # Simulate Baron kill event
        event = FeedEvent(
            source="pandascore",
            event_type="baron",
            game="lol",
            data={"team": "T1"},
            timestamp=datetime.now(UTC).timestamp(),
            match_id="match_123"
        )

        # Classify event
        significant = setup_components["detector"].classify(event)

        assert significant.is_significant
        assert significant.impact_score >= 0.8
        assert significant.favored_team == "T1"

        # Find market and evaluate opportunity
        mapping = mapper.find_market(game="lol", teams=["T1", "Gen.G"])
        opportunity = engine.evaluate_opportunity(significant, mapping)

        assert opportunity is not None
        assert opportunity.side == "BUY"
        assert opportunity.outcome == "YES"
        assert opportunity.edge_pct >= 0.10

    @pytest.mark.asyncio
    async def test_csgo_match_point_opportunity(self, setup_components):
        """CS:GO match point should create opportunity."""
        engine = setup_components["engine"]
        mapper = setup_components["mapper"]
        detector = setup_components["detector"]

        # Add CS:GO mapping
        mapper.add_mapping(
            game="csgo",
            event_identifier="Navi_vs_FaZe_Major",
            polymarket_id="0xcsgo456",
            outcomes={"Navi": "YES", "FaZe": "NO"}
        )

        # Simulate match point round win
        event = FeedEvent(
            source="pandascore",
            event_type="round_end",
            game="csgo",
            data={"winner": "Navi", "score": {"Navi": 15, "FaZe": 10}},
            timestamp=datetime.now(UTC).timestamp()
        )

        significant = detector.classify(event)

        assert significant.is_significant
        assert significant.impact_score >= 0.6  # Match point is important

        mapping = mapper.find_market(game="csgo", teams=["Navi", "FaZe"])
        opportunity = engine.evaluate_opportunity(significant, mapping)

        assert opportunity is not None

    @pytest.mark.asyncio
    async def test_no_opportunity_when_market_fair(self, setup_components):
        """No opportunity if market already reflects event."""
        engine = setup_components["engine"]
        mapper = setup_components["mapper"]
        polymarket = setup_components["polymarket"]

        # Market already at 94% (near extreme, limited room to move)
        # With baron impact 0.85, base_movement = 0.153
        # At 94%: room_to_move = 0.06, effective_movement = 0.153 * 0.12 = 0.018
        # Fair price = 0.94 + 0.018 = 0.958
        # Edge = 0.018 which is below 2% threshold
        polymarket.get_best_prices.return_value = (0.93, 0.95)

        event = FeedEvent(
            source="pandascore",
            event_type="baron",
            game="lol",
            data={"team": "T1"},
            timestamp=datetime.now(UTC).timestamp()
        )

        significant = setup_components["detector"].classify(event)
        mapping = mapper.find_market(game="lol", teams=["T1", "Gen.G"])
        opportunity = engine.evaluate_opportunity(significant, mapping)

        # At 94% market price, the edge should be below minimum threshold (2%)
        # because there's very limited room for price to move up
        assert opportunity is None or opportunity.edge_pct < 0.02

    @pytest.mark.asyncio
    async def test_position_sizing_respects_limits(self, setup_components):
        """Position size should respect risk limits."""
        engine = setup_components["engine"]
        engine.capital = 10000
        engine.max_position_pct = 0.10

        from src.arb.reality_arb import ArbOpportunity

        opportunity = ArbOpportunity(
            market_id="0x123",
            side="BUY",
            outcome="YES",
            current_price=0.50,
            estimated_fair_price=0.70,
            edge_pct=0.40,  # Very high edge
            trigger_event="test",
            timestamp=datetime.now(UTC).timestamp()
        )

        size = engine.calculate_position_size(opportunity)

        # Even with 40% edge, should be capped at 10% of capital
        assert size <= 1000
        assert size > 0
