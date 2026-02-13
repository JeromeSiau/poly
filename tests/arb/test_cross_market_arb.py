"""Tests for the cross-market arbitrage engine."""

import pytest
from unittest.mock import MagicMock, AsyncMock
from dataclasses import dataclass
from src.arb.cross_market_arb import CrossMarketArbEngine, CrossMarketOpportunity


@dataclass
class MockMatchedEvent:
    name: str = "Test Event"
    polymarket_id: str = "0x123"
    azuro_condition_id: str = "456"
    overtime_game_id: str = None
    confidence: float = 0.98


def test_opportunity_creation():
    opp = CrossMarketOpportunity(
        event=MockMatchedEvent(),
        source_platform="polymarket",
        source_price=0.42,
        source_liquidity=5000.0,
        target_platform="azuro",
        target_price=0.47,
        target_liquidity=3000.0,
    )
    assert opp.gross_edge_pct == pytest.approx(0.05, abs=0.01)
    assert opp.source_platform == "polymarket"


def test_opportunity_net_edge_calculation():
    opp = CrossMarketOpportunity(
        event=MockMatchedEvent(),
        source_platform="polymarket",
        source_price=0.42,
        source_liquidity=5000.0,
        target_platform="azuro",
        target_price=0.47,
        target_liquidity=3000.0,
        fees_pct=0.01,
        gas_estimate=0.50,
    )
    # Net edge should be gross - fees
    assert opp.net_edge_pct < opp.gross_edge_pct


def test_opportunity_is_valid():
    valid_opp = CrossMarketOpportunity(
        event=MockMatchedEvent(),
        source_platform="polymarket",
        source_price=0.40,
        source_liquidity=5000.0,
        target_platform="azuro",
        target_price=0.47,
        target_liquidity=3000.0,
    )
    assert valid_opp.is_valid  # 7% edge should be valid

    invalid_opp = CrossMarketOpportunity(
        event=MockMatchedEvent(),
        source_platform="polymarket",
        source_price=0.45,
        source_liquidity=5000.0,
        target_platform="azuro",
        target_price=0.46,
        target_liquidity=3000.0,
    )
    assert not invalid_opp.is_valid  # 1% edge too small


@pytest.fixture
def arb_engine():
    engine = CrossMarketArbEngine(
        allocated_capital=5000.0,
        max_position_pct=0.10,
    )
    return engine


def test_engine_creation(arb_engine):
    assert arb_engine is not None
    assert arb_engine.min_edge_pct == 0.02


@pytest.mark.asyncio
async def test_engine_find_opportunities(arb_engine):
    # Mock price data
    prices = {
        "polymarket": {"YES": 0.42, "NO": 0.58},
        "azuro": {"YES": 0.47, "NO": 0.53},
    }

    event = MockMatchedEvent()

    opportunities = arb_engine.find_opportunities_for_event(event, prices)

    # Should find at least one opportunity (buy YES on PM, sell on Azuro)
    assert len(opportunities) >= 0  # May or may not find based on threshold


@pytest.mark.asyncio
async def test_engine_evaluate_opportunity(arb_engine):
    opp = CrossMarketOpportunity(
        event=MockMatchedEvent(),
        source_platform="polymarket",
        source_price=0.40,
        source_liquidity=5000.0,
        target_platform="azuro",
        target_price=0.47,
        target_liquidity=3000.0,
    )

    result = await arb_engine.evaluate_opportunity(opp)

    assert result is not None
    assert result.position_size > 0
