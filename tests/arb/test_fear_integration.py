"""Integration tests for the full fear-selling pipeline.

Verifies the end-to-end flow: Gamma API -> FearMarketScanner -> FearSellingEngine
-> trade signal generation, including cluster limit enforcement.
"""

import httpx
import pytest
import respx

from src.arb.fear_engine import FearSellingEngine
from src.arb.fear_scanner import FearMarketScanner
from src.risk.manager import UnifiedRiskManager

GAMMA_API = "https://gamma-api.polymarket.com"

FEAR_EVENTS = [
    {
        "id": "evt1",
        "title": "US strikes Iran by March 31, 2026",
        "slug": "us-strikes-iran-march-2026",
        "endDate": "2026-04-01T00:00:00Z",
        "markets": [
            {
                "id": "mkt1",
                "conditionId": "0xabc",
                "question": "Will the US strike Iran by March 31, 2026?",
                "volume": "500000",
                "volumeNum": 500000,
                "liquidity": "80000",
                "liquidityNum": 80000,
                "tokens": [
                    {"token_id": "tok_yes_1", "outcome": "Yes", "price": 0.40},
                    {"token_id": "tok_no_1", "outcome": "No", "price": 0.60},
                ],
            }
        ],
    },
    {
        "id": "evt2",
        "title": "Khamenei out as Supreme Leader by June 30",
        "slug": "khamenei-out-june-2026",
        "endDate": "2026-07-01T00:00:00Z",
        "markets": [
            {
                "id": "mkt2",
                "conditionId": "0xdef",
                "question": "Will Khamenei be out as Supreme Leader by June 30?",
                "volume": "300000",
                "volumeNum": 300000,
                "liquidity": "60000",
                "liquidityNum": 60000,
                "tokens": [
                    {"token_id": "tok_yes_2", "outcome": "Yes", "price": 0.35},
                    {"token_id": "tok_no_2", "outcome": "No", "price": 0.65},
                ],
            }
        ],
    },
    {
        "id": "evt3",
        "title": "Will Taylor Swift win Grammy",
        "slug": "taylor-swift-grammy",
        "endDate": "2027-02-01T00:00:00Z",
        "markets": [
            {
                "id": "mkt3",
                "conditionId": "0xghi",
                "question": "Will Taylor Swift win Grammy?",
                "volume": "20000",
                "volumeNum": 20000,
                "liquidity": "5000",
                "liquidityNum": 5000,
                "tokens": [
                    {"token_id": "tok_yes_3", "outcome": "Yes", "price": 0.50},
                    {"token_id": "tok_no_3", "outcome": "No", "price": 0.50},
                ],
            }
        ],
    },
]


@pytest.mark.asyncio
@respx.mock
async def test_full_pipeline():
    """Full pipeline: discover -> score -> evaluate -> generate signals."""
    respx.get(f"{GAMMA_API}/events").mock(
        return_value=httpx.Response(200, json=FEAR_EVENTS)
    )
    risk_manager = UnifiedRiskManager(
        global_capital=100_000.0,
        reality_allocation_pct=0.0,
        crossmarket_allocation_pct=0.0,
        max_position_pct=0.10,
        daily_loss_limit_pct=0.05,
        fear_allocation_pct=1.0,
    )
    engine = FearSellingEngine(
        risk_manager=risk_manager, executor=None, min_fear_score=0.5
    )

    # Discover
    candidates = await engine._scanner.discover_markets(gamma_url=GAMMA_API)
    assert len(candidates) >= 2  # Iran + Khamenei

    # Evaluate
    signals = []
    for c in candidates:
        signal = engine.evaluate_candidate(c)
        if signal:
            signals.append(signal)

    assert len(signals) >= 1
    clusters = {s.cluster for s in signals}
    assert "iran" in clusters

    for s in signals:
        assert s.side == "BUY"
        assert s.outcome == "NO"
        assert s.size_usd > 0


@pytest.mark.asyncio
@respx.mock
async def test_cluster_limit_enforcement():
    """Should respect cluster correlation limits."""
    respx.get(f"{GAMMA_API}/events").mock(
        return_value=httpx.Response(200, json=FEAR_EVENTS)
    )
    risk_manager = UnifiedRiskManager(
        global_capital=50_000.0,
        reality_allocation_pct=0.0,
        crossmarket_allocation_pct=0.0,
        max_position_pct=0.10,
        daily_loss_limit_pct=0.05,
        fear_allocation_pct=1.0,
    )
    engine = FearSellingEngine(
        risk_manager=risk_manager,
        executor=None,
        max_cluster_pct=0.15,
        min_fear_score=0.5,
    )

    candidates = await engine._scanner.discover_markets(gamma_url=GAMMA_API)
    iran_candidates = [c for c in candidates if c.cluster == "iran"]

    signals = []
    for c in iran_candidates:
        signal = engine.evaluate_candidate(c)
        if signal:
            engine._cluster_exposure[c.cluster] = (
                engine._cluster_exposure.get(c.cluster, 0.0) + signal.size_usd
            )
            signals.append(signal)

    total_iran = sum(s.size_usd for s in signals)
    assert total_iran <= 50_000.0 * 0.15 + 1.0  # small float tolerance
