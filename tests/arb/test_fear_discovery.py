"""Tests for FearMarketScanner.discover_markets — Gamma API market discovery."""

import pytest
import respx
import httpx
from src.arb.fear_scanner import FearMarketScanner, FearMarketCandidate

GAMMA_API = "https://gamma-api.polymarket.com"

SAMPLE_EVENTS = [
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
                "outcomes": ["Yes", "No"],
                "outcomePrices": ["0.40", "0.60"],
                "clobTokenIds": ["tok_yes_1", "tok_no_1"],
            }
        ],
    },
    {
        "id": "evt2",
        "title": "Will Taylor Swift release new album by April",
        "slug": "taylor-swift-album",
        "endDate": "2026-05-01T00:00:00Z",
        "markets": [
            {
                "id": "mkt2",
                "conditionId": "0xdef",
                "question": "Taylor Swift album by April?",
                "volume": "30000",
                "volumeNum": 30000,
                "liquidity": "10000",
                "liquidityNum": 10000,
                "outcomes": ["Yes", "No"],
                "outcomePrices": ["0.50", "0.50"],
                "clobTokenIds": ["tok_yes_2", "tok_no_2"],
            }
        ],
    },
]


@pytest.mark.asyncio
@respx.mock
async def test_discover_fear_markets():
    respx.get(f"{GAMMA_API}/events").mock(
        return_value=httpx.Response(200, json=SAMPLE_EVENTS)
    )
    scanner = FearMarketScanner()
    candidates = await scanner.discover_markets(gamma_url=GAMMA_API)
    fear_candidates = [c for c in candidates if c.fear_score >= 0.5]
    assert len(fear_candidates) >= 1
    assert fear_candidates[0].cluster == "iran"


@pytest.mark.asyncio
@respx.mock
async def test_discover_handles_api_error():
    respx.get(f"{GAMMA_API}/events").mock(
        return_value=httpx.Response(500)
    )
    scanner = FearMarketScanner()
    candidates = await scanner.discover_markets(gamma_url=GAMMA_API)
    assert candidates == []
