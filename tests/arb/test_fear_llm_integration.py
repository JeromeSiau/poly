"""Tests for FearMarketScanner + FearClassifier LLM integration."""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx

from src.arb.fear_classifier import FearClassifier, ClassifiedMarket
from src.arb.fear_scanner import FearMarketScanner


def _gamma_events():
    """Build mock Gamma API events with a mix of keyword and non-keyword markets."""
    future = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
    return [
        {
            "title": "Will Russia use nuclear weapons?",
            "endDate": future,
            "markets": [
                {
                    "question": "Will Russia use nuclear weapons?",
                    "conditionId": "cond-nuclear",
                    "volumeNum": 60000,
                    "liquidityNum": 5000,
                    "tokens": [
                        {"outcome": "Yes", "price": 0.15, "token_id": "yes-1"},
                        {"outcome": "No", "price": 0.85, "token_id": "no-1"},
                    ],
                }
            ],
        },
        {
            "title": "Will diplomatic tensions escalate in South China Sea?",
            "endDate": future,
            "markets": [
                {
                    "question": "Will diplomatic tensions escalate in South China Sea?",
                    "conditionId": "cond-scs",
                    "volumeNum": 30000,
                    "liquidityNum": 4000,
                    "tokens": [
                        {"outcome": "Yes", "price": 0.25, "token_id": "yes-2"},
                        {"outcome": "No", "price": 0.75, "token_id": "no-2"},
                    ],
                }
            ],
        },
    ]


def _openai_response(results: list[dict]) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps({"results": results}),
                }
            }
        ]
    }


@pytest.mark.asyncio
@respx.mock
async def test_discover_with_classifier_confirms_and_discovers():
    """Scanner with classifier confirms keyword hits and discovers missed markets."""
    gamma_url = "https://gamma-api.polymarket.com"

    # Mock Gamma API
    respx.get(f"{gamma_url}/events").mock(
        return_value=httpx.Response(200, json=_gamma_events())
    )

    # Mock OpenAI: Pass 1 confirms the nuclear market
    # Pass 2 discovers the South China Sea market (no keyword match)
    call_count = 0

    def _openai_side_effect(request):
        nonlocal call_count
        call_count += 1
        body = json.loads(request.content)
        user_msg = body["messages"][1]["content"]

        if "nuclear" in user_msg:
            # Pass 1: confirm keyword candidate
            return httpx.Response(
                200,
                json=_openai_response(
                    [
                        {
                            "title": "Will Russia use nuclear weapons?",
                            "is_fear": True,
                            "cluster": "russia_ukraine",
                            "confidence": 0.95,
                        }
                    ]
                ),
            )
        else:
            # Pass 2: discover non-keyword market
            return httpx.Response(
                200,
                json=_openai_response(
                    [
                        {
                            "title": "Will diplomatic tensions escalate in South China Sea?",
                            "is_fear": True,
                            "cluster": "china_taiwan",
                            "confidence": 0.85,
                        }
                    ]
                ),
            )

    respx.post("https://api.openai.com/v1/chat/completions").mock(
        side_effect=_openai_side_effect
    )

    classifier = FearClassifier(api_key="test-key")
    scanner = FearMarketScanner(
        min_fear_score=0.0, classifier=classifier
    )

    candidates = await scanner.discover_markets(gamma_url=gamma_url)

    # Should have both: keyword-confirmed + LLM-discovered
    titles = {c.title for c in candidates}
    assert "Will Russia use nuclear weapons?" in titles
    assert "Will diplomatic tensions escalate in South China Sea?" in titles
    assert call_count == 2


@pytest.mark.asyncio
@respx.mock
async def test_discover_without_classifier_returns_all_price_range_markets():
    """Scanner without classifier returns all markets in price range (no keyword gate)."""
    gamma_url = "https://gamma-api.polymarket.com"

    respx.get(f"{gamma_url}/events").mock(
        return_value=httpx.Response(200, json=_gamma_events())
    )

    scanner = FearMarketScanner(min_fear_score=0.0, classifier=None)
    candidates = await scanner.discover_markets(gamma_url=gamma_url)

    # Without classifier, all markets in YES price range are returned
    titles = {c.title for c in candidates}
    assert "Will Russia use nuclear weapons?" in titles
    assert "Will diplomatic tensions escalate in South China Sea?" in titles

    # But the nuclear market should score higher due to keyword match
    scores = {c.title: c.fear_score for c in candidates}
    assert scores["Will Russia use nuclear weapons?"] > scores[
        "Will diplomatic tensions escalate in South China Sea?"
    ]


@pytest.mark.asyncio
@respx.mock
async def test_classifier_rejection_drops_keyword_candidate():
    """LLM rejecting a keyword candidate removes it from results."""
    gamma_url = "https://gamma-api.polymarket.com"
    future = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()

    # "fall" is a high-tier keyword but "Will stock market fall?" is not a
    # geopolitical fear market — LLM should reject it
    respx.get(f"{gamma_url}/events").mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "title": "Will stock market fall 10% in March?",
                    "endDate": future,
                    "markets": [
                        {
                            "question": "Will stock market fall 10% in March?",
                            "conditionId": "cond-fall",
                            "volumeNum": 50000,
                            "liquidityNum": 3000,
                            "tokens": [
                                {"outcome": "Yes", "price": 0.30, "token_id": "yes-f"},
                                {"outcome": "No", "price": 0.70, "token_id": "no-f"},
                            ],
                        }
                    ],
                }
            ],
        )
    )

    # LLM rejects — not a fear market
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json=_openai_response(
                [
                    {
                        "title": "Will stock market fall 10% in March?",
                        "is_fear": False,
                        "cluster": "economic",
                        "confidence": 0.90,
                    }
                ]
            ),
        )
    )

    classifier = FearClassifier(api_key="test-key")
    scanner = FearMarketScanner(min_fear_score=0.0, classifier=classifier)
    candidates = await scanner.discover_markets(gamma_url=gamma_url)

    # LLM rejected it, so it should be dropped
    assert len(candidates) == 0
