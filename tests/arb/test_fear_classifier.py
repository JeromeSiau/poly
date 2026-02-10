"""Tests for the GPT-5-nano FearClassifier."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from src.arb.fear_classifier import (
    OPENAI_CHAT_URL,
    ClassifiedMarket,
    FearClassifier,
)


def _openai_response(results: list[dict]) -> dict:
    """Build a mock OpenAI chat completion response."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": json.dumps({"results": results}),
                },
                "finish_reason": "stop",
            }
        ],
    }


@pytest.mark.asyncio
@respx.mock
async def test_classify_batch_confirms_fear_markets():
    """LLM confirms fear markets and returns ClassifiedMarket objects."""
    respx.post(OPENAI_CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            json=_openai_response(
                [
                    {
                        "title": "Will Russia use nuclear weapons in 2025?",
                        "is_fear": True,
                        "cluster": "russia_ukraine",
                        "confidence": 0.95,
                    },
                    {
                        "title": "Will Bitcoin reach $100k?",
                        "is_fear": False,
                        "cluster": "other",
                        "confidence": 0.90,
                    },
                ]
            ),
        )
    )

    classifier = FearClassifier(api_key="test-key")
    results = await classifier.classify_batch(
        ["Will Russia use nuclear weapons in 2025?", "Will Bitcoin reach $100k?"]
    )

    assert len(results) == 1
    assert results[0].title == "Will Russia use nuclear weapons in 2025?"
    assert results[0].is_fear is True
    assert results[0].cluster == "russia_ukraine"
    assert results[0].confidence == 0.95


@pytest.mark.asyncio
@respx.mock
async def test_classify_batch_filters_low_confidence():
    """Markets below min_confidence are excluded."""
    respx.post(OPENAI_CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            json=_openai_response(
                [
                    {
                        "title": "Will there be a trade war?",
                        "is_fear": True,
                        "cluster": "economic",
                        "confidence": 0.50,
                    },
                ]
            ),
        )
    )

    classifier = FearClassifier(api_key="test-key", min_confidence=0.70)
    results = await classifier.classify_batch(["Will there be a trade war?"])

    assert len(results) == 0


@pytest.mark.asyncio
@respx.mock
async def test_classify_batch_api_error_returns_empty():
    """API errors fail-open: return empty list."""
    respx.post(OPENAI_CHAT_URL).mock(
        return_value=httpx.Response(500, json={"error": "server error"})
    )

    classifier = FearClassifier(api_key="test-key")
    results = await classifier.classify_batch(["Will Iran strike Israel?"])

    assert results == []


@pytest.mark.asyncio
async def test_classify_batch_no_api_key_returns_empty():
    """No API key configured — returns empty list."""
    classifier = FearClassifier(api_key="")
    results = await classifier.classify_batch(["Will Iran strike Israel?"])

    assert results == []


@pytest.mark.asyncio
async def test_classify_batch_empty_titles_returns_empty():
    """Empty input returns empty list without API call."""
    classifier = FearClassifier(api_key="test-key")
    results = await classifier.classify_batch([])

    assert results == []


@pytest.mark.asyncio
@respx.mock
async def test_classify_batch_malformed_json_returns_empty():
    """Malformed JSON response returns empty list."""
    respx.post(OPENAI_CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "not valid json {{{",
                        }
                    }
                ]
            },
        )
    )

    classifier = FearClassifier(api_key="test-key")
    results = await classifier.classify_batch(["Will Iran attack?"])

    assert results == []


@pytest.mark.asyncio
@respx.mock
async def test_classify_batch_multiple_fear_markets():
    """Multiple fear markets returned correctly."""
    respx.post(OPENAI_CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            json=_openai_response(
                [
                    {
                        "title": "Will Iran strike Israel?",
                        "is_fear": True,
                        "cluster": "middle_east",
                        "confidence": 0.92,
                    },
                    {
                        "title": "Will China invade Taiwan by 2026?",
                        "is_fear": True,
                        "cluster": "china_taiwan",
                        "confidence": 0.88,
                    },
                    {
                        "title": "Will NATO expand further?",
                        "is_fear": True,
                        "cluster": "other",
                        "confidence": 0.60,
                    },
                ]
            ),
        )
    )

    classifier = FearClassifier(api_key="test-key", min_confidence=0.70)
    results = await classifier.classify_batch(
        [
            "Will Iran strike Israel?",
            "Will China invade Taiwan by 2026?",
            "Will NATO expand further?",
        ]
    )

    assert len(results) == 2
    assert results[0].cluster == "middle_east"
    assert results[1].cluster == "china_taiwan"
