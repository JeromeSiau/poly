import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.matching.llm_verifier import LLMVerifier, MatchResult


def test_match_result_creation():
    result = MatchResult(
        is_match=True,
        confidence=0.98,
        reasoning="Both events refer to the Kansas City Chiefs winning Super Bowl LIX",
    )
    assert result.is_match
    assert result.confidence == 0.98


def test_match_result_is_high_confidence():
    high = MatchResult(is_match=True, confidence=0.98, reasoning="")
    low = MatchResult(is_match=True, confidence=0.80, reasoning="")

    assert high.is_high_confidence(threshold=0.95)
    assert not low.is_high_confidence(threshold=0.95)


@pytest.mark.asyncio
async def test_llm_verifier_verify_match():
    verifier = LLMVerifier()

    mock_response = MagicMock()
    mock_response.content = [
        MagicMock(text='{"is_match": true, "confidence": 0.98, "reasoning": "Same event"}')
    ]

    with patch.object(verifier, "_client") as mock_client:
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        result = await verifier.verify_match(
            event1_name="Chiefs win Super Bowl LIX",
            event1_platform="polymarket",
            event2_name="Kansas City Chiefs to win Super Bowl LIX",
            event2_platform="overtime",
        )

        assert result.is_match
        assert result.confidence == 0.98


@pytest.mark.asyncio
async def test_llm_verifier_no_match():
    verifier = LLMVerifier()

    mock_response = MagicMock()
    mock_response.content = [
        MagicMock(text='{"is_match": false, "confidence": 0.95, "reasoning": "Different events"}')
    ]

    with patch.object(verifier, "_client") as mock_client:
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        result = await verifier.verify_match(
            event1_name="Chiefs win Super Bowl",
            event1_platform="polymarket",
            event2_name="Lakers win NBA Finals",
            event2_platform="overtime",
        )

        assert not result.is_match


@pytest.mark.asyncio
async def test_llm_verifier_handles_json_in_markdown():
    verifier = LLMVerifier()

    mock_response = MagicMock()
    mock_response.content = [
        MagicMock(text='```json\n{"is_match": true, "confidence": 0.97, "reasoning": "Same"}\n```')
    ]

    with patch.object(verifier, "_client") as mock_client:
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        result = await verifier.verify_match(
            event1_name="Test event 1",
            event1_platform="polymarket",
            event2_name="Test event 2",
            event2_platform="azuro",
        )

        assert result.is_match
        assert result.confidence == 0.97
