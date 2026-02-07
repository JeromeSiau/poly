"""Tests for LLM researcher that generates trading theses."""
import pytest
from unittest.mock import AsyncMock, patch

from src.screening.llm_researcher import LLMResearcher, ResearchResult


class TestResearchResult:
    def test_has_edge(self):
        r = ResearchResult(
            market_id="mkt-001", title="Will X happen?",
            thesis="Based on recent data, X is unlikely because...",
            recommended_side="NO", confidence=0.85,
            key_factors=["factor1", "factor2"], sources=["source1"],
        )
        assert r.has_edge

    def test_no_edge_low_confidence(self):
        r = ResearchResult(
            market_id="mkt-001", title="Test", thesis="Unclear",
            recommended_side="YES", confidence=0.4,
            key_factors=[], sources=[],
        )
        assert not r.has_edge


class TestLLMResearcher:
    def test_init(self):
        researcher = LLMResearcher()
        assert researcher.min_confidence == 0.7

    def test_build_research_prompt(self):
        researcher = LLMResearcher()
        prompt = researcher.build_research_prompt(
            title="Will Bitcoin reach $200K by end of 2026?",
            price_yes=0.15, volume_24h=500000, category="crypto",
        )
        assert "Bitcoin" in prompt
        assert "$200K" in prompt
        assert "15%" in prompt or "0.15" in prompt

    @pytest.mark.asyncio
    async def test_research_market_mocked(self):
        researcher = LLMResearcher()
        mock_response = {
            "thesis": "Based on current trends, BTC reaching $200K is unlikely.",
            "recommended_side": "NO", "confidence": 0.82,
            "key_factors": ["current price is $65K", "historical growth rates"],
            "sources": ["market data"],
        }
        with patch.object(researcher, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            result = await researcher.research_market(
                market_id="mkt-001",
                title="Will Bitcoin reach $200K by end of 2026?",
                price_yes=0.15, volume_24h=500000, category="crypto",
            )
        assert result.recommended_side == "NO"
        assert result.confidence == 0.82
        assert result.has_edge
