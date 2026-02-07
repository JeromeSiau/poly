"""LLM Research Agent for generating trading theses on Polymarket markets.

Uses Claude (or Perplexity via API) to perform deep research on screened
markets and generate actionable trading recommendations.
"""

import json
from dataclasses import dataclass
from typing import Any, Optional

import httpx
import structlog

from config.settings import settings

logger = structlog.get_logger()


@dataclass
class ResearchResult:
    market_id: str
    title: str
    thesis: str
    recommended_side: str  # "YES", "NO", or "SKIP"
    confidence: float
    key_factors: list[str]
    sources: list[str]

    @property
    def has_edge(self) -> bool:
        return self.confidence >= 0.7 and self.recommended_side != "SKIP"


class LLMResearcher:
    def __init__(self, min_confidence: float = 0.7):
        self.min_confidence = min_confidence

    def build_research_prompt(
        self, title: str, price_yes: float, volume_24h: float, category: str,
    ) -> str:
        return f"""Analyze this Polymarket prediction market and provide a trading recommendation.

Market: "{title}"
Category: {category}
Current YES price: {price_yes:.2f} ({price_yes*100:.0f}% implied probability)
24h Volume: ${volume_24h:,.0f}

Research the topic thoroughly and respond with a JSON object:
{{
    "thesis": "Your detailed analysis (2-3 paragraphs)",
    "recommended_side": "YES" or "NO" or "SKIP",
    "confidence": 0.0 to 1.0,
    "key_factors": ["factor 1", "factor 2", ...],
    "sources": ["relevant source 1", ...]
}}

Consider:
1. What is the base rate for this type of event?
2. Are there any recent developments that change the probability?
3. Is the current market price justified by available evidence?
4. What information asymmetry might exist?

Be conservative. Only recommend YES or NO if you have strong evidence
that the market is mispriced by at least 10%.
Respond ONLY with the JSON object."""

    async def _call_llm(self, prompt: str) -> dict[str, Any]:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": settings.ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": settings.LLM_MODEL,
                    "max_tokens": 2000,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()
            text = data["content"][0]["text"]
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                # LLM may wrap JSON in markdown fences
                import re
                match = re.search(r"\{.*\}", text, re.DOTALL)
                if match:
                    return json.loads(match.group())
                logger.warning("llm_invalid_json", text=text[:200])
                return {"thesis": "", "recommended_side": "SKIP", "confidence": 0.0, "key_factors": [], "sources": []}

    async def research_market(
        self, market_id: str, title: str, price_yes: float,
        volume_24h: float, category: str,
    ) -> ResearchResult:
        prompt = self.build_research_prompt(title, price_yes, volume_24h, category)
        result = await self._call_llm(prompt)
        return ResearchResult(
            market_id=market_id, title=title,
            thesis=result.get("thesis", ""),
            recommended_side=result.get("recommended_side", "SKIP"),
            confidence=result.get("confidence", 0.0),
            key_factors=result.get("key_factors", []),
            sources=result.get("sources", []),
        )
