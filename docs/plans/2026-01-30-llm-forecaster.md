# LLM Forecaster Module — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a module that uses multiple LLMs (Claude + GPT-4) to identify mispriced markets on Polymarket by comparing market prices against calibrated probability estimates.

**Architecture:** Scanner finds low-liquidity markets → Context builder enriches with news/data → Ensemble of LLMs generates probability estimates using superforecaster methodology → Aggregator combines predictions → Edge detector identifies opportunities → Telegram alerts for trading decisions.

**Tech Stack:** Python 3.11+, Anthropic SDK (Claude), OpenAI SDK (GPT-4), httpx for async HTTP, SQLAlchemy for persistence, python-telegram-bot for alerts.

---

## Prerequisites

- Project structure from main design document already in place
- API keys: Anthropic, OpenAI, Telegram
- Virtual environment with base dependencies

---

## Task 1: Forecaster Module Setup

**Files:**
- Create: `src/forecaster/__init__.py`
- Create: `tests/forecaster/__init__.py`
- Modify: `requirements.txt` (add LLM dependencies)

**Step 1: Update requirements.txt**

Add these dependencies:

```txt
# LLM Clients
anthropic>=0.40.0
openai>=1.50.0

# Additional for forecaster
tenacity>=8.0.0
```

**Step 2: Create module init files**

```python
# src/forecaster/__init__.py
"""LLM Forecaster module for identifying mispriced markets."""
```

```python
# tests/forecaster/__init__.py
"""Tests for forecaster module."""
```

**Step 3: Verify setup**

Run: `python -c "from src.forecaster import *; print('OK')"`

Expected: `OK`

**Step 4: Commit**

```bash
git add src/forecaster/ tests/forecaster/ requirements.txt
git commit -m "feat(forecaster): add module scaffolding"
```

---

## Task 2: Market Scanner

**Files:**
- Create: `src/forecaster/scanner.py`
- Create: `tests/forecaster/test_scanner.py`

**Step 1: Write the failing test**

```python
# tests/forecaster/test_scanner.py
"""Tests for market scanner."""

import pytest
from unittest.mock import AsyncMock


@pytest.fixture
def sample_markets():
    """Sample market data from Polymarket API."""
    return [
        {
            "condition_id": "0x1234",
            "question": "Will Bitcoin reach $100K by March 2026?",
            "description": "Resolves YES if BTC price...",
            "end_date_iso": "2026-03-31T23:59:59Z",
            "tokens": [
                {"outcome": "Yes", "price": 0.45},
                {"outcome": "No", "price": 0.55},
            ],
            "liquidity": 25000,
            "volume": 5000,
        },
        {
            "condition_id": "0x5678",
            "question": "Will Trump win 2028?",
            "description": "Resolves YES if...",
            "end_date_iso": "2028-11-15T23:59:59Z",
            "tokens": [
                {"outcome": "Yes", "price": 0.38},
                {"outcome": "No", "price": 0.62},
            ],
            "liquidity": 150000,  # Too liquid
            "volume": 50000,
        },
    ]


@pytest.mark.asyncio
async def test_scanner_filters_by_liquidity(sample_markets):
    """Scanner filters markets by liquidity bounds."""
    from src.forecaster.scanner import MarketScanner

    scanner = MarketScanner(min_liquidity=1000, max_liquidity=50000)
    scanner._fetch_raw_markets = AsyncMock(return_value=sample_markets)

    markets = await scanner.scan()

    assert len(markets) == 1
    assert markets[0]["condition_id"] == "0x1234"


@pytest.mark.asyncio
async def test_scanner_extracts_yes_price(sample_markets):
    """Scanner extracts YES token price."""
    from src.forecaster.scanner import MarketScanner

    scanner = MarketScanner(max_liquidity=200000)
    scanner._fetch_raw_markets = AsyncMock(return_value=sample_markets)

    markets = await scanner.scan()

    assert markets[0]["yes_price"] == 0.45
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/forecaster/test_scanner.py -v`

Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write src/forecaster/scanner.py**

```python
"""Market scanner for Polymarket."""

import httpx
import structlog
from datetime import datetime, timezone
from typing import Any

from config.settings import settings

logger = structlog.get_logger()


class MarketScanner:
    """Scans Polymarket for eligible markets."""

    POLYMARKET_API = "https://clob.polymarket.com"

    def __init__(
        self,
        min_liquidity: float = 1000,
        max_liquidity: float = 50000,
        min_days_to_resolution: int = 7,
    ):
        self.min_liquidity = min_liquidity
        self.max_liquidity = max_liquidity
        self.min_days_to_resolution = min_days_to_resolution
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def _fetch_raw_markets(self) -> list[dict[str, Any]]:
        """Fetch raw market data from Polymarket API."""
        client = await self._get_client()
        response = await client.get(
            f"{self.POLYMARKET_API}/markets",
            params={"limit": 100, "active": "true"},
        )
        response.raise_for_status()
        return response.json()

    def _extract_yes_price(self, market: dict[str, Any]) -> float:
        """Extract YES token price from market data."""
        tokens = market.get("tokens", [])
        for token in tokens:
            if token.get("outcome", "").lower() == "yes":
                return float(token.get("price", 0.5))
        return float(tokens[0].get("price", 0.5)) if tokens else 0.5

    def _is_eligible(self, market: dict[str, Any]) -> bool:
        """Check if market meets eligibility criteria."""
        liquidity = float(market.get("liquidity", 0))

        if liquidity < self.min_liquidity or liquidity > self.max_liquidity:
            return False

        end_date_str = market.get("end_date_iso")
        if end_date_str:
            try:
                end_date = datetime.fromisoformat(
                    end_date_str.replace("Z", "+00:00")
                )
                days_remaining = (end_date - datetime.now(timezone.utc)).days
                if days_remaining < self.min_days_to_resolution:
                    return False
            except (ValueError, TypeError):
                pass

        return True

    async def scan(self) -> list[dict[str, Any]]:
        """Scan and return eligible markets."""
        raw_markets = await self._fetch_raw_markets()
        logger.info("fetched_markets", count=len(raw_markets))

        eligible = []
        for market in raw_markets:
            if not self._is_eligible(market):
                continue

            eligible.append({
                "condition_id": market.get("condition_id"),
                "question": market.get("question"),
                "description": market.get("description", ""),
                "end_date": market.get("end_date_iso"),
                "yes_price": self._extract_yes_price(market),
                "liquidity": float(market.get("liquidity", 0)),
                "volume": float(market.get("volume", 0)),
            })

        logger.info("eligible_markets", count=len(eligible))
        return eligible

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/forecaster/test_scanner.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/forecaster/scanner.py tests/forecaster/test_scanner.py
git commit -m "feat(forecaster): add market scanner with liquidity filtering"
```

---

## Task 3: Context Builder

**Files:**
- Create: `src/forecaster/context.py`
- Create: `tests/forecaster/test_context.py`

**Step 1: Write the failing test**

```python
# tests/forecaster/test_context.py
"""Tests for context builder."""

import pytest
from unittest.mock import AsyncMock


@pytest.fixture
def sample_market():
    return {
        "condition_id": "0x1234",
        "question": "Will Bitcoin reach $100K by March 2026?",
        "description": "Resolves YES if BTC exceeds $100K.",
        "end_date": "2026-03-31T23:59:59Z",
        "yes_price": 0.45,
        "liquidity": 25000,
    }


@pytest.mark.asyncio
async def test_context_includes_question(sample_market):
    """Context includes the market question."""
    from src.forecaster.context import ContextBuilder

    builder = ContextBuilder()
    context = await builder.build(sample_market)

    assert sample_market["question"] in context


@pytest.mark.asyncio
async def test_context_includes_price(sample_market):
    """Context includes current market price."""
    from src.forecaster.context import ContextBuilder

    builder = ContextBuilder()
    context = await builder.build(sample_market)

    assert "45%" in context or "0.45" in context
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/forecaster/test_context.py -v`

Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write src/forecaster/context.py**

```python
"""Context builder for enriching market data."""

import httpx
import structlog
from datetime import datetime, timezone
from typing import Any

logger = structlog.get_logger()


class ContextBuilder:
    """Builds rich context for LLM forecasting."""

    def __init__(self):
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=15.0)
        return self._client

    def _format_market_info(self, market: dict[str, Any]) -> str:
        """Format basic market information."""
        return f"""## Market Information
**Question:** {market.get('question', 'N/A')}
**Description:** {market.get('description', 'N/A')}
**Resolution Date:** {market.get('end_date', 'N/A')}
**Current YES Price:** {market.get('yes_price', 0.5):.0%}
**Liquidity:** ${market.get('liquidity', 0):,.0f}"""

    def _calculate_time_context(self, market: dict[str, Any]) -> str:
        """Calculate time-related context."""
        end_date_str = market.get("end_date")
        if not end_date_str:
            return "**Time Remaining:** Unknown"

        try:
            end_date = datetime.fromisoformat(
                end_date_str.replace("Z", "+00:00")
            )
            days = (end_date - datetime.now(timezone.utc)).days

            if days < 7:
                return f"**Time Remaining:** {days} days (IMMINENT)"
            elif days < 30:
                return f"**Time Remaining:** {days} days (SHORT-TERM)"
            elif days < 180:
                return f"**Time Remaining:** {days} days (MEDIUM-TERM)"
            else:
                return f"**Time Remaining:** {days} days (LONG-TERM)"
        except (ValueError, TypeError):
            return "**Time Remaining:** Could not parse"

    def _market_sentiment(self, market: dict[str, Any]) -> str:
        """Describe market sentiment based on price."""
        price = market.get("yes_price", 0.5)
        if price < 0.15:
            return "Market considers this UNLIKELY (< 15%)"
        elif price < 0.35:
            return "Market leans NO (15-35%)"
        elif price < 0.65:
            return "Market is UNCERTAIN (35-65%)"
        elif price < 0.85:
            return "Market leans YES (65-85%)"
        else:
            return "Market considers this LIKELY (> 85%)"

    async def build(self, market: dict[str, Any]) -> str:
        """Build complete context for a market."""
        sections = [
            self._format_market_info(market),
            f"\n## Time Context\n{self._calculate_time_context(market)}",
            f"\n## Market Sentiment\n{self._market_sentiment(market)}",
        ]
        return "\n\n".join(sections)

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/forecaster/test_context.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/forecaster/context.py tests/forecaster/test_context.py
git commit -m "feat(forecaster): add context builder"
```

---

## Task 4: Superforecaster Prompts

**Files:**
- Create: `src/forecaster/prompts.py`
- Create: `tests/forecaster/test_prompts.py`

**Step 1: Write the failing test**

```python
# tests/forecaster/test_prompts.py
"""Tests for forecaster prompts."""

import pytest


def test_prompt_includes_question():
    """Prompt includes the market question."""
    from src.forecaster.prompts import build_forecast_prompt

    prompt = build_forecast_prompt(
        question="Will Bitcoin reach $100K?",
        context="Some context",
        market_price=0.45,
    )

    assert "Will Bitcoin reach $100K?" in prompt
    assert "45%" in prompt


def test_prompt_includes_methodology():
    """Prompt includes superforecasting methodology."""
    from src.forecaster.prompts import build_forecast_prompt

    prompt = build_forecast_prompt(
        question="Test",
        context="Test",
        market_price=0.50,
    )

    assert "base rate" in prompt.lower()


def test_prompt_requests_json():
    """Prompt requests JSON output."""
    from src.forecaster.prompts import build_forecast_prompt

    prompt = build_forecast_prompt(
        question="Test",
        context="Test",
        market_price=0.50,
    )

    assert "json" in prompt.lower()
    assert "probability" in prompt.lower()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/forecaster/test_prompts.py -v`

Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write src/forecaster/prompts.py**

```python
"""Prompt templates for LLM forecasting."""

SYSTEM_PROMPT = """You are a superforecaster. Your goal is to provide well-calibrated probability estimates.

Key principles:
1. Use base rates as starting points, then adjust
2. Consider both inside view (case-specific) and outside view (reference class)
3. Be precise: say 34%, not "around 30-35%"
4. Never say 0% or 100%
5. Acknowledge uncertainty

You will be evaluated on calibration: when you say 70%, it should happen 70% of the time."""


FORECAST_TEMPLATE = """## Question
{question}

## Current Market Price
The market prices YES at {market_price:.0%}.
Note: Markets can be wrong. Form your own view first.

## Context
{context}

## Your Task

Think step by step:

1. **Base Rate**: Historical frequency of similar events?

2. **Adjustments**:
   - Factors pushing probability HIGHER?
   - Factors pushing probability LOWER?

3. **Inside vs Outside View**:
   - Inside: What does case-specific analysis suggest?
   - Outside: What does the reference class suggest?

4. **Key Uncertainties**: What would most change your estimate?

5. **Final Probability**: Synthesize into a single number (1-99%).

## Output Format

Return JSON:
```json
{{
  "reasoning": "Your 2-3 paragraph analysis",
  "base_rate": 0.XX,
  "adjustments": [
    {{"factor": "description", "direction": "up/down", "magnitude": "small/medium/large"}}
  ],
  "key_uncertainties": ["uncertainty 1", "uncertainty 2"],
  "final_probability": 0.XX,
  "confidence": "low/medium/high"
}}
```

Important:
- `final_probability` must be between 0.01 and 0.99
- `confidence` reflects certainty about your estimate, not the probability itself"""


def build_forecast_prompt(
    question: str,
    context: str,
    market_price: float,
) -> str:
    """Build the complete forecast prompt."""
    return FORECAST_TEMPLATE.format(
        question=question,
        context=context,
        market_price=market_price,
    )


def get_system_prompt() -> str:
    """Get the system prompt."""
    return SYSTEM_PROMPT
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/forecaster/test_prompts.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/forecaster/prompts.py tests/forecaster/test_prompts.py
git commit -m "feat(forecaster): add superforecaster prompt templates"
```

---

## Task 5: LLM Ensemble

**Files:**
- Create: `src/forecaster/ensemble.py`
- Create: `tests/forecaster/test_ensemble.py`

**Step 1: Write the failing test**

```python
# tests/forecaster/test_ensemble.py
"""Tests for ensemble forecaster."""

import pytest
from unittest.mock import AsyncMock, patch


@pytest.fixture
def claude_response():
    return {
        "reasoning": "Based on trends...",
        "base_rate": 0.30,
        "adjustments": [],
        "key_uncertainties": ["Regulation"],
        "final_probability": 0.45,
        "confidence": "medium",
    }


@pytest.fixture
def gpt_response():
    return {
        "reasoning": "Analyzing data...",
        "base_rate": 0.25,
        "adjustments": [],
        "key_uncertainties": ["Market"],
        "final_probability": 0.42,
        "confidence": "medium",
    }


@pytest.mark.asyncio
async def test_ensemble_aggregates(claude_response, gpt_response):
    """Ensemble combines multiple model predictions."""
    from src.forecaster.ensemble import EnsembleForecaster

    forecaster = EnsembleForecaster()

    with patch.object(forecaster, "_query_claude", new_callable=AsyncMock) as m_claude:
        with patch.object(forecaster, "_query_gpt", new_callable=AsyncMock) as m_gpt:
            m_claude.return_value = claude_response
            m_gpt.return_value = gpt_response

            result = await forecaster.forecast(
                question="Test?",
                context="Context",
                market_price=0.50,
            )

    assert "aggregated_probability" in result
    assert 0.40 <= result["aggregated_probability"] <= 0.50


@pytest.mark.asyncio
async def test_ensemble_calculates_agreement(claude_response, gpt_response):
    """Ensemble calculates inter-model agreement."""
    from src.forecaster.ensemble import EnsembleForecaster

    forecaster = EnsembleForecaster()

    with patch.object(forecaster, "_query_claude", new_callable=AsyncMock) as m_claude:
        with patch.object(forecaster, "_query_gpt", new_callable=AsyncMock) as m_gpt:
            m_claude.return_value = claude_response
            m_gpt.return_value = gpt_response

            result = await forecaster.forecast("Test?", "Context", 0.50)

    assert "agreement" in result
    assert result["agreement"] > 0.8  # 0.45 and 0.42 are close


@pytest.mark.asyncio
async def test_ensemble_handles_failure(claude_response):
    """Ensemble handles single model failure."""
    from src.forecaster.ensemble import EnsembleForecaster

    forecaster = EnsembleForecaster()

    with patch.object(forecaster, "_query_claude", new_callable=AsyncMock) as m_claude:
        with patch.object(forecaster, "_query_gpt", new_callable=AsyncMock) as m_gpt:
            m_claude.return_value = claude_response
            m_gpt.side_effect = Exception("API error")

            result = await forecaster.forecast("Test?", "Context", 0.50)

    assert result is not None
    assert "aggregated_probability" in result
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/forecaster/test_ensemble.py -v`

Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write src/forecaster/ensemble.py**

```python
"""Ensemble forecaster combining multiple LLMs."""

import asyncio
import json
import math
import structlog
from typing import Any

import anthropic
import openai

from config.settings import settings
from src.forecaster.prompts import build_forecast_prompt, get_system_prompt

logger = structlog.get_logger()


class EnsembleForecaster:
    """Combines predictions from multiple LLMs."""

    def __init__(self):
        self.claude = anthropic.AsyncAnthropic(
            api_key=settings.ANTHROPIC_API_KEY,
        )
        self.openai = openai.AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
        )
        self.weights = {"claude": 0.55, "gpt": 0.45}

    def _parse_response(self, content: str) -> dict[str, Any] | None:
        """Parse JSON from LLM response."""
        try:
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                content = content[start:end].strip()
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                content = content[start:end].strip()
            return json.loads(content)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("parse_failed", error=str(e))
            return None

    async def _query_claude(
        self, question: str, context: str, market_price: float
    ) -> dict[str, Any] | None:
        """Query Claude."""
        prompt = build_forecast_prompt(question, context, market_price)
        try:
            response = await self.claude.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                system=get_system_prompt(),
                messages=[{"role": "user", "content": prompt}],
            )
            result = self._parse_response(response.content[0].text)
            if result:
                result["model"] = "claude"
            return result
        except Exception as e:
            logger.error("claude_failed", error=str(e))
            return None

    async def _query_gpt(
        self, question: str, context: str, market_price: float
    ) -> dict[str, Any] | None:
        """Query GPT-4."""
        prompt = build_forecast_prompt(question, context, market_price)
        try:
            response = await self.openai.chat.completions.create(
                model="gpt-4-turbo",
                max_tokens=2000,
                messages=[
                    {"role": "system", "content": get_system_prompt()},
                    {"role": "user", "content": prompt},
                ],
            )
            result = self._parse_response(response.choices[0].message.content)
            if result:
                result["model"] = "gpt"
            return result
        except Exception as e:
            logger.error("gpt_failed", error=str(e))
            return None

    def _calculate_agreement(self, probabilities: list[float]) -> float:
        """Calculate agreement (0-1) between predictions."""
        if len(probabilities) < 2:
            return 1.0
        mean = sum(probabilities) / len(probabilities)
        if mean == 0:
            return 1.0
        variance = sum((p - mean) ** 2 for p in probabilities) / len(probabilities)
        cv = math.sqrt(variance) / mean
        return max(0, 1 - cv)

    def _weighted_average(
        self, values: list[float], weights: list[float]
    ) -> float:
        """Calculate weighted average."""
        total_weight = sum(weights)
        return sum(v * w for v, w in zip(values, weights)) / total_weight

    def _extremize(self, prob: float, factor: float = 1.3) -> float:
        """Push probability toward extremes when confident."""
        prob = max(0.01, min(0.99, prob))
        logit = math.log(prob / (1 - prob))
        return 1 / (1 + math.exp(-logit * factor))

    async def forecast(
        self, question: str, context: str, market_price: float
    ) -> dict[str, Any]:
        """Generate ensemble forecast."""
        results = await asyncio.gather(
            self._query_claude(question, context, market_price),
            self._query_gpt(question, context, market_price),
            return_exceptions=True,
        )

        predictions = [r for r in results if isinstance(r, dict) and r]

        if not predictions:
            return {
                "aggregated_probability": 0.5,
                "individual_predictions": [],
                "agreement": 0.0,
                "confidence": "low",
                "error": "All models failed",
            }

        probs = [p["final_probability"] for p in predictions]
        weights = [self.weights.get(p.get("model", ""), 0.5) for p in predictions]

        aggregated = self._weighted_average(probs, weights)
        agreement = self._calculate_agreement(probs)

        if agreement > 0.85 and len(predictions) >= 2:
            aggregated = self._extremize(aggregated)

        confidences = [p.get("confidence", "medium") for p in predictions]
        if all(c == "high" for c in confidences) and agreement > 0.8:
            overall = "high"
        elif any(c == "low" for c in confidences) or agreement < 0.6:
            overall = "low"
        else:
            overall = "medium"

        reasoning = "\n\n---\n\n".join(
            f"**{p.get('model', '?').upper()}:** {p.get('reasoning', 'N/A')}"
            for p in predictions
        )

        return {
            "aggregated_probability": aggregated,
            "individual_predictions": predictions,
            "agreement": agreement,
            "confidence": overall,
            "reasoning": reasoning,
        }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/forecaster/test_ensemble.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/forecaster/ensemble.py tests/forecaster/test_ensemble.py
git commit -m "feat(forecaster): add multi-LLM ensemble with aggregation"
```

---

## Task 6: Opportunity Detector

**Files:**
- Create: `src/forecaster/detector.py`
- Create: `tests/forecaster/test_detector.py`

**Step 1: Write the failing test**

```python
# tests/forecaster/test_detector.py
"""Tests for opportunity detector."""

import pytest


@pytest.fixture
def bullish_case():
    """LLM thinks market underprices YES."""
    return {
        "market": {
            "condition_id": "0x1234",
            "question": "Will X happen?",
            "yes_price": 0.35,
            "liquidity": 10000,
        },
        "forecast": {
            "aggregated_probability": 0.55,
            "agreement": 0.85,
            "confidence": "high",
            "reasoning": "Strong evidence",
        },
    }


def test_identifies_buy_yes(bullish_case):
    """Detector identifies BUY_YES opportunity."""
    from src.forecaster.detector import OpportunityDetector

    detector = OpportunityDetector(min_edge=0.05)
    result = detector.evaluate(
        bullish_case["market"],
        bullish_case["forecast"],
    )

    assert result is not None
    assert result["action"] == "BUY_YES"
    assert result["edge"] > 0.05


def test_filters_low_edge():
    """Detector filters when edge too small."""
    from src.forecaster.detector import OpportunityDetector

    detector = OpportunityDetector(min_edge=0.10)

    market = {"condition_id": "0x", "yes_price": 0.50, "liquidity": 10000}
    forecast = {"aggregated_probability": 0.52, "agreement": 0.9, "confidence": "high"}

    result = detector.evaluate(market, forecast)
    assert result is None


def test_filters_low_confidence():
    """Detector filters when confidence low."""
    from src.forecaster.detector import OpportunityDetector

    detector = OpportunityDetector(min_edge=0.05)

    market = {"condition_id": "0x", "yes_price": 0.30, "liquidity": 10000}
    forecast = {"aggregated_probability": 0.60, "agreement": 0.5, "confidence": "low"}

    result = detector.evaluate(market, forecast)
    assert result is None


def test_calculates_position_size(bullish_case):
    """Detector calculates position size."""
    from src.forecaster.detector import OpportunityDetector

    detector = OpportunityDetector(min_edge=0.05)
    result = detector.evaluate(
        bullish_case["market"],
        bullish_case["forecast"],
    )

    assert "position_fraction" in result
    assert 0 < result["position_fraction"] < 0.20
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/forecaster/test_detector.py -v`

Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write src/forecaster/detector.py**

```python
"""Opportunity detector for mispriced markets."""

import structlog
from typing import Any

from config.settings import settings

logger = structlog.get_logger()


class OpportunityDetector:
    """Detects trading opportunities from LLM forecasts."""

    def __init__(
        self,
        min_edge: float = 0.08,
        min_agreement: float = 0.70,
        kelly_fraction: float = 0.25,
    ):
        self.min_edge = min_edge
        self.min_agreement = min_agreement
        self.kelly_fraction = kelly_fraction

    def _calculate_edge(
        self, market_price: float, llm_prob: float
    ) -> tuple[float, str]:
        """Calculate edge and action."""
        if llm_prob > market_price:
            edge = (llm_prob - market_price) / market_price
            action = "BUY_YES"
        else:
            no_market = 1 - market_price
            no_llm = 1 - llm_prob
            edge = (no_llm - no_market) / no_market
            action = "BUY_NO"
        return edge, action

    def _calculate_position(
        self, edge: float, agreement: float, confidence: str
    ) -> float:
        """Calculate position size using fractional Kelly."""
        adjusted_edge = edge * agreement
        kelly = adjusted_edge
        position = kelly * self.kelly_fraction

        multiplier = {"low": 0.5, "medium": 0.75, "high": 1.0}.get(confidence, 0.75)
        position *= multiplier

        return min(max(0, position), settings.MAX_POSITION_PCT)

    def evaluate(
        self, market: dict[str, Any], forecast: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Evaluate if market is an opportunity."""
        market_price = market.get("yes_price", 0.5)
        llm_prob = forecast.get("aggregated_probability", 0.5)
        agreement = forecast.get("agreement", 0.5)
        confidence = forecast.get("confidence", "medium")

        edge, action = self._calculate_edge(market_price, llm_prob)

        if edge < self.min_edge:
            return None
        if agreement < self.min_agreement:
            return None
        if confidence == "low":
            return None

        position = self._calculate_position(edge, agreement, confidence)

        logger.info(
            "opportunity_found",
            market_id=market.get("condition_id"),
            action=action,
            edge=f"{edge:.1%}",
        )

        return {
            "market_id": market.get("condition_id"),
            "question": market.get("question", ""),
            "action": action,
            "edge": edge,
            "market_price": market_price,
            "llm_probability": llm_prob,
            "agreement": agreement,
            "confidence": confidence,
            "position_fraction": position,
            "liquidity": market.get("liquidity", 0),
            "reasoning": forecast.get("reasoning", ""),
        }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/forecaster/test_detector.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/forecaster/detector.py tests/forecaster/test_detector.py
git commit -m "feat(forecaster): add opportunity detector with Kelly sizing"
```

---

## Task 7: Telegram Alerts

**Files:**
- Modify: `src/bot/telegram.py` (add forecaster alerts)
- Create: `tests/forecaster/test_alerts.py`

**Step 1: Write the failing test**

```python
# tests/forecaster/test_alerts.py
"""Tests for forecaster Telegram alerts."""

import pytest


@pytest.fixture
def opportunity():
    return {
        "market_id": "0x1234",
        "question": "Will Bitcoin reach $100K?",
        "action": "BUY_YES",
        "edge": 0.15,
        "market_price": 0.45,
        "llm_probability": 0.60,
        "agreement": 0.88,
        "confidence": "high",
        "position_fraction": 0.05,
        "liquidity": 25000,
        "reasoning": "Strong momentum...",
    }


def test_format_forecast_alert(opportunity):
    """Alert formatter produces readable message."""
    from src.bot.telegram import format_forecast_alert

    message = format_forecast_alert(opportunity)

    assert "Bitcoin" in message
    assert "BUY" in message
    assert "15%" in message or "15.0%" in message
    assert "45%" in message
    assert "60%" in message
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/forecaster/test_alerts.py -v`

Expected: FAIL

**Step 3: Add to src/bot/telegram.py**

```python
"""Telegram bot utilities."""


def format_forecast_alert(opportunity: dict) -> str:
    """Format a forecast opportunity as Telegram message."""
    action_emoji = "🟢" if opportunity["action"] == "BUY_YES" else "🔴"
    conf_emoji = {"high": "🎯", "medium": "📊", "low": "⚠️"}.get(
        opportunity.get("confidence", "medium"), "📊"
    )

    return f"""
{action_emoji} **FORECAST OPPORTUNITY**

**Market:** {opportunity.get('question', 'Unknown')}

**Signal:** {opportunity['action'].replace('_', ' ')}
**Edge:** {opportunity['edge']:.1%}

┌─────────────┬────────────┐
│ Market Price │ {opportunity['market_price']:.0%}        │
│ LLM Estimate │ {opportunity['llm_probability']:.0%}        │
│ Agreement    │ {opportunity['agreement']:.0%}        │
└─────────────┴────────────┘

{conf_emoji} **Confidence:** {opportunity.get('confidence', 'medium').upper()}
💰 **Position:** {opportunity['position_fraction']:.1%} of capital
💧 **Liquidity:** ${opportunity.get('liquidity', 0):,.0f}

**Reasoning:**
{opportunity.get('reasoning', 'N/A')[:400]}{'...' if len(opportunity.get('reasoning', '')) > 400 else ''}
""".strip()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/forecaster/test_alerts.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/bot/telegram.py tests/forecaster/test_alerts.py
git commit -m "feat(bot): add forecast opportunity alert formatting"
```

---

## Task 8: Orchestrator

**Files:**
- Create: `src/forecaster/orchestrator.py`
- Create: `tests/forecaster/test_orchestrator.py`

**Step 1: Write the failing test**

```python
# tests/forecaster/test_orchestrator.py
"""Tests for forecaster orchestrator."""

import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def mock_scanner():
    s = AsyncMock()
    s.scan.return_value = [{
        "condition_id": "0x1",
        "question": "Test?",
        "yes_price": 0.40,
        "liquidity": 20000,
    }]
    return s


@pytest.fixture
def mock_context():
    c = AsyncMock()
    c.build.return_value = "Context"
    return c


@pytest.fixture
def mock_ensemble():
    e = AsyncMock()
    e.forecast.return_value = {
        "aggregated_probability": 0.60,
        "agreement": 0.85,
        "confidence": "high",
        "reasoning": "Test",
    }
    return e


@pytest.fixture
def mock_detector():
    d = MagicMock()
    d.evaluate.return_value = {
        "market_id": "0x1",
        "action": "BUY_YES",
        "edge": 0.15,
    }
    return d


@pytest.mark.asyncio
async def test_orchestrator_full_cycle(
    mock_scanner, mock_context, mock_ensemble, mock_detector
):
    """Orchestrator runs full cycle."""
    from src.forecaster.orchestrator import ForecasterOrchestrator

    orch = ForecasterOrchestrator(
        scanner=mock_scanner,
        context_builder=mock_context,
        ensemble=mock_ensemble,
        detector=mock_detector,
    )

    opps = await orch.run_cycle()

    mock_scanner.scan.assert_called_once()
    mock_context.build.assert_called_once()
    mock_ensemble.forecast.assert_called_once()
    mock_detector.evaluate.assert_called_once()

    assert len(opps) == 1
    assert opps[0]["action"] == "BUY_YES"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/forecaster/test_orchestrator.py -v`

Expected: FAIL

**Step 3: Write src/forecaster/orchestrator.py**

```python
"""Orchestrator for the forecasting pipeline."""

import asyncio
import structlog
from typing import Any

from src.forecaster.scanner import MarketScanner
from src.forecaster.context import ContextBuilder
from src.forecaster.ensemble import EnsembleForecaster
from src.forecaster.detector import OpportunityDetector

logger = structlog.get_logger()


class ForecasterOrchestrator:
    """Coordinates the forecasting pipeline."""

    def __init__(
        self,
        scanner: MarketScanner | None = None,
        context_builder: ContextBuilder | None = None,
        ensemble: EnsembleForecaster | None = None,
        detector: OpportunityDetector | None = None,
    ):
        self.scanner = scanner or MarketScanner()
        self.context_builder = context_builder or ContextBuilder()
        self.ensemble = ensemble or EnsembleForecaster()
        self.detector = detector or OpportunityDetector()

    async def _process_market(self, market: dict[str, Any]) -> dict[str, Any] | None:
        """Process single market."""
        market_id = market.get("condition_id", "?")

        try:
            context = await self.context_builder.build(market)

            forecast = await self.ensemble.forecast(
                question=market.get("question", ""),
                context=context,
                market_price=market.get("yes_price", 0.5),
            )

            opportunity = self.detector.evaluate(market, forecast)
            return opportunity

        except Exception as e:
            logger.error("process_failed", market_id=market_id, error=str(e))
            return None

    async def run_cycle(self) -> list[dict[str, Any]]:
        """Run one complete cycle."""
        logger.info("starting_cycle")

        markets = await self.scanner.scan()
        logger.info("scanned", count=len(markets))

        if not markets:
            return []

        opportunities = []
        for market in markets:
            opp = await self._process_market(market)
            if opp:
                opportunities.append(opp)

        logger.info("cycle_complete", found=len(opportunities))
        return opportunities

    async def run_continuous(self, interval: int = 300) -> None:
        """Run continuously."""
        logger.info("continuous_mode", interval=interval)
        while True:
            try:
                await self.run_cycle()
            except Exception as e:
                logger.error("cycle_error", error=str(e))
            await asyncio.sleep(interval)

    async def close(self) -> None:
        """Cleanup."""
        await self.scanner.close()
        await self.context_builder.close()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/forecaster/test_orchestrator.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/forecaster/orchestrator.py tests/forecaster/test_orchestrator.py
git commit -m "feat(forecaster): add orchestrator for pipeline coordination"
```

---

## Task 9: CLI Entry Point

**Files:**
- Create: `scripts/run_forecaster.py`

**Step 1: Write the script**

```python
#!/usr/bin/env python3
"""CLI entry point for LLM Forecaster."""

import asyncio
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog

from src.forecaster.orchestrator import ForecasterOrchestrator
from src.bot.telegram import format_forecast_alert


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer() if verbose else structlog.processors.JSONRenderer(),
        ],
    )


async def run_single() -> None:
    """Run single cycle."""
    orch = ForecasterOrchestrator()
    try:
        opps = await orch.run_cycle()
        print(f"\nFound {len(opps)} opportunities:")
        for opp in opps:
            print(f"\n{format_forecast_alert(opp)}")
    finally:
        await orch.close()


async def run_continuous(interval: int) -> None:
    """Run continuous loop."""
    orch = ForecasterOrchestrator()
    try:
        await orch.run_continuous(interval)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        await orch.close()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="LLM Forecaster for Polymarket")
    parser.add_argument(
        "--mode",
        choices=["single", "continuous"],
        default="single",
        help="Run mode",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Scan interval in seconds (continuous mode)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.mode == "single":
        asyncio.run(run_single())
    else:
        asyncio.run(run_continuous(args.interval))


if __name__ == "__main__":
    main()
```

**Step 2: Test CLI**

Run: `python scripts/run_forecaster.py --help`

Expected:
```
usage: run_forecaster.py [-h] [--mode {single,continuous}] [--interval INTERVAL] [-v]
```

**Step 3: Commit**

```bash
git add scripts/run_forecaster.py
git commit -m "feat(cli): add forecaster CLI entry point"
```

---

## Task 10: Integration Test

**Files:**
- Create: `tests/test_forecaster_integration.py`

**Step 1: Write integration test**

```python
# tests/test_forecaster_integration.py
"""Integration tests for forecaster pipeline."""

import pytest
from unittest.mock import AsyncMock, patch


@pytest.fixture
def polymarket_response():
    return [{
        "condition_id": "0xintegration",
        "question": "Will AI achieve AGI by 2030?",
        "description": "Resolves YES if...",
        "end_date_iso": "2030-12-31T23:59:59Z",
        "tokens": [
            {"outcome": "Yes", "price": 0.25},
            {"outcome": "No", "price": 0.75},
        ],
        "liquidity": 30000,
        "volume": 5000,
    }]


@pytest.fixture
def llm_response():
    return {
        "reasoning": "Progress faster than expected...",
        "base_rate": 0.15,
        "adjustments": [],
        "key_uncertainties": ["Definition"],
        "final_probability": 0.38,
        "confidence": "medium",
    }


@pytest.mark.asyncio
async def test_full_pipeline(polymarket_response, llm_response):
    """Full pipeline detects opportunity."""
    from src.forecaster.orchestrator import ForecasterOrchestrator
    from src.forecaster.scanner import MarketScanner
    from src.forecaster.ensemble import EnsembleForecaster

    scanner = MarketScanner(max_liquidity=100000)
    scanner._fetch_raw_markets = AsyncMock(return_value=polymarket_response)

    ensemble = EnsembleForecaster()
    with patch.object(ensemble, "_query_claude", new_callable=AsyncMock) as m_claude:
        with patch.object(ensemble, "_query_gpt", new_callable=AsyncMock) as m_gpt:
            m_claude.return_value = llm_response
            m_gpt.return_value = llm_response

            orch = ForecasterOrchestrator(scanner=scanner, ensemble=ensemble)
            opps = await orch.run_cycle()

    # Market at 25%, LLM at 38% → ~52% edge
    assert len(opps) == 1
    assert opps[0]["action"] == "BUY_YES"
    assert opps[0]["edge"] > 0.10


@pytest.mark.asyncio
async def test_no_opportunity_when_aligned(polymarket_response):
    """No opportunity when LLM agrees with market."""
    from src.forecaster.orchestrator import ForecasterOrchestrator
    from src.forecaster.scanner import MarketScanner
    from src.forecaster.ensemble import EnsembleForecaster

    scanner = MarketScanner(max_liquidity=100000)
    scanner._fetch_raw_markets = AsyncMock(return_value=polymarket_response)

    aligned = {
        "reasoning": "Market correct",
        "final_probability": 0.26,  # Close to 25%
        "confidence": "high",
    }

    ensemble = EnsembleForecaster()
    with patch.object(ensemble, "_query_claude", new_callable=AsyncMock) as m_claude:
        with patch.object(ensemble, "_query_gpt", new_callable=AsyncMock) as m_gpt:
            m_claude.return_value = aligned
            m_gpt.return_value = aligned

            orch = ForecasterOrchestrator(scanner=scanner, ensemble=ensemble)
            opps = await orch.run_cycle()

    assert len(opps) == 0
```

**Step 2: Run integration test**

Run: `pytest tests/test_forecaster_integration.py -v`

Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_forecaster_integration.py
git commit -m "test: add forecaster integration tests"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Module setup | `src/forecaster/__init__.py` |
| 2 | Market scanner | `src/forecaster/scanner.py` |
| 3 | Context builder | `src/forecaster/context.py` |
| 4 | Prompts | `src/forecaster/prompts.py` |
| 5 | LLM ensemble | `src/forecaster/ensemble.py` |
| 6 | Opportunity detector | `src/forecaster/detector.py` |
| 7 | Telegram alerts | `src/bot/telegram.py` |
| 8 | Orchestrator | `src/forecaster/orchestrator.py` |
| 9 | CLI | `scripts/run_forecaster.py` |
| 10 | Integration tests | `tests/test_forecaster_integration.py` |

**Total:** 10 commits, ~500 lines production code + tests

---

**Plan complete and saved to `docs/plans/2026-01-30-llm-forecaster.md`.**

**Two execution options:**

1. **Subagent-Driven (this session)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

2. **Parallel Session (separate)** — Open new session with executing-plans, batch execution with checkpoints

**Which approach?**
