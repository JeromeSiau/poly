"""GPT-5-nano fear market classifier.

Second-pass LLM confirmation after keyword pre-filtering.  Takes market
titles that passed keyword scoring and asks GPT-5-nano to confirm whether
each market truly represents a fear-driven tail-risk event suitable for
contrarian NO bets.

The model is cheap ($0.05/1M input tokens) and thinking-capable, so it
handles geopolitical nuance well — inverted fear, soft geopolitics, and
ambiguous titles that keywords alone would miss or misclassify.

Uses httpx directly (no OpenAI SDK dependency).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import httpx
import structlog

logger = structlog.get_logger()

OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
MODEL = "gpt-5-nano"

_SYSTEM_PROMPT = """\
You classify Polymarket prediction-market titles for a contrarian NO-bet \
strategy that sells fear.

A market is a FEAR market if:
- It asks about a dramatic, scary, low-probability geopolitical or \
catastrophic event (war, strike, invasion, nuclear, coup, assassination, \
collapse, regime change, pandemic, etc.)
- Retail sentiment is likely to overprice YES due to fear/headlines.
- A contrarian NO bet would exploit that fear premium.

A market is NOT a fear market if:
- It asks about routine politics, elections, policy, economic indicators, \
sports, entertainment, or tech.
- The scary-sounding outcome is actually LIKELY (inverted fear — e.g. \
"Will ceasefire hold?" where YES is the calm outcome).
- It is a positive/hopeful event dressed in dramatic language.

For each title, respond with a JSON object:
{"results": [{"title": "...", "is_fear": true/false, "cluster": "...", "confidence": 0.0-1.0}]}

cluster must be one of: iran, russia_ukraine, china_taiwan, north_korea, \
us_military, middle_east, climate, pandemic, economic, other.

Respond ONLY with valid JSON, no markdown fences."""


@dataclass(slots=True)
class ClassifiedMarket:
    """Result of LLM fear classification for a single market."""

    title: str
    is_fear: bool
    cluster: str
    confidence: float


class FearClassifier:
    """Classifies market titles as fear-driven using GPT-5-nano."""

    def __init__(
        self,
        api_key: str,
        model: str = MODEL,
        min_confidence: float = 0.70,
        timeout: float = 30.0,
        batch_size: int = 20,
    ):
        self.api_key = api_key
        self.model = model
        self.min_confidence = min_confidence
        self.timeout = timeout
        self.batch_size = batch_size

    async def classify_batch(
        self, titles: list[str]
    ) -> list[ClassifiedMarket]:
        """Send a batch of titles to GPT-5-nano and return classifications.

        Titles are sent in a single prompt to minimise API calls.
        Returns only markets classified as fear with confidence >= min_confidence.
        On any error, returns an empty list (fail-open: keywords still work).
        """
        if not titles:
            return []
        if not self.api_key:
            logger.warning("fear_classifier_no_api_key")
            return []

        user_msg = "Classify these market titles:\n" + "\n".join(
            f"{i+1}. {t}" for i, t in enumerate(titles)
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        }

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    OPENAI_CHAT_URL,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "fear_classifier_api_error",
                status_code=exc.response.status_code,
                error=exc.response.text[:200],
            )
            return []
        except httpx.RequestError as exc:
            logger.warning(
                "fear_classifier_network_error",
                error_type=type(exc).__name__,
                error=str(exc) or "timeout or connection error",
            )
            return []

        return self._parse_response(data)

    def _parse_response(self, data: dict) -> list[ClassifiedMarket]:
        """Extract ClassifiedMarket list from OpenAI chat response."""
        try:
            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            results = parsed.get("results", [])
        except (KeyError, IndexError, json.JSONDecodeError) as exc:
            logger.warning("fear_classifier_parse_error", error=str(exc))
            return []

        classified: list[ClassifiedMarket] = []
        for item in results:
            try:
                cm = ClassifiedMarket(
                    title=item["title"],
                    is_fear=bool(item["is_fear"]),
                    cluster=str(item.get("cluster", "other")),
                    confidence=float(item.get("confidence", 0.0)),
                )
                if cm.is_fear and cm.confidence >= self.min_confidence:
                    classified.append(cm)
            except (KeyError, ValueError, TypeError):
                continue

        logger.info(
            "fear_classifier_result",
            input_count=len(results),
            fear_count=len(classified),
        )
        return classified
