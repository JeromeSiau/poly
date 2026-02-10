"""Fear Market Scanner — tail-risk / geopolitical market discovery and scoring.

Identifies Polymarket markets driven by fear and tail-risk events (wars,
strikes, invasions, regime change, etc.) and scores them for NO-bet
suitability.  The core thesis: retail overprices fear — dramatic headlines
push YES above the true base rate, creating an edge on NO.

Scoring criteria:
- Title contains fear/geopolitical keywords (tiered: high, medium, low)
- YES price in the fear sweet spot (0.05-0.65, ideal 0.20-0.50)
- Sufficient volume and liquidity for entry/exit
- Time to resolution (shorter = faster payoff, but >365d penalised)

Cluster detection maps markets to geopolitical theatres so that
correlated positions can be risk-managed together.

Base-rate estimation uses annualised event frequencies per cluster,
scaled to the market's time window via exponential decay.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx
import structlog

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Fear keyword tiers — higher tier = stronger fear signal
# ---------------------------------------------------------------------------

FEAR_KEYWORDS: dict[str, list[str]] = {
    "high": [
        "strike", "invade", "nuclear", "war", "attack",
        "bomb", "collapse", "fall", "regime change",
        "die", "killed", "coup", "assassinate",
    ],
    "medium": [
        "ceasefire", "resign", "impeach", "default", "recession",
        "shutdown", "sanctions", "regime", "annex", "deploy", "mobilize",
    ],
    "low": [
        "ban", "tariff", "out by", "out as", "fired",
        "charged", "indicted", "arrested",
    ],
}

# ---------------------------------------------------------------------------
# Geopolitical cluster patterns
# ---------------------------------------------------------------------------

CLUSTER_PATTERNS: dict[str, list[str]] = {
    "iran": ["iran", "khamenei", "iranian", "tehran", "persian"],
    "russia_ukraine": [
        "russia", "ukraine", "ukraine ceasefire", "kremlin", "putin", "zelensk",
    ],
    "china_taiwan": ["china", "taiwan", "beijing", "strait"],
    "north_korea": ["north korea", "pyongyang", "kim jong"],
    "us_military": ["us strike", "us invade", "pentagon", "us troops"],
    "middle_east": ["gaza", "israel", "hamas", "hezbollah", "lebanon"],
}

# Annualised base rates per cluster (probability of *any* qualifying event
# occurring within a full year).
_ANNUAL_RATES: dict[str, float] = {
    "iran": 0.10,
    "russia_ukraine": 0.15,
    "china_taiwan": 0.05,
    "north_korea": 0.03,
    "us_military": 0.08,
    "middle_east": 0.12,
    "other": 0.10,
}


# ---------------------------------------------------------------------------
# Dataclass: a single fear-market candidate
# ---------------------------------------------------------------------------

@dataclass
class FearMarketCandidate:
    """A candidate NO bet on a fear/tail-risk market."""

    condition_id: str
    token_id: str
    title: str
    yes_price: float
    no_price: float
    estimated_no_probability: float
    edge_pct: float
    volume_24h: float
    liquidity: float
    end_date_iso: str
    fear_score: float
    cluster: str

    @property
    def expected_return(self) -> float:
        """Expected value per dollar invested in NO."""
        win_payout = 1.0 - self.no_price
        loss = self.no_price
        p_no = self.estimated_no_probability
        return p_no * win_payout - (1 - p_no) * loss

    @property
    def is_valid(self) -> bool:
        """Minimum viability checks for placing a NO bet."""
        return (
            self.edge_pct > 0.01
            and self.liquidity >= 1000.0
            and self.volume_24h >= 5000.0
            and 0 < self.no_price < 1
        )


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

class FearMarketScanner:
    """Scans Polymarket for contrarian NO bets on fear-driven markets."""

    def __init__(
        self,
        min_yes_price: float = 0.05,
        max_yes_price: float = 0.65,
        min_liquidity: float = 1000.0,
        min_volume_24h: float = 5000.0,
        min_fear_score: float = 0.5,
    ):
        self.min_yes_price = min_yes_price
        self.max_yes_price = max_yes_price
        self.min_liquidity = min_liquidity
        self.min_volume_24h = min_volume_24h
        self.min_fear_score = min_fear_score

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_market(
        self,
        title: str,
        yes_price: float,
        volume_24h: float,
        end_date_days: int,
    ) -> float:
        """Score a market for fear-driven NO bet suitability (0.0 to 1.0).

        Higher scores indicate a stronger fear-bias signal and better
        NO-bet opportunity.
        """
        score = 0.15  # base score

        # --- Keyword tier bonus -----------------------------------------
        title_lower = title.lower()
        if any(kw in title_lower for kw in FEAR_KEYWORDS["high"]):
            score += 0.35
        elif any(kw in title_lower for kw in FEAR_KEYWORDS["medium"]):
            score += 0.20
        elif any(kw in title_lower for kw in FEAR_KEYWORDS["low"]):
            score += 0.10

        # --- YES price sweet spot ---------------------------------------
        if 0.20 <= yes_price <= 0.50:
            score += 0.20
        elif 0.10 <= yes_price <= 0.60:
            score += 0.10

        # --- Volume bonus -----------------------------------------------
        if volume_24h > 100_000:
            score += 0.15
        elif volume_24h > 50_000:
            score += 0.10
        elif volume_24h > 10_000:
            score += 0.05

        # --- Deadline bonus ---------------------------------------------
        if end_date_days <= 14:
            score += 0.15
        elif end_date_days <= 30:
            score += 0.10
        elif end_date_days <= 90:
            score += 0.05
        elif end_date_days > 365:
            score -= 0.10

        return max(0.0, min(1.0, score))

    # ------------------------------------------------------------------
    # Cluster detection
    # ------------------------------------------------------------------

    def detect_cluster(self, title: str) -> str:
        """Match *title* against CLUSTER_PATTERNS and return cluster name.

        Returns ``"other"`` when no pattern matches.
        """
        title_lower = title.lower()
        for cluster, keywords in CLUSTER_PATTERNS.items():
            if any(kw in title_lower for kw in keywords):
                return cluster
        return "other"

    # ------------------------------------------------------------------
    # Base-rate estimation
    # ------------------------------------------------------------------

    def estimate_base_rate(
        self,
        end_date_days: int,
        cluster: str,
    ) -> float:
        """Estimate the true probability of YES using annualised cluster rates.

        Uses exponential decay to convert an annual rate to the market's
        time window::

            daily_rate  = 1 - (1 - annual_rate) ^ (1/365)
            probability = 1 - (1 - daily_rate) ^ end_date_days

        Result is clamped to [0.01, 0.50].
        """
        annual_rate = _ANNUAL_RATES.get(cluster, _ANNUAL_RATES["other"])
        daily_rate = 1.0 - (1.0 - annual_rate) ** (1.0 / 365.0)
        prob = 1.0 - (1.0 - daily_rate) ** end_date_days
        return max(0.01, min(0.50, prob))

    # ------------------------------------------------------------------
    # Candidate filtering
    # ------------------------------------------------------------------

    def filter_candidates(
        self, markets: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Filter markets whose YES price falls within the configured range."""
        candidates = []
        for market in markets:
            tokens = market.get("tokens", [])
            yes_price = None
            for token in tokens:
                if token.get("outcome") == "Yes":
                    yes_price = token.get("price", 1.0)
                    break

            if (
                yes_price is not None
                and self.min_yes_price <= yes_price <= self.max_yes_price
            ):
                candidates.append(market)

        return candidates

    # ------------------------------------------------------------------
    # Gamma API market discovery
    # ------------------------------------------------------------------

    async def discover_markets(
        self,
        gamma_url: str = "https://gamma-api.polymarket.com",
        limit: int = 100,
    ) -> list[FearMarketCandidate]:
        """Fetch active events from the Gamma API and build scored candidates.

        Queries ``{gamma_url}/events`` for active, non-closed events, then
        iterates each event's markets to extract YES/NO prices and token IDs.
        Markets are filtered by YES price range, scored, clustered, and
        enriched with a base-rate edge estimate.

        Returns candidates sorted by *fear_score* descending.  On any API
        error the method logs a warning and returns an empty list.
        """
        now = datetime.now(timezone.utc)

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{gamma_url}/events",
                    params={"limit": limit, "active": "true", "closed": "false"},
                )
                resp.raise_for_status()
                events = resp.json()
        except (httpx.HTTPStatusError, httpx.RequestError) as exc:
            logger.warning("gamma_api_error", error=str(exc))
            return []

        candidates: list[FearMarketCandidate] = []

        for event in events:
            title = event.get("title", "")
            end_date_str = event.get("endDate", "")

            # Compute days to resolution
            try:
                end_dt = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
                end_date_days = max(1, (end_dt - now).days)
            except (ValueError, TypeError):
                end_date_days = 90  # fallback

            for market in event.get("markets", []):
                tokens = market.get("tokens", [])

                # Extract YES / NO prices and token IDs
                yes_price: float | None = None
                no_price: float | None = None
                no_token_id: str = ""
                for token in tokens:
                    outcome = token.get("outcome", "")
                    if outcome == "Yes":
                        yes_price = float(token.get("price", 1.0))
                    elif outcome == "No":
                        no_price = float(token.get("price", 0.0))
                        no_token_id = token.get("token_id", "")

                if yes_price is None or no_price is None:
                    continue

                # Filter by YES price range
                if not (self.min_yes_price <= yes_price <= self.max_yes_price):
                    continue

                question = market.get("question", title)
                volume_24h = float(market.get("volumeNum", market.get("volume", 0)))
                liquidity = float(market.get("liquidityNum", market.get("liquidity", 0)))
                condition_id = market.get("conditionId", market.get("id", ""))

                # Scoring, clustering, base-rate
                fear_score = self.score_market(question, yes_price, volume_24h, end_date_days)
                cluster = self.detect_cluster(question)
                estimated_yes_prob = self.estimate_base_rate(end_date_days, cluster)
                estimated_no_prob = 1.0 - estimated_yes_prob
                edge = estimated_no_prob - no_price

                candidates.append(
                    FearMarketCandidate(
                        condition_id=condition_id,
                        token_id=no_token_id,
                        title=question,
                        yes_price=yes_price,
                        no_price=no_price,
                        estimated_no_probability=estimated_no_prob,
                        edge_pct=edge,
                        volume_24h=volume_24h,
                        liquidity=liquidity,
                        end_date_iso=end_date_str,
                        fear_score=fear_score,
                        cluster=cluster,
                    )
                )

        # Sort by fear_score descending
        candidates.sort(key=lambda c: c.fear_score, reverse=True)
        return candidates
