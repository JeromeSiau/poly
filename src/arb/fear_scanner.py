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
from typing import Any

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
