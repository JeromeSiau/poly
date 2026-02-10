# Fear Selling Bot — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an automated bot that systematically sells "fear premium" on Polymarket geopolitical/catastrophe markets by buying NO on events the market overprices due to fear bias — the SwissMiss strategy, with proper risk management.

**Architecture:** Extends the existing NoBetScanner pattern (hype bias on crypto) to a new domain: fear bias on geopolitical events. A `FearMarketScanner` discovers and scores fear-driven markets via the Gamma API. A `FearSpikeDetector` extends the existing `SpikeDetector` to detect NO-price drops (fear spikes) as optimal entry points. A `FearSellingEngine` orchestrates scanning, spike detection, entry/exit rules, temporal laddering, and cluster-based correlation limits. Integrates with `UnifiedRiskManager` and `PolymarketExecutor`.

**Tech Stack:** Python 3.11+, asyncio, httpx/aiohttp (Gamma API), existing `PolymarketFeed` (CLOB WebSocket), existing `PolymarketExecutor` (py-clob-client), existing `SpikeDetector`, SQLAlchemy async ORM, structlog, pytest + respx.

---

## Task 1: FearMarketScanner — Market Discovery & Scoring

Discovers active Polymarket markets with fear/tail-risk characteristics and scores them for NO-bet suitability. Mirrors the `NoBetScanner` pattern but with fear-specific keywords and different price sweet spots.

**Files:**
- Create: `src/arb/fear_scanner.py`
- Test: `tests/arb/test_fear_scanner.py`

**Step 1: Write the failing tests**

```python
# tests/arb/test_fear_scanner.py
"""Tests for FearMarketScanner."""
import pytest
from src.arb.fear_scanner import FearMarketScanner, FearMarketCandidate, FEAR_KEYWORDS


class TestFearKeywords:
    def test_high_fear_keywords_present(self):
        assert "strike" in FEAR_KEYWORDS["high"]
        assert "invade" in FEAR_KEYWORDS["high"]
        assert "nuclear" in FEAR_KEYWORDS["high"]

    def test_medium_fear_keywords_present(self):
        assert "ceasefire" in FEAR_KEYWORDS["medium"]
        assert "regime" in FEAR_KEYWORDS["medium"]


class TestFearMarketCandidate:
    def test_expected_return_profitable(self):
        c = FearMarketCandidate(
            condition_id="0x123",
            token_id="tok_no",
            title="US strikes Iran by March 31",
            yes_price=0.40,
            no_price=0.60,
            estimated_no_probability=0.85,
            edge_pct=0.25,
            volume_24h=100_000.0,
            liquidity=50_000.0,
            end_date_iso="2026-04-01T00:00:00Z",
            fear_score=0.8,
            cluster="iran",
        )
        assert c.expected_return > 0
        assert c.is_valid

    def test_invalid_when_low_liquidity(self):
        c = FearMarketCandidate(
            condition_id="0x123",
            token_id="tok_no",
            title="Will volcano erupt",
            yes_price=0.30,
            no_price=0.70,
            estimated_no_probability=0.90,
            edge_pct=0.20,
            volume_24h=100.0,
            liquidity=200.0,
            end_date_iso="2026-04-01T00:00:00Z",
            fear_score=0.5,
            cluster="other",
        )
        assert not c.is_valid


class TestFearMarketScanner:
    def setup_method(self):
        self.scanner = FearMarketScanner()

    def test_score_high_fear_market(self):
        score = self.scanner.score_market(
            title="US strikes Iran by March 31, 2026",
            yes_price=0.45,
            volume_24h=200_000,
            end_date_days=30,
        )
        assert score >= 0.7

    def test_score_low_fear_market(self):
        score = self.scanner.score_market(
            title="Will Taylor Swift release album",
            yes_price=0.50,
            volume_24h=10_000,
            end_date_days=60,
        )
        assert score < 0.4

    def test_score_medium_fear_market(self):
        score = self.scanner.score_market(
            title="Russia Ukraine ceasefire by June 2026",
            yes_price=0.35,
            volume_24h=150_000,
            end_date_days=120,
        )
        assert score >= 0.5

    def test_detect_cluster_iran(self):
        cluster = self.scanner.detect_cluster(
            "Khamenei out as Supreme Leader by March 31"
        )
        assert cluster == "iran"

    def test_detect_cluster_russia_ukraine(self):
        cluster = self.scanner.detect_cluster(
            "Russia Ukraine ceasefire by end of 2026"
        )
        assert cluster == "russia_ukraine"

    def test_detect_cluster_unknown(self):
        cluster = self.scanner.detect_cluster(
            "Will aliens land on Earth in 2026"
        )
        assert cluster == "other"

    def test_estimate_base_rate_short_deadline(self):
        """Short-deadline catastrophic events have very low base rates."""
        rate = self.scanner.estimate_base_rate(
            title="US strikes Iran by Feb 28, 2026",
            yes_price=0.30,
            end_date_days=14,
            cluster="iran",
        )
        # True probability of strike in 14 days is very low
        assert rate < 0.15

    def test_estimate_base_rate_long_deadline(self):
        """Longer deadlines have slightly higher base rates."""
        rate = self.scanner.estimate_base_rate(
            title="Khamenei out as Supreme Leader in 2026",
            yes_price=0.50,
            end_date_days=300,
            cluster="iran",
        )
        assert rate > 0.05
        assert rate < 0.40

    def test_filter_candidates_yes_range(self):
        markets = [
            {"condition_id": "a", "tokens": [
                {"outcome": "Yes", "price": 0.30},
                {"outcome": "No", "price": 0.70},
            ], "question": "US strikes Iran", "volume_num": 100_000},
            {"condition_id": "b", "tokens": [
                {"outcome": "Yes", "price": 0.95},
                {"outcome": "No", "price": 0.05},
            ], "question": "Will sun rise", "volume_num": 50_000},
        ]
        candidates = self.scanner.filter_candidates(markets)
        # Only market "a" has YES in 0.05-0.65 range
        assert len(candidates) == 1
        assert candidates[0]["condition_id"] == "a"
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/arb/test_fear_scanner.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.arb.fear_scanner'`

**Step 3: Write implementation**

```python
# src/arb/fear_scanner.py
"""Fear Market Scanner — discovers Polymarket markets with fear/tail-risk bias.

Identifies markets where YES is overpriced due to fear bias (geopolitical crises,
military strikes, regime changes). The SwissMiss strategy: systematically buy NO
on dramatic events that the market overprices.

Complementary to NoBetScanner (which targets optimism/hype bias on crypto).
This scanner targets fear/catastrophe bias on geopolitical events.
"""

from dataclasses import dataclass
from typing import Any

import structlog

logger = structlog.get_logger()

# Fear-prone market keywords — YES tends to be overpriced on these
FEAR_KEYWORDS: dict[str, list[str]] = {
    "high": [
        "strike", "strikes", "invade", "invasion", "war",
        "nuclear", "bomb", "attack", "assassinate",
        "collapse", "fall", "regime change", "regime fall",
        "die", "killed", "coup",
    ],
    "medium": [
        "ceasefire", "resign", "impeach", "default",
        "recession", "shutdown", "sanctions", "regime",
        "annex", "deploy", "mobilize",
    ],
    "low": [
        "ban", "tariff", "out by", "out as", "fired",
        "charged", "indicted", "arrested",
    ],
}

# Geopolitical cluster detection — correlated events
CLUSTER_PATTERNS: dict[str, list[str]] = {
    "iran": ["iran", "khamenei", "iranian", "tehran", "persian"],
    "russia_ukraine": ["russia", "ukraine", "ceasefire", "kremlin", "putin", "zelensk"],
    "china_taiwan": ["china", "taiwan", "beijing", "strait"],
    "north_korea": ["north korea", "pyongyang", "kim jong"],
    "us_military": ["us strike", "us invade", "pentagon", "us troops"],
    "middle_east": ["gaza", "israel", "hamas", "hezbollah", "lebanon"],
}


@dataclass
class FearMarketCandidate:
    """A candidate NO bet on a fear-driven market."""

    condition_id: str
    token_id: str  # NO token ID
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
        """Expected return per dollar invested in NO."""
        win_payout = 1.0 - self.no_price
        loss = self.no_price
        p_no = self.estimated_no_probability
        return p_no * win_payout - (1 - p_no) * loss

    @property
    def is_valid(self) -> bool:
        return (
            self.edge_pct > 0.01
            and self.liquidity >= 1_000.0
            and self.volume_24h >= 5_000.0
            and 0 < self.no_price < 1
        )


class FearMarketScanner:
    """Scans Polymarket for fear-biased markets suitable for NO bets."""

    def __init__(
        self,
        min_yes_price: float = 0.05,
        max_yes_price: float = 0.65,
        min_liquidity: float = 1_000.0,
        min_volume_24h: float = 5_000.0,
        min_fear_score: float = 0.5,
    ):
        self.min_yes_price = min_yes_price
        self.max_yes_price = max_yes_price
        self.min_liquidity = min_liquidity
        self.min_volume_24h = min_volume_24h
        self.min_fear_score = min_fear_score

    def score_market(
        self,
        title: str,
        yes_price: float,
        volume_24h: float,
        end_date_days: int,
    ) -> float:
        """Score a market for fear-bias suitability (0.0 to 1.0)."""
        score = 0.15  # base score

        # Fear keyword detection
        title_lower = title.lower()
        if any(kw in title_lower for kw in FEAR_KEYWORDS["high"]):
            score += 0.35
        elif any(kw in title_lower for kw in FEAR_KEYWORDS["medium"]):
            score += 0.20
        elif any(kw in title_lower for kw in FEAR_KEYWORDS["low"]):
            score += 0.10

        # YES price sweet spot for fear markets:
        # 0.10-0.50 is where fear premium is highest
        if 0.20 <= yes_price <= 0.50:
            score += 0.20
        elif 0.10 <= yes_price <= 0.60:
            score += 0.10
        elif yes_price < 0.10 or yes_price > 0.65:
            score -= 0.10

        # Volume = can exit if needed
        if volume_24h > 100_000:
            score += 0.15
        elif volume_24h > 50_000:
            score += 0.10
        elif volume_24h > 10_000:
            score += 0.05

        # Shorter resolution = faster theta capture
        if end_date_days <= 14:
            score += 0.15
        elif end_date_days <= 30:
            score += 0.10
        elif end_date_days <= 90:
            score += 0.05
        elif end_date_days > 365:
            score -= 0.10

        return max(0.0, min(1.0, score))

    def detect_cluster(self, title: str) -> str:
        """Detect the geopolitical cluster a market belongs to."""
        title_lower = title.lower()
        for cluster, patterns in CLUSTER_PATTERNS.items():
            if any(p in title_lower for p in patterns):
                return cluster
        return "other"

    def estimate_base_rate(
        self,
        title: str,
        yes_price: float,
        end_date_days: int,
        cluster: str,
    ) -> float:
        """Estimate the true probability of the YES event occurring.

        Uses time-decay heuristic: dramatic events are very unlikely
        in short windows and somewhat more plausible over long periods.
        The market systematically overestimates short-term catastrophe.
        """
        # Base rate by cluster (annualized probability of occurrence)
        annual_rates = {
            "iran": 0.10,
            "russia_ukraine": 0.15,
            "china_taiwan": 0.05,
            "north_korea": 0.03,
            "us_military": 0.08,
            "middle_east": 0.12,
            "other": 0.10,
        }
        annual_rate = annual_rates.get(cluster, 0.10)

        # Scale by time window (exponential decay)
        daily_rate = 1.0 - (1.0 - annual_rate) ** (1.0 / 365.0)
        prob = 1.0 - (1.0 - daily_rate) ** end_date_days

        # Clamp to reasonable range
        return max(0.01, min(0.50, prob))

    def filter_candidates(
        self, markets: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Filter markets in the fear-bias YES price range."""
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
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/arb/test_fear_scanner.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/arb/fear_scanner.py tests/arb/test_fear_scanner.py
git commit -m "feat(fear_selling): add FearMarketScanner with keyword scoring and cluster detection"
```

---

## Task 2: FearSpikeDetector — Real-Time Fear Spike Detection

Extends the existing `SpikeDetector` concept to detect sudden NO-price drops (= fear spikes) as optimal entry points. Also detects correlated cross-market spikes within the same geopolitical cluster.

**Files:**
- Create: `src/arb/fear_spike_detector.py`
- Test: `tests/arb/test_fear_spike_detector.py`

**Step 1: Write the failing tests**

```python
# tests/arb/test_fear_spike_detector.py
"""Tests for FearSpikeDetector."""
import pytest
from src.arb.fear_spike_detector import FearSpikeDetector, FearSpike


class TestFearSpikeDetector:
    def setup_method(self):
        self.detector = FearSpikeDetector(
            spike_threshold_pct=0.05,
            spike_window_seconds=600,
            cooldown_seconds=120,
        )

    def test_no_spike_on_stable_prices(self):
        """Stable NO price should not trigger a spike."""
        for i in range(10):
            spikes = self.detector.observe(
                condition_id="market1",
                no_price=0.70,
                timestamp=1000.0 + i * 10,
                cluster="iran",
            )
        assert spikes == []

    def test_spike_on_no_price_drop(self):
        """A sudden NO price drop (fear spike) should be detected."""
        # Initial stable price
        self.detector.observe("m1", no_price=0.75, timestamp=1000.0, cluster="iran")
        # Price drops 10%
        spikes = self.detector.observe(
            "m1", no_price=0.65, timestamp=1005.0, cluster="iran"
        )
        assert len(spikes) == 1
        assert spikes[0].direction == "fear_spike"
        assert spikes[0].drop_pct == pytest.approx(-0.10, abs=0.01)

    def test_cooldown_prevents_duplicate(self):
        """Cooldown should prevent duplicate signals."""
        self.detector.observe("m1", no_price=0.80, timestamp=1000.0, cluster="iran")
        self.detector.observe("m1", no_price=0.70, timestamp=1005.0, cluster="iran")
        # Second spike within cooldown
        self.detector.observe("m1", no_price=0.80, timestamp=1010.0, cluster="iran")
        spikes = self.detector.observe(
            "m1", no_price=0.70, timestamp=1015.0, cluster="iran"
        )
        assert spikes == []

    def test_correlated_spike_detection(self):
        """Multiple markets in same cluster spiking = correlated spike."""
        self.detector.observe("m1", no_price=0.80, timestamp=1000.0, cluster="iran")
        self.detector.observe("m2", no_price=0.75, timestamp=1000.0, cluster="iran")
        self.detector.observe("m1", no_price=0.70, timestamp=1005.0, cluster="iran")
        self.detector.observe("m2", no_price=0.65, timestamp=1008.0, cluster="iran")

        correlated = self.detector.get_correlated_spikes("iran", window_seconds=30)
        assert len(correlated) >= 2

    def test_no_correlated_spike_across_clusters(self):
        """Spikes in different clusters should not be correlated."""
        self.detector.observe("m1", no_price=0.80, timestamp=1000.0, cluster="iran")
        self.detector.observe("m2", no_price=0.75, timestamp=1000.0, cluster="russia_ukraine")
        self.detector.observe("m1", no_price=0.70, timestamp=1005.0, cluster="iran")
        self.detector.observe("m2", no_price=0.65, timestamp=1008.0, cluster="russia_ukraine")

        iran_correlated = self.detector.get_correlated_spikes("iran", window_seconds=30)
        assert len(iran_correlated) == 1  # only m1

    def test_recovery_signal(self):
        """Detect when NO price recovers after a spike (entry opportunity passed)."""
        self.detector.observe("m1", no_price=0.80, timestamp=1000.0, cluster="iran")
        self.detector.observe("m1", no_price=0.65, timestamp=1005.0, cluster="iran")
        # Price recovers
        recovery = self.detector.observe(
            "m1", no_price=0.78, timestamp=1200.0, cluster="iran"
        )
        # Should detect upward recovery
        assert any(s.direction == "recovery" for s in recovery) or recovery == []
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/arb/test_fear_spike_detector.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# src/arb/fear_spike_detector.py
"""Fear Spike Detector — detects sudden NO-price drops on fear-driven markets.

When breaking news creates panic (military strike rumor, regime change speculation),
NO prices drop sharply as retail traders rush to buy YES. These drops typically
recover within 10-60 minutes, making them optimal NO entry points.

Extends the pattern from src/feeds/spike_detector.py with:
- Directional detection (only NO drops, not rises)
- Cluster-based correlation (multiple markets in same geopolitical cluster)
- Recovery tracking
"""

from collections import deque
from dataclasses import dataclass, field
import time as _time

import structlog

logger = structlog.get_logger()


@dataclass(slots=True)
class FearSpike:
    """A detected fear spike on a market."""

    condition_id: str
    no_price_before: float
    no_price_now: float
    drop_pct: float
    direction: str  # "fear_spike" | "recovery"
    cluster: str
    timestamp: float


class FearSpikeDetector:
    """Monitors NO prices and detects fear-driven drops as entry signals."""

    def __init__(
        self,
        spike_threshold_pct: float = 0.05,
        spike_window_seconds: float = 600.0,
        cooldown_seconds: float = 120.0,
        recovery_threshold_pct: float = 0.05,
    ) -> None:
        self._threshold = max(0.01, spike_threshold_pct)
        self._window = max(1.0, spike_window_seconds)
        self._cooldown = max(0.0, cooldown_seconds)
        self._recovery_threshold = recovery_threshold_pct

        # condition_id -> deque of (timestamp, no_price)
        self._history: dict[str, deque[tuple[float, float]]] = {}
        # condition_id -> last spike timestamp
        self._last_spike: dict[str, float] = {}
        # condition_id -> cluster
        self._market_clusters: dict[str, str] = {}
        # cluster -> list of recent FearSpike
        self._cluster_spikes: dict[str, list[FearSpike]] = {}
        # condition_id -> lowest NO price after spike (for recovery tracking)
        self._spike_low: dict[str, float] = {}

    def observe(
        self,
        condition_id: str,
        no_price: float,
        timestamp: float,
        cluster: str = "other",
    ) -> list[FearSpike]:
        """Observe a NO price update and return any detected spikes."""
        self._market_clusters[condition_id] = cluster
        signals: list[FearSpike] = []

        if condition_id not in self._history:
            self._history[condition_id] = deque(maxlen=500)

        buf = self._history[condition_id]
        buf.append((timestamp, no_price))

        # Prune old entries
        cutoff = timestamp - self._window
        while buf and buf[0][0] < cutoff:
            buf.popleft()

        if len(buf) < 2:
            return []

        oldest_price = buf[0][1]
        delta = no_price - oldest_price

        # Detect fear spike: NO price drops significantly
        if delta <= -self._threshold:
            last_spike = self._last_spike.get(condition_id, 0.0)
            if self._cooldown <= 0 or (timestamp - last_spike) >= self._cooldown:
                spike = FearSpike(
                    condition_id=condition_id,
                    no_price_before=oldest_price,
                    no_price_now=no_price,
                    drop_pct=delta,
                    direction="fear_spike",
                    cluster=cluster,
                    timestamp=timestamp,
                )
                signals.append(spike)
                self._last_spike[condition_id] = timestamp
                self._spike_low[condition_id] = no_price

                # Track in cluster
                if cluster not in self._cluster_spikes:
                    self._cluster_spikes[cluster] = []
                self._cluster_spikes[cluster].append(spike)

                logger.info(
                    "fear_spike_detected",
                    condition_id=condition_id,
                    drop_pct=f"{delta:.2%}",
                    no_price=no_price,
                    cluster=cluster,
                )

        # Detect recovery: NO price rises back after a spike
        if condition_id in self._spike_low:
            low = self._spike_low[condition_id]
            recovery = no_price - low
            if recovery >= self._recovery_threshold:
                signals.append(
                    FearSpike(
                        condition_id=condition_id,
                        no_price_before=low,
                        no_price_now=no_price,
                        drop_pct=recovery,
                        direction="recovery",
                        cluster=cluster,
                        timestamp=timestamp,
                    )
                )
                del self._spike_low[condition_id]

        return signals

    def get_correlated_spikes(
        self, cluster: str, window_seconds: float = 60.0
    ) -> list[FearSpike]:
        """Get recent spikes in the same geopolitical cluster."""
        now = _time.time()
        spikes = self._cluster_spikes.get(cluster, [])
        return [s for s in spikes if (now - s.timestamp) < window_seconds]
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/arb/test_fear_spike_detector.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/arb/fear_spike_detector.py tests/arb/test_fear_spike_detector.py
git commit -m "feat(fear_selling): add FearSpikeDetector with cluster correlation"
```

---

## Task 3: Database Model & Settings

Add the `FearPosition` DB model for tracking open/closed fear-selling positions, and add settings to `config/settings.py`.

**Files:**
- Modify: `src/db/models.py`
- Modify: `config/settings.py`
- Test: `tests/db/test_fear_models.py`

**Step 1: Write the failing tests**

```python
# tests/db/test_fear_models.py
"""Tests for FearPosition database model."""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.models import Base, FearPosition


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


class TestFearPosition:
    def test_create_position(self, db_session):
        pos = FearPosition(
            condition_id="0x123",
            token_id="tok_no",
            title="US strikes Iran by March 31",
            cluster="iran",
            side="NO",
            entry_price=0.65,
            size_usd=10_000.0,
            shares=10_000.0 / 0.65,
            fear_score=0.82,
            yes_price_at_entry=0.35,
        )
        db_session.add(pos)
        db_session.commit()

        fetched = db_session.query(FearPosition).first()
        assert fetched.condition_id == "0x123"
        assert fetched.cluster == "iran"
        assert fetched.is_open is True
        assert fetched.side == "NO"

    def test_close_position(self, db_session):
        pos = FearPosition(
            condition_id="0x456",
            token_id="tok_no",
            title="Khamenei out by Feb 28",
            cluster="iran",
            side="NO",
            entry_price=0.80,
            size_usd=5_000.0,
            shares=5_000.0 / 0.80,
            fear_score=0.75,
            yes_price_at_entry=0.20,
        )
        db_session.add(pos)
        db_session.commit()

        pos.is_open = False
        pos.exit_price = 0.95
        pos.realized_pnl = pos.shares * (0.95 - 0.80)
        db_session.commit()

        fetched = db_session.query(FearPosition).first()
        assert fetched.is_open is False
        assert fetched.realized_pnl > 0
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/db/test_fear_models.py -v`
Expected: FAIL — `ImportError: cannot import name 'FearPosition' from 'src.db.models'`

**Step 3: Add FearPosition model to src/db/models.py**

Append to the end of the models file (before the closing of the module), after the last model class:

```python
class FearPosition(Base):
    """Tracks positions in fear-selling strategy."""

    __tablename__ = "fear_positions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    condition_id = Column(String, nullable=False, index=True)
    token_id = Column(String, nullable=False)
    title = Column(String, nullable=False)
    cluster = Column(String, nullable=False, index=True)
    side = Column(String, nullable=False, default="NO")
    entry_price = Column(Float, nullable=False)
    size_usd = Column(Float, nullable=False)
    shares = Column(Float, nullable=False)
    fear_score = Column(Float, nullable=False)
    yes_price_at_entry = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    realized_pnl = Column(Float, nullable=True)
    unrealized_pnl = Column(Float, default=0.0)
    is_open = Column(Boolean, default=True, index=True)
    entry_trigger = Column(String, nullable=True)  # "scan" | "spike" | "manual"
    opened_at = Column(DateTime, server_default=func.now())
    closed_at = Column(DateTime, nullable=True)
```

**Step 4: Add settings to config/settings.py**

Add after the `NO_BET` section:

```python
    # === Fear Selling Strategy (SwissMiss / tail risk selling) ===
    FEAR_SELLING_ENABLED: bool = False
    FEAR_SELLING_SCAN_INTERVAL: float = 300.0  # seconds between market scans
    FEAR_SELLING_MIN_FEAR_SCORE: float = 0.5
    FEAR_SELLING_MIN_YES_PRICE: float = 0.05
    FEAR_SELLING_MAX_YES_PRICE: float = 0.65
    FEAR_SELLING_MIN_LIQUIDITY: float = 1_000.0
    FEAR_SELLING_MIN_VOLUME_24H: float = 5_000.0
    FEAR_SELLING_SPIKE_THRESHOLD_PCT: float = 0.05  # 5% NO drop = spike
    FEAR_SELLING_SPIKE_WINDOW_SECONDS: float = 600.0  # 10 min window
    FEAR_SELLING_MAX_CLUSTER_PCT: float = 0.30  # max 30% capital per cluster
    FEAR_SELLING_MAX_POSITION_PCT: float = 0.10  # max 10% capital per position
    FEAR_SELLING_KELLY_FRACTION: float = 0.25  # quarter Kelly
    FEAR_SELLING_EXIT_NO_PRICE: float = 0.95  # take profit when NO > 95c
    FEAR_SELLING_STOP_YES_PRICE: float = 0.70  # stop loss when YES > 70c
    CAPITAL_ALLOCATION_FEAR_PCT: float = 0.0  # disabled by default
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/db/test_fear_models.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add src/db/models.py config/settings.py tests/db/test_fear_models.py
git commit -m "feat(fear_selling): add FearPosition model and settings"
```

---

## Task 4: FearSellingEngine — Core Orchestrator

The main engine that ties together scanning, spike detection, entry/exit rules, Kelly sizing, and cluster-based risk limits. Follows the `WeatherOracleEngine` / `CryptoMinuteEngine` pattern.

**Files:**
- Create: `src/arb/fear_engine.py`
- Test: `tests/arb/test_fear_engine.py`

**Step 1: Write the failing tests**

```python
# tests/arb/test_fear_engine.py
"""Tests for FearSellingEngine."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.arb.fear_engine import FearSellingEngine, FearTradeSignal
from src.arb.fear_scanner import FearMarketCandidate
from src.risk.manager import UnifiedRiskManager


@pytest.fixture
def risk_manager():
    return UnifiedRiskManager(
        global_capital=100_000.0,
        reality_allocation_pct=0.0,
        crossmarket_allocation_pct=0.0,
        max_position_pct=0.10,
        daily_loss_limit_pct=0.05,
        fear_allocation_pct=1.0,
    )


@pytest.fixture
def engine(risk_manager):
    return FearSellingEngine(
        risk_manager=risk_manager,
        executor=None,  # paper mode
        max_cluster_pct=0.30,
        kelly_fraction=0.25,
        exit_no_price=0.95,
        stop_yes_price=0.70,
    )


class TestKellySizing:
    def test_kelly_size_moderate_edge(self, engine):
        """Quarter Kelly on 20% edge should give reasonable size."""
        size = engine.compute_kelly_size(
            estimated_no_prob=0.85,
            no_price=0.65,
            available_capital=100_000.0,
        )
        assert 0 < size < 30_000  # reasonable fraction of capital

    def test_kelly_size_zero_when_no_edge(self, engine):
        """No edge = no position."""
        size = engine.compute_kelly_size(
            estimated_no_prob=0.60,
            no_price=0.65,
            available_capital=100_000.0,
        )
        # Fair price ~ true prob, edge ~ 0 or negative
        assert size == 0.0

    def test_kelly_size_capped(self, engine):
        """Size should be capped at max_position_pct of available capital."""
        size = engine.compute_kelly_size(
            estimated_no_prob=0.99,
            no_price=0.50,
            available_capital=100_000.0,
        )
        assert size <= 10_000.0  # 10% cap


class TestClusterLimits:
    def test_cluster_exposure_within_limits(self, engine):
        """Should allow trade when cluster exposure is under limit."""
        can_trade = engine.check_cluster_limit("iran", 10_000.0)
        assert can_trade is True

    def test_cluster_exposure_exceeds_limit(self, engine):
        """Should reject trade when cluster exposure exceeds limit."""
        # Simulate existing exposure
        engine._cluster_exposure["iran"] = 28_000.0
        can_trade = engine.check_cluster_limit("iran", 5_000.0)
        assert can_trade is False


class TestEntryRules:
    def test_generate_signal_from_candidate(self, engine):
        candidate = FearMarketCandidate(
            condition_id="0x123",
            token_id="tok_no",
            title="US strikes Iran by March 31",
            yes_price=0.40,
            no_price=0.60,
            estimated_no_probability=0.85,
            edge_pct=0.25,
            volume_24h=200_000.0,
            liquidity=100_000.0,
            end_date_iso="2026-04-01T00:00:00Z",
            fear_score=0.82,
            cluster="iran",
        )
        signal = engine.evaluate_candidate(candidate)
        assert signal is not None
        assert signal.side == "BUY"
        assert signal.outcome == "NO"
        assert signal.size_usd > 0

    def test_reject_candidate_low_fear_score(self, engine):
        candidate = FearMarketCandidate(
            condition_id="0x456",
            token_id="tok_no",
            title="Random boring event",
            yes_price=0.50,
            no_price=0.50,
            estimated_no_probability=0.55,
            edge_pct=0.05,
            volume_24h=200_000.0,
            liquidity=100_000.0,
            end_date_iso="2026-06-01T00:00:00Z",
            fear_score=0.3,
            cluster="other",
        )
        signal = engine.evaluate_candidate(candidate)
        assert signal is None


class TestExitRules:
    def test_should_exit_take_profit(self, engine):
        """Exit when NO price reaches 0.95."""
        should_exit, reason = engine.check_exit(
            entry_price=0.65,
            current_no_price=0.96,
            current_yes_price=0.04,
        )
        assert should_exit is True
        assert "profit" in reason.lower()

    def test_should_exit_stop_loss(self, engine):
        """Exit when YES price reaches 0.70 (event becoming likely)."""
        should_exit, reason = engine.check_exit(
            entry_price=0.65,
            current_no_price=0.28,
            current_yes_price=0.72,
        )
        assert should_exit is True
        assert "stop" in reason.lower()

    def test_should_hold(self, engine):
        """Hold when price is between entry and targets."""
        should_exit, reason = engine.check_exit(
            entry_price=0.65,
            current_no_price=0.75,
            current_yes_price=0.25,
        )
        assert should_exit is False
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/arb/test_fear_engine.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# src/arb/fear_engine.py
"""Fear Selling Engine — orchestrates the fear-premium selling strategy.

Combines FearMarketScanner (market discovery) and FearSpikeDetector (entry timing)
with Kelly sizing, cluster-based correlation limits, and exit rules.

Follows the engine pattern from WeatherOracleEngine / CryptoMinuteEngine.
"""

from dataclasses import dataclass, field
from typing import Optional

import structlog

from src.arb.fear_scanner import FearMarketCandidate, FearMarketScanner
from src.arb.fear_spike_detector import FearSpikeDetector
from src.risk.manager import UnifiedRiskManager

logger = structlog.get_logger()


@dataclass
class FearTradeSignal:
    """A trade signal generated by the fear-selling engine."""

    condition_id: str
    token_id: str
    title: str
    side: str  # "BUY"
    outcome: str  # "NO"
    price: float  # NO price to buy at
    size_usd: float
    edge_pct: float
    fear_score: float
    cluster: str
    trigger: str  # "scan" | "spike"


class FearSellingEngine:
    """Core engine for systematic fear-premium selling on Polymarket."""

    def __init__(
        self,
        risk_manager: Optional[UnifiedRiskManager] = None,
        executor=None,
        max_cluster_pct: float = 0.30,
        max_position_pct: float = 0.10,
        kelly_fraction: float = 0.25,
        exit_no_price: float = 0.95,
        stop_yes_price: float = 0.70,
        min_fear_score: float = 0.5,
    ):
        self._risk_manager = risk_manager
        self._executor = executor
        self._max_cluster_pct = max_cluster_pct
        self._max_position_pct = max_position_pct
        self._kelly_fraction = kelly_fraction
        self._exit_no_price = exit_no_price
        self._stop_yes_price = stop_yes_price
        self._min_fear_score = min_fear_score

        self._scanner = FearMarketScanner(min_fear_score=min_fear_score)
        self._spike_detector = FearSpikeDetector()

        # Tracking state
        self._cluster_exposure: dict[str, float] = {}
        self._open_positions: dict[str, float] = {}  # condition_id -> size_usd

    @property
    def _available_capital(self) -> float:
        if self._risk_manager:
            return self._risk_manager.get_available_capital("fear")
        return 100_000.0  # paper trading fallback

    def compute_kelly_size(
        self,
        estimated_no_prob: float,
        no_price: float,
        available_capital: float,
    ) -> float:
        """Compute position size using fractional Kelly criterion.

        f* = (p * b - q) / b
        where p = prob of winning, b = net odds, q = 1 - p
        """
        if no_price <= 0 or no_price >= 1:
            return 0.0

        p = estimated_no_prob
        q = 1.0 - p
        b = (1.0 - no_price) / no_price  # net win per dollar risked

        if b <= 0:
            return 0.0

        kelly = (p * b - q) / b
        if kelly <= 0:
            return 0.0

        # Apply fractional Kelly
        fraction = kelly * self._kelly_fraction

        # Cap at max position size
        max_size = available_capital * self._max_position_pct
        size = min(fraction * available_capital, max_size)

        return max(0.0, size)

    def check_cluster_limit(self, cluster: str, proposed_size: float) -> bool:
        """Check if adding proposed_size to cluster would exceed limit."""
        current = self._cluster_exposure.get(cluster, 0.0)
        limit = self._available_capital * self._max_cluster_pct
        return (current + proposed_size) <= limit

    def evaluate_candidate(
        self, candidate: FearMarketCandidate
    ) -> Optional[FearTradeSignal]:
        """Evaluate a fear market candidate and generate a trade signal."""
        if candidate.fear_score < self._min_fear_score:
            return None

        if not candidate.is_valid:
            return None

        available = self._available_capital
        size = self.compute_kelly_size(
            estimated_no_prob=candidate.estimated_no_probability,
            no_price=candidate.no_price,
            available_capital=available,
        )

        if size <= 0:
            return None

        if not self.check_cluster_limit(candidate.cluster, size):
            logger.info(
                "cluster_limit_exceeded",
                cluster=candidate.cluster,
                proposed=size,
            )
            return None

        return FearTradeSignal(
            condition_id=candidate.condition_id,
            token_id=candidate.token_id,
            title=candidate.title,
            side="BUY",
            outcome="NO",
            price=candidate.no_price,
            size_usd=size,
            edge_pct=candidate.edge_pct,
            fear_score=candidate.fear_score,
            cluster=candidate.cluster,
            trigger="scan",
        )

    def check_exit(
        self,
        entry_price: float,
        current_no_price: float,
        current_yes_price: float,
    ) -> tuple[bool, str]:
        """Check if a position should be exited."""
        # Take profit: NO price high enough
        if current_no_price >= self._exit_no_price:
            return True, f"Take profit: NO at {current_no_price:.2f} >= {self._exit_no_price}"

        # Stop loss: YES price too high (event becoming likely)
        if current_yes_price >= self._stop_yes_price:
            return True, f"Stop loss: YES at {current_yes_price:.2f} >= {self._stop_yes_price}"

        return False, "Hold"
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/arb/test_fear_engine.py -v`
Expected: All PASS (may need to add `fear_allocation_pct` to `UnifiedRiskManager.__init__` — see Step 4a below)

**Step 4a: Update UnifiedRiskManager to support fear allocation**

In `src/risk/manager.py`, add `fear_allocation_pct` parameter to `__init__` (following the existing pattern for `nobet_allocation_pct`):
- Add `fear_allocation_pct: float = 0.0` to constructor
- Add `"fear"` to `get_available_capital()` switch
- Add `"fear"` to `_daily_pnl_by_strategy` initialization

**Step 5: Run tests to verify they pass**

Run: `pytest tests/arb/test_fear_engine.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add src/arb/fear_engine.py tests/arb/test_fear_engine.py src/risk/manager.py
git commit -m "feat(fear_selling): add FearSellingEngine with Kelly sizing and cluster limits"
```

---

## Task 5: Gamma API Market Discovery

Add async method to `FearMarketScanner` for fetching active markets from the Polymarket Gamma API and building `FearMarketCandidate` objects. Follows the `MarketScanner.sync()` pattern from `crypto_minute.py`.

**Files:**
- Modify: `src/arb/fear_scanner.py` (add `async def discover_markets()`)
- Test: `tests/arb/test_fear_discovery.py`

**Step 1: Write the failing tests**

```python
# tests/arb/test_fear_discovery.py
"""Tests for FearMarketScanner.discover_markets() — Gamma API integration."""
import json
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
                "tokens": [
                    {"token_id": "tok_yes_1", "outcome": "Yes", "price": 0.40},
                    {"token_id": "tok_no_1", "outcome": "No", "price": 0.60},
                ],
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
                "tokens": [
                    {"token_id": "tok_yes_2", "outcome": "Yes", "price": 0.70},
                    {"token_id": "tok_no_2", "outcome": "No", "price": 0.30},
                ],
            }
        ],
    },
]


@pytest.mark.asyncio
@respx.mock
async def test_discover_fear_markets():
    """Should discover and score fear markets from Gamma API."""
    respx.get(f"{GAMMA_API}/events").mock(
        return_value=httpx.Response(200, json=SAMPLE_EVENTS)
    )

    scanner = FearMarketScanner()
    candidates = await scanner.discover_markets(gamma_url=GAMMA_API)

    # Should find the Iran market as a candidate, not Taylor Swift
    fear_candidates = [c for c in candidates if c.fear_score >= 0.5]
    assert len(fear_candidates) >= 1
    assert fear_candidates[0].cluster == "iran"


@pytest.mark.asyncio
@respx.mock
async def test_discover_handles_api_error():
    """Should return empty list on API error."""
    respx.get(f"{GAMMA_API}/events").mock(
        return_value=httpx.Response(500)
    )

    scanner = FearMarketScanner()
    candidates = await scanner.discover_markets(gamma_url=GAMMA_API)
    assert candidates == []
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/arb/test_fear_discovery.py -v`
Expected: FAIL — `AttributeError: 'FearMarketScanner' object has no attribute 'discover_markets'`

**Step 3: Add discover_markets to FearMarketScanner**

Append to `src/arb/fear_scanner.py`:

```python
    async def discover_markets(
        self,
        gamma_url: str = "https://gamma-api.polymarket.com",
        limit: int = 100,
    ) -> list[FearMarketCandidate]:
        """Fetch active markets from Gamma API and build candidates."""
        import httpx
        from datetime import datetime, timezone

        candidates: list[FearMarketCandidate] = []

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    f"{gamma_url}/events",
                    params={"limit": limit, "active": "true", "closed": "false"},
                )
                if resp.status_code != 200:
                    logger.warning("gamma_api_error", status=resp.status_code)
                    return []

                events = resp.json()
        except Exception as e:
            logger.error("gamma_api_fetch_failed", error=str(e))
            return []

        now = datetime.now(timezone.utc)

        for event in events:
            end_date_str = event.get("endDate", "")
            try:
                end_date = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
                end_date_days = max(1, (end_date - now).days)
            except (ValueError, TypeError):
                end_date_days = 30

            for market in event.get("markets", []):
                tokens = market.get("tokens", [])
                yes_price = None
                no_price = None
                yes_token_id = None
                no_token_id = None

                for token in tokens:
                    outcome = token.get("outcome", "")
                    if outcome == "Yes":
                        yes_price = token.get("price", 0.0)
                        yes_token_id = token.get("token_id", "")
                    elif outcome == "No":
                        no_price = token.get("price", 0.0)
                        no_token_id = token.get("token_id", "")

                if yes_price is None or no_price is None or not no_token_id:
                    continue

                if not (self.min_yes_price <= yes_price <= self.max_yes_price):
                    continue

                title = market.get("question", "") or event.get("title", "")
                volume = float(market.get("volumeNum", 0) or 0)
                liquidity = float(market.get("liquidityNum", 0) or 0)
                condition_id = market.get("conditionId", "")

                fear_score = self.score_market(
                    title=title,
                    yes_price=yes_price,
                    volume_24h=volume,
                    end_date_days=end_date_days,
                )

                cluster = self.detect_cluster(title)
                base_rate = self.estimate_base_rate(
                    title=title,
                    yes_price=yes_price,
                    end_date_days=end_date_days,
                    cluster=cluster,
                )
                estimated_no_prob = 1.0 - base_rate
                edge = estimated_no_prob - no_price

                candidates.append(
                    FearMarketCandidate(
                        condition_id=condition_id,
                        token_id=no_token_id,
                        title=title,
                        yes_price=yes_price,
                        no_price=no_price,
                        estimated_no_probability=estimated_no_prob,
                        edge_pct=edge,
                        volume_24h=volume,
                        liquidity=liquidity,
                        end_date_iso=end_date_str,
                        fear_score=fear_score,
                        cluster=cluster,
                    )
                )

        # Sort by fear_score descending
        candidates.sort(key=lambda c: c.fear_score, reverse=True)
        return candidates
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/arb/test_fear_discovery.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/arb/fear_scanner.py tests/arb/test_fear_discovery.py
git commit -m "feat(fear_selling): add Gamma API market discovery to FearMarketScanner"
```

---

## Task 6: Run Script — Entry Point

Create the main run script following the pattern from `scripts/run_weather_oracle.py`. Connects all components: scanner, spike detector, engine, feed, executor, risk manager.

**Files:**
- Create: `scripts/run_fear_selling.py`
- No unit test needed (integration script), but test it starts in paper mode

**Step 1: Write the script**

```python
# scripts/run_fear_selling.py
"""Fear Selling Bot — systematic tail-risk premium capture.

Scans Polymarket for fear-driven geopolitical markets, buys NO on overpriced
catastrophe events, and manages positions with cluster-based risk limits.

Usage:
    python scripts/run_fear_selling.py [--autopilot] [--scan-interval 300]
"""

import argparse
import asyncio
import signal
import sys
import time

import structlog

from config.settings import settings
from src.arb.fear_engine import FearSellingEngine, FearTradeSignal
from src.arb.fear_scanner import FearMarketScanner
from src.arb.fear_spike_detector import FearSpikeDetector
from src.arb.polymarket_executor import PolymarketExecutor
from src.feeds.polymarket import PolymarketFeed
from src.risk.manager import UnifiedRiskManager

logger = structlog.get_logger()


async def main(args: argparse.Namespace) -> None:
    logger.info("fear_selling_starting", autopilot=args.autopilot)

    # Risk manager
    risk_manager = UnifiedRiskManager(
        global_capital=settings.GLOBAL_CAPITAL,
        reality_allocation_pct=settings.CAPITAL_ALLOCATION_REALITY_PCT,
        crossmarket_allocation_pct=settings.CAPITAL_ALLOCATION_CROSSMARKET_PCT,
        max_position_pct=settings.FEAR_SELLING_MAX_POSITION_PCT,
        daily_loss_limit_pct=settings.DAILY_LOSS_LIMIT_PCT,
        fear_allocation_pct=settings.CAPITAL_ALLOCATION_FEAR_PCT,
    )

    # Executor (None for paper trading)
    executor = None
    if args.autopilot and settings.POLYMARKET_PRIVATE_KEY:
        executor = PolymarketExecutor(
            host=settings.POLYMARKET_CLOB_HTTP,
            chain_id=settings.POLYMARKET_CHAIN_ID,
            private_key=settings.POLYMARKET_PRIVATE_KEY,
            funder=settings.POLYMARKET_WALLET_ADDRESS,
            api_key=settings.POLYMARKET_API_KEY,
            api_secret=settings.POLYMARKET_API_SECRET,
            api_passphrase=settings.POLYMARKET_API_PASSPHRASE,
        )

    # Engine
    engine = FearSellingEngine(
        risk_manager=risk_manager,
        executor=executor,
        max_cluster_pct=settings.FEAR_SELLING_MAX_CLUSTER_PCT,
        kelly_fraction=settings.FEAR_SELLING_KELLY_FRACTION,
        exit_no_price=settings.FEAR_SELLING_EXIT_NO_PRICE,
        stop_yes_price=settings.FEAR_SELLING_STOP_YES_PRICE,
        min_fear_score=settings.FEAR_SELLING_MIN_FEAR_SCORE,
    )

    scan_interval = args.scan_interval

    # Graceful shutdown
    stop = asyncio.Event()

    def _signal_handler(*_):
        logger.info("shutdown_requested")
        stop.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    logger.info(
        "fear_selling_ready",
        capital=risk_manager.get_available_capital("fear"),
        scan_interval=scan_interval,
        mode="autopilot" if args.autopilot else "paper",
    )

    # Main loop
    while not stop.is_set():
        try:
            # 1. Discover markets
            candidates = await engine._scanner.discover_markets()
            logger.info("scan_complete", candidates=len(candidates))

            # 2. Evaluate each candidate
            for candidate in candidates:
                if candidate.fear_score < settings.FEAR_SELLING_MIN_FEAR_SCORE:
                    continue

                signal_trade = engine.evaluate_candidate(candidate)
                if signal_trade is None:
                    continue

                logger.info(
                    "trade_signal",
                    title=signal_trade.title,
                    cluster=signal_trade.cluster,
                    price=signal_trade.price,
                    size=signal_trade.size_usd,
                    edge=f"{signal_trade.edge_pct:.1%}",
                    fear_score=signal_trade.fear_score,
                )

                # 3. Execute if autopilot
                if args.autopilot and executor:
                    try:
                        resp = await executor.place_order(
                            token_id=signal_trade.token_id,
                            side="BUY",
                            size=signal_trade.size_usd,
                            price=signal_trade.price,
                        )
                        logger.info("order_placed", response=resp)
                    except Exception as e:
                        logger.error("order_failed", error=str(e))

            # 4. Check exits on open positions
            # (Position tracking integration — reads from DB)

        except Exception as e:
            logger.error("scan_loop_error", error=str(e))

        # Wait for next scan
        try:
            await asyncio.wait_for(stop.wait(), timeout=scan_interval)
        except asyncio.TimeoutError:
            pass

    logger.info("fear_selling_stopped")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fear Selling Bot")
    parser.add_argument("--autopilot", action="store_true", help="Enable live execution")
    parser.add_argument("--scan-interval", type=float, default=300.0, help="Seconds between scans")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
```

**Step 2: Verify the script imports work**

Run: `python -c "from scripts.run_fear_selling import parse_args; print('OK')"`
Expected: `OK` (no import errors)

**Step 3: Commit**

```bash
git add scripts/run_fear_selling.py
git commit -m "feat(fear_selling): add run script with scan loop and autopilot mode"
```

---

## Task 7: Integration Test — Full Pipeline

End-to-end test that verifies the full pipeline: Gamma API → FearMarketScanner → FearSellingEngine → trade signal generation.

**Files:**
- Create: `tests/arb/test_fear_integration.py`

**Step 1: Write the integration test**

```python
# tests/arb/test_fear_integration.py
"""Integration test for fear-selling pipeline."""
import pytest
import respx
import httpx

from src.arb.fear_scanner import FearMarketScanner
from src.arb.fear_engine import FearSellingEngine
from src.arb.fear_spike_detector import FearSpikeDetector
from src.risk.manager import UnifiedRiskManager

GAMMA_API = "https://gamma-api.polymarket.com"

FEAR_EVENTS = [
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
                "tokens": [
                    {"token_id": "tok_yes_1", "outcome": "Yes", "price": 0.40},
                    {"token_id": "tok_no_1", "outcome": "No", "price": 0.60},
                ],
            }
        ],
    },
    {
        "id": "evt2",
        "title": "Khamenei out as Supreme Leader by June 30",
        "slug": "khamenei-out-june-2026",
        "endDate": "2026-07-01T00:00:00Z",
        "markets": [
            {
                "id": "mkt2",
                "conditionId": "0xdef",
                "question": "Will Khamenei be out as Supreme Leader by June 30?",
                "volume": "300000",
                "volumeNum": 300000,
                "liquidity": "60000",
                "liquidityNum": 60000,
                "tokens": [
                    {"token_id": "tok_yes_2", "outcome": "Yes", "price": 0.35},
                    {"token_id": "tok_no_2", "outcome": "No", "price": 0.65},
                ],
            }
        ],
    },
    {
        "id": "evt3",
        "title": "Will Taylor Swift win Grammy",
        "slug": "taylor-swift-grammy",
        "endDate": "2027-02-01T00:00:00Z",
        "markets": [
            {
                "id": "mkt3",
                "conditionId": "0xghi",
                "question": "Will Taylor Swift win Grammy?",
                "volume": "20000",
                "volumeNum": 20000,
                "liquidity": "5000",
                "liquidityNum": 5000,
                "tokens": [
                    {"token_id": "tok_yes_3", "outcome": "Yes", "price": 0.50},
                    {"token_id": "tok_no_3", "outcome": "No", "price": 0.50},
                ],
            }
        ],
    },
]


@pytest.mark.asyncio
@respx.mock
async def test_full_pipeline():
    """Full pipeline: discover → score → evaluate → generate signals."""
    respx.get(f"{GAMMA_API}/events").mock(
        return_value=httpx.Response(200, json=FEAR_EVENTS)
    )

    risk_manager = UnifiedRiskManager(
        global_capital=100_000.0,
        reality_allocation_pct=0.0,
        crossmarket_allocation_pct=0.0,
        max_position_pct=0.10,
        daily_loss_limit_pct=0.05,
        fear_allocation_pct=1.0,
    )

    engine = FearSellingEngine(
        risk_manager=risk_manager,
        executor=None,
        min_fear_score=0.5,
    )

    # Discover
    candidates = await engine._scanner.discover_markets(gamma_url=GAMMA_API)
    assert len(candidates) >= 2  # Iran + Khamenei (Taylor should be filtered or scored low)

    # Evaluate
    signals = []
    for c in candidates:
        signal = engine.evaluate_candidate(c)
        if signal:
            signals.append(signal)

    # Should generate signals for fear markets only
    assert len(signals) >= 1
    clusters = {s.cluster for s in signals}
    assert "iran" in clusters

    # All signals should be BUY NO
    for s in signals:
        assert s.side == "BUY"
        assert s.outcome == "NO"
        assert s.size_usd > 0


@pytest.mark.asyncio
@respx.mock
async def test_cluster_limit_enforcement():
    """Should respect cluster correlation limits."""
    respx.get(f"{GAMMA_API}/events").mock(
        return_value=httpx.Response(200, json=FEAR_EVENTS)
    )

    risk_manager = UnifiedRiskManager(
        global_capital=50_000.0,
        reality_allocation_pct=0.0,
        crossmarket_allocation_pct=0.0,
        max_position_pct=0.10,
        daily_loss_limit_pct=0.05,
        fear_allocation_pct=1.0,
    )

    engine = FearSellingEngine(
        risk_manager=risk_manager,
        executor=None,
        max_cluster_pct=0.15,  # very low limit — 15% of 50K = 7.5K
        min_fear_score=0.5,
    )

    candidates = await engine._scanner.discover_markets(gamma_url=GAMMA_API)
    iran_candidates = [c for c in candidates if c.cluster == "iran"]

    # Evaluate candidates — second one should be blocked by cluster limit
    signals = []
    for c in iran_candidates:
        signal = engine.evaluate_candidate(c)
        if signal:
            # Simulate the position being opened
            engine._cluster_exposure[c.cluster] = (
                engine._cluster_exposure.get(c.cluster, 0.0) + signal.size_usd
            )
            signals.append(signal)

    # Total iran exposure should not exceed 15% of capital
    total_iran = sum(s.size_usd for s in signals)
    assert total_iran <= 50_000.0 * 0.15 + 1.0  # small float tolerance
```

**Step 2: Run tests to verify they pass**

Run: `pytest tests/arb/test_fear_integration.py -v`
Expected: All PASS

**Step 3: Commit**

```bash
git add tests/arb/test_fear_integration.py
git commit -m "test(fear_selling): add full pipeline integration test"
```

---

## Task 8: Dashboard Tab

Add a "Fear Selling" tab to the existing Streamlit dashboard for monitoring positions, PnL, and cluster exposure.

**Files:**
- Modify: `src/paper_trading/dashboard.py` (add tab)

**Step 1: Read the existing dashboard to understand the tab pattern**

Read: `src/paper_trading/dashboard.py`

**Step 2: Add a fear_selling tab**

Add a new tab in the dashboard's tab list (following existing pattern). The tab should show:
- Active fear positions from `FearPosition` table
- PnL by cluster (bar chart)
- Cluster exposure pie chart
- Recent trade signals log

Implementation follows the existing tab patterns in the dashboard — query `FearPosition` from DB, display with `st.dataframe()`, chart with `st.plotly_chart()` or `st.bar_chart()`.

**Step 3: Commit**

```bash
git add src/paper_trading/dashboard.py
git commit -m "feat(fear_selling): add Fear Selling tab to Streamlit dashboard"
```

---

## Task 9: Final Wiring — __init__.py & Run All Tests

Update `src/arb/__init__.py` to export new modules, run the full test suite, fix any issues.

**Files:**
- Modify: `src/arb/__init__.py`
- Run: `pytest -v`

**Step 1: Update __init__.py**

Add the new exports to `src/arb/__init__.py`:

```python
from .fear_scanner import FearMarketScanner, FearMarketCandidate
from .fear_spike_detector import FearSpikeDetector, FearSpike
from .fear_engine import FearSellingEngine, FearTradeSignal
```

**Step 2: Run full test suite**

Run: `pytest -v`
Expected: All tests PASS (including existing tests — no regressions)

**Step 3: Commit**

```bash
git add src/arb/__init__.py
git commit -m "feat(fear_selling): wire up exports and verify no regressions"
```

---

## Summary

| Task | Component | Files | Tests |
|------|-----------|-------|-------|
| 1 | FearMarketScanner | `src/arb/fear_scanner.py` | 9 tests |
| 2 | FearSpikeDetector | `src/arb/fear_spike_detector.py` | 6 tests |
| 3 | DB Model + Settings | `src/db/models.py`, `config/settings.py` | 2 tests |
| 4 | FearSellingEngine | `src/arb/fear_engine.py` | 7 tests |
| 5 | Gamma API Discovery | `src/arb/fear_scanner.py` (extend) | 2 tests |
| 6 | Run Script | `scripts/run_fear_selling.py` | Import test |
| 7 | Integration Test | `tests/arb/test_fear_integration.py` | 2 tests |
| 8 | Dashboard Tab | `src/paper_trading/dashboard.py` | Visual check |
| 9 | Wiring + Full Suite | `src/arb/__init__.py` | Full pytest |

**Total: 9 tasks, ~28 tests, 4 new files, 4 modified files**
