# Event-Driven Sports Sniper Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an event-driven daemon that detects live sports events (goals, match endings) via Polymarket CLOB WebSocket price spikes and Odds API score changes, then buys the winning outcome across ALL related conditions of the same event before the market fully converges.

**Architecture:** Two parallel signal loops feed a shared action queue. Loop 1 (CLOB WS) subscribes to all active sport conditions and detects price spikes (>15% move in <60s) for instant reaction at T+3-10s. Loop 2 (Odds API scores) polls every 2-3 minutes to detect score changes, providing event context and triggering trades on sibling conditions (O/U, Draw, BTTS) that haven't spiked yet. Both loops funnel `TradeIntent`s through the existing `TwoSidedInventoryEngine` for position management, sizing, and paper fills.

**Tech Stack:** asyncio, websockets (existing PolymarketFeed), httpx, existing TwoSidedInventoryEngine, existing OddsApiClient, SQLite paper persistence

---

## Task 1: Odds API Scores Client

Add a `fetch_scores()` method to the existing `OddsApiClient` that calls The Odds API `/v4/sports/{sport}/scores` endpoint and returns structured live score data.

**Files:**
- Modify: `src/feeds/odds_api.py`
- Test: `tests/feeds/test_odds_api.py`

**Step 1: Write the failing test**

Add to `tests/feeds/test_odds_api.py`:

```python
@pytest.mark.asyncio
async def test_fetch_scores_returns_live_games(respx_mock):
    """fetch_scores should parse completed and in-progress games."""
    payload = [
        {
            "id": "abc123",
            "sport_key": "soccer_epl",
            "sport_title": "EPL",
            "commence_time": "2026-02-09T15:00:00Z",
            "home_team": "Liverpool",
            "away_team": "Arsenal",
            "completed": False,
            "scores": [
                {"name": "Liverpool", "score": "2"},
                {"name": "Arsenal", "score": "1"},
            ],
            "last_updated": "2026-02-09T15:45:00Z",
        },
        {
            "id": "def456",
            "sport_key": "soccer_epl",
            "sport_title": "EPL",
            "commence_time": "2026-02-09T13:00:00Z",
            "home_team": "Chelsea",
            "away_team": "Spurs",
            "completed": True,
            "scores": [
                {"name": "Chelsea", "score": "0"},
                {"name": "Spurs", "score": "1"},
            ],
            "last_updated": "2026-02-09T14:52:00Z",
        },
    ]
    respx_mock.get("https://api.the-odds-api.com/v4/sports/soccer_epl/scores").mock(
        return_value=httpx.Response(
            200,
            json=payload,
            headers={"x-requests-remaining": "9500", "x-requests-used": "500"},
        )
    )

    client = OddsApiClient(api_key="test-key")
    async with httpx.AsyncClient() as http:
        result = await client.fetch_scores(http, sports=["soccer_epl"])

    assert len(result.games) == 2
    liv = result.games[0]
    assert liv.home_team == "Liverpool"
    assert liv.away_team == "Arsenal"
    assert liv.home_score == 2
    assert liv.away_score == 1
    assert liv.completed is False
    che = result.games[1]
    assert che.completed is True
    assert che.away_score == 1
    assert result.usage.remaining == 9500


@pytest.mark.asyncio
async def test_fetch_scores_detects_change(respx_mock):
    """ScoreTracker should detect when a score changes between polls."""
    from src.feeds.odds_api import ScoreTracker, LiveGame

    tracker = ScoreTracker()
    game = LiveGame(
        event_id="abc123", sport="soccer_epl", home_team="Liverpool",
        away_team="Arsenal", home_score=1, away_score=0, completed=False,
    )
    changes = tracker.update([game])
    assert len(changes) == 0  # first observation, no change

    game_updated = LiveGame(
        event_id="abc123", sport="soccer_epl", home_team="Liverpool",
        away_team="Arsenal", home_score=2, away_score=0, completed=False,
    )
    changes = tracker.update([game_updated])
    assert len(changes) == 1
    assert changes[0].event_id == "abc123"
    assert changes[0].home_score == 2
    assert changes[0].prev_home_score == 1
    assert changes[0].change_type == "score_change"


@pytest.mark.asyncio
async def test_fetch_scores_detects_completion(respx_mock):
    """ScoreTracker should detect game completion."""
    from src.feeds.odds_api import ScoreTracker, LiveGame

    tracker = ScoreTracker()
    game = LiveGame(
        event_id="abc123", sport="soccer_epl", home_team="Liverpool",
        away_team="Arsenal", home_score=2, away_score=1, completed=False,
    )
    tracker.update([game])

    game_ended = LiveGame(
        event_id="abc123", sport="soccer_epl", home_team="Liverpool",
        away_team="Arsenal", home_score=2, away_score=1, completed=True,
    )
    changes = tracker.update([game_ended])
    assert len(changes) == 1
    assert changes[0].change_type == "completed"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/feeds/test_odds_api.py::test_fetch_scores_returns_live_games tests/feeds/test_odds_api.py::test_fetch_scores_detects_change tests/feeds/test_odds_api.py::test_fetch_scores_detects_completion -v`
Expected: FAIL - `OddsApiClient has no attribute fetch_scores`

**Step 3: Write minimal implementation**

Add to `src/feeds/odds_api.py`:

```python
@dataclass(slots=True)
class LiveGame:
    event_id: str
    sport: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    completed: bool
    commence_time: Optional[datetime] = None
    last_updated: Optional[datetime] = None


@dataclass(slots=True)
class ScoresSnapshot:
    games: list[LiveGame]
    usage: OddsApiUsage


@dataclass(slots=True)
class ScoreChange:
    event_id: str
    sport: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    prev_home_score: int
    prev_away_score: int
    completed: bool
    change_type: str  # "score_change" | "completed"


class ScoreTracker:
    """Detects score changes between consecutive polls."""

    def __init__(self) -> None:
        self._prev: dict[str, LiveGame] = {}

    def update(self, games: list[LiveGame]) -> list[ScoreChange]:
        changes: list[ScoreChange] = []
        for game in games:
            prev = self._prev.get(game.event_id)
            if prev is not None:
                if game.completed and not prev.completed:
                    changes.append(ScoreChange(
                        event_id=game.event_id, sport=game.sport,
                        home_team=game.home_team, away_team=game.away_team,
                        home_score=game.home_score, away_score=game.away_score,
                        prev_home_score=prev.home_score, prev_away_score=prev.away_score,
                        completed=True, change_type="completed",
                    ))
                elif game.home_score != prev.home_score or game.away_score != prev.away_score:
                    changes.append(ScoreChange(
                        event_id=game.event_id, sport=game.sport,
                        home_team=game.home_team, away_team=game.away_team,
                        home_score=game.home_score, away_score=game.away_score,
                        prev_home_score=prev.home_score, prev_away_score=prev.away_score,
                        completed=game.completed, change_type="score_change",
                    ))
            self._prev[game.event_id] = game
        return changes
```

Add `fetch_scores()` method to `OddsApiClient`:

```python
async def fetch_scores(
    self,
    client: httpx.AsyncClient,
    sports: list[str],
    days_from: int = 1,
) -> ScoresSnapshot:
    games: list[LiveGame] = []
    usage = OddsApiUsage()

    for sport in sports:
        endpoint = f"{self.base_url}/sports/{sport}/scores"
        params = {
            "apiKey": self.api_key,
            "daysFrom": str(days_from),
            "dateFormat": self.date_format,
        }
        try:
            response = await client.get(endpoint, params=params)
            response.raise_for_status()
        except Exception as exc:
            logger.warning("odds_api_scores_error", sport=sport, error=str(exc))
            continue

        usage = self._parse_usage_headers(response.headers)
        payload = response.json()
        if not isinstance(payload, list):
            continue

        for row in payload:
            scores = row.get("scores")
            if not isinstance(scores, list) or len(scores) < 2:
                continue
            home = str(row.get("home_team", "")).strip()
            away = str(row.get("away_team", "")).strip()
            if not home or not away:
                continue
            home_score = 0
            away_score = 0
            for s in scores:
                name = str(s.get("name", "")).strip()
                try:
                    val = int(s.get("score", 0))
                except (TypeError, ValueError):
                    val = 0
                if name == home:
                    home_score = val
                elif name == away:
                    away_score = val
            games.append(LiveGame(
                event_id=f"{sport}:{row.get('id', '')}",
                sport=sport,
                home_team=home,
                away_team=away,
                home_score=home_score,
                away_score=away_score,
                completed=bool(row.get("completed", False)),
                commence_time=_parse_datetime(row.get("commence_time")),
                last_updated=_parse_datetime(row.get("last_updated")),
            ))

    return ScoresSnapshot(games=games, usage=usage)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/feeds/test_odds_api.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/feeds/odds_api.py tests/feeds/test_odds_api.py
git commit -m "feat(odds-api): add fetch_scores and ScoreTracker for live event detection"
```

---

## Task 2: Event-to-Conditions Mapper

Build a mapper that groups Polymarket conditions by sporting event using `event_slug` from the Gamma API, so when one condition spikes, we know all sibling conditions to trade.

**Files:**
- Create: `src/analysis/event_condition_mapper.py`
- Test: `tests/analysis/test_event_condition_mapper.py`

**Step 1: Write the failing test**

```python
# tests/analysis/test_event_condition_mapper.py
import pytest
from src.analysis.event_condition_mapper import EventConditionMapper


def test_build_from_gamma_markets():
    """Mapper groups conditions by event_slug."""
    raw_markets = [
        {
            "conditionId": "cid_win",
            "question": "Will Jaguares win?",
            "events": [{"slug": "col1-jag-dep-2026-02-08"}],
            "outcomes": '["Yes","No"]',
            "clobTokenIds": '["tok_win_yes","tok_win_no"]',
        },
        {
            "conditionId": "cid_draw",
            "question": "Will it be a draw?",
            "events": [{"slug": "col1-jag-dep-2026-02-08"}],
            "outcomes": '["Yes","No"]',
            "clobTokenIds": '["tok_draw_yes","tok_draw_no"]',
        },
        {
            "conditionId": "cid_ou15",
            "question": "O/U 1.5 goals",
            "events": [{"slug": "col1-jag-dep-2026-02-08"}],
            "outcomes": '["Over","Under"]',
            "clobTokenIds": '["tok_ou_over","tok_ou_under"]',
        },
        {
            "conditionId": "cid_other",
            "question": "Will Liverpool win?",
            "events": [{"slug": "epl-liv-ars-2026-02-09"}],
            "outcomes": '["Yes","No"]',
            "clobTokenIds": '["tok_other_yes","tok_other_no"]',
        },
    ]

    mapper = EventConditionMapper()
    mapper.build(raw_markets)

    siblings = mapper.siblings_of("cid_win")
    sibling_cids = {c["conditionId"] for c in siblings}
    assert sibling_cids == {"cid_draw", "cid_ou15"}

    assert mapper.event_slug_for("cid_win") == "col1-jag-dep-2026-02-08"
    assert mapper.event_slug_for("cid_other") == "epl-liv-ars-2026-02-09"
    assert mapper.all_conditions_for_event("col1-jag-dep-2026-02-08") == 3

    # token lookup
    assert mapper.token_ids_for("cid_win") == ["tok_win_yes", "tok_win_no"]


def test_match_score_change_to_event():
    """Mapper can find event_slug from team names."""
    raw_markets = [
        {
            "conditionId": "cid_win",
            "question": "Will Jaguares de Córdoba FC win on 2026-02-08?",
            "events": [{"slug": "col1-jag-dep-2026-02-08"}],
            "outcomes": '["Yes","No"]',
            "clobTokenIds": '["tok_a","tok_b"]',
        },
    ]
    mapper = EventConditionMapper()
    mapper.build(raw_markets)

    slugs = mapper.find_events_by_teams("Jaguares", "Deportivo Pereira")
    assert "col1-jag-dep-2026-02-08" in slugs
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/analysis/test_event_condition_mapper.py -v`
Expected: FAIL - `No module named 'src.analysis.event_condition_mapper'`

**Step 3: Write minimal implementation**

```python
# src/analysis/event_condition_mapper.py
"""Maps Polymarket conditions to sporting events for cross-condition trading."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any, Optional


def _parse_json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except (json.JSONDecodeError, TypeError):
            return []
    return []


def _extract_event_slug(raw: dict[str, Any]) -> str:
    events = raw.get("events", [])
    if isinstance(events, str):
        events = _parse_json_list(events)
    if isinstance(events, list) and events:
        slug = events[0].get("slug") if isinstance(events[0], dict) else None
        if isinstance(slug, str):
            return slug
    return str(raw.get("slug", ""))


def _normalize_for_search(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower().strip())


class EventConditionMapper:
    """Groups Polymarket conditions by event_slug for cross-condition trading."""

    def __init__(self) -> None:
        self._cid_to_slug: dict[str, str] = {}
        self._slug_to_cids: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._cid_to_entry: dict[str, dict[str, Any]] = {}
        self._slug_to_teams: dict[str, str] = {}  # slug -> normalized question text

    def build(self, raw_markets: list[dict[str, Any]]) -> None:
        self._cid_to_slug.clear()
        self._slug_to_cids.clear()
        self._cid_to_entry.clear()
        self._slug_to_teams.clear()

        for raw in raw_markets:
            cid = str(raw.get("conditionId") or "")
            if not cid:
                continue
            slug = _extract_event_slug(raw)
            if not slug:
                continue

            outcomes = _parse_json_list(raw.get("outcomes"))
            token_ids = _parse_json_list(raw.get("clobTokenIds"))
            question = str(raw.get("question") or "")

            entry = {
                "conditionId": cid,
                "question": question,
                "outcomes": outcomes,
                "clobTokenIds": token_ids,
                "event_slug": slug,
            }

            self._cid_to_slug[cid] = slug
            self._slug_to_cids[slug].append(entry)
            self._cid_to_entry[cid] = entry
            if slug not in self._slug_to_teams:
                self._slug_to_teams[slug] = _normalize_for_search(question)

    def siblings_of(self, condition_id: str) -> list[dict[str, Any]]:
        slug = self._cid_to_slug.get(condition_id, "")
        if not slug:
            return []
        return [e for e in self._slug_to_cids[slug] if e["conditionId"] != condition_id]

    def event_slug_for(self, condition_id: str) -> str:
        return self._cid_to_slug.get(condition_id, "")

    def all_conditions_for_event(self, event_slug: str) -> int:
        return len(self._slug_to_cids.get(event_slug, []))

    def conditions_for_event(self, event_slug: str) -> list[dict[str, Any]]:
        return list(self._slug_to_cids.get(event_slug, []))

    def token_ids_for(self, condition_id: str) -> list[str]:
        entry = self._cid_to_entry.get(condition_id)
        if entry is None:
            return []
        return list(entry.get("clobTokenIds", []))

    def all_token_ids(self) -> list[str]:
        out: list[str] = []
        for entry in self._cid_to_entry.values():
            out.extend(entry.get("clobTokenIds", []))
        return out

    def find_events_by_teams(self, *team_fragments: str) -> list[str]:
        normalized = [_normalize_for_search(f) for f in team_fragments if f]
        if not normalized:
            return []
        matches: list[str] = []
        for slug, text in self._slug_to_teams.items():
            if all(frag in text for frag in normalized):
                matches.append(slug)
        if not matches:
            for slug, text in self._slug_to_teams.items():
                if any(frag in text for frag in normalized):
                    matches.append(slug)
        return matches
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/analysis/test_event_condition_mapper.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/analysis/event_condition_mapper.py tests/analysis/test_event_condition_mapper.py
git commit -m "feat: add EventConditionMapper for cross-condition event grouping"
```

---

## Task 3: CLOB WebSocket Price Spike Detector

Build a `SpikeDetector` that monitors price changes from the CLOB WebSocket and emits signals when a condition's mid-price moves more than a threshold within a time window.

**Files:**
- Create: `src/feeds/spike_detector.py`
- Test: `tests/feeds/test_spike_detector.py`

**Step 1: Write the failing test**

```python
# tests/feeds/test_spike_detector.py
import time
import pytest
from src.feeds.spike_detector import SpikeDetector, SpikeSignal


def test_detects_spike_above_threshold():
    """A 20% move in <60s triggers a spike."""
    detector = SpikeDetector(threshold_pct=0.15, window_seconds=60)
    now = time.time()

    detector.observe("cid1", "Yes", 0.50, now - 30)
    detector.observe("cid1", "Yes", 0.50, now - 20)
    signals = detector.observe("cid1", "Yes", 0.72, now)  # +22%

    assert len(signals) == 1
    assert signals[0].condition_id == "cid1"
    assert signals[0].outcome == "Yes"
    assert signals[0].direction == "up"
    assert abs(signals[0].delta - 0.22) < 0.01


def test_no_spike_below_threshold():
    """A 10% move should not trigger with 15% threshold."""
    detector = SpikeDetector(threshold_pct=0.15, window_seconds=60)
    now = time.time()

    detector.observe("cid1", "Yes", 0.50, now - 10)
    signals = detector.observe("cid1", "Yes", 0.59, now)

    assert len(signals) == 0


def test_spike_only_fires_once_per_cooldown():
    """Same condition should not re-fire within cooldown."""
    detector = SpikeDetector(threshold_pct=0.15, window_seconds=60, cooldown_seconds=120)
    now = time.time()

    detector.observe("cid1", "Yes", 0.50, now - 30)
    signals1 = detector.observe("cid1", "Yes", 0.72, now)
    assert len(signals1) == 1

    # Second spike within cooldown - should NOT fire
    detector.observe("cid1", "Yes", 0.72, now + 5)
    signals2 = detector.observe("cid1", "Yes", 0.92, now + 10)
    assert len(signals2) == 0


def test_detects_downward_spike():
    """Spike detection works for drops too (useful for the other side)."""
    detector = SpikeDetector(threshold_pct=0.15, window_seconds=60)
    now = time.time()

    detector.observe("cid1", "No", 0.50, now - 10)
    signals = detector.observe("cid1", "No", 0.28, now)

    assert len(signals) == 1
    assert signals[0].direction == "down"
    assert abs(signals[0].delta - (-0.22)) < 0.01
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/feeds/test_spike_detector.py -v`
Expected: FAIL - `No module named 'src.feeds.spike_detector'`

**Step 3: Write minimal implementation**

```python
# src/feeds/spike_detector.py
"""Detects price spikes on Polymarket CLOB WebSocket feed."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class SpikeSignal:
    condition_id: str
    outcome: str
    price_before: float
    price_now: float
    delta: float
    direction: str  # "up" | "down"
    timestamp: float


class SpikeDetector:
    """Monitors price observations and emits signals on large moves."""

    def __init__(
        self,
        threshold_pct: float = 0.15,
        window_seconds: float = 60.0,
        cooldown_seconds: float = 120.0,
    ) -> None:
        self._threshold = max(0.01, threshold_pct)
        self._window = max(1.0, window_seconds)
        self._cooldown = max(0.0, cooldown_seconds)
        # key -> deque of (timestamp, price)
        self._history: dict[str, deque[tuple[float, float]]] = {}
        # key -> last spike timestamp
        self._last_spike: dict[str, float] = {}

    def _key(self, condition_id: str, outcome: str) -> str:
        return f"{condition_id}:{outcome}"

    def observe(
        self,
        condition_id: str,
        outcome: str,
        price: float,
        timestamp: float,
    ) -> list[SpikeSignal]:
        key = self._key(condition_id, outcome)

        if key not in self._history:
            self._history[key] = deque(maxlen=500)

        buf = self._history[key]
        buf.append((timestamp, price))

        # Prune old entries outside window
        cutoff = timestamp - self._window
        while buf and buf[0][0] < cutoff:
            buf.popleft()

        if len(buf) < 2:
            return []

        # Compare current price to oldest in window
        oldest_price = buf[0][1]
        delta = price - oldest_price

        if abs(delta) < self._threshold:
            return []

        # Cooldown check
        last_spike = self._last_spike.get(key, 0.0)
        if self._cooldown > 0 and (timestamp - last_spike) < self._cooldown:
            return []

        self._last_spike[key] = timestamp

        return [
            SpikeSignal(
                condition_id=condition_id,
                outcome=outcome,
                price_before=oldest_price,
                price_now=price,
                delta=delta,
                direction="up" if delta > 0 else "down",
                timestamp=timestamp,
            )
        ]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/feeds/test_spike_detector.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/feeds/spike_detector.py tests/feeds/test_spike_detector.py
git commit -m "feat: add SpikeDetector for CLOB WebSocket price spike detection"
```

---

## Task 4: Sniper Signal Router

Build the `SniperRouter` that takes spike signals and score changes, maps them to trade actions across sibling conditions, and produces `TradeIntent` lists ready for the engine.

**Files:**
- Create: `src/arb/sniper_router.py`
- Test: `tests/arb/test_sniper_router.py`

**Step 1: Write the failing test**

```python
# tests/arb/test_sniper_router.py
import pytest
from src.arb.sniper_router import SniperRouter
from src.feeds.spike_detector import SpikeSignal
from src.feeds.odds_api import ScoreChange
from src.analysis.event_condition_mapper import EventConditionMapper


def _build_mapper() -> EventConditionMapper:
    mapper = EventConditionMapper()
    mapper.build([
        {
            "conditionId": "cid_win",
            "question": "Will Jaguares win?",
            "events": [{"slug": "col1-jag-dep"}],
            "outcomes": '["Yes","No"]',
            "clobTokenIds": '["tok_w_y","tok_w_n"]',
        },
        {
            "conditionId": "cid_draw",
            "question": "Draw?",
            "events": [{"slug": "col1-jag-dep"}],
            "outcomes": '["Yes","No"]',
            "clobTokenIds": '["tok_d_y","tok_d_n"]',
        },
    ])
    return mapper


def test_spike_on_win_yes_triggers_siblings():
    """Spike up on Win/Yes should also trigger Draw/No buy."""
    mapper = _build_mapper()
    router = SniperRouter(mapper=mapper)

    spike = SpikeSignal(
        condition_id="cid_win", outcome="Yes",
        price_before=0.50, price_now=0.80, delta=0.30,
        direction="up", timestamp=1000.0,
    )
    actions = router.route_spike(spike)

    # Should want to buy Win/Yes AND Draw/No (if team is winning, draw less likely)
    cids = {a.condition_id for a in actions}
    assert "cid_win" in cids
    assert "cid_draw" in cids

    # Win/Yes should be BUY, Draw should be BUY No
    win_action = next(a for a in actions if a.condition_id == "cid_win")
    assert win_action.outcome == "Yes"
    assert win_action.side == "BUY"

    draw_action = next(a for a in actions if a.condition_id == "cid_draw")
    assert draw_action.outcome == "No"
    assert draw_action.side == "BUY"


def test_score_change_triggers_buy_on_all_conditions():
    """Score change should trigger trades on all event conditions."""
    mapper = _build_mapper()
    router = SniperRouter(mapper=mapper)

    change = ScoreChange(
        event_id="soccer_col1:abc",
        sport="soccer_col1",
        home_team="Jaguares",
        away_team="Deportivo Pereira",
        home_score=1, away_score=0,
        prev_home_score=0, prev_away_score=0,
        completed=False, change_type="score_change",
    )
    actions = router.route_score_change(change)

    cids = {a.condition_id for a in actions}
    assert "cid_win" in cids  # home team scored → buy Win/Yes
    assert "cid_draw" in cids  # goal scored → buy Draw/No
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/arb/test_sniper_router.py -v`
Expected: FAIL - `No module named 'src.arb.sniper_router'`

**Step 3: Write minimal implementation**

```python
# src/arb/sniper_router.py
"""Routes spike signals and score changes to trade actions across sibling conditions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from src.analysis.event_condition_mapper import EventConditionMapper
from src.feeds.spike_detector import SpikeSignal
from src.feeds.odds_api import ScoreChange


@dataclass(slots=True)
class SniperAction:
    condition_id: str
    outcome: str
    side: str  # "BUY"
    reason: str  # "spike_primary" | "spike_sibling" | "score_change"
    urgency: float  # higher = more urgent (used for ordering)
    source_event_slug: str


class SniperRouter:
    """Maps signals to concrete buy actions across conditions."""

    def __init__(self, mapper: EventConditionMapper) -> None:
        self._mapper = mapper

    def route_spike(self, spike: SpikeSignal) -> list[SniperAction]:
        actions: list[SniperAction] = []
        slug = self._mapper.event_slug_for(spike.condition_id)
        if not slug:
            return actions

        # Primary: buy the spiking side
        if spike.direction == "up":
            actions.append(SniperAction(
                condition_id=spike.condition_id,
                outcome=spike.outcome,
                side="BUY",
                reason="spike_primary",
                urgency=abs(spike.delta),
                source_event_slug=slug,
            ))

        # Siblings: infer direction for related conditions
        for sibling in self._mapper.siblings_of(spike.condition_id):
            sibling_cid = sibling["conditionId"]
            sibling_outcomes = sibling.get("outcomes", [])
            question = str(sibling.get("question", "")).lower()

            outcome_to_buy = self._infer_sibling_outcome(
                spike=spike,
                sibling_question=question,
                sibling_outcomes=sibling_outcomes,
            )
            if outcome_to_buy:
                actions.append(SniperAction(
                    condition_id=sibling_cid,
                    outcome=outcome_to_buy,
                    side="BUY",
                    reason="spike_sibling",
                    urgency=abs(spike.delta) * 0.7,
                    source_event_slug=slug,
                ))

        return actions

    def route_score_change(self, change: ScoreChange) -> list[SniperAction]:
        actions: list[SniperAction] = []

        # Find matching event slugs
        slugs = self._mapper.find_events_by_teams(change.home_team, change.away_team)
        if not slugs:
            slugs = self._mapper.find_events_by_teams(change.home_team)
        if not slugs:
            return actions

        home_scoring = change.home_score > change.prev_home_score
        away_scoring = change.away_score > change.prev_away_score

        for slug in slugs:
            for cond in self._mapper.conditions_for_event(slug):
                cid = cond["conditionId"]
                question = str(cond.get("question", "")).lower()
                outcomes = cond.get("outcomes", [])

                outcome = self._infer_outcome_from_score(
                    question=question,
                    outcomes=outcomes,
                    home_scoring=home_scoring,
                    away_scoring=away_scoring,
                    completed=change.completed,
                    home_score=change.home_score,
                    away_score=change.away_score,
                )
                if outcome:
                    urgency = 1.0 if change.completed else 0.6
                    actions.append(SniperAction(
                        condition_id=cid,
                        outcome=outcome,
                        side="BUY",
                        reason="score_change",
                        urgency=urgency,
                        source_event_slug=slug,
                    ))

        return actions

    @staticmethod
    def _infer_sibling_outcome(
        spike: SpikeSignal,
        sibling_question: str,
        sibling_outcomes: list[str],
    ) -> Optional[str]:
        if not sibling_outcomes:
            return None

        is_draw = "draw" in sibling_question
        is_ou = "o/u" in sibling_question or "over" in sibling_question

        # If the main market spiked up (team winning), draw becomes less likely
        if is_draw and spike.direction == "up":
            return "No" if "No" in sibling_outcomes else None

        # For O/U and other markets, we can't infer direction from a single spike
        # Just flag the condition but let the engine decide based on orderbook
        if is_ou:
            return None

        # Generic: if spike.direction == "up" on a win market, buy Yes on sibling win markets
        if spike.direction == "up" and "Yes" in sibling_outcomes:
            return "Yes"

        return None

    @staticmethod
    def _infer_outcome_from_score(
        question: str,
        outcomes: list[str],
        home_scoring: bool,
        away_scoring: bool,
        completed: bool,
        home_score: int,
        away_score: int,
    ) -> Optional[str]:
        if not outcomes:
            return None

        is_draw = "draw" in question
        is_win = "win" in question

        if is_draw:
            if home_score == away_score:
                return "Yes" if "Yes" in outcomes else None
            else:
                return "No" if "No" in outcomes else None

        if is_win and (home_scoring or away_scoring):
            # Check which team the question refers to
            if home_scoring and not away_scoring:
                return "Yes" if "Yes" in outcomes else None
            elif away_scoring and not home_scoring:
                return "No" if "No" in outcomes else None

        if completed:
            if home_score > away_score:
                return "Yes" if "Yes" in outcomes else None
            elif away_score > home_score:
                return "No" if "No" in outcomes else None

        return None
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/arb/test_sniper_router.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/arb/sniper_router.py tests/arb/test_sniper_router.py
git commit -m "feat: add SniperRouter for cross-condition trade routing"
```

---

## Task 5: Sniper Daemon Main Loop

Build the main `run_sniper.py` script that orchestrates: Gamma market fetch → event mapper build → CLOB WS subscription → spike detection loop + Odds API scores polling loop → trade execution via paper fills.

**Files:**
- Create: `scripts/run_sniper.py`
- Modify: `run_two_sided.sh` (add `sniper` strategy style)
- Test: `tests/scripts/test_run_sniper.py` (integration smoke test)

**Step 1: Write the failing integration test**

```python
# tests/scripts/test_run_sniper.py
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.feeds.spike_detector import SpikeDetector, SpikeSignal
from src.feeds.odds_api import ScoreTracker, LiveGame, ScoreChange
from src.analysis.event_condition_mapper import EventConditionMapper
from src.arb.sniper_router import SniperRouter, SniperAction


@pytest.mark.asyncio
async def test_sniper_end_to_end_spike_to_actions():
    """Full pipeline: market data → mapper → spike → router → actions."""
    # 1. Build mapper from fake Gamma data
    mapper = EventConditionMapper()
    mapper.build([
        {
            "conditionId": "cid_win",
            "question": "Will Team A win?",
            "events": [{"slug": "epl-tea-teb"}],
            "outcomes": '["Yes","No"]',
            "clobTokenIds": '["tok1","tok2"]',
        },
        {
            "conditionId": "cid_draw",
            "question": "Will it end in a draw?",
            "events": [{"slug": "epl-tea-teb"}],
            "outcomes": '["Yes","No"]',
            "clobTokenIds": '["tok3","tok4"]',
        },
    ])

    # 2. Simulate spike on cid_win
    detector = SpikeDetector(threshold_pct=0.15, window_seconds=60)
    import time
    now = time.time()
    detector.observe("cid_win", "Yes", 0.45, now - 20)
    spikes = detector.observe("cid_win", "Yes", 0.78, now)
    assert len(spikes) == 1

    # 3. Route spike to actions
    router = SniperRouter(mapper=mapper)
    actions = router.route_spike(spikes[0])

    assert len(actions) >= 2
    cids = {a.condition_id for a in actions}
    assert "cid_win" in cids
    assert "cid_draw" in cids

    # 4. Check action details
    win_action = next(a for a in actions if a.condition_id == "cid_win")
    assert win_action.side == "BUY"
    assert win_action.outcome == "Yes"
    assert win_action.reason == "spike_primary"

    draw_action = next(a for a in actions if a.condition_id == "cid_draw")
    assert draw_action.side == "BUY"
    assert draw_action.outcome == "No"
    assert draw_action.reason == "spike_sibling"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/scripts/test_run_sniper.py -v`
Expected: PASS (this is the integration test validating the components from Tasks 1-4 work together)

**Step 3: Write the daemon script**

```python
# scripts/run_sniper.py
"""Event-driven sports sniper daemon.

Detects live sports events via Polymarket CLOB WebSocket price spikes
and Odds API score changes, then buys the winning outcome across all
related conditions before the market fully converges.

Usage:
    uv run python scripts/run_sniper.py \
        --scores-sports soccer_epl,soccer_la_liga,soccer_serie_a \
        --scores-interval 120 \
        --spike-threshold 0.15 \
        --spike-window 60 \
        --max-order 10 \
        --strategy-tag sniper_v1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

import httpx
import structlog

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import Settings
from src.analysis.event_condition_mapper import EventConditionMapper
from src.arb.sniper_router import SniperRouter, SniperAction
from src.arb.two_sided_inventory import TwoSidedInventoryEngine, TradeIntent
from src.feeds.odds_api import OddsApiClient, ScoreTracker
from src.feeds.spike_detector import SpikeDetector

logger = structlog.get_logger()

GAMMA_API = "https://gamma-api.polymarket.com/markets"
CLOB_BOOK_URL = "https://clob.polymarket.com/book"

settings = Settings()


async def fetch_sport_markets(
    client: httpx.AsyncClient,
    limit: int = 1500,
    event_prefixes: Optional[list[str]] = None,
    max_days_to_end: float = 3.0,
) -> list[dict[str, Any]]:
    """Fetch active binary sport markets from Gamma API."""
    all_markets: list[dict[str, Any]] = []
    batch = 100
    offset = 0
    now = time.time()

    while len(all_markets) < limit:
        resp = await client.get(
            GAMMA_API,
            params={"limit": min(batch, limit - len(all_markets)), "offset": offset, "active": "true", "closed": "false"},
        )
        if resp.status_code != 200:
            break
        rows = resp.json()
        if not isinstance(rows, list) or not rows:
            break
        for raw in rows:
            outcomes = json.loads(raw.get("outcomes", "[]")) if isinstance(raw.get("outcomes"), str) else (raw.get("outcomes") or [])
            clob_ids = json.loads(raw.get("clobTokenIds", "[]")) if isinstance(raw.get("clobTokenIds"), str) else (raw.get("clobTokenIds") or [])
            if len(outcomes) != 2 or len(clob_ids) < 2:
                continue
            # Event prefix filter
            events = raw.get("events", [])
            if isinstance(events, str):
                try:
                    events = json.loads(events)
                except Exception:
                    events = []
            slug = events[0].get("slug", "") if isinstance(events, list) and events and isinstance(events[0], dict) else ""
            if event_prefixes and slug:
                prefix = slug.split("-", 1)[0]
                if prefix not in event_prefixes:
                    continue
            all_markets.append(raw)
        offset += batch
        if len(rows) < batch:
            break

    return all_markets


async def fetch_orderbook(
    client: httpx.AsyncClient,
    token_id: str,
) -> dict[str, Any]:
    """Fetch CLOB orderbook for a single token."""
    try:
        resp = await client.get(CLOB_BOOK_URL, params={"token_id": token_id})
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {}


def action_to_intent(
    action: SniperAction,
    mapper: EventConditionMapper,
    price: float,
    size_usd: float,
) -> TradeIntent:
    """Convert a SniperAction to a TradeIntent for the engine."""
    token_ids = mapper.token_ids_for(action.condition_id)
    outcomes = []
    entry = mapper._cid_to_entry.get(action.condition_id, {})
    outcomes = entry.get("outcomes", [])
    token_id = ""
    if outcomes and token_ids:
        try:
            idx = outcomes.index(action.outcome)
            token_id = token_ids[idx] if idx < len(token_ids) else ""
        except ValueError:
            pass

    return TradeIntent(
        condition_id=action.condition_id,
        title=entry.get("question", ""),
        outcome=action.outcome,
        token_id=token_id,
        side="BUY",
        price=price,
        size_usd=size_usd,
        edge_pct=0.0,  # Will be filled by engine evaluation
        reason=action.reason,
    )


async def scores_loop(
    client: httpx.AsyncClient,
    odds_client: OddsApiClient,
    tracker: ScoreTracker,
    router: SniperRouter,
    action_queue: asyncio.Queue,
    sports: list[str],
    interval: float,
) -> None:
    """Poll Odds API /scores and emit actions on changes."""
    while True:
        try:
            snapshot = await odds_client.fetch_scores(client, sports=sports)
            changes = tracker.update(snapshot.games)
            for change in changes:
                actions = router.route_score_change(change)
                for action in actions:
                    await action_queue.put(action)
                if actions:
                    logger.info(
                        "score_change_detected",
                        event_id=change.event_id,
                        score=f"{change.home_score}-{change.away_score}",
                        change_type=change.change_type,
                        actions=len(actions),
                    )
            if snapshot.usage.remaining is not None:
                logger.debug("odds_api_credits", remaining=snapshot.usage.remaining)
        except Exception as exc:
            logger.warning("scores_loop_error", error=str(exc))
        await asyncio.sleep(interval)


async def spike_monitor_loop(
    client: httpx.AsyncClient,
    detector: SpikeDetector,
    router: SniperRouter,
    mapper: EventConditionMapper,
    action_queue: asyncio.Queue,
    poll_interval: float = 2.0,
    book_concurrency: int = 40,
) -> None:
    """Poll CLOB orderbooks for all tracked tokens and detect spikes."""
    # This uses REST polling (cheaper to implement than full WS subscription management).
    # For production, replace with PolymarketFeed WebSocket subscription.
    while True:
        try:
            all_entries = list(mapper._cid_to_entry.values())
            sem = asyncio.Semaphore(book_concurrency)
            now = time.time()

            async def check_condition(entry: dict) -> None:
                cid = entry["conditionId"]
                token_ids = entry.get("clobTokenIds", [])
                outcomes = entry.get("outcomes", [])

                for i, tid in enumerate(token_ids):
                    if i >= len(outcomes):
                        break
                    async with sem:
                        book = await fetch_orderbook(client, tid)
                    bids = book.get("bids", [])
                    asks = book.get("asks", [])
                    if not bids and not asks:
                        continue
                    best_bid = float(bids[0]["price"]) if bids else 0.0
                    best_ask = float(asks[0]["price"]) if asks else 0.0
                    mid = (best_bid + best_ask) / 2 if best_bid > 0 and best_ask > 0 else max(best_bid, best_ask)
                    if mid <= 0:
                        continue

                    spikes = detector.observe(cid, outcomes[i], mid, now)
                    for spike in spikes:
                        actions = router.route_spike(spike)
                        for action in actions:
                            await action_queue.put(action)
                        if actions:
                            logger.info(
                                "spike_detected",
                                condition_id=cid,
                                outcome=outcomes[i],
                                delta=f"{spike.delta:.3f}",
                                direction=spike.direction,
                                actions=len(actions),
                            )

            tasks = [check_condition(e) for e in all_entries]
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as exc:
            logger.warning("spike_loop_error", error=str(exc))
        await asyncio.sleep(poll_interval)


async def execution_loop(
    client: httpx.AsyncClient,
    action_queue: asyncio.Queue,
    mapper: EventConditionMapper,
    max_order_usd: float,
    strategy_tag: str,
) -> None:
    """Consume actions from queue and log/paper-fill them."""
    while True:
        action: SniperAction = await action_queue.get()
        try:
            logger.info(
                "sniper_action",
                condition_id=action.condition_id,
                outcome=action.outcome,
                side=action.side,
                reason=action.reason,
                urgency=f"{action.urgency:.3f}",
                event_slug=action.source_event_slug,
            )
            # TODO: Wire up paper fills via TwoSidedPaperRecorder
            # For now, log actions for validation
        except Exception as exc:
            logger.warning("execution_error", error=str(exc))
        finally:
            action_queue.task_done()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Event-driven sports sniper")
    p.add_argument("--scores-sports", type=str, default="soccer_epl,soccer_la_liga,soccer_serie_a,soccer_brazil_serie_a,soccer_argentina_primera_division")
    p.add_argument("--scores-interval", type=float, default=120.0, help="Seconds between Odds API score polls")
    p.add_argument("--spike-threshold", type=float, default=0.15, help="Min price move to trigger spike (0.15 = 15%%)")
    p.add_argument("--spike-window", type=float, default=60.0, help="Spike detection window in seconds")
    p.add_argument("--spike-cooldown", type=float, default=120.0, help="Min seconds between spikes on same condition")
    p.add_argument("--spike-poll-interval", type=float, default=2.0, help="Seconds between CLOB orderbook polls")
    p.add_argument("--book-concurrency", type=int, default=40)
    p.add_argument("--market-limit", type=int, default=1500)
    p.add_argument("--event-prefixes", type=str, default="epl,lal,sea,fl1,por,bun,tur,arg,col1,nba,nfl,cbb,atp,wta,ucl,cs2")
    p.add_argument("--max-order", type=float, default=10.0)
    p.add_argument("--mapper-refresh-seconds", type=float, default=300.0, help="Re-fetch Gamma markets every N seconds")
    p.add_argument("--strategy-tag", type=str, default="sniper_v1")
    return p


async def main() -> None:
    args = build_parser().parse_args()

    odds_client = OddsApiClient(api_key=settings.ODDS_API_KEY)
    tracker = ScoreTracker()
    detector = SpikeDetector(
        threshold_pct=args.spike_threshold,
        window_seconds=args.spike_window,
        cooldown_seconds=args.spike_cooldown,
    )
    mapper = EventConditionMapper()
    action_queue: asyncio.Queue[SniperAction] = asyncio.Queue()

    prefixes = [p.strip() for p in args.event_prefixes.split(",") if p.strip()] or None
    scores_sports = [s.strip() for s in args.scores_sports.split(",") if s.strip()]

    async with httpx.AsyncClient(timeout=20.0) as client:
        # Initial market fetch + mapper build
        logger.info("fetching_markets", limit=args.market_limit)
        raw_markets = await fetch_sport_markets(client, limit=args.market_limit, event_prefixes=prefixes)
        mapper.build(raw_markets)
        logger.info("mapper_built", conditions=len(mapper._cid_to_entry), events=len(mapper._slug_to_cids))

        router = SniperRouter(mapper=mapper)

        # Launch parallel loops
        tasks = [
            asyncio.create_task(scores_loop(
                client, odds_client, tracker, router, action_queue,
                sports=scores_sports, interval=args.scores_interval,
            )),
            asyncio.create_task(spike_monitor_loop(
                client, detector, router, mapper, action_queue,
                poll_interval=args.spike_poll_interval,
                book_concurrency=args.book_concurrency,
            )),
            asyncio.create_task(execution_loop(
                client, action_queue, mapper,
                max_order_usd=args.max_order,
                strategy_tag=args.strategy_tag,
            )),
        ]

        # Periodic mapper refresh
        async def refresh_mapper():
            while True:
                await asyncio.sleep(args.mapper_refresh_seconds)
                try:
                    new_markets = await fetch_sport_markets(client, limit=args.market_limit, event_prefixes=prefixes)
                    mapper.build(new_markets)
                    logger.info("mapper_refreshed", conditions=len(mapper._cid_to_entry))
                except Exception as exc:
                    logger.warning("mapper_refresh_error", error=str(exc))

        tasks.append(asyncio.create_task(refresh_mapper()))

        logger.info("sniper_started", scores_sports=scores_sports, spike_threshold=args.spike_threshold)
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
```

**Step 4: Run integration test**

Run: `pytest tests/scripts/test_run_sniper.py -v`
Expected: ALL PASS

**Step 5: Smoke test the daemon starts**

Run: `timeout 10 .venv/bin/python scripts/run_sniper.py --spike-poll-interval 999 --scores-interval 999 2>&1 || true`
Expected: logs `fetching_markets`, `mapper_built`, `sniper_started`, then timeout

**Step 6: Commit**

```bash
git add scripts/run_sniper.py tests/scripts/test_run_sniper.py
git commit -m "feat: add event-driven sports sniper daemon with spike + scores detection"
```

---

## Task 6: Wire Paper Fills and Snapshot Persistence

Connect the sniper execution loop to the existing `TwoSidedPaperRecorder` for DB persistence, and add a `--dry-run` mode that only logs without recording.

**Files:**
- Modify: `scripts/run_sniper.py` (add paper fill logic to `execution_loop`)
- Test: `tests/scripts/test_run_sniper.py` (add persistence test)

This task depends on Tasks 1-5 being complete. Implementation details:

- Import `TwoSidedPaperRecorder` from `scripts/run_two_sided_inventory.py`
- For each `SniperAction`, fetch the current orderbook to get the actual ask price
- Build a `TradeIntent` via `action_to_intent()` with the live ask
- Apply the fill through the `TwoSidedInventoryEngine` for position management
- Persist via `TwoSidedPaperRecorder.persist_fill()`
- Log the fill with reason, edge, and event context

**Step 1-5:** Follow TDD pattern: write test for paper fill output → implement → verify → commit.

```bash
git commit -m "feat(sniper): wire paper fills and DB persistence"
```

---

## Post-Implementation Validation

After all tasks are complete, run:

```bash
# All unit tests
pytest tests/feeds/test_odds_api.py tests/feeds/test_spike_detector.py tests/analysis/test_event_condition_mapper.py tests/arb/test_sniper_router.py tests/scripts/test_run_sniper.py -v

# Smoke test daemon
timeout 30 .venv/bin/python scripts/run_sniper.py --scores-interval 120 --spike-poll-interval 5 --market-limit 500

# Compare sniper detections vs RN1 activity
.venv/bin/python scripts/snapshot_rn1_vs_local_entries.py --hours 6 --strategy-tag sniper_v1
```
