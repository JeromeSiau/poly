"""Match Polymarket markets to bookmaker events/markets/outcomes.

This module focuses on practical matching for sports markets:
- event-level match (teams + kickoff proximity + text similarity)
- market-level match (moneyline / draw binary / totals with line)
- outcome mapping (Polymarket outcomes -> bookmaker outcome semantics)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from src.matching.normalizer import EventNormalizer

GENERIC_OUTCOMES = {"yes", "no", "over", "under"}
DRAW_WORDS = ("draw", "tie", "match nul")


def _parse_json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
            return decoded if isinstance(decoded, list) else []
        except json.JSONDecodeError:
            return []
    return []


def _as_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_datetime(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return _as_utc(value)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if isinstance(value, str) and value.strip():
        raw = value.strip()
        try:
            if raw.endswith("Z"):
                raw = raw.replace("Z", "+00:00")
            return _as_utc(datetime.fromisoformat(raw))
        except ValueError:
            return None
    return None


def _canonical_market_type(market_type: str) -> str:
    lowered = market_type.lower().strip()
    if lowered in {"moneyline", "h2h", "match_winner", "winner", "1x2"}:
        return "moneyline"
    if lowered in {"totals", "total", "over_under", "ou", "o/u"}:
        return "totals"
    return lowered


@dataclass(slots=True)
class BookmakerMarket:
    """One bookmaker market (e.g. moneyline, totals 2.5)."""

    market_type: str
    outcomes: dict[str, float]
    line: Optional[float] = None
    market_id: str = ""


@dataclass(slots=True)
class BookmakerEvent:
    """Bookmaker event with teams and available markets."""

    event_id: str
    home_team: str
    away_team: str
    starts_at: Optional[datetime] = None
    sport: str = ""
    league: str = ""
    markets: list[BookmakerMarket] = field(default_factory=list)


@dataclass(slots=True)
class BookmakerMarketMatch:
    """Resolved mapping for one Polymarket market."""

    condition_id: str
    polymarket_question: str
    bookmaker_event_id: str
    bookmaker_market_type: str
    bookmaker_market_id: str
    confidence: float
    line: Optional[float]
    outcome_map: dict[str, str]
    team_home: str
    team_away: str


@dataclass(slots=True)
class _ParsedPolymarketMarket:
    condition_id: str
    question: str
    outcomes: list[str]
    teams: list[str]
    start_time: Optional[datetime]
    market_type: str
    line: Optional[float]
    yes_target_team: Optional[str]


class BookmakerMatcher:
    """Match Polymarket sports markets with bookmaker events."""

    def __init__(
        self,
        normalizer: Optional[EventNormalizer] = None,
        min_event_confidence: float = 0.62,
        kickoff_tolerance_minutes: int = 120,
    ) -> None:
        self._normalizer = normalizer or EventNormalizer()
        self._min_event_confidence = min_event_confidence
        self._kickoff_tolerance_minutes = kickoff_tolerance_minutes

    def match_market(
        self,
        polymarket_market: dict[str, Any],
        bookmaker_events: list[BookmakerEvent],
    ) -> Optional[BookmakerMarketMatch]:
        """Match one Polymarket market to bookmaker event+market+outcomes."""
        parsed = self._parse_polymarket_market(polymarket_market)
        if not parsed.question or not parsed.outcomes:
            return None

        event, event_conf = self._select_best_event(parsed, bookmaker_events)
        if event is None or event_conf < self._min_event_confidence:
            return None

        picked_market = self._pick_bookmaker_market(parsed, event)
        if picked_market is None:
            return None

        outcome_map = self._build_outcome_map(parsed, event, picked_market)
        if not outcome_map:
            return None

        confidence = min(0.999, event_conf + 0.05)
        return BookmakerMarketMatch(
            condition_id=parsed.condition_id,
            polymarket_question=parsed.question,
            bookmaker_event_id=event.event_id,
            bookmaker_market_type=_canonical_market_type(picked_market.market_type),
            bookmaker_market_id=picked_market.market_id,
            confidence=confidence,
            line=picked_market.line,
            outcome_map=outcome_map,
            team_home=event.home_team,
            team_away=event.away_team,
        )

    def match_all(
        self,
        polymarket_markets: list[dict[str, Any]],
        bookmaker_events: list[BookmakerEvent],
    ) -> list[BookmakerMarketMatch]:
        """Match many Polymarket markets."""
        matches: list[BookmakerMarketMatch] = []
        for market in polymarket_markets:
            resolved = self.match_market(market, bookmaker_events)
            if resolved is not None:
                matches.append(resolved)
        return matches

    @staticmethod
    def to_condition_index(
        matches: list[BookmakerMarketMatch],
    ) -> dict[str, BookmakerMarketMatch]:
        """Handy index: condition_id -> match."""
        return {m.condition_id: m for m in matches}

    def _parse_polymarket_market(self, market: dict[str, Any]) -> _ParsedPolymarketMarket:
        question = str(market.get("question", "") or market.get("title", "")).strip()
        outcomes = [str(x) for x in _parse_json_list(market.get("outcomes", []))]
        if not outcomes and isinstance(market.get("tokens"), list):
            outcomes = [str(x.get("outcome", "")) for x in market["tokens"] if x.get("outcome")]

        start_time = (
            _parse_datetime(market.get("endDate"))
            or _parse_datetime(market.get("startDate"))
            or self._normalizer.extract_date(question)
        )

        teams = self._extract_teams(question, outcomes, market)
        market_type, line, yes_target_team = self._classify_market(question, outcomes)
        if yes_target_team and yes_target_team not in teams:
            teams.append(yes_target_team)

        condition_id = str(
            market.get("conditionId")
            or market.get("condition_id")
            or market.get("id")
            or ""
        )
        return _ParsedPolymarketMarket(
            condition_id=condition_id,
            question=question,
            outcomes=outcomes,
            teams=teams,
            start_time=start_time,
            market_type=market_type,
            line=line,
            yes_target_team=yes_target_team,
        )

    def _extract_teams(
        self,
        question: str,
        outcomes: list[str],
        market: dict[str, Any],
    ) -> list[str]:
        found: list[str] = []

        def add_team(name: str) -> None:
            cleaned = self._normalizer.normalize_team(name.strip())
            if cleaned and cleaned not in found:
                found.append(cleaned)

        # Extract from "X vs Y" text if present.
        sources = [question]
        events = market.get("events", [])
        if isinstance(events, str):
            events = _parse_json_list(events)
        if isinstance(events, list) and events:
            first_title = events[0].get("title")
            if isinstance(first_title, str) and first_title:
                sources.append(first_title)

        for source in sources:
            pieces = re.split(r"\s+vs\.?\s+", source, maxsplit=1, flags=re.IGNORECASE)
            if len(pieces) == 2:
                left = self._clean_team_fragment(pieces[0])
                right = self._clean_team_fragment(pieces[1])
                if left:
                    add_team(left)
                if right:
                    add_team(right)

        # Extract with existing alias-aware normalizer.
        for team in self._normalizer.extract_teams(question):
            if team not in found:
                found.append(team)

        # Fallback for named binary outcomes (tennis/esports winner markets).
        lowered = [o.lower().strip() for o in outcomes]
        if len(outcomes) == 2 and not set(lowered).issubset(GENERIC_OUTCOMES):
            for outcome in outcomes:
                add_team(outcome)

        return found

    @staticmethod
    def _clean_team_fragment(value: str) -> str:
        text = value.strip()
        text = re.sub(r"^will\s+", "", text, flags=re.IGNORECASE)
        text = re.sub(r":.*$", "", text).strip()
        text = re.sub(r"\s+end in a draw.*$", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"\s+win on .*$", "", text, flags=re.IGNORECASE).strip()
        text = text.strip(" ?!.,")
        return text

    def _classify_market(
        self,
        question: str,
        outcomes: list[str],
    ) -> tuple[str, Optional[float], Optional[str]]:
        lowered_q = question.lower()
        lowered_outcomes = [o.lower().strip() for o in outcomes]

        line_match = re.search(
            r"(?:o\/u|over\/under|total(?:s)?)\s*([0-9]+(?:\.[0-9]+)?)",
            lowered_q,
        )
        line = float(line_match.group(1)) if line_match else None

        if line is not None or set(lowered_outcomes) == {"over", "under"}:
            return "totals", line, None

        if "end in a draw" in lowered_q:
            return "draw_binary", None, None

        if set(lowered_outcomes) == {"yes", "no"}:
            winner_match = re.search(r"will\s+(.+?)\s+win(?:\s+on|\?|$)", question, re.IGNORECASE)
            if winner_match:
                target = self._normalizer.normalize_team(winner_match.group(1))
                return "yes_no_team_win", None, target
            return "yes_no_generic", None, None

        return "moneyline", None, None

    def _select_best_event(
        self,
        parsed: _ParsedPolymarketMarket,
        events: list[BookmakerEvent],
    ) -> tuple[Optional[BookmakerEvent], float]:
        best_event: Optional[BookmakerEvent] = None
        best_score = 0.0

        for event in events:
            score = self._score_event_candidate(parsed, event)
            if score > best_score:
                best_score = score
                best_event = event

        return best_event, best_score

    def _score_event_candidate(
        self,
        parsed: _ParsedPolymarketMarket,
        event: BookmakerEvent,
    ) -> float:
        home = self._normalizer.normalize_team(event.home_team)
        away = self._normalizer.normalize_team(event.away_team)
        event_teams = {home, away}
        pm_teams = set(parsed.teams)

        if pm_teams:
            overlap = len(pm_teams & event_teams)
            if overlap >= 2:
                team_score = 1.0
            elif overlap == 1:
                team_score = 0.80
            else:
                # Fallback fuzzy team matching for partial names.
                best_fuzzy = 0.0
                for team in pm_teams:
                    best_fuzzy = max(
                        best_fuzzy,
                        self._normalizer.calculate_similarity(team, home),
                        self._normalizer.calculate_similarity(team, away),
                    )
                if best_fuzzy < 0.52:
                    return 0.0
                team_score = 0.45 + 0.40 * best_fuzzy
        else:
            team_score = 0.35

        # Strong boost for explicit "Will <team> win?" markets.
        if parsed.market_type == "yes_no_team_win" and parsed.yes_target_team:
            side = self._map_team_to_side(parsed.yes_target_team, event)
            if side is not None:
                team_score = max(team_score, 0.90)

        description = f"{event.home_team} vs {event.away_team} - {event.league}".strip()
        similarity = self._normalizer.calculate_similarity(parsed.question, description)

        if parsed.start_time and event.starts_at:
            parsed_ts = _as_utc(parsed.start_time)
            event_ts = _as_utc(event.starts_at)
            delta_min = abs((parsed_ts - event_ts).total_seconds()) / 60.0
            same_day = parsed_ts.date() == event_ts.date()
            hard_window = self._kickoff_tolerance_minutes * (12 if same_day else 3)
            if delta_min > hard_window and len(pm_teams) >= 2:
                return 0.0
            soft_window = self._kickoff_tolerance_minutes * (6 if same_day else 1)
            time_score = max(0.0, 1.0 - (delta_min / soft_window))
        else:
            time_score = 0.5

        return 0.55 * team_score + 0.35 * similarity + 0.10 * time_score

    def _pick_bookmaker_market(
        self,
        parsed: _ParsedPolymarketMarket,
        event: BookmakerEvent,
    ) -> Optional[BookmakerMarket]:
        if not event.markets:
            return None

        if parsed.market_type == "totals":
            totals = [
                m for m in event.markets
                if _canonical_market_type(m.market_type) == "totals"
            ]
            if not totals:
                return None
            if parsed.line is None:
                return totals[0]
            line_matches = [m for m in totals if m.line is not None]
            if not line_matches:
                return None
            line_matches.sort(key=lambda m: abs(float(m.line) - parsed.line))
            best = line_matches[0]
            if best.line is None or abs(float(best.line) - parsed.line) > 0.11:
                return None
            return best

        moneylines = [
            m for m in event.markets
            if _canonical_market_type(m.market_type) == "moneyline"
        ]
        if not moneylines:
            return None
        return moneylines[0]

    def _build_outcome_map(
        self,
        parsed: _ParsedPolymarketMarket,
        event: BookmakerEvent,
        market: BookmakerMarket,
    ) -> dict[str, str]:
        lowered_outcomes = {o.lower().strip() for o in parsed.outcomes}
        canonical_market = _canonical_market_type(market.market_type)
        keys = self._canonical_outcome_keys(event, market)

        if canonical_market == "totals":
            if "over" not in keys or "under" not in keys:
                return {}
            outcome_map: dict[str, str] = {}
            for outcome in parsed.outcomes:
                lowered = outcome.lower().strip()
                if lowered.startswith("over"):
                    outcome_map[outcome] = "over"
                elif lowered.startswith("under"):
                    outcome_map[outcome] = "under"
            return outcome_map

        if parsed.market_type == "draw_binary" and lowered_outcomes == {"yes", "no"}:
            if "draw" not in keys:
                return {}
            return {
                self._find_outcome(parsed.outcomes, "yes"): "draw",
                self._find_outcome(parsed.outcomes, "no"): "1-draw",
            }

        if parsed.market_type == "yes_no_team_win" and lowered_outcomes == {"yes", "no"}:
            if parsed.yes_target_team is None:
                return {}
            yes_key = self._map_team_to_side(parsed.yes_target_team, event)
            if yes_key is None:
                return {}
            return {
                self._find_outcome(parsed.outcomes, "yes"): yes_key,
                self._find_outcome(parsed.outcomes, "no"): f"1-{yes_key}",
            }

        if parsed.market_type in {"moneyline", "yes_no_generic"}:
            outcome_map = self._map_named_outcomes(parsed.outcomes, event, keys)
            return outcome_map

        return {}

    def _canonical_outcome_keys(
        self,
        event: BookmakerEvent,
        market: BookmakerMarket,
    ) -> dict[str, str]:
        home = self._normalizer.normalize_team(event.home_team)
        away = self._normalizer.normalize_team(event.away_team)
        resolved: dict[str, str] = {}

        for raw_key in market.outcomes.keys():
            key = raw_key.lower().strip()
            normalized = self._normalizer.normalize_team(key)
            canonical: Optional[str] = None

            if key in {"home", "1"}:
                canonical = "home"
            elif key in {"away", "2"}:
                canonical = "away"
            elif key in {"draw", "x", "tie"} or any(word in key for word in DRAW_WORDS):
                canonical = "draw"
            elif key.startswith("over"):
                canonical = "over"
            elif key.startswith("under"):
                canonical = "under"
            elif normalized == home:
                canonical = "home"
            elif normalized == away:
                canonical = "away"

            if canonical is not None and canonical not in resolved:
                resolved[canonical] = raw_key

        return resolved

    @staticmethod
    def _find_outcome(outcomes: list[str], name: str) -> str:
        target = name.lower().strip()
        for outcome in outcomes:
            if outcome.lower().strip() == target:
                return outcome
        return name

    def _map_team_to_side(self, team: str, event: BookmakerEvent) -> Optional[str]:
        target = self._normalizer.normalize_team(team)
        home = self._normalizer.normalize_team(event.home_team)
        away = self._normalizer.normalize_team(event.away_team)

        if target == home:
            return "home"
        if target == away:
            return "away"

        sim_home = self._normalizer.calculate_similarity(target, home)
        sim_away = self._normalizer.calculate_similarity(target, away)
        if sim_home >= sim_away and sim_home >= 0.55:
            return "home"
        if sim_away > sim_home and sim_away >= 0.55:
            return "away"
        return None

    def _map_named_outcomes(
        self,
        outcomes: list[str],
        event: BookmakerEvent,
        canonical_keys: dict[str, str],
    ) -> dict[str, str]:
        outcome_map: dict[str, str] = {}
        home = self._normalizer.normalize_team(event.home_team)
        away = self._normalizer.normalize_team(event.away_team)

        for outcome in outcomes:
            lowered = outcome.lower().strip()
            if any(word in lowered for word in DRAW_WORDS):
                if "draw" in canonical_keys:
                    outcome_map[outcome] = "draw"
                continue

            team = self._normalizer.normalize_team(outcome)

            # Token-level fallback for partial names like "Wild" vs "Thiago Seyboth Wild".
            if re.search(rf"\b{re.escape(team)}\b", home):
                outcome_map[outcome] = "home"
                continue
            if re.search(rf"\b{re.escape(team)}\b", away):
                outcome_map[outcome] = "away"
                continue

            home_sim = max(
                self._normalizer.calculate_similarity(team, home),
                self._normalizer.calculate_similarity(lowered, home),
            )
            away_sim = max(
                self._normalizer.calculate_similarity(team, away),
                self._normalizer.calculate_similarity(lowered, away),
            )

            if home_sim >= away_sim and home_sim >= 0.45:
                outcome_map[outcome] = "home"
            elif away_sim > home_sim and away_sim >= 0.45:
                outcome_map[outcome] = "away"

        return outcome_map
