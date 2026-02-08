"""Odds API client for external fair price construction.

Transforms The Odds API payload into normalized bookmaker events that can be
matched against Polymarket markets.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import median
from typing import Any, Optional

import httpx
import structlog

from src.matching.bookmaker_matcher import BookmakerEvent, BookmakerMarket
from src.matching.normalizer import EventNormalizer

logger = structlog.get_logger()


def _parse_datetime(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str) and value:
        raw = value.strip()
        if raw.endswith("Z"):
            raw = raw.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(raw)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def _safe_decimal_odds(value: Any) -> Optional[float]:
    try:
        odds = float(value)
    except (TypeError, ValueError):
        return None
    if odds <= 1.0:
        return None
    return odds


def _devig(odds_by_outcome: dict[str, float]) -> dict[str, float]:
    implied: dict[str, float] = {}
    for outcome, odds in odds_by_outcome.items():
        if odds <= 1.0:
            continue
        implied[outcome] = 1.0 / odds

    total = sum(implied.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in implied.items()}


def _normalize_probabilities(probabilities: dict[str, float]) -> dict[str, float]:
    filtered = {k: float(v) for k, v in probabilities.items() if v > 0}
    total = sum(filtered.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in filtered.items()}


@dataclass(slots=True)
class OddsApiUsage:
    remaining: Optional[int] = None
    used: Optional[int] = None
    last: Optional[int] = None


@dataclass(slots=True)
class OddsApiSnapshot:
    events: list[BookmakerEvent]
    usage: OddsApiUsage


class OddsApiClient:
    """Fetches odds and normalizes them for the matcher layer."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.the-odds-api.com/v4",
        odds_format: str = "decimal",
        date_format: str = "iso",
        normalizer: Optional[EventNormalizer] = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.odds_format = odds_format
        self.date_format = date_format
        self._normalizer = normalizer or EventNormalizer()

    async def fetch_events(
        self,
        client: httpx.AsyncClient,
        sports: list[str],
        regions: str,
        markets: str,
    ) -> OddsApiSnapshot:
        if not self.api_key:
            return OddsApiSnapshot(events=[], usage=OddsApiUsage())

        events: dict[str, BookmakerEvent] = {}
        usage = OddsApiUsage()

        for sport in sports:
            endpoint = f"{self.base_url}/sports/{sport}/odds"
            params = {
                "apiKey": self.api_key,
                "regions": regions,
                "markets": markets,
                "oddsFormat": self.odds_format,
                "dateFormat": self.date_format,
            }
            try:
                response = await client.get(endpoint, params=params)
                response.raise_for_status()
            except Exception as exc:
                logger.warning("odds_api_fetch_error", sport=sport, error=str(exc))
                continue

            usage = self._parse_usage_headers(response.headers)
            payload = response.json()
            if not isinstance(payload, list):
                continue

            for row in payload:
                parsed = self._parse_event_row(row=row, sport=sport)
                if parsed is None:
                    continue
                events[parsed.event_id] = parsed

        return OddsApiSnapshot(events=list(events.values()), usage=usage)

    @staticmethod
    def _parse_usage_headers(headers: httpx.Headers) -> OddsApiUsage:
        def _to_int(value: Optional[str]) -> Optional[int]:
            if value is None:
                return None
            try:
                return int(value)
            except ValueError:
                return None

        return OddsApiUsage(
            remaining=_to_int(headers.get("x-requests-remaining")),
            used=_to_int(headers.get("x-requests-used")),
            last=_to_int(headers.get("x-requests-last")),
        )

    def _parse_event_row(
        self,
        row: dict[str, Any],
        sport: str,
    ) -> Optional[BookmakerEvent]:
        home_team = str(row.get("home_team", "")).strip()
        away_team = str(row.get("away_team", "")).strip()
        if not home_team or not away_team:
            return None

        raw_event_id = str(row.get("id", "")).strip()
        starts_at = _parse_datetime(row.get("commence_time"))
        event_id = f"{sport}:{raw_event_id}" if raw_event_id else f"{sport}:{home_team}:{away_team}"

        h2h_rows: list[dict[str, float]] = []
        totals_rows_by_line: dict[float, list[dict[str, float]]] = defaultdict(list)

        bookmakers = row.get("bookmakers", [])
        if not isinstance(bookmakers, list):
            bookmakers = []

        for bookmaker in bookmakers:
            markets = bookmaker.get("markets", [])
            if not isinstance(markets, list):
                continue

            for market in markets:
                key = str(market.get("key", "")).lower().strip()
                outcomes = market.get("outcomes", [])
                if not isinstance(outcomes, list):
                    continue

                if key == "h2h":
                    probs = self._extract_h2h_probabilities(
                        outcomes=outcomes,
                        home_team=home_team,
                        away_team=away_team,
                    )
                    if probs:
                        h2h_rows.append(probs)
                elif key == "totals":
                    line, probs = self._extract_totals_probabilities(outcomes)
                    if line is not None and probs:
                        totals_rows_by_line[line].append(probs)

        markets: list[BookmakerMarket] = []
        if h2h_rows:
            aggregated = self._aggregate_probability_rows(h2h_rows)
            if "home" in aggregated and "away" in aggregated:
                markets.append(
                    BookmakerMarket(
                        market_type="moneyline",
                        outcomes=aggregated,
                        market_id=f"{event_id}:h2h",
                    )
                )

        for line, rows in sorted(totals_rows_by_line.items()):
            aggregated = self._aggregate_probability_rows(rows)
            if "over" in aggregated and "under" in aggregated:
                markets.append(
                    BookmakerMarket(
                        market_type="totals",
                        outcomes=aggregated,
                        line=float(line),
                        market_id=f"{event_id}:totals:{line:.2f}",
                    )
                )

        if not markets:
            return None

        return BookmakerEvent(
            event_id=event_id,
            home_team=home_team,
            away_team=away_team,
            starts_at=starts_at,
            sport=sport,
            league=str(row.get("sport_title", "")).strip(),
            markets=markets,
        )

    def _extract_h2h_probabilities(
        self,
        outcomes: list[dict[str, Any]],
        home_team: str,
        away_team: str,
    ) -> dict[str, float]:
        home_norm = self._normalizer.normalize_team(home_team)
        away_norm = self._normalizer.normalize_team(away_team)
        odds_by_key: dict[str, float] = {}

        for outcome in outcomes:
            name = str(outcome.get("name", "")).strip()
            odds = _safe_decimal_odds(outcome.get("price"))
            if not name or odds is None:
                continue

            name_lower = name.lower().strip()
            normalized = self._normalizer.normalize_team(name)
            canonical: Optional[str] = None

            if name_lower in {"draw", "tie"}:
                canonical = "draw"
            elif normalized == home_norm:
                canonical = "home"
            elif normalized == away_norm:
                canonical = "away"
            else:
                sim_home = self._normalizer.calculate_similarity(normalized, home_norm)
                sim_away = self._normalizer.calculate_similarity(normalized, away_norm)
                if sim_home >= sim_away and sim_home >= 0.55:
                    canonical = "home"
                elif sim_away > sim_home and sim_away >= 0.55:
                    canonical = "away"

            if canonical is not None and canonical not in odds_by_key:
                odds_by_key[canonical] = odds

        if "home" not in odds_by_key or "away" not in odds_by_key:
            return {}

        return _devig(odds_by_key)

    @staticmethod
    def _extract_totals_probabilities(
        outcomes: list[dict[str, Any]],
    ) -> tuple[Optional[float], dict[str, float]]:
        odds_by_key: dict[str, float] = {}
        line: Optional[float] = None

        for outcome in outcomes:
            name = str(outcome.get("name", "")).lower().strip()
            odds = _safe_decimal_odds(outcome.get("price"))
            if odds is None:
                continue

            point = outcome.get("point")
            try:
                parsed_line = float(point) if point is not None else None
            except (TypeError, ValueError):
                parsed_line = None

            if name == "over":
                odds_by_key["over"] = odds
                if parsed_line is not None:
                    line = parsed_line
            elif name == "under":
                odds_by_key["under"] = odds
                if parsed_line is not None:
                    line = parsed_line

        if line is None or "over" not in odds_by_key or "under" not in odds_by_key:
            return None, {}

        return line, _devig(odds_by_key)

    @staticmethod
    def _aggregate_probability_rows(rows: list[dict[str, float]]) -> dict[str, float]:
        if not rows:
            return {}

        by_key: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            for key, value in row.items():
                if value > 0:
                    by_key[key].append(value)

        if not by_key:
            return {}

        aggregated = {key: median(values) for key, values in by_key.items() if values}
        return _normalize_probabilities(aggregated)
