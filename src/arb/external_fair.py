"""External fair-value runtime backed by Odds API.

Manages the full lifecycle of bookmaker-based fair pricing:
fetch snapshots, cache in DB, match conditions, and blend with
market-derived fair values.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
import structlog
from sqlalchemy import select

from src.arb.two_sided_inventory import MarketSnapshot
from src.db.database import get_sync_session, init_db
from src.db.models import OddsApiCache
from src.feeds.odds_api import OddsApiClient, OddsApiSnapshot, OddsApiUsage
from src.matching.bookmaker_matcher import (
    BookmakerEvent,
    BookmakerMarket,
    BookmakerMarketMatch,
    BookmakerMatcher,
)
from src.utils.parsing import (
    _clamp,
    _ensure_sync_db_url,
    _extract_outcome_price_map,
    _normalize_outcome_label,
    _parse_datetime,
    _to_float,
)

logger = structlog.get_logger()


@dataclass(slots=True)
class ExternalFairStats:
    odds_events: int = 0
    matched_conditions: int = 0
    applied_conditions: int = 0
    credits_remaining: Optional[int] = None
    credits_used: Optional[int] = None
    credits_last_call: Optional[int] = None


def _find_market_by_id(event: BookmakerEvent, market_id: str) -> Optional[BookmakerMarket]:
    for market in event.markets:
        if market.market_id == market_id:
            return market
    return None


def _resolve_probability(odds_map: dict[str, float], mapping_key: str) -> Optional[float]:
    if not mapping_key:
        return None
    if mapping_key.startswith("1-"):
        base_key = mapping_key[2:]
        base = odds_map.get(base_key)
        if base is None:
            return None
        return _clamp(1.0 - float(base), 0.001, 0.999)
    value = odds_map.get(mapping_key)
    if value is None:
        return None
    return _clamp(float(value), 0.001, 0.999)


def _lookup_outcome_price(outcome_prices: dict[str, float], outcome: str) -> Optional[float]:
    direct = outcome_prices.get(outcome)
    if direct is not None:
        return direct
    target = _normalize_outcome_label(outcome)
    for key, price in outcome_prices.items():
        if _normalize_outcome_label(key) == target:
            return price
    return None


def _build_gamma_timing_fair(
    *,
    snapshot: MarketSnapshot,
    raw_market: Optional[dict[str, Any]],
    now_ts: float,
    min_prob: float,
    min_gap: float,
    require_ended: bool,
) -> Optional[dict[str, float]]:
    if raw_market is None:
        return None

    if require_ended:
        end_dt = _parse_datetime(raw_market.get("endDate"))
        if end_dt is None or now_ts < end_dt.timestamp():
            return None

    probs = _extract_outcome_price_map(raw_market)
    if len(probs) < 2:
        return None

    out_a, out_b = snapshot.outcome_order[:2]
    p_a = probs.get(out_a)
    p_b = probs.get(out_b)
    if p_a is None or p_b is None:
        return None

    max_p = max(p_a, p_b)
    gap = abs(p_a - p_b)
    if max_p < min_prob or gap < min_gap:
        return None

    winner = out_a if p_a >= p_b else out_b
    loser = out_b if winner == out_a else out_a
    return {winner: 0.999, loser: 0.001}


class ExternalFairRuntime:
    """Odds API + matcher runtime cache for low-credit operation."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        sports: list[str],
        regions: str,
        markets: str,
        min_refresh_seconds: float,
        min_match_confidence: float,
        blend: float,
        shared_cache_db_url: str = "",
        shared_cache_ttl_seconds: float = 0.0,
    ) -> None:
        self._sports = sports or ["upcoming"]
        self._regions = regions
        self._markets = markets
        self._min_refresh_seconds = max(1.0, min_refresh_seconds)
        self._blend = _clamp(blend, 0.0, 1.0)

        self._odds = OddsApiClient(api_key=api_key, base_url=base_url)
        self._matcher = BookmakerMatcher(min_event_confidence=min_match_confidence)
        self._shared_cache_db_url = (
            _ensure_sync_db_url(shared_cache_db_url) if shared_cache_db_url else ""
        )
        self._shared_cache_ttl_seconds = max(0.0, shared_cache_ttl_seconds)
        sports_key = ",".join(sorted(self._sports))
        cache_spec = f"{base_url}|sports:{sports_key}|regions:{regions}|markets:{markets}"
        self._cache_key = f"odds:{hashlib.sha1(cache_spec.encode('utf-8')).hexdigest()}"
        if self._shared_cache_db_url and self._shared_cache_ttl_seconds > 0:
            init_db(self._shared_cache_db_url)

        self._last_refresh_ts: float = 0.0
        self._events_by_id: dict[str, BookmakerEvent] = {}
        self._matches_by_condition: dict[str, BookmakerMarketMatch] = {}
        self._stats = ExternalFairStats()

    @staticmethod
    def _event_to_payload(event: BookmakerEvent) -> dict[str, Any]:
        return {
            "event_id": event.event_id,
            "home_team": event.home_team,
            "away_team": event.away_team,
            "starts_at": event.starts_at.isoformat() if event.starts_at else None,
            "sport": event.sport,
            "league": event.league,
            "markets": [
                {
                    "market_type": market.market_type,
                    "outcomes": {k: float(v) for k, v in market.outcomes.items()},
                    "line": market.line,
                    "market_id": market.market_id,
                }
                for market in event.markets
            ],
        }

    @staticmethod
    def _payload_to_event(payload: dict[str, Any]) -> Optional[BookmakerEvent]:
        event_id = str(payload.get("event_id") or "")
        home_team = str(payload.get("home_team") or "")
        away_team = str(payload.get("away_team") or "")
        if not event_id or not home_team or not away_team:
            return None

        raw_markets = payload.get("markets")
        if not isinstance(raw_markets, list):
            raw_markets = []
        markets: list[BookmakerMarket] = []
        for row in raw_markets:
            if not isinstance(row, dict):
                continue
            raw_outcomes = row.get("outcomes")
            if not isinstance(raw_outcomes, dict):
                continue
            outcomes: dict[str, float] = {}
            for key, value in raw_outcomes.items():
                outcome_name = str(key).strip()
                if not outcome_name:
                    continue
                outcomes[outcome_name] = _to_float(value, default=0.0)
            if not outcomes:
                continue
            markets.append(
                BookmakerMarket(
                    market_type=str(row.get("market_type") or ""),
                    outcomes=outcomes,
                    line=_to_float(row.get("line"), default=0.0)
                    if row.get("line") is not None
                    else None,
                    market_id=str(row.get("market_id") or ""),
                )
            )

        if not markets:
            return None

        return BookmakerEvent(
            event_id=event_id,
            home_team=home_team,
            away_team=away_team,
            starts_at=_parse_datetime(payload.get("starts_at")),
            sport=str(payload.get("sport") or ""),
            league=str(payload.get("league") or ""),
            markets=markets,
        )

    def _load_shared_snapshot_with_usage(self, now_ts: float) -> Optional[OddsApiSnapshot]:
        if not self._shared_cache_db_url or self._shared_cache_ttl_seconds <= 0:
            return None

        session = get_sync_session(self._shared_cache_db_url)
        try:
            row = session.execute(
                select(OddsApiCache).where(OddsApiCache.cache_key == self._cache_key)
            ).scalar_one_or_none()
        finally:
            session.close()

        if row is None or row.fetched_at is None:
            return None
        fetched_at = row.fetched_at
        if fetched_at.tzinfo is None:
            fetched_at = fetched_at.replace(tzinfo=timezone.utc)
        age = now_ts - fetched_at.timestamp()
        if age > self._shared_cache_ttl_seconds:
            return None

        raw_payload = row.payload
        if not isinstance(raw_payload, list):
            return None
        events: list[BookmakerEvent] = []
        for item in raw_payload:
            if not isinstance(item, dict):
                continue
            parsed = self._payload_to_event(item)
            if parsed is not None:
                events.append(parsed)

        usage = OddsApiUsage(
            remaining=row.credits_remaining,
            used=row.credits_used,
            last=row.credits_last_call,
        )
        return OddsApiSnapshot(events=events, usage=usage)

    def _store_shared_snapshot(self, snapshot: OddsApiSnapshot, now_ts: float) -> None:
        if not self._shared_cache_db_url or self._shared_cache_ttl_seconds <= 0:
            return

        payload = [self._event_to_payload(event) for event in snapshot.events]
        fetched_at = datetime.fromtimestamp(now_ts, tz=timezone.utc).replace(tzinfo=None)

        session = get_sync_session(self._shared_cache_db_url)
        try:
            row = session.execute(
                select(OddsApiCache).where(OddsApiCache.cache_key == self._cache_key)
            ).scalar_one_or_none()
            if row is None:
                row = OddsApiCache(
                    cache_key=self._cache_key,
                    payload=payload,
                    credits_remaining=snapshot.usage.remaining,
                    credits_used=snapshot.usage.used,
                    credits_last_call=snapshot.usage.last,
                    fetched_at=fetched_at,
                )
                session.add(row)
            else:
                row.payload = payload
                row.credits_remaining = snapshot.usage.remaining
                row.credits_used = snapshot.usage.used
                row.credits_last_call = snapshot.usage.last
                row.fetched_at = fetched_at
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def _apply_snapshot(
        self,
        snapshot: OddsApiSnapshot,
        raw_markets: list[dict[str, Any]],
    ) -> None:
        self._events_by_id = {event.event_id: event for event in snapshot.events}
        matches = self._matcher.match_all(raw_markets, snapshot.events)
        self._matches_by_condition = {match.condition_id: match for match in matches}
        self._stats = ExternalFairStats(
            odds_events=len(snapshot.events),
            matched_conditions=len(matches),
            applied_conditions=0,
            credits_remaining=snapshot.usage.remaining,
            credits_used=snapshot.usage.used,
            credits_last_call=snapshot.usage.last,
        )

    async def refresh_if_needed(
        self,
        client: httpx.AsyncClient,
        raw_markets: list[dict[str, Any]],
        now_ts: float,
        force: bool = False,
    ) -> None:
        needs_refresh = force or (now_ts - self._last_refresh_ts >= self._min_refresh_seconds)
        if not needs_refresh and self._events_by_id:
            return

        cached = self._load_shared_snapshot_with_usage(now_ts)
        if cached is not None:
            self._last_refresh_ts = now_ts
            self._apply_snapshot(cached, raw_markets)
            logger.info(
                "external_fair_refreshed",
                source="db_cache",
                sports=self._sports,
                odds_events=self._stats.odds_events,
                matched_conditions=self._stats.matched_conditions,
                credits_remaining=self._stats.credits_remaining,
                credits_used=self._stats.credits_used,
                credits_last_call=self._stats.credits_last_call,
            )
            return

        snapshot: OddsApiSnapshot = await self._odds.fetch_events(
            client=client,
            sports=self._sports,
            regions=self._regions,
            markets=self._markets,
        )
        self._last_refresh_ts = now_ts
        self._store_shared_snapshot(snapshot, now_ts)
        self._apply_snapshot(snapshot, raw_markets)
        logger.info(
            "external_fair_refreshed",
            source="api",
            sports=self._sports,
            odds_events=self._stats.odds_events,
            matched_conditions=self._stats.matched_conditions,
            credits_remaining=self._stats.credits_remaining,
            credits_used=self._stats.credits_used,
            credits_last_call=self._stats.credits_last_call,
        )

    def fair_for_snapshot(
        self,
        snapshot: MarketSnapshot,
        market_fair: dict[str, float],
    ) -> Optional[dict[str, float]]:
        match = self._matches_by_condition.get(snapshot.condition_id)
        if match is None:
            return None

        event = self._events_by_id.get(match.bookmaker_event_id)
        if event is None:
            return None
        market = _find_market_by_id(event, match.bookmaker_market_id)
        if market is None:
            return None

        external: dict[str, float] = {}
        for outcome in snapshot.outcome_order:
            mapping_key = match.outcome_map.get(outcome, "")
            probability = _resolve_probability(market.outcomes, mapping_key)
            if probability is None:
                return None
            external[outcome] = probability

        # Keep binary outcomes normalized.
        total = sum(external.values())
        if total > 0:
            external = {k: _clamp(v / total, 0.001, 0.999) for k, v in external.items()}

        if self._blend < 1.0:
            external = {
                outcome: _clamp(
                    self._blend * external[outcome] + (1.0 - self._blend) * market_fair[outcome],
                    0.001,
                    0.999,
                )
                for outcome in snapshot.outcome_order
            }

        self._stats.applied_conditions += 1
        return external

    @property
    def stats(self) -> ExternalFairStats:
        return self._stats
