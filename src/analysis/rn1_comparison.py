"""Compare local two-sided behavior against RN1 public wallet activity."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from statistics import median
from typing import Any, Optional, Sequence

import httpx
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.models import LiveObservation, PaperTrade

DATA_API_ACTIVITY = "https://data-api.polymarket.com/activity"
DATA_API_ACTIVITY_MAX_OFFSET = 3000
DEFAULT_RN1_WALLET = "0x2005d16a84ceefa912d4e380cd32e7ff827875ea"
TWO_SIDED_EVENT_TYPE = "two_sided_inventory"


@dataclass(slots=True)
class ActivityEvent:
    timestamp: int
    condition_id: str
    outcome: str
    side: str
    size_usd: float
    reason: str
    strategy_tag: str


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_sync_db_url(value: str) -> str:
    raw = (value or "").strip()
    if not raw:
        return "sqlite:///data/arb.db"
    if "://" not in raw:
        return f"sqlite:///{raw}"
    scheme, suffix = raw.split("://", 1)
    if "+" in scheme:
        scheme = scheme.split("+", 1)[0]
    return f"{scheme}://{suffix}"


def _parse_ts(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _dt_to_ts(value: Optional[datetime]) -> int:
    if value is None:
        return 0
    dt = value
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def _rolling_max_events_per_minute(events: Sequence[ActivityEvent]) -> int:
    if not events:
        return 0
    ordered = sorted(events, key=lambda item: item.timestamp)
    left = 0
    best = 0
    for right, current in enumerate(ordered):
        while left <= right and current.timestamp - ordered[left].timestamp > 60:
            left += 1
        size = right - left + 1
        if size > best:
            best = size
    return best


def load_local_two_sided_events(
    *,
    db_url: str,
    window_hours: float,
    strategy_tag: Optional[str],
) -> list[ActivityEvent]:
    """Load local two-sided activity from paper tables."""
    engine = create_engine(_normalize_sync_db_url(db_url))
    session = sessionmaker(bind=engine)()
    cutoff_dt = datetime.utcnow() - timedelta(hours=max(0.1, window_hours))
    try:
        query = (
            session.query(PaperTrade, LiveObservation)
            .join(LiveObservation, LiveObservation.id == PaperTrade.observation_id)
            .filter(LiveObservation.event_type == TWO_SIDED_EVENT_TYPE)
            .filter(PaperTrade.created_at >= cutoff_dt)
            .order_by(PaperTrade.created_at.asc(), PaperTrade.id.asc())
        )
        rows = query.all()
    finally:
        session.close()

    out: list[ActivityEvent] = []
    for trade, observation in rows:
        state = observation.game_state if isinstance(observation.game_state, dict) else {}
        tag = str(state.get("strategy_tag") or "default")
        if strategy_tag and tag != strategy_tag:
            continue
        side = str(trade.side or state.get("side") or "").upper()
        if side not in {"BUY", "SELL"}:
            continue
        ts = _dt_to_ts(trade.created_at or observation.timestamp)
        if ts <= 0:
            continue
        condition_id = str(state.get("condition_id") or observation.match_id or "")
        if not condition_id:
            continue
        outcome = str(state.get("outcome") or "")
        reason = str(state.get("reason") or "")
        out.append(
            ActivityEvent(
                timestamp=ts,
                condition_id=condition_id,
                outcome=outcome,
                side=side,
                size_usd=_safe_float(trade.size),
                reason=reason,
                strategy_tag=tag,
            )
        )
    return out


def fetch_rn1_activity(
    *,
    wallet: str,
    window_hours: float,
    page_limit: int = 500,
    max_pages: int = 7,
    timeout_seconds: float = 25.0,
) -> list[ActivityEvent]:
    """Fetch RN1 activity from Polymarket data API."""
    cutoff_ts = int((datetime.now(timezone.utc) - timedelta(hours=max(0.1, window_hours))).timestamp())
    out: list[ActivityEvent] = []

    with httpx.Client(timeout=timeout_seconds) as client:
        for page in range(max(1, max_pages)):
            offset = page * page_limit
            if offset > DATA_API_ACTIVITY_MAX_OFFSET:
                break
            try:
                resp = client.get(
                    DATA_API_ACTIVITY,
                    params={
                        "user": wallet,
                        "limit": max(1, page_limit),
                        "offset": max(0, offset),
                    },
                )
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                # Activity pagination currently hard-fails (400) beyond a max offset.
                # Stop gracefully and keep data collected so far.
                if exc.response is not None and exc.response.status_code == 400:
                    break
                raise
            payload = resp.json()
            if not isinstance(payload, list) or not payload:
                break

            stop = False
            for row in payload:
                if not isinstance(row, dict):
                    continue
                ts = _parse_ts(row.get("timestamp"))
                if ts <= 0:
                    continue
                if ts < cutoff_ts:
                    stop = True
                    continue
                row_type = str(row.get("type") or "").upper()
                if row_type not in {"TRADE", "MERGE"}:
                    continue
                side = str(row.get("side") or "").upper()
                if row_type == "MERGE":
                    side = "MERGE"
                if side not in {"BUY", "SELL", "MERGE"}:
                    continue
                out.append(
                    ActivityEvent(
                        timestamp=ts,
                        condition_id=str(row.get("conditionId") or ""),
                        outcome=str(row.get("outcome") or ""),
                        side=side,
                        size_usd=_safe_float(row.get("usdcSize") or row.get("size")),
                        reason=row_type.lower(),
                        strategy_tag="RN1",
                    )
                )

            if stop or len(payload) < page_limit:
                break
    return out


def summarize_behavior(
    events: Sequence[ActivityEvent],
    *,
    window_hours: float,
) -> dict[str, Any]:
    """Build behavior metrics from activity events."""
    if not events:
        return {
            "events_total": 0,
            "events_trade": 0,
            "events_merge": 0,
            "buy_count": 0,
            "sell_count": 0,
            "buy_share_of_trades": 0.0,
            "sell_share_of_trades": 0.0,
            "merge_share_of_events": 0.0,
            "unique_conditions": 0,
            "multi_outcome_conditions": 0,
            "multi_outcome_ratio": 0.0,
            "size_total_usd": 0.0,
            "size_mean_usd": 0.0,
            "size_median_usd": 0.0,
            "size_p90_usd": 0.0,
            "cadence_per_minute_window": 0.0,
            "cadence_per_minute_active": 0.0,
            "max_events_per_minute": 0,
            "top_reasons": {},
            "top_tags": {},
            "window_start_ts": 0,
            "window_end_ts": 0,
            "active_span_seconds": 0,
        }

    ordered = sorted(events, key=lambda item: item.timestamp)
    start_ts = ordered[0].timestamp
    end_ts = ordered[-1].timestamp
    span_seconds = max(1, end_ts - start_ts)
    window_seconds = max(1.0, window_hours * 3600.0)

    reason_counter: Counter[str] = Counter()
    tag_counter: Counter[str] = Counter()
    side_counter: Counter[str] = Counter()
    condition_outcomes: dict[str, set[str]] = defaultdict(set)
    sizes: list[float] = []

    for item in ordered:
        reason_counter[item.reason or "unknown"] += 1
        tag_counter[item.strategy_tag or "default"] += 1
        side_counter[item.side] += 1
        if item.condition_id:
            condition_outcomes[item.condition_id].add(item.outcome or "__unknown__")
        if item.size_usd > 0:
            sizes.append(item.size_usd)

    events_total = len(ordered)
    events_merge = side_counter.get("MERGE", 0)
    events_trade = events_total - events_merge
    buy_count = side_counter.get("BUY", 0)
    sell_count = side_counter.get("SELL", 0)

    unique_conditions = len(condition_outcomes)
    multi_outcome_conditions = sum(1 for outcomes in condition_outcomes.values() if len(outcomes) >= 2)
    multi_outcome_ratio = (
        (multi_outcome_conditions / unique_conditions) if unique_conditions > 0 else 0.0
    )

    sizes_sorted = sorted(sizes)
    p90 = 0.0
    if sizes_sorted:
        p90_idx = int(math.ceil(0.90 * len(sizes_sorted))) - 1
        p90 = sizes_sorted[max(0, min(len(sizes_sorted) - 1, p90_idx))]

    return {
        "events_total": events_total,
        "events_trade": events_trade,
        "events_merge": events_merge,
        "buy_count": buy_count,
        "sell_count": sell_count,
        "buy_share_of_trades": (buy_count / events_trade) if events_trade > 0 else 0.0,
        "sell_share_of_trades": (sell_count / events_trade) if events_trade > 0 else 0.0,
        "merge_share_of_events": (events_merge / events_total) if events_total > 0 else 0.0,
        "unique_conditions": unique_conditions,
        "multi_outcome_conditions": multi_outcome_conditions,
        "multi_outcome_ratio": multi_outcome_ratio,
        "size_total_usd": sum(sizes),
        "size_mean_usd": (sum(sizes) / len(sizes)) if sizes else 0.0,
        "size_median_usd": median(sizes) if sizes else 0.0,
        "size_p90_usd": p90,
        "cadence_per_minute_window": events_total / (window_seconds / 60.0),
        "cadence_per_minute_active": events_total / (span_seconds / 60.0),
        "max_events_per_minute": _rolling_max_events_per_minute(ordered),
        "top_reasons": dict(reason_counter.most_common(8)),
        "top_tags": dict(tag_counter.most_common(8)),
        "window_start_ts": start_ts,
        "window_end_ts": end_ts,
        "active_span_seconds": span_seconds,
    }


def _ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def build_gaps(local: dict[str, Any], rn1: dict[str, Any]) -> dict[str, float]:
    """Compute key behavior gaps between local strategy and RN1."""
    return {
        "cadence_window_ratio_local_vs_rn1": _ratio(
            _safe_float(local.get("cadence_per_minute_window")),
            _safe_float(rn1.get("cadence_per_minute_window")),
        ),
        "cadence_active_ratio_local_vs_rn1": _ratio(
            _safe_float(local.get("cadence_per_minute_active")),
            _safe_float(rn1.get("cadence_per_minute_active")),
        ),
        "size_median_ratio_local_vs_rn1": _ratio(
            _safe_float(local.get("size_median_usd")),
            _safe_float(rn1.get("size_median_usd")),
        ),
        "size_mean_ratio_local_vs_rn1": _ratio(
            _safe_float(local.get("size_mean_usd")),
            _safe_float(rn1.get("size_mean_usd")),
        ),
        "multi_outcome_ratio_gap": (
            _safe_float(local.get("multi_outcome_ratio")) - _safe_float(rn1.get("multi_outcome_ratio"))
        ),
        "buy_share_gap": (
            _safe_float(local.get("buy_share_of_trades")) - _safe_float(rn1.get("buy_share_of_trades"))
        ),
        "merge_share_gap": (
            _safe_float(local.get("merge_share_of_events")) - _safe_float(rn1.get("merge_share_of_events"))
        ),
        "unique_conditions_ratio_local_vs_rn1": _ratio(
            _safe_float(local.get("unique_conditions")),
            _safe_float(rn1.get("unique_conditions")),
        ),
    }


def build_recommendations(local: dict[str, Any], rn1: dict[str, Any], gaps: dict[str, float]) -> list[str]:
    """Generate short actionable recommendations from gap metrics."""
    recos: list[str] = []

    cadence_ratio = _safe_float(gaps.get("cadence_window_ratio_local_vs_rn1"))
    if cadence_ratio < 0.25:
        recos.append(
            "Cadence trop basse vs RN1: reduis interval/cooldown et augmente max-orders-per-cycle pour capter plus d'opportunites."
        )

    multi_gap = _safe_float(gaps.get("multi_outcome_ratio_gap"))
    if multi_gap < -0.15:
        recos.append(
            "Coverage deux-cotes trop faible: augmente la priorite des signaux par paire complete (Yes+No) avant les single-leg."
        )

    buy_gap = _safe_float(gaps.get("buy_share_gap"))
    if buy_gap > 0.20:
        recos.append(
            "Trop de BUY vs RN1: renforce les sorties (exit-edge, stale/risk exits, merge) pour recycler l'inventaire."
        )

    local_median = _safe_float(local.get("size_median_usd"))
    rn1_median = _safe_float(rn1.get("size_median_usd"))
    if local_median > 0 and rn1_median > 0 and local_median > 3.0 * rn1_median:
        recos.append(
            "Ticket median trop gros vs RN1: decoupe les entrees en ordres plus petits et plus frequents."
        )

    if not recos:
        recos.append("Les gaps principaux sont reduits. Continue a optimiser la qualite des fills et la vitesse d'execution.")
    return recos


def build_comparison_report(
    *,
    db_url: str,
    window_hours: float = 6.0,
    strategy_tag: Optional[str] = None,
    rn1_wallet: str = DEFAULT_RN1_WALLET,
    page_limit: int = 500,
    max_pages: int = 7,
) -> dict[str, Any]:
    """Build a local-vs-RN1 comparison report."""
    safe_window = max(0.1, float(window_hours))
    local_events = load_local_two_sided_events(
        db_url=db_url,
        window_hours=safe_window,
        strategy_tag=strategy_tag,
    )
    rn1_events = fetch_rn1_activity(
        wallet=rn1_wallet,
        window_hours=safe_window,
        page_limit=page_limit,
        max_pages=max_pages,
    )
    local_summary = summarize_behavior(local_events, window_hours=safe_window)
    rn1_summary = summarize_behavior(rn1_events, window_hours=safe_window)
    gaps = build_gaps(local_summary, rn1_summary)
    recos = build_recommendations(local_summary, rn1_summary, gaps)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window_hours": safe_window,
        "filters": {
            "strategy_tag": strategy_tag,
            "rn1_wallet": rn1_wallet,
            "db_url": db_url,
            "page_limit": int(page_limit),
            "max_pages": int(max_pages),
        },
        "local": local_summary,
        "rn1": rn1_summary,
        "gaps": gaps,
        "recommendations": recos,
    }
