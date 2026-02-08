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


def _normalize_rn1_row_type(row: dict[str, Any]) -> str:
    row_type = str(row.get("type") or "").upper()
    if row_type:
        return row_type
    side = str(row.get("side") or "").upper()
    price = _safe_float(row.get("price"), default=0.0)
    if side in {"BUY", "SELL"} and price > 0:
        return "TRADE"
    return "UNKNOWN"


def _normalize_rn1_side(row: dict[str, Any], row_type: str) -> str:
    if row_type == "MERGE":
        return "MERGE"
    side = str(row.get("side") or "").upper()
    if side in {"BUY", "SELL"}:
        return side
    if row_type == "REDEEM":
        return "REDEEM"
    return "UNKNOWN"


def _market_type_from_title(title: str) -> str:
    t = (title or "").lower()
    if "both teams to score" in t:
        return "btts"
    if "o/u" in t or "over" in t or "under" in t:
        return "over_under"
    if "draw" in t:
        return "draw"
    if "map " in t:
        return "map"
    if "set " in t:
        return "set"
    if "will " in t and " win on " in t:
        return "winner"
    return "other"


def _league_prefix_from_event_slug(event_slug: str) -> str:
    raw = (event_slug or "").strip()
    if not raw:
        return "unknown"
    return raw.split("-", 1)[0]


def _price_bucket(price: float) -> str:
    if price <= 0:
        return "n/a"
    if price < 0.1:
        return "<0.10"
    if price < 0.2:
        return "0.10-0.20"
    if price < 0.4:
        return "0.20-0.40"
    if price < 0.6:
        return "0.40-0.60"
    if price < 0.8:
        return "0.60-0.80"
    if price < 0.9:
        return "0.80-0.90"
    return ">=0.90"


def fetch_rn1_raw_activity(
    *,
    wallet: str,
    window_hours: float,
    page_limit: int = 500,
    max_pages: int = 7,
    timeout_seconds: float = 25.0,
) -> list[dict[str, Any]]:
    """Fetch and normalize RN1 activity rows (TRADE/MERGE/REDEEM)."""
    cutoff_ts = int((datetime.now(timezone.utc) - timedelta(hours=max(0.1, window_hours))).timestamp())
    rows: list[dict[str, Any]] = []

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
                row_type = _normalize_rn1_row_type(row)
                side = _normalize_rn1_side(row, row_type)
                if row_type == "UNKNOWN":
                    continue
                usdc_size = _safe_float(row.get("usdcSize") or row.get("size"))
                price = _safe_float(row.get("price"))
                shares = (usdc_size / price) if (row_type == "TRADE" and price > 0 and usdc_size > 0) else 0.0
                rows.append(
                    {
                        "timestamp": ts,
                        "datetime_utc": datetime.fromtimestamp(ts, timezone.utc).isoformat(),
                        "type": row_type,
                        "side": side,
                        "condition_id": str(row.get("conditionId") or ""),
                        "title": str(row.get("title") or ""),
                        "slug": str(row.get("slug") or ""),
                        "event_slug": str(row.get("eventSlug") or ""),
                        "league_prefix": _league_prefix_from_event_slug(str(row.get("eventSlug") or "")),
                        "market_type": _market_type_from_title(str(row.get("title") or "")),
                        "outcome": str(row.get("outcome") or ""),
                        "price": price,
                        "price_bucket": _price_bucket(price),
                        "usdc_size": usdc_size,
                        "shares": shares,
                        "tx_hash": str(row.get("transactionHash") or ""),
                    }
                )

            if stop or len(payload) < page_limit:
                break

    # Deduplicate by tx + condition + type + timestamp + outcome + size + price.
    seen: set[tuple[Any, ...]] = set()
    deduped: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: item["timestamp"]):
        key = (
            row.get("tx_hash"),
            row.get("condition_id"),
            row.get("type"),
            row.get("timestamp"),
            row.get("outcome"),
            row.get("usdc_size"),
            row.get("price"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


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
    out: list[ActivityEvent] = []
    rows = fetch_rn1_raw_activity(
        wallet=wallet,
        window_hours=window_hours,
        page_limit=page_limit,
        max_pages=max_pages,
        timeout_seconds=timeout_seconds,
    )
    for row in rows:
        row_type = str(row.get("type") or "").upper()
        side = str(row.get("side") or "").upper()
        if row_type not in {"TRADE", "MERGE"}:
            continue
        if side not in {"BUY", "SELL", "MERGE"}:
            continue
        out.append(
            ActivityEvent(
                timestamp=int(row.get("timestamp") or 0),
                condition_id=str(row.get("condition_id") or ""),
                outcome=str(row.get("outcome") or ""),
                side=side,
                size_usd=_safe_float(row.get("usdc_size")),
                reason=row_type.lower(),
                strategy_tag="RN1",
            )
        )
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


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    qq = max(0.0, min(1.0, q))
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * qq))
    return float(ordered[idx])


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


def build_rn1_transaction_report(
    *,
    window_hours: float = 6.0,
    rn1_wallet: str = DEFAULT_RN1_WALLET,
    page_limit: int = 500,
    max_pages: int = 7,
    include_transactions: bool = False,
    transaction_limit: int = 2000,
    top_conditions: int = 50,
) -> dict[str, Any]:
    """Build a deep RN1 transaction-level report."""
    safe_window = max(0.1, float(window_hours))
    rows = fetch_rn1_raw_activity(
        wallet=rn1_wallet,
        window_hours=safe_window,
        page_limit=page_limit,
        max_pages=max_pages,
    )
    if not rows:
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "window_hours": safe_window,
            "filters": {
                "rn1_wallet": rn1_wallet,
                "page_limit": int(page_limit),
                "max_pages": int(max_pages),
            },
            "summary": {
                "events_total": 0,
                "trade_count": 0,
                "merge_count": 0,
                "redeem_count": 0,
            },
            "conditions": [],
            "transactions": [],
        }

    type_counts: Counter[str] = Counter()
    side_counts: Counter[str] = Counter()
    league_counts: Counter[str] = Counter()
    market_type_counts: Counter[str] = Counter()
    price_bucket_counts: Counter[str] = Counter()
    outcome_counts: Counter[str] = Counter()
    tx_batch_sizes: Counter[str] = Counter(row.get("tx_hash") for row in rows if row.get("tx_hash"))
    per_minute_counts: Counter[int] = Counter(int(row["timestamp"]) // 60 for row in rows)

    cond_stats: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_type = str(row.get("type") or "UNKNOWN")
        side = str(row.get("side") or "UNKNOWN")
        condition_id = str(row.get("condition_id") or "")
        price = _safe_float(row.get("price"))
        usdc_size = _safe_float(row.get("usdc_size"))
        shares = _safe_float(row.get("shares"))
        outcome = str(row.get("outcome") or "")

        type_counts[row_type] += 1
        side_counts[side] += 1
        league_counts[str(row.get("league_prefix") or "unknown")] += 1
        market_type_counts[str(row.get("market_type") or "other")] += 1
        price_bucket_counts[str(row.get("price_bucket") or "n/a")] += 1
        if outcome:
            outcome_counts[outcome] += 1

        if not condition_id:
            continue
        stats = cond_stats.get(condition_id)
        if stats is None:
            stats = {
                "condition_id": condition_id,
                "title": str(row.get("title") or ""),
                "event_slug": str(row.get("event_slug") or ""),
                "league_prefix": str(row.get("league_prefix") or "unknown"),
                "market_type": str(row.get("market_type") or "other"),
                "events": 0,
                "trade_count": 0,
                "merge_count": 0,
                "redeem_count": 0,
                "buy_count": 0,
                "sell_count": 0,
                "buy_usdc": 0.0,
                "sell_usdc": 0.0,
                "merge_usdc": 0.0,
                "redeem_usdc": 0.0,
                "trade_prices": [],
                "trade_sizes": [],
                "outcomes_seen": set(),
                "outcome_acc": defaultdict(lambda: {"shares": 0.0, "cost": 0.0, "trades": 0}),
                "last_trade_ts": 0,
                "merge_lags": [],
            }
            cond_stats[condition_id] = stats

        stats["events"] += 1
        if row_type == "TRADE":
            stats["trade_count"] += 1
            if side == "BUY":
                stats["buy_count"] += 1
                stats["buy_usdc"] += usdc_size
            elif side == "SELL":
                stats["sell_count"] += 1
                stats["sell_usdc"] += usdc_size
            if usdc_size > 0:
                stats["trade_sizes"].append(usdc_size)
            if price > 0:
                stats["trade_prices"].append(price)
            if outcome:
                stats["outcomes_seen"].add(outcome)
                if shares > 0:
                    slot = stats["outcome_acc"][outcome]
                    slot["shares"] += shares
                    slot["cost"] += usdc_size
                    slot["trades"] += 1
            stats["last_trade_ts"] = int(row.get("timestamp") or 0)
        elif row_type == "MERGE":
            stats["merge_count"] += 1
            stats["merge_usdc"] += usdc_size
            ts = int(row.get("timestamp") or 0)
            last_trade_ts = int(stats.get("last_trade_ts") or 0)
            if ts > 0 and last_trade_ts > 0 and ts >= last_trade_ts:
                stats["merge_lags"].append(ts - last_trade_ts)
        elif row_type == "REDEEM":
            stats["redeem_count"] += 1
            stats["redeem_usdc"] += usdc_size

    condition_rows: list[dict[str, Any]] = []
    for condition_id, stats in cond_stats.items():
        outcomes_seen = set(stats["outcomes_seen"])
        is_multi_outcome = len(outcomes_seen) >= 2
        pair_cost_est = None
        complete_set_shares_est = 0.0
        locked_edge_est = 0.0
        locked_pnl_est = 0.0

        if is_multi_outcome:
            # Use two largest-share outcomes to estimate complete-set economics.
            outcomes_by_shares = sorted(
                (
                    (outcome, data["shares"], data["cost"])
                    for outcome, data in stats["outcome_acc"].items()
                    if data["shares"] > 0
                ),
                key=lambda item: item[1],
                reverse=True,
            )
            if len(outcomes_by_shares) >= 2:
                (_, s1, c1), (_, s2, c2) = outcomes_by_shares[:2]
                avg1 = (c1 / s1) if s1 > 0 else 0.0
                avg2 = (c2 / s2) if s2 > 0 else 0.0
                pair_cost_est = avg1 + avg2
                complete_set_shares_est = min(s1, s2)
                locked_edge_est = max(0.0, 1.0 - pair_cost_est)
                locked_pnl_est = complete_set_shares_est * locked_edge_est

        trade_prices = stats["trade_prices"]
        trade_sizes = stats["trade_sizes"]
        merge_lags = stats["merge_lags"]
        condition_rows.append(
            {
                "condition_id": condition_id,
                "title": stats["title"],
                "event_slug": stats["event_slug"],
                "league_prefix": stats["league_prefix"],
                "market_type": stats["market_type"],
                "events": int(stats["events"]),
                "trade_count": int(stats["trade_count"]),
                "merge_count": int(stats["merge_count"]),
                "redeem_count": int(stats["redeem_count"]),
                "buy_count": int(stats["buy_count"]),
                "sell_count": int(stats["sell_count"]),
                "buy_usdc": float(stats["buy_usdc"]),
                "sell_usdc": float(stats["sell_usdc"]),
                "merge_usdc": float(stats["merge_usdc"]),
                "redeem_usdc": float(stats["redeem_usdc"]),
                "outcomes_seen": sorted(outcomes_seen),
                "is_multi_outcome": bool(is_multi_outcome),
                "median_trade_price": median(trade_prices) if trade_prices else 0.0,
                "median_trade_usdc": median(trade_sizes) if trade_sizes else 0.0,
                "pair_cost_est": pair_cost_est,
                "complete_set_shares_est": float(complete_set_shares_est),
                "locked_edge_est": float(locked_edge_est),
                "locked_pnl_est": float(locked_pnl_est),
                "merge_lag_sec_median": median(merge_lags) if merge_lags else None,
            }
        )

    condition_rows.sort(
        key=lambda item: (item.get("locked_pnl_est") or 0.0, item.get("buy_usdc") or 0.0),
        reverse=True,
    )

    enriched_rows: list[dict[str, Any]] = []
    if include_transactions:
        prev_ts: Optional[int] = None
        prev_cond_ts: dict[str, int] = {}
        cond_event_idx: Counter[str] = Counter()
        cond_trade_idx: Counter[str] = Counter()
        cond_trade_usdc_cum: defaultdict[str, float] = defaultdict(float)
        cond_outcomes_seen_cum: dict[str, set[str]] = defaultdict(set)
        cond_last_trade_ts: dict[str, int] = {}
        multi_cond_set = {
            str(item["condition_id"])
            for item in condition_rows
            if bool(item.get("is_multi_outcome"))
        }

        for idx, row in enumerate(rows, start=1):
            condition_id = str(row.get("condition_id") or "")
            row_type = str(row.get("type") or "UNKNOWN")
            side = str(row.get("side") or "UNKNOWN")
            ts = int(row.get("timestamp") or 0)
            usdc_size = _safe_float(row.get("usdc_size"))
            outcome = str(row.get("outcome") or "")

            cond_event_idx[condition_id] += 1
            if row_type == "TRADE":
                cond_trade_idx[condition_id] += 1
                cond_trade_usdc_cum[condition_id] += usdc_size
                if outcome:
                    cond_outcomes_seen_cum[condition_id].add(outcome)

            seconds_since_prev_event = (ts - prev_ts) if (prev_ts is not None and ts >= prev_ts) else None
            prev_same_cond = prev_cond_ts.get(condition_id)
            seconds_since_prev_condition = (
                (ts - prev_same_cond)
                if (prev_same_cond is not None and ts >= prev_same_cond)
                else None
            )
            prev_ts = ts
            if condition_id:
                prev_cond_ts[condition_id] = ts

            seconds_since_last_trade_same_condition = None
            if row_type == "MERGE" and condition_id in cond_last_trade_ts:
                last_trade_ts = cond_last_trade_ts[condition_id]
                if ts >= last_trade_ts:
                    seconds_since_last_trade_same_condition = ts - last_trade_ts
            if row_type == "TRADE":
                cond_last_trade_ts[condition_id] = ts

            enriched_rows.append(
                {
                    "global_index": idx,
                    "timestamp": ts,
                    "datetime_utc": row.get("datetime_utc"),
                    "type": row_type,
                    "side": side,
                    "condition_id": condition_id,
                    "title": row.get("title"),
                    "event_slug": row.get("event_slug"),
                    "league_prefix": row.get("league_prefix"),
                    "market_type": row.get("market_type"),
                    "outcome": outcome,
                    "price": _safe_float(row.get("price")),
                    "price_bucket": row.get("price_bucket"),
                    "usdc_size": usdc_size,
                    "shares": _safe_float(row.get("shares")),
                    "tx_hash": row.get("tx_hash"),
                    "tx_batch_size": int(tx_batch_sizes.get(str(row.get("tx_hash") or ""), 0)),
                    "condition_event_index": int(cond_event_idx[condition_id]),
                    "condition_trade_index": int(cond_trade_idx[condition_id]),
                    "condition_trade_usdc_cum": float(cond_trade_usdc_cum[condition_id]),
                    "condition_outcomes_seen_cum": int(len(cond_outcomes_seen_cum[condition_id])),
                    "is_multi_outcome_condition": bool(condition_id in multi_cond_set),
                    "seconds_since_prev_event": seconds_since_prev_event,
                    "seconds_since_prev_condition_event": seconds_since_prev_condition,
                    "seconds_since_last_trade_same_condition": seconds_since_last_trade_same_condition,
                }
            )

        if transaction_limit > 0 and len(enriched_rows) > transaction_limit:
            enriched_rows = enriched_rows[-transaction_limit:]

    start_ts = int(rows[0]["timestamp"])
    end_ts = int(rows[-1]["timestamp"])
    span_seconds = max(1, end_ts - start_ts)

    trade_usdc_values = [
        _safe_float(row.get("usdc_size"))
        for row in rows
        if str(row.get("type")) == "TRADE"
    ]
    pair_cost_values = [
        _safe_float(item.get("pair_cost_est"))
        for item in condition_rows
        if item.get("pair_cost_est") is not None
    ]
    locked_edge_values = [
        _safe_float(item.get("locked_edge_est"))
        for item in condition_rows
        if _safe_float(item.get("locked_edge_est")) > 0
    ]
    merge_lag_values = [
        _safe_float(item.get("merge_lag_sec_median"))
        for item in condition_rows
        if item.get("merge_lag_sec_median") is not None
    ]
    trade_count_values = [int(item.get("trade_count") or 0) for item in condition_rows]

    summary = {
        "events_total": len(rows),
        "trade_count": int(type_counts.get("TRADE", 0)),
        "merge_count": int(type_counts.get("MERGE", 0)),
        "redeem_count": int(type_counts.get("REDEEM", 0)),
        "buy_count": int(side_counts.get("BUY", 0)),
        "sell_count": int(side_counts.get("SELL", 0)),
        "buy_share_of_trades": _ratio(
            _safe_float(side_counts.get("BUY", 0)),
            _safe_float(type_counts.get("TRADE", 0)),
        ),
        "merge_share_of_events": _ratio(
            _safe_float(type_counts.get("MERGE", 0)),
            _safe_float(len(rows)),
        ),
        "unique_conditions": len(cond_stats),
        "multi_outcome_conditions": sum(1 for item in condition_rows if bool(item.get("is_multi_outcome"))),
        "multi_outcome_ratio": _ratio(
            sum(1 for item in condition_rows if bool(item.get("is_multi_outcome"))),
            len(cond_stats),
        ),
        "trade_usdc_median": median(trade_usdc_values) if trade_usdc_values else 0.0,
        "trade_usdc_p90": _quantile(trade_usdc_values, 0.90),
        "events_per_minute_window": len(rows) / (safe_window * 60.0),
        "events_per_minute_active": len(rows) / (span_seconds / 60.0),
        "max_events_per_minute": max(per_minute_counts.values()) if per_minute_counts else 0,
        "window_start_ts": start_ts,
        "window_end_ts": end_ts,
        "active_span_seconds": span_seconds,
    }

    league_buy_usdc: Counter[str] = Counter()
    market_type_buy_usdc: Counter[str] = Counter()
    for item in condition_rows:
        buy_usdc = _safe_float(item.get("buy_usdc"))
        league_buy_usdc[str(item.get("league_prefix") or "unknown")] += buy_usdc
        market_type_buy_usdc[str(item.get("market_type") or "other")] += buy_usdc

    total_buy_usdc = sum(league_buy_usdc.values())
    dominant_leagues = league_buy_usdc.most_common(8)
    dominant_market_types = market_type_buy_usdc.most_common(8)
    inferred_league_filter = [
        league
        for league, amt in dominant_leagues
        if total_buy_usdc > 0 and (amt / total_buy_usdc) >= 0.05
    ]
    inferred_market_filter = [
        market_type
        for market_type, amt in dominant_market_types
        if total_buy_usdc > 0 and (amt / total_buy_usdc) >= 0.08
    ]
    method_inference = {
        "dominant_leagues_by_buy_usdc": dict(dominant_leagues),
        "dominant_market_types_by_buy_usdc": dict(dominant_market_types),
        "pair_cost_est_median": median(pair_cost_values) if pair_cost_values else 0.0,
        "pair_cost_est_p25": _quantile(pair_cost_values, 0.25),
        "pair_cost_est_p75": _quantile(pair_cost_values, 0.75),
        "locked_edge_est_median": median(locked_edge_values) if locked_edge_values else 0.0,
        "locked_edge_est_p75": _quantile(locked_edge_values, 0.75),
        "merge_lag_sec_median": median(merge_lag_values) if merge_lag_values else 0.0,
        "condition_trade_count_median": median(trade_count_values) if trade_count_values else 0.0,
        "condition_trade_count_p75": _quantile([float(v) for v in trade_count_values], 0.75),
        "inferred_filters": {
            "league_prefix_in": inferred_league_filter,
            "market_type_in": inferred_market_filter,
            "pair_cost_est_max": _quantile(pair_cost_values, 0.75) if pair_cost_values else 1.0,
            "trade_usdc_median_band": {
                "min": _quantile(trade_usdc_values, 0.25),
                "max": _quantile(trade_usdc_values, 0.75),
            },
        },
        "notes": [
            "RN1 est quasi exclusivement BUY puis MERGE/REDEEM.",
            "Les conditions multi-outcome sont prioritaires pour constituer des complete sets.",
            "Les merges sont utilises pour realiser le lock-in sur le spread Yes+No.",
        ],
    }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window_hours": safe_window,
        "filters": {
            "rn1_wallet": rn1_wallet,
            "page_limit": int(page_limit),
            "max_pages": int(max_pages),
            "include_transactions": bool(include_transactions),
            "transaction_limit": int(transaction_limit),
        },
        "summary": summary,
        "distribution": {
            "type_counts": dict(type_counts),
            "side_counts": dict(side_counts),
            "league_prefix_top": dict(league_counts.most_common(20)),
            "market_type_counts": dict(market_type_counts),
            "price_bucket_counts": dict(price_bucket_counts),
            "outcome_top": dict(outcome_counts.most_common(20)),
        },
        "method_inference": method_inference,
        "conditions": condition_rows[: max(1, int(top_conditions))],
        "transactions": enriched_rows,
    }


def aggregate_local_conditions(
    *,
    db_url: str,
    window_hours: float,
    strategy_tag: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Aggregate local two-sided events per condition."""
    events = load_local_two_sided_events(
        db_url=db_url,
        window_hours=window_hours,
        strategy_tag=strategy_tag,
    )
    by_condition: dict[str, dict[str, Any]] = {}
    for item in events:
        cid = str(item.condition_id or "")
        if not cid:
            continue
        cur = by_condition.get(cid)
        if cur is None:
            cur = {
                "condition_id": cid,
                "events": 0,
                "buy_count": 0,
                "sell_count": 0,
                "buy_usdc": 0.0,
                "sell_usdc": 0.0,
                "total_usdc": 0.0,
                "outcomes_seen": set(),
                "reasons": Counter(),
                "strategy_tags": Counter(),
            }
            by_condition[cid] = cur
        cur["events"] += 1
        cur["total_usdc"] += _safe_float(item.size_usd)
        cur["outcomes_seen"].add(item.outcome or "__unknown__")
        cur["reasons"][item.reason or "unknown"] += 1
        cur["strategy_tags"][item.strategy_tag or "default"] += 1
        if item.side == "BUY":
            cur["buy_count"] += 1
            cur["buy_usdc"] += _safe_float(item.size_usd)
        elif item.side == "SELL":
            cur["sell_count"] += 1
            cur["sell_usdc"] += _safe_float(item.size_usd)

    rows: list[dict[str, Any]] = []
    for cur in by_condition.values():
        rows.append(
            {
                "condition_id": cur["condition_id"],
                "events": int(cur["events"]),
                "buy_count": int(cur["buy_count"]),
                "sell_count": int(cur["sell_count"]),
                "buy_usdc": float(cur["buy_usdc"]),
                "sell_usdc": float(cur["sell_usdc"]),
                "total_usdc": float(cur["total_usdc"]),
                "outcomes_seen_count": int(len(cur["outcomes_seen"])),
                "reasons": dict(cur["reasons"]),
                "strategy_tags": dict(cur["strategy_tags"]),
            }
        )
    rows.sort(key=lambda item: (item.get("events") or 0, item.get("total_usdc") or 0), reverse=True)
    return rows


def build_rn1_vs_local_condition_report(
    *,
    db_url: str,
    window_hours: float = 6.0,
    strategy_tag: Optional[str] = None,
    rn1_wallet: str = DEFAULT_RN1_WALLET,
    page_limit: int = 500,
    max_pages: int = 7,
    top_conditions: int = 100,
) -> dict[str, Any]:
    """Compare RN1 conditions with local conditions by exact condition_id."""
    safe_window = max(0.1, float(window_hours))
    rn1_report = build_rn1_transaction_report(
        window_hours=safe_window,
        rn1_wallet=rn1_wallet,
        page_limit=page_limit,
        max_pages=max_pages,
        include_transactions=False,
        top_conditions=5000,
    )
    rn1_conditions = rn1_report.get("conditions", [])
    local_conditions = aggregate_local_conditions(
        db_url=db_url,
        window_hours=safe_window,
        strategy_tag=strategy_tag,
    )

    rn1_by_cid = {str(item.get("condition_id")): item for item in rn1_conditions}
    local_by_cid = {str(item.get("condition_id")): item for item in local_conditions}
    rn1_ids = set(rn1_by_cid.keys())
    local_ids = set(local_by_cid.keys())
    overlap_ids = rn1_ids & local_ids
    rn1_only_ids = rn1_ids - local_ids
    local_only_ids = local_ids - rn1_ids

    overlap: list[dict[str, Any]] = []
    for cid in overlap_ids:
        r = rn1_by_cid[cid]
        l = local_by_cid[cid]
        overlap.append(
            {
                "condition_id": cid,
                "title": r.get("title"),
                "event_slug": r.get("event_slug"),
                "league_prefix": r.get("league_prefix"),
                "market_type": r.get("market_type"),
                "rn1_trade_count": int(r.get("trade_count") or 0),
                "rn1_merge_count": int(r.get("merge_count") or 0),
                "rn1_buy_usdc": float(r.get("buy_usdc") or 0.0),
                "rn1_locked_edge_est": float(r.get("locked_edge_est") or 0.0),
                "rn1_locked_pnl_est": float(r.get("locked_pnl_est") or 0.0),
                "local_events": int(l.get("events") or 0),
                "local_buy_count": int(l.get("buy_count") or 0),
                "local_sell_count": int(l.get("sell_count") or 0),
                "local_buy_usdc": float(l.get("buy_usdc") or 0.0),
                "local_sell_usdc": float(l.get("sell_usdc") or 0.0),
                "local_total_usdc": float(l.get("total_usdc") or 0.0),
                "local_reason_top": dict(sorted((l.get("reasons") or {}).items(), key=lambda kv: kv[1], reverse=True)[:5]),
                "activity_ratio_local_vs_rn1": _ratio(
                    float(l.get("events") or 0.0),
                    float(r.get("events") or 0.0),
                ),
                "buy_usdc_ratio_local_vs_rn1": _ratio(
                    float(l.get("buy_usdc") or 0.0),
                    float(r.get("buy_usdc") or 0.0),
                ),
            }
        )
    overlap.sort(
        key=lambda item: (item.get("rn1_buy_usdc") or 0.0, item.get("rn1_trade_count") or 0),
        reverse=True,
    )

    rn1_only = [
        {
            "condition_id": cid,
            "title": rn1_by_cid[cid].get("title"),
            "event_slug": rn1_by_cid[cid].get("event_slug"),
            "league_prefix": rn1_by_cid[cid].get("league_prefix"),
            "market_type": rn1_by_cid[cid].get("market_type"),
            "rn1_trade_count": int(rn1_by_cid[cid].get("trade_count") or 0),
            "rn1_merge_count": int(rn1_by_cid[cid].get("merge_count") or 0),
            "rn1_buy_usdc": float(rn1_by_cid[cid].get("buy_usdc") or 0.0),
            "rn1_locked_edge_est": float(rn1_by_cid[cid].get("locked_edge_est") or 0.0),
            "rn1_locked_pnl_est": float(rn1_by_cid[cid].get("locked_pnl_est") or 0.0),
        }
        for cid in rn1_only_ids
    ]
    rn1_only.sort(
        key=lambda item: (item.get("rn1_buy_usdc") or 0.0, item.get("rn1_trade_count") or 0),
        reverse=True,
    )

    local_only = [
        {
            "condition_id": cid,
            "local_events": int(local_by_cid[cid].get("events") or 0),
            "local_buy_count": int(local_by_cid[cid].get("buy_count") or 0),
            "local_sell_count": int(local_by_cid[cid].get("sell_count") or 0),
            "local_buy_usdc": float(local_by_cid[cid].get("buy_usdc") or 0.0),
            "local_total_usdc": float(local_by_cid[cid].get("total_usdc") or 0.0),
            "local_reason_top": dict(
                sorted((local_by_cid[cid].get("reasons") or {}).items(), key=lambda kv: kv[1], reverse=True)[:5]
            ),
        }
        for cid in local_only_ids
    ]
    local_only.sort(
        key=lambda item: (item.get("local_events") or 0, item.get("local_total_usdc") or 0.0),
        reverse=True,
    )

    overlap_ratio_vs_rn1 = _ratio(len(overlap_ids), len(rn1_ids))
    recos: list[str] = []
    if overlap_ratio_vs_rn1 < 0.15:
        recos.append(
            "Tu joues trop peu de conditions RN1 (overlap faible). Priorite: reproduire son universe de conditions avant d'ajuster les edges."
        )
    if overlap:
        median_activity_ratio = median([_safe_float(item.get("activity_ratio_local_vs_rn1")) for item in overlap])
        if median_activity_ratio < 0.20:
            recos.append(
                "Sur les conditions communes, ton intensite est beaucoup plus faible. Augmente la frequence d'execution sur les memes conditions."
            )
    if not recos:
        recos.append("Overlap et intensite deviennent proches. Prochaine etape: comparer la qualite de prix et la latence condition par condition.")

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window_hours": safe_window,
        "filters": {
            "db_url": db_url,
            "strategy_tag": strategy_tag,
            "rn1_wallet": rn1_wallet,
            "page_limit": int(page_limit),
            "max_pages": int(max_pages),
        },
        "summary": {
            "rn1_conditions": len(rn1_ids),
            "local_conditions": len(local_ids),
            "overlap_conditions": len(overlap_ids),
            "rn1_only_conditions": len(rn1_only_ids),
            "local_only_conditions": len(local_only_ids),
            "overlap_ratio_vs_rn1": overlap_ratio_vs_rn1,
        },
        "recommendations": recos,
        "overlap_top": overlap[: max(1, int(top_conditions))],
        "rn1_only_top": rn1_only[: max(1, int(top_conditions))],
        "local_only_top": local_only[: max(1, int(top_conditions))],
    }


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
