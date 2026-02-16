"""Slot analytics API — mounted on the main trades API.

Provides aggregated slot data for the ML dashboard:
heatmap (timing x move -> WR), calibration, per-symbol stats.

Supports filtering by slot duration (5m / 15m) via ?duration= parameter.

Note: by-hour and by-day queries use MySQL functions (FROM_UNIXTIME, HOUR,
WEEKDAY). These endpoints require a MySQL-backed DATABASE_URL.
"""

from __future__ import annotations

import time
from typing import Optional

from fastapi import APIRouter, Query
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from config.settings import settings

router = APIRouter(tags=["slots"])

# Lazy DB connection
_engine = None
_factory = None

_DURATION_MAP = {"5m": 300, "15m": 900}


def _get_factory() -> Optional[async_sessionmaker]:
    global _engine, _factory
    if _factory is not None:
        return _factory
    url = settings.DATABASE_URL
    if not url:
        return None
    kwargs = {"pool_pre_ping": True, "echo": False}
    if "mysql" in url or "mariadb" in url:
        kwargs["pool_size"] = 3
    _engine = create_async_engine(url, **kwargs)
    _factory = async_sessionmaker(bind=_engine, class_=AsyncSession, expire_on_commit=False)
    return _factory


def _wr(wins, total) -> float:
    return round(int(wins or 0) / int(total) * 100, 1) if total else 0.0


def _duration_clause(prefix: str, duration: Optional[str]) -> str:
    """Build SQL clause for slot_duration filtering."""
    if duration and duration in _DURATION_MAP:
        return f"AND {prefix}.slot_duration = :slot_duration"
    return ""


def _duration_params(duration: Optional[str]) -> dict:
    """Return params dict entry for slot_duration if needed."""
    if duration and duration in _DURATION_MAP:
        return {"slot_duration": _DURATION_MAP[duration]}
    return {}


def _timing_case(duration: Optional[str]) -> tuple[str, float]:
    """Return (SQL CASE expression, max_minutes) for timing bins.

    5m slots  -> 1-minute bins: 0-1, 1-2, 2-3, 3-4, 4-5
    15m slots -> 2-minute bins: 0-2, 2-4, 4-6, 6-8, 8-10, 10-12
    """
    if duration == "5m":
        return (
            """CASE
                WHEN ss.minutes_into_slot < 1 THEN '0-1'
                WHEN ss.minutes_into_slot < 2 THEN '1-2'
                WHEN ss.minutes_into_slot < 3 THEN '2-3'
                WHEN ss.minutes_into_slot < 4 THEN '3-4'
                ELSE '4-5'
            END""",
            5.0,
        )
    # Default: 15m bins (also used for "all")
    return (
        """CASE
            WHEN ss.minutes_into_slot < 2 THEN '0-2'
            WHEN ss.minutes_into_slot < 4 THEN '2-4'
            WHEN ss.minutes_into_slot < 6 THEN '4-6'
            WHEN ss.minutes_into_slot < 8 THEN '6-8'
            WHEN ss.minutes_into_slot < 10 THEN '8-10'
            ELSE '10-12'
        END""",
        12.0,
    )


def _calibration_window(duration: Optional[str]) -> tuple[float, float]:
    """Return (min_minutes, max_minutes) for calibration snapshot window.

    5m  -> 1-4 min (mid-slot for 5m)
    15m -> 4-10 min (existing)
    """
    if duration == "5m":
        return (1.0, 4.0)
    return (4.0, 10.0)


@router.get("/slots")
async def slot_analytics(
    hours: float = Query(default=168.0, ge=1.0, le=2160.0, description="Lookback window in hours."),
    symbol: Optional[str] = Query(default=None, description="Filter by symbol (BTC, ETH, SOL, XRP)."),
    duration: Optional[str] = Query(default=None, description="Slot duration filter: 5m, 15m, or null for all."),
    start_ts: Optional[int] = Query(default=None, description="Absolute start unix timestamp (overrides hours)."),
    end_ts: Optional[int] = Query(default=None, description="Absolute end unix timestamp (overrides hours)."),
) -> dict:
    """Slot analytics for ML dashboard: heatmap, calibration, per-symbol stats."""
    factory = _get_factory()
    if factory is None:
        return {"error": "DATABASE_URL not configured"}

    cutoff = start_ts if start_ts is not None else int(time.time() - hours * 3600)
    end_clause_sr = "AND sr.slot_ts <= :end_ts" if end_ts is not None else ""
    end_clause_ss = "AND ss.slot_ts <= :end_ts" if end_ts is not None else ""
    sym_clause = "AND sr.symbol = :symbol" if symbol else ""
    sym_clause_ss = "AND ss.symbol = :symbol" if symbol else ""
    dur_clause_sr = _duration_clause("sr", duration)
    dur_clause_ss = _duration_clause("ss", duration)
    params: dict = {"cutoff": cutoff, **_duration_params(duration)}
    if end_ts is not None:
        params["end_ts"] = end_ts
    if symbol:
        params["symbol"] = symbol.upper()

    timing_case, max_minutes = _timing_case(duration)
    cal_min, cal_max = _calibration_window(duration)

    try:
        async with factory() as session:
            # --- Summary ---
            row = (await session.execute(text(f"""
                SELECT COUNT(*) total,
                       SUM(resolved_up IS NOT NULL) resolved,
                       SUM(resolved_up IS NULL) unresolved,
                       MIN(slot_ts) first_ts, MAX(slot_ts) last_ts
                FROM slot_resolutions sr
                WHERE slot_ts > :cutoff {end_clause_sr} {sym_clause} {dur_clause_sr}
            """), params)).mappings().first()

            total = int(row["total"] or 0)
            if total == 0:
                return {
                    "total_slots": 0, "resolved": 0, "unresolved": 0,
                    "snapshot_count": 0, "by_symbol": [], "heatmap": [],
                    "calibration": [], "by_hour": [], "by_day": [],
                    "duration": duration,
                }

            # --- Snapshot count ---
            snap = (await session.execute(text(f"""
                SELECT COUNT(*) cnt FROM slot_snapshots ss
                WHERE slot_ts > :cutoff {end_clause_ss} {sym_clause_ss} {dur_clause_ss}
            """), params)).mappings().first()

            # --- By symbol ---
            sym_rows = (await session.execute(text(f"""
                SELECT symbol, COUNT(*) total,
                       SUM(resolved_up = 1) wins
                FROM slot_resolutions sr
                WHERE slot_ts > :cutoff {end_clause_sr} AND resolved_up IS NOT NULL {sym_clause} {dur_clause_sr}
                GROUP BY symbol ORDER BY symbol
            """), params)).mappings().all()

            # --- Heatmap: timing x move -> WR ---
            hm_rows = (await session.execute(text(f"""
                SELECT
                    {timing_case} timing_bin,
                    CASE
                        WHEN ss.dir_move_pct < -0.2 THEN '< -0.2'
                        WHEN ss.dir_move_pct < -0.1 THEN '-0.2/-0.1'
                        WHEN ss.dir_move_pct < 0 THEN '-0.1/0'
                        WHEN ss.dir_move_pct < 0.1 THEN '0/0.1'
                        WHEN ss.dir_move_pct < 0.2 THEN '0.1/0.2'
                        ELSE '> 0.2'
                    END move_bin,
                    COUNT(*) total,
                    SUM(sr.resolved_up = 1) wins
                FROM slot_snapshots ss
                JOIN slot_resolutions sr
                    ON ss.symbol = sr.symbol AND ss.slot_ts = sr.slot_ts
                    AND ss.slot_duration = sr.slot_duration
                WHERE sr.resolved_up IS NOT NULL
                    AND ss.slot_ts > :cutoff {end_clause_ss}
                    AND ss.minutes_into_slot <= :max_minutes
                    AND ss.dir_move_pct IS NOT NULL
                    {sym_clause} {dur_clause_sr}
                GROUP BY timing_bin, move_bin
            """), {**params, "max_minutes": max_minutes})).mappings().all()

            # --- Calibration: bid_up vs actual P(up) ---
            cal_rows = (await session.execute(text(f"""
                SELECT
                    ROUND(ss.bid_up * 20) / 20 bid_bucket,
                    AVG(ss.bid_up) avg_bid,
                    COUNT(*) total,
                    SUM(sr.resolved_up = 1) wins
                FROM slot_snapshots ss
                JOIN slot_resolutions sr
                    ON ss.symbol = sr.symbol AND ss.slot_ts = sr.slot_ts
                    AND ss.slot_duration = sr.slot_duration
                WHERE sr.resolved_up IS NOT NULL
                    AND ss.bid_up IS NOT NULL
                    AND ss.bid_up BETWEEN 0.10 AND 0.95
                    AND ss.minutes_into_slot BETWEEN :cal_min AND :cal_max
                    AND ss.slot_ts > :cutoff {end_clause_ss}
                    {sym_clause} {dur_clause_sr}
                GROUP BY bid_bucket HAVING total >= 3
                ORDER BY bid_bucket
            """), {**params, "cal_min": cal_min, "cal_max": cal_max})).mappings().all()

            # --- By hour (from resolutions, no snapshot join needed) ---
            hour_rows = (await session.execute(text(f"""
                SELECT
                    HOUR(FROM_UNIXTIME(slot_ts)) hour_utc,
                    COUNT(*) total,
                    SUM(resolved_up = 1) wins
                FROM slot_resolutions sr
                WHERE resolved_up IS NOT NULL AND slot_ts > :cutoff {end_clause_sr} {sym_clause} {dur_clause_sr}
                GROUP BY hour_utc ORDER BY hour_utc
            """), params)).mappings().all()

            # --- By day of week ---
            day_rows = (await session.execute(text(f"""
                SELECT
                    WEEKDAY(FROM_UNIXTIME(slot_ts)) day_of_week,
                    COUNT(*) total,
                    SUM(resolved_up = 1) wins
                FROM slot_resolutions sr
                WHERE resolved_up IS NOT NULL AND slot_ts > :cutoff {end_clause_sr} {sym_clause} {dur_clause_sr}
                GROUP BY day_of_week ORDER BY day_of_week
            """), params)).mappings().all()

            return {
                "total_slots": total,
                "resolved": int(row["resolved"] or 0),
                "unresolved": int(row["unresolved"] or 0),
                "snapshot_count": int(snap["cnt"] or 0),
                "first_ts": int(row["first_ts"]) if row["first_ts"] else None,
                "last_ts": int(row["last_ts"]) if row["last_ts"] else None,
                "duration": duration,
                "by_symbol": [
                    {"symbol": r["symbol"], "wins": int(r["wins"] or 0),
                     "total": int(r["total"]), "wr": _wr(r["wins"], r["total"])}
                    for r in sym_rows
                ],
                "heatmap": [
                    {"timing": r["timing_bin"], "move": r["move_bin"],
                     "wins": int(r["wins"] or 0), "total": int(r["total"]),
                     "wr": _wr(r["wins"], r["total"])}
                    for r in hm_rows
                ],
                "calibration": [
                    {"bid_bucket": round(float(r["bid_bucket"]), 2),
                     "avg_bid": round(float(r["avg_bid"]), 3),
                     "wins": int(r["wins"] or 0), "total": int(r["total"]),
                     "wr": _wr(r["wins"], r["total"])}
                    for r in cal_rows
                ],
                "by_hour": [
                    {"hour": int(r["hour_utc"]), "wins": int(r["wins"] or 0),
                     "total": int(r["total"]), "wr": _wr(r["wins"], r["total"])}
                    for r in hour_rows
                ],
                "by_day": [
                    {"day": int(r["day_of_week"]), "wins": int(r["wins"] or 0),
                     "total": int(r["total"]), "wr": _wr(r["wins"], r["total"])}
                    for r in day_rows
                ],
            }
    except Exception as exc:
        return {"error": f"Query failed: {str(exc)[:200]}"}


@router.get("/slots/stoploss")
async def slot_stoploss(
    hours: float = Query(default=168.0, ge=1.0, le=2160.0, description="Lookback window in hours."),
    symbol: Optional[str] = Query(default=None, description="Filter by symbol."),
    peak: float = Query(default=0.75, ge=0.50, le=0.95, description="Min bid_up peak to consider."),
    duration: Optional[str] = Query(default=None, description="Slot duration filter: 5m, 15m, or null for all."),
    start_ts: Optional[int] = Query(default=None, description="Absolute start unix timestamp (overrides hours)."),
    end_ts: Optional[int] = Query(default=None, description="Absolute end unix timestamp (overrides hours)."),
) -> dict:
    """Stop-loss threshold sweep: WR by dip depth after bid reaches peak."""
    factory = _get_factory()
    if factory is None:
        return {"error": "DATABASE_URL not configured"}

    cutoff = start_ts if start_ts is not None else int(time.time() - hours * 3600)
    end_clause_ss = "AND ss.slot_ts <= :end_ts" if end_ts is not None else ""
    end_clause_sr = "AND sr.slot_ts <= :end_ts" if end_ts is not None else ""
    sym_clause = "AND peaked.symbol = :symbol" if symbol else ""
    dur_clause_ss = _duration_clause("ss", duration)
    dur_clause_sr = _duration_clause("sr", duration)
    max_minutes = 5.0 if duration == "5m" else 13.0
    params: dict = {"cutoff": cutoff, "peak": peak, **_duration_params(duration)}
    if end_ts is not None:
        params["end_ts"] = end_ts
    if symbol:
        params["symbol"] = symbol.upper()

    try:
        async with factory() as session:
            rows = (await session.execute(text(f"""
                WITH thresholds AS (
                    SELECT 0.95 AS t UNION SELECT 0.90 UNION SELECT 0.85
                    UNION SELECT 0.80 UNION SELECT 0.75 UNION SELECT 0.70
                    UNION SELECT 0.65 UNION SELECT 0.60 UNION SELECT 0.55
                    UNION SELECT 0.50 UNION SELECT 0.45 UNION SELECT 0.40
                    UNION SELECT 0.35 UNION SELECT 0.30 UNION SELECT 0.25
                    UNION SELECT 0.20 UNION SELECT 0.15 UNION SELECT 0.10
                    UNION SELECT 0.05
                ),
                peak_minute AS (
                    SELECT ss.symbol, ss.slot_ts, ss.slot_duration,
                           MIN(ss.minutes_into_slot) AS first_peak_min
                    FROM slot_snapshots ss
                    WHERE ss.bid_up >= :peak
                        AND ss.minutes_into_slot <= :max_minutes
                        AND ss.slot_ts > :cutoff {end_clause_ss}
                        {dur_clause_ss}
                    GROUP BY ss.symbol, ss.slot_ts, ss.slot_duration
                ),
                peaked AS (
                    SELECT sr.symbol, sr.slot_ts, sr.resolved_up,
                           MIN(CASE WHEN ss.minutes_into_slot >= pm.first_peak_min
                                    THEN ss.bid_up END) AS min_bid_after_peak
                    FROM slot_resolutions sr
                    JOIN peak_minute pm
                        ON pm.symbol = sr.symbol AND pm.slot_ts = sr.slot_ts
                        AND pm.slot_duration = sr.slot_duration
                    JOIN slot_snapshots ss
                        ON ss.symbol = sr.symbol AND ss.slot_ts = sr.slot_ts
                        AND ss.slot_duration = sr.slot_duration
                        AND ss.minutes_into_slot <= :max_minutes
                    WHERE sr.resolved_up IS NOT NULL
                        AND sr.slot_ts > :cutoff {end_clause_sr}
                        AND ss.bid_up IS NOT NULL
                        {dur_clause_sr}
                    GROUP BY sr.symbol, sr.slot_ts, sr.resolved_up
                )
                SELECT
                    t.t AS threshold,
                    COUNT(*) AS total,
                    SUM(peaked.resolved_up = 1) AS wins,
                    SUM(peaked.resolved_up = 0) AS losses,
                    SUM(peaked.min_bid_after_peak <= t.t) AS triggered,
                    SUM(peaked.min_bid_after_peak <= t.t AND peaked.resolved_up = 0) AS true_saves,
                    SUM(peaked.min_bid_after_peak <= t.t AND peaked.resolved_up = 1) AS false_exits
                FROM peaked
                CROSS JOIN thresholds t
                WHERE 1=1 {sym_clause}
                GROUP BY t.t
                ORDER BY t.t DESC
            """), {**params, "max_minutes": max_minutes})).mappings().all()

            if not rows:
                return {"peak": peak, "total_peaked": 0, "thresholds": []}

            total_peaked = int(rows[0]["total"]) if rows else 0
            thresholds = []
            for r in rows:
                total = int(r["total"])
                wins = int(r["wins"] or 0)
                losses = int(r["losses"] or 0)
                triggered = int(r["triggered"] or 0)
                true_saves = int(r["true_saves"] or 0)
                false_exits = int(r["false_exits"] or 0)
                hold_wr = round(wins / total * 100, 1) if total else 0
                precision = round(true_saves / triggered * 100, 1) if triggered else 0
                thresholds.append({
                    "threshold": float(r["threshold"]),
                    "total": total,
                    "wins": wins,
                    "losses": losses,
                    "triggered": triggered,
                    "true_saves": true_saves,
                    "false_exits": false_exits,
                    "hold_wr": hold_wr,
                    "precision": precision,
                })

            return {"peak": peak, "total_peaked": total_peaked, "thresholds": thresholds}
    except Exception as exc:
        return {"error": f"Query failed: {str(exc)[:200]}"}
