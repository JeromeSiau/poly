"""FastAPI endpoint for browsing all paper trades with optional tag filtering."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import FastAPI, Query
from sqlalchemy import select

from src.db.database import get_sync_session, init_db
from src.db.models import LiveObservation as LO, PaperTrade as PT

app = FastAPI(title="Trades API", version="1.0.0")

DB_URL = "sqlite:///data/arb.db"


@app.on_event("startup")
def _startup() -> None:
    init_db(DB_URL)


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@app.get("/trades")
def list_trades(
    tag: Optional[str] = Query(default=None, description="Filter by strategy_tag (substring match)."),
    event_type: Optional[str] = Query(default=None, description="Filter by event_type (exact match)."),
    hours: float = Query(default=24.0, ge=0.1, le=720.0, description="Lookback window in hours."),
    limit: int = Query(default=200, ge=1, le=2000, description="Max rows returned."),
) -> dict:
    """Return trades joined with their observations, newest first."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    session = get_sync_session(DB_URL)
    try:
        q = (
            select(LO, PT)
            .outerjoin(PT, PT.observation_id == LO.id)
            .where(LO.timestamp >= cutoff)
            .order_by(LO.timestamp.desc())
        )

        if event_type:
            q = q.where(LO.event_type == event_type)

        rows = session.execute(q).all()

        trades = []
        for obs, trade in rows:
            gs = obs.game_state or {}

            # Substring match on strategy_tag inside game_state
            if tag and tag not in gs.get("strategy_tag", ""):
                continue

            entry = {
                "id": obs.id,
                "timestamp": obs.timestamp.isoformat() if obs.timestamp else None,
                "event_type": obs.event_type,
                "match_id": obs.match_id,
                "strategy_tag": gs.get("strategy_tag"),
                "title": gs.get("title"),
                "outcome": gs.get("outcome"),
                "side": gs.get("side"),
                "symbol": gs.get("symbol"),
                "slug": gs.get("slug"),
                "model_prediction": obs.model_prediction,
                "polymarket_price": obs.polymarket_price,
            }

            if trade:
                entry.update({
                    "trade_id": trade.id,
                    "entry_price": trade.entry_price,
                    "exit_price": trade.exit_price,
                    "size": trade.size,
                    "pnl": trade.pnl,
                    "edge_theoretical": trade.edge_theoretical,
                    "edge_realized": trade.edge_realized,
                })

            # Extra fields from game_state for crypto_minute
            for key in ("gap_pct", "gap_bucket", "time_bucket", "time_remaining_s",
                        "spot_at_entry", "spot_at_resolution", "sub_strategy"):
                if key in gs:
                    entry[key] = gs[key]

            trades.append(entry)
            if len(trades) >= limit:
                break

        # Summary stats
        pnls = [t["pnl"] for t in trades if t.get("pnl") is not None]
        wins = sum(1 for p in pnls if p > 0)
        losses = sum(1 for p in pnls if p <= 0)

        return {
            "count": len(trades),
            "total_pnl": round(sum(pnls), 4) if pnls else 0,
            "wins": wins,
            "losses": losses,
            "winrate": round(wins / len(pnls) * 100, 1) if pnls else 0,
            "filters": {"tag": tag, "event_type": event_type, "hours": hours},
            "trades": trades,
        }
    finally:
        session.close()


@app.get("/tags")
def list_tags(
    hours: float = Query(default=24.0, ge=0.1, le=720.0, description="Lookback window in hours."),
) -> dict:
    """Return all distinct strategy_tags and event_types seen recently."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    session = get_sync_session(DB_URL)
    try:
        rows = session.execute(
            select(LO.event_type, LO.game_state)
            .where(LO.timestamp >= cutoff)
        ).all()

        tags: dict[str, int] = {}
        event_types: dict[str, int] = {}
        for event_type, gs in rows:
            event_types[event_type] = event_types.get(event_type, 0) + 1
            st = (gs or {}).get("strategy_tag", "unknown")
            tags[st] = tags.get(st, 0) + 1

        return {
            "hours": hours,
            "strategy_tags": tags,
            "event_types": event_types,
        }
    finally:
        session.close()
