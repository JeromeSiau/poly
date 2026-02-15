"""FastAPI endpoint for browsing all paper trades with optional tag filtering."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import FastAPI, Query
from sqlalchemy import select

from config.settings import settings
from src.api.slots_api import router as slots_router
from src.api.winrate import fetch_activity, analyse, resolve_open_markets
from src.db.database import get_sync_session, init_db
from src.db.models import LiveObservation as LO, PaperTrade as PT

app = FastAPI(title="Trades API", version="1.0.0")
app.include_router(slots_router)

DB_URL = "sqlite:///data/arb.db"

_LIVE_MODES = {"live", "live_fill", "live_settlement", "autopilot"}


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
    mode: Optional[str] = Query(default=None, description="'live' or 'paper'. Filters by game_state.mode."),
    is_open: Optional[bool] = Query(default=None, description="true=open positions only, false=closed only."),
    hours: float = Query(default=24.0, ge=0.1, le=720.0, description="Lookback window in hours."),
    limit: int = Query(default=200, ge=1, le=2000, description="Max rows returned."),
) -> dict:
    """Return trades joined with their observations, newest first."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    session = get_sync_session(DB_URL)
    try:
        # Use INNER JOIN when is_open is set (we need PT.is_open to filter)
        if is_open is not None:
            q = (
                select(LO, PT)
                .join(PT, PT.observation_id == LO.id)
                .where(LO.timestamp >= cutoff)
                .where(PT.is_open == is_open)
                .order_by(LO.timestamp.desc())
            )
        else:
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

            # Mode filter: check game_state.mode
            if mode is not None:
                obs_mode = str(gs.get("mode", "paper")).lower()
                if mode == "live" and obs_mode not in _LIVE_MODES:
                    continue
                if mode == "paper" and obs_mode in _LIVE_MODES:
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
                    "is_open": bool(trade.is_open) if trade.is_open is not None else None,
                    "closed_at": trade.closed_at.isoformat() if trade.closed_at else None,
                })

            # Extra fields from game_state (crypto_minute + td_maker analytics)
            for key in ("gap_pct", "gap_bucket", "time_bucket", "time_remaining_s",
                        "spot_at_entry", "spot_at_resolution", "sub_strategy",
                        "dir_move_pct", "minutes_into_slot",
                        "token_id", "category", "fee_cost"):
                if key in gs:
                    entry[key] = gs[key]

            # Include full game_state for programmatic consumers
            entry["game_state"] = gs

            trades.append(entry)
            if len(trades) >= limit:
                break

        # Summary stats
        pnls = [t["pnl"] for t in trades if t.get("pnl") is not None]
        wins = sum(1 for p in pnls if p > 0)
        losses = sum(1 for p in pnls if p <= 0)
        still_open = sum(1 for t in trades if t.get("is_open") is True)

        return {
            "count": len(trades),
            "total_pnl": round(sum(pnls), 4) if pnls else 0,
            "wins": wins,
            "losses": losses,
            "winrate": round(wins / len(pnls) * 100, 1) if pnls else 0,
            "still_open": still_open,
            "filters": {
                "tag": tag,
                "event_type": event_type,
                "mode": mode,
                "is_open": is_open,
                "hours": hours,
            },
            "trades": trades,
        }
    finally:
        session.close()


# ---------------------------------------------------------------------------
# /balance — paper or live USDC balance
# ---------------------------------------------------------------------------

def _fetch_live_balance() -> float:
    """Fetch on-chain USDC.e balance via Polygon RPC (no executor needed)."""
    import httpx

    usdc_e = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
    selector = "0x70a08231"
    rpc_url = settings.POLYGON_RPC_URL or "https://polygon-rpc.com"
    wallet = settings.POLYMARKET_WALLET_ADDRESS
    if not wallet:
        raise ValueError("POLYMARKET_WALLET_ADDRESS not configured")
    padded = wallet.lower().replace("0x", "").zfill(64)
    data = selector + padded
    resp = httpx.post(
        rpc_url,
        json={"jsonrpc": "2.0", "method": "eth_call",
              "params": [{"to": usdc_e, "data": data}, "latest"], "id": 1},
        timeout=10,
    )
    result = resp.json().get("result", "0x0")
    return int(result, 16) / 1e6


def _fetch_positions(wallet: str) -> list[dict]:
    """Fetch open positions from Polymarket Data API."""
    import httpx

    positions: list[dict] = []
    offset = 0
    with httpx.Client(timeout=15) as client:
        while True:
            resp = client.get(
                "https://data-api.polymarket.com/positions",
                params={"user": wallet, "sizeThreshold": 0.1,
                        "limit": 100, "offset": offset},
            )
            if resp.status_code != 200:
                break
            batch = resp.json()
            if not isinstance(batch, list) or not batch:
                break
            positions.extend(batch)
            if len(batch) < 100:
                break
            offset += len(batch)
    return positions


@app.get("/balance")
def balance(
    mode: str = Query(default="paper", description="'live' or 'paper'."),
) -> dict:
    """Return current balance for paper or live mode.

    Live mode returns portfolio value = USDC cash + market value of open positions
    (same as Polymarket profile page).
    """
    if mode == "live":
        try:
            wallet = settings.POLYMARKET_WALLET_ADDRESS
            if not wallet:
                raise ValueError("POLYMARKET_WALLET_ADDRESS not configured")

            cash = _fetch_live_balance()
            positions = _fetch_positions(wallet)

            positions_value = sum(float(p.get("currentValue", 0)) for p in positions)
            unrealized_pnl = sum(float(p.get("cashPnl", 0)) for p in positions)
            portfolio = cash + positions_value

            return {
                "mode": "live",
                "portfolio": round(portfolio, 2),
                "cash": round(cash, 2),
                "positions_value": round(positions_value, 2),
                "unrealized_pnl": round(unrealized_pnl, 2),
                "n_positions": len(positions),
                # Keep backward compat
                "balance": round(portfolio, 2),
            }
        except Exception as exc:
            return {"balance": 0.0, "mode": "live", "error": str(exc)}

    # Paper: starting capital + sum of closed paper-mode pnl
    session = get_sync_session(DB_URL)
    try:
        rows = session.execute(
            select(PT.pnl, LO.game_state)
            .join(LO, LO.id == PT.observation_id)
            .where(PT.is_open == False)  # noqa: E712
            .where(PT.pnl.isnot(None))
        ).all()
        paper_pnl = sum(
            pnl for pnl, gs in rows
            if str((gs or {}).get("mode", "paper")).lower() not in _LIVE_MODES
        )
        bal = settings.PAPER_STARTING_CAPITAL + paper_pnl
        return {"balance": round(bal, 4), "mode": "paper"}
    finally:
        session.close()


# ---------------------------------------------------------------------------
# /positions — live positions from Polymarket Data API
# ---------------------------------------------------------------------------

@app.get("/positions")
def positions(
    mode: str = Query(default="live", description="'live' or 'paper'."),
) -> dict:
    """Return open positions. Live: Polymarket Data API. Paper: internal DB."""
    if mode == "live":
        wallet = settings.POLYMARKET_WALLET_ADDRESS
        if not wallet:
            return {"positions": [], "error": "No wallet configured"}
        try:
            raw = _fetch_positions(wallet)
            # Filter out resolved positions (curPrice 0 or 1 = market settled)
            active = [p for p in raw if 0 < float(p.get("curPrice", 0)) < 1]
            items = [
                {
                    "title": p.get("title", ""),
                    "slug": p.get("slug", ""),
                    "outcome": p.get("outcome", ""),
                    "size": round(float(p.get("size", 0)), 2),
                    "avg_price": round(float(p.get("avgPrice", 0)), 4),
                    "cur_price": round(float(p.get("curPrice", 0)), 4),
                    "value": round(float(p.get("currentValue", 0)), 2),
                    "pnl": round(float(p.get("cashPnl", 0)), 2),
                    "pnl_pct": round(float(p.get("percentPnl", 0)), 1),
                }
                for p in active
            ]
            return {"mode": "live", "count": len(items), "positions": items}
        except Exception as exc:
            return {"positions": [], "error": str(exc)}

    # Paper: open positions from internal DB
    session = get_sync_session(DB_URL)
    try:
        rows = session.execute(
            select(LO, PT)
            .join(PT, PT.observation_id == LO.id)
            .where(PT.is_open == True)  # noqa: E712
        ).all()
        items = []
        for obs, trade in rows:
            gs = obs.game_state or {}
            obs_mode = str(gs.get("mode", "paper")).lower()
            if obs_mode in _LIVE_MODES:
                continue
            items.append({
                "title": gs.get("title", ""),
                "slug": gs.get("slug", ""),
                "outcome": gs.get("outcome", ""),
                "size": round(trade.size, 2) if trade.size else 0,
                "avg_price": round(trade.entry_price, 4) if trade.entry_price else 0,
                "cur_price": None,
                "value": None,
                "pnl": None,
                "pnl_pct": None,
            })
        return {"mode": "paper", "count": len(items), "positions": items}
    finally:
        session.close()


# ---------------------------------------------------------------------------
# /winrate — on-chain (live) or DB-based (paper)
# ---------------------------------------------------------------------------

def _winrate_paper(hours: float) -> dict:
    """Compute paper-mode win rate from the internal DB."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    session = get_sync_session(DB_URL)
    try:
        rows = session.execute(
            select(LO, PT)
            .join(PT, PT.observation_id == LO.id)
            .where(LO.timestamp >= cutoff)
        ).all()

        # Filter to paper-mode observations only
        paper_rows = []
        for obs, trade in rows:
            gs = obs.game_state or {}
            obs_mode = str(gs.get("mode", "paper")).lower()
            if obs_mode not in _LIVE_MODES:
                paper_rows.append((obs, trade))

        # Separate resolved vs open
        resolved = [(o, t) for o, t in paper_rows if t.pnl is not None]
        still_open = [(o, t) for o, t in paper_rows if t.pnl is None]

        wins = [(o, t) for o, t in resolved if t.pnl > 0]
        losses = [(o, t) for o, t in resolved if t.pnl <= 0]

        total_pnl = sum(t.pnl for _, t in resolved)
        total_cost = sum(t.size for _, t in paper_rows)
        win_pnl = sum(t.pnl for _, t in wins)
        loss_pnl = sum(t.pnl for _, t in losses)

        return {
            "mode": "paper",
            "hours": hours,
            "total_markets": len(paper_rows),
            "resolved": len(resolved),
            "still_open": len(still_open),
            "wins": len(wins),
            "losses": len(losses),
            "winrate": round(len(wins) / len(resolved) * 100, 1) if resolved else 0,
            "total_pnl": round(total_pnl, 2),
            "total_invested": round(total_cost, 2),
            "roi_pct": round(total_pnl / total_cost * 100, 1) if total_cost > 0 else 0,
            "avg_win": round(win_pnl / len(wins), 2) if wins else 0,
            "avg_loss": round(loss_pnl / len(losses), 2) if losses else 0,
            "profit_factor": round(abs(win_pnl / loss_pnl), 2) if loss_pnl < 0 else None,
        }
    finally:
        session.close()


@app.get("/winrate")
def winrate(
    hours: float = Query(default=24.0, ge=0.1, le=720.0, description="Lookback window in hours."),
    wallet: Optional[str] = Query(default=None, description="Wallet address (default: from settings)."),
    mode: Optional[str] = Query(default="live", description="'live' (on-chain) or 'paper' (DB)."),
) -> dict:
    """Win rate from on-chain Polymarket wallet activity or paper DB."""
    if mode == "paper":
        return _winrate_paper(hours)

    addr = wallet or settings.POLYMARKET_WALLET_ADDRESS
    if not addr:
        return {"error": "No wallet configured"}

    rows = fetch_activity(addr, hours)
    markets = analyse(rows)
    resolve_open_markets(markets)

    resolved = [m for m in markets if m["status"] in ("WIN", "LOSS")]
    still_open = [m for m in markets if m["status"] == "OPEN"]
    wins = [m for m in resolved if m["status"] == "WIN"]
    losses = [m for m in resolved if m["status"] == "LOSS"]

    total_pnl = sum(m["pnl"] for m in resolved)
    total_cost = sum(m["cost"] for m in markets)
    win_pnl = sum(m["pnl"] for m in wins)
    loss_pnl = sum(m["pnl"] for m in losses)

    return {
        "hours": hours,
        "wallet": addr,
        "total_markets": len(markets),
        "resolved": len(resolved),
        "still_open": len(still_open),
        "wins": len(wins),
        "losses": len(losses),
        "winrate": round(len(wins) / len(resolved) * 100, 1) if resolved else 0,
        "total_pnl": round(total_pnl, 2),
        "total_invested": round(total_cost, 2),
        "roi_pct": round(total_pnl / total_cost * 100, 1) if total_cost > 0 else 0,
        "avg_win": round(win_pnl / len(wins), 2) if wins else 0,
        "avg_loss": round(loss_pnl / len(losses), 2) if losses else 0,
        "profit_factor": round(abs(win_pnl / loss_pnl), 2) if loss_pnl < 0 else None,
        "markets": [
            {
                "title": m["title"],
                "status": m["status"],
                "pnl": round(m["pnl"], 2),
                "cost": round(m["cost"], 2),
                "avg_entry": round(m["avg_entry"], 2),
                "outcome": m["outcome"],
                "n_fills": m["n_fills"],
                "timestamp": m["first_ts"],
            }
            for m in markets
            if m["status"] in ("WIN", "LOSS")
        ],
    }


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
