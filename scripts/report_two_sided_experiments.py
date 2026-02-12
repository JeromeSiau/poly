#!/usr/bin/env python3
"""Report two-sided paper trading results by experiment tag and market pair.

Usage:
    uv run python scripts/report_two_sided_experiments.py --db data/arb.db
    uv run python scripts/report_two_sided_experiments.py --strategy-tag edge_1p2_0p2 --top 20
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.models import LiveObservation, PaperTrade

TWO_SIDED_EVENT_TYPE = "two_sided_inventory"


@dataclass(slots=True)
class TradeRow:
    strategy_tag: str
    run_id: str
    condition_id: str
    title: str
    outcome: str
    side: str
    shares: float
    size_usd: float
    edge_theoretical: float
    edge_realized: float
    pnl: float
    created_at: datetime


@dataclass(slots=True)
class PairSummary:
    strategy_tag: str
    condition_id: str
    title: str
    trades: int
    sells: int
    win_rate: float
    realized_pnl: float
    gross_notional: float
    net_shares: float
    avg_edge_theoretical: float
    avg_edge_realized_sells: float


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


def _parse_since(raw: Optional[str]) -> Optional[datetime]:
    if not raw:
        return None
    return datetime.strptime(raw, "%Y-%m-%d")


def _extract_trade_row(trade: PaperTrade, observation: LiveObservation) -> Optional[TradeRow]:
    if observation.event_type != TWO_SIDED_EVENT_TYPE:
        return None
    game_state = observation.game_state if isinstance(observation.game_state, dict) else {}

    strategy_tag = str(game_state.get("strategy_tag") or "default")
    run_id = str(game_state.get("run_id") or "unknown")
    condition_id = str(game_state.get("condition_id") or observation.match_id or "")
    if not condition_id:
        return None
    title = str(game_state.get("title") or condition_id)
    outcome = str(game_state.get("outcome") or "")
    side = str(trade.side or game_state.get("side") or "").upper()
    if side not in {"BUY", "SELL"}:
        return None

    fill_price = _safe_float(
        trade.simulated_fill_price if trade.simulated_fill_price is not None else trade.entry_price,
        default=0.0,
    )
    size_usd = _safe_float(trade.size, default=0.0)
    if size_usd <= 0:
        return None

    shares = _safe_float(game_state.get("shares"), default=0.0)
    if shares <= 0 and fill_price > 0:
        shares = size_usd / fill_price

    created_at = trade.created_at or observation.timestamp
    if created_at is None:
        created_at = datetime.utcnow()

    return TradeRow(
        strategy_tag=strategy_tag,
        run_id=run_id,
        condition_id=condition_id,
        title=title,
        outcome=outcome,
        side=side,
        shares=shares,
        size_usd=size_usd,
        edge_theoretical=_safe_float(trade.edge_theoretical),
        edge_realized=_safe_float(trade.edge_realized),
        pnl=_safe_float(trade.pnl),
        created_at=created_at,
    )


def load_rows(db_url: str, since: Optional[datetime], strategy_tag: Optional[str]) -> list[TradeRow]:
    engine = create_engine(_normalize_sync_db_url(db_url))
    session = sessionmaker(bind=engine)()
    try:
        query = (
            session.query(PaperTrade, LiveObservation)
            .join(LiveObservation, LiveObservation.id == PaperTrade.observation_id)
            .filter(LiveObservation.event_type == TWO_SIDED_EVENT_TYPE)
        )
        if since is not None:
            query = query.filter(PaperTrade.created_at >= since)
        rows = query.order_by(PaperTrade.created_at.asc(), PaperTrade.id.asc()).all()
    finally:
        session.close()

    parsed: list[TradeRow] = []
    for trade, observation in rows:
        row = _extract_trade_row(trade, observation)
        if row is None:
            continue
        if strategy_tag and row.strategy_tag != strategy_tag:
            continue
        parsed.append(row)
    return parsed


def summarize_pairs(rows: list[TradeRow]) -> list[PairSummary]:
    acc: dict[tuple[str, str, str], dict[str, float]] = {}
    for row in rows:
        key = (row.strategy_tag, row.condition_id, row.title)
        cur = acc.get(key)
        if cur is None:
            cur = {
                "trades": 0.0,
                "sells": 0.0,
                "wins": 0.0,
                "realized_pnl": 0.0,
                "gross_notional": 0.0,
                "net_shares": 0.0,
                "sum_edge_theoretical": 0.0,
                "sum_edge_realized_sells": 0.0,
            }
            acc[key] = cur

        cur["trades"] += 1
        cur["realized_pnl"] += row.pnl
        cur["gross_notional"] += row.size_usd
        cur["sum_edge_theoretical"] += row.edge_theoretical
        if row.side == "BUY":
            cur["net_shares"] += row.shares
        else:
            cur["net_shares"] -= row.shares
            cur["sells"] += 1
            cur["sum_edge_realized_sells"] += row.edge_realized
            if row.pnl > 0:
                cur["wins"] += 1

    summaries: list[PairSummary] = []
    for (tag, condition_id, title), cur in acc.items():
        sells = int(cur["sells"])
        trades = int(cur["trades"])
        wins = int(cur["wins"])
        summaries.append(
            PairSummary(
                strategy_tag=tag,
                condition_id=condition_id,
                title=title,
                trades=trades,
                sells=sells,
                win_rate=(wins / sells) if sells > 0 else 0.0,
                realized_pnl=cur["realized_pnl"],
                gross_notional=cur["gross_notional"],
                net_shares=cur["net_shares"],
                avg_edge_theoretical=(cur["sum_edge_theoretical"] / trades) if trades > 0 else 0.0,
                avg_edge_realized_sells=(cur["sum_edge_realized_sells"] / sells) if sells > 0 else 0.0,
            )
        )
    return summaries


def print_report(rows: list[TradeRow], summaries: list[PairSummary], top: int) -> None:
    if not rows:
        print("No two-sided paper trades found for this filter.")
        return

    tags = sorted({row.strategy_tag for row in rows})
    print("=" * 90)
    print(f"Two-sided experiments | trades={len(rows)} | strategy_tags={len(tags)}")
    print("=" * 90)

    for tag in tags:
        tag_rows = [row for row in rows if row.strategy_tag == tag]
        tag_pairs = [item for item in summaries if item.strategy_tag == tag]
        sells = sum(1 for row in tag_rows if row.side == "SELL")
        pnl = sum(row.pnl for row in tag_rows)
        gross = sum(row.size_usd for row in tag_rows)
        open_pairs = sum(1 for item in tag_pairs if abs(item.net_shares) > 1e-9)
        win_sells = sum(1 for row in tag_rows if row.side == "SELL" and row.pnl > 0)
        win_rate = (win_sells / sells) if sells > 0 else 0.0

        print(
            f"\n[tag={tag}] trades={len(tag_rows)} sells={sells} win_rate={win_rate:.1%} "
            f"realized_pnl=${pnl:,.2f} gross_notional=${gross:,.2f} open_pairs={open_pairs}"
        )

        best = sorted(tag_pairs, key=lambda x: x.realized_pnl, reverse=True)[:top]
        worst = sorted(tag_pairs, key=lambda x: x.realized_pnl)[:top]

        print("  Best pairs:")
        print("    pnl       trades sells net_shares cond_id        title")
        for item in best:
            print(
                f"    {item.realized_pnl:>8.2f}  {item.trades:>6} {item.sells:>5} "
                f"{item.net_shares:>10.2f} {item.condition_id[:12]:<12} {item.title[:50]}"
            )

        print("  Worst pairs:")
        print("    pnl       trades sells net_shares cond_id        title")
        for item in worst:
            print(
                f"    {item.realized_pnl:>8.2f}  {item.trades:>6} {item.sells:>5} "
                f"{item.net_shares:>10.2f} {item.condition_id[:12]:<12} {item.title[:50]}"
            )


def write_csv(path: Path, summaries: list[PairSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "strategy_tag",
                "condition_id",
                "title",
                "trades",
                "sells",
                "win_rate",
                "realized_pnl",
                "gross_notional",
                "net_shares",
                "avg_edge_theoretical",
                "avg_edge_realized_sells",
            ]
        )
        for item in sorted(summaries, key=lambda x: (x.strategy_tag, -x.realized_pnl)):
            writer.writerow(
                [
                    item.strategy_tag,
                    item.condition_id,
                    item.title,
                    item.trades,
                    item.sells,
                    item.win_rate,
                    item.realized_pnl,
                    item.gross_notional,
                    item.net_shares,
                    item.avg_edge_theoretical,
                    item.avg_edge_realized_sells,
                ]
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report two-sided paper P&L by experiment and pair.")
    parser.add_argument("--db", type=str, default="data/arb.db", help="DB path or SQLAlchemy URL.")
    parser.add_argument("--since", type=str, default=None, help="Filter start date (YYYY-MM-DD).")
    parser.add_argument(
        "--strategy-tag",
        type=str,
        default=None,
        help="Optional strategy tag filter (e.g. edge_1p2_0p2).",
    )
    parser.add_argument("--top", type=int, default=10, help="Top/bottom pairs to print per tag.")
    parser.add_argument("--csv-out", type=str, default=None, help="Optional CSV output path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    since = _parse_since(args.since)
    rows = load_rows(db_url=args.db, since=since, strategy_tag=args.strategy_tag)
    summaries = summarize_pairs(rows)
    print_report(rows, summaries, top=max(1, args.top))
    if args.csv_out:
        out = Path(args.csv_out)
        write_csv(out, summaries)
        print(f"\nCSV saved: {out}")


if __name__ == "__main__":
    main()
