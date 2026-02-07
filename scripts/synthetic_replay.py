#!/usr/bin/env python3
# scripts/synthetic_replay.py
"""Synthetic replay for Reality Arb without PandaScore/Polymarket.

Usage:
    uv run python scripts/synthetic_replay.py \
        --input data/synthetic/lol_match.json

Notes:
- `true_prob` is interpreted as the true win probability of teams[0].
- The market price lags behind `true_prob` by `market.lag` each event.
- `market.fill_mode` supports "fok" (reject if not enough liquidity) or "partial".
"""

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

# Add project root to path for imports
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

from src.arb.reality_arb import RealityArbEngine
from src.feeds.base import FeedEvent
from src.realtime.event_detector import EventDetector
from src.realtime.market_mapper import MarketMapper
from src.arb.position_manager import PositionAction


def clamp_price(value: float) -> float:
    return max(0.01, min(0.99, value))


@dataclass
class MarketModel:
    """Simple market model with lagged reaction to true probability."""

    mid: float
    spread: float
    lag: float
    bid_size: float
    ask_size: float
    true_price: float

    def apply_true_price(self, true_price: float) -> None:
        self.true_price = clamp_price(true_price)
        # Market reacts with lag: move a fraction toward the true price
        self.mid = clamp_price(self.mid + (self.true_price - self.mid) * self.lag)

    def best_prices(self) -> tuple[float, float]:
        half = self.spread / 2.0
        bid = clamp_price(self.mid - half)
        ask = clamp_price(self.mid + half)
        return bid, ask


class SyntheticPolymarketFeed:
    """Minimal feed returning synthetic bid/ask prices."""

    def __init__(self, model: MarketModel):
        self.model = model

    def get_best_prices(self, market_id: str, outcome: str) -> tuple[float, float]:
        return self.model.best_prices()

    def get_best_levels(
        self, market_id: str, outcome: str
    ) -> tuple[float, float, float, float]:
        bid, ask = self.model.best_prices()
        return (bid, self.model.bid_size, ask, self.model.ask_size)


class ExecutionSimulator:
    """Simple execution simulator with liquidity and partial fills."""

    def __init__(self, model: MarketModel, fill_mode: str, min_fill_usd: float) -> None:
        self.model = model
        self.fill_mode = fill_mode
        self.min_fill_usd = min_fill_usd

    def execute(self, side: str, size: float, price: float) -> dict[str, Any]:
        side = side.upper()
        if price <= 0 or size <= 0:
            return {"status": "REJECTED", "filled_size": 0.0}

        available_shares = (
            self.model.ask_size if side == "BUY" else self.model.bid_size
        )
        available_usd = available_shares * price

        if size <= available_usd:
            filled_size = size
            status = "FILLED"
        elif self.fill_mode == "partial" and available_usd >= self.min_fill_usd:
            filled_size = available_usd
            status = "PARTIAL"
        else:
            return {"status": "REJECTED", "filled_size": 0.0}

        filled_shares = filled_size / price
        if side == "BUY":
            self.model.ask_size = max(0.0, self.model.ask_size - filled_shares)
        else:
            self.model.bid_size = max(0.0, self.model.bid_size - filled_shares)

        return {"status": status, "filled_size": filled_size}


def _merge_event_data(raw_event: dict[str, Any]) -> dict[str, Any]:
    data = dict(raw_event.get("data", {}))
    for key in (
        "team",
        "killer_team",
        "game_time_minutes",
        "gold_diff",
        "kill_diff",
        "tower_diff",
        "dragon_diff",
        "baron_diff",
    ):
        if key in raw_event and key not in data:
            data[key] = raw_event[key]
    return data


async def run_replay(input_path: Path, verbose: bool, save_json: Optional[Path]) -> None:
    raw = json.loads(input_path.read_text())

    game = raw.get("game", "lol")
    teams = raw.get("teams", [])
    winner = raw.get("winner")
    match_id = raw.get("match_id", "synthetic_match")
    events = raw.get("events", [])

    if len(teams) != 2:
        raise ValueError("Input must include exactly two teams.")
    if winner not in teams:
        raise ValueError("Winner must be one of the teams in the match.")

    market_cfg = raw.get("market", {})
    risk_cfg = raw.get("risk", {})

    market_id = market_cfg.get("market_id", "synthetic_market")
    initial_price = float(market_cfg.get("initial_price", 0.5))
    spread = float(market_cfg.get("spread", 0.02))
    lag = float(market_cfg.get("lag", 0.3))
    bid_size = float(market_cfg.get("bid_size", 500))
    ask_size = float(market_cfg.get("ask_size", 500))
    fill_mode = str(market_cfg.get("fill_mode", "fok")).lower()
    min_fill_usd = float(market_cfg.get("min_fill_usd", 25.0))
    if fill_mode not in ("fok", "partial"):
        raise ValueError("market.fill_mode must be 'fok' or 'partial'.")

    model = MarketModel(
        mid=clamp_price(initial_price),
        spread=spread,
        lag=lag,
        bid_size=bid_size,
        ask_size=ask_size,
        true_price=clamp_price(initial_price),
    )
    feed = SyntheticPolymarketFeed(model)
    executor = ExecutionSimulator(model, fill_mode=fill_mode, min_fill_usd=min_fill_usd)

    mapper = MarketMapper()
    mapping = mapper.add_mapping(
        game=game,
        event_identifier=f"{game}_{teams[0]}_vs_{teams[1]}_synthetic",
        polymarket_id=market_id,
        outcomes={teams[0]: "YES", teams[1]: "NO"},
    )

    detector = EventDetector()
    engine = RealityArbEngine(
        polymarket_feed=feed,
        event_detector=detector,
        market_mapper=mapper,
        autopilot=False,
    )

    capital = float(risk_cfg.get("capital", engine.capital))
    engine.capital = capital
    engine.min_edge_pct = float(risk_cfg.get("min_edge_pct", engine.min_edge_pct))
    engine.max_position_pct = float(risk_cfg.get("max_position_pct", engine.max_position_pct))
    engine.position_manager.max_position_per_market = engine.capital * engine.max_position_pct

    summary = {
        "events_total": 0,
        "events_significant": 0,
        "opportunities": 0,
        "edges": [],
        "actions": {
            "open": 0,
            "add": 0,
            "hold": 0,
            "close": 0,
            "no_action": 0,
            "rate_limited": 0,
        },
        "fills": {
            "filled": 0,
            "partial": 0,
            "rejected": 0,
        },
        "partial_closes": 0,
    }

    def _update_existing_position_prices() -> None:
        existing = engine.position_manager.get_position(market_id)
        if not existing:
            return

        outcome = existing.outcome or mapping.get_outcome_for_team(existing.team)
        if not outcome:
            return

        best_bid, best_ask = feed.get_best_prices(market_id, outcome)
        if best_bid is not None and best_ask is not None:
            existing.update_price((best_bid + best_ask) / 2.0)

    for idx, raw_event in enumerate(events, start=1):
        summary["events_total"] += 1

        data = _merge_event_data(raw_event)
        event = FeedEvent(
            source="synthetic",
            event_type=raw_event.get("event_type", "unknown"),
            game=game,
            data=data,
            timestamp=float(raw_event.get("timestamp", time.time() - 10)),
            match_id=raw_event.get("match_id", match_id),
        )

        significant = detector.classify(event)
        if significant.is_significant:
            summary["events_significant"] += 1

        _update_existing_position_prices()

        opportunity = engine.evaluate_opportunity(significant, mapping)

        edge = None
        action = None
        fill_status = None

        if opportunity is not None:
            summary["opportunities"] += 1
            edge = opportunity.edge_pct
            summary["edges"].append(edge)

            decision = engine.position_manager.evaluate(
                market_id=opportunity.market_id,
                favored_team=significant.favored_team,
                fair_price=opportunity.estimated_fair_price,
                current_market_price=(
                    (opportunity.best_bid + opportunity.best_ask) / 2
                    if opportunity.best_bid is not None and opportunity.best_ask is not None
                    else opportunity.current_price
                ),
                edge_pct=opportunity.edge_pct,
                min_edge=engine.min_edge_pct,
                suggested_size=engine.calculate_position_size(opportunity),
            )

            action = decision.action.value
            if action in summary["actions"]:
                summary["actions"][action] += 1

            if decision.action in (PositionAction.OPEN, PositionAction.ADD):
                trade_size = decision.size
                if trade_size > 0:
                    fill = executor.execute("BUY", trade_size, opportunity.current_price)
                    fill_status = fill["status"]
                    filled_size = fill.get("filled_size", 0.0)

                    if fill_status == "FILLED":
                        summary["fills"]["filled"] += 1
                    elif fill_status == "PARTIAL":
                        summary["fills"]["partial"] += 1
                    else:
                        summary["fills"]["rejected"] += 1

                    if filled_size > 0:
                        if decision.action == PositionAction.OPEN:
                            engine.position_manager.open_position(
                                market_id=opportunity.market_id,
                                team=significant.favored_team,
                                outcome=opportunity.outcome,
                                entry_price=opportunity.current_price,
                                size=filled_size,
                                trigger_event=opportunity.trigger_event,
                                outcome_token_id=opportunity.token_id,
                            )
                        else:
                            engine.position_manager.add_to_position(
                                market_id=opportunity.market_id,
                                additional_price=opportunity.current_price,
                                additional_size=filled_size,
                            )

            elif decision.action == PositionAction.CLOSE:
                existing = engine.position_manager.get_position(opportunity.market_id)
                if existing:
                    outcome = existing.outcome or mapping.get_outcome_for_team(existing.team)
                    best_bid, _ = feed.get_best_prices(opportunity.market_id, outcome)
                    exit_price = best_bid if best_bid is not None else opportunity.current_price

                    fill = executor.execute("SELL", existing.size, exit_price)
                    fill_status = fill["status"]
                    filled_size = fill.get("filled_size", 0.0)

                    if fill_status == "FILLED":
                        summary["fills"]["filled"] += 1
                        engine.position_manager.close_position(
                            market_id=existing.market_id,
                            exit_price=exit_price,
                            reason=decision.reason,
                        )
                    elif fill_status == "PARTIAL":
                        summary["fills"]["partial"] += 1
                        if filled_size > 0:
                            engine.position_manager.close_position_partial(
                                market_id=existing.market_id,
                                exit_price=exit_price,
                                size_to_close=filled_size,
                                reason=decision.reason,
                            )
                            summary["partial_closes"] += 1
                    else:
                        summary["fills"]["rejected"] += 1

        if verbose:
            bid, ask = model.best_prices()
            action_label = action or "none"
            edge_label = f"{edge:.2%}" if edge is not None else "n/a"
            fill_label = fill_status or "n/a"
            print(
                f"{idx:02d} {event.event_type:14} team={data.get('team', 'n/a'):<6} "
                f"impact={significant.impact_score:.2f} edge={edge_label:>8} "
                f"action={action_label:10} fill={fill_label:8} "
                f"mkt=({bid:.2f}/{ask:.2f})"
            )

        true_prob = raw_event.get("true_prob")
        if true_prob is not None:
            model.apply_true_price(float(true_prob))

    await engine.close_match_position(market_id=market_id, winner=winner)

    realized = engine.position_manager.get_total_realized_pnl()
    unrealized = engine.position_manager.get_total_unrealized_pnl()
    avg_edge = (
        sum(summary["edges"]) / len(summary["edges"]) if summary["edges"] else 0.0
    )

    print("\n" + "=" * 60)
    print("SYNTHETIC REPLAY SUMMARY")
    print("=" * 60)
    print(f"Match: {teams[0]} vs {teams[1]} | Winner: {winner}")
    print(f"Events: {summary['events_total']} | Significant: {summary['events_significant']}")
    print(f"Opportunities: {summary['opportunities']}")
    print(
        "Actions: "
        f"open={summary['actions']['open']} "
        f"add={summary['actions']['add']} "
        f"hold={summary['actions']['hold']} "
        f"close={summary['actions']['close']} "
        f"no_action={summary['actions']['no_action']}"
    )
    print(
        "Fills: "
        f"full={summary['fills']['filled']} "
        f"partial={summary['fills']['partial']} "
        f"rejected={summary['fills']['rejected']}"
    )
    if summary["partial_closes"]:
        print(f"Partial closes: {summary['partial_closes']}")
    print(f"Avg edge: {avg_edge:.2%}")
    print(f"Realized PnL: ${realized:,.2f}")
    print(f"Unrealized PnL: ${unrealized:,.2f}")
    print("=" * 60 + "\n")

    if save_json:
        report = {
            "summary": summary,
            "avg_edge": avg_edge,
            "realized_pnl": realized,
            "unrealized_pnl": unrealized,
            "closed_positions": engine.position_manager.get_closed_positions(),
        }
        save_json.write_text(json.dumps(report, indent=2))
        print(f"Saved report to: {save_json}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthetic replay for Reality Arb strategy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/synthetic/lol_match.json",
        help="Path to synthetic match JSON",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-event decisions",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default="",
        help="Optional path to save JSON report",
    )

    args = parser.parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    save_json = Path(args.save_json) if args.save_json else None

    asyncio.run(run_replay(input_path, args.verbose, save_json))


if __name__ == "__main__":
    main()
