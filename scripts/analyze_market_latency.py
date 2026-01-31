#!/usr/bin/env python3
"""Analyze Polymarket price reaction latency for esports markets.

This script helps validate the reality arbitrage strategy by analyzing
how quickly Polymarket prices react to game events. It looks for:
1. Sharp price movements (>3%) that indicate significant game events
2. "Continuation" patterns where multiple moves happen in sequence
3. Average reaction time of the market

If the market consistently takes several minutes to fully price events,
there may be exploitable edge with faster data sources like PandaScore.

Usage:
    uv run python scripts/analyze_market_latency.py
    uv run python scripts/analyze_market_latency.py --limit 30 --min-move 5
    uv run python scripts/analyze_market_latency.py --match-type BO3
"""

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx
import structlog

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = structlog.get_logger()

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
LOL_SERIES_ID = "10311"


@dataclass
class PriceMove:
    """Represents a significant price movement."""
    timestamp: int
    elapsed_min: int
    prev_price: float
    new_price: float
    change_pct: float
    direction: int  # 1 = up, -1 = down


@dataclass
class ContinuationPattern:
    """Represents a continuation pattern (delayed market reaction)."""
    elapsed_min: int
    num_moves: int
    total_change_pct: float
    duration_seconds: int


@dataclass
class MatchAnalysis:
    """Analysis results for a single match."""
    title: str
    winner: str
    data_points: int
    significant_moves: list[PriceMove]
    continuations: list[ContinuationPattern]


def fetch_resolved_matches(limit: int = 50, match_type: str | None = None) -> list[dict]:
    """Fetch recently resolved LoL matches from Polymarket."""
    with httpx.Client(timeout=30.0) as client:
        response = client.get(f"{GAMMA_API}/events", params={
            "series_id": LOL_SERIES_ID,
            "closed": "true",
            "limit": limit,
            "order": "endDate",
            "ascending": "false",
        })
        response.raise_for_status()
        events = response.json()

    # Filter by match type if specified
    if match_type:
        events = [e for e in events if f"({match_type})" in e.get("title", "")]

    return events


def get_price_history(token_id: str, start_ts: int, end_ts: int) -> list[dict]:
    """Get price history with 1-minute fidelity."""
    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.get(f"{CLOB_API}/prices-history", params={
                "market": token_id,
                "startTs": start_ts - 3600,  # Start 1h before for context
                "endTs": end_ts,
                "fidelity": 1,  # 1 minute candles
            })
            if response.status_code == 200:
                return response.json().get("history", [])
    except Exception as e:
        logger.warning("price_history_error", error=str(e))
    return []


def find_significant_moves(
    history: list[dict],
    start_ts: int,
    min_move_pct: float = 3.0
) -> list[PriceMove]:
    """Find significant price movements in history."""
    moves = []

    for i in range(1, len(history)):
        prev = history[i - 1]
        curr = history[i]

        prev_price = prev.get("p", 0.5)
        curr_price = curr.get("p", 0.5)
        change = curr_price - prev_price
        change_pct = abs(change) * 100

        if change_pct >= min_move_pct:
            ts = curr.get("t", 0)
            elapsed = (ts - start_ts) // 60

            moves.append(PriceMove(
                timestamp=ts,
                elapsed_min=elapsed,
                prev_price=prev_price,
                new_price=curr_price,
                change_pct=change_pct,
                direction=1 if change > 0 else -1,
            ))

    return moves


def find_continuation_patterns(
    moves: list[PriceMove],
    max_gap_seconds: int = 300
) -> list[ContinuationPattern]:
    """Find continuation patterns (multiple moves in same direction within time window)."""
    if not moves:
        return []

    patterns = []
    i = 0

    while i < len(moves):
        move = moves[i]
        direction = move.direction
        start_ts = move.timestamp
        count = 1
        total_change = move.change_pct
        last_ts = start_ts

        j = i + 1
        while j < len(moves):
            next_move = moves[j]
            time_gap = next_move.timestamp - start_ts

            if time_gap <= max_gap_seconds and next_move.direction == direction:
                count += 1
                total_change += next_move.change_pct
                last_ts = next_move.timestamp
                j += 1
            else:
                break

        if count >= 2:
            patterns.append(ContinuationPattern(
                elapsed_min=move.elapsed_min,
                num_moves=count,
                total_change_pct=total_change,
                duration_seconds=last_ts - start_ts,
            ))

        i = j if j > i + 1 else i + 1

    return patterns


def analyze_match(event: dict, min_move_pct: float = 3.0) -> MatchAnalysis | None:
    """Analyze a single match for price reaction patterns."""
    title = event.get("title", "")
    markets = event.get("markets", [])

    if not markets:
        return None

    market = markets[0]
    start_date = event.get("startDate", "")
    end_date = event.get("endDate", "")

    try:
        start_dt = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
        start_ts = int(start_dt.timestamp())
        end_ts = int(end_dt.timestamp())

        token_ids = json.loads(market.get("clobTokenIds", "[]"))
        outcomes = json.loads(market.get("outcomes", "[]"))
        final_prices = json.loads(market.get("outcomePrices", "[]"))

        if not token_ids or not outcomes:
            return None

        winner = outcomes[0] if final_prices[0] == "1" else outcomes[1] if len(outcomes) > 1 else "?"

    except Exception as e:
        logger.debug("parse_error", title=title, error=str(e))
        return None

    # Get price history
    history = get_price_history(token_ids[0], start_ts, end_ts)

    if len(history) < 10:
        return None

    # Find significant moves and continuation patterns
    moves = find_significant_moves(history, start_ts, min_move_pct)
    continuations = find_continuation_patterns(moves)

    return MatchAnalysis(
        title=title[:60],
        winner=winner,
        data_points=len(history),
        significant_moves=moves,
        continuations=continuations,
    )


def print_analysis_report(results: list[MatchAnalysis], min_move_pct: float) -> None:
    """Print analysis report to console."""
    print("=" * 70)
    print("ANALYSE DU DÉLAI DE RÉACTION DU MARCHÉ POLYMARKET")
    print("=" * 70)
    print(f"\nMatchs analysés: {len(results)}")
    print(f"Seuil de mouvement: ≥{min_move_pct}%")
    print()

    total_moves = 0
    total_continuations = 0
    all_continuation_moves = []
    all_continuation_durations = []

    for r in results:
        total_moves += len(r.significant_moves)
        total_continuations += len(r.continuations)

        print(f"{r.title}")
        print(f"  Gagnant: {r.winner}")
        print(f"  Points de données: {r.data_points}")
        print(f"  Mouvements ≥{min_move_pct}%: {len(r.significant_moves)}")

        if r.continuations:
            for c in r.continuations:
                all_continuation_moves.append(c.num_moves)
                all_continuation_durations.append(c.duration_seconds)
                print(f"  ⚡ Continuation à min {c.elapsed_min}: "
                      f"{c.num_moves} moves = {c.total_change_pct:.1f}% "
                      f"({c.duration_seconds}s)")
        print()

    print("=" * 70)
    print("RÉSUMÉ")
    print("=" * 70)

    print(f"\nTotal mouvements significatifs: {total_moves}")
    print(f"Total patterns de continuation: {total_continuations}")

    if all_continuation_moves:
        avg_moves = sum(all_continuation_moves) / len(all_continuation_moves)
        avg_duration = sum(all_continuation_durations) / len(all_continuation_durations)

        print(f"Moves moyens par continuation: {avg_moves:.1f}")
        print(f"Durée moyenne des continuations: {avg_duration:.0f}s")

        print(f"\n{'='*70}")
        print("🎯 INTERPRÉTATION")
        print("=" * 70)

        if total_continuations / len(results) >= 2:
            print(f"\n✅ SIGNAL POSITIF:")
            print(f"   - {total_continuations} continuations sur {len(results)} matchs")
            print(f"   - Le marché met en moyenne {avg_duration:.0f}s ({avg_duration/60:.1f} min)")
            print(f"     pour pleinement intégrer les events majeurs")
            print(f"\n   → Un avantage de 30-40s (PandaScore) permettrait d'entrer")
            print(f"     AVANT ces séquences de réaction et de capturer l'edge")

            # Estimate potential edge
            avg_change = sum(c.total_change_pct for r in results for c in r.continuations) / total_continuations
            print(f"\n   💰 Edge potentiel moyen par continuation: ~{avg_change/2:.1f}%")
            print(f"      (en entrant au début de la séquence)")
        else:
            print(f"\n⚠️  SIGNAL FAIBLE:")
            print(f"   - Seulement {total_continuations} continuations sur {len(results)} matchs")
            print(f"   - Le marché réagit peut-être assez rapidement")
            print(f"   - L'edge de PandaScore pourrait être limité")
    else:
        print("\n❌ PAS DE PATTERNS DÉTECTÉS")
        print("   - Essayez avec un seuil de mouvement plus bas (--min-move 2)")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Polymarket price reaction latency for esports markets"
    )
    parser.add_argument(
        "--limit", type=int, default=30,
        help="Number of matches to analyze (default: 30)"
    )
    parser.add_argument(
        "--min-move", type=float, default=3.0,
        help="Minimum price movement %% to consider significant (default: 3.0)"
    )
    parser.add_argument(
        "--match-type", type=str, default=None,
        choices=["BO1", "BO3", "BO5"],
        help="Filter by match type (default: all)"
    )

    args = parser.parse_args()

    print(f"Récupération des matchs résolus (limit={args.limit})...")
    events = fetch_resolved_matches(limit=args.limit, match_type=args.match_type)
    print(f"Trouvé {len(events)} matchs\n")

    print("Analyse des prix en cours...")
    results = []

    for event in events:
        result = analyze_match(event, min_move_pct=args.min_move)
        if result:
            results.append(result)

    print()
    print_analysis_report(results, args.min_move)


if __name__ == "__main__":
    main()
