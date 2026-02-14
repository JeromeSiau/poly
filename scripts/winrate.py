#!/usr/bin/env python3
"""Win rate CLI from Polymarket on-chain wallet activity.

Usage:
    ./run scripts/winrate.py                     # default: 24h
    ./run scripts/winrate.py --hours 17          # last 17h
    ./run scripts/winrate.py --wallet 0x...      # explicit wallet
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from typing import Any

from config.settings import settings
from src.api.winrate import fetch_activity, analyse, resolve_open_markets


def print_report(markets: list[dict[str, Any]], hours: float) -> None:
    resolved = [m for m in markets if m["status"] in ("WIN", "LOSS")]
    still_open = [m for m in markets if m["status"] == "OPEN"]

    wins = [m for m in resolved if m["status"] == "WIN"]
    losses = [m for m in resolved if m["status"] == "LOSS"]

    total_pnl = sum(m["pnl"] for m in resolved)
    total_cost = sum(m["cost"] for m in markets)
    win_pnl = sum(m["pnl"] for m in wins)
    loss_pnl = sum(m["pnl"] for m in losses)

    print()
    print("=" * 70)
    print(f"  Polymarket Win Rate — last {hours}h")
    print("=" * 70)
    print()
    print(f"  Total markets:    {len(markets)}")
    print(f"  Resolved:         {len(resolved)}")
    print(f"  Still open:       {len(still_open)}")
    print()
    print(f"  Wins:             {len(wins)}")
    print(f"  Losses:           {len(losses)}")
    if resolved:
        print(f"  Win rate:         {len(wins)/len(resolved)*100:.1f}%")
    print()
    print(f"  Total PnL:        ${total_pnl:+.2f}")
    print(f"  Total invested:   ${total_cost:.2f}")
    if total_cost > 0:
        print(f"  ROI:              {total_pnl/total_cost*100:+.1f}%")
    if wins:
        print(f"  Avg win:          ${win_pnl/len(wins):+.2f}")
    if losses:
        print(f"  Avg loss:         ${loss_pnl/len(losses):+.2f}")
    if loss_pnl < 0:
        print(f"  Profit factor:    {abs(win_pnl/loss_pnl):.2f}")

    if still_open:
        open_cost = sum(m["cost"] for m in still_open)
        print()
        print(f"  Open exposure:    ${open_cost:.2f} across {len(still_open)} markets")

    if resolved:
        print()
        print("-" * 70)
        print("  Resolved markets:")
        print("-" * 70)
        for m in sorted(resolved, key=lambda x: x["first_ts"]):
            ts = datetime.fromtimestamp(m["first_ts"], timezone.utc).strftime("%m-%d %H:%M")
            tag = "W" if m["status"] == "WIN" else "L"
            title = (m["title"] or m["slug"] or "?")[:45]
            print(f"  [{tag}] ${m['pnl']:+6.2f}  @{m['avg_entry']:.2f}  "
                  f"${m['cost']:6.2f}  {ts}  {title}")

    if len(resolved) > 5:
        print()
        print("-" * 70)
        top_wins = sorted(resolved, key=lambda x: x["pnl"], reverse=True)[:5]
        print("  Top 5 wins:")
        for m in top_wins:
            if m["pnl"] <= 0:
                break
            title = (m["title"] or m["slug"] or "?")[:45]
            print(f"    ${m['pnl']:+.2f}  {title}")

        top_losses = sorted(resolved, key=lambda x: x["pnl"])[:5]
        print("  Top 5 losses:")
        for m in top_losses:
            if m["pnl"] >= 0:
                break
            title = (m["title"] or m["slug"] or "?")[:45]
            print(f"    ${m['pnl']:+.2f}  {title}")

    print()
    print("=" * 70)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Win rate from Polymarket wallet activity"
    )
    parser.add_argument("--hours", type=float, default=24.0,
                        help="Lookback window in hours (default: 24)")
    parser.add_argument("--wallet", type=str, default="",
                        help="Wallet address (default: from .env)")
    args = parser.parse_args()

    wallet = args.wallet or settings.POLYMARKET_WALLET_ADDRESS
    if not wallet:
        print("ERROR: no wallet — set POLYMARKET_WALLET_ADDRESS or use --wallet",
              file=sys.stderr)
        sys.exit(1)

    print(f"Wallet: {wallet}")
    print(f"Window: {args.hours}h")
    print("Fetching activity from Polymarket...")

    rows = fetch_activity(wallet, args.hours)
    if not rows:
        print(f"No activity found in the last {args.hours}h")
        sys.exit(0)

    print(f"Found {len(rows)} activity records")

    markets = analyse(rows)
    resolve_open_markets(markets)
    print_report(markets, args.hours)


if __name__ == "__main__":
    main()
