#!/usr/bin/env python3
# scripts/performance_report.py
"""Generate paper trading performance report.

Usage:
    uv run python scripts/performance_report.py \
        --since 2026-01-31 \
        --output reports/performance.html
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.models import Base, LiveObservation, PaperTrade
from src.paper_trading.metrics import PaperTradingMetrics, TradeRecord


def main():
    parser = argparse.ArgumentParser(description="Generate performance report")
    parser.add_argument(
        "--since",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="data/arb.db",
        help="Database path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/performance.html",
        help="Output HTML path",
    )

    args = parser.parse_args()

    # Load data
    engine = create_engine(f"sqlite:///{args.db}")
    Session = sessionmaker(bind=engine)
    session = Session()

    query = session.query(PaperTrade)
    if args.since:
        since_date = datetime.strptime(args.since, "%Y-%m-%d")
        query = query.filter(PaperTrade.created_at >= since_date)

    trades_db = query.all()
    session.close()

    if not trades_db:
        print("No trades found")
        return

    # Convert to TradeRecords
    trades = [
        TradeRecord(
            timestamp=t.created_at,
            edge_theoretical=t.edge_theoretical,
            edge_realized=t.edge_realized or 0,
            pnl=t.pnl or 0,
            size=t.size,
        )
        for t in trades_db
    ]

    # Calculate metrics
    metrics = PaperTradingMetrics(trades)
    summary = metrics.as_dict()

    # Print summary
    print("\n" + "=" * 50)
    print("PAPER TRADING PERFORMANCE")
    print("=" * 50)
    print(f"Period: {args.since or 'All time'}")
    print(f"Trades: {summary['n_trades']}")
    print("-" * 50)
    print(f"Total P&L:      ${summary['total_pnl']:,.2f}")
    print(f"Win Rate:       {summary['win_rate']:.1%}")
    print(f"Sharpe Ratio:   {summary['sharpe_ratio']:.2f}")
    print(f"Max Drawdown:   ${summary['max_drawdown']:,.2f}")
    print(f"Profit Factor:  {summary['profit_factor']:.2f}")
    print("-" * 50)
    print(f"Avg Edge (theoretical): {summary['avg_edge_theoretical']:.2%}")
    print(f"Avg Edge (realized):    {summary['avg_edge_realized']:.2%}")
    print("=" * 50)

    # Generate HTML report (simple version)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Paper Trading Performance</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .metric {{ display: inline-block; padding: 20px; margin: 10px; background: #f5f5f5; }}
        .metric-value {{ font-size: 32px; font-weight: bold; }}
        .positive {{ color: #28a745; }}
        .negative {{ color: #dc3545; }}
    </style>
</head>
<body>
    <h1>Paper Trading Performance</h1>
    <p>Period: {args.since or 'All time'}</p>

    <div class="metrics">
        <div class="metric">
            <div class="metric-value {'positive' if summary['total_pnl'] > 0 else 'negative'}">
                ${summary['total_pnl']:,.2f}
            </div>
            <div>Total P&L</div>
        </div>
        <div class="metric">
            <div class="metric-value">{summary['n_trades']}</div>
            <div>Trades</div>
        </div>
        <div class="metric">
            <div class="metric-value">{summary['win_rate']:.1%}</div>
            <div>Win Rate</div>
        </div>
        <div class="metric">
            <div class="metric-value">{summary['sharpe_ratio']:.2f}</div>
            <div>Sharpe Ratio</div>
        </div>
    </div>

    <h2>Edge Analysis</h2>
    <ul>
        <li>Avg Theoretical Edge: {summary['avg_edge_theoretical']:.2%}</li>
        <li>Avg Realized Edge: {summary['avg_edge_realized']:.2%}</li>
        <li>Edge Capture: {(summary['avg_edge_realized'] / summary['avg_edge_theoretical'] * 100) if summary['avg_edge_theoretical'] > 0 else 0:.1f}%</li>
    </ul>

    <h2>Risk Metrics</h2>
    <ul>
        <li>Max Drawdown: ${summary['max_drawdown']:,.2f}</li>
        <li>Profit Factor: {summary['profit_factor']:.2f}</li>
    </ul>
</body>
</html>
"""

    output_path.write_text(html)
    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
