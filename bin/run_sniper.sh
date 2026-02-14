#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

# Usage:
#   run_sniper.sh                           # paper, defaults
#   run_sniper.sh --live                    # live trading
#   run_sniper.sh --live --capital 1000     # live, custom capital

MODE="paper"
MIN_PRICE="${MIN_PRICE:-0.95}"
CAPITAL="${CAPITAL:-500}"
RISK_PCT="${RISK_PCT:-0.01}"
MAX_PER_MARKET="${MAX_PER_MARKET:-0.05}"
SCAN_INTERVAL="${SCAN_INTERVAL:-15}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --paper)          MODE="paper"; shift ;;
    --live)           MODE="live"; shift ;;
    --min-price)      MIN_PRICE="$2"; shift 2 ;;
    --capital)        CAPITAL="$2"; shift 2 ;;
    --risk-pct)       RISK_PCT="$2"; shift 2 ;;
    --max-per-market) MAX_PER_MARKET="$2"; shift 2 ;;
    --scan-interval)  SCAN_INTERVAL="$2"; shift 2 ;;
    *)                shift ;;
  esac
done

TAG="sniper_engine_p${MIN_PRICE}"
LOG_FILE="$BASE/logs/${TAG}.log"

MODE_FLAG="--paper"
[[ "$MODE" == "live" ]] && MODE_FLAG="--live"

exec >> "$LOG_FILE" 2>&1
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] start tag=$TAG mode=$MODE min_price=$MIN_PRICE capital=$CAPITAL"

exec "$PYTHON" "$BASE/scripts/run_sniper.py" \
  $MODE_FLAG \
  --min-price "$MIN_PRICE" \
  --capital "$CAPITAL" \
  --risk-pct "$RISK_PCT" \
  --max-per-market "$MAX_PER_MARKET" \
  --scan-interval "$SCAN_INTERVAL" \
  --strategy-tag "$TAG" --db-url "$DB_URL" \
  "${CB_ARGS[@]}"
