#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

# Args:
#   1: symbols     -> "BTCUSDT,ETHUSDT" (default)
#   2: timeframes  -> "300,900" (default)
#   3: budget      -> 200 (USD per market)
#   4: min_edge    -> 0.01 (1%)
#   5: mode        -> "paper" (default) or "live"
SYMBOLS="${1:-BTCUSDT,ETHUSDT}"
TIMEFRAMES="${2:-300,900}"
BUDGET="${3:-200}"
MIN_EDGE="${4:-0.01}"
MODE="${5:-paper}"

# Tuning knobs (overridable via env)
ENTRY_WINDOW="${ENTRY_WINDOW:-30}"
FEE_BPS="${FEE_BPS:-100}"
MAX_CONCURRENT="${MAX_CONCURRENT:-10}"
DISCOVERY_LEAD="${DISCOVERY_LEAD:-120}"
POLL_INTERVAL="${POLL_INTERVAL:-5.0}"

TAG="crypto_2s_${SYMBOLS//,/_}_${TIMEFRAMES//,/_}"
LOG_FILE="$BASE/logs/crypto_two_sided_${TAG}.log"

EXTRA_FLAGS="--paper"
if [ "$MODE" = "live" ]; then
    EXTRA_FLAGS="--autopilot"
fi

exec >> "$LOG_FILE" 2>&1
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] start symbols=$SYMBOLS timeframes=$TIMEFRAMES budget=$BUDGET min_edge=$MIN_EDGE mode=$MODE entry_window=$ENTRY_WINDOW fee_bps=$FEE_BPS max_concurrent=$MAX_CONCURRENT"

exec "$PYTHON" "$BASE/scripts/run_crypto_two_sided.py" \
  --symbols "$SYMBOLS" \
  --timeframes "$TIMEFRAMES" \
  --budget "$BUDGET" \
  --min-edge "$MIN_EDGE" \
  --entry-window "$ENTRY_WINDOW" \
  --fee-bps "$FEE_BPS" \
  --max-concurrent "$MAX_CONCURRENT" \
  --discovery-lead "$DISCOVERY_LEAD" \
  --poll-interval "$POLL_INTERVAL" \
  --tag "$TAG" \
  $EXTRA_FLAGS \
  "${CB_ARGS[@]}"
