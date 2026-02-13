#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

# Passive time-decay maker for Kalshi crypto bracket markets.
#
# Usage:
#   run_kalshi_td_maker.sh                       # paper, defaults
#   run_kalshi_td_maker.sh --live                 # real orders
#   run_kalshi_td_maker.sh --demo --live          # demo sandbox + real orders
#   run_kalshi_td_maker.sh --target-bid 78        # custom bid (cents)

MODE="paper"
DEMO=""
TARGET_BID="${TARGET_BID:-75}"
ORDER_SIZE="${ORDER_SIZE:-10}"
MAX_EXPOSURE="${MAX_EXPOSURE:-200}"
SYMBOLS="${SYMBOLS:-KXBTCD,KXETHD}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --paper)        MODE="paper"; shift ;;
    --live)         MODE="live"; shift ;;
    --demo)         DEMO="--demo"; shift ;;
    --target-bid)   TARGET_BID="$2"; shift 2 ;;
    --order-size)   ORDER_SIZE="$2"; shift 2 ;;
    --max-exposure) MAX_EXPOSURE="$2"; shift 2 ;;
    --symbols)      SYMBOLS="$2"; shift 2 ;;
    *)              shift ;;
  esac
done

TAG="kalshi_td_maker_bid${TARGET_BID}"
LOG_FILE="$BASE/logs/${TAG}.log"

MODE_FLAG="--paper"
[[ "$MODE" == "live" ]] && MODE_FLAG="--live"

exec >> "$LOG_FILE" 2>&1
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] start tag=$TAG mode=$MODE target_bid=${TARGET_BID}c order_size=$ORDER_SIZE max_exposure=$MAX_EXPOSURE symbols=$SYMBOLS"

exec "$PYTHON" "$BASE/scripts/run_kalshi_td_maker.py" \
  $MODE_FLAG $DEMO \
  --target-bid "$TARGET_BID" \
  --order-size "$ORDER_SIZE" \
  --max-exposure "$MAX_EXPOSURE" \
  --symbols "$SYMBOLS" \
  "${CB_ARGS[@]}"
