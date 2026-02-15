#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

# Usage:
#   run_slot_collector.sh                         # default: all 4 symbols
#   run_slot_collector.sh --symbols BTC,ETH,SOL   # custom symbols

SYMBOLS="${SYMBOLS:-BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT}"
DISCOVERY_INTERVAL="${DISCOVERY_INTERVAL:-60}"
SNAPSHOT_INTERVAL="${SNAPSHOT_INTERVAL:-30}"
RESOLUTION_INTERVAL="${RESOLUTION_INTERVAL:-60}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --symbols)              SYMBOLS="$2"; shift 2 ;;
    --discovery-interval)   DISCOVERY_INTERVAL="$2"; shift 2 ;;
    --snapshot-interval)    SNAPSHOT_INTERVAL="$2"; shift 2 ;;
    --resolution-interval)  RESOLUTION_INTERVAL="$2"; shift 2 ;;
    *)                      shift ;;
  esac
done

TAG="slot_collector_${SYMBOLS//,/_}"
LOG_FILE="$BASE/logs/${TAG}.log"

exec >> "$LOG_FILE" 2>&1
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] start tag=$TAG symbols=$SYMBOLS"

exec "$PYTHON" "$BASE/scripts/run_slot_collector.py" \
  --symbols "$SYMBOLS" \
  --discovery-interval "$DISCOVERY_INTERVAL" \
  --snapshot-interval "$SNAPSHOT_INTERVAL" \
  --resolution-interval "$RESOLUTION_INTERVAL" \
  --strategy-tag "$TAG"
