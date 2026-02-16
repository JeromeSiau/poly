#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

# 5-minute slot collector — BTC only (only symbol with 5m markets on Polymarket).
# Uses 15s snapshot interval for ~20 snapshots per 5m slot.

SYMBOLS="${SYMBOLS:-BTCUSDT}"
DISCOVERY_INTERVAL="${DISCOVERY_INTERVAL:-60}"
SNAPSHOT_INTERVAL="${SNAPSHOT_INTERVAL:-15}"
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

TAG="slot_collector_5m_${SYMBOLS//,/_}"
LOG_FILE="$BASE/logs/${TAG}.log"

exec >> "$LOG_FILE" 2>&1
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] start tag=$TAG symbols=$SYMBOLS duration=5m"

exec "$PYTHON" "$BASE/scripts/run_slot_collector.py" \
  --symbols "$SYMBOLS" \
  --slot-duration 5m \
  --discovery-interval "$DISCOVERY_INTERVAL" \
  --snapshot-interval "$SNAPSHOT_INTERVAL" \
  --resolution-interval "$RESOLUTION_INTERVAL" \
  --strategy-tag "$TAG"
