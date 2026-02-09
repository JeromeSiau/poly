#!/bin/bash
set -euo pipefail

# Portable base path (works locally and on server).
BASE="$(cd "$(dirname "$0")" && pwd)"

# Args:
#   1: strategy -> "both" (default), "time_decay", "long_vol"
#   2: symbols  -> "BTCUSDT,ETHUSDT" (default)
#   3: td threshold (default: 0.88)
#   4: lv threshold (default: 0.15)
#   5: scan interval in seconds (default: 3.0)
STRATEGY="${1:-both}"
SYMBOLS="${2:-BTCUSDT,ETHUSDT}"
TD_THRESHOLD="${3:-0.88}"
LV_THRESHOLD="${4:-0.15}"
SCAN_INTERVAL="${5:-3.0}"

# Entry window (overridable via env)
MIN_ENTRY_TIME="${MIN_ENTRY_TIME:-120}"
MAX_ENTRY_TIME="${MAX_ENTRY_TIME:-300}"

# Gap thresholds (overridable via env)
TD_MIN_GAP_PCT="${TD_MIN_GAP_PCT:-0.3}"
LV_MAX_GAP_PCT="${LV_MAX_GAP_PCT:-0.5}"

# Paper trading sizing (overridable via env)
PAPER_SIZE_USD="${PAPER_SIZE_USD:-10.0}"
PAPER_CAPITAL="${PAPER_CAPITAL:-1000.0}"

TAG="crypto_minute_${STRATEGY}_${SYMBOLS//,/_}_td${TD_THRESHOLD}_lv${LV_THRESHOLD}"

cd "$BASE"
export PYTHONPATH="$BASE"
export PYTHONUNBUFFERED=1
mkdir -p "$BASE/logs" "$BASE/data"
LOG_FILE="$BASE/logs/crypto_minute_${TAG}.log"

# Inject settings via env
export CRYPTO_MINUTE_SYMBOLS="$SYMBOLS"
export CRYPTO_MINUTE_TD_THRESHOLD="$TD_THRESHOLD"
export CRYPTO_MINUTE_LV_THRESHOLD="$LV_THRESHOLD"
export CRYPTO_MINUTE_SCAN_INTERVAL="$SCAN_INTERVAL"
export CRYPTO_MINUTE_MIN_ENTRY_TIME="$MIN_ENTRY_TIME"
export CRYPTO_MINUTE_MAX_ENTRY_TIME="$MAX_ENTRY_TIME"
export CRYPTO_MINUTE_TD_MIN_GAP_PCT="$TD_MIN_GAP_PCT"
export CRYPTO_MINUTE_LV_MAX_GAP_PCT="$LV_MAX_GAP_PCT"
export CRYPTO_MINUTE_PAPER_SIZE_USD="$PAPER_SIZE_USD"
export CRYPTO_MINUTE_PAPER_CAPITAL="$PAPER_CAPITAL"

exec >> "$LOG_FILE" 2>&1
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] start strategy=$STRATEGY symbols=$SYMBOLS td_threshold=$TD_THRESHOLD lv_threshold=$LV_THRESHOLD scan_interval=$SCAN_INTERVAL min_entry_time=$MIN_ENTRY_TIME max_entry_time=$MAX_ENTRY_TIME td_min_gap=$TD_MIN_GAP_PCT lv_max_gap=$LV_MAX_GAP_PCT paper_size=$PAPER_SIZE_USD paper_capital=$PAPER_CAPITAL"

exec "$BASE/.venv/bin/python" "$BASE/scripts/run_crypto_minute.py" \
  --symbols "$SYMBOLS" \
  --td-threshold "$TD_THRESHOLD" \
  --lv-threshold "$LV_THRESHOLD" \
  --scan-interval "$SCAN_INTERVAL"
