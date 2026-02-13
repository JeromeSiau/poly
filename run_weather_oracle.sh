#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

# Args:
#   1: mode -> "watch" (default), "scan"
#   2: scan interval in seconds (default: 300)
MODE="${1:-watch}"
SCAN_INTERVAL="${2:-300.0}"

# Overridable via env
MAX_ENTRY_PRICE="${MAX_ENTRY_PRICE:-0.12}"
MIN_CONFIDENCE="${MIN_CONFIDENCE:-0.65}"
PAPER_SIZE_USD="${PAPER_SIZE_USD:-3.0}"
MAX_DAILY_SPEND="${MAX_DAILY_SPEND:-50.0}"
FORECAST_DAYS="${FORECAST_DAYS:-7}"

LOG_FILE="$BASE/logs/weather_oracle.log"

# Inject settings via env
export WEATHER_ORACLE_SCAN_INTERVAL="$SCAN_INTERVAL"
export WEATHER_ORACLE_MAX_ENTRY_PRICE="$MAX_ENTRY_PRICE"
export WEATHER_ORACLE_MIN_FORECAST_CONFIDENCE="$MIN_CONFIDENCE"
export WEATHER_ORACLE_PAPER_SIZE_USD="$PAPER_SIZE_USD"
export WEATHER_ORACLE_MAX_DAILY_SPEND="$MAX_DAILY_SPEND"
export WEATHER_ORACLE_FORECAST_DAYS="$FORECAST_DAYS"

exec >> "$LOG_FILE" 2>&1
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] start mode=$MODE interval=$SCAN_INTERVAL max_entry=$MAX_ENTRY_PRICE min_conf=$MIN_CONFIDENCE paper_size=$PAPER_SIZE_USD max_daily=$MAX_DAILY_SPEND"

exec "$PYTHON" "$BASE/scripts/run_weather_oracle.py" \
  "$MODE" \
  --interval "$SCAN_INTERVAL" \
  --max-entry-price "$MAX_ENTRY_PRICE" \
  --min-confidence "$MIN_CONFIDENCE" \
  --paper-size "$PAPER_SIZE_USD" \
  --max-daily-spend "$MAX_DAILY_SPEND" \
  "${CB_ARGS[@]}"
