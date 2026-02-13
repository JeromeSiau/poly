#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

# Args:
#   1: mode           -> "paper" (default) or "autopilot"
#   2: scan interval  -> seconds between scans (default: 300)
MODE="${1:-paper}"
SCAN_INTERVAL="${2:-${SCAN_INTERVAL:-300}}"

# Fear scoring (overridable via env)
MIN_FEAR_SCORE="${MIN_FEAR_SCORE:-0.5}"
MIN_YES_PRICE="${MIN_YES_PRICE:-0.15}"
MAX_YES_PRICE="${MAX_YES_PRICE:-0.65}"

# Risk limits (overridable via env)
MAX_CLUSTER_PCT="${MAX_CLUSTER_PCT:-0.30}"
MAX_POSITION_PCT="${MAX_POSITION_PCT:-0.10}"
KELLY_FRACTION="${KELLY_FRACTION:-0.25}"

# Exit rules (overridable via env)
EXIT_NO_PRICE="${EXIT_NO_PRICE:-0.95}"
STOP_YES_PRICE="${STOP_YES_PRICE:-0.70}"

# Spike detection (overridable via env)
SPIKE_THRESHOLD="${SPIKE_THRESHOLD:-0.05}"
SPIKE_WINDOW="${SPIKE_WINDOW:-600}"

# Capital allocation (overridable via env)
FEAR_ALLOCATION="${FEAR_ALLOCATION:-20}"

AUTOPILOT_FLAG=()
if [[ "$MODE" == "autopilot" ]]; then
  AUTOPILOT_FLAG=("--autopilot")
fi

LOG_FILE="$BASE/logs/fear_selling.log"

# Inject settings via env
export FEAR_SELLING_ENABLED=true
export FEAR_SELLING_SCAN_INTERVAL="$SCAN_INTERVAL"
export FEAR_SELLING_MIN_FEAR_SCORE="$MIN_FEAR_SCORE"
export FEAR_SELLING_MIN_YES_PRICE="$MIN_YES_PRICE"
export FEAR_SELLING_MAX_YES_PRICE="$MAX_YES_PRICE"
export FEAR_SELLING_MAX_CLUSTER_PCT="$MAX_CLUSTER_PCT"
export FEAR_SELLING_MAX_POSITION_PCT="$MAX_POSITION_PCT"
export FEAR_SELLING_KELLY_FRACTION="$KELLY_FRACTION"
export FEAR_SELLING_EXIT_NO_PRICE="$EXIT_NO_PRICE"
export FEAR_SELLING_STOP_YES_PRICE="$STOP_YES_PRICE"
export FEAR_SELLING_SPIKE_THRESHOLD_PCT="$SPIKE_THRESHOLD"
export FEAR_SELLING_SPIKE_WINDOW_SECONDS="$SPIKE_WINDOW"
export CAPITAL_ALLOCATION_FEAR_PCT="$FEAR_ALLOCATION"

exec >> "$LOG_FILE" 2>&1
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] start mode=$MODE interval=$SCAN_INTERVAL fear_score=$MIN_FEAR_SCORE cluster_pct=$MAX_CLUSTER_PCT kelly=$KELLY_FRACTION exit_no=$EXIT_NO_PRICE stop_yes=$STOP_YES_PRICE alloc=$FEAR_ALLOCATION"

exec "$PYTHON" "$BASE/scripts/run_fear_selling.py" \
  --scan-interval "$SCAN_INTERVAL" \
  "${AUTOPILOT_FLAG[@]}" \
  "${CB_ARGS[@]}"
