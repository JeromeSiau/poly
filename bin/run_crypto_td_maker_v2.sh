#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

# === Crypto TD Maker v2 — data-optimized parameters ===
# Based on slot snapshot analysis (Feb 2026, 68K snapshots, 2K+ slots):
#   - Entry window: min 0-9 only (edge +2.5% vs -17% after min 12)
#   - Price range: 0.75-0.90 (edge +3.9% to +4.9%)
#   - Avoid hours 17-21 UTC (edge -4% to -10%)
#   - Spread boost 2x (edge +5% when spread > 2c)
#   - Chainlink move cap 0.20% (small moves = +4% edge, big = -8%)

SYMBOLS="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT"
MODE="paper"
TARGET_BID="${TARGET_BID:-0.75}"
MAX_BID="${MAX_BID:-0.90}"
WALLET_USD=""
ORDER_SIZE_OVERRIDE=""
MAX_EXPOSURE_OVERRIDE=""
LADDER_RUNGS="${LADDER_RUNGS:-1}"
MIN_MOVE_PCT="${MIN_MOVE_PCT:-0}"
MAX_MOVE_PCT="${MAX_MOVE_PCT:-0.20}"
MIN_ENTRY_MINUTES="${MIN_ENTRY_MINUTES:-0}"
MAX_ENTRY_MINUTES="${MAX_ENTRY_MINUTES:-9}"
STOPLOSS_PEAK="${STOPLOSS_PEAK:-0.75}"
STOPLOSS_EXIT="${STOPLOSS_EXIT:-0.40}"
STOPLOSS_FAIR_MARGIN="${STOPLOSS_FAIR_MARGIN:-0.10}"
ENTRY_FAIR_MARGIN="${ENTRY_FAIR_MARGIN:-0}"
SPREAD_SIZE_MULT="${SPREAD_SIZE_MULT:-2.0}"
AVOID_HOURS="${AVOID_HOURS:-17 18 19 20 21}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --symbols)        SYMBOLS="$2"; shift 2 ;;
    --paper)          MODE="paper"; shift ;;
    --live)           MODE="live"; shift ;;
    --target-bid)     TARGET_BID="$2"; shift 2 ;;
    --max-bid)        MAX_BID="$2"; shift 2 ;;
    --wallet)         WALLET_USD="$2"; shift 2 ;;
    --order-size)     ORDER_SIZE_OVERRIDE="$2"; shift 2 ;;
    --max-exposure)   MAX_EXPOSURE_OVERRIDE="$2"; shift 2 ;;
    --ladder-rungs)   LADDER_RUNGS="$2"; shift 2 ;;
    --min-move-pct)   MIN_MOVE_PCT="$2"; shift 2 ;;
    --max-move-pct)   MAX_MOVE_PCT="$2"; shift 2 ;;
    --min-entry-minutes) MIN_ENTRY_MINUTES="$2"; shift 2 ;;
    --max-entry-minutes) MAX_ENTRY_MINUTES="$2"; shift 2 ;;
    --stoploss-peak)  STOPLOSS_PEAK="$2"; shift 2 ;;
    --stoploss-exit)  STOPLOSS_EXIT="$2"; shift 2 ;;
    --spread-size-mult) SPREAD_SIZE_MULT="$2"; shift 2 ;;
    --avoid-hours)    AVOID_HOURS="$2"; shift 2 ;;
    *)                echo "WARNING: unknown arg '$1' — ignored" >&2; shift ;;
  esac
done

TAG="crypto_td_v2_${SYMBOLS//,/_}_bid${TARGET_BID}"
LOG_FILE="$BASE/logs/${TAG}.log"

DISCOVERY_INTERVAL="${DISCOVERY_INTERVAL:-60}"
MAKER_INTERVAL="${MAKER_INTERVAL:-0.5}"

MODE_FLAG="--paper"
[[ "$MODE" == "live" ]] && MODE_FLAG="--live"

SIZING_ARGS=()
if [[ -n "$WALLET_USD" ]]; then
  ORDER_SIZE="$(awk -v w="$WALLET_USD" 'BEGIN{v=w*0.025; if(v<1) v=1; printf "%.2f", v}')"
  MAX_EXPOSURE="$(awk -v w="$WALLET_USD" 'BEGIN{v=w*0.50; if(v<50) v=50; printf "%.2f", v}')"
  [[ -n "$ORDER_SIZE_OVERRIDE" ]] && ORDER_SIZE="$ORDER_SIZE_OVERRIDE"
  [[ -n "$MAX_EXPOSURE_OVERRIDE" ]] && MAX_EXPOSURE="$MAX_EXPOSURE_OVERRIDE"
  SIZING_ARGS=(--wallet "$WALLET_USD" --order-size "$ORDER_SIZE" --max-exposure "$MAX_EXPOSURE")
elif [[ -n "$ORDER_SIZE_OVERRIDE" || -n "$MAX_EXPOSURE_OVERRIDE" ]]; then
  [[ -n "$ORDER_SIZE_OVERRIDE" ]] && SIZING_ARGS+=(--order-size "$ORDER_SIZE_OVERRIDE")
  [[ -n "$MAX_EXPOSURE_OVERRIDE" ]] && SIZING_ARGS+=(--max-exposure "$MAX_EXPOSURE_OVERRIDE")
fi

exec >> "$LOG_FILE" 2>&1
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] start tag=$TAG mode=$MODE wallet=${WALLET_USD:-auto}"

# shellcheck disable=SC2086
exec "$PYTHON" "$BASE/scripts/run_crypto_td_maker.py" \
  --symbols "$SYMBOLS" $MODE_FLAG \
  --target-bid "$TARGET_BID" --max-bid "$MAX_BID" \
  --ladder-rungs "$LADDER_RUNGS" \
  --min-move-pct "$MIN_MOVE_PCT" --max-move-pct "$MAX_MOVE_PCT" \
  --min-entry-minutes "$MIN_ENTRY_MINUTES" --max-entry-minutes "$MAX_ENTRY_MINUTES" \
  --stoploss-peak "$STOPLOSS_PEAK" --stoploss-exit "$STOPLOSS_EXIT" \
  --stoploss-fair-margin "$STOPLOSS_FAIR_MARGIN" \
  --entry-fair-margin "$ENTRY_FAIR_MARGIN" \
  --spread-size-mult "$SPREAD_SIZE_MULT" \
  --avoid-hours-utc $AVOID_HOURS \
  "${SIZING_ARGS[@]}" \
  --discovery-interval "$DISCOVERY_INTERVAL" --maker-interval "$MAKER_INTERVAL" \
  --strategy-tag "$TAG" --db-url "$DB_URL" \
  "${CB_ARGS[@]}"
