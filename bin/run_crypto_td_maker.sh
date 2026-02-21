#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

# Usage:
#   run_crypto_td_maker.sh                         # paper, defaults
#   run_crypto_td_maker.sh --live                  # live, auto-detect wallet
#   run_crypto_td_maker.sh --live --wallet 500     # live, manual wallet

SYMBOLS="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT"
MODE="paper"
TARGET_BID="${TARGET_BID:-0.75}"
MAX_BID="${MAX_BID:-0.85}"
WALLET_USD=""
ORDER_SIZE_OVERRIDE=""
MAX_EXPOSURE_OVERRIDE=""
LADDER_RUNGS="${LADDER_RUNGS:-1}"
MIN_MOVE_PCT="${MIN_MOVE_PCT:-0.20}"
MAX_MOVE_PCT="${MAX_MOVE_PCT:-0}"
MIN_ENTRY_MINUTES="${MIN_ENTRY_MINUTES:-10}"
MAX_ENTRY_MINUTES="${MAX_ENTRY_MINUTES:-0}"
ENTRY_MODE="${ENTRY_MODE:-hardcoded}"
MIN_EDGE="${MIN_EDGE:-0.03}"
MODEL_PATH="${MODEL_PATH:-}"
STOPLOSS_PEAK="${STOPLOSS_PEAK:-0.75}"
STOPLOSS_EXIT="${STOPLOSS_EXIT:-0.40}"
STOPLOSS_FAIR_MARGIN="${STOPLOSS_FAIR_MARGIN:-0.10}"
ENTRY_FAIR_MARGIN="${ENTRY_FAIR_MARGIN:-0}"
TAKER_MODE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --symbols)        SYMBOLS="$2"; shift 2 ;;
    --paper)          MODE="paper"; shift ;;
    --live)           MODE="live"; shift ;;
    --taker)          TAKER_MODE="1"; shift ;;
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
    --stoploss-fair-margin) STOPLOSS_FAIR_MARGIN="$2"; shift 2 ;;
    --entry-fair-margin) ENTRY_FAIR_MARGIN="$2"; shift 2 ;;
    --entry-mode) ENTRY_MODE="$2"; shift 2 ;;
    --min-edge)   MIN_EDGE="$2"; shift 2 ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    *)                echo "WARNING: unknown arg '$1' — ignored" >&2; shift ;;
  esac
done

if [[ -n "$TAKER_MODE" ]]; then
    TAG="crypto_td_taker_${SYMBOLS//,/_}_bid${TARGET_BID}"
elif [[ "$ENTRY_MODE" == "ml-dynamic" ]]; then
    TAG="crypto_td_ml_${SYMBOLS//,/_}_edge${MIN_EDGE}"
elif [[ "$ENTRY_MODE" == "ml-hybrid" ]]; then
    TAG="crypto_td_mlh_${SYMBOLS//,/_}_bid${TARGET_BID}"
else
    TAG="crypto_td_maker_${SYMBOLS//,/_}_bid${TARGET_BID}"
fi
LOG_FILE="$BASE/logs/${TAG}.log"

DISCOVERY_INTERVAL="${DISCOVERY_INTERVAL:-60}"
MAKER_INTERVAL="${MAKER_INTERVAL:-0.5}"

MODE_FLAG="--paper"
[[ "$MODE" == "live" ]] && MODE_FLAG="--live"

# Build wallet/sizing args — if --wallet given, compute sizing in shell;
# otherwise let Python auto-detect from Polymarket balance.
# --order-size / --max-exposure override the wallet-derived defaults.
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

ML_ARGS=()
if [[ "$ENTRY_MODE" != "hardcoded" ]]; then
  ML_ARGS+=(--entry-mode "$ENTRY_MODE")
  [[ -n "$MODEL_PATH" ]] && ML_ARGS+=(--model-path "$MODEL_PATH")
  [[ "$ENTRY_MODE" == "ml-dynamic" ]] && ML_ARGS+=(--min-edge "$MIN_EDGE")
fi

exec "$PYTHON" "$BASE/scripts/run_crypto_td_maker.py" \
  --symbols "$SYMBOLS" $MODE_FLAG \
  --target-bid "$TARGET_BID" --max-bid "$MAX_BID" \
  --ladder-rungs "$LADDER_RUNGS" \
  --min-move-pct "$MIN_MOVE_PCT" --max-move-pct "$MAX_MOVE_PCT" \
  --min-entry-minutes "$MIN_ENTRY_MINUTES" --max-entry-minutes "$MAX_ENTRY_MINUTES" \
  --stoploss-peak "$STOPLOSS_PEAK" --stoploss-exit "$STOPLOSS_EXIT" \
  --stoploss-fair-margin "$STOPLOSS_FAIR_MARGIN" \
  --entry-fair-margin "$ENTRY_FAIR_MARGIN" \
  "${SIZING_ARGS[@]}" "${ML_ARGS[@]}" \
  --discovery-interval "$DISCOVERY_INTERVAL" --maker-interval "$MAKER_INTERVAL" \
  --strategy-tag "$TAG" --db-url "$DB_URL" \
  "${CB_ARGS[@]}" \
  ${TAKER_MODE:+--taker}
