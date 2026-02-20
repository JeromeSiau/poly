#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

# ML-driven TD maker: entry decided by XGBoost with decaying edge threshold.
# Uses the same Python script as the hardcoded maker, just with --entry-mode ml-dynamic.
#
# Usage:
#   run_crypto_td_ml.sh                         # paper, defaults
#   run_crypto_td_ml.sh --live --wallet 500      # live

SYMBOLS="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT"
MODE="paper"
TARGET_BID="${TARGET_BID:-0.75}"
MAX_BID="${MAX_BID:-0.85}"
WALLET_USD=""
ORDER_SIZE_OVERRIDE=""
MAX_EXPOSURE_OVERRIDE=""
MIN_EDGE="${MIN_EDGE:-0.05}"
MODEL_PATH="${MODEL_PATH:-data/models/td_model_latest.joblib}"
STOPLOSS_PEAK="${STOPLOSS_PEAK:-0.75}"
STOPLOSS_EXIT="${STOPLOSS_EXIT:-0.40}"
STOPLOSS_FAIR_MARGIN="${STOPLOSS_FAIR_MARGIN:-0.10}"

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
    --min-edge)       MIN_EDGE="$2"; shift 2 ;;
    --model-path)     MODEL_PATH="$2"; shift 2 ;;
    --stoploss-peak)  STOPLOSS_PEAK="$2"; shift 2 ;;
    --stoploss-exit)  STOPLOSS_EXIT="$2"; shift 2 ;;
    --stoploss-fair-margin) STOPLOSS_FAIR_MARGIN="$2"; shift 2 ;;
    *)                echo "WARNING: unknown arg '$1' — ignored" >&2; shift ;;
  esac
done

TAG="crypto_td_ml_${SYMBOLS//,/_}_edge${MIN_EDGE}"
LOG_FILE="$BASE/logs/${TAG}.log"

DISCOVERY_INTERVAL="${DISCOVERY_INTERVAL:-60}"
MAKER_INTERVAL="${MAKER_INTERVAL:-5}"

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
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] start tag=$TAG mode=$MODE model=$MODEL_PATH min_edge=$MIN_EDGE"

exec "$PYTHON" "$BASE/scripts/run_crypto_td_maker.py" \
  --symbols "$SYMBOLS" $MODE_FLAG \
  --entry-mode ml-dynamic \
  --model-path "$MODEL_PATH" \
  --min-edge "$MIN_EDGE" \
  --target-bid "$TARGET_BID" --max-bid "$MAX_BID" \
  --stoploss-peak "$STOPLOSS_PEAK" --stoploss-exit "$STOPLOSS_EXIT" \
  --stoploss-fair-margin "$STOPLOSS_FAIR_MARGIN" \
  "${SIZING_ARGS[@]}" \
  --discovery-interval "$DISCOVERY_INTERVAL" --maker-interval "$MAKER_INTERVAL" \
  --strategy-tag "$TAG" --db-url "$DB_URL" \
  "${CB_ARGS[@]}"
