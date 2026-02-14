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
    *)                shift ;;
  esac
done

TAG="crypto_td_maker_${SYMBOLS//,/_}_bid${TARGET_BID}"
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

exec "$PYTHON" "$BASE/scripts/run_crypto_td_maker.py" \
  --symbols "$SYMBOLS" $MODE_FLAG \
  --target-bid "$TARGET_BID" --max-bid "$MAX_BID" \
  --ladder-rungs "$LADDER_RUNGS" \
  "${SIZING_ARGS[@]}" \
  --discovery-interval "$DISCOVERY_INTERVAL" --maker-interval "$MAKER_INTERVAL" \
  --strategy-tag "$TAG" --db-url "$DB_URL" \
  "${CB_ARGS[@]}"
