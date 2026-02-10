#!/bin/bash
set -euo pipefail

# WebSocket-driven crypto market maker for 15-min binary markets.
#
# Uses BinanceFeed (WS) + PolymarketFeed (WS) for real-time data,
# places GTC limit orders at the bid, and earns the spread.
#
# Usage:
#   run_crypto_maker.sh [--symbols BTCUSDT,ETHUSDT] [--paper|--live] [--wallet 200]
#   run_crypto_maker.sh BTCUSDT,ETHUSDT paper 200   # positional (legacy)

BASE="$(cd "$(dirname "$0")" && pwd)"

# Defaults
SYMBOLS="BTCUSDT,ETHUSDT"
MODE="paper"
WALLET_USD="${WALLET_USD:-200}"

# Parse args: support both named (--symbols X) and positional (X paper 200)
_positional=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --symbols)   SYMBOLS="$2"; shift 2 ;;
    --paper)     MODE="paper"; shift ;;
    --live)      MODE="live"; shift ;;
    --wallet)    WALLET_USD="$2"; shift 2 ;;
    *)           _positional+=("$1"); shift ;;
  esac
done
# Fallback to positional args
[[ ${#_positional[@]} -ge 1 ]] && SYMBOLS="${_positional[0]}"
[[ ${#_positional[@]} -ge 2 ]] && MODE="${_positional[1]}"
[[ ${#_positional[@]} -ge 3 ]] && WALLET_USD="${_positional[2]}"

TAG="crypto_maker_${SYMBOLS//,/_}_${WALLET_USD}usd"

cd "$BASE"
export PYTHONPATH="$BASE"
export PYTHONUNBUFFERED=1
mkdir -p "$BASE/logs" "$BASE/data"
LOG_FILE="$BASE/logs/${TAG}.log"

# Dynamic sizing from wallet:
MIN_ORDER="$(awk -v w="$WALLET_USD" 'BEGIN{v=w*0.025; if(v<1) v=1; printf "%.2f", v}')"
MAX_ORDER="$(awk -v w="$WALLET_USD" -v min="$MIN_ORDER" 'BEGIN{v=w*0.04; if(v<min+1) v=min+1; printf "%.2f", v}')"
MAX_OUTCOME_INV="$(awk -v w="$WALLET_USD" -v maxo="$MAX_ORDER" 'BEGIN{v=w*0.125; if(v<maxo*2) v=maxo*2; printf "%.2f", v}')"
MAX_MARKET_NET="$(awk -v w="$WALLET_USD" -v maxo="$MAX_ORDER" 'BEGIN{v=w*0.06; if(v<maxo*1.2) v=maxo*1.2; printf "%.2f", v}')"

# Overridable intervals:
MAKER_INTERVAL="${MAKER_INTERVAL:-0.5}"
DISCOVERY_INTERVAL="${DISCOVERY_INTERVAL:-60}"
FILL_CHECK_INTERVAL="${FILL_CHECK_INTERVAL:-3}"

MODE_FLAG="--paper"
if [[ "$MODE" == "live" ]]; then
  MODE_FLAG="--live"
fi

exec >> "$LOG_FILE" 2>&1
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] start tag=$TAG mode=$MODE symbols=$SYMBOLS wallet_usd=$WALLET_USD min_order=$MIN_ORDER max_order=$MAX_ORDER max_outcome_inv=$MAX_OUTCOME_INV max_market_net=$MAX_MARKET_NET"

exec "$BASE/.venv/bin/python" "$BASE/scripts/run_crypto_maker.py" \
  --symbols "$SYMBOLS" \
  $MODE_FLAG \
  --maker-interval "$MAKER_INTERVAL" \
  --discovery-interval "$DISCOVERY_INTERVAL" \
  --fill-check-interval "$FILL_CHECK_INTERVAL" \
  --min-edge 0.001 \
  --min-order "$MIN_ORDER" \
  --max-order "$MAX_ORDER" \
  --max-outcome-inv "$MAX_OUTCOME_INV" \
  --max-market-net "$MAX_MARKET_NET" \
  --max-hold-seconds 900 \
  --strategy-tag "$TAG" \
  --db-url "sqlite+aiosqlite:///data/arb.db"
