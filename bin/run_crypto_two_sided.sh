#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

# Crypto 15-minute market making via two-sided inventory engine.
#
# Args:
#   1: symbols    -> "BTCUSDT,ETHUSDT" (default)
#   2: mode       -> "paper" (default) or "live"
#   3: wallet USD -> 200 (default)
#   4: min edge   -> 0.005 (default, 0.5%)
#   5: exit edge  -> 0.002 (default, 0.2%)

SYMBOLS="${1:-BTCUSDT,ETHUSDT}"
MODE="${2:-paper}"
WALLET_USD="${3:-${WALLET_USD:-200}}"
MIN_EDGE="${4:-0.005}"
EXIT_EDGE="${5:-0.002}"

TAG="crypto_two_sided_${SYMBOLS//,/_}_${WALLET_USD}usd"
LOG_FILE="$BASE/logs/${TAG}.log"

# Dynamic sizing from wallet
MIN_ORDER="$(awk -v w="$WALLET_USD" 'BEGIN{v=w*0.025; if(v<1) v=1; printf "%.2f", v}')"
MAX_ORDER="$(awk -v w="$WALLET_USD" -v min="$MIN_ORDER" 'BEGIN{v=w*0.04; if(v<min+1) v=min+1; printf "%.2f", v}')"
MAX_OUTCOME_INV="$(awk -v w="$WALLET_USD" -v maxo="$MAX_ORDER" 'BEGIN{v=w*0.125; if(v<maxo*2) v=maxo*2; printf "%.2f", v}')"
MAX_MARKET_NET="$(awk -v w="$WALLET_USD" -v maxo="$MAX_ORDER" 'BEGIN{v=w*0.06; if(v<maxo*1.2) v=maxo*1.2; printf "%.2f", v}')"

# Crypto-specific overrides (overridable via env)
WATCH_INTERVAL="${WATCH_INTERVAL:-3}"
SIGNAL_COOLDOWN="${SIGNAL_COOLDOWN:-5}"
MAX_ORDERS_PER_CYCLE="${MAX_ORDERS_PER_CYCLE:-8}"
MAX_BOOK_CONCURRENCY="${MAX_BOOK_CONCURRENCY:-10}"

EXEC_FLAGS=("--paper-fill")
if [[ "$MODE" == "live" ]]; then
  EXEC_FLAGS=("--autopilot" "--no-paper-fill")
fi

exec >> "$LOG_FILE" 2>&1
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] start tag=$TAG mode=$MODE symbols=$SYMBOLS wallet_usd=$WALLET_USD min_edge=$MIN_EDGE exit_edge=$EXIT_EDGE min_order=$MIN_ORDER max_order=$MAX_ORDER max_outcome_inv=$MAX_OUTCOME_INV max_market_net=$MAX_MARKET_NET interval=$WATCH_INTERVAL"

exec "$PYTHON" "$BASE/scripts/run_two_sided_inventory.py" \
  watch \
  --crypto-symbols "$SYMBOLS" \
  --no-external-fair \
  --maker-mode \
  "${EXEC_FLAGS[@]}" \
  --interval "$WATCH_INTERVAL" \
  --signal-cooldown "$SIGNAL_COOLDOWN" \
  --min-edge "$MIN_EDGE" --exit-edge "$EXIT_EDGE" \
  --min-order "$MIN_ORDER" --max-order "$MAX_ORDER" \
  --max-outcome-inv "$MAX_OUTCOME_INV" --max-market-net "$MAX_MARKET_NET" \
  --max-orders-per-cycle "$MAX_ORDERS_PER_CYCLE" \
  --max-book-concurrency "$MAX_BOOK_CONCURRENCY" \
  --max-hold-seconds 900 \
  --no-settle-resolved --no-pair-merge \
  --strategy-tag "$TAG" --db-url "$DB_URL" \
  "${CB_ARGS[@]}"
