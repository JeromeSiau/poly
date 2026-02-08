#!/bin/bash
set -euo pipefail

# Portable base path (works locally and on server).
BASE="$(cd "$(dirname "$0")" && pwd)"

# Args:
#   1: min edge (default: 2.0%)
#   2: exit edge (default: 0.6%)
#   3: strategy tag (default generated from edges)
#   4: mode -> "paper" (default) or "live"
#   5: wallet size in USD (default: 200)
MIN_EDGE="${1:-0.02}"
EXIT_EDGE="${2:-0.006}"
WALLET_USD="${5:-${WALLET_USD:-200}}"
TAG="${3:-edge_${MIN_EDGE//./p}_${EXIT_EDGE//./p}_${WALLET_USD}usd}"
MODE="${4:-paper}"

cd "$BASE"
export PYTHONPATH="$BASE"
export PYTHONUNBUFFERED=1
mkdir -p "$BASE/logs"
LOG_FILE="$BASE/logs/two_sided_${TAG}.log"

# Dynamic sizing from wallet:
#   min_order        = 2.5% wallet
#   max_order        = 4.0% wallet
#   max_outcome_inv  = 12.5% wallet
#   max_market_net   = 6.0% wallet
MIN_ORDER="$(awk -v w="$WALLET_USD" 'BEGIN{v=w*0.025; if(v<1) v=1; printf "%.2f", v}')"
MAX_ORDER="$(awk -v w="$WALLET_USD" -v min="$MIN_ORDER" 'BEGIN{v=w*0.04; if(v<min+1) v=min+1; printf "%.2f", v}')"
MAX_OUTCOME_INV="$(awk -v w="$WALLET_USD" -v maxo="$MAX_ORDER" 'BEGIN{v=w*0.125; if(v<maxo*2) v=maxo*2; printf "%.2f", v}')"
MAX_MARKET_NET="$(awk -v w="$WALLET_USD" -v maxo="$MAX_ORDER" 'BEGIN{v=w*0.06; if(v<maxo*1.2) v=maxo*1.2; printf "%.2f", v}')"
MAX_ORDERS_PER_CYCLE=1

EXEC_FLAGS=("--paper-fill")
if [[ "$MODE" == "live" ]]; then
  EXEC_FLAGS=("--autopilot" "--no-paper-fill")
fi

exec >> "$LOG_FILE" 2>&1
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] start tag=$TAG mode=$MODE wallet_usd=$WALLET_USD min_edge=$MIN_EDGE exit_edge=$EXIT_EDGE min_order=$MIN_ORDER max_order=$MAX_ORDER max_outcome_inv=$MAX_OUTCOME_INV max_market_net=$MAX_MARKET_NET"

exec "$BASE/.venv/bin/python" "$BASE/scripts/run_two_sided_inventory.py" \
  watch \
  --external-fair \
  "${EXEC_FLAGS[@]}" \
  --include-nonsports \
  --min-liquidity 500 \
  --min-volume-24h 100 \
  --max-days-to-end 3 \
  --min-edge "$MIN_EDGE" \
  --exit-edge "$EXIT_EDGE" \
  --min-order "$MIN_ORDER" \
  --max-order "$MAX_ORDER" \
  --max-outcome-inv "$MAX_OUTCOME_INV" \
  --max-market-net "$MAX_MARKET_NET" \
  --max-orders-per-cycle "$MAX_ORDERS_PER_CYCLE" \
  --max-hold-seconds 21600 \
  --strategy-tag "$TAG" \
  --db-url sqlite+aiosqlite:///data/arb.db \
  --odds-shared-cache \
  --odds-shared-cache-ttl-seconds 900
