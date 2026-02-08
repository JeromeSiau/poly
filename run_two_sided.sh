#!/bin/bash
set -euo pipefail

# Portable base path (works locally and on server).
BASE="$(cd "$(dirname "$0")" && pwd)"

# Args:
#   1: min edge (default: 2.0%)
#   2: exit edge (default: 0.6%)
#   3: strategy tag (default generated from edges + fair mode)
#   4: mode -> "paper" (default) or "live"
#   5: wallet size in USD (default: 200)
#   6: fair mode -> "external_only" (default), "timing_only", "hybrid"
#   7: fair blend (used only for hybrid, default: 0.6)
#   8: strategy style -> "default" (default) or "rn1_mimic"
MIN_EDGE="${1:-0.02}"
EXIT_EDGE="${2:-0.006}"
WALLET_USD="${5:-${WALLET_USD:-200}}"
FAIR_MODE="${6:-${FAIR_MODE:-external_only}}"
FAIR_BLEND="${7:-${FAIR_BLEND:-0.6}}"
STRATEGY_STYLE="${8:-${STRATEGY_STYLE:-default}}"
TAG="${3:-edge_${MIN_EDGE//./p}_${EXIT_EDGE//./p}_${WALLET_USD}usd_${FAIR_MODE}_${STRATEGY_STYLE}}"
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
WATCH_INTERVAL="${WATCH_INTERVAL:-5}"
SIGNAL_COOLDOWN="${SIGNAL_COOLDOWN:-8}"
MAX_ORDERS_PER_CYCLE="${MAX_ORDERS_PER_CYCLE:-4}"
PAIR_MERGE_MIN_EDGE="${PAIR_MERGE_MIN_EDGE:-0.003}"
MIN_LIQUIDITY="${MIN_LIQUIDITY:-500}"
MIN_VOLUME_24H="${MIN_VOLUME_24H:-100}"
MAX_DAYS_TO_END="${MAX_DAYS_TO_END:-1}"
INCLUDE_NONSPORTS="${INCLUDE_NONSPORTS:-0}"
EVENT_PREFIXES="${EVENT_PREFIXES:-}"

UNIVERSE_FLAGS=()
if [[ "$INCLUDE_NONSPORTS" == "1" ]]; then
  UNIVERSE_FLAGS+=("--include-nonsports")
fi

EXEC_FLAGS=("--paper-fill")
if [[ "$MODE" == "live" ]]; then
  EXEC_FLAGS=("--autopilot" "--no-paper-fill")
fi

FAIR_FLAGS=()
case "$FAIR_MODE" in
  external_only)
    FAIR_FLAGS+=("--external-fair" "--fair-blend" "1.0")
    ;;
  timing_only)
    FAIR_FLAGS+=("--no-external-fair")
    ;;
  hybrid)
    FAIR_FLAGS+=("--external-fair" "--fair-blend" "$FAIR_BLEND")
    ;;
  *)
    echo "Invalid FAIR_MODE: $FAIR_MODE (expected external_only|timing_only|hybrid)" >&2
    exit 1
    ;;
esac

STYLE_FLAGS=()
case "$STRATEGY_STYLE" in
  default)
    ;;
  rn1_mimic)
    # RN1-like profile: avoid classic SELL/settlement loop, keep BUY pressure.
    STYLE_FLAGS+=("--buy-only" "--no-settle-resolved")
    # More RN1-like market coverage/cadence defaults.
    if [[ "$WATCH_INTERVAL" == "5" ]]; then WATCH_INTERVAL="1"; fi
    if [[ "$SIGNAL_COOLDOWN" == "8" ]]; then SIGNAL_COOLDOWN="1"; fi
    if [[ "$MAX_ORDERS_PER_CYCLE" == "4" ]]; then MAX_ORDERS_PER_CYCLE="12"; fi
    if [[ "$PAIR_MERGE_MIN_EDGE" == "0.003" ]]; then PAIR_MERGE_MIN_EDGE="0.001"; fi
    if [[ "$MIN_LIQUIDITY" == "500" ]]; then MIN_LIQUIDITY="100"; fi
    if [[ "$MIN_VOLUME_24H" == "100" ]]; then MIN_VOLUME_24H="20"; fi
    if [[ "$MAX_DAYS_TO_END" == "1" ]]; then MAX_DAYS_TO_END="3"; fi
    if [[ "$INCLUDE_NONSPORTS" == "0" ]]; then INCLUDE_NONSPORTS="1"; fi
    if [[ -z "$EVENT_PREFIXES" ]]; then EVENT_PREFIXES="epl,cs2,lal,nba,fl1,sea,por,tur,cbb"; fi
    ;;
  *)
    echo "Invalid STRATEGY_STYLE: $STRATEGY_STYLE (expected default|rn1_mimic)" >&2
    exit 1
    ;;
esac

# Rebuild universe flags after strategy-style overrides.
UNIVERSE_FLAGS=()
if [[ "$INCLUDE_NONSPORTS" == "1" ]]; then
  UNIVERSE_FLAGS+=("--include-nonsports")
fi
if [[ -n "$EVENT_PREFIXES" ]]; then
  UNIVERSE_FLAGS+=("--event-prefixes" "$EVENT_PREFIXES")
fi

exec >> "$LOG_FILE" 2>&1
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] start tag=$TAG mode=$MODE wallet_usd=$WALLET_USD fair_mode=$FAIR_MODE fair_blend=$FAIR_BLEND strategy_style=$STRATEGY_STYLE min_edge=$MIN_EDGE exit_edge=$EXIT_EDGE min_order=$MIN_ORDER max_order=$MAX_ORDER max_outcome_inv=$MAX_OUTCOME_INV max_market_net=$MAX_MARKET_NET interval=$WATCH_INTERVAL signal_cooldown=$SIGNAL_COOLDOWN max_orders_per_cycle=$MAX_ORDERS_PER_CYCLE pair_merge_min_edge=$PAIR_MERGE_MIN_EDGE min_liquidity=$MIN_LIQUIDITY min_volume_24h=$MIN_VOLUME_24H max_days_to_end=$MAX_DAYS_TO_END include_nonsports=$INCLUDE_NONSPORTS event_prefixes=$EVENT_PREFIXES"

exec "$BASE/.venv/bin/python" "$BASE/scripts/run_two_sided_inventory.py" \
  watch \
  "${FAIR_FLAGS[@]}" \
  "${STYLE_FLAGS[@]}" \
  "${EXEC_FLAGS[@]}" \
  --interval "$WATCH_INTERVAL" \
  --signal-cooldown "$SIGNAL_COOLDOWN" \
  "${UNIVERSE_FLAGS[@]}" \
  --min-liquidity "$MIN_LIQUIDITY" \
  --min-volume-24h "$MIN_VOLUME_24H" \
  --max-days-to-end "$MAX_DAYS_TO_END" \
  --min-edge "$MIN_EDGE" \
  --exit-edge "$EXIT_EDGE" \
  --min-order "$MIN_ORDER" \
  --max-order "$MAX_ORDER" \
  --max-outcome-inv "$MAX_OUTCOME_INV" \
  --max-market-net "$MAX_MARKET_NET" \
  --max-orders-per-cycle "$MAX_ORDERS_PER_CYCLE" \
  --max-hold-seconds 21600 \
  --pair-merge \
  --pair-merge-min-edge "$PAIR_MERGE_MIN_EDGE" \
  --strategy-tag "$TAG" \
  --db-url sqlite+aiosqlite:///data/arb.db \
  --odds-shared-cache \
  --odds-shared-cache-ttl-seconds 900
