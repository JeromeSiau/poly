#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

# Args:
#   1: mode        -> "paper" (default) or "dry"
#   2: wallet USD  -> (default: 200)
#   3: sports      -> Odds API sport keys, comma-separated
SNIPER_MODE="${1:-paper}"
WALLET_USD="${2:-${WALLET_USD:-200}}"
SCORES_SPORTS="${3:-${SCORES_SPORTS:-soccer_epl,soccer_spain_la_liga,soccer_italy_serie_a,soccer_brazil_campeonato,soccer_argentina_primera_division}}"

# Scoring loop (overridable via env)
SCORES_INTERVAL="${SCORES_INTERVAL:-120}"

# Spike detection (overridable via env)
SPIKE_THRESHOLD="${SPIKE_THRESHOLD:-0.15}"
SPIKE_WINDOW="${SPIKE_WINDOW:-60}"
SPIKE_COOLDOWN="${SPIKE_COOLDOWN:-120}"
SPIKE_POLL_INTERVAL="${SPIKE_POLL_INTERVAL:-2.0}"

# Market scanning (overridable via env)
MARKET_LIMIT="${MARKET_LIMIT:-1500}"
BOOK_CONCURRENCY="${BOOK_CONCURRENCY:-40}"
EVENT_PREFIXES="${EVENT_PREFIXES:-epl,lal,sea,fl1,por,bun,tur,arg,col1,nba,nfl,cbb,atp,wta,ucl,cs2,super}"
MAPPER_REFRESH="${MAPPER_REFRESH:-300}"

# Sizing from wallet (overridable via env)
MAX_ORDER="${MAX_ORDER:-$(awk -v w="$WALLET_USD" 'BEGIN{v=w*0.05; if(v<5) v=5; printf "%.2f", v}')}"
MAX_OUTCOME_INV="${MAX_OUTCOME_INV:-$(awk -v w="$WALLET_USD" -v maxo="$MAX_ORDER" 'BEGIN{v=w*0.50; if(v<maxo*6) v=maxo*6; printf "%.2f", v}')}"
MAX_MARKET_NET="${MAX_MARKET_NET:-$(awk -v w="$WALLET_USD" -v maxo="$MAX_ORDER" 'BEGIN{v=w*0.40; if(v<maxo*4) v=maxo*4; printf "%.2f", v}')}"

STRATEGY_TAG="${STRATEGY_TAG:-sniper_sports}"

DRY_RUN_FLAG=()
if [[ "$SNIPER_MODE" == "dry" ]]; then
  DRY_RUN_FLAG=("--dry-run")
fi

LOG_FILE="$BASE/logs/sniper_${STRATEGY_TAG}.log"

exec >> "$LOG_FILE" 2>&1
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] start mode=$SNIPER_MODE wallet_usd=$WALLET_USD scores_sports=$SCORES_SPORTS scores_interval=$SCORES_INTERVAL spike_threshold=$SPIKE_THRESHOLD spike_window=$SPIKE_WINDOW spike_cooldown=$SPIKE_COOLDOWN spike_poll_interval=$SPIKE_POLL_INTERVAL market_limit=$MARKET_LIMIT book_concurrency=$BOOK_CONCURRENCY event_prefixes=$EVENT_PREFIXES max_order=$MAX_ORDER max_outcome_inv=$MAX_OUTCOME_INV max_market_net=$MAX_MARKET_NET strategy_tag=$STRATEGY_TAG"

exec "$PYTHON" "$BASE/scripts/run_sniper.py" \
  --scores-sports "$SCORES_SPORTS" \
  --scores-interval "$SCORES_INTERVAL" \
  --spike-threshold "$SPIKE_THRESHOLD" \
  --spike-window "$SPIKE_WINDOW" \
  --spike-cooldown "$SPIKE_COOLDOWN" \
  --spike-poll-interval "$SPIKE_POLL_INTERVAL" \
  --market-limit "$MARKET_LIMIT" \
  --book-concurrency "$BOOK_CONCURRENCY" \
  --event-prefixes "$EVENT_PREFIXES" \
  --max-order "$MAX_ORDER" \
  --mapper-refresh-seconds "$MAPPER_REFRESH" \
  --strategy-tag "$STRATEGY_TAG" \
  --wallet-usd "$WALLET_USD" \
  --max-outcome-inv "$MAX_OUTCOME_INV" \
  --max-market-net "$MAX_MARKET_NET" \
  --db-url "$DB_URL" \
  "${DRY_RUN_FLAG[@]}" \
  "${CB_ARGS[@]}"
