#!/bin/bash
set -euo pipefail

BASE="$(cd "$(dirname "$0")/.." && pwd)"
DB_PATH="$BASE/data/arb.db"
MAKE_BACKUP=1
ASSUME_YES=0
KEEP_CACHE=0

usage() {
  cat <<'EOF'
Usage:
  ./reset_arb_db.sh [options]

Options:
  --db <path>       SQLite file path (default: ./data/arb.db)
  --yes             No confirmation prompt
  --no-backup       Skip backup copy before reset
  --keep-cache      Keep odds_api_cache table data
  -h, --help        Show this help

Examples:
  ./reset_arb_db.sh
  ./reset_arb_db.sh --yes
  ./reset_arb_db.sh --db /home/ploi/orb.lvlup-dev.com/data/arb.db --yes
  ./reset_arb_db.sh --yes --keep-cache
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --db)
      shift
      [[ $# -gt 0 ]] || { echo "Missing value after --db" >&2; exit 1; }
      DB_PATH="$1"
      ;;
    --yes)
      ASSUME_YES=1
      ;;
    --no-backup)
      MAKE_BACKUP=0
      ;;
    --keep-cache)
      KEEP_CACHE=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
  shift
done

if [[ "$DB_PATH" != /* ]]; then
  DB_PATH="$BASE/$DB_PATH"
fi

if [[ ! -f "$DB_PATH" ]]; then
  echo "Database not found: $DB_PATH" >&2
  exit 1
fi

if ! command -v sqlite3 >/dev/null 2>&1; then
  echo "sqlite3 not found in PATH." >&2
  exit 1
fi

table_exists() {
  local table="$1"
  sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='$table';"
}

row_count() {
  local table="$1"
  if [[ "$(table_exists "$table")" -eq 1 ]]; then
    sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM $table;"
  else
    echo "0"
  fi
}

TARGET_TABLES=("paper_trades" "live_observations" "fear_positions")
if [[ "$KEEP_CACHE" -eq 0 ]]; then
  TARGET_TABLES+=("odds_api_cache")
fi

echo "Database: $DB_PATH"
for t in "${TARGET_TABLES[@]}"; do
  echo "Before $t: $(row_count "$t") rows"
done

if [[ "$ASSUME_YES" -ne 1 ]]; then
  read -r -p "Reset these tables now? [y/N] " ANSWER
  case "${ANSWER:-}" in
    y|Y|yes|YES)
      ;;
    *)
      echo "Cancelled."
      exit 0
      ;;
  esac
fi

if [[ "$MAKE_BACKUP" -eq 1 ]]; then
  BACKUP_DIR="$(dirname "$DB_PATH")"
  BACKUP_PATH="$BACKUP_DIR/arb_backup_$(date +%Y%m%d_%H%M%S).db"
  cp "$DB_PATH" "$BACKUP_PATH"
  echo "Backup created: $BACKUP_PATH"
fi

SQL="BEGIN;"
DELETED_TABLES=()
for t in "${TARGET_TABLES[@]}"; do
  if [[ "$(table_exists "$t")" -eq 1 ]]; then
    SQL+="DELETE FROM $t;"
    DELETED_TABLES+=("$t")
  fi
done

if [[ "$(table_exists "sqlite_sequence")" -eq 1 && "${#DELETED_TABLES[@]}" -gt 0 ]]; then
  NAMES=""
  for t in "${DELETED_TABLES[@]}"; do
    if [[ -n "$NAMES" ]]; then
      NAMES+=","
    fi
    NAMES+="'$t'"
  done
  SQL+="DELETE FROM sqlite_sequence WHERE name IN ($NAMES);"
fi
SQL+="COMMIT;"

sqlite3 "$DB_PATH" "$SQL"
sqlite3 "$DB_PATH" "VACUUM;"

for t in "${TARGET_TABLES[@]}"; do
  echo "After  $t: $(row_count "$t") rows"
done

echo "Reset complete."
