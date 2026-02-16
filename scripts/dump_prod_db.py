#!/usr/bin/env python3
"""Dump production MySQL database and import into local MySQL.

Usage:
    ./run scripts/dump_prod_db.py                                        # all tables
    ./run scripts/dump_prod_db.py --tables slot_snapshots slot_resolutions  # specific tables
    ./run scripts/dump_prod_db.py --dry-run                              # show commands only
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

from config.settings import settings


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DUMP_PATH = PROJECT_ROOT / "data" / "prod_dump.sql"

_TABLE_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
_PASSWORD_RE = re.compile(r"-p'[^']*'")
_MYSQL_PWD_RE = re.compile(r"MYSQL_PWD='[^']*'")


def _redact(cmd_str: str) -> str:
    """Replace password values in a command string for safe display."""
    s = _PASSWORD_RE.sub("-p'***'", cmd_str)
    s = _MYSQL_PWD_RE.sub("MYSQL_PWD='***'", s)
    return s


def parse_mysql_url(url: str) -> dict[str, str]:
    """Parse a SQLAlchemy MySQL URL into host/user/pass/db components.

    Handles URLs like:
        mysql+aiomysql://user:pass@host/dbname
        mysql+aiomysql://user:pass@host:3306/dbname
    """
    # Strip the dialect prefix so urlparse handles it correctly
    # e.g. "mysql+aiomysql://..." -> "mysql://..."
    if "+aiomysql" in url:
        url = url.replace("+aiomysql", "")
    elif "+pymysql" in url:
        url = url.replace("+pymysql", "")

    parsed = urlparse(url)

    if parsed.scheme != "mysql":
        print(f"ERROR: DATABASE_URL is not MySQL (scheme={parsed.scheme})",
              file=sys.stderr)
        sys.exit(1)

    return {
        "host": parsed.hostname or "localhost",
        "port": str(parsed.port) if parsed.port else "3306",
        "user": parsed.username or "",
        "password": parsed.password or "",
        "database": parsed.path.lstrip("/"),
    }


def validate_settings() -> None:
    """Check that all required PROD_* settings are configured."""
    missing = []
    if not settings.PROD_SSH_HOST:
        missing.append("PROD_SSH_HOST")
    if not settings.PROD_DB_NAME:
        missing.append("PROD_DB_NAME")
    if not settings.PROD_DB_USER:
        missing.append("PROD_DB_USER")
    if not settings.PROD_DB_PASS:
        missing.append("PROD_DB_PASS")
    if missing:
        print(f"ERROR: Missing env vars: {', '.join(missing)}", file=sys.stderr)
        print("Set them in .env — see .env.example for template.", file=sys.stderr)
        sys.exit(1)


def build_dump_cmd(tables: list[str] | None) -> list[str]:
    """Build the SSH + mysqldump command."""
    # Use MYSQL_PWD env var on the remote side to hide password from `ps`
    mysqldump = (
        f"MYSQL_PWD='{settings.PROD_DB_PASS}'"
        f" mysqldump --single-transaction --skip-lock-tables"
        f" -h {settings.PROD_DB_HOST}"
        f" -u {settings.PROD_DB_USER}"
        f" {settings.PROD_DB_NAME}"
    )
    if tables:
        for t in tables:
            if not _TABLE_NAME_RE.match(t):
                print(f"ERROR: Invalid table name: {t!r}", file=sys.stderr)
                sys.exit(1)
        mysqldump += " " + " ".join(tables)

    return [
        "ssh", settings.PROD_SSH_HOST,
        mysqldump,
    ]


def build_import_cmd(local: dict[str, str]) -> str:
    """Build the mysql import command (run via shell for < redirect)."""
    parts = ["mysql", f"-h {local['host']}", f"-P {local['port']}",
             f"-u {local['user']}"]
    if local["password"]:
        parts.append(f"-p'{local['password']}'")
    parts.append(local["database"])
    return " ".join(parts) + f" < {DUMP_PATH}"


def run_cmd(cmd: list[str] | str, description: str, *,
            dry_run: bool = False, shell: bool = False,
            stdout_file: str | None = None) -> None:
    """Run a command, printing progress and handling errors."""
    cmd_str = cmd if isinstance(cmd, str) else " ".join(cmd)
    print(f"\n>> {description}")
    print(f"   {_redact(cmd_str)}")

    if dry_run:
        print("   [dry-run] skipped")
        return

    try:
        if stdout_file:
            with open(stdout_file, "w") as f:
                result = subprocess.run(
                    cmd, stdout=f, stderr=subprocess.PIPE,
                    shell=shell, text=True,
                )
        else:
            result = subprocess.run(
                cmd, capture_output=True, shell=shell, text=True,
            )

        if result.returncode != 0:
            print(f"   FAILED (exit code {result.returncode})", file=sys.stderr)
            if result.stderr:
                for line in result.stderr.strip().splitlines()[:10]:
                    print(f"   stderr: {line}", file=sys.stderr)
            sys.exit(1)

        print("   OK")

    except FileNotFoundError as e:
        print(f"   FAILED: command not found — {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"   FAILED: {e}", file=sys.stderr)
        sys.exit(1)


def print_row_counts(local: dict[str, str], *, dry_run: bool = False) -> None:
    """Print row counts for all tables in the local database."""
    print("\n>> Verifying row counts...")

    if dry_run:
        print("   [dry-run] skipped")
        return

    query = (
        "SELECT table_name, table_rows "
        "FROM information_schema.tables "
        f"WHERE table_schema = '{local['database']}' "
        "ORDER BY table_name"
    )

    parts = ["mysql", f"-h {local['host']}", f"-P {local['port']}",
             f"-u {local['user']}"]
    if local["password"]:
        parts.append(f"-p'{local['password']}'")
    parts += ["-e", f'"{query}"']
    cmd_str = " ".join(parts)

    try:
        result = subprocess.run(
            cmd_str, capture_output=True, shell=True, text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            print()
            for line in result.stdout.strip().splitlines():
                print(f"   {line}")
        else:
            print("   (could not read row counts)")
    except Exception:
        print("   (could not read row counts)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dump production MySQL DB and import locally"
    )
    parser.add_argument("--tables", nargs="+", default=None,
                        help="Specific tables to dump (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show commands without executing")
    args = parser.parse_args()

    # --- Validate ---------------------------------------------------------
    validate_settings()
    local = parse_mysql_url(settings.DATABASE_URL)

    # Safety: refuse to import into the production database
    if (local["host"] == settings.PROD_DB_HOST
            and local["database"] == settings.PROD_DB_NAME):
        print("ERROR: DATABASE_URL points to the production database — aborting.",
              file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("  Production DB Dump")
    print("=" * 60)
    print(f"  Source:  {settings.PROD_SSH_HOST} -> "
          f"{settings.PROD_DB_HOST}/{settings.PROD_DB_NAME}")
    print(f"  Target:  {local['host']}:{local['port']}/{local['database']}")
    print(f"  Tables:  {', '.join(args.tables) if args.tables else 'ALL'}")
    print(f"  Dump:    {DUMP_PATH}")
    if args.dry_run:
        print("  Mode:    DRY RUN")

    # --- Ensure data/ dir exists ------------------------------------------
    DUMP_PATH.parent.mkdir(parents=True, exist_ok=True)

    # --- Step 1: mysqldump via SSH ----------------------------------------
    dump_cmd = build_dump_cmd(args.tables)
    run_cmd(dump_cmd, "Dumping production database via SSH...",
            dry_run=args.dry_run, stdout_file=str(DUMP_PATH))

    # --- Check dump file --------------------------------------------------
    if not args.dry_run:
        size = os.path.getsize(DUMP_PATH)
        print(f"   Dump size: {size / 1024 / 1024:.1f} MB")
        if size == 0:
            print("ERROR: Dump file is empty — mysqldump likely failed.",
                  file=sys.stderr)
            sys.exit(1)

    # --- Step 2: Import into local MySQL ----------------------------------
    import_cmd = build_import_cmd(local)
    run_cmd(import_cmd, "Importing into local MySQL...",
            dry_run=args.dry_run, shell=True)

    # --- Step 3: Verify ---------------------------------------------------
    print_row_counts(local, dry_run=args.dry_run)

    print()
    print("=" * 60)
    print(f"  Done. Dump file kept at: {DUMP_PATH}")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
