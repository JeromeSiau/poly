#!/usr/bin/env python3
"""Smart deploy restart: only restart Ploi daemons affected by code changes.

Usage: python bin/smart_restart.py <old_head_sha>

Reads PLOI_API_TOKEN and PLOI_SERVER_ID from config/settings.py (.env).
"""

import json
import subprocess
import sys
import urllib.request

sys.path.insert(0, ".")
from config.settings import settings

PLOI_API = "https://ploi.io/api"

# ---------------------------------------------------------------------------
# Dependency map: file-path prefix  →  affected daemon keys
#
# A daemon "key" is a substring matched against the daemon command reported by
# the Ploi API (e.g. "run_fear_selling" matches a command containing
# "run_fear_selling.sh" or "run_fear_selling.py").
#
# The special key "ALL" means every daemon must be restarted.
# ---------------------------------------------------------------------------

RULES: list[tuple[str, list[str] | str]] = [
    # ── Shared infrastructure → restart everything ──────────────────────
    ("src/execution/",      "ALL"),
    ("src/risk/",           "ALL"),
    ("src/utils/",          "ALL"),
    ("config/",             "ALL"),
    ("requirements.txt",    "ALL"),

    # ── Shared components → subset of daemons ───────────────────────────
    ("src/arb/polymarket_executor", ["run_fear_selling", "run_crypto_td_maker",
                                     "run_crypto_two_sided", "run_crypto_maker"]),
    ("src/feeds/polymarket",        ["run_crypto_td_maker", "run_crypto_two_sided",
                                     "run_crypto_maker"]),
    ("src/feeds/kalshi",            ["run_kalshi_td_maker"]),
    ("src/db/td_orders",            ["run_crypto_td_maker", "run_kalshi_td_maker"]),
    ("src/db/",                     ["run_fear_selling", "run_trades_api",
                                     "run_dashboard", "streamlit"]),

    # ── Strategy-specific modules ───────────────────────────────────────
    ("src/arb/fear_",               ["run_fear_selling"]),
    ("src/arb/weather_oracle",      ["run_weather_oracle"]),
    ("src/arb/crypto_minute",       ["run_crypto_minute", "run_crypto_maker"]),
    ("src/arb/crypto_two_sided",    ["run_crypto_two_sided"]),
    ("src/arb/crypto_td",           ["run_crypto_td_maker"]),

    # ── API / dashboard ─────────────────────────────────────────────────
    ("src/api/trades_api",          ["run_trades_api"]),
    ("src/api/winrate",             ["run_trades_api"]),
    ("src/api/rn1_compare",         ["run_rn1_compare_api"]),
    ("src/paper_trading/",          ["run_dashboard", "streamlit", "run_auto_redeem"]),
    ("src/ml/",                     ["run_dashboard", "streamlit"]),

    # ── Scripts (entry points) ──────────────────────────────────────────
    ("scripts/run_fear_selling.py",      ["run_fear_selling"]),
    ("scripts/run_weather_oracle.py",    ["run_weather_oracle"]),
    ("scripts/run_crypto_minute.py",     ["run_crypto_minute", "run_crypto_maker"]),
    ("scripts/run_crypto_td_maker.py",   ["run_crypto_td_maker"]),
    ("scripts/run_crypto_two_sided.py",  ["run_crypto_two_sided"]),
    ("scripts/run_auto_redeem.py",       ["run_auto_redeem"]),
    ("scripts/run_kalshi_td_maker.py",   ["run_kalshi_td_maker"]),

    # ── Shell wrappers ──────────────────────────────────────────────────
    ("bin/run_fear_selling.sh",      ["run_fear_selling"]),
    ("bin/run_weather_oracle.sh",    ["run_weather_oracle"]),
    ("bin/run_crypto_minute.sh",     ["run_crypto_minute"]),
    ("bin/run_crypto_maker.sh",      ["run_crypto_maker"]),
    ("bin/run_crypto_td_maker.sh",   ["run_crypto_td_maker"]),
    ("bin/run_crypto_two_sided.sh",  ["run_crypto_two_sided"]),
    ("bin/run_trades_api.sh",        ["run_trades_api"]),
    ("bin/run_dashboard.sh",         ["run_dashboard", "streamlit"]),
    ("bin/run_rn1_compare_api.sh",   ["run_rn1_compare_api"]),
]


def get_changed_files(old_head: str) -> list[str]:
    """Return files changed between old_head and current HEAD."""
    result = subprocess.run(
        ["git", "diff", "--name-only", old_head, "HEAD"],
        capture_output=True, text=True, check=True,
    )
    return [f for f in result.stdout.strip().splitlines() if f]


def resolve_affected_keys(changed_files: list[str]) -> set[str]:
    """Map changed files to daemon keys that need a restart."""
    keys: set[str] = set()
    for path in changed_files:
        for pattern, targets in RULES:
            if path.startswith(pattern):
                if targets == "ALL":
                    return {"ALL"}
                keys.update(targets)
    return keys


def ploi_request(method: str, endpoint: str, body: dict | None = None) -> dict:
    """Make an authenticated request to the Ploi API."""
    url = f"{PLOI_API}{endpoint}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {settings.PLOI_API_TOKEN}")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def list_daemons() -> list[dict]:
    """Fetch all daemons from Ploi, handling pagination."""
    daemons = []
    page = 1
    while True:
        resp = ploi_request("GET", f"/servers/{settings.PLOI_SERVER_ID}/daemons?page={page}")
        daemons.extend(resp.get("data", []))
        meta = resp.get("meta", {})
        if page >= meta.get("last_page", 1):
            break
        page += 1
    return daemons


def restart_daemon(daemon_id: int) -> None:
    """Restart a single daemon via Ploi API."""
    ploi_request("POST", f"/servers/{settings.PLOI_SERVER_ID}/daemons/{daemon_id}/restart")


def match_daemon(daemon: dict, keys: set[str]) -> bool:
    """Check if a daemon's command matches any of the affected keys."""
    cmd = daemon.get("command", "")
    return any(key in cmd for key in keys)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python bin/smart_restart.py <old_head_sha>")
        sys.exit(1)

    old_head = sys.argv[1]

    # 1. What changed?
    changed = get_changed_files(old_head)
    if not changed:
        print("[smart_restart] No files changed, nothing to restart.")
        return

    print(f"[smart_restart] {len(changed)} file(s) changed:")
    for f in changed[:20]:
        print(f"  {f}")
    if len(changed) > 20:
        print(f"  ... and {len(changed) - 20} more")

    # 2. Which daemon keys are affected?
    keys = resolve_affected_keys(changed)
    if not keys:
        print("[smart_restart] No daemon-relevant files changed, skipping restart.")
        return

    restart_all = "ALL" in keys
    if restart_all:
        print("[smart_restart] Shared code changed → restarting ALL daemons.")
    else:
        print(f"[smart_restart] Affected daemons: {', '.join(sorted(keys))}")

    # 3. Validate Ploi credentials
    if not settings.PLOI_API_TOKEN or not settings.PLOI_SERVER_ID:
        print("[smart_restart] ERROR: PLOI_API_TOKEN and PLOI_SERVER_ID must be set in .env")
        sys.exit(1)

    # 4. Fetch daemons from Ploi
    daemons = list_daemons()
    print(f"[smart_restart] Found {len(daemons)} daemon(s) on Ploi.")

    # 5. Restart matching daemons
    restarted = 0
    for d in daemons:
        if d.get("status") != "active":
            continue
        if restart_all or match_daemon(d, keys):
            name = d.get("command", "?")[:80]
            print(f"  ↻ Restarting daemon {d['id']}: {name}")
            try:
                restart_daemon(d["id"])
                restarted += 1
            except Exception as e:
                print(f"  ✗ Failed to restart daemon {d['id']}: {e}")

    print(f"[smart_restart] Done. Restarted {restarted}/{len(daemons)} daemon(s).")


if __name__ == "__main__":
    main()
