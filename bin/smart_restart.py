#!/usr/bin/env python3
"""Smart deploy restart: only restart supervisor daemons affected by code changes.

Usage: python3 bin/smart_restart.py <old_head_sha>

Reads supervisor configs from /etc/supervisor/conf.d/worker-*.conf to discover
daemons, then maps git-changed files to affected daemons and restarts only those.
"""

import glob
import subprocess
import sys

SITE_DIR = "/home/ploi/poly.lvlup-dev.com"

# ---------------------------------------------------------------------------
# Dependency map: file-path prefix → affected daemon keys
#
# A daemon "key" is a substring matched against the daemon's command in its
# supervisor config (e.g. "run_fear_selling" matches a command containing
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
                                     "run_crypto_maker",
                                     "run_crypto_two_sided_maker",
                                     "run_sniper"]),
    ("src/feeds/polymarket",        ["run_crypto_td_maker",
                                     "run_crypto_maker", "run_crypto_two_sided_maker",
                                     "run_sniper", "run_slot_collector",
                                     "run_slot_collector_5m"]),
    ("src/feeds/chainlink",         ["run_crypto_td_maker", "run_slot_collector",
                                     "run_slot_collector_5m"]),
    ("src/feeds/kalshi",            ["run_kalshi_td_maker"]),
    ("src/db/td_orders",            ["run_crypto_td_maker", "run_kalshi_td_maker",
                                     "run_crypto_two_sided_maker"]),
    ("src/db/slot_",                ["run_slot_collector", "run_slot_collector_5m"]),
    ("src/db/",                     ["run_fear_selling", "run_trades_api",
                                     "run_dashboard", "streamlit"]),

    # ── Strategy-specific modules ───────────────────────────────────────
    ("src/arb/fear_",               ["run_fear_selling"]),
    ("src/arb/weather_oracle",      ["run_weather_oracle"]),
    ("src/arb/crypto_minute",       ["run_crypto_minute", "run_crypto_maker"]),
    ("src/arb/crypto_td",           ["run_crypto_td_maker"]),
    ("src/arb/sniper_engine",       ["run_sniper"]),
    ("src/feeds/polymarket_scanner", ["run_sniper"]),

    # ── API / dashboard ─────────────────────────────────────────────────
    ("src/api/trades_api",          ["run_trades_api"]),
    ("src/api/slots_api",           ["run_trades_api"]),
    ("src/api/winrate",             ["run_trades_api"]),
    ("src/api/rn1_compare",         ["run_rn1_compare_api"]),
    ("src/dashboard/",              ["run_dashboard", "streamlit"]),
    ("src/paper_trading/",          ["run_auto_redeem"]),
    ("src/ml/",                     ["run_dashboard", "streamlit"]),

    # ── Scripts (entry points) ──────────────────────────────────────────
    ("scripts/run_fear_selling.py",      ["run_fear_selling"]),
    ("scripts/run_weather_oracle.py",    ["run_weather_oracle"]),
    ("scripts/run_crypto_minute.py",     ["run_crypto_minute", "run_crypto_maker"]),
    ("scripts/run_crypto_td_maker.py",   ["run_crypto_td_maker"]),
    ("scripts/run_auto_redeem.py",       ["run_auto_redeem"]),
    ("scripts/run_kalshi_td_maker.py",   ["run_kalshi_td_maker"]),
    ("scripts/run_crypto_two_sided_maker.py", ["run_crypto_two_sided_maker"]),
    ("scripts/run_sniper.py",           ["run_sniper"]),
    ("scripts/run_slot_collector.py",    ["run_slot_collector", "run_slot_collector_5m"]),

    # ── Shell wrappers ──────────────────────────────────────────────────
    ("bin/run_fear_selling.sh",      ["run_fear_selling"]),
    ("bin/run_weather_oracle.sh",    ["run_weather_oracle"]),
    ("bin/run_crypto_minute.sh",     ["run_crypto_minute"]),
    ("bin/run_crypto_maker.sh",      ["run_crypto_maker"]),
    ("bin/run_crypto_td_maker.sh",   ["run_crypto_td_maker"]),
    ("bin/run_crypto_two_sided_maker.sh", ["run_crypto_two_sided_maker"]),
    ("bin/run_trades_api.sh",        ["run_trades_api"]),
    ("bin/run_dashboard.sh",         ["run_dashboard", "streamlit"]),
    ("bin/run_rn1_compare_api.sh",   ["run_rn1_compare_api"]),
    ("bin/run_sniper.sh",           ["run_sniper"]),
    ("bin/run_slot_collector.sh",    ["run_slot_collector"]),
    ("bin/run_slot_collector_5m.sh", ["run_slot_collector_5m"]),
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


def discover_daemons() -> list[dict]:
    """Read supervisor configs to find our site's daemons."""
    daemons = []
    for conf_path in sorted(glob.glob("/etc/supervisor/conf.d/worker-*.conf")):
        name = conf_path.rsplit("/", 1)[-1].replace(".conf", "")
        command = ""
        directory = ""
        with open(conf_path) as f:
            for line in f:
                if line.strip().startswith("command="):
                    command = line.strip().split("=", 1)[1]
                if line.strip().startswith("directory="):
                    directory = line.strip().split("=", 1)[1].rstrip("/")
        # Include daemons that belong to our site (match command or directory)
        if SITE_DIR in command or directory == SITE_DIR:
            daemons.append({"name": name, "command": command})
    return daemons


def match_daemon(daemon: dict, keys: set[str]) -> bool:
    """Check if a daemon's command matches any of the affected keys."""
    cmd = daemon["command"]
    return any(key in cmd for key in keys)


def restart(name: str) -> bool:
    """Restart a supervisor daemon group. Returns True on success."""
    result = subprocess.run(
        ["sudo", "supervisorctl", "restart", f"{name}:*"],
        capture_output=True, text=True,
    )
    return result.returncode == 0


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python3 bin/smart_restart.py <old_head_sha>")
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
        print("[smart_restart] Shared code changed -> restarting ALL daemons.")
    else:
        print(f"[smart_restart] Affected keys: {', '.join(sorted(keys))}")

    # 3. Discover daemons from supervisor configs
    daemons = discover_daemons()
    print(f"[smart_restart] Found {len(daemons)} daemon(s) for {SITE_DIR}:")
    for d in daemons:
        print(f"  {d['name']}: {d['command'][:80]}")

    # 4. Restart matching daemons
    restarted = 0
    for d in daemons:
        if restart_all or match_daemon(d, keys):
            ok = restart(d["name"])
            status = "ok" if ok else "FAILED"
            print(f"  -> restart {d['name']}: {status}")
            if ok:
                restarted += 1

    print(f"[smart_restart] Done. Restarted {restarted}/{len(daemons)} daemon(s).")


if __name__ == "__main__":
    main()
