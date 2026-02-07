#!/usr/bin/env python3
"""
Data refresh runner. Add scripts to STEPS below.

Usage:
    python ref.py              # run all steps
    python ref.py --skip-bbg   # skip Bloomberg steps
    python ref.py --only ust   # run single step
"""

import subprocess
import sys
import os
import re
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
LOG_DIR = SCRIPT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG: Add new scripts here
# ─────────────────────────────────────────────────────────────────────────────
# Format: (key, script, description, requires_bbg)

STEPS = [
    ("cusips",    "pull_ust_cusips_postgres.py", "sec.auctioned_securities",  False),
    ("ust",       "pull_daily_postgres.py",      "md.ust_eod",                True),
    ("fut",       "pull_fut_eod.py",             "md.fut_eod",                True),
    ("dlv",       "pull_dlv_basket.py",          "deliverable baskets",       True),
    ("headline",  "build_headline.py",           "md.headline",               False),
    ("breakeven", "build_breakeven.py",          "md.breakeven",              False),
]

# ─────────────────────────────────────────────────────────────────────────────


def run(script: str) -> tuple[bool, str, float]:
    """Run script, return (success, output, duration_sec)."""
    start = datetime.now()
    result = subprocess.run(
        [sys.executable, script],
        cwd=SCRIPT_DIR,
        capture_output=True,
        text=True,
    )
    duration = (datetime.now() - start).total_seconds()
    output = result.stdout + result.stderr
    return result.returncode == 0, output, duration


def parse_rows(output: str) -> int | None:
    """Extract row count from output."""
    for pattern in [r"(\d[\d,]*)\s+rows?", r"Success:\s*(\d+)"]:
        m = re.search(pattern, output, re.IGNORECASE)
        if m:
            return int(m.group(1).replace(",", ""))
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-bbg", action="store_true")
    parser.add_argument("--only", type=str)
    args = parser.parse_args()

    ts = datetime.now()
    log_path = LOG_DIR / f"ref_{ts:%Y%m%d_%H%M%S}.log"
    log = open(log_path, "w")

    def out(msg):
        print(msg)
        log.write(msg + "\n")

    out(f"\n{'═'*60}")
    out(f"  REF - {ts:%Y-%m-%d %H:%M:%S}")
    out(f"{'═'*60}\n")

    results = []
    for key, script, desc, needs_bbg in STEPS:
        if args.only and args.only != key:
            continue
        if args.skip_bbg and needs_bbg:
            out(f"⏭️  {desc:<30} (skipped)")
            continue

        out(f"▶  {desc:<30} ", )
        ok, output, dur = run(script)
        rows = parse_rows(output)

        status = "✅" if ok else "❌"
        rows_str = f"{rows:,} rows" if rows else ("up to date" if "up to date" in output.lower() else "")
        out(f"   {status} {dur:>5.1f}s  {rows_str}")

        log.write(f"\n--- {script} ---\n{output}\n")
        results.append((key, ok, dur, rows))

    # Summary
    out(f"\n{'─'*60}")
    total_time = sum(r[2] for r in results)
    total_rows = sum(r[3] or 0 for r in results)
    failed = sum(1 for r in results if not r[1])

    out(f"  ⏱️  {total_time:.1f}s total | 📊 {total_rows:,} rows | {'❌ ' + str(failed) + ' failed' if failed else '✅ all ok'}")
    out(f"  📄 {log_path}\n")

    log.close()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
