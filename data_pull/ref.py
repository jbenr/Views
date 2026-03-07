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
import threading
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
    ("strips",    "pull_strips.py",              "md.strips_eod",             True),
    ("index",     "pull_index_eod.py",           "md.index_eod",              True),
    ("cftc",      "pull_cftc.py",                "md.cftc",                   False),
    ("headline",  "build_headline.py",           "md.headline",               False),
    ("breakeven", "build_breakeven.py",          "md.breakeven",              False),
]

# ─────────────────────────────────────────────────────────────────────────────


def run(script: str, log_file) -> tuple[bool, str, float]:
    """Run script, streaming output live to terminal and capturing it for the log."""
    start = datetime.now()
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}

    proc = subprocess.Popen(
        [sys.executable, "-u", script],
        cwd=SCRIPT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )

    captured = {"stdout": [], "stderr": []}

    def stream(pipe, name, prefix=""):
        """Read pipe line-by-line, print live and capture."""
        for raw in iter(pipe.readline, b""):
            line = raw.decode("utf-8", errors="replace")
            # Print live — use \r-aware write for tqdm compatibility
            if line.startswith("\r") or "\r" in line:
                sys.stderr.write(f"   {line}")
                sys.stderr.flush()
            else:
                sys.stderr.write(f"   {prefix}{line}")
                sys.stderr.flush()
            captured[name].append(line)
        pipe.close()

    t_out = threading.Thread(target=stream, args=(proc.stdout, "stdout"), daemon=True)
    t_err = threading.Thread(target=stream, args=(proc.stderr, "stderr", ""), daemon=True)
    t_out.start()
    t_err.start()

    proc.wait()
    t_out.join(timeout=5)
    t_err.join(timeout=5)

    duration = (datetime.now() - start).total_seconds()
    full_output = "".join(captured["stdout"]) + "".join(captured["stderr"])

    # write full output to log
    log_file.write(f"\n--- {script} ---\n{full_output}\n")

    return proc.returncode == 0, full_output, duration


def extract_error(output: str, max_lines: int = 4) -> str:
    """Pull the most useful error context from captured output."""
    lines = [l.rstrip() for l in output.strip().splitlines() if l.strip()]
    if not lines:
        return "no output"

    # look for common Python error patterns from the bottom
    for i in range(len(lines) - 1, -1, -1):
        if any(lines[i].startswith(p) for p in ("Traceback", "Error:", "Exception")):
            # return from the error header to the end
            return "\n".join(lines[i:][-max_lines:])
        if "Error" in lines[i] or "Exception" in lines[i]:
            return "\n".join(lines[max(0, i - 1):][-max_lines:])

    # fallback: last N lines
    return "\n".join(lines[-max_lines:])


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

    # figure out which steps we're running
    active_steps = []
    for key, script, desc, needs_bbg in STEPS:
        if args.only and args.only != key:
            continue
        if args.skip_bbg and needs_bbg:
            continue
        active_steps.append((key, script, desc, needs_bbg))

    ts = datetime.now()
    log_path = LOG_DIR / f"ref_{ts:%Y%m%d_%H%M%S}.log"
    log = open(log_path, "w", encoding="utf-8")

    def out(msg):
        print(msg)
        log.write(msg + "\n")

    out(f"\n{'='*60}")
    out(f"  REF - {ts:%Y-%m-%d %H:%M:%S}")
    out(f"{'='*60}\n")

    total = len(active_steps)
    results = []
    skipped = []

    for key, script, desc, needs_bbg in STEPS:
        if args.only and args.only != key:
            continue
        if args.skip_bbg and needs_bbg:
            skipped.append(desc)
            continue

        step_num = len(results) + 1
        out(f"[{step_num}/{total}]  {desc}")

        ok, output, dur = run(script, log)
        rows = parse_rows(output)

        status = "OK" if ok else "FAIL"
        rows_str = f"{rows:,} rows" if rows else ("up to date" if "up to date" in output.lower() else "")
        out(f"       {'>' if ok else 'x'} {status} {dur:>5.1f}s  {rows_str}")

        if not ok:
            err = extract_error(output)
            for line in err.splitlines():
                out(f"       | {line}")
            out("")

        results.append((key, ok, dur, rows))

    # skipped steps
    if skipped:
        out(f"\nSkipped ({len(skipped)}): {', '.join(skipped)}")

    # Summary
    out(f"\n{'_'*60}")
    total_time = sum(r[2] for r in results)
    total_rows = sum(r[3] or 0 for r in results)
    failed = [r[0] for r in results if not r[1]]
    passed = sum(1 for r in results if r[1])

    if failed:
        out(f"\n  {passed} passed, {len(failed)} failed: {', '.join(failed)}")
    else:
        out(f"\n  All {passed} steps passed")
    out(f"  {total_time:.1f}s total | {total_rows:,} rows")
    out(f"  log: {log_path}\n")

    log.close()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
