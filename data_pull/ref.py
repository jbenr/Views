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

import psycopg
from zoneinfo import ZoneInfo

DB_DSN = os.getenv("DB_DSN", "postgresql://benjils:snickers@raptor:5432/markets")
ET     = ZoneInfo("America/New_York")

# enable ANSI on Windows
if os.name == "nt":
    os.system("")

_ANSI = re.compile(r"\033\[[0-9;]*m")

class C:
    RESET = "\033[0m"
    BOLD  = "\033[1m"
    DIM   = "\033[2m"
    RED   = "\033[91m"
    GREEN = "\033[92m"
    CYAN  = "\033[96m"

def c(text, *codes) -> str:
    return "".join(codes) + str(text) + C.RESET


def _fmt_et(d: datetime) -> str:
    """Apr 30, 2026 9:43:22pm ET"""
    d = d.astimezone(ET)
    h = int(d.strftime('%I'))
    return f"{d.strftime('%b')} {d.day}, {d.year} {h}:{d.strftime('%M:%S')}{d.strftime('%p').lower()} ET"

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
    ("swpn_vol",  "pull_swaption_vol.py",        "md.swaption_vol",           True),
    ("headline",  "build_headline.py",           "md.headline",               False),
    ("breakeven", "build_breakeven.py",          "md.breakeven",              False),
]

# ─────────────────────────────────────────────────────────────────────────────


def run(script: str, log_file, extra: list[str] | None = None) -> tuple[bool, str, float]:
    """Run script, streaming output live to terminal and capturing it for the log."""
    start = datetime.now()
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    cmd = [sys.executable, "-u", script] + (extra or [])

    proc = subprocess.Popen(
        cmd,
        cwd=SCRIPT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )

    captured = {"stdout": [], "stderr": []}

    def stream(pipe, name, prefix=""):
        """Read pipe line-by-line, print live and capture. Suppress Python tracebacks."""
        in_traceback = False
        for raw in iter(pipe.readline, b""):
            line = raw.decode("utf-8", errors="replace")
            stripped = line.strip()
            suppress = False

            if stripped == "Traceback (most recent call last):":
                in_traceback = True
                suppress = True
            elif in_traceback:
                suppress = True
                if stripped and not stripped.startswith("File ") and not line.startswith(" "):
                    in_traceback = False  # exception type line — traceback done

            if not suppress:
                if line.startswith("\r") or "\r" in line:
                    sys.stderr.write(f"   {line}")
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


def _settle_live_rows():
    """After EOD BDH run, clear is_live flag for today's rows."""
    try:
        with psycopg.connect(DB_DSN) as conn:
            with conn.cursor() as cur:
                for tbl in ['md.fut_eod', 'md.ust_eod', 'md.index_eod', 'md.strips_eod', 'md.swaption_vol']:
                    cur.execute(f"UPDATE {tbl} SET is_live = FALSE WHERE ts = CURRENT_DATE AND is_live = TRUE")
            conn.commit()
    except Exception:
        pass


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-bbg", action="store_true")
    parser.add_argument("--only", type=str)
    parser.add_argument("--live",  action="store_true", help="L1 live: fut + index (BDP)")
    parser.add_argument("--live2", action="store_true", help="L2 live: + ust + strips (BDP)")
    args = parser.parse_args()

    ts = datetime.now()
    log_path = LOG_DIR / f"ref_{ts:%Y%m%d_%H%M%S}.log"
    log = open(log_path, "w", encoding="utf-8")

    def out(msg):
        print(msg)
        log.write(_ANSI.sub("", msg) + "\n")

    # ── live mode ──────────────────────────────────────────────────────────────
    if args.live or args.live2:
        level = 2 if args.live2 else 1
        label = f"REF --live{'2' if args.live2 else ''}"
        out(f"\n{c('='*60, C.DIM)}")
        out(f"  {c(label + ' - ' + _fmt_et(ts), C.BOLD)}")
        out(f"{c('='*60, C.DIM)}\n")
        out(f"[1/1]  live level {level}")
        ok, output, dur = run("pull_live.py", log, extra=["--level", str(level), "--write"])
        rows = parse_rows(output)
        rows_str = c(f"{rows:,} rows", C.DIM) if rows else ""
        tick = c(">", C.GREEN) if ok else c("x", C.RED)
        stat = c("OK", C.GREEN) if ok else c("FAIL", C.RED)
        out(f"       {tick} {stat} {dur:>5.1f}s  {rows_str}")
        if not ok:
            err = extract_error(output, max_lines=1)
            for line in err.splitlines():
                out(c(f"       | {line}", C.RED, C.DIM))
        out(f"\n{c('_'*60, C.DIM)}")
        summary = c("passed", C.GREEN) if ok else c("failed", C.RED)
        out(f"\n  {summary}  {c(f'{dur:.1f}s total', C.DIM)}")
        out(c(f"  log: {log_path}", C.DIM))
        log.close()
        sys.exit(0 if ok else 1)

    # ── EOD mode ───────────────────────────────────────────────────────────────
    # figure out which steps we're running
    active_steps = []
    for key, script, desc, needs_bbg in STEPS:
        if args.only and args.only != key:
            continue
        if args.skip_bbg and needs_bbg:
            continue
        active_steps.append((key, script, desc, needs_bbg))

    out(f"\n{c('='*60, C.DIM)}")
    out(f"  {c('REF - ' + _fmt_et(ts), C.BOLD)}")
    out(f"{c('='*60, C.DIM)}\n")

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

        tick     = c(">", C.GREEN) if ok else c("x", C.RED)
        stat     = c("OK", C.GREEN) if ok else c("FAIL", C.RED)
        rows_str = c(f"{rows:,} rows", C.DIM) if rows else (c("up to date", C.DIM) if "up to date" in output.lower() else "")
        out(f"       {tick} {stat} {dur:>5.1f}s  {rows_str}")

        if not ok:
            err = extract_error(output, max_lines=1)
            for line in err.splitlines():
                out(c(f"       | {line}", C.RED, C.DIM))
            out("")

        results.append((key, ok, dur, rows))

    # settle any live rows written earlier today
    if not args.only:
        _settle_live_rows()

    # skipped steps
    if skipped:
        out(c(f"\nSkipped ({len(skipped)}): {', '.join(skipped)}", C.DIM))

    # Summary
    out(f"\n{c('_'*60, C.DIM)}")
    total_time = sum(r[2] for r in results)
    total_rows = sum(r[3] or 0 for r in results)
    failed = [r[0] for r in results if not r[1]]
    passed = sum(1 for r in results if r[1])

    if failed:
        out(f"\n  {c(str(passed) + ' passed', C.GREEN)}, {c(str(len(failed)) + ' failed: ' + ', '.join(failed), C.RED)}")
    else:
        out(f"\n  {c('All ' + str(passed) + ' steps passed', C.GREEN)}")
    out(c(f"  {total_time:.1f}s total | {total_rows:,} rows", C.DIM))
    out(c(f"  log: {log_path}", C.DIM))

    log.close()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
