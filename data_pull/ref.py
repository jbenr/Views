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
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

import psycopg
from zoneinfo import ZoneInfo

def _with_connect_timeout(dsn: str, seconds: int = 10) -> str:
    if "connect_timeout=" in dsn:
        return re.sub(r"connect_timeout=\d+", f"connect_timeout={seconds}", dsn)
    sep = "&" if "?" in dsn else "?"
    return f"{dsn}{sep}connect_timeout={seconds}"


DB_DSN = _with_connect_timeout(
    os.getenv("DB_DSN", "postgresql://benjils:snickers@raptor:5432/markets"),
    seconds=10,
)
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


def _safe_dsn_label(dsn: str) -> str:
    """Return connection target without leaking the password."""
    try:
        parsed = urlsplit(dsn)
        query = dict(parse_qsl(parsed.query))
        user = f"{parsed.username}@" if parsed.username else ""
        host = parsed.hostname or "unknown-host"
        port = f":{parsed.port}" if parsed.port else ""
        db = parsed.path.lstrip("/") or "unknown-db"
        timeout = query.get("connect_timeout")
        suffix = f" connect_timeout={timeout}s" if timeout else ""
        return f"{user}{host}{port}/{db}{suffix}"
    except Exception:
        return "<unparseable DB_DSN>"

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
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1", "DB_DSN": DB_DSN}
    cmd = [sys.executable, "-u", script] + (extra or [])

    log_file.write(f"\n--- {script} ---\n")
    log_file.write(f"started: {_fmt_et(start)}\n")
    log_file.write(f"cwd: {SCRIPT_DIR}\n")
    log_file.write(f"cmd: {' '.join(cmd)}\n")
    log_file.write(f"db: {_safe_dsn_label(DB_DSN)}\n")
    log_file.flush()

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
                    suppress = False      # ...and it is the one line worth showing
            elif stripped.startswith("Connecting to Postgres:"):
                suppress = True

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
    log_file.write(f"exit_code: {proc.returncode}\n")
    log_file.write(f"duration_s: {duration:.1f}\n")
    log_file.write("output:\n")
    log_file.write(full_output)
    log_file.write("\n")
    log_file.flush()

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


_wsl_keepalive_proc: subprocess.Popen | None = None


def start_wsl_keepalive() -> None:
    """Hold a wsl.exe process attached for the run's duration.

    WSL2 recycles its VM within roughly a minute of no wsl.exe process being
    attached to it, regardless of vmIdleTimeout, silently dropping any live
    TCP connections into it -- including the Postgres port-forward. A
    one-shot wake command isn't enough to keep it up for a multi-minute
    pull; holding a process attached for the whole run is.
    """
    global _wsl_keepalive_proc
    if os.name != "nt" or os.getenv("REF_SKIP_WSL_WAKE") == "1":
        return
    distro = os.getenv("REF_WSL_DISTRO", "Ubuntu")
    try:
        _wsl_keepalive_proc = subprocess.Popen(
            ["wsl", "-d", distro, "-u", "root", "--", "sleep", "infinity"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception:
        _wsl_keepalive_proc = None


def stop_wsl_keepalive() -> None:
    global _wsl_keepalive_proc
    if _wsl_keepalive_proc is None:
        return
    _wsl_keepalive_proc.terminate()
    try:
        _wsl_keepalive_proc.wait(timeout=5)
    except Exception:
        _wsl_keepalive_proc.kill()
    _wsl_keepalive_proc = None


def wake_wsl_postgres(report=None) -> tuple[bool, str, float]:
    """On Windows, keep WSL alive and start the local Postgres cluster if present."""
    start = datetime.now()
    if os.name != "nt":
        return True, "skipped on non-Windows", 0.0
    if os.getenv("REF_SKIP_WSL_WAKE") == "1":
        return True, "skipped by REF_SKIP_WSL_WAKE=1", 0.0

    distro = os.getenv("REF_WSL_DISTRO", "Ubuntu")
    script = (
        "pgrep -f 'sleep infinity' >/dev/null || "
        "nohup sleep infinity >/dev/null 2>&1 & "
        "service postgresql start >/dev/null 2>&1 || true; "
        "pg_isready -h 127.0.0.1 -p 5432 || true"
    )

    try:
        result = subprocess.run(
            ["wsl", "-d", distro, "-u", "root", "--", "bash", "-lc", script],
            capture_output=True,
            text=True,
            timeout=90,
        )
    except Exception as e:
        dur = (datetime.now() - start).total_seconds()
        return False, f"WSL wake failed: {str(e).splitlines()[0]}", dur

    dur = (datetime.now() - start).total_seconds()
    msg = " ".join((result.stdout + result.stderr).split())
    if result.returncode != 0:
        return False, f"WSL wake rc={result.returncode}: {msg or 'no output'}", dur
    return True, msg or f"started {distro}", dur


def wait_for_db(max_wait_s: int = 180, report=None) -> tuple[bool, str, float]:
    """Wait until Postgres accepts SQL, not just TCP, before child scripts run."""
    start = datetime.now()
    deadline = start.timestamp() + max_wait_s
    last_err = ""
    attempt = 0

    while datetime.now().timestamp() < deadline:
        attempt += 1
        try:
            with psycopg.connect(DB_DSN) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT pg_is_in_recovery(), now()")
                    in_recovery, _ = cur.fetchone()
                    if not in_recovery:
                        return True, f"ready on attempt {attempt}", (datetime.now() - start).total_seconds()
                    last_err = "database still in recovery"
        except Exception as e:
            last_err = str(e).split("\n")[0].strip()

        elapsed = (datetime.now() - start).total_seconds()
        time_left = max(0, int(deadline - datetime.now().timestamp()))
        msg = f"attempt {attempt}: {last_err[:70]} ({elapsed:.1f}s elapsed, {time_left}s left)"
        if report is not None:
            report(c(f"       | {msg}", C.DIM))
        else:
            print(f"\r  {msg}", end="", flush=True)
        time.sleep(3)

    if report is None:
        print()
    return False, last_err or "timed out waiting for Postgres", (datetime.now() - start).total_seconds()

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
    parser.add_argument("--db-wait", type=int, default=180, help="seconds to wait for Postgres readiness before running pulls")
    args = parser.parse_args()

    ts = datetime.now()
    log_path = LOG_DIR / f"ref_{ts:%Y%m%d_%H%M%S}.log"
    log = open(log_path, "w", encoding="utf-8")

    def out(msg):
        print(msg, flush=True)
        log.write(_ANSI.sub("", msg) + "\n")
        log.flush()

    # ── live mode ──────────────────────────────────────────────────────────────
    if args.live or args.live2:
        level = 2 if args.live2 else 1
        label = f"REF --live{'2' if args.live2 else ''}"
        out(f"\n{c('='*60, C.DIM)}")
        out(f"  {c(label + ' - ' + _fmt_et(ts), C.BOLD)}")
        out(f"{c('='*60, C.DIM)}\n")

        start_wsl_keepalive()
        try:
            wake_ok, wake_msg, wake_dur = wake_wsl_postgres(report=out)
            wake_tick = c(">", C.GREEN) if wake_ok else c("!", C.RED)
            wake_stat = c("OK", C.GREEN) if wake_ok else c("WARN", C.RED)
            out(f"[startup] waking WSL + Postgres")
            out(f"       {wake_tick} {wake_stat} {wake_dur:>5.1f}s  {c(wake_msg, C.DIM)}")

            db_ok, db_msg, db_dur = wait_for_db(max_wait_s=args.db_wait, report=out)
            tick = c(">", C.GREEN) if db_ok else c("x", C.RED)
            stat = c("OK", C.GREEN) if db_ok else c("FAIL", C.RED)
            out(f"[startup] waiting for Postgres to accept connections")
            out(f"       {tick} {stat} {db_dur:>5.1f}s  {c(db_msg, C.DIM)}")
            if not db_ok:
                out(c(f"       | {db_msg}", C.RED, C.DIM))
                out(c(f"  log: {log_path}", C.DIM))
                log.close()
                sys.exit(1)
            out("")

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
        finally:
            stop_wsl_keepalive()

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
    out(c(f"db: {_safe_dsn_label(DB_DSN)}", C.DIM))
    out(c(f"mode: eod | only={args.only or 'all'} | skip_bbg={args.skip_bbg} | db_wait={args.db_wait}s", C.DIM))

    if not active_steps:
        out(c(f"no steps selected; check --only={args.only!r} or --skip-bbg", C.RED))
        out(c(f"  log: {log_path}", C.DIM))
        log.close()
        sys.exit(1)

    step_keys = ", ".join(key for key, *_ in active_steps)
    out(c(f"steps: {len(active_steps)} selected [{step_keys}]", C.DIM))
    out("")

    out("[startup] waking WSL + Postgres")
    start_wsl_keepalive()
    wake_ok, wake_msg, wake_dur = wake_wsl_postgres(report=out)
    wake_tick = c(">", C.GREEN) if wake_ok else c("!", C.RED)
    wake_stat = c("OK", C.GREEN) if wake_ok else c("WARN", C.RED)
    out(f"       {wake_tick} {wake_stat} {wake_dur:>5.1f}s  {c(wake_msg, C.DIM)}")
    if not wake_ok:
        out(c("       | continuing to DB readiness check anyway", C.RED, C.DIM))
    out("")

    try:
        out("[startup] waiting for Postgres to accept connections")
        db_ok, db_msg, db_dur = wait_for_db(max_wait_s=args.db_wait, report=out)
        tick = c(">", C.GREEN) if db_ok else c("x", C.RED)
        stat = c("OK", C.GREEN) if db_ok else c("FAIL", C.RED)
        out(f"       {tick} {stat} {db_dur:>5.1f}s  {c(db_msg, C.DIM)}")
        if not db_ok:
            out(c(f"       | {db_msg}", C.RED, C.DIM))
            out(c(f"  log: {log_path}", C.DIM))
            log.close()
            sys.exit(1)
        out("")

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
    finally:
        stop_wsl_keepalive()


if __name__ == "__main__":
    main()
