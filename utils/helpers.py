"""Polars / pandas input conversion helpers."""

from __future__ import annotations
import atexit
import contextlib
import os
import re
import subprocess
import sys
import threading
import time
import polars as pl
import pandas as pd
import psycopg
from typing import Union

DB_DSN = os.getenv("DB_DSN", "postgresql://benjils:snickers@raptor:5432/markets?connect_timeout=10")
WSL_KEEPALIVE_NAME = "views-wsl-keepalive"


def _ensure_wsl_keepalive() -> None:
    """Keep WSL attached for a bounded window so Postgres does not idle out."""
    if os.name != "nt" or os.getenv("VIEWS_SKIP_WSL_KEEPALIVE") == "1":
        return

    distro = os.getenv("VIEWS_WSL_DISTRO", os.getenv("REF_WSL_DISTRO", "Ubuntu"))
    seconds = int(os.getenv("VIEWS_WSL_KEEPALIVE_SECONDS", "3600"))
    script = (
        f"pkill -f '^{WSL_KEEPALIVE_NAME} ' >/dev/null 2>&1 || true; "
        f"nohup bash -c 'exec -a {WSL_KEEPALIVE_NAME} sleep {seconds}' "
        ">/dev/null 2>&1 & "
        "nohup service postgresql start >/dev/null 2>&1 &"
    )
    try:
        subprocess.run(
            ["wsl", "-d", distro, "-u", "root", "--", "bash", "-lc", script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=90,
        )
    except Exception:
        # Best-effort only; _connect's normal wake/retry path will report issues.
        return


_wsl_hold: subprocess.Popen | None = None


def hold_wsl() -> bool:
    """Pin the WSL VM up for the lifetime of this process. Returns True if held.

    WSL2 recycles its VM about a minute after the last wsl.exe process
    detaches, and `vmIdleTimeout` does not prevent it. When that happens
    postgres is killed without a clean shutdown, so the next connection finds
    it replaying WAL and reporting "the database system is starting up".

    _ensure_wsl_keepalive() cannot prevent this: it backgrounds a sleep
    *inside* the VM and lets wsl.exe exit, and it is the attached Windows-side
    process that counts. Holding one open is the only thing that works, which
    is why long-running callers (a dashboard, a data pull) need this and
    one-shot queries do not.

    Idempotent, best-effort, and released automatically at interpreter exit.
    """
    global _wsl_hold
    if os.name != "nt" or os.getenv("VIEWS_SKIP_WSL_KEEPALIVE") == "1":
        return False
    if _wsl_hold is not None and _wsl_hold.poll() is None:
        return True
    distro = os.getenv("VIEWS_WSL_DISTRO", os.getenv("REF_WSL_DISTRO", "Ubuntu"))
    try:
        _ensure_wsl_keepalive()  # also starts postgres if the VM was down
        _wsl_hold = subprocess.Popen(
            ["wsl", "-d", distro, "-u", "root", "--", "sleep", "infinity"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        _wsl_hold = None
        return False
    atexit.register(release_wsl)
    return True


def release_wsl() -> None:
    """Drop the WSL hold taken by hold_wsl(). Safe to call more than once."""
    global _wsl_hold
    if _wsl_hold is None:
        return
    proc, _wsl_hold = _wsl_hold, None
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except Exception:
        proc.kill()


def _validate_conn(conn: psycopg.Connection) -> psycopg.Connection:
    """Return conn only after a trivial query succeeds."""
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
    except Exception:
        conn.close()
        raise
    return conn


def _connect(dsn: str | None = None, wake: bool = True) -> psycopg.Connection:
    """Open a DB connection. On Windows, wakes WSL+postgres automatically if the first attempt fails.

    wake=False makes one short-timeout attempt and raises on failure: no WSL
    wake, no retry ladder, no progress output. For callers that must not block
    on a sleeping database because they already have something to fall back
    on -- a UI render drawing from a local cache, say. The wake path can take
    half a minute when postgres is replaying WAL, which is far too long to
    hold a page render.
    """
    dsn = dsn or DB_DSN

    probe = re.sub(r"connect_timeout=\d+", "connect_timeout=3", dsn)
    if "connect_timeout" not in probe:
        probe += ("&" if "?" in probe else "?") + "connect_timeout=3"

    if not wake:
        return _validate_conn(psycopg.connect(probe))

    if os.name != "nt":
        return _validate_conn(psycopg.connect(dsn))

    try:
        return _validate_conn(psycopg.connect(probe))
    except Exception as e:
        probe_err = str(e).split("\n")[0].strip()
        starting_up = "starting up" in probe_err.lower()

    t0   = time.time()
    lock = threading.Lock()

    def _log(msg: str) -> None:
        with lock:
            sys.stdout.write(f"\r{' ' * 72}\r  {msg}\n")
            sys.stdout.flush()

    def _start_ticker(label: str) -> threading.Event:
        phase_t0 = time.time()
        stop = threading.Event()
        def _tick():
            while not stop.wait(0.1):
                elapsed  = time.time() - phase_t0
                secs_int = int(elapsed)
                dots = "." * (((max(secs_int, 1) - 1) % 3) + 1)
                with lock:
                    sys.stdout.write(f"\r  {label}  {elapsed:.1f}s{dots:<4}")
                    sys.stdout.flush()
        threading.Thread(target=_tick, daemon=True).start()
        return stop

    _log(f"probe: {probe_err}")

    # ── phase 1: wake WSL and start postgres ─────────────────────────────────
    if not starting_up:
        _log("step 1 — waking WSL...")
        stop1 = _start_ticker("waking WSL")
        _ensure_wsl_keepalive()
        stop1.set()
        _log(f"step 1 done  [{time.time() - t0:.1f}s]")
    else:
        _log("step 1 — skipped (postgres already responding, just not ready yet)")

    # ── phase 2: wait for postgres to accept connections ──────────────────────
    _log("step 2 — waiting for postgres...")
    stop2 = _start_ticker("postgres")

    try:
        for attempt in range(1, 20):
            time.sleep(2)
            try:
                conn = _validate_conn(psycopg.connect(dsn))
                stop2.set()
                _log(f"attempt {attempt}: ready!  [{time.time() - t0:.1f}s total]")
                return conn
            except Exception as e:
                _log(f"attempt {attempt}: {str(e).split(chr(10))[0].strip()}")

        stop2.set()
        return _validate_conn(psycopg.connect(dsn))  # final - raises naturally if still down
    except Exception:
        stop2.set()
        raise


@contextlib.contextmanager
def timed(label: str):
    """Print label, tick dots every second while block runs, then print elapsed time."""
    print(label, end="", flush=True)
    stop = threading.Event()
    t0 = time.time()
    def _tick():
        while not stop.wait(1.0):
            print(".", end="", flush=True)
    threading.Thread(target=_tick, daemon=True).start()
    try:
        yield
    finally:
        stop.set()
        print(f" {time.time() - t0:.1f}s", flush=True)


def to_pl_series(s: Union[pl.Series, pd.Series]) -> pl.Series:
    if isinstance(s, pd.Series):
        return pl.from_pandas(s)
    return s


def to_pl_df(df: Union[pl.DataFrame, pd.DataFrame]) -> pl.DataFrame:
    if isinstance(df, pd.DataFrame):
        return pl.from_pandas(df)
    return df


def fix_outliers(
    expr: pl.Expr,
    *,
    hi: float | None = None,
    lo: float | None = None,
) -> pl.Expr:
    """Replace values outside (lo, hi) with linear interpolation from neighbors."""
    mask = pl.lit(False)
    if hi is not None:
        mask = mask | (expr > hi)
    if lo is not None:
        mask = mask | (expr < lo)
    return pl.when(mask).then(None).otherwise(expr).interpolate()


def query_db(
    sql: str,
    params: list | tuple | None = None,
    dsn: str | None = None,
    wake: bool = True,
) -> pd.DataFrame:
    """Run a SQL query and return a DataFrame. Opens and closes the connection for you.

    wake=False fails fast instead of waking a sleeping WSL/postgres -- see
    _connect. It also skips the retry, since retrying is the thing being
    opted out of.
    """
    if not wake:
        with _connect(dsn, wake=False) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])

    last_err: Exception | None = None
    for attempt in range(2):
        try:
            with _connect(dsn) as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    rows = cur.fetchall()
                    cols = [d.name for d in cur.description]
            break
        except psycopg.OperationalError as exc:
            last_err = exc
            if attempt == 1:
                raise
            time.sleep(2)
    else:
        raise last_err or RuntimeError("query failed")
    return pd.DataFrame(rows, columns=cols)


def query_df(conn, sql: str, params: list | tuple | None = None) -> pd.DataFrame:
    """Run a SQL query on an existing connection. Use query_db() instead for auto-managed connections."""
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
        cols = [d.name for d in cur.description]
    return pd.DataFrame(rows, columns=cols)
