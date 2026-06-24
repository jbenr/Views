"""Polars / pandas input conversion helpers."""

from __future__ import annotations
import contextlib
import os
import re
import subprocess
import threading
import time
import polars as pl
import pandas as pd
import psycopg
from typing import Union

DB_DSN = os.getenv("DB_DSN", "postgresql://benjils:snickers@raptor:5432/markets?connect_timeout=10")


def _connect(dsn: str | None = None) -> psycopg.Connection:
    """Open a DB connection. On Windows, wakes WSL+postgres automatically if the first attempt fails."""
    dsn = dsn or DB_DSN

    # Probe with a short timeout so we don't wait 10s to discover WSL is down.
    probe = re.sub(r"connect_timeout=\d+", "connect_timeout=3", dsn)
    if "connect_timeout" not in probe:
        probe += ("&" if "?" in probe else "?") + "connect_timeout=3"

    try:
        return psycopg.connect(probe)
    except Exception:
        if os.name != "nt":
            raise  # not Windows — nothing to wake, surface the error

    print("DB unreachable — starting WSL postgres...", end="", flush=True)
    subprocess.run(
        ["wsl", "-d", "Ubuntu", "--", "service", "postgresql", "start"],
        capture_output=True, timeout=30,
    )
    for _ in range(5):
        time.sleep(2)
        try:
            conn = psycopg.connect(dsn)
            print(" ready.", flush=True)
            return conn
        except Exception:
            pass

    return psycopg.connect(dsn)  # final attempt — raises naturally if still down


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


def query_db(sql: str, params: list | tuple | None = None, dsn: str | None = None) -> pd.DataFrame:
    """Run a SQL query and return a DataFrame. Opens and closes the connection for you."""
    with _connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
            cols = [d.name for d in cur.description]
    return pd.DataFrame(rows, columns=cols)


def query_df(conn, sql: str, params: list | tuple | None = None) -> pd.DataFrame:
    """Run a SQL query on an existing connection. Use query_db() instead for auto-managed connections."""
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
        cols = [d.name for d in cur.description]
    return pd.DataFrame(rows, columns=cols)
