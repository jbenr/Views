"""Polars / pandas input conversion helpers."""

from __future__ import annotations
import contextlib
import os
import threading
import time
import polars as pl
import pandas as pd
import psycopg
from typing import Union

DB_DSN = os.getenv("DB_DSN", "postgresql://benjils:snickers@raptor:5432/markets?connect_timeout=10")


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
    with psycopg.connect(dsn or DB_DSN) as conn:
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
