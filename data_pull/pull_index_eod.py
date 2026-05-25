#!/usr/bin/env python3
"""
Pull EOD data for index/generic tickers from Bloomberg into TimescaleDB.

- Reads ticker universe from utils/tickers.py
- Uses Bloomberg BDH via Bbg helper
- Stores into md.index_eod (Timescale hypertable), keyed by (ticker, ts)
- Incremental: each run only fetches dates after max(ts) already stored

Usage:
    python pull_index_eod.py                     # incremental update
    python pull_index_eod.py --backfill 2020-01-01  # force start date
"""

from __future__ import annotations

import os
import sys
import datetime as dt
import argparse

import numpy as np
import pandas as pd
import psycopg
from tqdm import tqdm

sys.path.append(os.path.expanduser("~/werk/Views"))
from data_pull.berg import Bbg
from utils import tickers as tkr

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DB_DSN = os.getenv("DB_DSN", "postgresql://benjils:snickers@raptor:5432/markets")
BBG_FIELDS = ["PX_LAST", "PX_OPEN", "PX_HIGH", "PX_LOW", "VOLUME"]
BATCH_SIZE = 50

# Collect all tickers from tickers.py
ALL_TICKERS = (
    tkr.ust + tkr.crv + tkr.fly +
    tkr.sofr_ois + tkr.swp_spd +
    tkr.tips + tkr.be + tkr.zc +
    tkr.stonk + tkr.sector +
    tkr.prec_met + tkr.enrgy + tkr.ag + tkr.ind_met +
    tkr.mtg
)


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def get_conn():
    return psycopg.connect(DB_DSN)


def ensure_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("CREATE SCHEMA IF NOT EXISTS md;")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS md.index_eod (
                ts        date NOT NULL,
                ticker    text NOT NULL,
                px_last   double precision,
                px_open   double precision,
                px_high   double precision,
                px_low    double precision,
                volume    double precision,
                source    text NOT NULL DEFAULT 'BGN',
                created_at timestamptz NOT NULL DEFAULT now(),
                PRIMARY KEY (ticker, ts)
            );
        """)
        cur.execute("SELECT create_hypertable('md.index_eod', 'ts', if_not_exists => TRUE);")
    conn.commit()


def get_date_range(conn, force_start: dt.date | None = None) -> tuple[dt.date | None, dt.date | None]:
    today = dt.date.today()
    if force_start:
        return (force_start, today)

    with conn.cursor() as cur:
        cur.execute("SELECT max(ts)::date FROM md.index_eod;")
        row = cur.fetchone()

    if row is None or row[0] is None:
        return (dt.date(2000, 1, 1), today)

    max_date = row[0]
    if isinstance(max_date, dt.datetime):
        max_date = max_date.date()

    return (max_date, today)


def upsert(conn, df: pd.DataFrame) -> int:
    if df.empty:
        return 0

    sql = """
    INSERT INTO md.index_eod (ts, ticker, px_last, px_open, px_high, px_low, volume, source)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (ticker, ts) DO UPDATE SET
      px_last = EXCLUDED.px_last,
      px_open = EXCLUDED.px_open,
      px_high = EXCLUDED.px_high,
      px_low  = EXCLUDED.px_low,
      volume  = EXCLUDED.volume,
      source  = EXCLUDED.source;
    """

    rows = [
        (row.ts, row.ticker, row.px_last, row.px_open, row.px_high, row.px_low, row.volume, row.source)
        for row in df.itertuples(index=False)
    ]

    with conn.cursor() as cur:
        cur.executemany(sql, rows)
    conn.commit()
    return len(rows)


# ---------------------------------------------------------------------------
# Bloomberg fetch
# ---------------------------------------------------------------------------

def fetch_bdh(
    bbg: Bbg, conn, tickers: list[str],
    start: dt.date, end: dt.date,
    batch_size: int = BATCH_SIZE,
) -> int:
    expected_cols = ["ts", "ticker", "px_last", "px_open", "px_high", "px_low", "volume", "source"]
    total_inserted = 0
    n_batches = (len(tickers) + batch_size - 1) // batch_size

    for i in tqdm(
        range(0, len(tickers), batch_size),
        total=n_batches, desc="Bloomberg BDH batches", unit="batch",
    ):
        batch = tickers[i : i + batch_size]

        data = bbg.bdh(batch, BBG_FIELDS, start=start, end=end, periodicity="DAILY")

        batch_dfs: list[pd.DataFrame] = []
        for ticker, df in data.items():
            if df is None or df.empty or "error" in df.columns:
                continue

            df = df.rename(columns={
                "PX_LAST": "px_last", "PX_OPEN": "px_open",
                "PX_HIGH": "px_high", "PX_LOW": "px_low",
                "VOLUME": "volume",
            })
            df = df.reset_index().rename(columns={"date": "ts"})
            df["ticker"] = ticker
            df["source"] = "BGN"

            for col in expected_cols:
                if col not in df.columns:
                    df[col] = pd.NA

            for col in expected_cols:
                if df[col].dtype == object and df[col].isna().all():
                    df[col] = np.nan
            batch_dfs.append(df[expected_cols])

        batch_dfs = [df for df in batch_dfs if df['px_last'].notna().any()]
        if not batch_dfs:
            continue

        inserted = upsert(conn, pd.concat(batch_dfs, ignore_index=True))
        total_inserted += inserted

    return total_inserted


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backfill", type=str, help="Force start date (YYYY-MM-DD)")
    args = parser.parse_args()

    force_start = dt.datetime.strptime(args.backfill, "%Y-%m-%d").date() if args.backfill else None

    print(f"Connecting to Postgres: {DB_DSN}")
    conn = get_conn()
    ensure_table(conn)

    start, end = get_date_range(conn, force_start)
    if start is None:
        print("No new dates to pull. Up to date.")
        conn.close()
        return

    tickers = list(dict.fromkeys(ALL_TICKERS))  # dedupe, preserve order
    print(f"Pulling {len(tickers)} tickers from {start} to {end}...")

    bbg = Bbg()
    inserted = fetch_bdh(bbg, conn, tickers, start, end)

    if inserted == 0:
        print("Bloomberg returned no data. Nothing inserted.")
    else:
        print(f"✅ Upserted {inserted:,} rows into md.index_eod")

    conn.close()


if __name__ == "__main__":
    main()
