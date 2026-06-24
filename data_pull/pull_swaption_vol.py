"""Pull swaption vol surface from Bloomberg into md.swaption_vol.

Fetches normal vol (PX_LAST, bps/yr) for the full expiry × tenor × strike
surface. Each row is stored with expiry/tenor/strike as explicit columns so
the surface can be queried and reshaped without parsing ticker strings.

Incremental: each run fetches from max(ts) already in the table.

Usage:
    python pull_swaption_vol.py
    python pull_swaption_vol.py --backfill 2010-01-01
"""

from __future__ import annotations

import os
import sys
import datetime as dt
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import psycopg
from tqdm import tqdm

sys.path.append(os.path.expanduser("~/werk/Views"))
from data_pull.berg import Bbg
from utils import tickers as tkr


# ── config ────────────────────────────────────────────────────────────────────

DB_DSN     = os.getenv("DB_DSN", "postgresql://benjils:snickers@raptor:5432/markets")
BATCH_SIZE = 50


# ── db helpers ────────────────────────────────────────────────────────────────

def get_conn():
    return psycopg.connect(DB_DSN)


def ensure_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("CREATE SCHEMA IF NOT EXISTS md;")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS md.swaption_vol (
                ts         date        NOT NULL,
                expiry     text        NOT NULL,
                tenor      int         NOT NULL,
                strike     int         NOT NULL,
                vol        double precision,
                source     text        NOT NULL DEFAULT 'BGN',
                created_at timestamptz NOT NULL DEFAULT now(),
                is_live    boolean     NOT NULL DEFAULT FALSE,
                PRIMARY KEY (ts, expiry, tenor, strike)
            );
        """)
        cur.execute("SELECT create_hypertable('md.swaption_vol', 'ts', if_not_exists => TRUE);")
    conn.commit()


def get_grouped_pulls(
    conn, tickers: list[str], meta: dict, force_start: dt.date | None = None
) -> dict[dt.date, list[str]]:
    """Return {start_date: [tickers]} grouping by per-surface-point max(ts).

    Each Bloomberg ticker maps to an (expiry, tenor, strike) point. New points added
    to the surface fall back to 2010-01-01 so they backfill automatically.
    """
    today = dt.date.today()

    if force_start:
        return {force_start: list(tickers)}

    with conn.cursor() as cur:
        cur.execute("""
            SELECT expiry, tenor, strike, MAX(ts)::date
            FROM md.swaption_vol
            GROUP BY expiry, tenor, strike
        """)
        db_maxes = {(row[0], row[1], row[2]): row[3] for row in cur.fetchall()}

    fallback = dt.date(2010, 1, 1)
    groups: dict[dt.date, list[str]] = defaultdict(list)
    for ticker in tickers:
        m = meta.get(ticker, {})
        key = (m.get("expiry"), m.get("tenor"), m.get("strike"))
        start = db_maxes.get(key, fallback)
        start = start.date() if isinstance(start, dt.datetime) else start
        if start < today:
            groups[start].append(ticker)

    return dict(groups)


def upsert(conn, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    sql = """
    INSERT INTO md.swaption_vol (ts, expiry, tenor, strike, vol, source)
    VALUES (%s, %s, %s, %s, %s, %s)
    ON CONFLICT (ts, expiry, tenor, strike) DO UPDATE SET
        vol    = EXCLUDED.vol,
        source = EXCLUDED.source;
    """
    rows = [
        (row.ts, row.expiry, row.tenor, row.strike, row.vol, row.source)
        for row in df.itertuples(index=False)
    ]
    with conn.cursor() as cur:
        cur.executemany(sql, rows)
    conn.commit()
    return len(rows)


# ── bloomberg fetch ───────────────────────────────────────────────────────────

def fetch_bdh(
    bbg: Bbg,
    conn,
    tickers: list[str],
    meta: dict,
    start: dt.date,
    end: dt.date,
    batch_size: int = BATCH_SIZE,
) -> int:
    total_inserted = 0
    n_batches = (len(tickers) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(tickers), batch_size), total=n_batches, desc="Bloomberg BDH batches", unit="batch"):
        batch = tickers[i : i + batch_size]
        data  = bbg.bdh(batch, ["PX_LAST"], start=start, end=end, periodicity="DAILY")

        batch_rows = []
        for ticker, df in data.items():
            if df is None or df.empty or "error" in df.columns:
                continue
            if ticker not in meta:
                continue
            m = meta[ticker]
            df = (df
                  .rename(columns={"PX_LAST": "vol"})
                  .reset_index()
                  .rename(columns={"date": "ts"}))
            df["expiry"] = m["expiry"]
            df["tenor"]  = m["tenor"]
            df["strike"] = m["strike"]
            df["source"] = "BGN"
            batch_rows.append(df[["ts", "expiry", "tenor", "strike", "vol", "source"]])

        batch_rows = [d for d in batch_rows if d["vol"].notna().any()]
        if not batch_rows:
            continue

        total_inserted += upsert(conn, pd.concat(batch_rows, ignore_index=True))

    return total_inserted


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backfill", type=str, help="Force start date (YYYY-MM-DD)")
    args = parser.parse_args()

    force_start = dt.datetime.strptime(args.backfill, "%Y-%m-%d").date() if args.backfill else None

    print(f"Connecting to Postgres: {DB_DSN}")
    conn = get_conn()
    ensure_table(conn)

    tickers = list(tkr.swpn_meta.keys())
    groups = get_grouped_pulls(conn, tickers, tkr.swpn_meta, force_start)

    if not groups:
        print("No new dates to pull. Up to date.")
        conn.close()
        return

    today = dt.date.today()
    n_tickers = sum(len(v) for v in groups.values())
    print(f"Pulling {n_tickers} swaption vol tickers across {len(groups)} date group(s)...")

    bbg = Bbg()
    total_inserted = 0

    for start, group_tickers in sorted(groups.items()):
        if len(groups) > 1:
            print(f"  {len(group_tickers)} tickers from {start} to {today}...")
        inserted = fetch_bdh(bbg, conn, group_tickers, tkr.swpn_meta, start, today)
        total_inserted += inserted

    if total_inserted == 0:
        print("Bloomberg returned no data. Nothing inserted.")
    else:
        print(f"Upserted {total_inserted:,} rows into md.swaption_vol")

    conn.close()


if __name__ == "__main__":
    main()
