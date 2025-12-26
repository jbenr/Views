#!/usr/bin/env python3
"""
Pull daily UST bond/note history from Bloomberg into TimescaleDB.

- Reads CUSIPs from public.auctioned_securities
- Uses Bloomberg BDH via Bbg helper (utils.bbg.Bbg)
- Stores into md.ust_eod (Timescale hypertable), keyed by (cusip, ts)
- Incremental: each run only fetches dates after max(ts) already stored
"""

from __future__ import annotations

import os
import datetime as dt
from typing import List, Tuple

import pandas as pd
import psycopg  # pip install "psycopg[binary]"
from tqdm import tqdm

from berg import Bbg  # adjust if you put Bbg somewhere else

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DB_DSN = os.getenv("DB_DSN", "postgresql://benjils:snickers@raptor:5432/markets")

# Bloomberg fields to pull
BBG_FIELDS = [
    "PX_LAST",
    "YLD_YTM_MID",
    "PX_BID",
    "PX_ASK",
    "YLD_YTM_BID",
    "YLD_YTM_ASK",
]

BATCH_SIZE = 5  # number of CUSIPs per BDH request


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def get_conn():
    return psycopg.connect(DB_DSN)


def ensure_ust_eod_table(conn) -> None:
    """Create md.ust_eod table and hypertable if they don't exist."""
    with conn.cursor() as cur:
        # schema
        cur.execute("CREATE SCHEMA IF NOT EXISTS md;")

        # table
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS md.ust_eod (
                ts          date    NOT NULL,
                cusip       text    NOT NULL,
                px_last     double precision,
                yld_ytm_mid double precision,
                px_bid      double precision,
                px_ask      double precision,
                bid_yield   double precision,
                ask_yield   double precision,
                source      text    NOT NULL DEFAULT 'BGN',
                created_at  timestamptz NOT NULL DEFAULT now(),
                PRIMARY KEY (cusip, ts)
            );
            """
        )

        # hypertable (safe to call repeatedly if_not_exists => TRUE)
        cur.execute(
            "SELECT create_hypertable('md.ust_eod', 'ts', if_not_exists => TRUE);"
        )

    conn.commit()


def get_cusips(conn) -> List[str]:
    """Distinct UST Note/Bond CUSIPs from auctioned_securities."""
    sql = """
        SELECT DISTINCT cusip
        FROM auctioned_securities
        WHERE cusip IS NOT NULL
          AND security_type IN ('Note', 'Bond')
        ORDER BY cusip;
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
    return [r[0] for r in rows]


def get_date_range(conn) -> tuple[dt.date, dt.date]:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT min(ts)::date, max(ts)::date
            FROM md.ust_eod;
        """)
        row = cur.fetchone()

    today = dt.date.today()

    if row is None or row[0] is None:
        # No data yet → pull from historical start
        return (dt.date(1990, 1, 1), today)

    start, end = row

    # Ensure both are `date` objects
    if isinstance(start, dt.datetime):
        start = start.date()
    if isinstance(end, dt.datetime):
        end = end.date()

    # Don’t allow future start
    if start > today:
        start = today

    # Start from day after last saved date to avoid duplicate pull
    start = end + dt.timedelta(days=1)
    if start > today:
        start = today

    return (start, today)


# ---------------------------------------------------------------------------
# Bloomberg fetch
# ---------------------------------------------------------------------------

def fetch_bdh_for_cusips(
    bbg: Bbg,
    conn,
    cusips: List[str],
    start: dt.date,
    end: dt.date,
    batch_size: int = BATCH_SIZE,
) -> int:
    """
    Call BDH in batches and upsert each batch into md.ust_eod.

    Returns total rows upserted.
    """
    expected_cols = [
        "ts",
        "cusip",
        "px_last",
        "yld_ytm_mid",
        "px_bid",
        "px_ask",
        "bid_yield",
        "ask_yield",
        "source",
    ]

    total_inserted = 0
    n_batches = (len(cusips) + batch_size - 1) // batch_size

    for i in tqdm(
        range(0, len(cusips), batch_size),
        total=n_batches,
        desc="Bloomberg BDH batches",
        unit="batch",
    ):
        batch_cusips = cusips[i : i + batch_size]
        tickers = [f"{c} Govt" for c in batch_cusips]

        data = bbg.bdh(
            tickers,
            BBG_FIELDS,
            start=start,
            end=end,
            periodicity="DAILY",
        )

        batch_dfs: list[pd.DataFrame] = []

        for ticker, df in data.items():
            if df is None or df.empty:
                continue
            if "error" in df.columns:
                # Optional: log the error
                # print(f"BDH error for {ticker}: {df['error'].iloc[0]}")
                continue

            cusip = ticker.split()[0]

            df = df.rename(
                columns={
                    "PX_LAST": "px_last",
                    "YLD_YTM_MID": "yld_ytm_mid",
                    "PX_BID": "px_bid",
                    "PX_ASK": "px_ask",
                    "YLD_YTM_BID": "bid_yield",
                    "YLD_YTM_ASK": "ask_yield",
                }
            )

            df = df.reset_index().rename(columns={"date": "ts"})
            df["cusip"] = cusip
            df["source"] = "BGN"

            # Ensure all expected columns are present
            for col in expected_cols:
                if col not in df.columns:
                    df[col] = pd.NA

            batch_dfs.append(df[expected_cols])

        if not batch_dfs:
            continue

        batch_df = pd.concat(batch_dfs, ignore_index=True)
        inserted = upsert_ust_eod(conn, batch_df)
        total_inserted += inserted

    return total_inserted


# ---------------------------------------------------------------------------
# Upsert into Timescale
# ---------------------------------------------------------------------------

def upsert_ust_eod(conn, df: pd.DataFrame) -> int:
    """Insert/update rows into md.ust_eod using ON CONFLICT (cusip, ts)."""
    if df.empty:
        return 0

    rows = [
        (
            row.ts,
            row.cusip,
            row.px_last,
            row.yld_ytm_mid,
            row.px_bid,
            row.px_ask,
            row.bid_yield,
            row.ask_yield,
            row.source,
        )
        for row in df.itertuples(index=False)
    ]

    sql = """
    INSERT INTO md.ust_eod
      (ts, cusip, px_last, yld_ytm_mid, px_bid, px_ask, bid_yield, ask_yield, source)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (cusip, ts) DO UPDATE SET
      px_last     = EXCLUDED.px_last,
      yld_ytm_mid = EXCLUDED.yld_ytm_mid,
      px_bid      = EXCLUDED.px_bid,
      px_ask      = EXCLUDED.px_ask,
      bid_yield   = EXCLUDED.bid_yield,
      ask_yield   = EXCLUDED.ask_yield,
      source      = EXCLUDED.source;
    """

    with conn.cursor() as cur:
        cur.executemany(sql, rows)

    conn.commit()
    return len(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Connecting to Postgres: {DB_DSN}")
    with get_conn() as conn:
        ensure_ust_eod_table(conn)

        cusips = get_cusips(conn)
        if not cusips:
            print("No CUSIPs found in auctioned_securities (Note/Bond). Nothing to do.")
            return

        start, end = get_date_range(conn)
        if start is None or end is None:
            print("No new dates to pull. You're up to date.")
            return

        print(f"Found {len(cusips)} CUSIPs.")
        print(f"Pulling Bloomberg history from {start} to {end}...")

        bbg = Bbg()  # uses localhost:8194 by default

        inserted = fetch_bdh_for_cusips(bbg, conn, cusips, start, end)
        if inserted == 0:
            print("Bloomberg returned no data. Nothing inserted.")
        else:
            print(f"✅ Upserted {inserted} rows into md.ust_eod")


if __name__ == "__main__":
    main()
