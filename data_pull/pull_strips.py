#!/usr/bin/env python3
"""
Pull Treasury STRIPS universe + daily prices into TimescaleDB.

- Discovers STRIPS CUSIPs from Treasury Direct fiscal data API (MSPD STRIPS table)
- Fetches daily prices/yields from Bloomberg BDH
- Stores metadata into sec.strips, EOD data into md.strips_eod (hypertable)
- Incremental: each run only fetches dates after max(ts) already stored

Usage:
    python pull_strips.py                        # incremental update
    python pull_strips.py --backfill 2020-01-01  # force start date
    python pull_strips.py full                   # full historical refresh
"""

from __future__ import annotations

import os
import sys
import datetime as dt
import argparse

import requests
import pandas as pd
import psycopg
from tqdm import tqdm

sys.path.append(os.path.expanduser("~/werk/Views"))
from data_pull.berg import Bbg

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DB_DSN = os.getenv("DB_DSN", "postgresql://benjils:snickers@raptor:5432/markets")

# Treasury Direct fiscal data API — MSPD STRIPS components
# NOTE: verify this endpoint returns CUSIP-level STRIPS data. If not, try:
#   v1/debt/mspd/mspd_table_3_market  (filter for STRIPS)
#   v2/accounting/od/strips
STRIPS_URL = (
    "https://api.fiscaldata.treasury.gov/"
    "services/api/fiscal_service/v1/debt/mspd/mspd_table_3_market"
)
PAGE_SIZE = 10_000

BBG_FIELDS = ["PX_LAST", "YLD_YTM_MID"]
BATCH_SIZE = 50


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def get_conn():
    return psycopg.connect(DB_DSN)


def ensure_tables(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("CREATE SCHEMA IF NOT EXISTS sec;")
        cur.execute("CREATE SCHEMA IF NOT EXISTS md;")

        cur.execute("""
            CREATE TABLE IF NOT EXISTS sec.strips (
                cusip              text PRIMARY KEY,
                security_desc      text,
                maturity_date      date,
                strip_type         text,
                outstanding_amt    double precision,
                record_date        date,
                created_at         timestamptz NOT NULL DEFAULT now(),
                updated_at         timestamptz NOT NULL DEFAULT now()
            );
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_strips_maturity ON sec.strips (maturity_date);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_strips_type ON sec.strips (strip_type);")

        cur.execute("""
            CREATE TABLE IF NOT EXISTS md.strips_eod (
                ts          date NOT NULL,
                cusip       text NOT NULL,
                px_last     double precision,
                yld_ytm_mid double precision,
                source      text NOT NULL DEFAULT 'BGN',
                created_at  timestamptz NOT NULL DEFAULT now(),
                PRIMARY KEY (cusip, ts)
            );
        """)
        cur.execute("SELECT create_hypertable('md.strips_eod', 'ts', if_not_exists => TRUE);")
    conn.commit()


def get_max_eod_date(conn) -> dt.date | None:
    with conn.cursor() as cur:
        cur.execute("SELECT max(ts)::date FROM md.strips_eod;")
        row = cur.fetchone()
    if row and row[0]:
        v = row[0]
        return v.date() if isinstance(v, dt.datetime) else v
    return None


def get_active_strip_cusips(conn) -> list[str]:
    """CUSIPs from sec.strips where maturity > today."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT cusip FROM sec.strips
            WHERE maturity_date > CURRENT_DATE
            ORDER BY cusip;
        """)
        return [r[0] for r in cur.fetchall()]


# ---------------------------------------------------------------------------
# Treasury Direct: STRIPS universe
# ---------------------------------------------------------------------------

def fetch_strips_universe() -> pd.DataFrame:
    """
    Pull STRIPS data from Treasury Direct fiscal data API.
    Returns DataFrame with CUSIP-level STRIPS info.
    """
    rows = []
    page = 1

    while True:
        params = {
            "page[size]": PAGE_SIZE,
            "page[number]": page,
            "filter": "security_class2:eq:Strip",
        }
        resp = requests.get(STRIPS_URL, params=params, timeout=60)
        resp.raise_for_status()
        payload = resp.json()

        data = payload.get("data", [])
        if not data:
            break
        rows.extend(data)

        if not payload.get("links", {}).get("next"):
            break
        page += 1

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df


def normalize_strips(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal normalization of Treasury Direct STRIPS data."""
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Parse dates
    for col in [c for c in df.columns if "date" in c]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # Parse numeric
    for col in [c for c in df.columns if "amt" in c or "amount" in c]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def upsert_strips_metadata(conn, df: pd.DataFrame) -> int:
    """Upsert STRIPS universe into sec.strips."""
    if df.empty:
        return 0

    # Map whatever columns we got to our table
    # This is flexible — we take what Treasury Direct gives us
    col_map = {}
    for src in df.columns:
        if "cusip" in src:
            col_map[src] = "cusip"
        elif "maturity" in src and "date" in src:
            col_map[src] = "maturity_date"
        elif "security_desc" in src or "security_description" in src:
            col_map[src] = "security_desc"
        elif src in ("security_class2", "security_class"):
            col_map[src] = "strip_type"
        elif "outstanding" in src and "amt" in src:
            col_map[src] = "outstanding_amt"
        elif src == "record_date":
            col_map[src] = "record_date"

    if "cusip" not in col_map.values():
        print("Warning: no CUSIP column found in STRIPS data", file=sys.stderr)
        return 0

    mapped = df.rename(columns=col_map)
    keep = [c for c in ["cusip", "security_desc", "maturity_date", "strip_type",
                         "outstanding_amt", "record_date"] if c in mapped.columns]
    mapped = mapped[keep].drop_duplicates(subset=["cusip"], keep="last")

    sql = """
    INSERT INTO sec.strips (cusip, security_desc, maturity_date, strip_type,
                            outstanding_amt, record_date, updated_at)
    VALUES (%s, %s, %s, %s, %s, %s, now())
    ON CONFLICT (cusip) DO UPDATE SET
      security_desc   = EXCLUDED.security_desc,
      maturity_date   = EXCLUDED.maturity_date,
      strip_type      = EXCLUDED.strip_type,
      outstanding_amt = EXCLUDED.outstanding_amt,
      record_date     = EXCLUDED.record_date,
      updated_at      = now();
    """

    rows = []
    for _, r in mapped.iterrows():
        rows.append((
            r.get("cusip"), r.get("security_desc"), r.get("maturity_date"),
            r.get("strip_type"), r.get("outstanding_amt"), r.get("record_date"),
        ))

    with conn.cursor() as cur:
        cur.executemany(sql, rows)
    conn.commit()
    return len(rows)


# ---------------------------------------------------------------------------
# Bloomberg: STRIPS prices
# ---------------------------------------------------------------------------

def fetch_strips_eod(
    bbg: Bbg, conn, cusips: list[str],
    start: dt.date, end: dt.date,
    batch_size: int = BATCH_SIZE,
) -> int:
    total_inserted = 0
    n_batches = (len(cusips) + batch_size - 1) // batch_size

    for i in tqdm(
        range(0, len(cusips), batch_size),
        total=n_batches, desc="Bloomberg BDH batches", unit="batch",
    ):
        batch_cusips = cusips[i : i + batch_size]
        tickers = [f"{c} Govt" for c in batch_cusips]

        data = bbg.bdh(tickers, BBG_FIELDS, start=start, end=end, periodicity="DAILY")

        batch_dfs: list[pd.DataFrame] = []
        for ticker, df in data.items():
            if df is None or df.empty or "error" in df.columns:
                continue

            cusip = ticker.split()[0]
            df = df.rename(columns={"PX_LAST": "px_last", "YLD_YTM_MID": "yld_ytm_mid"})
            df = df.reset_index().rename(columns={"date": "ts"})
            df["cusip"] = cusip
            df["source"] = "BGN"

            for col in ["ts", "cusip", "px_last", "yld_ytm_mid", "source"]:
                if col not in df.columns:
                    df[col] = pd.NA

            batch_dfs.append(df[["ts", "cusip", "px_last", "yld_ytm_mid", "source"]])

        if not batch_dfs:
            continue

        inserted = upsert_strips_eod(conn, pd.concat(batch_dfs, ignore_index=True))
        total_inserted += inserted

    return total_inserted


def upsert_strips_eod(conn, df: pd.DataFrame) -> int:
    if df.empty:
        return 0

    sql = """
    INSERT INTO md.strips_eod (ts, cusip, px_last, yld_ytm_mid, source)
    VALUES (%s, %s, %s, %s, %s)
    ON CONFLICT (cusip, ts) DO UPDATE SET
      px_last     = EXCLUDED.px_last,
      yld_ytm_mid = EXCLUDED.yld_ytm_mid,
      source      = EXCLUDED.source;
    """

    rows = [
        (row.ts, row.cusip, row.px_last, row.yld_ytm_mid, row.source)
        for row in df.itertuples(index=False)
    ]

    with conn.cursor() as cur:
        cur.executemany(sql, rows)
    conn.commit()
    return len(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backfill", type=str, help="Force start date (YYYY-MM-DD)")
    parser.add_argument("mode", nargs="?", default="incremental")
    args = parser.parse_args()

    conn = get_conn()
    ensure_tables(conn)

    # Step 1: refresh STRIPS universe from Treasury Direct
    print("Fetching STRIPS universe from Treasury Direct...")
    try:
        raw = fetch_strips_universe()
        if raw.empty:
            print("  Warning: no STRIPS data returned from API. Proceeding with existing sec.strips.")
        else:
            df = normalize_strips(raw)
            n = upsert_strips_metadata(conn, df)
            print(f"  Upserted {n:,} STRIPS into sec.strips")
    except Exception as e:
        print(f"  Warning: Treasury Direct fetch failed ({e}). Proceeding with existing sec.strips.")

    # Step 2: pull EOD from Bloomberg
    cusips = get_active_strip_cusips(conn)
    if not cusips:
        print("No active STRIPS CUSIPs found. Nothing to pull from Bloomberg.")
        conn.close()
        return

    force_start = dt.datetime.strptime(args.backfill, "%Y-%m-%d").date() if args.backfill else None
    if args.mode.lower() in ("full", "all", "refresh"):
        start, end = dt.date(2000, 1, 1), dt.date.today()
    elif force_start:
        start, end = force_start, dt.date.today()
    else:
        max_date = get_max_eod_date(conn)
        if max_date is None:
            start, end = dt.date(2000, 1, 1), dt.date.today()
        else:
            start = max_date
            end = dt.date.today()
            if start >= end:
                print("Up to date.")
                conn.close()
                return

    print(f"Pulling {len(cusips)} STRIPS CUSIPs from {start} to {end}...")
    bbg = Bbg()
    inserted = fetch_strips_eod(bbg, conn, cusips, start, end)

    if inserted == 0:
        print("Bloomberg returned no data. Nothing inserted.")
    else:
        print(f"✅ Upserted {inserted:,} rows into md.strips_eod")

    conn.close()


if __name__ == "__main__":
    main()
