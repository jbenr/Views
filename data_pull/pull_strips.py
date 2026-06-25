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
from collections import defaultdict

import numpy as np
import requests
import pandas as pd
import psycopg
from tqdm import tqdm

sys.path.append(os.path.expanduser("~/werk/Views"))
from data_pull.berg import Bbg
from data_pull.blacklist import STRIP_CUSIPS

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DB_DSN = os.getenv("DB_DSN", "postgresql://benjils:snickers@raptor:5432/markets")

# MSPD Table 5 — "Holdings of Treasury Securities in Stripped Form"
# Each row covers one parent Note/Bond/TIPS with its principal-STRIP CUSIP, the
# parent's coupon and maturity, outstanding, and how much has been stripped.
STRIPS_URL = (
    "https://api.fiscaldata.treasury.gov/"
    "services/api/fiscal_service/v1/debt/mspd/mspd_table_5"
)
LATEST_DATE_URL = STRIPS_URL  # same endpoint, queried with page size 1
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
                parent_cusip       text,
                created_at         timestamptz NOT NULL DEFAULT now(),
                updated_at         timestamptz NOT NULL DEFAULT now()
            );
        """)
        cur.execute("ALTER TABLE sec.strips ADD COLUMN IF NOT EXISTS parent_cusip text;")
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


def get_grouped_pulls(
    conn, cusips: list[str], force_start: dt.date | None = None
) -> dict[dt.date, list[str]]:
    """Return {start_date: [cusips]} grouping by per-CUSIP max(ts).

    New CUSIPs not yet in the DB fall back to 2000-01-01 so they get full history
    on the first pull — no manual backfill needed when new STRIPS are discovered.
    """
    today = dt.date.today()

    if force_start:
        return {force_start: list(cusips)}

    with conn.cursor() as cur:
        cur.execute("SELECT cusip, MAX(ts)::date FROM md.strips_eod GROUP BY cusip")
        db_maxes = {row[0]: row[1] for row in cur.fetchall()}

    fallback = dt.date(2000, 1, 1)
    groups: dict[dt.date, list[str]] = defaultdict(list)
    for cusip in cusips:
        start = db_maxes.get(cusip, fallback)
        start = start.date() if isinstance(start, dt.datetime) else start
        if start <= today:
            groups[start].append(cusip)

    return dict(groups)


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

def _latest_record_date() -> str | None:
    """Most recent monthly snapshot date in MSPD Table 5 (YYYY-MM-DD)."""
    resp = requests.get(
        STRIPS_URL,
        params={"page[size]": 1, "sort": "-record_date", "fields": "record_date"},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json().get("data", [])
    return data[0]["record_date"] if data else None


def fetch_strips_universe() -> pd.DataFrame:
    """
    Pull the latest snapshot of MSPD Table 5 (Holdings of Treasury Securities in
    Stripped Form). Returns one row per parent bond with its principal-STRIP CUSIP.
    """
    record_date = _latest_record_date()
    if record_date is None:
        return pd.DataFrame()

    rows = []
    page = 1
    while True:
        params = {
            "page[size]": PAGE_SIZE,
            "page[number]": page,
            "filter": f"record_date:eq:{record_date}",
        }
        resp = requests.get(STRIPS_URL, params=params, timeout=60)
        resp.raise_for_status()
        payload = resp.json()

        data = payload.get("data", [])
        if not data:
            break
        rows.extend(data)

        meta = payload.get("meta", {})
        total_pages = (meta.get("pagination") or {}).get("total-pages", page)
        if page >= total_pages:
            break
        page += 1

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def normalize_strips(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce MSPD Table 5 fields and project into our schema."""
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    for col in ("record_date", "maturity_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="ISO8601", errors="coerce")

    for col in ("outstanding_amt", "portion_unstripped_amt",
                "portion_stripped_amt", "reconstituted_amt", "interest_rate_pct"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    out = pd.DataFrame({
        "cusip":           df.get("cusip"),
        "parent_cusip":    df.get("security_class2_desc"),
        "strip_type":      df.get("security_class1_desc"),
        "maturity_date":   df.get("maturity_date"),
        "outstanding_amt": df.get("portion_stripped_amt"),  # strip-level, in $K
        "record_date":     df.get("record_date"),
    })
    out["security_desc"] = (
        out["strip_type"].fillna("").astype(str) + " STRIP "
        + out["maturity_date"].dt.strftime("%Y-%m-%d").fillna("")
    ).str.strip()

    # Only keep rows where some portion has actually been stripped — otherwise
    # the CUSIP won't quote on Bloomberg.
    out = out[out["outstanding_amt"].fillna(0) > 0]
    out = out.dropna(subset=["cusip"]).drop_duplicates(subset=["cusip"], keep="last")
    return out


def upsert_strips_metadata(conn, df: pd.DataFrame) -> int:
    """Upsert STRIPS universe into sec.strips."""
    if df.empty:
        return 0

    sql = """
    INSERT INTO sec.strips (cusip, security_desc, maturity_date, strip_type,
                            outstanding_amt, record_date, parent_cusip, updated_at)
    VALUES (%s, %s, %s, %s, %s, %s, %s, now())
    ON CONFLICT (cusip) DO UPDATE SET
      security_desc   = EXCLUDED.security_desc,
      maturity_date   = EXCLUDED.maturity_date,
      strip_type      = EXCLUDED.strip_type,
      outstanding_amt = EXCLUDED.outstanding_amt,
      record_date     = EXCLUDED.record_date,
      parent_cusip    = EXCLUDED.parent_cusip,
      updated_at      = now();
    """

    def _clean(v):
        if v is None:
            return None
        try:
            if pd.isna(v):
                return None
        except (TypeError, ValueError):
            pass
        return v

    rows = [
        tuple(_clean(v) for v in (
            r.cusip, r.security_desc, r.maturity_date, r.strip_type,
            r.outstanding_amt, r.record_date, r.parent_cusip,
        ))
        for r in df.itertuples(index=False)
    ]

    with conn.cursor() as cur:
        cur.executemany(sql, rows)
    conn.commit()
    return len(rows)


# ---------------------------------------------------------------------------
# Coupon STRIPS — discover CUSIPs from parent bonds' coupon schedules
# ---------------------------------------------------------------------------
#
# MSPD Table 5 only enumerates principal STRIPS. Coupon STRIPS (TINTs) are
# fungible across parents — every coupon paid on the same date collapses into
# one CUSIP — so we derive them as:
#   1. all stripable Note/Bond parents from sec.auctioned_securities
#   2. all of their semi-annual coupon payment dates (working back from maturity)
#   3. ask Bloomberg `S MM/DD/YY Govt` -> ID_CUSIP for each unique date
# Irregular first-coupon strips (off the standard semi-annual grid) come from
# the parent's tint_cusip_1 / tint_cusip_2 fields directly.

BDP_BATCH = 500


def _generate_coupon_dates(dated: dt.date, maturity: dt.date) -> list[dt.date]:
    """Semi-annual coupon dates strictly after dated, up to and including maturity.

    Walks backward from maturity in 6-month steps, anchored on the maturity day/month.
    """
    out: list[dt.date] = []
    y, m, d = maturity.year, maturity.month, maturity.day
    while True:
        try:
            cand = dt.date(y, m, d)
        except ValueError:  # e.g. Feb 30 -> roll to end of month
            cand = dt.date(y, m, 28)
        if cand <= dated:
            break
        out.append(cand)
        # step back 6 months
        m -= 6
        if m <= 0:
            m += 12
            y -= 1
    return sorted(out)


def get_parent_coupon_schedules(conn) -> pd.DataFrame:
    """Stripable Note/Bond parents with future maturity + their schedule fields."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT DISTINCT ON (cusip)
                cusip,
                security_type,
                dated_date::date     AS dated_date,
                maturity_date::date  AS maturity_date,
                first_int_payment_date::date AS first_int_payment_date,
                int_payment_frequency,
                corpus_cusip,
                NULLIF(tint_cusip_1, 'null') AS tint_cusip_1,
                NULLIF(tint_cusip_2, 'null') AS tint_cusip_2
            FROM sec.auctioned_securities
            WHERE security_type IN ('Note', 'Bond')
              AND maturity_date > CURRENT_DATE
              AND dated_date IS NOT NULL
              AND NULLIF(corpus_cusip, 'null') IS NOT NULL
            ORDER BY cusip, record_date DESC;
        """)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
    return pd.DataFrame(rows, columns=cols)


def fetch_coupon_strip_universe(bbg: Bbg, conn) -> pd.DataFrame:
    """Discover coupon STRIP CUSIPs and return a DataFrame ready for upsert."""
    parents = get_parent_coupon_schedules(conn)
    if parents.empty:
        return pd.DataFrame()

    # Standard semi-annual coupon dates from each parent (future only).
    today = dt.date.today()
    all_dates: set[dt.date] = set()
    for r in parents.itertuples(index=False):
        all_dates.update(
            d for d in _generate_coupon_dates(r.dated_date, r.maturity_date)
            if d >= today
        )

    dates_sorted = sorted(all_dates)
    print(f"  Resolving {len(dates_sorted):,} unique coupon dates via Bloomberg…")

    tickers = [f"S {d.strftime('%m/%d/%y')} Govt" for d in dates_sorted]
    ticker_to_date = dict(zip(tickers, dates_sorted))

    resolved: list[dict] = []
    for i in tqdm(range(0, len(tickers), BDP_BATCH),
                  desc="Bloomberg BDP batches", unit="batch"):
        batch = tickers[i:i + BDP_BATCH]
        df = bbg.bdp(batch, ["ID_CUSIP", "MATURITY"])
        if df.empty:
            continue
        for ticker, row in df.iterrows():
            cusip = row.get("ID_CUSIP")
            if not cusip or (isinstance(cusip, float) and pd.isna(cusip)):
                continue
            mat = row.get("MATURITY") or ticker_to_date.get(ticker)
            resolved.append({
                "cusip": str(cusip),
                "maturity_date": pd.to_datetime(mat).date() if mat else ticker_to_date[ticker],
                "strip_type": "Coupon STRIP",
                "parent_cusip": None,
            })

    # Add irregular first-coupon TINT CUSIPs that Treasury publishes directly,
    # but only when the first coupon hasn't paid yet.
    for r in parents.itertuples(index=False):
        if r.first_int_payment_date is None or r.first_int_payment_date < today:
            continue
        for tint in (r.tint_cusip_1, r.tint_cusip_2):
            if tint and tint not in (None, "", "null"):
                resolved.append({
                    "cusip": str(tint),
                    "maturity_date": r.first_int_payment_date,
                    "strip_type": "Coupon STRIP",
                    "parent_cusip": r.cusip,
                })

    if not resolved:
        return pd.DataFrame()

    out = pd.DataFrame(resolved).dropna(subset=["cusip", "maturity_date"])
    out["security_desc"] = (
        "Coupon STRIP " + out["maturity_date"].astype(str)
    )
    out["outstanding_amt"] = None
    out["record_date"] = dt.date.today()
    out = out.drop_duplicates(subset=["cusip"], keep="last")
    return out[["cusip", "security_desc", "maturity_date", "strip_type",
                "outstanding_amt", "record_date", "parent_cusip"]]


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

        strip_cols = ["ts", "cusip", "px_last", "yld_ytm_mid", "source"]
        batch_df = pd.concat(
            [df.dropna(axis=1, how="all") for df in batch_dfs], ignore_index=True
        )
        for col in strip_cols:
            if col not in batch_df.columns:
                batch_df[col] = np.nan
        inserted = upsert_strips_eod(conn, batch_df[strip_cols])
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

    # Step 2: discover coupon STRIP CUSIPs from parent bonds + Bloomberg lookup.
    bbg = Bbg()
    print("Discovering coupon STRIP CUSIPs from parent bond schedules…")
    try:
        coupons = fetch_coupon_strip_universe(bbg, conn)
        if coupons.empty:
            print("  No coupon STRIPS discovered.")
        else:
            n = upsert_strips_metadata(conn, coupons)
            print(f"  Upserted {n:,} coupon STRIPS into sec.strips")
    except Exception as e:
        print(f"  Warning: coupon STRIP discovery failed ({e}). Continuing.")

    # Step 3: pull EOD from Bloomberg
    cusips = [c for c in get_active_strip_cusips(conn) if c not in STRIP_CUSIPS]
    if not cusips:
        print("No active STRIPS CUSIPs found. Nothing to pull from Bloomberg.")
        conn.close()
        return

    force_start = dt.datetime.strptime(args.backfill, "%Y-%m-%d").date() if args.backfill else None
    if args.mode.lower() in ("full", "all", "refresh"):
        force_start = dt.date(2000, 1, 1)

    groups = get_grouped_pulls(conn, cusips, force_start)

    if not groups:
        print("Up to date.")
        conn.close()
        return

    today = dt.date.today()
    n_cusips = sum(len(v) for v in groups.values())
    print(f"Pulling {n_cusips} STRIPS CUSIPs across {len(groups)} date group(s)...")

    total_inserted = 0
    for start, group_cusips in sorted(groups.items()):
        if len(groups) > 1:
            print(f"  {len(group_cusips)} CUSIPs from {start} to {today}...")
        inserted = fetch_strips_eod(bbg, conn, group_cusips, start, today)
        total_inserted += inserted

    if total_inserted == 0:
        print("Bloomberg returned no data. Nothing inserted.")
    else:
        print(f"✅ Upserted {total_inserted:,} rows into md.strips_eod")

    conn.close()


if __name__ == "__main__":
    main()
