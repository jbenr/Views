#!/usr/bin/env python3
# pull_ust_cusips_postgres.py
#
# Full + incremental loader for Treasury Securities Auctions Data
# using the Fiscal Data API only.

import os
import sys
import json
from typing import Any, Dict, List

from datetime import datetime

import requests
import pandas as pd

try:
    import psycopg  # type: ignore
except ImportError:
    import psycopg2 as psycopg  # type: ignore


# -----------------------------------------------------------------------------#
# Config
# -----------------------------------------------------------------------------#

FISCAL_URL = (
    "https://api.fiscaldata.treasury.gov/"
    "services/api/fiscal_service/v1/accounting/od/auctions_query"
)

DB_DSN = os.getenv("DB_DSN", "postgresql://benjils:snickers@localhost:5432/markets")
TABLE = "auctioned_securities"
PAGE_SIZE = 10_000  # Fiscal Data API max is typically 10k per page


# -----------------------------------------------------------------------------#
# Helpers
# -----------------------------------------------------------------------------#

def get_conn():
    return psycopg.connect(DB_DSN)


def _pg_type(dtype) -> str:
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "TIMESTAMPTZ"
    if pd.api.types.is_integer_dtype(dtype):
        return "BIGINT"
    if pd.api.types.is_float_dtype(dtype):
        return "DOUBLE PRECISION"
    if pd.api.types.is_bool_dtype(dtype):
        return "BOOLEAN"
    return "TEXT"


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal normalization: parse dates & key numerics, but keep column names as-is."""
    df = df.copy()

    # Date-like fields in auctions_query
    for col in (
        "record_date",
        "auction_date",
        "issue_date",
        "maturity_date",
        "announcement_date",
        "dated_date",
        "call_date",
        "first_int_payment_date",
    ):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Some important numeric fields; you can add more if you care
    for col in (
        "total_accepted",
        "total_tendered",
        "bid_to_cover_ratio",
        "int_rate",
        "avg_med_yield",
        "high_yield",
        "low_yield",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def fetch_all_auctions(extra_params: Dict[str, Any] | None = None) -> pd.DataFrame:
    """
    Pull all pages from the Fiscal Data auctions_query endpoint.
    If extra_params is given, merge into the request params (e.g. filters).
    """
    rows: List[Dict[str, Any]] = []
    page = 1

    while True:
        params: Dict[str, Any] = {
            "page[size]": PAGE_SIZE,
            "page[number]": page,
        }
        if extra_params:
            params.update(extra_params)

        resp = requests.get(FISCAL_URL, params=params, timeout=60)
        resp.raise_for_status()
        payload = resp.json()

        data = payload.get("data", [])
        if not data:
            break

        rows.extend(data)

        links = payload.get("links", {})
        if not links.get("next"):
            # No more pages
            break

        page += 1

    if not rows:
        return pd.DataFrame()

    return _normalize(pd.DataFrame(rows))


def _ensure_schema_and_indexes(conn, df: pd.DataFrame) -> None:
    """
    Ensure table exists with columns based on df (if not already there),
    and that useful indexes exist.
    """
    cols = [f'"{name}" {_pg_type(dtype)}' for name, dtype in df.dtypes.items()]
    ddl = f'CREATE TABLE IF NOT EXISTS "{TABLE}" ({", ".join(cols)});'

    with conn.cursor() as cur:
        cur.execute(ddl)

        # Unique index on (cusip, auction_date) enables ON CONFLICT DO NOTHING
        if {"cusip", "auction_date"}.issubset(df.columns):
            cur.execute(
                f'CREATE UNIQUE INDEX IF NOT EXISTS '
                f'idx_{TABLE}_cusip_auction_date '
                f'ON "{TABLE}"("cusip","auction_date");'
            )

        if "cusip" in df.columns:
            cur.execute(
                f'CREATE INDEX IF NOT EXISTS idx_{TABLE}_cusip '
                f'ON "{TABLE}"("cusip");'
            )

        if "auction_date" in df.columns:
            cur.execute(
                f'CREATE INDEX IF NOT EXISTS idx_{TABLE}_auction_date '
                f'ON "{TABLE}"("auction_date");'
            )

    conn.commit()


def _insert_dataframe(conn, df: pd.DataFrame) -> int:
    """
    Insert rows from df into TABLE.
    - Aligns df to existing DB columns: ignores extra columns that DB doesn't have.
    - Uses ON CONFLICT (cusip, auction_date) DO NOTHING when those columns exist.
    Returns number of rows actually inserted.
    """
    if df.empty:
        return 0

    # Discover DB schema
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = %s
            """,
            (TABLE,),
        )
        db_cols = [row[0] for row in cur.fetchall()]

    # Keep only columns that exist in the DB
    cols = [c for c in db_cols if c in df.columns]
    if not cols:
        return 0

    df = df[cols]

    col_list = ", ".join(f'"{c}"' for c in cols)
    placeholders = "(" + ",".join(["%s"] * len(cols)) + ")"

    conflict_clause = ""
    if {"cusip", "auction_date"}.issubset(cols):
        conflict_clause = ' ON CONFLICT ("cusip","auction_date") DO NOTHING'

    sql = f'INSERT INTO "{TABLE}" ({col_list}) VALUES {placeholders}{conflict_clause};'

    rows = [
        tuple(None if pd.isna(v) else v for v in row)
        for row in df.to_numpy()
    ]

    with conn.cursor() as cur:
        cur.executemany(sql, rows)
        inserted = cur.rowcount
    conn.commit()

    return inserted


def _get_max_auction_date(conn) -> datetime | None:
    """Return the max auction_date currently in the table, or None if empty/missing."""
    with conn.cursor() as cur:
        cur.execute(f'SELECT max("auction_date") FROM "{TABLE}";')
        row = cur.fetchone()
    return row[0] if row and row[0] is not None else None


# -----------------------------------------------------------------------------#
# Modes
# -----------------------------------------------------------------------------#

def full_refresh() -> None:
    """
    Full historical load: hit the Fiscal Data API for all auctions and
    rebuild the table from scratch.
    """
    print("📡 Fetching full auctions history from Fiscal Data API…")
    df = fetch_all_auctions()

    if df.empty:
        print("No data returned from API for full refresh.", file=sys.stderr)
        sys.exit(1)

    before = len(df)
    if {"cusip", "auction_date"}.issubset(df.columns):
        df = df.sort_values(["cusip", "auction_date"]).drop_duplicates(
            ["cusip", "auction_date"], keep="last"
        )
    elif "cusip" in df.columns:
        df = df.sort_values("cusip").drop_duplicates(["cusip"], keep="last")
    after = len(df)

    with get_conn() as conn:
        _ensure_schema_and_indexes(conn, df)
        with conn.cursor() as cur:
            cur.execute(f'TRUNCATE TABLE "{TABLE}";')
        conn.commit()

        inserted = _insert_dataframe(conn, df)

    print(
        f"✅ Full refresh complete.\n"
        f"   API rows: {before}\n"
        f"   After dedupe: {after}\n"
        f"   Inserted into {TABLE}: {inserted}"
    )


def incremental_update() -> None:
    """
    Incremental update:
    - Look up max(auction_date) in DB.
    - Ask API only for auctions with auction_date > that.
    - Insert with ON CONFLICT DO NOTHING.
    """
    with get_conn() as conn:
        max_dt = _get_max_auction_date(conn)

        # If table empty / doesn't exist yet, just do full refresh
        if max_dt is None:
            print("No existing data found. Running full refresh instead.")
            conn.close()
            full_refresh()
            return

        cutoff = max_dt.date().isoformat()
        print(f"📡 Fetching auctions with auction_date > {cutoff} …")

        df = fetch_all_auctions(
            extra_params={
                "filter": f"auction_date:gt:{cutoff}",
            }
        )

        if df.empty:
            print(
                "✅ No new auctions returned by API.\n"
                f"   Database is up to date as of auction_date {cutoff}."
            )
            return

        api_rows = len(df)
        if {"cusip", "auction_date"}.issubset(df.columns):
            df = df.sort_values(["cusip", "auction_date"]).drop_duplicates(
                ["cusip", "auction_date"], keep="last"
            )
        elif "cusip" in df.columns:
            df = df.sort_values("cusip").drop_duplicates(["cusip"], keep="last")
        deduped_rows = len(df)

        _ensure_schema_and_indexes(conn, df)
        inserted = _insert_dataframe(conn, df)

    print(
        f"✅ Incremental update complete.\n"
        f"   API rows returned: {api_rows}\n"
        f"   After dedupe: {deduped_rows}\n"
        f"   New rows inserted into {TABLE}: {inserted}"
    )


# -----------------------------------------------------------------------------#
# Entry point
# -----------------------------------------------------------------------------#

def main():
    mode = "incremental"
    if len(sys.argv) > 1 and sys.argv[1].lower() in {"full", "all", "refresh"}:
        mode = "full"

    if mode == "full":
        full_refresh()
    else:
        incremental_update()


if __name__ == "__main__":
    main()
