#!/usr/bin/env python3
# pull_ust_cusips_postgres.py
#
# Full + incremental loader for Treasury Securities Auctions Data (Fiscal Data API)
# -> loads into Postgres (schema-qualified table), supports full refresh + incremental updates.

import os
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import requests
import pandas as pd

try:
    import psycopg  # type: ignore
    _PSYCOPG_V3 = True
except ImportError:
    import psycopg2 as psycopg  # type: ignore
    _PSYCOPG_V3 = False


# ────────────────────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────────────────────

FISCAL_URL = (
    "https://api.fiscaldata.treasury.gov/"
    "services/api/fiscal_service/v1/accounting/od/auctions_query"
)
DB_DSN = os.getenv("DB_DSN", "postgresql://benjils:snickers@raptor:5432/markets")
TABLE = os.getenv("AUCTIONS_TABLE", "sec.auctioned_securities")  # schema.table
PAGE_SIZE = 10_000  # Fiscal Data API max page size is typically 10k
INCREMENTAL_LOOKBACK_DAYS = 7

# Split schema + table safely, and build a properly-quoted qualified name.
if "." not in TABLE:
    raise ValueError(f"TABLE must be schema-qualified like 'sec.auctioned_securities', got: {TABLE!r}")
SCHEMA, TABLE_NAME = TABLE.split(".", 1)


def qname() -> str:
    """Quoted schema-qualified table name: "sec"."auctioned_securities"."""
    return f'"{SCHEMA}"."{TABLE_NAME}"'


# ────────────────────────────────────────────────────────────────────────────────
# DB helpers
# ────────────────────────────────────────────────────────────────────────────────

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


def _ensure_schema_and_indexes(conn, df: pd.DataFrame) -> None:
    """Create schema/table (columns inferred from df) + a few useful indexes."""
    cols = [f'"{name}" {_pg_type(dtype)}' for name, dtype in df.dtypes.items()]
    ddl_schema = f'CREATE SCHEMA IF NOT EXISTS "{SCHEMA}";'
    ddl_table = f'CREATE TABLE IF NOT EXISTS {qname()} ({", ".join(cols)});'

    with conn.cursor() as cur:
        cur.execute(ddl_schema)
        cur.execute(ddl_table)

        # Index names should avoid dots; use schema_table prefix.
        idx_prefix = f"{SCHEMA}_{TABLE_NAME}"

        if {"cusip", "auction_date"}.issubset(df.columns):
            cur.execute(
                f'CREATE UNIQUE INDEX IF NOT EXISTS idx_{idx_prefix}_cusip_auction_date '
                f'ON {qname()}("cusip","auction_date");'
            )
        if "cusip" in df.columns:
            cur.execute(
                f'CREATE INDEX IF NOT EXISTS idx_{idx_prefix}_cusip '
                f'ON {qname()}("cusip");'
            )
        if "auction_date" in df.columns:
            cur.execute(
                f'CREATE INDEX IF NOT EXISTS idx_{idx_prefix}_auction_date '
                f'ON {qname()}("auction_date");'
            )

    conn.commit()


def _get_db_columns(conn) -> List[str]:
    """List columns in the target table using schema+table filters (correct for schema-qualified names)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
            ORDER BY ordinal_position
            """,
            (SCHEMA, TABLE_NAME),
        )
        return [r[0] for r in cur.fetchall()]


def _insert_dataframe(conn, df: pd.DataFrame) -> int:
    """
    Insert rows from df into the target table.
    - Only inserts columns that exist in DB (ignores extras).
    - Uses ON CONFLICT (cusip, auction_date) DO UPDATE when possible.
    Returns rows inserted (best-effort; rowcount can be driver-dependent).
    """
    if df.empty:
        return 0

    db_cols = _get_db_columns(conn)
    cols = [c for c in db_cols if c in df.columns]
    if not cols:
        return 0

    df = df[cols]
    col_list = ", ".join(f'"{c}"' for c in cols)
    placeholders = "(" + ",".join(["%s"] * len(cols)) + ")"

    conflict = ""
    if {"cusip", "auction_date"}.issubset(cols):
        update_cols = [c for c in cols if c not in {"cusip", "auction_date"}]
        assignments = ", ".join(
            f'"{c}" = EXCLUDED."{c}"' for c in update_cols
        )
        conflict = f' ON CONFLICT ("cusip","auction_date") DO UPDATE SET {assignments}'

    sql = f"INSERT INTO {qname()} ({col_list}) VALUES {placeholders}{conflict};"

    rows = [tuple(None if pd.isna(v) else v for v in row) for row in df.to_numpy()]

    with conn.cursor() as cur:
        cur.executemany(sql, rows)
        inserted = cur.rowcount  # may be -1 in some drivers; still okay as a signal
    conn.commit()
    return inserted


def _get_max_record_date(conn) -> Optional[datetime]:
    """Max publication date in table; None if table empty or missing column."""
    with conn.cursor() as cur:
        cur.execute(f'SELECT max("record_date") FROM {qname()};')
        row = cur.fetchone()
    return row[0] if row and row[0] is not None else None


# ────────────────────────────────────────────────────────────────────────────────
# API helpers
# ────────────────────────────────────────────────────────────────────────────────

def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Parse known date/numeric fields; keep original column names."""
    df = df.copy()

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
            df[col] = pd.to_datetime(df[col], format="ISO8601", errors="coerce")

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


def fetch_all_auctions(extra_params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Pull all pages from Fiscal Data auctions_query endpoint (optionally filtered)."""
    rows: List[Dict[str, Any]] = []
    page = 1

    while True:
        params: Dict[str, Any] = {"page[size]": PAGE_SIZE, "page[number]": page}
        if extra_params:
            params.update(extra_params)

        resp = requests.get(FISCAL_URL, params=params, timeout=60)
        resp.raise_for_status()
        payload = resp.json()

        data = payload.get("data", [])
        if not data:
            break
        rows.extend(data)

        if not payload.get("links", {}).get("next"):
            break
        page += 1

    return _normalize(pd.DataFrame(rows)) if rows else pd.DataFrame()


def _dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """Prefer the latest published row per (cusip, auction_date) or cusip."""
    if df.empty:
        return df
    if {"cusip", "auction_date"}.issubset(df.columns):
        sort_cols = ["cusip", "auction_date"]
        if "record_date" in df.columns:
            sort_cols.append("record_date")
        return df.sort_values(sort_cols).drop_duplicates(["cusip", "auction_date"], keep="last")
    if "cusip" in df.columns:
        return df.sort_values("cusip").drop_duplicates(["cusip"], keep="last")
    return df


# ────────────────────────────────────────────────────────────────────────────────
# Modes
# ────────────────────────────────────────────────────────────────────────────────

def full_refresh() -> None:
    """Fetch full history and rebuild table contents."""
    print("📡 Fetching full auctions history from Fiscal Data API…")
    df = fetch_all_auctions()
    if df.empty:
        print("No data returned from API for full refresh.", file=sys.stderr)
        sys.exit(1)

    before = len(df)
    df = _dedupe(df)
    after = len(df)

    with get_conn() as conn:
        _ensure_schema_and_indexes(conn, df)
        with conn.cursor() as cur:
            cur.execute(f"TRUNCATE TABLE {qname()};")
        conn.commit()
        inserted = _insert_dataframe(conn, df)

    print(
        "✅ Full refresh complete.\n"
        f"   API rows: {before}\n"
        f"   After dedupe: {after}\n"
        f"   Inserted into {SCHEMA}.{TABLE_NAME}: {inserted}"
    )


def incremental_update() -> None:
    """
    Incremental:
    - get max(record_date) from DB
    - refetch an inclusive publication-date overlap
    - upsert new or revised auction rows
    """
    with get_conn() as conn:
        try:
            max_dt = _get_max_record_date(conn)
        except Exception:
            # First run / table not created yet → bootstrap with full refresh.
            max_dt = None

        if max_dt is None:
            print("No existing data found (or table missing). Running full refresh instead.")
            full_refresh()
            return

        latest = max_dt.date()
        cutoff = (latest - timedelta(days=INCREMENTAL_LOOKBACK_DAYS)).isoformat()
        print(f"📡 Fetching auctions with record_date >= {cutoff} …")
        df = fetch_all_auctions(extra_params={"filter": f"record_date:gte:{cutoff}"})

        if df.empty:
            print(f"✅ No auctions returned by API. Up to date as of record_date {latest}.")
            return

        api_rows = len(df)
        df = _dedupe(df)
        deduped_rows = len(df)

        _ensure_schema_and_indexes(conn, df)
        inserted = _insert_dataframe(conn, df)

    print(
        "✅ Incremental update complete.\n"
        f"   API rows returned: {api_rows}\n"
        f"   After dedupe: {deduped_rows}\n"
        f"   Rows upserted into {SCHEMA}.{TABLE_NAME}: {inserted}"
    )


# ────────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────────

def main() -> None:
    mode = "incremental"
    if len(sys.argv) > 1 and sys.argv[1].lower() in {"full", "all", "refresh"}:
        mode = "full"
    (full_refresh if mode == "full" else incremental_update)()


if __name__ == "__main__":
    main()
