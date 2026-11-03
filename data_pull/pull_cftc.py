#!/usr/bin/env python3
"""
Pull CFTC Traders in Financial Futures (TFF) positioning data into TimescaleDB.

- Downloads from CFTC.gov (2006-2016 archive, yearly zips from 2017, current week text file)
- Two reports, same columns:
    futures   -> md.cftc           (futures only)
    combined  -> md.cftc_combined  (futures + options, options delta-adjusted)
  Combined minus futures-only is each trader group's delta-equivalent options position.
- Incremental: each run only fetches weeks after max(ts) already stored
- No Bloomberg dependency

Usage:
    python pull_cftc.py              # incremental update, both reports
    python pull_cftc.py backfill     # upsert all history from 2006, both reports (non-destructive)
    python pull_cftc.py full         # truncate and reload all history, both reports
"""

from __future__ import annotations

import io
import os
import sys
import zipfile
import datetime as dt

import requests
import pandas as pd
import psycopg
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DB_DSN = os.getenv("DB_DSN", "postgresql://benjils:snickers@raptor:5432/markets")

BASE_URL = "https://www.cftc.gov"
FIRST_YEAR = 2006
ARCHIVE_LAST_YEAR = 2016  # the CFTC packs 2006-2016 into one file per report
REPORTS = {
    "futures": dict(
        table="md.cftc",
        archive=BASE_URL + "/files/dea/history/fin_fut_txt_2006_2016.zip",
        yearly=BASE_URL + "/files/dea/history/fut_fin_txt_{year}.zip",
        current=BASE_URL + "/dea/newcot/FinFutWk.txt",
    ),
    "combined": dict(
        table="md.cftc_combined",
        archive=BASE_URL + "/files/dea/history/fin_com_txt_2006_2016.zip",
        yearly=BASE_URL + "/files/dea/history/com_fin_txt_{year}.zip",
        current=BASE_URL + "/dea/newcot/FinComWk.txt",
    ),
}

# Short names for contracts we care about most (by CFTC_Contract_Market_Code)
SHORT_NAMES = {
    "042601": "UST_2Y",   "044601": "UST_5Y",   "043602": "UST_10Y",
    "020601": "UST_BOND",  "020604": "ULTRA_BOND", "043607": "ULTRA_10Y",
    "134741": "SOFR_3M",   "134742": "SOFR_1M",
    "13874A": "ES",        "13874+": "SP500",
    "209742": "NQ",        "20974+": "NASDAQ",
    "1170E1": "VIX",       "098662": "DXY",       "045601": "FF",
}
CODE_COLUMN = "CFTC_Contract_Market_Code"  # read as text: codes have leading zeros


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def get_conn():
    return psycopg.connect(DB_DSN)


def _pg_type(dtype) -> str:
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "TIMESTAMPTZ"
    if pd.api.types.is_integer_dtype(dtype):
        return "BIGINT"
    if pd.api.types.is_float_dtype(dtype):
        return "DOUBLE PRECISION"
    return "TEXT"


def ensure_table(conn, df: pd.DataFrame, table: str) -> None:
    """Create the report table from DataFrame columns, plus hypertable + indexes."""
    name = table.split(".")[1]
    cols = [f'"{c}" {_pg_type(df[c].dtype)}' for c in df.columns]
    copy_futures = table != REPORTS["futures"]["table"] and table_exists(conn, REPORTS["futures"]["table"])
    with conn.cursor() as cur:
        cur.execute("CREATE SCHEMA IF NOT EXISTS md;")
        if copy_futures:  # same columns and types as the futures-only table, so the two subtract cleanly
            cur.execute(f"CREATE TABLE IF NOT EXISTS {table} "
                        f"(LIKE {REPORTS['futures']['table']} INCLUDING DEFAULTS INCLUDING CONSTRAINTS INCLUDING INDEXES);")
        else:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    {', '.join(cols)},
                    created_at timestamptz NOT NULL DEFAULT now(),
                    PRIMARY KEY (cftc_contract_market_code, ts)
                );
            """)
        cur.execute(f"SELECT create_hypertable('{table}', 'ts', if_not_exists => TRUE);")
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{name}_short_name "
            f"ON {table} (short_name) WHERE short_name IS NOT NULL;"
        )
    conn.commit()


def table_exists(conn, table: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s) IS NOT NULL;", (table,))
        return bool(cur.fetchone()[0])


def get_max_date(conn, table: str) -> dt.date | None:
    if not table_exists(conn, table):
        return None
    with conn.cursor() as cur:
        cur.execute(f"SELECT max(ts)::date FROM {table};")
        row = cur.fetchone()
    if row and row[0]:
        v = row[0]
        return v.date() if isinstance(v, dt.datetime) else v
    return None


def numeric_columns(conn, table: str) -> set[str]:
    schema, name = table.split(".")
    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_schema = %s AND table_name = %s "
            "AND data_type IN ('bigint', 'integer', 'double precision', 'numeric', 'real')",
            (schema, name),
        )
        return {r[0] for r in cur.fetchall()}


def upsert(conn, df: pd.DataFrame, table: str) -> int:
    if df.empty:
        return 0

    # Older files carry text placeholders (".") in some numeric columns; the table's types win.
    for col in numeric_columns(conn, table) & set(df.columns):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    cols = list(df.columns)
    col_list = ", ".join(f'"{c}"' for c in cols)
    placeholders = ", ".join(["%s"] * len(cols))
    update_cols = [c for c in cols if c not in ("ts", "cftc_contract_market_code")]
    update_set = ", ".join(f'"{c}" = EXCLUDED."{c}"' for c in update_cols)

    sql = f"""
    INSERT INTO {table} ({col_list}) VALUES ({placeholders})
    ON CONFLICT (cftc_contract_market_code, ts) DO UPDATE SET {update_set};
    """

    rows = [
        tuple(None if pd.isna(v) else v for v in row)
        for row in df.to_numpy(dtype=object)
    ]

    with conn.cursor() as cur:
        cur.executemany(sql, rows)
    conn.commit()
    return len(rows)


# ---------------------------------------------------------------------------
# Download + normalize
# ---------------------------------------------------------------------------

def _read_zip(url: str) -> pd.DataFrame:
    resp = requests.get(url, timeout=180)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        txt = [n for n in zf.namelist() if n.endswith(".txt")]
        if not txt:
            return pd.DataFrame()
        with zf.open(txt[0]) as f:
            return pd.read_csv(f, low_memory=False, on_bad_lines="skip", dtype={CODE_COLUMN: str})


def download_current_week(report: dict, header_cols: list[str] | None = None) -> pd.DataFrame:
    resp = requests.get(report["current"], timeout=60)
    resp.raise_for_status()
    if header_cols:
        return pd.read_csv(
            io.StringIO(resp.text), header=None, names=header_cols,
            low_memory=False, on_bad_lines="skip", dtype={CODE_COLUMN: str},
        )
    return pd.read_csv(io.StringIO(resp.text), low_memory=False, on_bad_lines="skip", dtype={CODE_COLUMN: str})


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal cleanup: clean col names, parse date, add short_name."""
    # Clean column names: strip whitespace, lowercase, underscores
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Parse the report date → ts
    date_col = [c for c in df.columns if "report_date" in c and "yyyy" in c]
    if date_col:
        df["ts"] = pd.to_datetime(df[date_col[0]], format="mixed", dayfirst=False).dt.normalize()
        df = df.drop(columns=date_col)
        df = df.dropna(subset=["ts"])

    # Clean contract code and add short_name
    if "cftc_contract_market_code" in df.columns:
        df["cftc_contract_market_code"] = df["cftc_contract_market_code"].astype(str).str.strip()
        df["short_name"] = df["cftc_contract_market_code"].map(SHORT_NAMES)

    # Strip whitespace from string columns, leaving missing values missing (not the text "nan")
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].where(df[col].isna(), df[col].astype(str).str.strip())

    return df.drop_duplicates(subset=["cftc_contract_market_code", "ts"], keep="last")


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def fetch_data(report: dict, start_year: int = FIRST_YEAR) -> pd.DataFrame:
    current_year = dt.date.today().year
    dfs = []

    if start_year <= ARCHIVE_LAST_YEAR:
        print("  2006-2016 archive")
        dfs.append(_read_zip(report["archive"]))

    for year in tqdm(range(max(start_year, ARCHIVE_LAST_YEAR + 1), current_year + 1), desc="CFTC years"):
        try:
            df = _read_zip(report["yearly"].format(year=year))
            if not df.empty:
                dfs.append(df)
        except requests.HTTPError:
            if year != current_year:
                print(f"  Warning: failed to download {year}", file=sys.stderr)

    header_cols = next((list(d.columns) for d in dfs if not d.empty), None)
    try:
        dfs.append(download_current_week(report, header_cols))
    except Exception:
        pass

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def load_history(report: dict, truncate: bool) -> None:
    """All history from 2006: upsert (backfill) or truncate and reload (full)."""
    table = report["table"]
    print(f"Downloading all CFTC TFF history for {table}...")
    raw = fetch_data(report)
    if raw.empty:
        print("No data returned from CFTC.", file=sys.stderr)
        sys.exit(1)

    df = normalize(raw)
    print(f"  Parsed {len(df):,} rows, {len(df.columns)} columns, {df['ts'].min():%Y-%m-%d} to {df['ts'].max():%Y-%m-%d}")

    conn = get_conn()
    ensure_table(conn, df, table)
    if truncate:
        with conn.cursor() as cur:
            cur.execute(f"TRUNCATE TABLE {table};")
        conn.commit()

    inserted = upsert(conn, df, table)
    print(f"✅ Upserted {inserted:,} rows into {table}")
    conn.close()


def incremental_update(report: dict) -> None:
    table = report["table"]
    conn = get_conn()
    max_date = get_max_date(conn, table)

    if max_date is None:
        print(f"No existing data in {table}. Loading all history.")
        conn.close()
        load_history(report, truncate=False)
        return

    print(f"Fetching CFTC data for {table} since {max_date}...")
    raw = fetch_data(report, start_year=max_date.year)
    if raw.empty:
        print("✅ No new data from CFTC. Up to date.")
        conn.close()
        return

    df = normalize(raw)
    df = df[df["ts"] > pd.Timestamp(max_date)]

    if df.empty:
        print("✅ Up to date.")
        conn.close()
        return

    print(f"  {len(df):,} new rows")
    inserted = upsert(conn, df, table)
    print(f"✅ Upserted {inserted:,} rows into {table}")
    conn.close()


def main() -> None:
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "incremental"
    if mode not in {"incremental", "backfill", "full", "all", "refresh"}:
        sys.exit(f"Unknown mode {mode!r}: use incremental, backfill or full.")
    for report in REPORTS.values():
        if mode == "incremental":
            incremental_update(report)
        else:
            load_history(report, truncate=mode != "backfill")


if __name__ == "__main__":
    main()
