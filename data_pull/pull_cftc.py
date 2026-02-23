#!/usr/bin/env python3
"""
Pull CFTC Traders in Financial Futures (TFF) positioning data into TimescaleDB.

- Downloads from CFTC.gov (historical zips + current week text file)
- Stores into md.cftc (Timescale hypertable)
- Incremental: each run only fetches weeks after max(ts) already stored
- No Bloomberg dependency

Usage:
    python pull_cftc.py              # incremental update
    python pull_cftc.py full         # full historical refresh from 2006
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

DB_DSN = os.getenv("DB_DSN", "postgresql://benjils:snickers@localhost:5432/markets")

BASE_URL = "https://www.cftc.gov"
HISTORICAL_URL = BASE_URL + "/files/dea/history/fut_fin_txt_{year}.zip"
CURRENT_URL = BASE_URL + "/dea/newcot/FinFutWk.txt"
FIRST_YEAR = 2006

# Short names for contracts we care about most (by CFTC_Contract_Market_Code)
SHORT_NAMES = {
    "042601": "UST_2Y",   "044601": "UST_5Y",   "043602": "UST_10Y",
    "020601": "UST_BOND",  "020604": "ULTRA_BOND", "043607": "ULTRA_10Y",
    "134741": "SOFR_3M",   "134742": "SOFR_1M",
    "13874A": "ES",        "13874+": "SP500",
    "209742": "NQ",        "20974+": "NASDAQ",
    "1170E1": "VIX",       "098662": "DXY",       "045601": "FF",
}


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


def ensure_table(conn, df: pd.DataFrame) -> None:
    """Create md.cftc from DataFrame columns, plus hypertable + indexes."""
    cols = [f'"{c}" {_pg_type(df[c].dtype)}' for c in df.columns]
    with conn.cursor() as cur:
        cur.execute("CREATE SCHEMA IF NOT EXISTS md;")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS md.cftc (
                {', '.join(cols)},
                created_at timestamptz NOT NULL DEFAULT now(),
                PRIMARY KEY (cftc_contract_market_code, ts)
            );
        """)
        cur.execute("SELECT create_hypertable('md.cftc', 'ts', if_not_exists => TRUE);")
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_cftc_short_name "
            "ON md.cftc (short_name) WHERE short_name IS NOT NULL;"
        )
    conn.commit()


def get_max_date(conn) -> dt.date | None:
    with conn.cursor() as cur:
        cur.execute("SELECT max(ts)::date FROM md.cftc;")
        row = cur.fetchone()
    if row and row[0]:
        v = row[0]
        return v.date() if isinstance(v, dt.datetime) else v
    return None


def upsert(conn, df: pd.DataFrame) -> int:
    if df.empty:
        return 0

    cols = list(df.columns)
    col_list = ", ".join(f'"{c}"' for c in cols)
    placeholders = ", ".join(["%s"] * len(cols))
    update_cols = [c for c in cols if c not in ("ts", "cftc_contract_market_code")]
    update_set = ", ".join(f'"{c}" = EXCLUDED."{c}"' for c in update_cols)

    sql = f"""
    INSERT INTO md.cftc ({col_list}) VALUES ({placeholders})
    ON CONFLICT (cftc_contract_market_code, ts) DO UPDATE SET {update_set};
    """

    rows = [
        tuple(None if pd.isna(v) else v for v in row)
        for row in df.to_numpy()
    ]

    with conn.cursor() as cur:
        cur.executemany(sql, rows)
    conn.commit()
    return len(rows)


# ---------------------------------------------------------------------------
# Download + normalize
# ---------------------------------------------------------------------------

def download_year(year: int) -> pd.DataFrame:
    url = HISTORICAL_URL.format(year=year)
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        txt = [n for n in zf.namelist() if n.endswith(".txt")]
        if not txt:
            return pd.DataFrame()
        with zf.open(txt[0]) as f:
            return pd.read_csv(f, low_memory=False, on_bad_lines="skip")


def download_current_week(header_cols: list[str] | None = None) -> pd.DataFrame:
    resp = requests.get(CURRENT_URL, timeout=60)
    resp.raise_for_status()
    if header_cols:
        return pd.read_csv(
            io.StringIO(resp.text), header=None, names=header_cols,
            low_memory=False, on_bad_lines="skip",
        )
    return pd.read_csv(io.StringIO(resp.text), low_memory=False, on_bad_lines="skip")


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal cleanup: clean col names, parse date, add short_name."""
    # Clean column names: strip whitespace, lowercase, underscores
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Parse the report date → ts
    date_col = [c for c in df.columns if "report_date" in c and "yyyy" in c]
    if date_col:
        df["ts"] = pd.to_datetime(df[date_col[0]], format="mixed", dayfirst=False)
        df = df.drop(columns=date_col)
        df = df.dropna(subset=["ts"])

    # Clean contract code and add short_name
    if "cftc_contract_market_code" in df.columns:
        df["cftc_contract_market_code"] = df["cftc_contract_market_code"].astype(str).str.strip()
        df["short_name"] = df["cftc_contract_market_code"].map(SHORT_NAMES)

    # Strip whitespace from string columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    return df


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def fetch_data(start_year: int = FIRST_YEAR) -> pd.DataFrame:
    current_year = dt.date.today().year
    dfs = []
    header_cols = None

    for year in tqdm(range(start_year, current_year + 1), desc="CFTC years"):
        try:
            df = download_year(year)
            if not df.empty:
                if header_cols is None:
                    header_cols = list(df.columns)
                dfs.append(df)
        except requests.HTTPError:
            if year == current_year:
                pass
            else:
                print(f"  Warning: failed to download {year}", file=sys.stderr)

    try:
        dfs.append(download_current_week(header_cols))
    except Exception:
        pass

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def full_refresh() -> None:
    print("Downloading all CFTC TFF history...")
    raw = fetch_data()
    if raw.empty:
        print("No data returned from CFTC.", file=sys.stderr)
        sys.exit(1)

    df = normalize(raw)
    print(f"  Parsed {len(df):,} rows, {len(df.columns)} columns")

    conn = get_conn()
    ensure_table(conn, df)
    with conn.cursor() as cur:
        cur.execute("TRUNCATE TABLE md.cftc;")
    conn.commit()

    inserted = upsert(conn, df)
    print(f"✅ Upserted {inserted:,} rows into md.cftc")
    conn.close()


def incremental_update() -> None:
    conn = get_conn()

    try:
        max_date = get_max_date(conn)
    except Exception:
        max_date = None

    if max_date is None:
        print("No existing data. Running full refresh.")
        conn.close()
        full_refresh()
        return

    print(f"Fetching CFTC data since {max_date}...")
    raw = fetch_data(start_year=max_date.year)
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
    inserted = upsert(conn, df)
    print(f"✅ Upserted {inserted:,} rows into md.cftc")
    conn.close()


def main() -> None:
    mode = "incremental"
    if len(sys.argv) > 1 and sys.argv[1].lower() in {"full", "all", "refresh"}:
        mode = "full"
    (full_refresh if mode == "full" else incremental_update)()


if __name__ == "__main__":
    main()
