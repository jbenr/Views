#!/usr/bin/env python3
# pull_ust_cusips.py
import os
import time
import json
import sys
from typing import List, Dict, Any

import requests
import pandas as pd
import duckdb

TD_URL = "https://www.treasurydirect.gov/TA_WS/securities/auctioned"

# Ensure /data exists at the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

DB_PATH = os.path.join(DATA_DIR, "ust.db")
TABLE = "auctioned_securities"


def fetch_all(timeout=60, retries=5, backoff=2.0) -> List[Dict[str, Any]]:
    """Download full auctioned-securities payload with simple retries."""
    last_err = None
    for i in range(retries):
        try:
            r = requests.get(TD_URL, timeout=timeout, headers={"Accept": "application/json"})
            r.raise_for_status()
            return r.json() if r.headers.get("Content-Type", "").startswith("application/json") else json.loads(r.text)
        except Exception as e:
            last_err = e
            time.sleep(backoff**i)
    raise RuntimeError(f"Failed to fetch data after {retries} attempts: {last_err}")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    for col in ("issue_date", "auction_date", "announcement_date", "maturity_date", "dated_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in ("interest_rate", "refcpi_on_issue_date"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def main():
    data = fetch_all()
    if not data:
        print("No data returned.", file=sys.stderr)
        sys.exit(1)

    df = normalize_columns(pd.DataFrame(data))

    # Deduplicate
    if {"cusip", "issue_date"}.issubset(df.columns):
        df = df.sort_values(["cusip", "issue_date"]).drop_duplicates(["cusip", "issue_date"], keep="last")
    elif "cusip" in df.columns:
        df = df.sort_values(["cusip"]).drop_duplicates(["cusip"], keep="last")

    # Connect to /data/ust.db at the project root
    con = duckdb.connect(DB_PATH)
    con.execute(f"CREATE TABLE IF NOT EXISTS {TABLE} AS SELECT * FROM df LIMIT 0")
    con.execute(f"DELETE FROM {TABLE}")
    con.register("df", df)
    con.execute(f"INSERT INTO {TABLE} SELECT * FROM df")

    if "cusip" in df.columns:
        con.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_cusip ON {TABLE}(cusip)")
    if "issue_date" in df.columns:
        con.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_issue_date ON {TABLE}(issue_date)")

    n = con.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
    print(f"✅ Loaded {n} rows into {DB_PATH}:{TABLE}")


if __name__ == "__main__":
    main()
