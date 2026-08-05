#!/usr/bin/env python3
"""
Pull daily rates futures history from Bloomberg into TimescaleDB.

- Pulls from a configurable list of tickers (generic or specific contracts)
- Uses Bloomberg BDH via Bbg helper
- Stores into md.fut_eod (Timescale hypertable), keyed by (contract, ts)
- Also populates sec.fut_contracts with contract metadata
- Incremental: each generic resumes from its own max(ts), with a short overlap

Usage:
    python pull_fut_eod.py                         # incremental update
    python pull_fut_eod.py --backfill 2020-01-01  # non-destructive backfill
    python pull_fut_eod.py full                    # full historical refresh from 2000
"""

from __future__ import annotations

import os
import sys
import datetime as dt
from collections import defaultdict
from typing import List

import pandas as pd
import psycopg
from tqdm import tqdm

from berg import Bbg

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DB_DSN = os.getenv("DB_DSN", "postgresql://benjils:snickers@raptor:5432/markets")

# Bloomberg fields for EOD data
BBG_EOD_FIELDS = [
    "PX_LAST",
    "VOLUME",
    "OPEN_INT",
    "FUT_CUR_GEN_TICKER",  # Actual contract ticker (e.g., "TYH6") for each date
]

# Bloomberg fields for contract metadata
BBG_CONTRACT_FIELDS = [
    "FUT_NOTICE_FIRST",
    "LAST_TRADEABLE_DT",
    "FUT_DLV_DT_FIRST",
    "FUT_DLV_DT_LAST",
    "FUT_CONTRACT_DT",
    "PARSEKYABLE_DES",  # Returns actual contract ticker like "TUH6 Comdty"
]

BATCH_SIZE = 5
HISTORICAL_START = dt.date(2000, 1, 1)
INCREMENTAL_OVERLAP_DAYS = 7

# Bloomberg generics discover the active contracts; stored rows retain the
# specific contract returned by FUT_CUR_GEN_TICKER.
GENERIC_RANKS = {
    "TU": range(1, 3),
    "FV": range(1, 3),
    "TY": range(1, 3),
    "UXY": range(1, 3),
    "US": range(1, 3),
    "WN": range(1, 3),
    "FF": range(1, 9),
    "SER": range(1, 9),
    "SFR": range(1, 9),
}
TICKERS = [
    f"{root}{rank} Comdty"
    for root, ranks in GENERIC_RANKS.items()
    for rank in ranks
]


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def get_conn():
    return psycopg.connect(DB_DSN)


def ensure_tables(conn) -> None:
    """Create md.fut_eod and sec.fut_contracts tables if they don't exist."""
    with conn.cursor() as cur:
        cur.execute("CREATE SCHEMA IF NOT EXISTS md;")
        cur.execute("CREATE SCHEMA IF NOT EXISTS sec;")

        # Contract metadata table (security master data)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS sec.fut_contracts (
                contract            text PRIMARY KEY,
                generic_ticker      text NOT NULL,
                contract_month      date,
                first_notice_date   date,
                last_trade_date     date,
                first_delivery_date date,
                last_delivery_date  date,
                created_at          timestamptz NOT NULL DEFAULT now(),
                updated_at          timestamptz NOT NULL DEFAULT now()
            );
            """
        )

        # EOD data table
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS md.fut_eod (
                ts              date NOT NULL,
                generic_ticker  text NOT NULL,
                contract        text NOT NULL,
                px_last         double precision,
                volume          double precision,
                open_interest   double precision,
                source          text NOT NULL DEFAULT 'BGN',
                created_at      timestamptz NOT NULL DEFAULT now(),
                updated_at      timestamptz NOT NULL DEFAULT now(),
                PRIMARY KEY (contract, ts)
            );
            """
        )

        # Create hypertable
        cur.execute(
            "SELECT create_hypertable('md.fut_eod', 'ts', if_not_exists => TRUE);"
        )

        # Index on generic_ticker for fast lookups
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS ix_fut_eod_generic_ticker 
            ON md.fut_eod (generic_ticker, ts);
            """
        )

    conn.commit()


def normalize_generic_ticker(ticker: str) -> str:
    """Return the stored generic name (for example, ``SFR1``)."""
    return ticker.replace(" Comdty", "").strip()


def get_grouped_pulls(
    conn,
    tickers: list[str],
    force_start: dt.date | None = None,
) -> dict[dt.date, list[str]]:
    """Group generics by pull start, using a separate watermark per series."""
    if force_start is not None:
        return {force_start: list(tickers)}

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT generic_ticker, MAX(ts)::date
            FROM md.fut_eod
            GROUP BY generic_ticker
            """
        )
        rows = cur.fetchall()

    db_maxes: dict[str, dt.date] = {}
    for ticker, max_ts in rows:
        if max_ts is None:
            continue
        key = normalize_generic_ticker(ticker)
        value = max_ts.date() if isinstance(max_ts, dt.datetime) else max_ts
        if key not in db_maxes or value > db_maxes[key]:
            db_maxes[key] = value

    today = dt.date.today()
    overlap = dt.timedelta(days=INCREMENTAL_OVERLAP_DAYS)
    groups: dict[dt.date, list[str]] = defaultdict(list)
    for ticker in tickers:
        last_ts = db_maxes.get(normalize_generic_ticker(ticker))
        start = HISTORICAL_START if last_ts is None else max(HISTORICAL_START, last_ts - overlap)
        if start <= today:
            groups[start].append(ticker)

    return dict(groups)


def parse_contract_ticker(ticker: str) -> tuple[str, str]:
    """
    Parse a Bloomberg futures ticker into (generic_root, contract).
    
    Examples:
        'TYH6 Comdty' -> ('TY', 'TYH6')
        'TY1 Comdty'  -> ('TY', 'TY1')
        'UXYH6 Comdty' -> ('UXY', 'UXYH6')
        'WN1 Comdty' -> ('WN', 'WN1')
    """
    # Known rates futures roots
    KNOWN_ROOTS = ["SFR", "SER", "UXY", "WN", "TU", "FV", "TY", "US", "UB", "Z3N", "FF"]
    
    clean = ticker.replace(" Comdty", "").strip()
    
    # Check known roots first (longest match)
    for root in sorted(KNOWN_ROOTS, key=len, reverse=True):
        if clean.startswith(root):
            return (root, clean)
    
    # Fallback: find where month code or digit starts
    # Month codes: F,G,H,J,K,M,N,Q,U,V,X,Z
    for i, char in enumerate(clean):
        if char.isdigit():
            return (clean[:i], clean)
    
    return (clean, clean)


# ---------------------------------------------------------------------------
# Bloomberg fetch
# ---------------------------------------------------------------------------

def fetch_contract_metadata(bbg: Bbg, tickers: List[str]) -> pd.DataFrame:
    """Fetch contract metadata using BDP for a list of tickers."""
    if not tickers:
        return pd.DataFrame()

    data = bbg.bdp(tickers, BBG_CONTRACT_FIELDS)
    
    if data is None or data.empty:
        return pd.DataFrame()

    records = []
    for ticker in tickers:
        if ticker not in data.index:
            continue
            
        row = data.loc[ticker]
        
        # Get actual contract ticker from PARSEKYABLE_DES (e.g., "TUH6 Comdty" -> "TUH6")
        actual_contract = None
        if pd.notna(row.get("PARSEKYABLE_DES")):
            actual_contract = str(row["PARSEKYABLE_DES"]).replace(" Comdty", "").strip()
        
        # If we couldn't get actual contract, skip
        if not actual_contract:
            print(f"Warning: Could not get actual contract for {ticker}, skipping metadata")
            continue
        
        # Get the root (TY, FV, etc.) from the actual contract
        generic_root, _ = parse_contract_ticker(actual_contract + " Comdty")
        
        # Parse contract month from FUT_CONTRACT_DT (YYYYMMDD)
        contract_month = None
        if pd.notna(row.get("FUT_CONTRACT_DT")):
            try:
                dt_int = int(row["FUT_CONTRACT_DT"])
                contract_month = dt.date(dt_int // 10000, (dt_int // 100) % 100, 1)
            except (ValueError, TypeError):
                pass
        
        # Fallback: derive contract_month from first_delivery_date
        if contract_month is None and pd.notna(row.get("FUT_DLV_DT_FIRST")):
            try:
                dlv_date = pd.to_datetime(row["FUT_DLV_DT_FIRST"]).date()
                contract_month = dt.date(dlv_date.year, dlv_date.month, 1)
            except (ValueError, TypeError):
                pass
        
        records.append({
            "contract": actual_contract,
            "generic_ticker": generic_root,
            "contract_month": contract_month,
            "first_notice_date": row.get("FUT_NOTICE_FIRST"),
            "last_trade_date": row.get("LAST_TRADEABLE_DT"),
            "first_delivery_date": row.get("FUT_DLV_DT_FIRST"),
            "last_delivery_date": row.get("FUT_DLV_DT_LAST"),
        })

    return pd.DataFrame(records)


def upsert_contract_metadata(conn, df: pd.DataFrame) -> int:
    """Insert/update contract metadata."""
    if df.empty:
        return 0

    rows = [
        (
            row.contract,
            row.generic_ticker,
            row.contract_month,
            row.first_notice_date,
            row.last_trade_date,
            row.first_delivery_date,
            row.last_delivery_date,
        )
        for row in df.itertuples(index=False)
    ]

    sql = """
    INSERT INTO sec.fut_contracts
      (contract, generic_ticker, contract_month, first_notice_date, 
       last_trade_date, first_delivery_date, last_delivery_date, updated_at)
    VALUES (%s, %s, %s, %s, %s, %s, %s, now())
    ON CONFLICT (contract) DO UPDATE SET
      generic_ticker      = EXCLUDED.generic_ticker,
      contract_month      = EXCLUDED.contract_month,
      first_notice_date   = EXCLUDED.first_notice_date,
      last_trade_date     = EXCLUDED.last_trade_date,
      first_delivery_date = EXCLUDED.first_delivery_date,
      last_delivery_date  = EXCLUDED.last_delivery_date,
      updated_at          = now();
    """

    with conn.cursor() as cur:
        cur.executemany(sql, rows)
    conn.commit()
    return len(rows)


def fetch_eod_for_tickers(
    bbg: Bbg,
    conn,
    tickers: List[str],
    start: dt.date,
    end: dt.date,
) -> int:
    """
    Call BDH in batches and upsert into md.fut_eod.
    Returns total rows upserted.
    """
    expected_cols = [
        "ts",
        "generic_ticker",
        "contract",
        "px_last",
        "volume",
        "open_interest",
        "source",
    ]

    total_inserted = 0
    n_batches = (len(tickers) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in tqdm(
        range(0, len(tickers), BATCH_SIZE),
        total=n_batches,
        desc="Bloomberg BDH batches",
        unit="batch",
    ):
        batch_tickers = tickers[i : i + BATCH_SIZE]

        data = bbg.bdh(
            batch_tickers,
            BBG_EOD_FIELDS,
            start=start,
            end=end,
            periodicity="DAILY",
        )

        batch_dfs: list[pd.DataFrame] = []

        for ticker, df in data.items():
            if df is None or df.empty:
                continue
            if "error" in df.columns:
                continue

            # Keep the full generic ticker (e.g., TY1, TY2) to track front/back month
            generic_ticker = ticker.replace(" Comdty", "").strip()

            df = df.rename(
                columns={
                    "PX_LAST": "px_last",
                    "VOLUME": "volume",
                    "OPEN_INT": "open_interest",
                    "FUT_CUR_GEN_TICKER": "contract",
                }
            )

            df = df.reset_index().rename(columns={"date": "ts"})
            df["generic_ticker"] = generic_ticker
            
            # If FUT_CUR_GEN_TICKER wasn't returned, fall back to the input ticker
            if "contract" not in df.columns or df["contract"].isna().all():
                df["contract"] = generic_ticker
            
            df["source"] = "BGN"

            for col in expected_cols:
                if col not in df.columns:
                    df[col] = pd.NA

            batch_dfs.append(df[expected_cols])

        if not batch_dfs:
            continue

        batch_df = pd.concat(batch_dfs, ignore_index=True)
        inserted = upsert_fut_eod(conn, batch_df)
        total_inserted += inserted

    return total_inserted


def upsert_fut_eod(conn, df: pd.DataFrame) -> int:
    """Insert/update rows into md.fut_eod."""
    if df.empty:
        return 0

    rows = [
        (
            row.ts,
            row.generic_ticker,
            row.contract,
            row.px_last,
            row.volume,
            row.open_interest,
            row.source,
        )
        for row in df.itertuples(index=False)
    ]

    sql = """
    INSERT INTO md.fut_eod
      (ts, generic_ticker, contract, px_last, volume, open_interest, source)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (contract, ts) DO UPDATE SET
      generic_ticker = EXCLUDED.generic_ticker,
      px_last        = EXCLUDED.px_last,
      volume         = EXCLUDED.volume,
      open_interest  = EXCLUDED.open_interest,
      source         = EXCLUDED.source;
    """

    with conn.cursor() as cur:
        cur.executemany(sql, rows)
    conn.commit()
    return len(rows)


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def populate_contracts_from_eod(conn, bbg: Bbg) -> int:
    """
    Populate sec.fut_contracts from distinct contracts found in md.fut_eod.
    Fetches metadata from Bloomberg for each actual contract.
    """
    # Fetch metadata only for contracts not yet present in the security master.
    with conn.cursor() as cur:
        cur.execute("""
            SELECT DISTINCT e.contract
            FROM md.fut_eod e
            LEFT JOIN sec.fut_contracts c ON c.contract = e.contract
            WHERE e.contract IS NOT NULL
              AND c.contract IS NULL
            ORDER BY e.contract;
        """)
        contracts = [row[0] for row in cur.fetchall()]
    
    if not contracts:
        return 0
    
    print(f"Found {len(contracts)} distinct contracts in md.fut_eod, fetching metadata...")
    
    def to_date(val):
        """Convert Bloomberg date value to Python date."""
        if val is None or pd.isna(val):
            return None
        try:
            # Handle pandas Timestamp
            if isinstance(val, pd.Timestamp):
                return val.date()
            # Handle datetime
            if isinstance(val, dt.datetime):
                return val.date()
            # Handle date
            if isinstance(val, dt.date):
                return val
            # Handle numeric (Excel-style date or epoch)
            if isinstance(val, (int, float)):
                return pd.to_datetime(val).date()
            # Handle string
            return pd.to_datetime(val).date()
        except (ValueError, TypeError):
            return None
    
    # Fetch metadata in batches
    records = []
    for i in tqdm(range(0, len(contracts), BATCH_SIZE), desc="Fetching contract metadata"):
        batch = contracts[i:i + BATCH_SIZE]
        tickers = [f"{c} Comdty" for c in batch]
        
        try:
            data = bbg.bdp(tickers, BBG_CONTRACT_FIELDS)
        except Exception as e:
            print(f"Warning: BDP failed for batch {batch}: {e}")
            continue
        
        if data is None or data.empty:
            continue
        
        for ticker in tickers:
            if ticker not in data.index:
                continue
            
            row = data.loc[ticker]
            contract = ticker.replace(" Comdty", "").strip()
            generic_root, _ = parse_contract_ticker(ticker)
            
            # Parse contract month from FUT_CONTRACT_DT
            contract_month = None
            if pd.notna(row.get("FUT_CONTRACT_DT")):
                try:
                    dt_int = int(row["FUT_CONTRACT_DT"])
                    contract_month = dt.date(dt_int // 10000, (dt_int // 100) % 100, 1)
                except (ValueError, TypeError):
                    pass
            
            # Fallback: derive from first_delivery_date
            if contract_month is None:
                dlv_date = to_date(row.get("FUT_DLV_DT_FIRST"))
                if dlv_date:
                    contract_month = dt.date(dlv_date.year, dlv_date.month, 1)
            
            records.append({
                "contract": contract,
                "generic_ticker": generic_root,
                "contract_month": contract_month,
                "first_notice_date": to_date(row.get("FUT_NOTICE_FIRST")),
                "last_trade_date": to_date(row.get("LAST_TRADEABLE_DT")),
                "first_delivery_date": to_date(row.get("FUT_DLV_DT_FIRST")),
                "last_delivery_date": to_date(row.get("FUT_DLV_DT_LAST")),
            })
    
    if not records:
        return 0
    
    df = pd.DataFrame(records)
    return upsert_contract_metadata(conn, df)


def full_refresh() -> None:
    """Full historical load from HISTORICAL_START to today."""
    print("📡 Running full historical refresh...")
    
    start = HISTORICAL_START
    end = dt.date.today()
    
    with get_conn() as conn:
        ensure_tables(conn)
        
        # Truncate existing data
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE md.fut_eod;")
            cur.execute("TRUNCATE TABLE sec.fut_contracts;")
        conn.commit()
        
        print(f"Tickers: {TICKERS}")
        print(f"Pulling Bloomberg history from {start} to {end}...")
        
        bbg = Bbg()
        
        # Pull EOD data FIRST
        inserted = fetch_eod_for_tickers(bbg, conn, TICKERS, start, end)
        
        if inserted == 0:
            print("Bloomberg returned no EOD data. Nothing inserted.")
        else:
            print(f"✅ Upserted {inserted} rows into md.fut_eod")
            
            # Now populate sec.fut_contracts from the actual contracts we found
            n_contracts = populate_contracts_from_eod(conn, bbg)
            print(f"✅ Full refresh complete. Upserted {n_contracts} contract records into sec.fut_contracts")


def incremental_update(force_start: dt.date | None = None) -> None:
    """Update each generic from its own watermark, or force a backfill date."""
    with get_conn() as conn:
        ensure_tables(conn)

        groups = get_grouped_pulls(conn, TICKERS, force_start)
        end = dt.date.today()

        if not groups:
            print("✅ No dates to pull.")
            return

        n_tickers = sum(len(group) for group in groups.values())
        print(f"📡 Pulling {n_tickers} futures generics across {len(groups)} date group(s)...")

        bbg = Bbg()

        total_inserted = 0
        for start, group_tickers in sorted(groups.items()):
            print(f"  {len(group_tickers)} tickers from {start} to {end}...")
            total_inserted += fetch_eod_for_tickers(
                bbg, conn, group_tickers, start, end
            )

        if total_inserted == 0:
            print("✅ Bloomberg returned no new EOD data. You're up to date.")
        else:
            print(f"✅ Upserted {total_inserted} rows into md.fut_eod")

            # Update contract metadata for any new contracts
            n_contracts = populate_contracts_from_eod(conn, bbg)
            print(f"✅ Incremental update complete. Upserted {n_contracts} contract records into sec.fut_contracts")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    force_start = None
    if "--backfill" in sys.argv:
        idx = sys.argv.index("--backfill")
        try:
            force_start = dt.datetime.strptime(sys.argv[idx + 1], "%Y-%m-%d").date()
        except (IndexError, ValueError):
            raise SystemExit("--backfill requires a date in YYYY-MM-DD format")

    mode = "incremental"
    if len(sys.argv) > 1 and sys.argv[1].lower() in {"full", "all", "refresh"}:
        mode = "full"
    
    print(f"Connecting to Postgres: {DB_DSN}")
    
    if mode == "full":
        full_refresh()
    else:
        incremental_update(force_start)

if __name__ == "__main__":
    main()
