#!/usr/bin/env python3
"""
Build md.breakeven table from md.headline.

Calculates inflation breakeven levels (TIPS yield - nominal yield) for
matching tenors: 5Y, 10Y, 20Y, 30Y.

Usage:
    python build_breakeven.py          # incremental (append new dates)
    python build_breakeven.py --rebuild # full rebuild from scratch
"""

from __future__ import annotations

import argparse
import os

import psycopg

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DB_DSN = os.getenv("DB_DSN", "postgresql://benjils:snickers@raptor:5432/markets")

# Tenors that have both nominals and TIPS
BREAKEVEN_TENORS = ("5-Year", "10-Year", "20-Year", "30-Year")


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


def get_conn():
    return psycopg.connect(DB_DSN)


def ensure_breakeven_table(conn) -> None:
    """Create md.breakeven if it doesn't exist."""
    with conn.cursor() as cur:
        cur.execute("CREATE SCHEMA IF NOT EXISTS md;")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS md.breakeven (
                ts              DATE NOT NULL,
                tenor           TEXT NOT NULL,
                status          TEXT NOT NULL,
                nominal_cusip   TEXT,
                tips_cusip      TEXT,
                nominal_yield   DOUBLE PRECISION,
                tips_yield      DOUBLE PRECISION,
                breakeven       DOUBLE PRECISION,
                PRIMARY KEY (ts, tenor, status)
            );
            """
        )
        # Index for faster lookups by date
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_breakeven_ts 
            ON md.breakeven (ts);
            """
        )
    conn.commit()


# ---------------------------------------------------------------------------
# Core SQL for breakeven generation
# ---------------------------------------------------------------------------

BREAKEVEN_SQL = """
WITH tips_first_issue AS (
    -- Get first auction date for each TIPS cusip (ignore reopenings)
    SELECT 
        cusip,
        original_security_term AS tenor,
        MIN(auction_date) AS first_auction_date
    FROM auctioned_securities
    WHERE inflation_index_security = 'Yes'
      AND security_type IN ('Note', 'Bond')
      AND original_security_term = ANY(%s)
    GROUP BY cusip, original_security_term
),
tips_nominal_pairs AS (
    -- For each TIPS, find the nominal that was current on the TIPS first auction date
    SELECT 
        t.cusip AS tips_cusip,
        t.tenor,
        t.first_auction_date,
        h.cusip AS nominal_cusip
    FROM tips_first_issue t
    JOIN md.headline h 
        ON h.ts = t.first_auction_date::date
        AND h.tenor = t.tenor
        AND h.asset_class = 'nominal'
        AND h.status = 'c'
)
SELECT 
    ht.ts,
    p.tenor,
    ht.status,
    p.nominal_cusip,
    p.tips_cusip,
    en.yld_ytm_mid AS nominal_yield,
    ht.yld_ytm_mid AS tips_yield,
    en.yld_ytm_mid - ht.yld_ytm_mid AS breakeven
FROM tips_nominal_pairs p
JOIN md.headline ht 
    ON ht.cusip = p.tips_cusip
    AND ht.asset_class = 'tips'
    AND ht.status IN ('c', 'o', 'oo')
JOIN md.ust_eod en 
    ON en.cusip = p.nominal_cusip 
    AND en.ts = ht.ts
WHERE en.yld_ytm_mid IS NOT NULL
  {date_filter}
"""


# ---------------------------------------------------------------------------
# Build functions
# ---------------------------------------------------------------------------


def get_max_breakeven_date(conn):
    """Get the max date currently in md.breakeven, or None if empty."""
    with conn.cursor() as cur:
        cur.execute("SELECT MAX(ts) FROM md.breakeven;")
        row = cur.fetchone()
    return row[0] if row and row[0] else None


def get_max_headline_date(conn):
    """Get the max date in md.headline."""
    with conn.cursor() as cur:
        cur.execute("SELECT MAX(ts) FROM md.headline;")
        row = cur.fetchone()
    return row[0] if row and row[0] else None


def rebuild_breakeven(conn) -> int:
    """Truncate and rebuild md.breakeven from scratch."""
    sql = BREAKEVEN_SQL.format(date_filter="")

    with conn.cursor() as cur:
        cur.execute("TRUNCATE md.breakeven;")
        cur.execute(
            f"""
            INSERT INTO md.breakeven 
                (ts, tenor, status, nominal_cusip, tips_cusip, nominal_yield, tips_yield, breakeven) 
            {sql}
            """,
            (list(BREAKEVEN_TENORS),),
        )
        row_count = cur.rowcount
    conn.commit()
    return row_count


def incremental_breakeven(conn) -> int:
    """Append only new dates to md.breakeven."""
    max_breakeven = get_max_breakeven_date(conn)
    max_headline = get_max_headline_date(conn)

    if max_headline is None:
        print("No data in md.headline. Run build_headline.py first.")
        return 0

    if max_breakeven is None:
        # Table is empty, do full rebuild
        print("md.breakeven is empty. Running full rebuild...")
        return rebuild_breakeven(conn)

    if max_breakeven >= max_headline:
        print(f"md.breakeven is up to date (max date: {max_breakeven}).")
        return 0

    # Only process dates after max_breakeven
    date_filter = "AND ht.ts > %s"
    sql = BREAKEVEN_SQL.format(date_filter=date_filter)

    with conn.cursor() as cur:
        cur.execute(
            f"""
            INSERT INTO md.breakeven 
                (ts, tenor, status, nominal_cusip, tips_cusip, nominal_yield, tips_yield, breakeven) 
            {sql}
            """,
            (list(BREAKEVEN_TENORS), max_breakeven),
        )
        row_count = cur.rowcount
    conn.commit()
    return row_count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Build md.breakeven table")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Full rebuild (truncate and repopulate)",
    )
    args = parser.parse_args()

    print(f"Connecting to Postgres: {DB_DSN}")
    with get_conn() as conn:
        ensure_breakeven_table(conn)

        if args.rebuild:
            print("Rebuilding md.breakeven from scratch...")
            rows = rebuild_breakeven(conn)
            print(f"✅ Inserted {rows:,} rows into md.breakeven")
        else:
            print("Running incremental update...")
            rows = incremental_breakeven(conn)
            if rows > 0:
                print(f"✅ Inserted {rows:,} new rows into md.breakeven")


if __name__ == "__main__":
    main()