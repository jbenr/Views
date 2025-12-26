#!/usr/bin/env python3
"""
Build md.headline table from md.ust_eod and auctioned_securities.

Labels each bond as current (c), single-old (o), double-old (oo), etc.
for headline tenors: 2Y, 3Y, 5Y, 7Y, 10Y, 20Y, 30Y nominals + 5Y, 10Y, 20Y, 30Y TIPS.

Usage:
    python build_headline.py          # incremental (append new dates)
    python build_headline.py --rebuild # full rebuild from scratch
"""

from __future__ import annotations

import argparse
import os

import psycopg

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DB_DSN = os.getenv("DB_DSN", "postgresql://benjils:snickers@raptor:5432/markets")

HEADLINE_TENORS = (
    "2-Year",
    "3-Year",
    "5-Year",
    "7-Year",
    "10-Year",
    "20-Year",
    "30-Year",
)


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


def get_conn():
    return psycopg.connect(DB_DSN)


def ensure_headline_table(conn) -> None:
    """Create md.headline if it doesn't exist."""
    with conn.cursor() as cur:
        cur.execute("CREATE SCHEMA IF NOT EXISTS md;")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS md.headline (
                ts              DATE NOT NULL,
                tenor           TEXT NOT NULL,
                asset_class     TEXT NOT NULL,
                cusip           TEXT NOT NULL,
                status          TEXT NOT NULL,
                px_last         DOUBLE PRECISION,
                yld_ytm_mid     DOUBLE PRECISION,
                PRIMARY KEY (ts, tenor, asset_class, status)
            );
            """
        )
        # Index for faster lookups by date
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_headline_ts 
            ON md.headline (ts);
            """
        )
        # Index for cusip lookups
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_headline_cusip 
            ON md.headline (cusip);
            """
        )
    conn.commit()


# ---------------------------------------------------------------------------
# Core SQL for headline generation
# ---------------------------------------------------------------------------

HEADLINE_SQL = """
WITH first_auction AS (
    SELECT 
        cusip, 
        MIN(auction_date) AS auction_date,
        MAX(maturity_date) AS maturity_date,
        original_security_term,
        inflation_index_security
    FROM auctioned_securities
    WHERE security_type IN ('Note', 'Bond')
      AND original_security_term IN %(tenors)s
    GROUP BY cusip, original_security_term, inflation_index_security
),
ranked AS (
    SELECT 
        d.ts,
        d.cusip,
        d.px_last,
        d.yld_ytm_mid,
        c.original_security_term AS tenor,
        CASE WHEN c.inflation_index_security = 'Yes' THEN 'tips' ELSE 'nominal' END AS asset_class,
        ROW_NUMBER() OVER (
            PARTITION BY d.ts, c.original_security_term, c.inflation_index_security
            ORDER BY c.auction_date DESC
        ) AS rank
    FROM md.ust_eod d
    JOIN first_auction c ON d.cusip = c.cusip
    WHERE d.ts >= c.auction_date
      AND d.ts < c.maturity_date
      {date_filter}
)
SELECT 
    ts, tenor, asset_class, cusip,
    CASE rank
        WHEN 1 THEN 'c'
        WHEN 2 THEN 'o'
        WHEN 3 THEN 'oo'
        WHEN 4 THEN 'ooo'
        WHEN 5 THEN 'oooo'
    END AS status,
    px_last, 
    yld_ytm_mid
FROM ranked
WHERE rank <= 5
"""


# ---------------------------------------------------------------------------
# Build functions
# ---------------------------------------------------------------------------


def get_max_headline_date(conn):
    """Get the max date currently in md.headline, or None if empty."""
    with conn.cursor() as cur:
        cur.execute("SELECT MAX(ts) FROM md.headline;")
        row = cur.fetchone()
    return row[0] if row and row[0] else None


def get_max_eod_date(conn):
    """Get the max date in md.ust_eod."""
    with conn.cursor() as cur:
        cur.execute("SELECT MAX(ts) FROM md.ust_eod;")
        row = cur.fetchone()
    return row[0] if row and row[0] else None


def rebuild_headline(conn) -> int:
    """Truncate and rebuild md.headline from scratch."""
    sql = HEADLINE_SQL.format(date_filter="")

    with conn.cursor() as cur:
        cur.execute("TRUNCATE md.headline;")
        cur.execute(
            f"INSERT INTO md.headline (ts, tenor, asset_class, cusip, status, px_last, yld_ytm_mid) {sql}",
            {"tenors": HEADLINE_TENORS},
        )
        row_count = cur.rowcount
    conn.commit()
    return row_count


def incremental_headline(conn) -> int:
    """Append only new dates to md.headline."""
    max_headline = get_max_headline_date(conn)
    max_eod = get_max_eod_date(conn)

    if max_eod is None:
        print("No data in md.ust_eod. Nothing to do.")
        return 0

    if max_headline is None:
        # Table is empty, do full rebuild
        print("md.headline is empty. Running full rebuild...")
        return rebuild_headline(conn)

    if max_headline >= max_eod:
        print(f"md.headline is up to date (max date: {max_headline}).")
        return 0

    # Only process dates after max_headline
    date_filter = "AND d.ts > %(max_date)s"
    sql = HEADLINE_SQL.format(date_filter=date_filter)

    with conn.cursor() as cur:
        cur.execute(
            f"INSERT INTO md.headline (ts, tenor, asset_class, cusip, status, px_last, yld_ytm_mid) {sql}",
            {"tenors": HEADLINE_TENORS, "max_date": max_headline},
        )
        row_count = cur.rowcount
    conn.commit()
    return row_count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Build md.headline table")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Full rebuild (truncate and repopulate)",
    )
    args = parser.parse_args()

    print(f"Connecting to Postgres: {DB_DSN}")
    with get_conn() as conn:
        ensure_headline_table(conn)

        if args.rebuild:
            print("Rebuilding md.headline from scratch...")
            rows = rebuild_headline(conn)
            print(f"✅ Inserted {rows:,} rows into md.headline")
        else:
            print("Running incremental update...")
            rows = incremental_headline(conn)
            if rows > 0:
                print(f"✅ Inserted {rows:,} new rows into md.headline")


if __name__ == "__main__":
    main()
