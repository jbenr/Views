"""Add an `updated_at` column to the four market-data tables.

`created_at` records when a row first appeared and is correct as it stands --
the live BDP pulls upsert existing rows all day without touching it, which is
the honest behaviour for a column with that name. The cost is that nothing in
the database records the *last* write, so freshness readouts built on
MAX(created_at) sit frozen at the day's first insert no matter how many live
refreshes have landed since.

This adds the missing half. `created_at` keeps meaning "first seen",
`updated_at` means "last written", and the pair is what lets a caller tell a
new bar from a revised one.

Idempotent: safe to re-run. The column is added nullable with a default rather
than NOT NULL, so existing rows are left NULL and no table rewrite happens --
766k rows across four Timescale hypertables would otherwise be rewritten for
no gain. Readers COALESCE back to `created_at` for those rows.

    python data_pull/migrate_updated_at.py --dry-run
    python data_pull/migrate_updated_at.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.helpers import _connect

TABLES = ["md.index_eod", "md.fut_eod", "md.ust_eod", "md.strips_eod"]


def pending(conn) -> list[str]:
    """Which of TABLES do not yet have an updated_at column."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT table_schema || '.' || table_name
            FROM information_schema.columns
            WHERE column_name = 'updated_at'
              AND table_schema || '.' || table_name = ANY(%s)
            """,
            [TABLES],
        )
        have = {row[0] for row in cur.fetchall()}
    return [t for t in TABLES if t not in have]


def statements(table: str) -> list[str]:
    """DDL for one table: add the column, then attach the default.

    Kept as two statements on purpose. `ADD COLUMN ... DEFAULT now()` uses a
    volatile default, which makes PostgreSQL rewrite every existing row;
    adding the column bare and setting the default afterwards is a pair of
    catalog-only operations that new inserts still pick up.
    """
    return [
        f"ALTER TABLE {table} ADD COLUMN updated_at timestamptz",
        f"ALTER TABLE {table} ALTER COLUMN updated_at SET DEFAULT now()",
    ]


def main(dry_run: bool = False) -> dict:
    # _connect, not psycopg.connect: on Windows the box lets WSL+postgres
    # sleep, and this is the path that wakes it and retries.
    with _connect() as conn:
        todo = pending(conn)
        if not todo:
            print("all four tables already have updated_at -- nothing to do")
            return {"applied": [], "skipped": TABLES}

        print(f"{'would apply' if dry_run else 'applying'} to: {', '.join(todo)}")
        for table in todo:
            for sql in statements(table):
                print(f"  {sql}")
                if not dry_run:
                    with conn.cursor() as cur:
                        cur.execute(sql)
        if dry_run:
            conn.rollback()
            print("dry run -- rolled back")
        else:
            conn.commit()
            print(f"done: {len(todo)} table(s) altered")

    return {"applied": [] if dry_run else todo, "skipped": [t for t in TABLES if t not in todo]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="print the DDL, change nothing")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
