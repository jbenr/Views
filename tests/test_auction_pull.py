"""Tests for Treasury auction incremental refresh behavior."""

import datetime as dt
import sys
from pathlib import Path

import pandas as pd


DATA_PULL = Path(__file__).resolve().parents[1] / "data_pull"
sys.path.insert(0, str(DATA_PULL))

import pull_ust_cusips_postgres as auctions


class FakeCursor:
    def __init__(self, rows=None, one=None):
        self.rows = rows or []
        self.one = one
        self.sql = None
        self.rowcount = 0

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def execute(self, sql, params=None):
        self.sql = sql

    def executemany(self, sql, rows):
        self.sql = sql
        self.rowcount = len(rows)

    def fetchall(self):
        return self.rows

    def fetchone(self):
        return self.one


class FakeConnection:
    def __init__(self, cursor):
        self.cursor_instance = cursor

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def cursor(self):
        return self.cursor_instance

    def commit(self):
        pass


def test_dedupe_keeps_latest_published_auction_row():
    df = pd.DataFrame({
        "cusip": ["912TEST", "912TEST"],
        "auction_date": pd.to_datetime(["2026-08-04", "2026-08-04"]),
        "record_date": pd.to_datetime(["2026-08-04", "2026-08-06"]),
        "bid_to_cover_ratio": [None, 2.75],
    })

    result = auctions._dedupe(df)

    assert len(result) == 1
    assert result.iloc[0]["record_date"] == pd.Timestamp("2026-08-06")
    assert result.iloc[0]["bid_to_cover_ratio"] == 2.75


def test_insert_updates_existing_auction_rows():
    columns = [(name,) for name in (
        "cusip", "auction_date", "record_date", "bid_to_cover_ratio"
    )]
    cursor = FakeCursor(rows=columns)
    conn = FakeConnection(cursor)
    df = pd.DataFrame({
        "cusip": ["912TEST"],
        "auction_date": [pd.Timestamp("2026-08-04")],
        "record_date": [pd.Timestamp("2026-08-06")],
        "bid_to_cover_ratio": [2.75],
    })

    assert auctions._insert_dataframe(conn, df) == 1
    assert 'DO UPDATE SET' in cursor.sql
    assert '"record_date" = EXCLUDED."record_date"' in cursor.sql


def test_incremental_pull_uses_inclusive_record_date_overlap(monkeypatch):
    max_record_date = dt.datetime(2026, 8, 6)
    conn = FakeConnection(FakeCursor(one=(max_record_date,)))
    captured = {}

    monkeypatch.setattr(auctions, "get_conn", lambda: conn)
    monkeypatch.setattr(
        auctions,
        "fetch_all_auctions",
        lambda extra_params=None: captured.update(extra_params) or pd.DataFrame(),
    )

    auctions.incremental_update()

    assert captured["filter"] == "record_date:gte:2026-07-30"
