"""Tests for futures universes and incremental pull grouping."""

import datetime as dt
import sys
from pathlib import Path


DATA_PULL = Path(__file__).resolve().parents[1] / "data_pull"
sys.path.insert(0, str(DATA_PULL))

import pull_dlv_basket as basket
import pull_fut_eod as fut
import pull_live as live


class FakeCursor:
    def __init__(self, rows):
        self.rows = rows
        self.sql = None
        self.params = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def execute(self, sql, params=None):
        self.sql = sql
        self.params = params

    def fetchall(self):
        return self.rows


class FakeConnection:
    def __init__(self, rows):
        self.cursor_instance = FakeCursor(rows)

    def cursor(self):
        return self.cursor_instance


def test_futures_universe_uses_eight_ff_ser_and_sfr_generics():
    assert fut.TICKERS[:2] == ["TU1 Comdty", "TU2 Comdty"]
    assert "FF1 Comdty" in fut.TICKERS
    assert "FF8 Comdty" in fut.TICKERS
    assert "FF9 Comdty" not in fut.TICKERS
    assert "SER1 Comdty" in fut.TICKERS
    assert "SER8 Comdty" in fut.TICKERS
    assert "SER9 Comdty" not in fut.TICKERS
    assert "SFR1 Comdty" in fut.TICKERS
    assert "SFR8 Comdty" in fut.TICKERS
    assert "SFR9 Comdty" not in fut.TICKERS
    assert live.FUT_TICKERS == fut.TICKERS


def test_contract_parser_keeps_ff_ser_and_sfr_roots():
    assert fut.parse_contract_ticker("FFZ6 Comdty") == ("FF", "FFZ6")
    assert fut.parse_contract_ticker("SERZ6 Comdty") == ("SER", "SERZ6")
    assert fut.parse_contract_ticker("SFRU6 Comdty") == ("SFR", "SFRU6")


def test_grouped_pulls_use_each_generic_watermark_and_backfill_new_series():
    today = dt.date.today()
    conn = FakeConnection([
        ("TY1", today - dt.timedelta(days=1)),
        ("SFR1 Comdty", today - dt.timedelta(days=2)),
    ])

    groups = fut.get_grouped_pulls(
        conn,
        ["TY1 Comdty", "SFR1 Comdty", "SFR2 Comdty"],
    )

    assert groups[today - dt.timedelta(days=8)] == ["TY1 Comdty"]
    assert groups[today - dt.timedelta(days=9)] == ["SFR1 Comdty"]
    assert groups[fut.HISTORICAL_START] == ["SFR2 Comdty"]


def test_grouped_pulls_can_force_a_non_destructive_backfill():
    start = dt.date(2020, 1, 1)
    tickers = ["FF1 Comdty", "SFR1 Comdty"]

    assert fut.get_grouped_pulls(FakeConnection([]), tickers, start) == {
        start: tickers
    }


def test_basket_query_only_selects_deliverable_roots():
    conn = FakeConnection([("TYZ5",)])

    assert basket.get_contracts(conn) == ["TYZ5"]
    assert conn.cursor_instance.params == (basket.DELIVERABLE_ROOTS,)
    assert "generic_ticker = ANY(%s)" in conn.cursor_instance.sql
    assert "FF" not in basket.DELIVERABLE_ROOTS
    assert "SER" not in basket.DELIVERABLE_ROOTS
    assert "SFR" not in basket.DELIVERABLE_ROOTS


def test_contract_metadata_query_only_selects_new_contracts():
    conn = FakeConnection([])

    assert fut.populate_contracts_from_eod(conn, object()) == 0
    assert "LEFT JOIN sec.fut_contracts" in conn.cursor_instance.sql
    assert "c.contract IS NULL" in conn.cursor_instance.sql
