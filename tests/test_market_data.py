"""Market-data transformation logic — the pure part, no DB required."""

import datetime as dt

import pandas as pd
import polars as pl
import pytest

from utils.market_data import align_columns, coverage_report, long_to_wide

LONG = pl.DataFrame({
    "ts": [dt.date(2024, 1, 1), dt.date(2024, 1, 1), dt.date(2024, 1, 2), dt.date(2024, 1, 2)],
    "ticker": ["USGG10YR Index", "SPX Index", "USGG10YR Index", "SPX Index"],
    "px": [4.0, 5000.0, 4.1, 5010.0],
})

TICKERS = {"10y": "USGG10YR Index", "spx": "SPX Index"}


def test_pivot_and_alias_rename():
    wide = long_to_wide(LONG, TICKERS)
    assert set(wide.columns) == {"ts", "10y", "spx"}
    assert len(wide) == 2
    assert wide["10y"].to_list() == [4.0, 4.1]


def test_bps_scaling_selected_columns():
    wide = long_to_wide(LONG, TICKERS, bps_cols=["10y"])
    assert wide["10y"].to_list() == pytest.approx([400.0, 410.0])
    assert wide["spx"].to_list() == [5000.0, 5010.0]


def test_bps_scaling_all():
    wide = long_to_wide(LONG, TICKERS, bps_cols="all")
    assert wide["spx"].to_list() == [500000.0, 501000.0]


def test_list_tickers_keeps_names():
    wide = long_to_wide(LONG, list(TICKERS.values()))
    assert "USGG10YR Index" in wide.columns


def test_to_pandas_indexed_by_ts():
    wide = long_to_wide(LONG, TICKERS, to_pandas=True)
    assert isinstance(wide, pd.DataFrame)
    assert wide.index.name == "ts"
    assert list(wide["10y"]) == [4.0, 4.1]


def test_pandas_long_input():
    wide = long_to_wide(LONG.to_pandas(), TICKERS)
    assert wide["spx"].to_list() == [5000.0, 5010.0]


def test_empty_input():
    empty = pl.DataFrame(schema={"ts": pl.Date, "ticker": pl.Utf8, "px": pl.Float64})
    assert long_to_wide(empty, TICKERS).is_empty()
    assert long_to_wide(empty, TICKERS, to_pandas=True).empty


def test_align_columns_drops_rows_without_full_overlap():
    data = pl.DataFrame(
        {
            "ts": [dt.date(2024, 1, 1), dt.date(2024, 1, 2), dt.date(2024, 1, 3)],
            "target": [1.0, None, 3.0],
            "feature": [10.0, 11.0, 12.0],
            "unused": [100.0, 101.0, 102.0],
        }
    )

    aligned = align_columns(data, ["target", "feature"])

    assert aligned.columns == ["ts", "target", "feature"]
    assert aligned["ts"].to_list() == [dt.date(2024, 1, 1), dt.date(2024, 1, 3)]


def test_coverage_report_includes_overlap_row():
    data = pl.DataFrame(
        {
            "ts": [dt.date(2024, 1, 1), dt.date(2024, 1, 2), dt.date(2024, 1, 3)],
            "target": [1.0, None, 3.0],
            "feature": [None, 11.0, 12.0],
        }
    )

    report = coverage_report(data, ["target", "feature"])
    rows = {r["series"]: r for r in report.to_dicts()}

    assert rows["target"]["observations"] == 2
    assert rows["feature"]["observations"] == 2
    assert rows["OVERLAP"]["observations"] == 1
    assert rows["OVERLAP"]["first_valid"] == dt.date(2024, 1, 3)
