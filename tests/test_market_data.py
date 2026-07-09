"""Market-data transformation logic — the pure part, no DB required."""

import datetime as dt

import pandas as pd
import polars as pl
import pytest

from utils.market_data import long_to_wide

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
