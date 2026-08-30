"""Shared 10Y / 10s30s panel for the three curve-research methods.

This module deliberately contains only data and derived economic inputs.  The
dislocation, relative-value, and fair-value files decide how to research that
panel; none of them owns a competing data recipe.
"""

from __future__ import annotations


import polars as pl

from utils.market_data import coverage_report, load_wide
from utils.rates import linear_forward

START = "2010-01-01"

TICKERS = {
    "10y": "USGG10YR Index",
    "10s30s": "USYC1030 Index",
    "real10y": "USGGT10Y Index",
    "be5": "USGGBE05 Index",
    "be10": "USGGBE10 Index",
    "move": "MOVE Index",
}
BPS_COLS = ["10y", "real10y", "be5", "be10"]


def add_features(data: pl.DataFrame) -> pl.DataFrame:
    """Add the forward-inflation factor used by the declared fair-value model."""
    return data.with_columns(
        linear_forward(pl.col("be5"), 5, pl.col("be10"), 10).alias("5y5y_infl")
    )


def load_data(start: str = START) -> pl.DataFrame:
    """Live research panel, with rates scaled to bps by the shared loader."""
    return add_features(load_wide(TICKERS, start=start, bps_cols=BPS_COLS))


def coverage(data: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """One standard coverage report for each 10s30s research file."""
    return coverage_report(data, columns)
