"""Shared 10Y / 10s30s panel for the three curve-research methods.

This module deliberately contains only data and derived economic inputs.  The
dislocation, relative-value, and fair-value files decide how to research that
panel; none of them owns a competing data recipe.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
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


def synthetic_data(n: int = 1500, seed: int = 41) -> pl.DataFrame:
    """Causal synthetic panel with conditional, RV, and fair-value structure.

    It only proves that the research files wire the intended calculations
    together.  It is not evidence for a live 10s30s relationship.
    """
    rng = np.random.default_rng(seed)
    level = 350.0 + np.cumsum(rng.normal(0.0, 1.8, n))
    inflation = 220.0 + np.cumsum(rng.normal(0.0, 0.45, n))
    move = np.empty(n)
    move[0] = 100.0
    for i in range(1, n):
        move[i] = 100.0 + 0.93 * (move[i - 1] - 100.0) + rng.normal(0.0, 2.0)

    # Curve-specific state: the common building block that all three methods
    # should be able to see from a different angle.
    residual = np.zeros(n)
    for i in range(1, n):
        residual[i] = 0.94 * residual[i - 1] + rng.normal(0.0, 1.8)

    be5 = inflation - 8.0 + np.cumsum(rng.normal(0.0, 0.15, n))
    be10 = inflation + np.cumsum(rng.normal(0.0, 0.15, n))
    real10y = level - be10
    fivey_fivey_infl = 2.0 * be10 - be5
    curve = 45.0 + 0.12 * level + 0.10 * fivey_fivey_infl + 0.08 * move + residual

    start_date = dt.date.fromisoformat(START)
    ts = pl.date_range(
        start_date, start_date + dt.timedelta(days=2 * n), interval="1d", eager=True
    )
    ts = ts.filter(ts.dt.weekday() <= 5)[:n]
    return pl.DataFrame(
        {
            "ts": ts,
            "10y": level,
            "10s30s": curve,
            "real10y": real10y,
            "be5": be5,
            "be10": be10,
            "move": move,
            "5y5y_infl": fivey_fivey_infl,
        }
    )


def coverage(data: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """One standard coverage report for each 10s30s research file."""
    return coverage_report(data, columns)
