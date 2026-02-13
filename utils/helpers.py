"""Polars / pandas input conversion helpers."""

from __future__ import annotations
import polars as pl
import pandas as pd
from typing import Union


def to_pl_series(s: Union[pl.Series, pd.Series]) -> pl.Series:
    if isinstance(s, pd.Series):
        return pl.from_pandas(s)
    return s


def to_pl_df(df: Union[pl.DataFrame, pd.DataFrame]) -> pl.DataFrame:
    if isinstance(df, pd.DataFrame):
        return pl.from_pandas(df)
    return df


def fix_outliers(
    expr: pl.Expr,
    *,
    hi: float | None = None,
    lo: float | None = None,
) -> pl.Expr:
    """Replace values outside (lo, hi) with linear interpolation from neighbors."""
    mask = pl.lit(False)
    if hi is not None:
        mask = mask | (expr > hi)
    if lo is not None:
        mask = mask | (expr < lo)
    return pl.when(mask).then(None).otherwise(expr).interpolate()
