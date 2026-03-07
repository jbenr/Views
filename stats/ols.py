"""Rolling OLS — no lookahead bias. Polars-native vectorized implementation."""

from __future__ import annotations
import numpy as np
import polars as pl
import pandas as pd
from typing import Union
from utils.helpers import to_pl_series


def roll_lr(
    x: Union[pl.Series, pd.Series, pl.DataFrame, pd.DataFrame],
    y: Union[pl.Series, pd.Series],
    lookback: int = 100,
    min_periods: int = None,
) -> pl.DataFrame:
    """y = alpha + beta*x, rolled over trailing `lookback` rows.

    Fully vectorized via rolling sums — no Python loop.
    Returns polars DataFrame with columns: x, y, alpha, beta, yhat, resid, r2.
    """
    # Normalize inputs
    if isinstance(x, (pd.DataFrame, pl.DataFrame)):
        x = x.to_series(0) if isinstance(x, pl.DataFrame) else pl.from_pandas(x.iloc[:, 0])
    else:
        x = to_pl_series(x)
    y = to_pl_series(y)

    if min_periods is None:
        min_periods = lookback

    # Align and drop nulls
    df = pl.DataFrame({"x": x, "y": y}).drop_nulls()

    # Rolling moments
    df = df.with_columns(
        pl.col("x").rolling_sum(lookback, min_periods=min_periods).alias("sum_x"),
        pl.col("y").rolling_sum(lookback, min_periods=min_periods).alias("sum_y"),
        (pl.col("x") * pl.col("x")).rolling_sum(lookback, min_periods=min_periods).alias("sum_xx"),
        (pl.col("x") * pl.col("y")).rolling_sum(lookback, min_periods=min_periods).alias("sum_xy"),
        (pl.col("y") * pl.col("y")).rolling_sum(lookback, min_periods=min_periods).alias("sum_yy"),
        pl.col("x").is_not_null().cast(pl.Int32).rolling_sum(lookback, min_periods=min_periods).cast(pl.Float64).alias("n"),
    )

    # Beta and alpha from normal equations
    df = df.with_columns(
        (
            (pl.col("n") * pl.col("sum_xy") - pl.col("sum_x") * pl.col("sum_y"))
            / (pl.col("n") * pl.col("sum_xx") - pl.col("sum_x") ** 2)
        ).alias("beta"),
    )
    df = df.with_columns(
        ((pl.col("sum_y") - pl.col("beta") * pl.col("sum_x")) / pl.col("n")).alias("alpha"),
    )

    # Fitted values and residuals
    df = df.with_columns(
        (pl.col("alpha") + pl.col("beta") * pl.col("x")).alias("yhat"),
    )
    df = df.with_columns(
        (pl.col("y") - pl.col("yhat")).alias("resid"),
    )

    # R-squared: 1 - SS_res / SS_tot
    # SS_tot = sum_yy - sum_y^2 / n  (rolling variance of y * n)
    # SS_res = rolling sum of resid^2
    df = df.with_columns(
        (pl.col("resid") ** 2).rolling_sum(lookback, min_periods=min_periods).alias("ss_res"),
        (pl.col("sum_yy") - pl.col("sum_y") ** 2 / pl.col("n")).alias("ss_tot"),
    )
    df = df.with_columns(
        pl.when(pl.col("ss_tot") > 0)
        .then(1.0 - pl.col("ss_res") / pl.col("ss_tot"))
        .otherwise(None)
        .alias("r2"),
    )

    return df.select("x", "y", "alpha", "beta", "yhat", "resid", "r2")


def roll_lr_diff(
    x: Union[pl.Series, pd.Series, pl.DataFrame, pd.DataFrame],
    y: Union[pl.Series, pd.Series],
    lookback: int = 100,
    min_periods: int = None,
) -> pl.DataFrame:
    """Changes-based rolling OLS: dy = alpha + beta*dx.

    Returns polars DataFrame with columns:
        x, y, dx, dy, alpha, beta, yhat, resid, resid_cum, r2

    - beta is the hedge ratio on daily changes (correct for beta-weighting).
    - resid is the single-period changes residual.
    - resid_cum is the cumulative residual in level space (for z-scoring).
    """
    if isinstance(x, (pd.DataFrame, pl.DataFrame)):
        x = x.to_series(0) if isinstance(x, pl.DataFrame) else pl.from_pandas(x.iloc[:, 0])
    else:
        x = to_pl_series(x)
    y = to_pl_series(y)

    levels = pl.DataFrame({"x": x, "y": y}).drop_nulls()
    levels = levels.with_columns(
        pl.col("x").diff().alias("dx"),
        pl.col("y").diff().alias("dy"),
    )

    # drop first row (null from diff) to match roll_lr output length
    levels = levels.slice(1)

    reg = roll_lr(levels["dx"], levels["dy"], lookback=lookback, min_periods=min_periods)

    out = levels.select("x", "y").with_columns(
        reg["x"].alias("dx"),
        reg["y"].alias("dy"),
        reg["alpha"],
        reg["beta"],
        reg["yhat"],
        reg["resid"],
        reg["resid"].cum_sum().alias("resid_cum"),
        reg["r2"],
    )
    return out


def roll_beta(x, y, lookback=100):
    return roll_lr(x, y, lookback)["beta"]


def roll_resid(x, y, lookback=100):
    return roll_lr(x, y, lookback)["resid"]
