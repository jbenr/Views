"""Ornstein-Uhlenbeck process: dX = theta*(mu - X)*dt + sigma*dW

Polars-native. Accepts both polars and pandas inputs.
"""

from __future__ import annotations
import warnings
import numpy as np
import polars as pl
import pandas as pd
from typing import Union
from utils.helpers import to_pl_series
from .ols import roll_lr

_LN2 = np.log(2)


def _to_numpy(series: Union[pl.Series, pd.Series]) -> np.ndarray:
    if isinstance(series, pl.Series):
        return series.drop_nulls().to_numpy().astype(float)
    return series.dropna().values.astype(float)


def half_life(series: Union[pl.Series, pd.Series]) -> float:
    """Half-life of mean reversion via AR(1): delta_x = phi*x_lag + eps."""
    s = _to_numpy(series)
    if len(s) < 3:
        return np.nan

    y, x = np.diff(s), s[:-1]
    xm = x.mean()
    dx = x - xm
    ss_xx = (dx * dx).sum()
    if ss_xx == 0:
        return np.nan

    phi = (dx * (y - y.mean())).sum() / ss_xx
    return -_LN2 / np.log(1 + phi) if phi < 0 else np.nan


def ou_params(series: Union[pl.Series, pd.Series], dt: float = 1.0) -> dict:
    """Estimate theta, mu, sigma from discrete series via AR(1) OLS."""
    s = _to_numpy(series)
    if len(s) < 3:
        return {"theta": np.nan, "mu": np.nan, "sigma": np.nan, "half_life": np.nan}

    y, x = np.diff(s), s[:-1]
    xm, ym = x.mean(), y.mean()
    dx = x - xm
    ss_xx = (dx * dx).sum()
    if ss_xx == 0:
        return {"theta": np.nan, "mu": np.nan, "sigma": np.nan, "half_life": np.nan}

    b = (dx * (y - ym)).sum() / ss_xx
    a = ym - b * xm
    theta = -b / dt
    mu = -a / b if b != 0 else np.nan
    sigma = (y - (a + b * x)).std() / np.sqrt(dt)
    hl = -_LN2 / np.log(1 + b) if b < 0 else np.nan

    return {"theta": theta, "mu": mu, "sigma": sigma, "half_life": hl}


def ou_zscore(
    series: Union[pl.Series, pd.Series], lookback: int = None
) -> pl.Series:
    """Z-score vs OU equilibrium (rolling if lookback given, else full-sample)."""
    s = to_pl_series(series).cast(pl.Float64)

    if lookback is not None:
        mu = s.rolling_mean(lookback)
        std = s.rolling_std(lookback)
    else:
        p = ou_params(series)
        mu = p["mu"]
        std = s.drop_nulls().std()

    return ((s - mu) / std).alias("ou_zscore")


def roll_half_life(
    series: Union[pl.Series, pd.Series],
    lookback: int = 100,
    min_periods: int = None,
) -> pl.Series:
    """Rolling half-life over trailing window.

    Vectorized: composes on roll_lr(lag, delta) to get phi, then
    half_life = -ln(2) / ln(1 + phi) element-wise. No Python loop.
    """
    s = to_pl_series(series).cast(pl.Float64)
    n = len(s)

    # AR(1) regression: delta = phi * lag + eps
    delta = s.diff()
    lag = s.shift(1)

    reg = roll_lr(lag, delta, lookback=lookback, min_periods=min_periods)
    phi = reg["beta"]

    # half_life = -ln(2) / ln(1 + phi), only valid where phi < 0
    # Polars evaluates all branches before masking, so the numpy divide-by-zero
    # warning fires even though the result is correctly null. Suppress it.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = (
            pl.DataFrame({"phi": phi})
            .select(
                (-_LN2 / (1.0 + pl.when(pl.col("phi") < 0)
                            .then(pl.col("phi"))
                            .otherwise(None)).log())
                .alias("half_life")
            )
            .to_series()
        )

    # Pad to match input length — roll_lr drops leading nulls internally
    if len(result) < n:
        pad = pl.Series("half_life", [None] * (n - len(result)), dtype=pl.Float64)
        result = pl.concat([pad, result])

    return result


def ou_summary(
    series: Union[pl.Series, pd.Series], lookback: int = None
) -> pl.DataFrame:
    """Single-row summary: theta, mu, sigma, half_life, current zscore & level."""
    s = to_pl_series(series).cast(pl.Float64)
    s_clean = s.drop_nulls()

    window = s_clean.tail(lookback) if lookback else s_clean
    p = ou_params(window)
    z = ou_zscore(s, lookback=lookback).drop_nulls().to_list()
    current_z = z[-1] if z else np.nan
    current_lvl = s_clean.to_list()[-1] if len(s_clean) > 0 else np.nan

    return pl.DataFrame(
        [
            {
                "theta": p["theta"],
                "mu": p["mu"],
                "sigma": p["sigma"],
                "half_life": p["half_life"],
                "current_zscore": current_z,
                "current_level": current_lvl,
            }
        ]
    )
