"""Residual diagnostics — horizon backtests and model-quality gates.

Shared by strategy research across the book: given a fair-value residual,
does fading it make money (horizon_backtest), and is the model behind it
stable enough to trust (beta_cv, quality_weight)?

Polars-native, accepts pandas or polars inputs.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
import polars as pl

from utils.helpers import to_pl_series


def horizon_backtest(
    resid: Union[pl.Series, pd.Series],
    horizons: tuple[int, ...] = (5, 20, 60),
    periods_per_year: int = 252,
    min_obs: int = 30,
) -> pl.DataFrame:
    """Backtest fading a residual at multiple horizons: IC, hit rate, Sharpe.

    P&L = sign(resid_t) * -(resid_{t+h} - resid_t) — positive when the
    residual reverts toward zero over the next h bars.

    Returns one row per horizon: h, n, ic (Spearman), hit, sharpe.
    """
    s = to_pl_series(resid).cast(pl.Float64)
    rows = []
    for h in horizons:
        joint = (
            pl.DataFrame({"s": s})
            .with_columns((pl.col("s").shift(-h) - pl.col("s")).alias("f"))
            .drop_nulls()
            .filter(pl.col("s") != 0)
        )
        n = len(joint)
        if n < min_obs:
            rows.append({"h": h, "n": n, "ic": None, "hit": None, "sharpe": None})
            continue

        stats_row = joint.select(
            pl.corr(pl.col("s"), -pl.col("f"), method="spearman").alias("ic"),
            (pl.col("s").sign() == (-pl.col("f")).sign()).mean().alias("hit"),
            (pl.col("s").sign() * -pl.col("f")).mean().alias("pnl_mean"),
            (pl.col("s").sign() * -pl.col("f")).std().alias("pnl_std"),
        ).row(0, named=True)

        sharpe = (
            stats_row["pnl_mean"] / stats_row["pnl_std"] * np.sqrt(periods_per_year / h)
            if stats_row["pnl_std"] and stats_row["pnl_std"] > 0
            else None
        )
        rows.append({
            "h": h,
            "n": n,
            "ic": round(float(stats_row["ic"]), 3),
            "hit": round(float(stats_row["hit"]), 3),
            "sharpe": round(float(sharpe), 2) if sharpe is not None else None,
        })

    return pl.DataFrame(
        rows,
        schema={"h": pl.Int64, "n": pl.Int64, "ic": pl.Float64,
                "hit": pl.Float64, "sharpe": pl.Float64},
    )


def beta_cv(
    beta: Union[pl.Series, pd.Series],
    lookback: int,
    cap: float = 2.0,
) -> pl.Series:
    """Rolling coefficient of variation of a hedge-ratio beta.

    std(beta) / mean(|beta|) over the trailing window, capped at `cap`.
    Low = stable relationship, high = the model is chasing a moving target.
    """
    b = to_pl_series(beta).cast(pl.Float64)
    denom = b.abs().rolling_mean(lookback)
    cv = b.rolling_std(lookback) / denom
    return (
        pl.DataFrame({"cv": cv})
        .select(
            pl.when(pl.col("cv").is_finite())
            .then(pl.col("cv"))
            .otherwise(None)
            .clip(0.0, cap)
            .alias("beta_cv")
        )
        .to_series()
    )


def quality_weight(
    r2: Union[pl.Series, pd.Series],
    cv: Union[pl.Series, pd.Series],
    cv_cap: float = 2.0,
) -> pl.Series:
    """Model-quality weight in [0, 1]: R² discounted by beta instability.

    weight = clip01(r2) × (1 − clip01(beta_cv / cv_cap)). Use it to gate or
    scale signals — a big residual from an unstable model is not a trade.
    """
    r = to_pl_series(r2).cast(pl.Float64).fill_null(0.0).clip(0.0, 1.0)
    c = to_pl_series(cv).cast(pl.Float64)
    penalty = (c / cv_cap).clip(0.0, 1.0)
    return ((r * (1.0 - penalty)).clip(0.0, 1.0)).alias("quality_weight")
