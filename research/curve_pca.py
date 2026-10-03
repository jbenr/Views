"""Fair value of the Treasury curve: the curve the PCA factors imply, and how far each tenor sits from it.

A k-factor PCA of curve levels (level, slope, curvature for k = 3) rebuilds
each day's curve from its broad shape. The gap at each tenor -- actual minus
that implied curve -- is what the factors do not explain: rich or cheap
against the rest of the curve. Fair values are point-in-time (each day
fitted only on the trailing window), so a residual can become a signal
without look-ahead.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from research.panel import YIELDS
from stats import fit_pca, roll_pca_fair_value
from utils.market_data import load_wide

PCA_TENORS = sorted(YIELDS, key=lambda t: int(t[:-1]))  # every headline point; the 20y starts in 2020
PCA_WINDOWS = [252, 504, 756, 1260]


def tenor_years(name: str) -> int:
    return int(name[:-1])


def load_curve(tenors: list[str], start: str) -> pl.DataFrame:
    """Generic Treasury yields in bp, one column per tenor, on days every tenor printed."""
    tenors = sorted(tenors, key=tenor_years)
    frame = load_wide({t: YIELDS[t] for t in tenors}, start=start, bps_cols="all")
    return frame.select("ts", *tenors).drop_nulls().sort("ts")


def pca_curve(tenors: list[str], start: str, window: int | None, n_components: int) -> dict:
    """Today's curve, its PCA-implied curve, and each tenor's residual in bp and in z-score.

    ``window`` is the trailing fit length (None = whole sample, in-sample).
    z is today's residual over the standard deviation of that tenor's own
    residual history. ``explained`` is the variance share of each kept factor
    in the latest fit.
    """
    curve = load_curve(tenors, start)
    tenors = list(curve.columns[1:])
    fitted = roll_pca_fair_value(curve.drop("ts"), lookback=window, n_components=n_components)
    resid = fitted["resid"]
    latest = curve.tail(1).row(0, named=True)
    fair = fitted["fair"].tail(1).row(0, named=True)
    rows = []
    for t in tenors:
        history = resid[t].drop_nulls()
        std = float(history.std()) if len(history) > 1 else np.nan
        rows.append({"tenor": t, "years": tenor_years(t), "actual_bp": latest[t], "fair_bp": fair[t],
                     "residual_bp": latest[t] - fair[t],
                     "z": (latest[t] - fair[t]) / std if std and np.isfinite(std) else None,
                     "resid_std_bp": std})
    sample = curve if window is None else curve.tail(window)
    fit = fit_pca(sample.select(tenors), n_components=n_components)
    return dict(curve=curve, as_of=latest["ts"], table=pl.DataFrame(rows),
                explained=fit["explained_variance"], cumulative=fit["cumulative_variance"])
