"""Small shared helpers for research studies.

These functions intentionally contain only data alignment and descriptive
forward-return scoring.  They do not know how a trade is executed and do not
choose a winner; that belongs to the caller and the true backtest.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import polars as pl


def aligned_panel(
    data: pl.DataFrame,
    required: Iterable[str],
    ts_col: str = "ts",
) -> pl.DataFrame:
    """Return a date-sorted common sample for named input columns."""
    names = list(dict.fromkeys(required))
    missing = [name for name in [ts_col, *names] if name not in data.columns]
    if missing:
        raise ValueError(f"research inputs missing columns: {missing}")
    return data.select([ts_col, *names]).drop_nulls(subset=names).sort(ts_col)


def forward_change(series: pl.Series, horizon: int) -> pl.Series:
    """Change from this bar to ``horizon`` bars ahead, input aligned."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    return (series.shift(-horizon) - series).alias(f"forward_{horizon}d")


def fade_scorecard(
    signal: pl.Series,
    level: pl.Series,
    horizons: Iterable[int],
    min_obs: int = 30,
) -> pl.DataFrame:
    """Forward diagnostics for fading a signed research signal.

    A positive signal is faded (short the level); a negative signal is bought.
    The table is a research diagnostic, not a trade simulation.
    """
    rows = []
    signal = signal.cast(pl.Float64)
    level = level.cast(pl.Float64)
    for horizon in horizons:
        fwd = forward_change(level, int(horizon))
        frame = pl.DataFrame({"signal": signal, "forward": fwd}).drop_nulls()
        n_obs = len(frame)
        if n_obs < min_obs:
            rows.append(
                {
                    "horizon": int(horizon),
                    "n_obs": n_obs,
                    "ic": None,
                    "hit_rate": None,
                    "mean_fade_move": None,
                    "sharpe": None,
                }
            )
            continue
        frame = frame.with_columns(
            (-pl.col("signal").sign() * pl.col("forward")).alias("fade_move")
        )
        summary = frame.select(
            pl.corr(pl.col("signal"), -pl.col("forward"), method="spearman").alias("ic"),
            (pl.col("fade_move") > 0).mean().alias("hit_rate"),
            pl.col("fade_move").mean().alias("mean_fade_move"),
            pl.col("fade_move").std().alias("std_fade_move"),
        ).row(0, named=True)
        std = summary["std_fade_move"]
        sharpe = (
            summary["mean_fade_move"] / std * np.sqrt(252.0 / horizon)
            if std is not None and std > 0
            else None
        )
        rows.append(
            {
                "horizon": int(horizon),
                "n_obs": n_obs,
                "ic": summary["ic"],
                "hit_rate": summary["hit_rate"],
                "mean_fade_move": summary["mean_fade_move"],
                "sharpe": sharpe,
            }
        )
    return pl.DataFrame(rows)


def threshold_scorecard(
    signal: pl.Series,
    level: pl.Series,
    thresholds: Iterable[float],
    horizons: Iterable[int],
    min_obs: int = 12,
) -> pl.DataFrame:
    """Score first threshold-crossing events rather than every daily bar."""
    signal_np = signal.cast(pl.Float64).to_numpy()
    level_np = level.cast(pl.Float64).to_numpy()
    rows = []
    previous = np.concatenate(([np.nan], signal_np[:-1]))
    for threshold in thresholds:
        crossed = ((signal_np >= threshold) & ~(previous >= threshold)) | (
            (signal_np <= -threshold) & ~(previous <= -threshold)
        )
        for horizon in horizons:
            valid = crossed.copy()
            valid[-int(horizon) :] = False
            indices = np.flatnonzero(valid & np.isfinite(signal_np))
            moves = -np.sign(signal_np[indices]) * (
                level_np[indices + int(horizon)] - level_np[indices]
            )
            moves = moves[np.isfinite(moves)]
            n_obs = len(moves)
            std = float(np.std(moves)) if n_obs else np.nan
            rows.append(
                {
                    "threshold": float(threshold),
                    "horizon": int(horizon),
                    "n_events": n_obs,
                    "hit_rate": float(np.mean(moves > 0)) if n_obs else None,
                    "mean_fade_move": float(np.mean(moves)) if n_obs else None,
                    "sharpe": (
                        float(np.mean(moves) / std * np.sqrt(252.0 / horizon))
                        if n_obs >= min_obs and std > 0
                        else None
                    ),
                }
            )
    return pl.DataFrame(rows)
