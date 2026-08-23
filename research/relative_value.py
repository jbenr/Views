"""Research weighted pair and principal-component relative-value packages.

Relative value begins with an actual package, not merely a predictive target.
The research is designed to establish:

* which hedge is defensible: fixed/DV01, rolling beta, or a PC factor;
* whether its residual is stationary, with sensible half-life and Hurst;
* whether the hedge ratio is stable enough to trade;
* whether hidden duration or other factor exposure remains after hedging;
* whether convergence holds across different rates regimes; and
* whether the real package has acceptable carry and roll.

This is a stronger, medium-horizon claim than a short-horizon dislocation.
PCA research has its own class because a fitted curve factor requires separate
leakage and factor-exposure scrutiny from a literal pair trade.
"""

from __future__ import annotations

from dataclasses import dataclass
import polars as pl

from stats import half_life, hurst_exponent, roll_lr_diff, roll_pc1_score

from .common import aligned_panel, fade_scorecard


def _zscore(series: pl.Series, lookback: int) -> pl.Series:
    return ((series - series.rolling_mean(lookback)) / series.rolling_std(lookback)).alias("signal")


def _residual_diagnostics(residual: pl.Series) -> pl.DataFrame:
    values = residual.drop_nulls().to_numpy().astype(float)
    if len(values) < 20:
        return pl.DataFrame({"n_obs": [len(values)], "half_life": [None], "hurst": [None]})
    return pl.DataFrame(
        {
            "n_obs": [len(values)],
            "half_life": [half_life(pl.Series(values))],
            "hurst": [hurst_exponent(pl.Series(values))],
        }
    )


@dataclass(frozen=True)
class PairRVStudy:
    """Research a literal two-leg relative-value package.

    ``left`` and ``right`` may be any level series.  ``weighting='beta'``
    estimates a rolling changes hedge ratio; ``weighting='fixed'`` uses the
    supplied weight, which is appropriate when the caller has already chosen
    DV01-neutral trade weights.
    """

    left: str
    right: str
    weighting: str = "beta"
    fixed_weight: float = 1.0
    beta_lookback: int = 126
    z_lookback: int = 126
    ts_col: str = "ts"

    def __post_init__(self) -> None:
        if self.left == self.right:
            raise ValueError("relative-value legs must be different")
        if self.weighting not in {"beta", "fixed"}:
            raise ValueError("weighting must be 'beta' or 'fixed'")

    def compute(self, data: pl.DataFrame) -> pl.DataFrame:
        frame = aligned_panel(data, [self.left, self.right], self.ts_col)
        left, right = frame[self.left].cast(pl.Float64), frame[self.right].cast(pl.Float64)
        if self.weighting == "fixed":
            beta = pl.Series("hedge_weight", [self.fixed_weight] * len(frame))
        else:
            reg = roll_lr_diff(right, left, lookback=self.beta_lookback)
            beta = pl.concat([pl.Series("hedge_weight", [None], dtype=pl.Float64), reg["beta"]])
        value = (left - beta * right).alias("rv_value")
        return frame.with_columns(beta, value, _zscore(value, self.z_lookback))

    def research(self, data: pl.DataFrame) -> dict[str, pl.DataFrame]:
        signals = self.compute(data)
        return {
            "signals": signals,
            "diagnostics": _residual_diagnostics(signals["rv_value"]),
            "horizons": fade_scorecard(signals["signal"], signals["rv_value"], (5, 10, 20, 40)),
        }


@dataclass(frozen=True)
class PCRelativeValueStudy:
    """Research a target against a point-in-time PC1 factor as its own RV path."""

    target: str
    panel: tuple[str, ...]
    pca_lookback: int = 252
    beta_lookback: int = 126
    z_lookback: int = 126
    ts_col: str = "ts"

    def __post_init__(self) -> None:
        if self.target not in self.panel:
            raise ValueError("PC panel must include the target so its exposure can be measured")

    def compute(self, data: pl.DataFrame) -> pl.DataFrame:
        frame = aligned_panel(data, [self.target, *self.panel], self.ts_col)
        pc1 = roll_pc1_score(frame.select(self.panel), lookback=self.pca_lookback).alias("pc1")
        reg = roll_lr_diff(pc1, frame[self.target], lookback=self.beta_lookback)
        # PC1 has its own warmup before the changes regression can begin.
        # roll_lr_diff removes those unavailable rows, so restore full input
        # alignment rather than assuming a single leading differencing null.
        pad = len(frame) - len(reg)
        beta = pl.concat([pl.Series("pc1_beta", [None] * pad, dtype=pl.Float64), reg["beta"]])
        residual = pl.concat([pl.Series("rv_value", [None] * pad, dtype=pl.Float64), reg["resid"]])
        return frame.with_columns(pc1, beta, residual, _zscore(residual, self.z_lookback))

    def research(self, data: pl.DataFrame) -> dict[str, pl.DataFrame]:
        signals = self.compute(data)
        return {
            "signals": signals,
            "diagnostics": _residual_diagnostics(signals["rv_value"]),
            "horizons": fade_scorecard(signals["signal"], signals["rv_value"], (5, 10, 20, 40)),
        }
