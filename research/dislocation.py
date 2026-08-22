"""Research short-horizon alpha from market dislocations and overreactions.

``DislocationStudy`` works with any target ``y`` and zero, one, or many
conditioning inputs ``x``.  It models *changes*, so its signal is today's
unusual move rather than a presumed long-run fair-value gap.  That makes this
the home for z-score, shock, and conditional-reversal research.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import polars as pl

from stats import roll_lr_diff, roll_mlr_diff

from .common import aligned_panel, fade_scorecard, threshold_scorecard


@dataclass(frozen=True)
class DislocationStudy:
    """A reusable conditional-overreaction study.

    ``features=()`` studies unusual moves in ``target`` itself.  With one or
    more features, it studies the part of the target move unexplained by
    contemporaneous feature moves.  Inputs are column names, so the same
    object works for rates, inflation, basis, vol, or cross-market panels.
    """

    target: str
    features: tuple[str, ...] = ()
    beta_lookback: int = 126
    z_lookback: int = 63
    ts_col: str = "ts"

    def __post_init__(self) -> None:
        if self.beta_lookback < 2:
            raise ValueError("beta_lookback must be >= 2")
        if self.z_lookback < 2:
            raise ValueError("z_lookback must be >= 2")
        if self.target in self.features:
            raise ValueError("target may not also be a dislocation feature")

    def compute(self, data: pl.DataFrame) -> pl.DataFrame:
        """Return target moves, conditional surprise, and standardized signal."""
        frame = aligned_panel(data, [self.target, *self.features], self.ts_col)
        target = frame[self.target].cast(pl.Float64)
        if not self.features:
            surprise = target.diff().alias("surprise")
            extras: dict[str, pl.Series] = {}
        elif len(self.features) == 1:
            feature = frame[self.features[0]].cast(pl.Float64)
            reg = roll_lr_diff(feature, target, lookback=self.beta_lookback)
            surprise = pl.concat(
                [pl.Series("surprise", [None], dtype=pl.Float64), reg["resid"]]
            )
            extras = {
                "predicted_move": pl.concat(
                    [pl.Series("predicted_move", [None], dtype=pl.Float64), reg["yhat"]]
                ),
                f"beta_{self.features[0]}": pl.concat(
                    [pl.Series(f"beta_{self.features[0]}", [None], dtype=pl.Float64), reg["beta"]]
                ),
                "r2": pl.concat([pl.Series("r2", [None], dtype=pl.Float64), reg["r2"]]),
            }
        else:
            reg = roll_mlr_diff(
                frame.select(self.features), target, lookback=self.beta_lookback
            )
            surprise = pl.concat(
                [pl.Series("surprise", [None], dtype=pl.Float64), reg["resid"]]
            )
            extras = {
                "predicted_move": pl.concat(
                    [pl.Series("predicted_move", [None], dtype=pl.Float64), reg["yhat"]]
                ),
                "r2": pl.concat([pl.Series("r2", [None], dtype=pl.Float64), reg["r2"]]),
                "condition_number": pl.concat(
                    [pl.Series("condition_number", [None], dtype=pl.Float64), reg["cond"]]
                ),
            }
            for feature in self.features:
                extras[f"beta_{feature}"] = pl.concat(
                    [
                        pl.Series(f"beta_{feature}", [None], dtype=pl.Float64),
                        reg[f"beta_{feature}"],
                    ]
                )

        z = ((surprise - surprise.rolling_mean(self.z_lookback)) /
             surprise.rolling_std(self.z_lookback)).alias("signal")
        return frame.with_columns(
            target.diff().alias("target_move"),
            surprise,
            z,
            *[series.alias(name) for name, series in extras.items()],
        )

    def research(
        self,
        data: pl.DataFrame,
        horizons: Iterable[int] = (1, 5, 10, 20),
        thresholds: Iterable[float] = (1.0, 1.5, 2.0, 2.5),
    ) -> dict[str, pl.DataFrame]:
        """Produce continuous and threshold-event evidence for this idea."""
        signal_frame = self.compute(data)
        return {
            "signals": signal_frame,
            "horizons": fade_scorecard(signal_frame["signal"], signal_frame[self.target], horizons),
            "events": threshold_scorecard(
                signal_frame["signal"], signal_frame[self.target], thresholds, horizons
            ),
        }
