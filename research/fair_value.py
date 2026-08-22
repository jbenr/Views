"""Research multi-factor fair value and error-correction relationships."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Mapping, Sequence

import polars as pl

from stats import half_life, hurst_exponent, roll_lr, roll_mlr

from .common import aligned_panel, fade_scorecard


@dataclass(frozen=True)
class FairValueStudy:
    """A rolling levels fair-value model for any target and factor list."""

    target: str
    factors: tuple[str, ...]
    lookback: int = 252
    ts_col: str = "ts"

    def __post_init__(self) -> None:
        if not self.factors:
            raise ValueError("fair value needs at least one factor")
        if self.target in self.factors:
            raise ValueError("target may not also be a fair-value factor")

    def compute(self, data: pl.DataFrame) -> pl.DataFrame:
        frame = aligned_panel(data, [self.target, *self.factors], self.ts_col)
        reg = roll_mlr(frame.select(self.factors), frame[self.target], lookback=self.lookback)
        # aligned_panel guarantees no missing inputs, so roll_mlr preserves order/length.
        return frame.with_columns(
            reg["yhat"].alias("fair_value"),
            reg["resid"].alias("residual"),
            reg["r2"].alias("r2"),
            reg["cond"].alias("condition_number"),
            *[reg[f"beta_{factor}"].alias(f"beta_{factor}") for factor in self.factors],
        )

    def research(self, data: pl.DataFrame) -> dict[str, pl.DataFrame]:
        signals = self.compute(data)
        residual = signals["residual"]
        ecm = roll_lr(residual.shift(1), signals[self.target].diff(), lookback=self.lookback)
        # The fair-value residual itself has a model warmup, then the ECM
        # loses another row to differencing.  Pad to the original panel, not
        # merely by one bar.
        ecm_speed = pl.concat(
            [
                pl.Series("error_correction", [None] * (len(signals) - len(ecm)), dtype=pl.Float64),
                ecm["beta"],
            ]
        )
        values = residual.drop_nulls().to_numpy().astype(float)
        diagnostics = pl.DataFrame(
            {
                "n_obs": [len(values)],
                "half_life": [half_life(pl.Series(values)) if len(values) >= 20 else None],
                "hurst": [hurst_exponent(pl.Series(values)) if len(values) >= 20 else None],
                "latest_error_correction": [ecm_speed.drop_nulls()[-1] if ecm_speed.drop_nulls().len() else None],
            }
        )
        return {
            "signals": signals.with_columns(ecm_speed),
            "diagnostics": diagnostics,
            "horizons": fade_scorecard(residual, signals[self.target], (5, 10, 20, 40)),
        }

    @classmethod
    def search(
        cls,
        data: pl.DataFrame,
        target: str,
        factor_families: Mapping[str, Sequence[str]],
        lookback: int = 252,
        max_factors: int = 3,
        ts_col: str = "ts",
    ) -> pl.DataFrame:
        """Compare controlled combinations: at most one factor per family.

        This is deliberately a family-aware exploration tool, not an
        unrestricted all-column combination search.  Every attempted factor
        set appears in the returned table for later selection accounting.
        """
        families = list(factor_families)
        choices = [[None, *factor_families[family]] for family in families]
        rows = []
        for selected in product(*choices):
            factors = tuple(value for value in selected if value is not None)
            if not factors or len(factors) > max_factors:
                continue
            study = cls(target=target, factors=factors, lookback=lookback, ts_col=ts_col)
            state = study.research(data)
            horizons = state["horizons"].filter(pl.col("horizon") == 20)
            diag = state["diagnostics"].row(0, named=True)
            row = {
                "factors": ",".join(factors),
                "n_factors": len(factors),
                "half_life": diag["half_life"],
                "hurst": diag["hurst"],
                "latest_error_correction": diag["latest_error_correction"],
            }
            if not horizons.is_empty():
                row.update(horizons.row(0, named=True))
            rows.append(row)
        return pl.DataFrame(rows).sort("ic", descending=True, nulls_last=True)
