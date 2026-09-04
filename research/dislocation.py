"""Research short-horizon alpha from market dislocations.

``DislocationStudy`` works with any target ``y`` and zero, one, or many
conditioning inputs ``x``.  It models *changes*, so its signal is today's
dislocation rather than a presumed long-run fair-value gap.  This is the home
for technically rigorous dislocation research.  An OU process is an optional
second-layer diagnostic: it can describe expected correction, half-life, and
time at risk only after the raw dislocation proves worth studying.

What this method is designed to learn:

* whether a target moved too far -- or not far enough -- given related moves;
* which forecast horizon contains the subsequent correction, if any;
* whether large dislocations differ from small ones;
* whether steepening and flattening (or positive/negative states generally)
  behave differently;
* which event, volatility, liquidity, and calendar environments support or
  negate the relationship; and
* whether conditional dislocation improves on a simple extreme-level baseline.

The base model is deliberately simple: a changes regression yields a raw
dislocation in native units.  The optional standardized score only makes
dislocations comparable across volatility regimes.  The optional OU state
does not create the signal; it tests whether the observed dislocation has a
stable convergence process worth trading.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import polars as pl

from backtest.lab import predict_scan
from stats import roll_lr_diff, roll_mlr_diff, roll_ou_features

from .common import aligned_panel, fade_scorecard, threshold_scorecard


@dataclass(frozen=True)
class DislocationStudy:
    """A reusable conditional-dislocation study.

    ``features=()`` studies unusual moves in ``target`` itself.  With one or
    more features, it studies the part of the target move unexplained by
    contemporaneous feature moves.  Inputs are column names, so the same
    object works for rates, inflation, basis, vol, or cross-market panels.
    """

    target: str
    features: tuple[str, ...] = ()
    beta_lookback: int = 126
    normalization_lookback: int = 63
    ou_lookback: int | None = 126
    residual_lookback: int = 1
    ts_col: str = "ts"

    def __post_init__(self) -> None:
        if self.beta_lookback < 2:
            raise ValueError("beta_lookback must be >= 2")
        if self.normalization_lookback < 2:
            raise ValueError("normalization_lookback must be >= 2")
        if self.ou_lookback is not None and self.ou_lookback < 2:
            raise ValueError("ou_lookback must be >= 2 or None")
        if self.residual_lookback < 1:
            raise ValueError("residual_lookback must be >= 1")
        if self.target in self.features:
            raise ValueError("target may not also be a dislocation feature")

    def compute(self, data: pl.DataFrame) -> pl.DataFrame:
        """Return target moves and raw/standardized conditional dislocation."""
        frame = aligned_panel(data, [self.target, *self.features], self.ts_col)
        target = frame[self.target].cast(pl.Float64)
        if not self.features:
            dislocation = target.diff().alias("dislocation")
            extras: dict[str, pl.Series] = {}
        elif len(self.features) == 1:
            feature = frame[self.features[0]].cast(pl.Float64)
            reg = roll_lr_diff(feature, target, lookback=self.beta_lookback)
            dislocation = pl.concat(
                [pl.Series("dislocation", [None], dtype=pl.Float64), reg["resid"]]
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
            dislocation = pl.concat(
                [pl.Series("dislocation", [None], dtype=pl.Float64), reg["resid"]]
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

        innovation = dislocation.alias("innovation")
        # A changes regression produces daily unexplained moves.  The actual
        # dislocation can be their trailing accumulation: a level-like gap
        # that resets every window rather than inheriting an arbitrary anchor.
        dislocation = innovation.rolling_sum(
            self.residual_lookback, min_samples=self.residual_lookback
        ).alias("dislocation")
        scale = dislocation.rolling_std(self.normalization_lookback).alias(
            "dislocation_scale"
        )
        score = (dislocation / scale).alias("dislocation_score")
        ou_extras: dict[str, pl.Series] = {}
        if self.ou_lookback is not None:
            ou = roll_ou_features(dislocation, lookback=self.ou_lookback)
            ou_extras = {
                f"dislocation_{name}": ou[name]
                for name in (
                    "ou_z", "ou_mean", "ou_sigma", "ou_rho", "ou_theta",
                    "expected_delta_1d", "half_life",
                )
            }
        return frame.with_columns(
            target.diff().alias("target_move"),
            innovation,
            dislocation,
            scale,
            score,
            # The raw dislocation is the default research signal.  A caller
            # may explicitly request dislocation_score when comparing states
            # across volatility regimes.
            dislocation.alias("signal"),
            *[series.alias(name) for name, series in extras.items()],
            *[series.alias(name) for name, series in ou_extras.items()],
        )

    def research(
        self,
        data: pl.DataFrame,
        horizons: Iterable[int] = (1, 5, 10, 20),
        thresholds: Iterable[float] = (5.0, 10.0, 15.0, 20.0),
        metric: str = "dislocation",
    ) -> dict[str, pl.DataFrame]:
        """Produce continuous and threshold-event evidence for this idea.

        ``metric='dislocation'`` uses native units (bps for a rates curve).
        ``metric='dislocation_score'`` is an optional volatility-normalized
        alternative; it changes scale, not the underlying regression.
        ``metric='dislocation_ou_z'`` asks the same question through the OU
        state and is available when ``ou_lookback`` is not ``None``.
        """
        signal_frame = self.compute(data)
        if metric not in {"dislocation", "dislocation_score", "dislocation_ou_z"}:
            raise ValueError(
                "metric must be 'dislocation', 'dislocation_score', or 'dislocation_ou_z'"
            )
        if metric not in signal_frame.columns:
            raise ValueError(f"metric {metric!r} needs ou_lookback to be set")
        signal = signal_frame[metric]
        return {
            "signals": signal_frame,
            "horizons": fade_scorecard(signal, signal_frame[self.target], horizons),
            "events": threshold_scorecard(
                signal, signal_frame[self.target], thresholds, horizons
            ),
        }


def dislocation_scan(
    data: pl.DataFrame,
    *,
    target: str,
    feature: str,
    beta_lookbacks: Iterable[int],
    residual_lookbacks: Iterable[int],
    normalization_lookbacks: Iterable[int],
    thresholds: Iterable[float],
    horizons: Iterable[int],
    gate_windows: Iterable[int] = (126, 252, 504),
    min_gate_history: int = 126,
    gate_names: Iterable[str] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Causal discovery scan for accumulated changes residuals.

    ``beta_lookback`` says how much history defines the daily conditional
    move. ``residual_lookback`` independently says how many daily misses form
    the level-like dislocation.  Gated and ungated threshold events are
    evaluated together by :func:`backtest.lab.predict_scan`; no gate gets to
    look at future percentile information.
    """
    frame = aligned_panel(data, [target, feature])
    y, x = frame[target].cast(pl.Float64), frame[feature].cast(pl.Float64)
    n = len(frame)
    null1 = pl.Series([None], dtype=pl.Float64)
    signals: list[np.ndarray] = []
    combos: list[dict] = []
    conditions: dict[str, list[np.ndarray]] = {
        "feature_level": [], "feature_move20": [], "target_vol20": [],
        "beta": [], "r2": [], "resid_vol20": [],
    }

    for beta_lb in beta_lookbacks:
        reg = roll_lr_diff(x, y, lookback=int(beta_lb))
        pad = n - len(reg)
        innovation = pl.concat([
            pl.Series([None] * pad, dtype=pl.Float64), reg["resid"],
        ])
        beta = pl.concat([pl.Series([None] * pad, dtype=pl.Float64), reg["beta"]])
        r2 = pl.concat([pl.Series([None] * pad, dtype=pl.Float64), reg["r2"]])
        for residual_lb in residual_lookbacks:
            dislocation = innovation.rolling_sum(
                int(residual_lb), min_samples=int(residual_lb)
            )
            for norm_lb in normalization_lookbacks:
                z = dislocation / dislocation.rolling_std(
                    int(norm_lb), min_samples=int(norm_lb)
                )
                signals.append(z.to_numpy().astype(float))
                combos.append({
                    "beta_lb": int(beta_lb),
                    "residual_lb": int(residual_lb),
                    "norm_lb": int(norm_lb),
                })
                conditions["feature_level"].append(x.to_numpy().astype(float))
                conditions["feature_move20"].append(x.diff(20).to_numpy().astype(float))
                conditions["target_vol20"].append(y.diff().rolling_std(20).to_numpy().astype(float))
                conditions["beta"].append(beta.to_numpy().astype(float))
                conditions["r2"].append(r2.to_numpy().astype(float))
                conditions["resid_vol20"].append(
                    dislocation.diff().rolling_std(20).to_numpy().astype(float)
                )

    matrix = np.column_stack(signals)
    names = list(conditions) if gate_names is None else list(gate_names)
    unknown = sorted(set(names) - set(conditions))
    if unknown:
        raise ValueError(f"unknown dislocation gate(s): {unknown}")
    gates = {name: np.column_stack(conditions[name]) for name in names}
    results = predict_scan(
        matrix,
        y.to_numpy(),
        entries=list(thresholds),
        horizons=list(horizons),
        combos=combos,
        gates=gates,
        gate_buckets="regime",
        gate_min_history=min_gate_history,
        gate_windows=list(gate_windows),
        entry_col="entry_z",
    ).with_columns(
        (pl.col("n_obs") / max(len(frame) / 252.0, 1.0)).alias("events_per_year")
    )
    return frame, results
