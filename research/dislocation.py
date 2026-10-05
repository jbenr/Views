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

import gc
import hashlib
import json
import os
import shutil
import warnings
from contextlib import contextmanager
from pathlib import Path

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import polars as pl

from backtest.lab import REGIME_GATE_BUCKETS, gate_allow_from_ranks, gate_percentile_rank, predict_scan
from research.regimes import STATES as REGIME_STATES, count_episodes, regime_column
from utils.market_data import align_columns
from backtest.validation import cv_fold_labels, cv_scores
from backtest.vector import run_vector
from stats.diagnostics import beta_cv, quality_weight
from stats import roll_lr, roll_lr_diff, roll_mlr_diff, roll_ou_features

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
    fit_on: Iterable[str] = ("changes",),
    device: str = "auto",
    signal_kind: str | Iterable[str] = "normalized",
    raw_thresholds: Iterable[float] = (1.0, 2.0, 3.0, 5.0, 10.0, 15.0),
    train_fraction: float = 1.0,
    progress: Callable[[int, int, str], None] | None = None,
    trade_legs: dict[str, float] | None = None,
    weight_columns: dict[str, str] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Causal discovery scan for accumulated changes or levels residuals.

    Changes fits daily moves then independently accumulates daily misses over
    ``residual_lookback`` bars. Levels fits the target level on the feature
    level; its residual is already a level gap, so it has no residual window.
    Gated and ungated cells compete with no future percentile information.
    """
    frame = aligned_panel(data, list(dict.fromkeys([target, feature, *(trade_legs or {}), *(weight_columns or {}).values()])))
    signal_kinds = list(dict.fromkeys([signal_kind] if isinstance(signal_kind, str) else signal_kind))
    if not signal_kinds or set(signal_kinds) - {"normalized", "ou_z", "raw"}:
        raise ValueError("Choose normalized, ou_z, raw, or a combination")
    raw_thresholds = list(raw_thresholds)
    if "raw" in signal_kinds and (not raw_thresholds or any(not np.isfinite(v) or v <= 0 for v in raw_thresholds)):
        raise ValueError("Raw residual thresholds must be positive finite values in target units")
    if not 0.5 <= train_fraction <= 1:
        raise ValueError("train_fraction must be between 0.5 and 1")
    beta_lookbacks = list(beta_lookbacks)
    residual_lookbacks = list(residual_lookbacks)
    normalization_lookbacks = list(normalization_lookbacks)
    thresholds, horizons, gate_windows = list(thresholds), list(horizons), list(gate_windows)
    gate_names = None if gate_names is None else list(gate_names)
    if min_gate_history < 1:
        raise ValueError("min_gate_history must be >= 1")
    if gate_names is None or gate_names:
        invalid_windows = [w for w in gate_windows if w < min_gate_history]
        if invalid_windows:
            raise ValueError(
                f"Gate percentile lookbacks {invalid_windows} are too short: "
                f"gates require at least {min_gate_history} valid observations. "
                f"Choose lookbacks >= {min_gate_history}, or select no gates."
            )
    y, x = frame[target].cast(pl.Float64), frame[feature].cast(pl.Float64)
    n = len(frame)
    null1 = pl.Series([None], dtype=pl.Float64)
    signals: list[np.ndarray] = []
    combos: list[dict] = []
    conditions: dict[str, list[np.ndarray]] = {
        # Feature and target state: the external environment in which the
        # dislocation occurred.
        "feature_level": [], "feature_move20": [], "feature_vol20": [],
        "target_level": [], "target_move20": [], "target_vol20": [],
        # Relationship quality/stability: direct equivalents of the mature
        # strategy funnel's gate menu, using the exact current-window R².
        "r2": [], "beta": [], "beta_cv": [], "model_quality": [],
        "beta_vol20": [], "beta_mom10": [], "r2_vol20": [], "r2_mom10": [],
        # Raw-residual state. OU gates intentionally wait for the later OU
        # pass; this discovery scan is residual-first.
        "resid_vol20": [], "resid_vol60": [], "resid_mom10": [],
        "resid_phi": [], "resid_half_life": [],
    }

    bases = list(dict.fromkeys(fit_on))
    unknown_bases = sorted(set(bases) - {"changes", "levels"})
    if unknown_bases:
        raise ValueError(
            f"unknown regression basis: {unknown_bases}; expected 'changes' or 'levels'"
        )
    if not bases:
        raise ValueError("choose at least one regression basis")

    def exact_window_r2(a: pl.Series, b: pl.Series, lookback: int) -> pl.Series:
        """True trailing single-factor R², not a sum of stale-fit errors.

        For a one-factor OLS with intercept, R² is the squared correlation
        inside the current regression window.  Computing it directly from the
        current window's sufficient statistics avoids scoring old residuals
        made with old coefficients.
        """
        moments = pl.DataFrame({"x": a, "y": b}).with_columns(
            pl.col("x").rolling_sum(lookback, min_samples=lookback).alias("sx"),
            pl.col("y").rolling_sum(lookback, min_samples=lookback).alias("sy"),
            (pl.col("x") * pl.col("x")).rolling_sum(
                lookback, min_samples=lookback
            ).alias("sxx"),
            (pl.col("y") * pl.col("y")).rolling_sum(
                lookback, min_samples=lookback
            ).alias("syy"),
            (pl.col("x") * pl.col("y")).rolling_sum(
                lookback, min_samples=lookback
            ).alias("sxy"),
            pl.col("x").is_not_null().cast(pl.Float64).rolling_sum(
                lookback, min_samples=lookback
            ).alias("n"),
        ).with_columns(
            (pl.col("n") * pl.col("sxx") - pl.col("sx") ** 2).alias("xx"),
            (pl.col("n") * pl.col("syy") - pl.col("sy") ** 2).alias("yy"),
            (pl.col("n") * pl.col("sxy") - pl.col("sx") * pl.col("sy")).alias("xy"),
        )
        return moments.select(
            pl.when((pl.col("xx") > 0) & (pl.col("yy") > 0))
            .then(pl.col("xy") ** 2 / (pl.col("xx") * pl.col("yy")))
            .otherwise(None)
            .alias("r2")
        )["r2"]

    def padded(reg: pl.DataFrame, name: str) -> pl.Series:
        return pl.concat([
            pl.Series([None] * (n - len(reg)), dtype=pl.Float64), reg[name],
        ])

    names = list(conditions) if gate_names is None else list(gate_names)
    unknown = sorted(set(names) - set(conditions))
    if unknown:
        raise ValueError(f"unknown dislocation gate(s): {unknown}")
    conditions = {name: [] for name in names}
    external = {
        "feature_level": x, "feature_move20": x.diff(20),
        "feature_vol20": x.diff().rolling_std(20),
        "target_level": y, "target_move20": y.diff(20),
        "target_vol20": y.diff().rolling_std(20),
    }
    external = {k: v.to_numpy().astype(float) for k, v in external.items() if k in conditions}
    beta_conditions = {}

    def add_signal(
        basis: str, beta_lb: int, residual_lb: int | None,
        dislocation: pl.Series, beta: pl.Series, r2: pl.Series,
    ) -> None:
        key = (basis, beta_lb)
        if key not in beta_conditions:
            stability = beta_cv(beta, lookback=beta_lb)
            values = {
                "beta": beta, "r2": r2, "beta_cv": stability,
                "model_quality": quality_weight(r2, stability),
                "beta_vol20": beta.diff().rolling_std(20), "beta_mom10": beta.diff(10),
                "r2_vol20": r2.diff().rolling_std(20), "r2_mom10": r2.diff(10),
            }
            beta_conditions[key] = {k: v.to_numpy().astype(float)
                                    for k, v in values.items() if k in conditions}
        residual = {
            "resid_vol20": dislocation.diff().rolling_std(20),
            "resid_vol60": dislocation.diff().rolling_std(60),
            "resid_mom10": dislocation.diff(10),
        }
        shared = {**external, **beta_conditions[key],
                  **{k: v.to_numpy().astype(float) for k, v in residual.items() if k in conditions}}
        for norm_lb in normalization_lookbacks:
            needs_ou = "ou_z" in signal_kinds or any(k in conditions for k in ("resid_phi", "resid_half_life"))
            ou = roll_ou_features(dislocation, lookback=int(norm_lb)) if needs_ou else None
            for kind in signal_kinds:
                z = dislocation if kind == "raw" else ou["ou_z"] if kind == "ou_z" else dislocation / dislocation.rolling_std(
                    int(norm_lb), min_samples=int(norm_lb)
                )
                signals.append(z.to_numpy().astype(float))
                combos.append({
                    "fit_on": basis, "beta_lb": int(beta_lb), "residual_lb": residual_lb,
                    "norm_lb": int(norm_lb), "signal_kind": kind,
                    "signal_units": "target units" if kind == "raw" else "standard deviations",
                })
                for name in names:
                    if name == "resid_phi":
                        value = (ou["ou_rho"] - 1).to_numpy().astype(float)
                    elif name == "resid_half_life":
                        value = ou["half_life"].to_numpy().astype(float)
                    else:
                        value = shared[name]
                    conditions[name].append(value)


    for basis in bases:
        for beta_lb in beta_lookbacks:
            if progress:
                progress(len(signals), 0, f"Fitting {basis} regression · lookback {beta_lb}")
            if basis == "changes":
                reg = roll_lr_diff(x, y, lookback=int(beta_lb))
                innovation = padded(reg, "resid")
                beta = padded(reg, "beta")
                r2 = exact_window_r2(x.diff(), y.diff(), int(beta_lb))
                for residual_lb in residual_lookbacks:
                    dislocation = innovation.rolling_sum(
                        int(residual_lb), min_samples=int(residual_lb)
                    )
                    add_signal(
                        basis, int(beta_lb), int(residual_lb), dislocation, beta, r2
                    )
            else:
                reg = roll_lr(x, y, lookback=int(beta_lb))
                # This residual is already a level gap. Re-accumulating it
                # would be a separate model, not a levels regression.
                add_signal(
                    basis, int(beta_lb), None, padded(reg, "resid"),
                    padded(reg, "beta"), exact_window_r2(x, y, int(beta_lb)),
                )

    matrix = np.column_stack(signals)
    names = list(conditions) if gate_names is None else list(gate_names)
    unknown = sorted(set(names) - set(conditions))
    if unknown:
        raise ValueError(f"unknown dislocation gate(s): {unknown}")
    # Batch by model to expose completed work and bound gate-matrix memory.
    cutoff = int(len(frame) * train_fraction)
    held_moves = {}
    if weight_columns:
        for h in horizons:
            moves = np.full(len(frame), np.nan)
            if h < len(frame):
                moves[:-h] = sum(frame[col].to_numpy()[:-h] * (frame[leg].to_numpy()[h:] - frame[leg].to_numpy()[:-h])
                                 for leg, col in weight_columns.items())
            held_moves[h] = moves
    parts = []
    if progress:
        progress(0, len(combos), f"Scoring {len(combos):,} models with batched gate statistics")
    for i, combo in enumerate(combos):
        local_gates = {name: conditions[name][i][:, None] for name in names}
        for sample in (["train", "test"] if cutoff < len(frame) else ["full"]):
            stop = cutoff if sample == "train" else len(frame)
            values = matrix[:stop, i:i+1].copy()
            levels = y.to_numpy()[:stop].copy()
            if sample == "test":
                # Keep gate warmup but exclude pre-split forward outcomes.
                levels[:cutoff] = np.nan
            forwards = None
            if held_moves:
                forwards = {h: v[:stop].copy() for h, v in held_moves.items()}
                for h, v in forwards.items():
                    v[max(0, stop-h):] = np.nan
                    if sample == "test":
                        v[:cutoff] = np.nan
            result = predict_scan(
                values, levels, entries=raw_thresholds if combo['signal_kind'] == 'raw' else thresholds, horizons=horizons,
                combos=[combo], gates={k: v[:stop] for k, v in local_gates.items()},
                gate_buckets="regime", gate_min_history=min_gate_history,
                gate_windows=gate_windows, entry_col="entry_z", device=device,
                forward_moves=forwards,
            ).with_columns(
                pl.lit(sample).alias("sample"),
                (pl.col("n_obs") / max((stop - (cutoff if sample == "test" else 0)) / 252, 1)).alias("events_per_year"),
            )
            parts.append(result)
        if progress:
            progress(i + 1, len(combos),
                     f"Scored {i + 1}/{len(combos)} · {combo['signal_kind']} · {combo['fit_on']} · beta {combo['beta_lb']} · residual {combo['residual_lb']} · norm {combo['norm_lb']}")
    results = pl.concat(parts, how="diagonal_relaxed")
    return frame, results


RANK_RULES = {
    "family": "Discovery Sharpe · family median across model lookbacks, then own",
    "sharpe": "Discovery Sharpe",
    "cv_mean": "CV · mean block Sharpe",
    "cv_worst": "CV · worst block Sharpe",
    "cv_positive": "CV · share of blocks with positive Sharpe",
    "pnl": "Discovery total P&L (bp)",
    "win_rate": "Discovery win rate",
    "n_trades": "Discovery trades",
    "avg_holding_days": "Average holding days",
}
RANK_COLUMNS = {"family": "family_median_sharpe", "sharpe": "sharpe",
                "cv_mean": "cv_mean_sharpe", "cv_worst": "cv_worst_sharpe", "cv_positive": "cv_positive_share",
                "pnl": "pnl_bps", "win_rate": "win_rate", "n_trades": "n_trades",
                "avg_holding_days": "avg_holding_days"}
# Kept per cell while backtesting so each can be ranked; float32 halves their memory.
_COMPACT_EXTRAS = ("pnl_bps", "win_rate", "avg_holding_days")


EXIT_COLUMNS = ["exit_style", "exit_param", "half_life_cap", "signal_stop", "stop_loss_bps", "max_entry_half_life"]
OPEN_ENDED_EMBARGO = 63  # CV buffer (bars) for exits without a fixed length


class _ScanPlan:
    """A discovery grid prepared once; backtests any model's cells on demand.

    Cell ``c`` of a model is gate ``c // n_exits`` crossed with exit spec
    ``c % n_exits``, in the order ``backtest_scan`` has always produced, so a
    cell's global index (model offset + cell) is its row in the full results.
    """

    def __init__(self, data, *, target, feature, legs, beta_lookbacks, residual_lookbacks,
                 normalization_lookbacks, thresholds, exit_params, stop_losses=(None,),
                 half_life_caps=(None,), signal_stops=(None,), max_half_lives=(None,), weight_columns=None,
                 fit_on=("changes",),
                 signal_kind="normalized", raw_thresholds=(1.0, 2.0, 3.0, 5.0, 10.0, 15.0), gate_names=(),
                 gate_windows=(126, 252, 504), min_gate_history=126, cost_bps=0.0, execution_lag=1,
                 train_fraction=0.7, cv_folds=0, regime_gates=(), min_regime_episodes=3):
        from research.dislocation_backtest import exit_cells

        signal_kinds = list(dict.fromkeys([signal_kind] if isinstance(signal_kind, str) else signal_kind))
        if not signal_kinds or set(signal_kinds) - {"normalized", "ou_z", "raw"}:
            raise ValueError("Choose normalized, ou_z, raw, or a combination")
        bases = list(dict.fromkeys(fit_on))
        if not bases or set(bases) - {"changes", "levels"}:
            raise ValueError("choose regression bases from 'changes' and 'levels'")
        exit_params = {style: list(values) for style, values in exit_params.items() if values}
        if not exit_params:
            raise ValueError("choose at least one exit style with at least one parameter")
        if any(float(v) < 1 for v in exit_params.get("time", [])):
            raise ValueError("time stops must be >= 1 bar")
        gate_names, gate_windows = list(gate_names), list(gate_windows)
        if gate_names and any(w < min_gate_history for w in gate_windows):
            raise ValueError(f"Gate percentile lookbacks must be >= {min_gate_history}; choose longer lookbacks or no gates.")
        if not 0.5 <= train_fraction <= 1:
            raise ValueError("train_fraction must be between 0.5 and 1")
        if execution_lag not in (0, 1):
            raise ValueError("execution_lag must be 0 or 1")

        self.target, self.feature, self.cost, self.lag = target, feature, cost_bps, execution_lag
        self.min_gate_history = min_gate_history
        weight_columns = weight_columns or {}
        regime_gates = list(dict.fromkeys(regime_gates))
        missing = [n for n in regime_gates if regime_column(n) not in data.columns]
        if missing:
            raise ValueError(f"regime state columns missing from the data: {missing} (see research.regimes.with_regimes)")
        # Regime states ride along without trimming the sample: a regime that starts late (implied vol
        # in 2022) simply has no state, so its gates are closed, before then.
        self.frame = align_columns(data, list(dict.fromkeys([target, feature, *legs, *weight_columns.values()])),
                                   optional=[regime_column(n) for n in regime_gates]).sort("ts")
        n = len(self.frame)
        self.cut = int(n * train_fraction)
        self.folds = int(cv_folds) if cv_folds and cv_folds >= 2 else 1
        self.embargo = max([int(round(float(v))) for v in exit_params.get("time", [])]
                           + ([OPEN_ENDED_EMBARGO] if set(exit_params) - {"time"} else []))
        self.labels = cv_fold_labels(n, self.cut, self.folds, embargo=self.embargo)
        self.leg_matrix = self.frame.select(list(legs)).to_numpy()
        self.leg_weights = np.array(list(legs.values()), dtype=float)
        self.entry_weights = (self.frame.select([weight_columns[leg] for leg in legs]).to_numpy()
                              if weight_columns else None)
        self.exits = {kind: exit_cells(raw_thresholds if kind == "raw" else thresholds, exit_params,
                                       stop_losses, half_life_caps, signal_stops, max_half_lives)
                      for kind in signal_kinds}
        self.models = [
            (kind, basis, int(beta), None if basis == "levels" else int(resid), int(norm))
            for kind in signal_kinds for basis in bases for beta in beta_lookbacks
            for resid in (residual_lookbacks if basis == "changes" else [None]) for norm in normalization_lookbacks
        ]
        self.gates = [("(none)", "all", None)] + [
            (name, bucket[0], int(window))
            for name in gate_names for window in gate_windows for bucket in REGIME_GATE_BUCKETS]
        # Macro regime gates: one per state, kept only with enough separate episodes in the discovery
        # period, since a regime seen once cannot tell a lasting edge from one lucky era.
        self.regime_episodes, self.skipped_regime_gates = {}, []
        for name in regime_gates:
            states = self.frame[regime_column(name)].head(self.cut).to_list()
            for state in REGIME_STATES[name]:
                count = count_episodes(states, state)
                self.regime_episodes[(name, state)] = count
                if count >= min_regime_episodes:
                    self.gates.append((f"regime:{name}", state, None))
                else:
                    self.skipped_regime_gates.append((name, state, count))
        sizes = [len(self.gates) * len(self.exits[m[0]]) for m in self.models]
        self.offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        # Feature/target conditions are identical for every model: rank them once.
        # Model conditions repeat only within one (signal, basis, beta) group, so
        # their ranks are dropped when the group changes, bounding memory.
        self._shared_ranks: dict = {}
        self._group_ranks: dict = {}
        self._group = None

    @property
    def n_cells(self) -> int:
        return int(self.offsets[-1])

    def model_state(self, i: int) -> pl.DataFrame:
        """Model ``i``'s signal and gate-condition frame (one row per row of ``self.frame``)."""
        from research.dislocation_backtest import signal_frame

        kind, basis, beta, resid, norm = self.models[i]
        return signal_frame(self.frame, target=self.target, feature=self.feature, fit_on=basis, beta_lb=beta,
                            residual_lb=resid, norm_lb=norm, signal_kind=kind)

    def model_rows(self, i: int, cells=None, state: pl.DataFrame | None = None):
        """Backtest model ``i`` (all its cells, or just ``cells``): rows, CV block sums, block sum squares.

        ``state`` reuses a model frame already built by ``model_state`` (the
        regime placebo swaps shifted regime columns into one).
        """
        kind, basis, beta, resid, norm = self.models[i]
        if (kind, basis, beta) != self._group:
            self._group, self._group_ranks = (kind, basis, beta), {}
        if state is None:
            state = self.model_state(i)
        spec = self.exits[kind]
        index = (np.arange(len(self.gates) * len(spec)) if cells is None
                 else np.asarray(cells, dtype=np.int64))
        gate_of, exit_of = index // len(spec), index % len(spec)
        needed = [g for g in np.unique(gate_of) if g > 0]
        position = {g: slot for slot, g in enumerate(needed)}
        masks = [self._mask(state, g) for g in needed]
        configs = pl.DataFrame({
            "model": np.zeros(len(index), dtype=np.int64),
            "gate": [position[g] if g > 0 else -1 for g in gate_of],
            "entry": [spec[e][0] for e in exit_of], "exit_style": [spec[e][1] for e in exit_of],
            "exit_param": [spec[e][2] for e in exit_of], "stop": [float(spec[e][3] or 0.0) for e in exit_of],
            "cap": [float(spec[e][4] or 0.0) for e in exit_of],
            "signal_stop": [float(spec[e][5] or 0.0) for e in exit_of],
            "max_hl": [float(spec[e][6] or 0.0) for e in exit_of],
        })
        result = run_vector(self.leg_matrix, self.leg_weights, state["signal"].to_numpy()[None, :], configs,
                            half_life=state["half_life"].to_numpy()[None, :],
                            gates=np.array(masks) if masks else None, entry_weights=self.entry_weights,
                            cost=self.cost, lag=self.lag, folds=self.labels)
        later = result.folds == self.folds
        discovery = ~later
        trades = result.fold_trades[:, discovery].sum(axis=1)
        wins = result.fold_wins[:, discovery].sum(axis=1)
        columns = {
            "signal_kind": kind,
            "signal_units": "target units" if kind == "raw" else "standard deviations",
            "fit_on": basis, "beta_lb": beta, "residual_lb": resid, "norm_lb": norm,
            "entry_z": [spec[e][0] for e in exit_of],
            "exit_style": [spec[e][1] for e in exit_of], "exit_param": [spec[e][2] for e in exit_of],
            "half_life_cap": pl.Series([spec[e][4] for e in exit_of], dtype=pl.Float64),
            "signal_stop": pl.Series([spec[e][5] for e in exit_of], dtype=pl.Float64),
            "stop_loss_bps": pl.Series([spec[e][3] for e in exit_of], dtype=pl.Float64),
            "max_entry_half_life": pl.Series([spec[e][6] for e in exit_of], dtype=pl.Float64),
            "gate": [self.gates[g][0] for g in gate_of], "gate_bucket": [self.gates[g][1] for g in gate_of],
            "gate_window": pl.Series([self.gates[g][2] for g in gate_of], dtype=pl.Int64),
            "regime_episodes": pl.Series([self._episodes(g) for g in gate_of], dtype=pl.Int64),
            "n_trades": trades,
            "trades_per_year": trades / max(self.cut / 252, 1e-9),
            "sharpe": result.sharpe(discovery),
            "pnl_bps": result.fold_sum[:, discovery].sum(axis=1),
            "win_rate": np.where(trades > 0, wins / np.maximum(trades, 1), np.nan),
            "avg_holding_days": result.metrics["avg_holding_days"],
        }
        if later.any() and result.fold_days[later].sum():
            columns.update(later_sharpe=result.sharpe(later), later_pnl_bps=result.fold_sum[:, later].sum(axis=1),
                           later_trades=result.fold_trades[:, later].sum(axis=1))
        sums = sumsq = None
        if self.folds >= 2:
            core = (result.folds >= 0) & (result.folds < self.folds)
            self.cv_days = result.fold_days[core]
            sums, sumsq = result.fold_sum[:, core], result.fold_sumsq[:, core]
            scores = cv_scores(self.cv_days, sums, sumsq)
            columns.update(cv_mean_sharpe=scores["mean"], cv_worst_sharpe=scores["worst"],
                           cv_positive_share=scores["positive_share"])
        return pl.DataFrame(columns), sums, sumsq

    def _episodes(self, g):
        """Episodes of a regime gate's state in the discovery period; None for other gates."""
        name, bucket, _ = self.gates[g]
        return self.regime_episodes.get((name.removeprefix("regime:"), bucket)) if name.startswith("regime:") else None

    def _mask(self, state, g):
        name, bucket, window = self.gates[g]
        if name.startswith("regime:"):
            return (state[f"gate_{name}"] == bucket).fill_null(False).to_numpy()
        values = state[f"gate_{name}"].to_numpy().astype(float)
        cache = self._shared_ranks if name.startswith(("feature_", "target_")) else self._group_ranks
        key = (name, window, hash(values.tobytes()))
        if key not in cache:
            cache[key] = gate_percentile_rank(values, min_history=self.min_gate_history, window=window)
        return gate_allow_from_ranks(cache[key], (name, bucket))


def backtest_scan(
    data: pl.DataFrame,
    *,
    target: str,
    feature: str,
    legs: dict[str, float],
    beta_lookbacks: Iterable[int],
    residual_lookbacks: Iterable[int],
    normalization_lookbacks: Iterable[int],
    thresholds: Iterable[float],
    exit_params: dict[str, Iterable[float]],
    stop_losses: Iterable[float | None] = (None,),
    half_life_caps: Iterable[float | None] = (None,),
    signal_stops: Iterable[float | None] = (None,),
    max_half_lives: Iterable[float | None] = (None,),
    weight_columns: dict[str, str] | None = None,
    fit_on: Iterable[str] = ("changes",),
    signal_kind: str | Iterable[str] = "normalized",
    raw_thresholds: Iterable[float] = (1.0, 2.0, 3.0, 5.0, 10.0, 15.0),
    gate_names: Iterable[str] = (),
    gate_windows: Iterable[int] = (126, 252, 504),
    min_gate_history: int = 126,
    cost_bps: float = 0.0,
    execution_lag: int = 1,
    train_fraction: float = 0.7,
    cv_folds: int = 0,
    regime_gates: Iterable[str] = (),
    min_regime_episodes: int = 3,
    progress: Callable[[int, int, str], None] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """Discovery by real trades: every model x gate x entry x exit, backtested.

    Each cell is a full trade specification -- first-crossing entry, one
    position at a time, one exit rule from ``exit_params`` (the Trade
    mechanics exits: time, band, revert_frac, half_life_frac) optionally with
    a half-life cap, a signal stop and a P&L stop -- run through
    ``backtest.vector`` with next-bar fills, costs and entry-frozen beta
    weights. Setups are therefore judged with the exits that suit them rather
    than one fixed holding period. The first ``train_fraction`` of bars is the
    discovery period that ranks cells; the rest is reported, never ranked on.

    With ``cv_folds >= 2`` the discovery period is also split into blocks
    (see ``backtest.validation.cv_fold_labels``). The embargo is the longest
    time stop, or ``OPEN_ENDED_EMBARGO`` bars when any exit has no fixed
    length. The per-block sums are returned for ``selection_checks``.
    """
    plan = _ScanPlan(
        data, target=target, feature=feature, legs=legs, beta_lookbacks=beta_lookbacks,
        residual_lookbacks=residual_lookbacks, normalization_lookbacks=normalization_lookbacks,
        thresholds=thresholds, exit_params=exit_params, stop_losses=stop_losses, half_life_caps=half_life_caps,
        signal_stops=signal_stops, max_half_lives=max_half_lives, weight_columns=weight_columns, fit_on=fit_on,
        signal_kind=signal_kind,
        raw_thresholds=raw_thresholds, gate_names=gate_names, gate_windows=gate_windows,
        min_gate_history=min_gate_history, cost_bps=cost_bps, execution_lag=execution_lag,
        train_fraction=train_fraction, cv_folds=cv_folds, regime_gates=regime_gates,
        min_regime_episodes=min_regime_episodes)
    parts, cv_sums, cv_sumsq = [], [], []
    with _gc_paused():
        for i, model in enumerate(plan.models):
            if progress:
                progress(i, len(plan.models), _model_message(i, plan))
            rows, sums, sumsq = plan.model_rows(i)
            parts.append(rows)
            if sums is not None:
                cv_sums.append(sums)
                cv_sumsq.append(sumsq)
    if progress:
        progress(len(plan.models), len(plan.models), f"Backtested {plan.n_cells:,} cells")
    results = pl.concat(parts, how="diagonal_relaxed").with_columns(
        pl.col("residual_lb").cast(pl.Int64), pl.col("win_rate").fill_nan(None))
    extras = {"cut": plan.cut, "folds": plan.folds, "labels": plan.labels, "embargo": plan.embargo}
    if plan.folds >= 2:
        extras.update(cv_days=plan.cv_days, cv_sums=np.vstack(cv_sums), cv_sumsq=np.vstack(cv_sumsq))
    return plan.frame, results, extras


def _model_message(i: int, plan: "_ScanPlan") -> str:
    kind, basis, beta, resid, norm = plan.models[i]
    return (f"Backtesting model {i + 1}/{len(plan.models)} · {kind} · {basis} · beta {beta} · "
            f"residual {resid or '—'} · norm {norm}")


def rank_board(results: pl.DataFrame, rank_by: str = "family", min_trades: int = 30) -> pl.DataFrame:
    """Discovery-period ranking of backtest-scan cells; never looks at later columns.

    ``family`` is the IC board's robustness idea kept on Sharpe: a cell's
    family is every model lookback sharing its signal, basis, entry, exit and
    gate, and families are ranked by their median Sharpe first.
    Returns every eligible cell with ``rank_score`` and a ``board_index``
    that addresses the row in ``results``.
    """
    if rank_by not in RANK_RULES:
        raise ValueError(f"rank_by must be one of {list(RANK_RULES)}")
    column = RANK_COLUMNS[rank_by]
    if column not in results.columns and column != "family_median_sharpe":
        raise ValueError(f"{RANK_RULES[rank_by]} needs a cross-validated discovery run")
    family = ["signal_kind", "fit_on", "entry_z", *EXIT_COLUMNS, "gate", "gate_bucket", "gate_window"]
    eligible = (results.with_row_index("board_index")
                .filter(pl.col("n_trades") >= min_trades)
                .with_columns(pl.col("sharpe").median().over(family).alias("family_median_sharpe")))
    keys = [column, "sharpe"] if column != "sharpe" else ["sharpe"]
    return eligible.with_columns(pl.col(column).alias("rank_score")).sort(
        [*keys, "board_index"], descending=[True] * len(keys) + [False], nulls_last=True)


def rank_board_file(path, rank_by: str = "family", min_trades: int = 30, top: int = 40) -> tuple[pl.DataFrame, int, int]:
    """``rank_board`` for a saved results file, without loading it.

    Saved discovery grids can hold 100M+ cells; reading one to show 40 rows
    filled the machine's memory. This streams the parquet file: filter to
    eligible cells, family medians by group, then the top rows. Returns
    (top rows, cells in the file, eligible cells), with the same ordering
    and columns as ``rank_board(...).head(top)``.
    """
    if rank_by not in RANK_RULES:
        raise ValueError(f"rank_by must be one of {list(RANK_RULES)}")
    column = RANK_COLUMNS[rank_by]
    scan = pl.scan_parquet(path)
    names = scan.collect_schema().names()
    if column not in names and column != "family_median_sharpe":
        raise ValueError(f"{RANK_RULES[rank_by]} needs a cross-validated discovery run")
    if "exit_style" not in names:  # saved before exits joined discovery: horizon-bar time stops
        scan = scan.with_columns(
            pl.lit("time").alias("exit_style"), pl.col("horizon").cast(pl.Float64).alias("exit_param"),
            *[pl.lit(None, dtype=pl.Float64).alias(c) for c in ("half_life_cap", "signal_stop", "stop_loss_bps")])
    if "max_entry_half_life" not in scan.collect_schema().names():  # saved before the entry half-life filter
        scan = scan.with_columns(pl.lit(None, dtype=pl.Float64).alias("max_entry_half_life"))
    family = ["signal_kind", "fit_on", "entry_z", *EXIT_COLUMNS, "gate", "gate_bucket", "gate_window"]
    keys = [column, "sharpe"] if column != "sharpe" else ["sharpe"]
    eligible = scan.with_row_index("board_index").filter(pl.col("n_trades") >= min_trades)
    stream = {"engine": "streaming"}
    # One small row per family: its median Sharpe and size. A median needs every
    # value of a group at once, so a 100M-row grid is grouped one slice at a time
    # (signal, basis, entry, exit style: each a few million rows). Families never
    # span slices, so the medians are exact.
    parts = ["signal_kind", "fit_on", "entry_z", "exit_style"]
    slices = eligible.select(parts).unique().collect(**stream).rows()
    families = pl.concat([
        eligible.filter(pl.all_horizontal([pl.col(c) == v for c, v in zip(parts, key)]))
        .select(*family, "sharpe").group_by(family)
        .agg(pl.col("sharpe").median().alias("family_median_sharpe"), pl.len().cast(pl.Int64).alias("rows"))
        .collect(**stream)
        for key in slices
    ]) if slices else eligible.select(*family, "sharpe").head(0).group_by(family).agg(
        pl.col("sharpe").median().alias("family_median_sharpe"), pl.len().cast(pl.Int64).alias("rows")).collect()
    cells = scan.select(pl.len()).collect(**stream).item()
    n_eligible = int(families["rows"].sum()) if len(families) else 0
    if rank_by == "family":
        # The top rows can only come from the best families: keep families until
        # they hold ``top`` rows, plus any tied with the last one kept.
        ordered = families.sort("family_median_sharpe", descending=True, nulls_last=True)
        reach = int((ordered["rows"].cum_sum() < top).sum())
        cutoff = ordered["family_median_sharpe"][min(reach, len(ordered) - 1)] if len(ordered) else None
        chosen = ordered.filter(pl.col("family_median_sharpe") >= cutoff) if cutoff is not None else ordered
        candidates = eligible.join(chosen.lazy().select(*family), on=family, how="semi", nulls_equal=True)
    else:
        # Rank on just the score columns, then fetch whole rows for the winners.
        winners = (eligible.select("board_index", *keys)
                   .sort([*keys, "board_index"], descending=[True] * len(keys) + [False], nulls_last=True)
                   .head(top).collect(**stream)["board_index"])
        candidates = eligible.filter(pl.col("board_index").is_in(winners.implode()))
    board = (candidates.collect(**stream)
             .join(families.select(*family, "family_median_sharpe"), on=family, how="left", nulls_equal=True)
             .with_columns(pl.col(column).alias("rank_score"))
             .sort([*keys, "board_index"], descending=[True] * len(keys) + [False], nulls_last=True)
             .head(top))
    return board, int(cells), n_eligible


BOARD_ROWS_SAVED = 200
MIN_TRADE_LEVELS = (10, 20, 30, 50, 100)


@contextmanager
def _gc_paused():
    """Pause Python's cyclic garbage collector for a backtest loop.

    With collection running, long discovery runs crashed natively: the
    collector walked an object left damaged by the compiled backtest code.
    With it paused, the same 84M-cell run completed. The loop creates almost
    no reference cycles, so pausing costs little; collection runs afterwards.
    """
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()
        gc.collect()


_CHECKPOINT_SOURCES = ("research/dislocation.py", "research/dislocation_backtest.py", "backtest/vector.py",
                       "backtest/lab.py", "backtest/validation.py", "stats/ols.py", "stats/ou.py")


def run_key(plan: "_ScanPlan", settings: dict) -> str:
    """Identity of one grid: settings, input data and the code that scores it.

    A checkpoint is only reused by a rerun with all three identical, so a
    resumed run is exactly the run that was interrupted.
    """
    root = Path(__file__).parent.parent
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True, default=str).encode())
    digest.update(plan.frame.hash_rows().to_numpy().tobytes())
    for name in _CHECKPOINT_SOURCES:
        digest.update((root / name).read_bytes())
    return digest.hexdigest()[:20]


def _compact_pass(plan: "_ScanPlan", progress=None, message=_model_message, checkpoint: Path | None = None) -> dict:
    """Backtest every cell, keeping only what ranking and the selection checks need.

    About 24 bytes per cell (Sharpe, trades, P&L, win rate, holding days),
    plus 20 for CV scores and 16 per CV block for the block sums when
    cross-validating, instead of a ~180-byte results row per cell. With ``checkpoint`` (a directory), each finished
    model's numbers are saved there, and models already saved are loaded
    instead of rerun, so an interrupted pass resumes where it stopped.
    """
    total = plan.n_cells
    out = {"sharpe": np.empty(total), "n_trades": np.empty(total, dtype=np.int32),
           **{name: np.empty(total, dtype=np.float32) for name in _COMPACT_EXTRAS}}
    if plan.folds >= 2:
        out.update(cv_mean_sharpe=np.empty(total), cv_worst_sharpe=np.empty(total),
                   cv_positive_share=np.empty(total, dtype=np.float32),
                   cv_sums=np.empty((total, plan.folds)), cv_sumsq=np.empty((total, plan.folds)))
    if checkpoint is not None:
        checkpoint.mkdir(parents=True, exist_ok=True)
    resumed = 0
    with _gc_paused():
        for i in range(len(plan.models)):
            part = slice(plan.offsets[i], plan.offsets[i + 1])
            saved = checkpoint / f"model_{i:05d}.npz" if checkpoint is not None else None
            if saved is not None and saved.is_file():
                with np.load(saved) as stored:
                    for name in out:
                        out[name][part] = stored[name]
                    if "cv_days" in stored:
                        plan.cv_days = stored["cv_days"]
                resumed += 1
                continue
            if progress:
                progress(i, len(plan.models), message(i, plan) + (f" · resumed {resumed} saved models" if resumed else ""))
            rows, sums, sumsq = plan.model_rows(i)
            out["sharpe"][part] = rows["sharpe"].to_numpy()
            out["n_trades"][part] = rows["n_trades"].to_numpy()
            for name in _COMPACT_EXTRAS:
                out[name][part] = rows[name].to_numpy()
            if plan.folds >= 2:
                out["cv_mean_sharpe"][part] = rows["cv_mean_sharpe"].to_numpy()
                out["cv_worst_sharpe"][part] = rows["cv_worst_sharpe"].to_numpy()
                out["cv_positive_share"][part] = rows["cv_positive_share"].to_numpy()
                out["cv_sums"][part], out["cv_sumsq"][part] = sums, sumsq
            del rows
            if saved is not None:
                partial = saved.with_suffix(".partial.npz")
                np.savez(partial, **{name: out[name][part] for name in out},
                         **({"cv_days": plan.cv_days} if plan.folds >= 2 else {}))
                os.replace(partial, saved)  # a model is either fully saved or absent
    if progress:
        progress(len(plan.models), len(plan.models),
                 f"Backtested {total:,} cells" + (f" ({resumed} models resumed from checkpoint)" if resumed else ""))
    return out


def _family_medians(plan: "_ScanPlan", compact: dict, eligible: np.ndarray) -> np.ndarray:
    """Each eligible cell's family median Sharpe (NaN elsewhere), as ``rank_board`` defines it.

    A family is one cell position across every model lookback of one signal
    and basis; those models are contiguous and share a cell layout.
    """
    medians = np.full(plan.n_cells, np.nan)
    groups: dict = {}
    for i, (kind, basis, *_rest) in enumerate(plan.models):
        groups.setdefault((kind, basis), []).append(i)
    for members in groups.values():
        start, stop = plan.offsets[members[0]], plan.offsets[members[-1] + 1]
        width = (stop - start) // len(members)
        values = np.where(eligible[start:stop], compact["sharpe"][start:stop], np.nan).reshape(len(members), width)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            family = np.nanmedian(values, axis=0)
        medians[start:stop] = np.where(eligible[start:stop], np.tile(family, len(members)), np.nan)
    return medians


def _top_cells(score: np.ndarray, sharpe: np.ndarray, eligible: np.ndarray, top: int) -> np.ndarray:
    """Global indices of the best ``top`` eligible cells: score, then Sharpe, descending; then index."""
    index = np.flatnonzero(eligible)
    if len(index) == 0:
        return index
    primary = np.where(np.isnan(score[index]), -np.inf, score[index])
    if len(index) > top:
        kth = np.partition(primary, len(primary) - top)[len(primary) - top]
        index, primary = index[primary >= kth], primary[primary >= kth]
    secondary = np.where(np.isnan(sharpe[index]), -np.inf, sharpe[index])
    order = np.lexsort((index, -secondary, -primary))
    return index[order][:top]


REGIME_SCOPE = "regime-"  # a board ranked over macro-regime-gated cells only is saved as rule "regime-<rule>"


def _rank_compact(plan, compact, rank_by: str, min_trades: int, top: int, family=None, restrict=None):
    """(top global indices, their rank scores, their family medians, eligible count) for one rule.

    ``family`` takes the family medians already computed for this minimum
    (they depend only on which cells are eligible), so ranking many rules at
    one level computes them once. ``restrict`` (a per-cell bool mask) ranks
    only those cells, e.g. the macro-regime-gated ones.
    """
    if rank_by not in RANK_RULES:
        raise ValueError(f"rank_by must be one of {list(RANK_RULES)}")
    if RANK_COLUMNS[rank_by] not in compact and rank_by != "family":
        raise ValueError(f"{RANK_RULES[rank_by]} needs a cross-validated discovery run" if rank_by.startswith("cv")
                         else f"{RANK_RULES[rank_by]} was not kept for this run")
    eligible = compact["n_trades"] >= min_trades
    if family is None:
        family = _family_medians(plan, compact, eligible)
    if restrict is not None:  # after the family medians: a family is one gate, so its median is unchanged
        eligible &= restrict
    score = family if rank_by == "family" else compact[RANK_COLUMNS[rank_by]]
    best = _top_cells(score, compact["sharpe"], eligible, top)
    return best, score[best], family[best], int(eligible.sum())


def _shift_regime(frame: pl.DataFrame, column: str, offset: int) -> pl.DataFrame:
    """The regime's states rotated ``offset`` days within its own history (leading days without a state stay so)."""
    values = frame[column].to_list()
    first = next((i for i, v in enumerate(values) if v is not None), len(values))
    known = values[first:]
    if known:
        k = offset % len(known)
        known = known[-k:] + known[:-k] if k else known
    return frame.with_columns(pl.Series(column, values[:first] + known, dtype=pl.Utf8))


def _regime_gated(plan) -> np.ndarray | None:
    """Per-cell mask of every macro-regime-gated cell, or None when the plan has no regime gates."""
    gates = [g for g, (gate, _, _) in enumerate(plan.gates) if gate.startswith("regime:")]
    if not gates:
        return None
    mask = np.zeros(plan.n_cells, dtype=bool)
    for i, model in enumerate(plan.models):
        start, stop = plan.offsets[i], plan.offsets[i + 1]
        mask[start:stop] = np.isin(np.arange(stop - start) // len(plan.exits[model[0]]), gates)
    return mask


def _regime_subset(plan, name: str):
    """One regime's gated cells, model by model: (local cell indices per model, a plan-like view for ranking).

    Every model has the same cells for one regime (its states x the model's
    exits), so the subset keeps the plan's model-by-model layout and the
    family ranking works on it unchanged.
    """
    import types
    gates = [g for g, (gate, _, _) in enumerate(plan.gates) if gate == f"regime:{name}"]
    local = [np.concatenate([g * len(plan.exits[m[0]]) + np.arange(len(plan.exits[m[0]])) for g in gates])
             if gates else np.array([], dtype=np.int64) for m in plan.models]
    offsets = np.concatenate([[0], np.cumsum([len(c) for c in local])]).astype(np.int64)
    return local, types.SimpleNamespace(models=plan.models, offsets=offsets, n_cells=int(offsets[-1]))


def regime_placebo(plan, compact, *, shifts: int, rank_by: str, min_trades: int, progress=None, **_unused) -> list[dict]:
    """Does each regime gate beat the same regime with its calendar shifted?

    For every regime with gates, the real score is the best of its gated
    cells (same rank rule and minimum trades as the board). Each placebo
    rotates that regime's daily states in time -- same episodes and lengths,
    wrong dates. Each model's signal is built once and every shifted calendar
    is swapped into it, so only the regime's gated cells are backtested again.
    """
    regimes = list(dict.fromkeys(gate.removeprefix("regime:") for gate, _, _ in plan.gates
                                 if gate.startswith("regime:")))
    if not regimes:
        return []
    metric = RANK_COLUMNS[rank_by] if rank_by != "family" else "sharpe"
    keys = list(dict.fromkeys(["sharpe", "n_trades", metric]))
    subsets = {name: _regime_subset(plan, name) for name in regimes}
    calendars, results = {}, {}
    for name in regimes:
        known = int(plan.frame[regime_column(name)].is_not_null().sum())
        offsets = [int(f * known) for f in (np.linspace(0.2, 0.8, shifts) if shifts > 1 else [0.5])]
        calendars[name] = [(offset, _shift_regime(plan.frame, regime_column(name), offset)[regime_column(name)])
                           for offset in offsets]
        n_sub = subsets[name][1].n_cells
        results[name] = [{k: np.zeros(n_sub, dtype=np.int32 if k == "n_trades" else float) for k in keys}
                         for _ in offsets]
    for i in range(len(plan.models)):
        if progress:
            progress(i, len(plan.models), f"Regime placebo · model {i + 1}/{len(plan.models)} · "
                                          f"{shifts} shifted calendars for {', '.join(regimes)}")
        state = plan.model_state(i)
        for name in regimes:
            local, view = subsets[name]
            part = slice(view.offsets[i], view.offsets[i + 1])
            for s, (_, calendar) in enumerate(calendars[name]):
                rows, _, _ = plan.model_rows(i, local[i], state=state.with_columns(calendar.alias(f"gate_regime:{name}")))
                for k in keys:
                    results[name][s][k][part] = rows[k].fill_nan(None).fill_null(np.nan).to_numpy() \
                        if k != "n_trades" else rows[k].to_numpy()
    out = []
    for name in regimes:
        local, view = subsets[name]
        real_cells = np.concatenate([plan.offsets[i] + local[i] for i in range(len(plan.models))])
        real = {k: compact[k][real_cells] for k in keys}
        best, scores, _, _ = _rank_compact(view, real, rank_by, min_trades, 1)
        real_score = float(scores[0]) if len(best) else float("nan")
        found = []
        for result in results[name]:
            pick, sub_scores, _, _ = _rank_compact(view, result, rank_by, min_trades, 1)
            found.append(float(sub_scores[0]) if len(pick) else float("nan"))
        valid = [x for x in found if np.isfinite(x)]
        beaten = sum(x >= real_score for x in valid) if np.isfinite(real_score) else len(valid)
        out.append({"regime": name, "states": [b for gate, b, _ in plan.gates if gate == f"regime:{name}"],
                    "real_score": real_score, "placebo_scores": found,
                    "shift_days": [offset for offset, _ in calendars[name]],
                    "p_value": (1 + beaten) / (1 + len(valid))})
    return out


def _board_rows(plan, picks: dict) -> dict:
    """Full results rows for chosen global cell indices, by rerunning only those cells."""
    wanted = np.unique(np.concatenate([idx for idx in picks.values()]) if picks else np.array([], dtype=np.int64))
    model_of = np.searchsorted(plan.offsets, wanted, side="right") - 1
    rows = {}
    for model in np.unique(model_of):
        cells = wanted[model_of == model]
        frame, _, _ = plan.model_rows(int(model), cells - plan.offsets[model])
        frame = frame.with_columns(pl.col("residual_lb").cast(pl.Int64), pl.col("win_rate").fill_nan(None))
        for cell, row in zip(cells, frame.iter_rows(named=True)):
            rows[int(cell)] = row
    return rows


def discovery_compact(
    data: pl.DataFrame,
    *,
    feature: str,
    rank_rules=tuple(RANK_RULES),
    min_trade_levels=MIN_TRADE_LEVELS,
    top: int = BOARD_ROWS_SAVED,
    selection_rule: str | None = None,
    selection_min_trades: int = 30,
    progress: Callable[[int, int, str], None] | None = None,
    checkpoints: Path | None = None,
    regime_placebos: int = 0,
    **scan_kwargs,
) -> dict:
    """``backtest_scan`` + ``rank_board`` for big grids, in a few GB instead of tens.

    Keeps only each cell's ranking numbers while backtesting, ranks every
    rule at every minimum-trade level, then reruns just the winning cells for
    their full rows. Boards equal ``rank_board(results, rule, min).head(top)``.
    Returns the aligned frame, boards keyed ``(rule, min_trades)``, cell and
    eligible counts, and (with CV) selection checks for ``selection_rule``.
    """
    plan = _ScanPlan(data, feature=feature, **scan_kwargs)
    checkpoint = checkpoints / run_key(plan, dict(scan_kwargs, feature=feature)) if checkpoints else None
    compact = _compact_pass(plan, progress, checkpoint=checkpoint)
    rules = [r for r in rank_rules if not r.startswith("cv") or plan.folds >= 2]
    levels = sorted(set(int(m) for m in min_trade_levels) | {int(selection_min_trades)})
    ranked = {}
    # Regime-gated cells rarely reach the top of a grid of millions, so their own boards are saved too.
    regime_mask = _regime_gated(plan)
    for m in levels:
        family = _family_medians(plan, compact, compact["n_trades"] >= m)
        for rule in rules:
            ranked[(rule, m)] = _rank_compact(plan, compact, rule, m, top, family)
            if regime_mask is not None:
                ranked[(REGIME_SCOPE + rule, m)] = _rank_compact(plan, compact, rule, m, top, family,
                                                                 restrict=regime_mask)
        del family
    del regime_mask
    if progress:
        progress(0, 0, f"Rebuilding full detail for the top {top} cells of {len(ranked)} boards")
    rows = _board_rows(plan, {key: value[0] for key, value in ranked.items()})
    boards, eligible = {}, {}
    for key, (best, scores, families, n_eligible) in ranked.items():
        if not key[0].startswith(REGIME_SCOPE):  # cells with enough trades, counted over the whole grid
            eligible[key[1]] = n_eligible
        # rank_score from the full row: some scores are ranked in float32 to save memory
        rule = key[0].removeprefix(REGIME_SCOPE)
        exact = None if rule == "family" else RANK_COLUMNS[rule]
        records = [{"board_index": int(c), **rows[int(c)], "family_median_sharpe": float(f),
                    "rank_score": float(rows[int(c)][exact] if exact else s)}
                   for c, s, f in zip(best, scores, families)]
        boards[key] = pl.DataFrame(records, infer_schema_length=None) if records else pl.DataFrame()
    checks = []
    if plan.folds >= 2 and selection_rule:
        from backtest.validation import selection_checks
        checks = selection_checks(plan.cv_days, compact["cv_sums"], compact["cv_sumsq"],
                                  {"cv_mean": "mean", "cv_worst": "worst"}.get(selection_rule, "sharpe"),
                                  eligible=compact["n_trades"] >= selection_min_trades)
    regimes_tested = []
    if regime_placebos and any(gate.startswith("regime:") for gate, _, _ in plan.gates):
        regimes_tested = regime_placebo(plan, compact, shifts=int(regime_placebos),
                                        rank_by=selection_rule or "family", min_trades=selection_min_trades,
                                        progress=progress)
    if checkpoint is not None:
        shutil.rmtree(checkpoint, ignore_errors=True)  # the run is complete; nothing to resume
    return {"frame": plan.frame, "boards": boards, "cells": plan.n_cells, "eligible": eligible,
            "checks": checks, "folds": plan.folds, "regime_placebo": regimes_tested,
            "skipped_regime_gates": plan.skipped_regime_gates}


def rank_results_file(path, rank_rules=("family", "sharpe", "cv_mean", "cv_worst"),
                      min_trade_levels=MIN_TRADE_LEVELS, top: int = BOARD_ROWS_SAVED, progress=None) -> dict:
    """``discovery_compact``'s boards for a saved full-results file, streamed in chunks.

    Older runs saved every cell. Their rows were written model by model in
    one fixed cell order, so the file is read a million rows at a time keeping
    only each cell's ranking numbers, ranked exactly as ``discovery_compact``
    ranks, and only the winning rows are then read in full.
    """
    import types
    import pyarrow.parquet as pq

    source = pq.ParquetFile(path)
    names = source.schema_arrow.names
    model_cols = ["signal_kind", "fit_on", "beta_lb", "residual_lb", "norm_lb"]
    cv = "cv_mean_sharpe" in names
    wanted = ["n_trades", "sharpe", *(["cv_mean_sharpe", "cv_worst_sharpe"] if cv else [])]
    # filled in place: a list of chunks plus their concatenation would hold every column twice
    total = source.metadata.num_rows
    compact = {name: np.empty(total, dtype=np.int32 if name == "n_trades" else np.float64) for name in wanted}
    starts, models, previous, seen = [], [], None, 0
    for b, batch in enumerate(source.iter_batches(batch_size=1_000_000, columns=model_cols + wanted)):
        frame = pl.from_arrow(batch)
        keys = frame.select(model_cols).rows()
        for offset in [i for i in range(len(keys)) if keys[i] != (keys[i - 1] if i else previous)]:
            starts.append(seen + offset)
            models.append(keys[offset])
        for name in wanted:
            compact[name][seen:seen + len(keys)] = frame[name].to_numpy()
        previous, seen = keys[-1], seen + len(keys)
        if progress:
            progress(seen, total, f"Reading cell scores · {seen:,} of {total:,} rows")
    offsets = np.array([*starts, seen], dtype=np.int64)
    plan = types.SimpleNamespace(models=models, offsets=offsets, n_cells=seen)
    widths: dict = {}
    for i, (kind, basis, *_rest) in enumerate(models):
        widths.setdefault((kind, basis), set()).add(int(offsets[i + 1] - offsets[i]))
    if any(len(w) != 1 for w in widths.values()):
        raise ValueError("results file is not laid out model by model with one cell grid per signal and basis")
    rules = [r for r in rank_rules if not r.startswith("cv") or cv]
    levels = sorted(set(int(m) for m in min_trade_levels))
    ranked = {(rule, m): _rank_compact(plan, compact, rule, m, top) for rule in rules for m in levels}
    wanted_rows = np.unique(np.concatenate([value[0] for value in ranked.values()])) if ranked else np.array([], int)
    group_starts = np.cumsum([0] + [source.metadata.row_group(g).num_rows for g in range(source.num_row_groups)])
    rows = {}
    for g in np.unique(np.searchsorted(group_starts, wanted_rows, side="right") - 1):
        inside = wanted_rows[(wanted_rows >= group_starts[g]) & (wanted_rows < group_starts[g + 1])]
        table = pl.from_arrow(source.read_row_group(int(g)).take(inside - group_starts[g]))
        for cell, row in zip(inside, table.iter_rows(named=True)):
            rows[int(cell)] = row
    boards = {}
    for key, (best, scores, families, _) in ranked.items():
        records = [{"board_index": int(c), **rows[int(c)], "family_median_sharpe": float(f), "rank_score": float(sc)}
                   for c, sc, f in zip(best, scores, families)]
        boards[key] = pl.DataFrame(records, infer_schema_length=None) if records else pl.DataFrame()
    return {"boards": boards, "cells": int(seen), "eligible": {m: int((compact["n_trades"] >= m).sum()) for m in levels}}


def placebo_scan(
    data: pl.DataFrame,
    *,
    feature: str,
    placebos: int,
    rank_by: str = "family",
    min_trades: int = 30,
    progress: Callable[[int, int, str], None] | None = None,
    checkpoints: Path | None = None,
    **scan_kwargs,
) -> list[dict]:
    """The best discovery score the same grid finds when the feature is scrambled.

    Each placebo circularly shifts the feature's daily changes by a different
    offset (20%..80% of the sample), keeping its volatility and
    autocorrelation but breaking any timing link to the target. The whole
    grid is rerun and ranked with the same rule, so the resulting top scores
    are what this search finds from noise alone. Progress counts models
    across all placebos, so a long run shows where it is and how long is left.
    """
    frame = align_columns(data, list(dict.fromkeys([scan_kwargs["target"], feature, *scan_kwargs["legs"],
                                                    *(scan_kwargs.get("weight_columns") or {}).values()])),
                          optional=[regime_column(n) for n in scan_kwargs.get("regime_gates", ())]).sort("ts")
    values = frame[feature].to_numpy().astype(float)
    moves = np.diff(values, prepend=values[0])
    offsets = np.linspace(0.2, 0.8, placebos) * len(values) if placebos > 1 else [0.5 * len(values)]
    out = []
    for i, offset in enumerate(offsets):
        def inner(done, total, message, i=i, offset=offset):
            if progress:
                progress(i * total + done, placebos * total,
                         f"Placebo {i + 1}/{placebos} (feature shifted {int(offset):,} bars) · {message}")
        shifted = frame.with_columns(pl.Series(feature, values[0] + np.cumsum(np.roll(moves, int(offset)))))
        # Only the placebo's top cell matters, so keep compact numbers, not rows.
        plan = _ScanPlan(shifted, feature=feature, **scan_kwargs)
        checkpoint = checkpoints / run_key(plan, dict(scan_kwargs, feature=feature)) if checkpoints else None
        compact = _compact_pass(plan, inner, checkpoint=checkpoint)
        if checkpoint is not None:
            shutil.rmtree(checkpoint, ignore_errors=True)
        best, scores, _, _ = _rank_compact(plan, compact, rank_by, min_trades, 1)
        out.append({"placebo": i + 1, "shift_bars": int(offset),
                    "best_score": float(scores[0]) if len(best) else float("nan"),
                    "best_sharpe": float(compact["sharpe"][best[0]]) if len(best) else float("nan")})
    return out
