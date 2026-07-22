"""The strategy template — a beta-hedged residual mean-reversion funnel.

A Strategy is one (x feature, y target) trade: fit a changes-based rolling
OLS of d(target) on d(feature), roll the daily residuals into level space,
and fade extremes of the residual (or its OU z-score), optionally gated by
a quantile regime condition. The class provides the whole research funnel;
a book module is just configuration:

    from backtest.strategy import Strategy

    STRATEGY = Strategy(
        name="tens_10s30s",
        module="book.curve.tens_10s30s",   # sweep workers import this
        path=Path(__file__),               # artifacts live in data/<name>/
        tickers={"10y": "USGG10YR Index", "10s30s": "USYC1030 Index"},
        bps_cols=["10y"],
        target="10s30s",
        feature="10y",
    )
    compute = STRATEGY.compute             # lab worker contract
    make_pipeline = STRATEGY.make_pipeline
    pipeline = STRATEGY.pipeline

    if __name__ == "__main__":
        STRATEGY.cli()

The research funnel. Each step saves its winners to a parquet in the strategy
module's ``data/<strategy name>/`` directory; the next step reads it — no
hand-copying between runs:

  1. --predict  cast a wide net: which (lookbacks, entry signal, threshold,
     horizon) cells show ANY forward predictability, with every gate
     condition/bucket as extra candidate cells. No trading mechanics — just
     IC / hit / fire rate. GPU-vectorized. Gated and ungated setups compete
     on raw IC in one leaderboard; the predict_top_n best distinct cells by
     NEIGHBORHOOD IC (median over adjacent grid cells — lone spikes are
     search noise and get dropped) are saved to setups_file.
  2. --exit  reads setups_file and runs an approximate TRADE backtest of
     each saved setup across exit styles — threshold bands, percent-of-
     dislocation-reverted, and half-life-scaled time stops: which exits
     pay, how long they hold, at what hit rate and PnL per trade. The best
     exit per setup is saved to exits_file.
  3. --sweep  reads exits_file and runs each (setup, exit) winner through
     the exact row-by-row engine — stops, costs, full trade mechanics — the
     live-simulation check before promotion. Saves each winner's full trade
     log (trades_file). Promotion ranking is the ROBUSTNESS board: sharpe
     with the best trade removed, best-trade share of PnL, and per-era
     consistency — a setup whose backtest hinges on one lucky trade must
     not outrank one that pays steadily.

Every grid below is a constructor field; override per strategy as needed.
Derived features (PC1 scores, synthetic forwards, ...) plug in via
feature_fn, which runs on the loaded panel before the model frame is cut.
"""

from __future__ import annotations

import datetime as dt
import math
import sys
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import polars as pl

import utils
from stats import beta_cv, horizon_backtest, roll_lr_diff, roll_ou_features
from utils.market_data import align_columns, coverage_report, load_wide

from .engine import (
    BacktestConfig,
    Engine,
    SignalConfig,
    SignalPipeline,
    TradeDef,
    print_summary,
    profit_target,
    trade_log,
)
from .lab import (
    ParamGrid,
    fast_scan,
    gate_allow_mask,
    gate_variant_count,
    neighbor_ic_stats,
    parse_gate,
    predict_scan,
    signal_matrix,
    stateful_exit_scan,
    sweep_strategy,
)
from .validation import (
    deflated_sharpe_ratio,
    effective_number_of_trials,
    event_overlap_diagnostics,
    probability_of_backtest_overfitting,
)

# the promoted-configuration shape every strategy starts from
DEFAULT_PARAMS = {
    # model fit: hedge ratio and OU state of the residual
    "beta_lb": 252,  # hedge-ratio lookback (days)
    "ou_lb": 252,  # OU-state lookback for z/mean/half-life
    # entry: which signal to fade, at what threshold (units follow the signal)
    "entry_signal": "residual",  # "residual" (bps) | "ou_z" (z)
    "entry_threshold": 20.0,  # positive signal -> short target
    # entry filters: what must ALSO be true at the bar to take the trade
    "z_gate": 0.5,  # residual entries only: OU z must confirm; None = off
    "half_life_min": 3.0,  # block unstable / too-fast OU fits
    "half_life_max": 120.0,  # block slow drifts masquerading as mean reversion
    "gate": None,  # quantile regime gate (condition, bucket) — see lab.parse_gate
    # exit: one primary style from the --exit menu + hard stop on top
    "exit_style": "revert_frac",  # "band" | "revert_frac" | "half_life_frac"
    "exit_param": 0.5,  # meaning follows the style (band level / frac / x half-life)
    "stop_loss_bps": 25.0,  # hard stop on adverse move
}


def _finite(value) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _half_life_ok(half_life, lo: float | None, hi: float | None) -> bool:
    """Entry sanity bounds on the OU half-life; lo=hi=None disables the check."""
    if lo is None and hi is None:
        return True
    if not _finite(half_life):
        return False
    hl = float(half_life)
    return (lo is None or hl >= lo) and (hi is None or hl <= hi)


def _entry_filter(z_gate: float | None, half_life_min: float, half_life_max: float):
    """Quantile gate + half-life sanity + optional OU-z confirmation.

    z_gate=None drops the OU-z confirmation entirely so the quantile gate
    (params["gate"] -> gate_allow) carries the entry filtering on its own.
    half_life_min=half_life_max=None drops the half-life sanity check.
    """

    def fn(direction: int, bar: dict) -> bool:
        # gate_allow arrives as 1.0/0.0 (engine floats extras); absent = no gate
        gate = bar.get("gate_allow")
        if gate is not None and gate != 1.0:
            return False

        if not _half_life_ok(bar.get("half_life"), half_life_min, half_life_max):
            return False

        if z_gate is None:
            return direction in (1, -1)

        ou_z = bar.get("ou_z")
        if not _finite(ou_z):
            return False
        # direction +1 is long target: residual should be cheap/negative.
        # direction -1 is short target: residual should be rich/positive.
        if direction == 1:
            return float(ou_z) <= -z_gate
        if direction == -1:
            return float(ou_z) >= z_gate
        return False

    return fn


def _normalize_predict_signals(signals: list[str]) -> list[str]:
    """Canonicalize configured predict signals and reject silent no-ops."""
    aliases = {"ou": "ou_z"}
    normalized = [aliases.get(signal, signal) for signal in signals]
    unknown = sorted(set(normalized) - {"residual", "ou_z"})
    if unknown:
        raise ValueError(
            f"unknown predict_entry_signals values: {unknown}; "
            "expected 'residual', 'ou', or 'ou_z'"
        )
    return list(dict.fromkeys(normalized))


def _setup_name(row: dict) -> str:
    """Short label for a saved setup, e.g. 'ou330/250/e2.7 h40 r2_mom10:high_75'."""
    if row["entry_signal"] == "residual":
        base = f"res{row['beta_lb']}/e{row['entry_threshold']:g}"
    else:
        base = f"ou{row['beta_lb']}/{row['ou_lb']}/e{row['entry_threshold']:g}"
    name = f"{base} h{row['predict_horizon']}"
    if row["gate"] != "(none)":
        name += f" {row['gate']}:{row['gate_bucket']}"
    return name


def _ann_sharpe(daily_pnl: np.ndarray) -> float:
    sd = float(daily_pnl.std())
    return float(daily_pnl.mean()) / sd * math.sqrt(252.0) if sd > 0 else 0.0


def _daily_pnl_from_trades(
    trades: pl.DataFrame, date_ix: dict, dlevel: np.ndarray
) -> np.ndarray:
    """Gross daily re-marking of a trade log: position[t-1] x d(level)."""
    pos = np.zeros(len(dlevel))
    for t in trades.iter_rows(named=True):
        i0, i1 = date_ix.get(t["entry_date"]), date_ix.get(t["exit_date"])
        if i0 is not None and i1 is not None:
            pos[i0:i1] = 1.0 if t["direction"] == "long" else -1.0
    return np.concatenate([[0.0], pos[:-1]]) * dlevel


def _winner_params(row: dict) -> dict:
    """Engine params for one (setup, exit) winner row from exits_file.
    z_gate and the half-life sanity bounds are off — the discovery scans
    never used them, and re-filtering the discovered event here starves the
    exact engine of the very trades steps 1-2 counted."""
    p = {
        "entry_signal": row["entry_signal"],
        "beta_lb": int(row["beta_lb"]),
        "entry_threshold": float(row["entry_threshold"]),
        "exit_style": row["exit_style"],
        "exit_param": float(row["exit_threshold"]),
        "gate": (
            None
            if row["gate"] in (None, "(none)")
            else (row["gate"], row["gate_bucket"])
        ),
        "z_gate": None,
        "half_life_min": None,
        "half_life_max": None,
    }
    if row["ou_lb"] is not None:
        p["ou_lb"] = int(row["ou_lb"])
    return p


@dataclass
class Strategy:
    """One (feature -> target) residual-fade strategy with its full funnel."""

    # identity
    name: str
    module: str  # importable path, used by sweep workers
    path: Path  # strategy module file; artifacts live in data/<name>/ below it
    tickers: dict[str, str]
    target: str
    feature: str
    bps_cols: list[str] = field(default_factory=list)
    start: str = "2010-01-01"
    family: str = "curve"

    # promoted configuration — what main() runs as the live signal, and the
    # base every mode overrides from
    default_params: dict = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    transaction_cost_bps: float = 0.1

    # data hooks
    synthetic_fn: Optional[Callable[[], pl.DataFrame]] = None
    feature_fn: Optional[Callable[[pl.DataFrame], pl.DataFrame]] = None

    # step 1 --predict: the setup search space. Wide on purpose.
    predict_entry_signals: list = field(
        default_factory=lambda: ["residual", "ou"]
    )
    predict_beta_lbs: list = field(default_factory=lambda: list(range(10, 501, 10)))
    predict_ou_lbs: list = field(default_factory=lambda: list(range(10, 501, 10)))
    predict_horizons: list = field(default_factory=lambda: [5, 10, 20, 40, 60, 100])
    predict_resid_thresholds_bps: list = field(
        default_factory=lambda: list(range(11, 31, 2))
    )
    predict_ou_z_thresholds: list = field(
        default_factory=lambda: np.arange(0.5, 3.1, 0.2).tolist()
    )
    predict_gate_buckets: object = "regime"  # named quantile regimes per condition
    gate_min_history: int = 252  # causal percentile warmup before gates may fire
    predict_min_obs: int = 30  # ignore cells with fewer threshold-crossing events
    predict_min_independent_events: int = 8  # non-overlapping forecast windows
    predict_top_n: int = 10  # distinct setups saved to setups_file for --exit
    predict_min_neighbors: int = 3  # corroborating grid neighbors a setup needs

    # step 2 --exit: exit styles per saved setup
    #   band            flat when |signal| <= band (0.0 = hold-until-reversal)
    #   revert_frac     exit once this fraction of the entry dislocation reverted
    #   half_life_frac  time stop at frac x the half-life measured at entry
    exit_styles: list = field(
        default_factory=lambda: ["band", "revert_frac", "half_life_frac"]
    )
    exit_resid_bands_bps: list = field(
        default_factory=lambda: [0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0]
    )
    exit_ou_z_bands: list = field(
        default_factory=lambda: [
            0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5,
        ]
    )
    exit_revert_fracs: list = field(default_factory=lambda: [0.25, 0.5, 0.75, 1.0])
    exit_half_life_fracs: list = field(
        default_factory=lambda: [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    )
    exit_min_trades: int = 8  # floor for leaderboards and the saved winners

    # step 3 --sweep: hard-stop overlay + robustness ranking
    sweep_stop_loss_bps: list = field(default_factory=lambda: [15.0, 25.0, 40.0])
    era_years: int = 4  # era length for the consistency check (pnl > 0 per era)
    validation_slices: int = 8  # CSCV partitions for finalist PBO

    def __post_init__(self):
        # Keep generated funnel artifacts grouped by strategy instead of
        # allowing them to accumulate beside the strategy modules.
        self.data_dir = self.path.parent / "data" / self.name
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.setups_file = self.data_dir / f"{self.name}_setups.parquet"
        self.exits_file = self.data_dir / f"{self.name}_exits.parquet"
        self.exit_results_file = self.data_dir / f"{self.name}_exit_results.parquet"
        self.sweep_results_file = (
            self.data_dir / f"{self.name}_sweep_results.parquet"
        )
        self.trades_file = self.data_dir / f"{self.name}_trades.parquet"
        self.validation_file = self.data_dir / f"{self.name}_validation.parquet"

        # bind once so module-level aliases and pipeline.compute_fn are the
        # same objects across every access (identity matters to the lab
        # worker contract and to tests)
        self.compute = self.compute  # type: ignore[method-assign]
        self.make_pipeline = self.make_pipeline  # type: ignore[method-assign]
        self.pipeline = self.make_pipeline()

    # -- data ---------------------------------------------------------------

    @property
    def model_columns(self) -> list[str]:
        return [self.target, self.feature]

    def load_data(self, start: str | None = None) -> pl.DataFrame:
        """Load the ticker panel from md.index_eod and add derived features."""
        data = load_wide(
            self.tickers, start=start or self.start, bps_cols=self.bps_cols
        )
        if self.feature_fn is not None:
            data = self.feature_fn(data)
        return data.with_columns(pl.col(self.model_columns).round(2))

    def model_frame(self, data: pl.DataFrame) -> pl.DataFrame:
        """Common-sample frame used by this model: ts, target, feature."""
        return align_columns(data, self.model_columns)

    def _data(self, use_db: bool) -> pl.DataFrame:
        if use_db:
            return self.load_data()
        if self.synthetic_fn is None:
            raise ValueError(f"{self.name}: no synthetic_fn configured")
        data = self.synthetic_fn()
        if self.feature_fn is not None:
            data = self.feature_fn(data)
        return data.with_columns(pl.col(self.model_columns).round(2))

    # -- signal construction ------------------------------------------------

    def _params(self, params: dict | None = None) -> dict:
        """Merge params and accept old param names as aliases."""
        raw = params or {}
        p = {**self.default_params, **raw}
        if "z_lb" in raw and "ou_lb" not in raw:
            p["ou_lb"] = raw["z_lb"]
        if "entry_resid_bps" in raw and "entry_threshold" not in raw:
            p["entry_threshold"] = raw["entry_resid_bps"]
        if "exit_reversion_frac" in raw and "exit_style" not in raw:
            p["exit_style"], p["exit_param"] = "revert_frac", raw["exit_reversion_frac"]
        return p

    def _gate_condition(self, frame: pl.DataFrame, p: dict) -> pl.Series:
        """Condition series for the gate param — the same menu as the
        fast/predict scan conditions, built from this signal frame."""
        name, _, _ = parse_gate(p["gate"])
        builders = {
            "r2": lambda: frame["r2"],
            "beta_cv": lambda: beta_cv(frame["beta"], lookback=p["beta_lb"]),
            "beta": lambda: frame["beta"],
            "beta_vol20": lambda: frame["beta"].diff().rolling_std(20),
            "beta_mom10": lambda: frame["beta"].diff(10),
            "r2_vol20": lambda: frame["r2"].diff().rolling_std(20),
            "r2_mom10": lambda: frame["r2"].diff(10),
            "resid_phi": lambda: frame["ou_rho"] - 1.0,
            "resid_half_life": lambda: frame["half_life"],
            "resid_vol20": lambda: frame["resid"].diff().rolling_std(20),
            "resid_mom10": lambda: frame["resid"].diff(10),
        }
        if name not in builders:
            raise ValueError(
                f"unknown gate condition {name!r}; known: {sorted(builders)}"
            )
        return builders[name]()

    def compute(self, data: pl.DataFrame, params: dict | None = None) -> pl.DataFrame:
        """Signal frame: beta-weighted target-vs-feature residual and its OU
        state. The tradable "signal" column follows params["entry_signal"]:
        the raw residual in bps, or its OU z-score."""
        p = self._params(params)
        if p["entry_signal"] not in {"residual", "ou_z"}:
            raise ValueError(
                f"unknown entry_signal={p['entry_signal']!r}; "
                "expected 'residual' or 'ou_z'"
            )

        y = data[self.target]
        x = data[self.feature]
        reg = roll_lr_diff(x, y, lookback=p["beta_lb"])

        # roll_lr_diff drops one row (first diff) - pad back to len(data)
        null1 = pl.Series([None], dtype=pl.Float64)
        resid_roll = pl.concat(
            [
                null1,
                reg["resid"].rolling_sum(p["beta_lb"], min_samples=p["beta_lb"]),
            ]
        )
        beta = pl.concat([null1, reg["beta"]])
        r2 = pl.concat([null1, reg["r2"]])

        ou = roll_ou_features(resid_roll, lookback=p["ou_lb"])
        signal = resid_roll if p["entry_signal"] == "residual" else ou["ou_z"]

        frame = pl.DataFrame(
            {
                "signal": signal,
                "resid": resid_roll,
                "ou_z": ou["ou_z"],
                "ou_mean": ou["ou_mean"],
                "ou_sigma": ou["ou_sigma"],
                "ou_rho": ou["ou_rho"],
                "ou_theta": ou["ou_theta"],
                "expected_delta_1d": ou["expected_delta_1d"],
                "half_life": ou["half_life"],
                "beta": beta,
                "r2": r2,
            }
        )
        if p["exit_style"] == "half_life_frac":
            # dynamic time stop: exit_param x the half-life measured at entry
            frame = frame.with_columns(
                (ou["half_life"] * p["exit_param"]).alias("time_stop")
            )
        if p.get("gate") is not None:
            allow = gate_allow_mask(
                self._gate_condition(frame, p),
                p["gate"],
                min_history=self.gate_min_history,
            )
            frame = frame.with_columns(pl.Series("gate_allow", allow))
        return frame

    def make_pipeline(self, params: dict | None = None) -> SignalPipeline:
        """Lab contract: build the pipeline for any param combo (sweeps call
        this). Exit styles mirror the --exit scan, one primary style per
        combo (the hard stop_loss_bps always applies on top):
          band            signal exits at +/- exit_param (units follow the signal)
          revert_frac     exit once exit_param of the entry dislocation reverted
          half_life_frac  time stop at exit_param x the half-life at entry
        """
        p = self._params(params)
        style, ep = p["exit_style"], float(p["exit_param"])
        exit_long = exit_short = None
        exit_fn = None
        if style == "band":
            exit_long, exit_short = -ep, ep
        elif style == "revert_frac":
            exit_fn = profit_target(ep)
        elif style != "half_life_frac":  # half_life_frac: time_stop from compute()
            raise ValueError(
                f"unknown exit_style={style!r}; "
                "expected 'band', 'revert_frac', or 'half_life_frac'"
            )
        # the OU-z confirmation only makes sense when the entry signal is the
        # raw residual; an ou_z entry already IS the z-score
        z_gate = p["z_gate"] if p["entry_signal"] == "residual" else None
        return SignalPipeline(
            name=self.name,
            trade_def=TradeDef.outright(self.name, self.target),
            compute_fn=self.compute if params is None else partial(self.compute, params=p),
            config=SignalConfig(
                entry_long=-p["entry_threshold"],
                entry_short=p["entry_threshold"],
                exit_long=exit_long,
                exit_short=exit_short,
                stop_loss_bps=p["stop_loss_bps"],
                time_stop_bars=None,
                exit_fn=exit_fn,
                entry_filter_fn=_entry_filter(
                    z_gate, p["half_life_min"], p["half_life_max"]
                ),
            ),
        )

    # -- funnel artifacts ---------------------------------------------------

    def _neighbor_stats(self, valid: pl.DataFrame, pool_size: int = 300) -> pl.DataFrame:
        """lab.neighbor_ic_stats on this strategy's predict grid."""
        return neighbor_ic_stats(
            valid,
            beta_lbs=self.predict_beta_lbs,
            ou_lbs=self.predict_ou_lbs,
            resid_thresholds=self.predict_resid_thresholds_bps,
            z_thresholds=self.predict_ou_z_thresholds,
            pool_size=pool_size,
        )

    def _select_setups(
        self, valid: pl.DataFrame, limit: int | None = None
    ) -> pl.DataFrame:
        """Best setups by neighborhood IC, one per (signal, lookbacks, gate)
        cell. Ranks on nbr_ic rather than the cell's own IC and requires at
        least predict_min_neighbors corroborating neighbors: after a
        multi-million-cell search the top raw ICs are selection flukes unless
        the surrounding cells agree. Dedupes threshold/horizon variants of
        the same cell so the saved setups are genuinely different trades."""
        best = (
            self._neighbor_stats(valid)
            .filter(pl.col("n_nbr") >= self.predict_min_neighbors)
            .sort("nbr_ic", descending=True)
            .unique(
                subset=["entry_signal", "beta_lb", "ou_lb", "gate", "gate_bucket"],
                keep="first",
                maintain_order=True,
            )
            .head(self.predict_top_n if limit is None else limit)
            .rename({"horizon": "predict_horizon"})
            .select(
                "entry_signal", "beta_lb", "ou_lb", "entry_threshold",
                "predict_horizon", "gate", "gate_bucket",
                "ic", "nbr_ic", "n_nbr", "hit_rate", "fire_rate", "n_obs",
            )
        )
        names = [_setup_name(r) for r in best.iter_rows(named=True)]
        return best.insert_column(0, pl.Series("name", names))

    def _add_overlap_diagnostics(
        self,
        setups: pl.DataFrame,
        data: pl.DataFrame,
        scans: list[dict] | None = None,
    ) -> pl.DataFrame:
        """Measure independent forecast episodes for shortlisted setups."""
        scan_lookup = {}
        if scans is not None:
            for scan in scans:
                for column, combo in enumerate(scan["combos"]):
                    scan_lookup[
                        (
                            scan["entry_signal"],
                            int(combo["beta_lb"]),
                            int(combo.get("ou_lb", 0)),
                        )
                    ] = (scan, column)

        rows = []
        for setup in setups.iter_rows(named=True):
            entry_signal = setup["entry_signal"]
            ou_lb = 0 if entry_signal == "residual" else int(setup["ou_lb"])
            cached = scan_lookup.get(
                (entry_signal, int(setup["beta_lb"]), ou_lb)
            )
            if cached is None:
                signal_kind = (
                    "residual" if entry_signal == "residual" else "ou_zscore"
                )
                matrix, _, conditions = signal_matrix(
                    data[self.feature],
                    data[self.target],
                    [int(setup["beta_lb"])],
                    [ou_lb],
                    return_conditions=True,
                    signal_kind=signal_kind,
                    lookback_name="ou_lb",
                )
                column = 0
            else:
                scan, column = cached
                matrix, conditions = scan["matrix"], scan["conditions"]
            signal = matrix[:, column]
            previous = np.concatenate([[np.nan], signal[:-1]])
            entry = float(setup["entry_threshold"])
            crossed = (
                ((signal >= entry) & ~(previous >= entry))
                | ((signal <= -entry) & ~(previous <= -entry))
            )
            gate = setup["gate"]
            if gate in (None, "(none)"):
                gate_ok = np.ones(len(signal), dtype=bool)
            else:
                gate_ok = gate_allow_mask(
                    conditions[gate][:, column],
                    (gate, setup["gate_bucket"]),
                    min_history=self.gate_min_history,
                )
            horizon = int(setup["predict_horizon"])
            valid_forward = np.arange(len(signal)) < len(signal) - horizon
            indices = np.flatnonzero(crossed & gate_ok & valid_forward)
            rows.append(event_overlap_diagnostics(indices, horizon))
        if not rows:
            return setups.with_columns(
                pl.lit(None, dtype=pl.Int64).alias("n_non_overlapping"),
                pl.lit(None, dtype=pl.Float64).alias("overlap_fraction"),
                pl.lit(None, dtype=pl.Float64).alias("median_event_spacing"),
            )
        return setups.with_columns(
            pl.Series(
                "n_non_overlapping", [row["n_non_overlapping"] for row in rows]
            ),
            pl.Series("overlap_fraction", [row["overlap_fraction"] for row in rows]),
            pl.Series("median_event_spacing", [row["median_spacing"] for row in rows]),
        )

    def load_setups(self, path: Path | None = None) -> list[dict]:
        """Setups saved by --predict, as the dicts exit_scan iterates over."""
        path = self.setups_file if path is None else path
        if not path.exists():
            raise FileNotFoundError(f"{path} not found - run --predict first")
        setups = []
        for r in pl.read_parquet(path).iter_rows(named=True):
            setup = {
                "name": r["name"],
                "entry_signal": r["entry_signal"],
                "beta_lb": int(r["beta_lb"]),
                "entry_threshold": float(r["entry_threshold"]),
                "predict_horizon": int(r["predict_horizon"]),
                "gate": (
                    None
                    if r["gate"] in (None, "(none)")
                    else (r["gate"], r["gate_bucket"])
                ),
            }
            if r["ou_lb"] is not None:
                setup["ou_lb"] = int(r["ou_lb"])
            setups.append(setup)
        return setups

    def load_exits(self, path: Path | None = None) -> pl.DataFrame:
        """Setup + exit winners saved by --exit (one row per setup)."""
        path = self.exits_file if path is None else path
        if not path.exists():
            raise FileNotFoundError(f"{path} not found - run --exit first")
        return pl.read_parquet(path)

    # -- modes --------------------------------------------------------------

    def main(self, use_db: bool = True, params: dict | None = None) -> dict:
        """Single run: coverage, horizon diagnostics, exact backtest, and the
        latest live signal line."""
        p = self._params(params)

        raw_data = self._data(use_db)
        coverage = coverage_report(raw_data, self.model_columns)
        data = self.model_frame(raw_data)

        print(f"model: y={self.target}  x=['{self.feature}']")
        print("\ncoverage / overlap:")
        utils.pdf(coverage)
        print("\nlatest aligned rows:")
        utils.pdf(data.tail(5))
        print(
            f"raw_rows={len(raw_data)}  model_rows={len(data)}  "
            f"{data['ts'].min()} -> {data['ts'].max()}  "
            f"cols={data.columns}  (source={'db' if use_db else 'synthetic'})  params={p}"
        )

        sig_frame = self.compute(data, params=p)
        diag = horizon_backtest(sig_frame["resid"])
        print("\nresidual horizon backtest (IC / hit / Sharpe):")
        utils.pdf(diag)

        engine = Engine(BacktestConfig(transaction_cost_bps=self.transaction_cost_bps))
        result = engine.add_signal(self.make_pipeline(p)).run(data)
        print_summary(result)

        last = sig_frame.row(-1, named=True)
        sig_val = last["signal"]
        resid = last["resid"]
        ou_z = last["ou_z"]
        half_life = last["half_life"]
        gate_ok = bool(last.get("gate_allow", True))
        units = "bps" if p["entry_signal"] == "residual" else "z"
        if sig_val is None or not _finite(sig_val):
            print("\nlatest signal: warmup - no signal yet")
        else:
            hl_ok = _half_life_ok(half_life, p["half_life_min"], p["half_life_max"])
            z_gate = p["z_gate"] if p["entry_signal"] == "residual" else None
            z_ok_short = z_gate is None or (_finite(ou_z) and ou_z >= z_gate)
            z_ok_long = z_gate is None or (_finite(ou_z) and ou_z <= -z_gate)
            if gate_ok and hl_ok and sig_val >= p["entry_threshold"] and z_ok_short:
                action = f"SHORT {self.target} (rich vs {self.feature})"
            elif gate_ok and hl_ok and sig_val <= -p["entry_threshold"] and z_ok_long:
                action = f"LONG {self.target} (cheap vs {self.feature})"
            else:
                action = "FLAT"
            z_rule = "" if z_gate is None else f", abs(ou_z)>={z_gate}"
            print(
                f"\nlatest signal: ts={data['ts'][-1]}  resid={resid:+.1f}bps  "
                f"ou_z={ou_z:+.2f}  half_life={half_life:.1f}d  "
                f"beta={last['beta']:+.3f}  r2={last['r2']:.2f}  action={action}  "
                f"(abs({p['entry_signal']})>={p['entry_threshold']}{units}{z_rule}, "
                f"exit={p['exit_style']}@{p['exit_param']})"
            )

        return {
            "raw_data": raw_data,
            "coverage": coverage,
            "data": data,
            "signals": sig_frame,
            "diag": diag,
            "result": result,
        }

    def predict(self, use_db: bool = True, device: str = "auto") -> dict:
        """GPU-friendly forward-horizon predictability scan with gate buckets."""
        entry_signals = _normalize_predict_signals(self.predict_entry_signals)
        raw_data = self._data(use_db)
        data = self.model_frame(raw_data)
        level = self.pipeline.trade_def.composite_series(data).to_numpy()

        scans = []

        if "residual" in entry_signals:
            resid, resid_combos, resid_conditions = signal_matrix(
                data[self.feature],
                data[self.target],
                self.predict_beta_lbs,
                [0],
                return_conditions=True,
                signal_kind="residual",
                lookback_name="ou_lb",
            )
            scans.append(
                {
                    "entry_signal": "residual",
                    "units": "bps",
                    "matrix": resid,
                    "combos": resid_combos,
                    "conditions": resid_conditions,
                    "thresholds": self.predict_resid_thresholds_bps,
                }
            )

        if "ou_z" in entry_signals:
            ou_z, ou_combos, ou_conditions = signal_matrix(
                data[self.feature],
                data[self.target],
                self.predict_beta_lbs,
                self.predict_ou_lbs,
                return_conditions=True,
                signal_kind="ou_zscore",
                lookback_name="ou_lb",
            )
            scans.append(
                {
                    "entry_signal": "ou_z",
                    "units": "z",
                    "matrix": ou_z,
                    "combos": ou_combos,
                    "conditions": ou_conditions,
                    "thresholds": self.predict_ou_z_thresholds,
                }
            )

        n_variants = (
            1 + len(scans[0]["conditions"]) * gate_variant_count(self.predict_gate_buckets)
            if scans
            else 1
        )
        n_evals = sum(
            scan["matrix"].shape[1]
            * len(scan["thresholds"])
            * len(self.predict_horizons)
            * n_variants
            for scan in scans
        )
        print(
            f"predict scan: signals={entry_signals}  "
            f"horizons={self.predict_horizons}  "
            f"model_columns={sum(scan['matrix'].shape[1] for scan in scans)}  "
            f"gate variants={n_variants}  evaluations={n_evals:,}  "
            f"(device={device})"
        )

        with utils.timed("running vectorized predictability scan"):
            result_blocks = []
            for scan in scans:
                block = predict_scan(
                    scan["matrix"],
                    level,
                    entries=scan["thresholds"],
                    horizons=self.predict_horizons,
                    combos=scan["combos"],
                    gates=scan["conditions"],
                    gate_buckets=self.predict_gate_buckets,
                    gate_min_history=self.gate_min_history,
                    device=device,
                    entry_col="entry_threshold",
                ).with_columns(
                    pl.lit(scan["entry_signal"]).alias("entry_signal"),
                    pl.lit(scan["units"]).alias("threshold_units"),
                )
                if scan["entry_signal"] == "residual":
                    block = block.with_columns(
                        pl.lit(None, dtype=pl.Int64).alias("ou_lb")
                    )
                result_blocks.append(block)

            results = pl.concat(result_blocks, how="diagonal_relaxed").sort(
                "ic", descending=True, nulls_last=True
            )

        show = [
            "entry_signal", "beta_lb", "ou_lb", "entry_threshold",
            "threshold_units", "horizon", "gate", "gate_bucket",
            "ic", "hit_rate", "fire_rate", "n_obs",
        ]
        valid = results.filter(
            (pl.col("n_obs") >= self.predict_min_obs) & pl.col("ic").is_finite()
        )
        print(
            f"\ntop 20 setups by IC, gated or ungated (n_obs >= {self.predict_min_obs}):"
        )
        utils.pdf(valid.select([c for c in show if c in valid.columns]).head(20))

        candidate_limit = max(self.predict_top_n * 10, self.predict_top_n)
        candidates = self._select_setups(valid, limit=candidate_limit)
        candidates = self._add_overlap_diagnostics(candidates, data, scans=scans)
        rejected = candidates.filter(
            pl.col("n_non_overlapping") < self.predict_min_independent_events
        )
        setups = (
            candidates.filter(
                pl.col("n_non_overlapping") >= self.predict_min_independent_events
            )
            .head(self.predict_top_n)
        )
        setups.write_parquet(self.setups_file)
        print(
            f"\ntop {len(setups)} distinct setups by neighborhood IC "
            f"(>= {self.predict_min_neighbors} neighbors, >= "
            f"{self.predict_min_independent_events} non-overlapping events), "
            f"saved for --exit "
            f"-> {self.setups_file}:"
        )
        utils.pdf(setups)
        if not rejected.is_empty():
            print(
                f"  rejected {len(rejected)} shortlisted cells with fewer than "
                f"{self.predict_min_independent_events} independent forecast windows"
            )

        return {"data": data, "results": results, "setups": setups}

    def exit_scan(self, use_db: bool = True, device: str = "auto") -> dict:
        """Vectorized exit scan: every (setup, entry, exit style, exit param)
        as an approximate trade backtest — which exits pay, how long they
        hold, at what hit rate. Setups come from setups_file (saved by
        --predict); the best exit per setup is saved to exits_file for
        --sweep. Bands run on device; the stateful styles (revert_frac,
        half_life_frac) are CPU."""
        setups = self.load_setups()
        raw_data = self._data(use_db)
        data = self.model_frame(raw_data)
        level = self.pipeline.trade_def.composite_series(data).to_numpy()

        scans = []
        for setup in setups:
            entry_signal = setup["entry_signal"]
            entry = float(setup["entry_threshold"])
            predict_horizon = int(setup["predict_horizon"])
            if entry_signal == "residual":
                signal_kind, units, lookbacks = "residual", "bps", [0]
                exits = [b for b in self.exit_resid_bands_bps if b < entry]
            elif entry_signal == "ou_z":
                signal_kind, units = "ou_zscore", "z"
                lookbacks = [int(setup["ou_lb"])]
                exits = [b for b in self.exit_ou_z_bands if b < entry]
            else:
                raise ValueError(f"unknown setup entry_signal={entry_signal!r}")

            matrix, combos, conditions = signal_matrix(
                data[self.feature],
                data[self.target],
                [int(setup["beta_lb"])],
                lookbacks,
                return_conditions=True,
                signal_kind=signal_kind,
                lookback_name="ou_lb",
            )

            gate_spec = setup.get("gate")
            if gate_spec is None:
                gate_name, gate_bucket = "(none)", "all"
                gate_ok = np.ones_like(matrix, dtype=bool)
            else:
                gate_name, _, _ = parse_gate(gate_spec)
                gate_bucket = (
                    gate_spec.get("bucket", gate_spec.get("kind"))
                    if isinstance(gate_spec, dict)
                    else gate_spec[1]
                )
                gate_ok = gate_allow_mask(
                    conditions[gate_name][:, 0],
                    gate_spec,
                    min_history=self.gate_min_history,
                )[:, None]

            # Match predict_scan's trigger event exactly: the gate must be
            # valid on the first bar crossing this setup's threshold.
            prev = np.concatenate([np.full((1, 1), np.nan), matrix[:-1]], axis=0)
            crossed = (
                ((matrix >= entry) & ~(prev >= entry))
                | ((matrix <= -entry) & ~(prev <= -entry))
            )
            forward = np.full((len(matrix), 1), np.nan)
            if predict_horizon < len(matrix):
                forward[:-predict_horizon, 0] = (
                    level[predict_horizon:] - level[:-predict_horizon]
                )
            combos = [
                {
                    **combos[0],
                    "setup": setup["name"],
                    "predict_horizon": predict_horizon,
                }
            ]
            scans.append(
                {
                    "entry_signal": entry_signal,
                    "units": units,
                    "matrix": matrix,
                    "combos": combos,
                    "entries": [entry],
                    "exits": exits,
                    "half_life": conditions["resid_half_life"],
                    "entry_allow": crossed & gate_ok & np.isfinite(forward),
                    "gate": gate_name,
                    "gate_bucket": str(gate_bucket),
                }
            )

        def _style_width(scan: dict) -> int:
            return (
                (len(scan["exits"]) if "band" in self.exit_styles else 0)
                + (len(self.exit_revert_fracs) if "revert_frac" in self.exit_styles else 0)
                + (
                    len(self.exit_half_life_fracs)
                    if "half_life_frac" in self.exit_styles
                    else 0
                )
            )

        n_evals = sum(
            scan["matrix"].shape[1] * len(scan["entries"]) * _style_width(scan)
            for scan in scans
        )
        print(
            f"exit scan: setups={len(setups)} (from {self.setups_file.name})  "
            f"styles={self.exit_styles}  "
            f"model_columns={sum(scan['matrix'].shape[1] for scan in scans)}  "
            f"evaluations={n_evals:,}  (device={device})"
        )

        n_tasks = len(scans) * len(self.exit_styles)
        t0 = time.time()
        task = 0

        def _task_progress(label: str):
            """Per-bar progress line for the stateful styles."""

            def cb(bar: int, total_bars: int) -> None:
                print(
                    f"\r  [{task}/{n_tasks}] {label}: bar {bar:,}/{total_bars:,}  "
                    f"({time.time() - t0:.1f}s elapsed) ",
                    end="",
                    flush=True,
                )

            return cb

        def _done(label: str, block_t0: float) -> None:
            print(
                f"\r  [{task}/{n_tasks}] {label}: done in {time.time() - block_t0:.1f}s"
                + " " * 30
            )

        result_blocks = []
        for scan in scans:
            styled = []
            if "band" in self.exit_styles:
                task += 1
                bt = time.time()
                styled.append(
                    fast_scan(
                        scan["matrix"],
                        level,
                        entries=scan["entries"],
                        exit_band=scan["exits"],
                        cost_bps=self.transaction_cost_bps,
                        combos=scan["combos"],
                        entry_allow=scan["entry_allow"],
                        device=device,
                        entry_col="entry_threshold",
                        exit_col="exit_threshold",
                    )
                    .drop("gate", "gate_bucket")  # ungated scan: constant columns
                    .with_columns(pl.lit("band").alias("exit_style"))
                )
                _done(f"{scan['entry_signal']} band", bt)
            if "revert_frac" in self.exit_styles:
                task += 1
                bt = time.time()
                styled.append(
                    stateful_exit_scan(
                        scan["matrix"],
                        level,
                        entries=scan["entries"],
                        exit_style="revert_frac",
                        exit_params=self.exit_revert_fracs,
                        entry_allow=scan["entry_allow"],
                        cost_bps=self.transaction_cost_bps,
                        combos=scan["combos"],
                        progress=_task_progress(f"{scan['entry_signal']} revert_frac"),
                    ).with_columns(pl.lit("revert_frac").alias("exit_style"))
                )
                _done(f"{scan['entry_signal']} revert_frac", bt)
            if "half_life_frac" in self.exit_styles:
                task += 1
                bt = time.time()
                styled.append(
                    stateful_exit_scan(
                        scan["matrix"],
                        level,
                        entries=scan["entries"],
                        exit_style="half_life_frac",
                        exit_params=self.exit_half_life_fracs,
                        half_life=scan["half_life"],
                        entry_allow=scan["entry_allow"],
                        cost_bps=self.transaction_cost_bps,
                        combos=scan["combos"],
                        progress=_task_progress(
                            f"{scan['entry_signal']} half_life_frac"
                        ),
                    ).with_columns(pl.lit("half_life_frac").alias("exit_style"))
                )
                _done(f"{scan['entry_signal']} half_life_frac", bt)
            block = pl.concat(styled, how="diagonal_relaxed").with_columns(
                pl.lit(scan["entry_signal"]).alias("entry_signal"),
                pl.lit(scan["units"]).alias("threshold_units"),
                pl.lit(scan["gate"]).alias("gate"),
                pl.lit(scan["gate_bucket"]).alias("gate_bucket"),
            )
            if scan["entry_signal"] == "residual":
                block = block.with_columns(pl.lit(None, dtype=pl.Int64).alias("ou_lb"))
            result_blocks.append(block)
        print(f"  exit scan done in {time.time() - t0:.1f}s")

        results = (
            pl.concat(result_blocks, how="diagonal_relaxed")
            .with_columns(
                pl.when(pl.col("n_trades") > 0)
                .then(pl.col("total_pnl_bps") / pl.col("n_trades"))
                .otherwise(None)
                .alias("pnl_per_trade_bps"),
                pl.when(pl.col("n_trades") > 0)
                .then(pl.col("n_bars_active") / pl.col("n_trades"))
                .otherwise(None)
                .alias("avg_hold_bars"),
            )
            .sort("sharpe", descending=True, nulls_last=True)
        )

        results.write_parquet(self.exit_results_file)

        valid = results.filter((pl.col("n_trades") > 0) & pl.col("sharpe").is_finite())

        exit_summary = (
            valid.group_by(
                "setup", "entry_signal", "beta_lb", "ou_lb", "entry_threshold",
                "predict_horizon", "gate", "gate_bucket", "exit_style",
                "exit_threshold",
            )
            .agg(
                pl.col("sharpe").median().alias("med_sharpe"),
                (pl.col("sharpe") > 0).mean().alias("pct_combos_positive"),
                pl.col("hit_rate").median().alias("med_hit_rate"),
                pl.col("pnl_per_trade_bps").median().alias("med_pnl_per_trade_bps"),
                pl.col("avg_hold_bars").median().alias("med_hold_bars"),
                pl.col("n_trades").median().alias("med_n_trades"),
            )
            .sort("med_sharpe", descending=True, nulls_last=True)
        )

        def _compact_board(frame: pl.DataFrame) -> pl.DataFrame:
            """Terminal-width view; full detail remains in the results file."""
            return frame.select(
                "setup",
                pl.col("predict_horizon").alias("h"),
                pl.col("exit_style").alias("exit"),
                pl.col("exit_threshold").round(2).alias("x"),
                pl.col("sharpe").round(3),
                pl.col("total_pnl_bps").round(1).alias("pnl"),
                pl.col("pnl_per_trade_bps").round(2).alias("pnl/trd"),
                (pl.col("hit_rate") * 100).round(1).alias("hit%"),
                pl.col("avg_hold_bars").round(1).alias("hold"),
                pl.col("n_trades").alias("n"),
            )

        robust = valid.filter(pl.col("n_trades") >= self.exit_min_trades)
        leaderboard = robust if not robust.is_empty() else valid
        sample_rule = (
            f"n_trades >= {self.exit_min_trades}"
            if not robust.is_empty()
            else f"all rows; none reached n_trades >= {self.exit_min_trades}"
        )
        for label, by in [
            ("sharpe", ["sharpe"]),
            ("hit rate", ["hit_rate", "sharpe"]),
            ("total pnl", ["total_pnl_bps", "sharpe"]),
        ]:
            print(
                f"\ntop 10 setups by {label} ({sample_rule}; "
                "approximate - verify via --sweep):"
            )
            board = leaderboard.sort(by, descending=[True] * len(by), nulls_last=True)
            utils.pdf(_compact_board(board.head(10)))

        # per setup (signal, lookbacks, entry), which exit takes the best sharpe
        best_exit = (
            leaderboard.sort("sharpe", descending=True)
            .group_by(
                "setup", "entry_signal", "beta_lb", "ou_lb", "entry_threshold",
                "predict_horizon", "gate", "gate_bucket",
                maintain_order=True,
            )
            .first()
        )
        winners = best_exit.sort("sharpe", descending=True).select(
            "setup", "entry_signal", "beta_lb", "ou_lb", "entry_threshold",
            "predict_horizon", "gate", "gate_bucket", "exit_style",
            "exit_threshold", "sharpe", "total_pnl_bps", "hit_rate",
            "pnl_per_trade_bps", "avg_hold_bars", "n_trades",
        )
        winners.write_parquet(self.exits_file)
        print(
            f"\nbest exit per setup ({sample_rule}), saved for --sweep "
            f"-> {self.exits_file}:"
        )
        utils.pdf(_compact_board(winners))

        print(f"\nsaved {len(results):,} exit rows -> {self.exit_results_file}")

        return {
            "data": data,
            "results": results,
            "exit_summary": exit_summary,
            "winners": winners,
        }

    def _sweep_grids(self, winners: pl.DataFrame | None = None) -> list[dict]:
        """One exact-engine sub-grid per (setup, exit) winner saved by --exit:
        the winner's params held fixed, crossed with the hard-stop overlay."""
        if winners is None:
            winners = self.load_exits()
        grids = []
        for row in winners.iter_rows(named=True):
            grid = {k: [v] for k, v in _winner_params(row).items()}
            grid["stop_loss_bps"] = self.sweep_stop_loss_bps
            grids.append(grid)
        return grids

    def _robustness(self, trades: pl.DataFrame, data: pl.DataFrame) -> pl.DataFrame:
        """Concentration / consistency ranking of the sweep trade log.

        Per setup: pnl and sharpe with the single best trade REMOVED (a
        promotable setup survives losing its luckiest trade), the best
        trade's share of total pnl, the median trade, and pnl>0 per
        era_years era. Daily pnl is gross re-marking so with/ex-best are
        computed identically. Ranked by sharpe_ex_best — this is the
        promotion ordering."""
        dates = data["ts"].to_list()
        date_ix = {d: i for i, d in enumerate(dates)}
        level = self.pipeline.trade_def.composite_series(data).to_numpy().astype(float)
        dlevel = np.concatenate([[0.0], np.diff(level)])
        y0 = dates[0].year
        n_eras = max(1, (dates[-1].year - y0 + 1) // self.era_years)

        rows = []
        for setup in trades["setup"].unique(maintain_order=True).to_list():
            st = trades.filter(pl.col("setup") == setup).sort(
                "pnl_bps", descending=True
            )
            total = float(st["pnl_bps"].sum())
            best = float(st["pnl_bps"][0])
            era_pnl = [0.0] * n_eras
            for t in st.iter_rows(named=True):
                era = min((t["entry_date"].year - y0) // self.era_years, n_eras - 1)
                era_pnl[era] += t["pnl_bps"]
            rows.append({
                "setup": setup,
                "n_trades": len(st),
                "total_pnl_bps": round(total, 1),
                "pnl_ex_best": round(total - best, 1),
                "best_trade_share": round(best / total, 2) if total > 0 else None,
                "median_trade_bps": round(float(st["pnl_bps"].median()), 2),
                "sharpe": round(
                    _ann_sharpe(_daily_pnl_from_trades(st, date_ix, dlevel)), 3
                ),
                "sharpe_ex_best": round(
                    _ann_sharpe(_daily_pnl_from_trades(st.slice(1), date_ix, dlevel)),
                    3,
                ),
                "eras_pos": f"{sum(p > 0 for p in era_pnl)}/{n_eras}",
            })
        return pl.DataFrame(rows).sort("sharpe_ex_best", descending=True)

    def _selection_validation(self, results: pl.DataFrame) -> pl.DataFrame:
        """DSR/PBO diagnostics for the exact finalist selection stage.

        This deliberately says *finalists*: the predict and exit scans are
        earlier selection stages whose full candidate return paths are not in
        this matrix. These figures are therefore a lower bound on the full
        research process's selection penalty, not a clean bill of health.
        """
        if "daily_pnl" not in results.columns or results.is_empty():
            return pl.DataFrame()
        paths = np.column_stack(
            [np.asarray(path, dtype=float) for path in results["daily_pnl"].to_list()]
        )
        sharpes = results["sharpe"].to_numpy().astype(float)
        n_eff, mean_corr = effective_number_of_trials(paths)
        selected = paths[:, 0]  # results are already sorted by exact Sharpe
        dsr_effective = deflated_sharpe_ratio(
            selected, sharpes, independent_trials=n_eff
        )
        dsr_raw = deflated_sharpe_ratio(
            selected, sharpes, independent_trials=float(len(sharpes))
        )

        row = {
            "scope": "exact_finalists_only",
            "n_trials": len(sharpes),
            "implied_independent_trials": n_eff,
            "mean_trial_correlation": mean_corr,
            "selected_sharpe": dsr_effective["selected_sharpe"],
            "expected_max_sharpe": dsr_effective["expected_max_sharpe"],
            "dsr": dsr_effective["dsr"],
            "expected_max_sharpe_raw_n": dsr_raw["expected_max_sharpe"],
            "dsr_raw_n": dsr_raw["dsr"],
            "return_skewness": dsr_effective["skewness"],
            "return_kurtosis": dsr_effective["kurtosis"],
            "n_observations": int(dsr_effective["n_obs"]),
            "pbo": None,
            "probability_oos_loss": None,
            "mean_is_sharpe": None,
            "mean_oos_sharpe": None,
            "mean_degradation": None,
            "median_oos_rank": None,
            "cscv_combinations": 0,
        }
        if len(sharpes) >= 2:
            n_slices = min(self.validation_slices, len(selected))
            n_slices -= n_slices % 2
            pbo = probability_of_backtest_overfitting(
                paths, n_slices=max(2, n_slices)
            )
            row.update({
                "pbo": pbo.pbo,
                "probability_oos_loss": pbo.probability_of_loss,
                "mean_is_sharpe": pbo.mean_is_sharpe,
                "mean_oos_sharpe": pbo.mean_oos_sharpe,
                "mean_degradation": pbo.mean_degradation,
                "median_oos_rank": pbo.median_oos_rank,
                "cscv_combinations": pbo.n_combinations,
            })
        return pl.DataFrame([row])

    def sweep(self, use_db: bool = True, n_jobs: int | None = None) -> dict:
        """Exact-engine sweep: one full backtest (stops, costs, trade
        mechanics) per saved (setup, exit) winner x hard stop, parallel
        across CPU cores, with live progress. Also re-runs each winner at
        its best stop to save the full trade log (trades_file), and prints
        the robustness board — the promotion ranking."""
        winners = self.load_exits()  # fail fast if --exit hasn't been run
        grids = self._sweep_grids(winners)
        raw_data = self._data(use_db)
        data = self.model_frame(raw_data)
        source = "db" if use_db else "synthetic"

        total = sum(len(ParamGrid(g)) for g in grids)
        print(
            f"sweep: {self.name}  winners={len(grids)} (from {self.exits_file.name})  "
            f"stops={self.sweep_stop_loss_bps}  combos={total:,}  rows={len(data)}  "
            f"(source={source})"
        )

        t0 = time.time()
        done_base = 0
        blocks = []
        for grid in grids:

            def _progress(done: int, sub_total: int, base: int = done_base) -> None:
                done_all = base + done
                if done_all % 20 == 0 or done == sub_total:
                    elapsed = time.time() - t0
                    rate = done_all / max(elapsed, 1e-9)
                    eta = (total - done_all) / max(rate, 1e-9)
                    print(
                        f"\r  {done_all:,}/{total:,} combos ({done_all / total:.0%})  "
                        f"{rate:,.1f}/s  eta {eta:,.0f}s ",
                        end="",
                        flush=True,
                    )

            blocks.append(
                sweep_strategy(
                    self.module,
                    data,
                    grid,
                    transaction_cost_bps=self.transaction_cost_bps,
                    n_jobs=n_jobs,
                    progress=_progress,
                    return_paths=True,
                )
            )
            done_base += len(blocks[-1])
        print(f"\n  done in {time.time() - t0:.1f}s")

        results = pl.concat(blocks, how="diagonal_relaxed").sort(
            "sharpe", descending=True, nulls_last=True
        )
        if "error" in results.columns:
            errors = results.filter(pl.col("error").is_not_null())
            if not errors.is_empty():
                print(
                    f"\nWARNING: {len(errors)} combos errored; "
                    f"first: {errors['error'][0]}"
                )
            results = results.filter(pl.col("error").is_null())

        validation = self._selection_validation(results)
        results = results.drop("daily_pnl")
        results.write_parquet(self.sweep_results_file)
        validation.write_parquet(self.validation_file)

        show = [
            "entry_signal", "beta_lb", "ou_lb", "entry_threshold", "exit_style",
            "exit_param", "stop_loss_bps", "gate", "sharpe", "total_pnl_bps",
            "hit_rate", "n_trades", "max_drawdown_bps",
        ]
        print("\ntop 10 by sharpe (raw engine ranking):")
        board = results.sort("sharpe", descending=True, nulls_last=True)
        utils.pdf(board.select([c for c in show if c in board.columns]).head(10))

        # trade log: one more engine run per winner at its best stop, so the
        # comparison app can chart every entry/exit
        trade_frames = []
        for row, block in zip(winners.iter_rows(named=True), blocks):
            ok = block
            if "error" in ok.columns:
                ok = ok.filter(pl.col("error").is_null())
            if ok.is_empty():
                continue
            best = ok.sort("sharpe", descending=True, nulls_last=True).row(0, named=True)
            p = {**_winner_params(row), "stop_loss_bps": float(best["stop_loss_bps"])}
            result = (
                Engine(BacktestConfig(transaction_cost_bps=self.transaction_cost_bps))
                .add_signal(self.make_pipeline(p))
                .run(data)
            )
            trade_frames.append(
                trade_log(result.closed_trades).with_columns(
                    pl.lit(row["setup"]).alias("setup"),
                    pl.lit(p["stop_loss_bps"]).alias("stop_loss_bps"),
                )
            )
        trades = (
            pl.concat(trade_frames, how="diagonal_relaxed")
            if trade_frames
            else pl.DataFrame()
        )
        trades.write_parquet(self.trades_file)

        robustness = (
            self._robustness(trades, data) if not trades.is_empty() else pl.DataFrame()
        )
        if not robustness.is_empty():
            print(
                "\nrobustness - the promotion ranking (by sharpe EX best trade; "
                "gross re-marked pnl):"
            )
            utils.pdf(robustness)

        if not validation.is_empty():
            print(
                "\nselection diagnostics - exact finalists only "
                "(earlier funnel trials are not included):"
            )
            utils.pdf(validation.select(
                "n_trials",
                pl.col("implied_independent_trials").round(1).alias("n_eff"),
                pl.col("mean_trial_correlation").round(3).alias("avg_corr"),
                pl.col("selected_sharpe").round(3).alias("sr"),
                pl.col("expected_max_sharpe").round(3).alias("sr_hurdle"),
                pl.col("dsr").round(3),
                pl.col("pbo").cast(pl.Float64).round(3),
                pl.col("probability_oos_loss").cast(pl.Float64).round(3).alias("p_loss"),
                pl.col("mean_degradation").cast(pl.Float64).round(3).alias("degradation"),
            ))

        print(f"\nsaved {len(results)} runs -> {self.sweep_results_file}")
        print(
            f"saved {len(trades)} trades across {len(trade_frames)} setups "
            f"-> {self.trades_file}"
        )
        print(f"saved selection diagnostics -> {self.validation_file}")

        return {"data": data, "results": results, "trades": trades,
                "robustness": robustness, "validation": validation}

    def cook(
        self,
        use_db: bool = True,
        device: str = "auto",
        n_jobs: int | None = None,
    ) -> dict:
        """The whole funnel in one shot: --predict -> --exit -> --sweep.

        Each step saves its winners for the next, exactly as when run
        separately; the returned dict carries all three states."""
        line = "=" * 72
        print(f"{line}\ncooking {self.name}  [1/3] --predict\n{line}")
        predict_state = self.predict(use_db=use_db, device=device)
        print(f"\n{line}\ncooking {self.name}  [2/3] --exit\n{line}")
        exit_state = self.exit_scan(use_db=use_db, device=device)
        print(f"\n{line}\ncooking {self.name}  [3/3] --sweep\n{line}")
        sweep_state = self.sweep(use_db=use_db, n_jobs=n_jobs)
        print(f"\n{line}\n{self.name} cooked.\n{line}")
        return {"predict": predict_state, "exit": exit_state, "sweep": sweep_state}

    # -- CLI ----------------------------------------------------------------

    def cli(self, argv: list[str] | None = None) -> dict:
        """Standard strategy-module CLI: modes --predict | --exit | --sweep |
        --cook (all three) (default: single run), flags --synthetic --cpu
        --gpu."""
        args = set(sys.argv[1:] if argv is None else argv)
        known = {
            "--synthetic", "--cpu", "--gpu",
            "--sweep", "--predict", "--exit", "--exits", "--fast", "--cook",
        }
        unknown = args - known
        if unknown:
            sys.exit(
                f"unknown argument(s): {sorted(unknown)}\n"
                "modes: --predict | --exit | --sweep | --cook (all three) "
                "(default: single run)  flags: --synthetic --cpu --gpu"
            )
        use_db = "--synthetic" not in args
        device = "cpu" if "--cpu" in args else ("gpu" if "--gpu" in args else "auto")
        if "--cook" in args:
            return self.cook(use_db=use_db, device=device)
        if "--sweep" in args:
            return self.sweep(use_db=use_db)
        if "--predict" in args:
            return self.predict(use_db=use_db, device=device)
        if args & {"--exit", "--exits", "--fast"}:  # --exits/--fast: aliases
            return self.exit_scan(use_db=use_db, device=device)
        return self.main(use_db=use_db)


def synthetic_pair(
    target: str,
    feature: str,
    n: int = 1500,
    seed: int = 21,
    start: str = "2010-01-01",
    feature_level: float = 350.0,
    target_base: float = 50.0,
    beta: float = 0.25,
) -> pl.DataFrame:
    """Standard synthetic (feature, target) pair: target explained by the
    feature plus an OU residual (half-life ~14d). The default synthetic_fn
    for any single-pair Strategy."""
    rng = np.random.default_rng(seed)

    x = feature_level + np.cumsum(rng.normal(0.0, 2.0, n))

    resid = np.zeros(n)
    theta, sigma = 0.05, 2.0
    for i in range(1, n):
        resid[i] = resid[i - 1] * (1 - theta) + rng.normal(0.0, sigma)

    y = target_base + beta * (x - feature_level) + resid

    start_date = dt.date.fromisoformat(start)
    ts = pl.date_range(
        start_date, start_date + dt.timedelta(days=2 * n), interval="1d", eager=True
    )
    ts = ts.filter(ts.dt.weekday() <= 5)[:n]

    return pl.DataFrame({"ts": ts, feature: x, target: y}).with_columns(
        pl.col([feature, target]).round(2)
    )
