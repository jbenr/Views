"""10s vs 10s30s - direction/curve interaction, scriptable standard output.

Question: is the 10s30s curve rich or cheap versus the level of 10Y rates?

Model:
    y = TARGET = 10s30s curve
    x = FEATURE = 10Y yield

Fit a changes-based rolling OLS of d(TARGET) on d(FEATURE), roll the daily
residuals into curve-bps level space, and fade raw residual extremes. OU state
is used as a gate and for exits/time stops rather than as the primary entry
threshold.

Directions are in CURVE space: positive residual means 10s30s is steep/rich vs
10Y -> short 10s30s; negative residual means 10s30s is flat/cheap vs 10Y ->
long 10s30s.

Backtest parameters live in DEFAULT_PARAMS. Override with main(params={...})
or search around them with the lab modes below.

    python -m book.curve.tens_10s30s              # single run, live DB
    python -m book.curve.tens_10s30s --synthetic  # single run, no DB
    python -m book.curve.tens_10s30s --predict    # setup search -> saves SETUPS_FILE
    python -m book.curve.tens_10s30s --exit       # exits per saved setup -> EXITS_FILE
    python -m book.curve.tens_10s30s --sweep      # exact engine + trade logs
    python book/curve/app.py                      # compare winners' trades (Dash)

    --cpu / --gpu force the scan device for --predict/--exit (default: auto).
    --exits / --fast are deprecated aliases for --exit.

Every mode returns a dict of state for interactive chaining: state = main().
"""

from __future__ import annotations

import datetime as dt
import math
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import polars as pl

import utils
from backtest import (
    BacktestConfig,
    Engine,
    ParamGrid,
    SignalConfig,
    SignalPipeline,
    TradeDef,
    fast_scan,
    gate_allow_mask,
    gate_variant_count,
    neighbor_ic_stats,
    parse_gate,
    predict_scan,
    print_summary,
    profit_target,
    signal_matrix,
    stateful_exit_scan,
    sweep_strategy,
    trade_log,
)
from stats import beta_cv, horizon_backtest, roll_lr_diff, roll_ou_features
from utils.market_data import align_columns, coverage_report, load_wide

# -- config -----------------------------------------------------------------

STRATEGY_FAMILY = "curve"
SIGNAL_NAME = "tens_10s30s"
MODULE = "book.curve.tens_10s30s"  # importable path, used by sweep workers

# funnel artifacts, kept next to this file: each mode saves its winners here
# for the next mode to read (--predict -> SETUPS_FILE -> --exit -> EXITS_FILE
# -> --sweep)
SETUPS_FILE = Path(__file__).with_name(f"{SIGNAL_NAME}_setups.parquet")
EXITS_FILE = Path(__file__).with_name(f"{SIGNAL_NAME}_exits.parquet")

# full results of the latest --exit / --sweep run, overwritten each run
# (--predict's saved results ARE the setups file above)
EXIT_RESULTS_FILE = Path(__file__).with_name(f"{SIGNAL_NAME}_exit_results.parquet")
SWEEP_RESULTS_FILE = Path(__file__).with_name(f"{SIGNAL_NAME}_sweep_results.parquet")

# trade log from the latest --sweep: every closed trade of every winner at its
# best stop, one engine run per winner. Feeds the comparison app (app.py).
TRADES_FILE = Path(__file__).with_name(f"{SIGNAL_NAME}_trades.parquet")

START = "2010-01-01"

TICKERS = {
    "10y": "USGG10YR Index",  # % -> scaled to bps at load
    "10s30s": "USYC1030 Index",  # already quoted in bps
}
BPS_COLS = ["10y"]

TARGET = "10s30s"
FEATURES = ["10y"]
FEATURE = FEATURES[0]
MODEL_COLUMNS = [TARGET, *FEATURES]
YIELD_COLS = MODEL_COLUMNS


# -- helpers ----------------------------------------------------------------


def load_data(start: str = START) -> pl.DataFrame:
    """Load 10Y and 10s30s from md.index_eod (requires access to raptor)."""
    return load_wide(TICKERS, start=start, bps_cols=BPS_COLS).with_columns(
        pl.col(YIELD_COLS).round(2)
    )


def model_frame(data: pl.DataFrame) -> pl.DataFrame:
    """Common-sample frame used by this model: ts, target, feature(s)."""
    return align_columns(data, MODEL_COLUMNS)


def synthetic_data(n: int = 1500, seed: int = 21) -> pl.DataFrame:
    """Synthetic substitute: 10s30s target explained by 10Y plus OU residual."""
    rng = np.random.default_rng(seed)

    tens = 350.0 + np.cumsum(rng.normal(0.0, 2.0, n))

    resid = np.zeros(n)  # OU: half-life around 14d
    theta, sigma = 0.05, 2.0
    for i in range(1, n):
        resid[i] = resid[i - 1] * (1 - theta) + rng.normal(0.0, sigma)

    slope = 50.0 + 0.25 * (tens - 350.0) + resid

    start_date = dt.date.fromisoformat(START)
    ts = pl.date_range(
        start_date, start_date + dt.timedelta(days=2 * n), interval="1d", eager=True
    )
    ts = ts.filter(ts.dt.weekday() <= 5)[:n]

    return pl.DataFrame({"ts": ts, "10y": tens, "10s30s": slope}).with_columns(
        pl.col(YIELD_COLS).round(2)
    )


def _params(params: dict | None = None) -> dict:
    """Merge params and accept old param names as aliases."""
    raw = params or {}
    p = {**DEFAULT_PARAMS, **raw}
    if "z_lb" in raw and "ou_lb" not in raw:
        p["ou_lb"] = raw["z_lb"]
    if "entry_resid_bps" in raw and "entry_threshold" not in raw:
        p["entry_threshold"] = raw["entry_resid_bps"]
    if "exit_reversion_frac" in raw and "exit_style" not in raw:
        p["exit_style"], p["exit_param"] = "revert_frac", raw["exit_reversion_frac"]
    return p


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
        # direction +1 is long target curve: residual should be cheap/negative.
        # direction -1 is short target curve: residual should be rich/positive.
        if direction == 1:
            return float(ou_z) <= -z_gate
        if direction == -1:
            return float(ou_z) >= z_gate
        return False

    return fn


def _gate_condition(frame: pl.DataFrame, p: dict) -> pl.Series:
    """Condition series for the gate param — the same menu as the fast/predict
    scan conditions (lab.CONDITION_NAMES), built from this signal frame."""
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
        raise ValueError(f"unknown gate condition {name!r}; known: {sorted(builders)}")
    return builders[name]()


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


def _neighbor_stats(valid: pl.DataFrame, pool_size: int = 300) -> pl.DataFrame:
    """lab.neighbor_ic_stats on this module's predict grid."""
    return neighbor_ic_stats(
        valid,
        beta_lbs=PREDICT_BETA_LBS,
        ou_lbs=PREDICT_OU_LBS,
        resid_thresholds=PREDICT_RESID_THRESHOLDS_BPS,
        z_thresholds=PREDICT_OU_Z_THRESHOLDS,
        pool_size=pool_size,
    )


def _select_setups(valid: pl.DataFrame) -> pl.DataFrame:
    """Best setups by neighborhood IC, one per (signal, lookbacks, gate) cell.

    Ranks on nbr_ic rather than the cell's own IC and requires at least
    PREDICT_MIN_NEIGHBORS corroborating neighbors: after a multi-million-cell
    search the top raw ICs are selection flukes unless the surrounding cells
    agree. Dedupes threshold/horizon variants of the same cell so the saved
    setups are PREDICT_TOP_N genuinely different trades."""
    best = (
        _neighbor_stats(valid)
        .filter(pl.col("n_nbr") >= PREDICT_MIN_NEIGHBORS)
        .sort("nbr_ic", descending=True)
        .unique(
            subset=["entry_signal", "beta_lb", "ou_lb", "gate", "gate_bucket"],
            keep="first",
            maintain_order=True,
        )
        .head(PREDICT_TOP_N)
        .rename({"horizon": "predict_horizon"})
        .select(
            "entry_signal", "beta_lb", "ou_lb", "entry_threshold",
            "predict_horizon", "gate", "gate_bucket",
            "ic", "nbr_ic", "n_nbr", "hit_rate", "fire_rate", "n_obs",
        )
    )
    names = [_setup_name(r) for r in best.iter_rows(named=True)]
    return best.insert_column(0, pl.Series("name", names))


def load_setups(path: Path | None = None) -> list[dict]:
    """Setups saved by --predict, as the dicts exit_scan iterates over."""
    path = SETUPS_FILE if path is None else path
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


def load_exits(path: Path | None = None) -> pl.DataFrame:
    """Setup + exit winners saved by --exit (one row per setup)."""
    path = EXITS_FILE if path is None else path
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run --exit first")
    return pl.read_parquet(path)


def compute(data: pl.DataFrame, params: dict | None = None) -> pl.DataFrame:
    """Signal frame: beta-weighted target-vs-feature residual and its OU state.

    The tradable "signal" column follows params["entry_signal"]: the raw
    residual in bps, or its OU z-score."""
    p = _params(params)
    if p["entry_signal"] not in {"residual", "ou_z"}:
        raise ValueError(
            f"unknown entry_signal={p['entry_signal']!r}; expected 'residual' or 'ou_z'"
        )

    y = data[TARGET]
    x = data[FEATURE]
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
        allow = gate_allow_mask(_gate_condition(frame, p), p["gate"])
        frame = frame.with_columns(pl.Series("gate_allow", allow))
    return frame


def make_pipeline(params: dict | None = None) -> SignalPipeline:
    """Lab contract: build the pipeline for any param combo (sweeps call this).

    Exit styles mirror the --exit scan, one primary style per combo (the hard
    stop_loss_bps always applies on top):
      band            signal exits at +/- exit_param (units follow the signal)
      revert_frac     exit once exit_param of the entry dislocation reverted
      half_life_frac  time stop at exit_param x the half-life at entry
    """
    p = _params(params)
    style, ep = p["exit_style"], float(p["exit_param"])
    exit_long = exit_short = None
    exit_fn = None
    if style == "band":
        exit_long, exit_short = -ep, ep
    elif style == "revert_frac":
        exit_fn = profit_target(ep)
    elif style != "half_life_frac":  # half_life_frac: time_stop column from compute()
        raise ValueError(
            f"unknown exit_style={style!r}; "
            "expected 'band', 'revert_frac', or 'half_life_frac'"
        )
    # the OU-z confirmation only makes sense when the entry signal is the raw
    # residual; an ou_z entry already IS the z-score
    z_gate = p["z_gate"] if p["entry_signal"] == "residual" else None
    return SignalPipeline(
        name=SIGNAL_NAME,
        trade_def=TradeDef.outright(SIGNAL_NAME, TARGET),
        compute_fn=compute if params is None else partial(compute, params=p),
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


# -- strategy parameters ----------------------------------------------------
#
# The research funnel. Each step saves its winners to a parquet next to this
# file; the next step reads it — no hand-copying between blocks:
#
#   1. --predict (PREDICT_*)  cast a wide net: which (lookbacks, entry signal,
#      threshold, horizon) cells show ANY forward predictability, with every
#      gate condition/bucket as extra candidate cells. No trading mechanics —
#      just IC / hit / fire rate. GPU-vectorized. Gated and ungated setups
#      compete on raw IC in one leaderboard; the PREDICT_TOP_N best distinct
#      cells by NEIGHBORHOOD IC (median over adjacent grid cells — lone
#      spikes are search noise and get dropped) are saved to SETUPS_FILE.
#   2. --exit (EXIT_*)  reads SETUPS_FILE and runs an approximate TRADE
#      backtest of each saved setup across exit styles — threshold bands,
#      percent-of-dislocation-reverted, and half-life-scaled time stops:
#      which exits pay, how long they hold, at what hit rate and PnL per
#      trade. The best exit per setup is saved to EXITS_FILE.
#   3. --sweep (SWEEP_*)  reads EXITS_FILE and runs each (setup, exit) winner
#      through the exact row-by-row engine — stops, costs, full trade
#      mechanics — the live-simulation check before promotion. Saves each
#      winner's full trade log (TRADES_FILE) for the comparison app.

# step 1 --predict: the setup search space. Wide on purpose — lookbacks x
# entry signal x threshold x horizon, plus every gate condition/bucket as
# extra candidate cells. Winners here define the setups steps 2-3 may use.
PREDICT_ENTRY_SIGNALS = ["residual", "ou"]  # candidate entry signals
PREDICT_BETA_LBS = list(range(10, 501, 10))
PREDICT_OU_LBS = list(range(10, 501, 10))
PREDICT_HORIZONS = [5, 10, 20, 40, 60, 100]  # forward windows (days)
PREDICT_RESID_THRESHOLDS_BPS = list(range(11, 31, 2))
PREDICT_OU_Z_THRESHOLDS = np.arange(0.5, 3.1, 0.2).tolist()
PREDICT_GATE_BUCKETS = "regime"  # named quantile regimes per condition
PREDICT_MIN_OBS = 30  # ignore cells with fewer threshold-crossing events
PREDICT_TOP_N = 10  # distinct setups saved to SETUPS_FILE for --exit
PREDICT_MIN_NEIGHBORS = 3  # corroborating grid neighbors a setup needs to be saved

# step 2 --exit: exit scan over the setups saved by --predict. Each setup
# carries its entry threshold, predictive horizon, and gate so the trade scan
# tests the discovered event rather than an ungated lookalike. Three styles:
#   band            flat when |signal| <= band. 0.0 = hold-until-reversal
#                   benchmark; bands only make sense below the entry threshold.
#   revert_frac     exit once this fraction of the point-in-time entry
#                   dislocation has reverted (1.0 = full reversion).
#   half_life_frac  time stop at frac x the residual half-life measured at
#                   entry — frac of the expected time to reversion.
EXIT_STYLES = ["band", "revert_frac", "half_life_frac"]
EXIT_RESID_BANDS_BPS = [
    0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0,
]  # per setup, only bands below its entry threshold are tested
EXIT_OU_Z_BANDS = [
    0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5,
]  # per setup, only bands below its entry threshold are tested
EXIT_REVERT_FRACS = [0.25, 0.5, 0.75, 1.0]  # frac of entry dislocation reverted
EXIT_HALF_LIFE_FRACS = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]  # x entry-time half-life
EXIT_MIN_TRADES = 8  # floor for the leaderboards and the winners saved to EXITS_FILE

# step 3 --sweep: the exact-engine check over the (setup, exit) winners saved
# by --exit, each held fixed and crossed with the hard-stop overlay — the one
# mechanic the approximate exit scan can't test. Promotion ranking is the
# ROBUSTNESS board: sharpe with the best trade removed, best-trade share of
# PnL, and per-era consistency — a setup whose backtest hinges on one lucky
# trade (e.g. the Dec-21 flattener) must not outrank one that pays steadily.
SWEEP_STOP_LOSS_BPS = [15.0, 25.0, 40.0]  # hard stop overlay, always on
ERA_YEARS = 4  # era length for the consistency check (pnl > 0 per era)

# DEFAULT_PARAMS is the promoted configuration — what main() runs as the live
# signal, and the base every mode overrides from.

DEFAULT_PARAMS = {
    # model fit: hedge ratio and OU state of the residual
    "beta_lb": 252,  # hedge-ratio lookback (days)
    "ou_lb": 252,  # OU-state lookback for z/mean/half-life
    # entry: which signal to fade, at what threshold (units follow the signal)
    "entry_signal": "residual",  # "residual" (bps) | "ou_z" (z)
    "entry_threshold": 20.0,  # positive signal -> short curve
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
TRANSACTION_COST_BPS = 0.1

pipeline = make_pipeline()


# -- modes -----------------------------------------------------------------


def main(use_db: bool = True, params: dict | None = None) -> dict:
    p = _params(params)

    raw_data = load_data() if use_db else synthetic_data()
    coverage = coverage_report(raw_data, MODEL_COLUMNS)
    data = model_frame(raw_data)

    print(f"model: y={TARGET}  x={FEATURES}")
    print("\ncoverage / overlap:")
    utils.pdf(coverage)
    print("\nlatest aligned rows:")
    utils.pdf(data.tail(5))
    print(
        f"raw_rows={len(raw_data)}  model_rows={len(data)}  "
        f"{data['ts'].min()} -> {data['ts'].max()}  "
        f"cols={data.columns}  (source={'db' if use_db else 'synthetic'})  params={p}"
    )

    sig_frame = compute(data, params=p)
    diag = horizon_backtest(sig_frame["resid"])
    print("\nresidual horizon backtest (IC / hit / Sharpe):")
    utils.pdf(diag)

    engine = Engine(BacktestConfig(transaction_cost_bps=TRANSACTION_COST_BPS))
    result = engine.add_signal(make_pipeline(p)).run(data)
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
            action = f"SHORT {TARGET} (curve steep/rich vs 10Y)"
        elif gate_ok and hl_ok and sig_val <= -p["entry_threshold"] and z_ok_long:
            action = f"LONG {TARGET} (curve flat/cheap vs 10Y)"
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


def _winner_params(row: dict) -> dict:
    """Engine params for one (setup, exit) winner row from EXITS_FILE.
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


def _robustness(trades: pl.DataFrame, data: pl.DataFrame) -> pl.DataFrame:
    """Concentration / consistency ranking of the sweep trade log.

    Per setup: pnl and sharpe with the single best trade REMOVED (a
    promotable setup survives losing its luckiest trade), the best trade's
    share of total pnl, the median trade, and pnl>0 per ERA_YEARS era.
    Daily pnl is gross re-marking so with/ex-best are computed identically.
    Ranked by sharpe_ex_best — this is the promotion ordering."""
    dates = data["ts"].to_list()
    date_ix = {d: i for i, d in enumerate(dates)}
    level = pipeline.trade_def.composite_series(data).to_numpy().astype(float)
    dlevel = np.concatenate([[0.0], np.diff(level)])
    y0 = dates[0].year
    n_eras = max(1, (dates[-1].year - y0 + 1) // ERA_YEARS)

    rows = []
    for setup in trades["setup"].unique(maintain_order=True).to_list():
        st = trades.filter(pl.col("setup") == setup).sort(
            "pnl_bps", descending=True
        )
        total = float(st["pnl_bps"].sum())
        best = float(st["pnl_bps"][0])
        era_pnl = [0.0] * n_eras
        for t in st.iter_rows(named=True):
            era_pnl[min((t["entry_date"].year - y0) // ERA_YEARS, n_eras - 1)] += t[
                "pnl_bps"
            ]
        rows.append({
            "setup": setup,
            "n_trades": len(st),
            "total_pnl_bps": round(total, 1),
            "pnl_ex_best": round(total - best, 1),
            "best_trade_share": round(best / total, 2) if total > 0 else None,
            "median_trade_bps": round(float(st["pnl_bps"].median()), 2),
            "sharpe": round(_ann_sharpe(_daily_pnl_from_trades(st, date_ix, dlevel)), 3),
            "sharpe_ex_best": round(
                _ann_sharpe(_daily_pnl_from_trades(st.slice(1), date_ix, dlevel)), 3
            ),
            "eras_pos": f"{sum(p > 0 for p in era_pnl)}/{n_eras}",
        })
    return pl.DataFrame(rows).sort("sharpe_ex_best", descending=True)


def _sweep_grids(winners: pl.DataFrame | None = None) -> list[dict]:
    """One exact-engine sub-grid per (setup, exit) winner saved by --exit:
    the winner's params held fixed, crossed with the hard-stop overlay."""
    if winners is None:
        winners = load_exits()
    grids = []
    for row in winners.iter_rows(named=True):
        grid = {k: [v] for k, v in _winner_params(row).items()}
        grid["stop_loss_bps"] = SWEEP_STOP_LOSS_BPS
        grids.append(grid)
    return grids




def sweep(use_db: bool = True, n_jobs: int | None = None) -> dict:
    """Exact-engine sweep: one full backtest (stops, costs, trade mechanics)
    per saved (setup, exit) winner x hard stop, parallel across CPU cores,
    with live progress. Also re-runs each winner at its best stop to save the
    full trade log (TRADES_FILE) for the comparison app."""
    winners = load_exits()  # fail fast if --exit hasn't been run
    grids = _sweep_grids(winners)
    raw_data = load_data() if use_db else synthetic_data()
    data = model_frame(raw_data)
    source = "db" if use_db else "synthetic"

    total = sum(len(ParamGrid(g)) for g in grids)
    print(
        f"sweep: {SIGNAL_NAME}  winners={len(grids)} (from {EXITS_FILE.name})  "
        f"stops={SWEEP_STOP_LOSS_BPS}  combos={total:,}  rows={len(data)}  "
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
                MODULE,
                data,
                grid,
                transaction_cost_bps=TRANSACTION_COST_BPS,
                n_jobs=n_jobs,
                progress=_progress,
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
                f"\nWARNING: {len(errors)} combos errored; first: {errors['error'][0]}"
            )
        results = results.filter(pl.col("error").is_null())

    results.write_parquet(SWEEP_RESULTS_FILE)

    show = [
        "entry_signal",
        "beta_lb",
        "ou_lb",
        "entry_threshold",
        "exit_style",
        "exit_param",
        "stop_loss_bps",
        "gate",
        "sharpe",
        "total_pnl_bps",
        "hit_rate",
        "n_trades",
        "max_drawdown_bps",
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
            Engine(BacktestConfig(transaction_cost_bps=TRANSACTION_COST_BPS))
            .add_signal(make_pipeline(p))
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
    trades.write_parquet(TRADES_FILE)

    robustness = (
        _robustness(trades, data) if not trades.is_empty() else pl.DataFrame()
    )
    if not robustness.is_empty():
        print(
            "\nrobustness - the promotion ranking (by sharpe EX best trade; "
            "gross re-marked pnl):"
        )
        utils.pdf(robustness)

    print(f"\nsaved {len(results)} runs -> {SWEEP_RESULTS_FILE}")
    print(
        f"saved {len(trades)} trades across {len(trade_frames)} setups "
        f"-> {TRADES_FILE}"
    )

    return {"data": data, "results": results, "trades": trades,
            "robustness": robustness}


def exit_scan(use_db: bool = True, device: str = "auto") -> dict:
    """Vectorized exit scan: every (setup, entry, exit style, exit param) as
    an approximate trade backtest — which exits pay, how long they hold, at
    what hit rate. Setups come from SETUPS_FILE (saved by --predict); the
    best exit per setup is saved to EXITS_FILE for --sweep. Bands run on
    device; the stateful styles (revert_frac, half_life_frac) are CPU."""
    setups = load_setups()
    raw_data = load_data() if use_db else synthetic_data()
    data = model_frame(raw_data)
    level = pipeline.trade_def.composite_series(data).to_numpy()

    scans = []
    for setup in setups:
        entry_signal = setup["entry_signal"]
        entry = float(setup["entry_threshold"])
        predict_horizon = int(setup["predict_horizon"])
        if entry_signal == "residual":
            signal_kind, units, lookbacks = "residual", "bps", [0]
            exits = [band for band in EXIT_RESID_BANDS_BPS if band < entry]
        elif entry_signal == "ou_z":
            signal_kind, units = "ou_zscore", "z"
            lookbacks = [int(setup["ou_lb"])]
            exits = [band for band in EXIT_OU_Z_BANDS if band < entry]
        else:
            raise ValueError(f"unknown setup entry_signal={entry_signal!r}")

        matrix, combos, conditions = signal_matrix(
            data[FEATURE],
            data[TARGET],
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
            gate_ok = gate_allow_mask(conditions[gate_name][:, 0], gate_spec)[:, None]

        # Match predict_scan's trigger event exactly: the gate must be valid
        # on the first bar crossing this setup's positive/negative threshold.
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
            (len(scan["exits"]) if "band" in EXIT_STYLES else 0)
            + (len(EXIT_REVERT_FRACS) if "revert_frac" in EXIT_STYLES else 0)
            + (len(EXIT_HALF_LIFE_FRACS) if "half_life_frac" in EXIT_STYLES else 0)
        )

    n_evals = sum(
        scan["matrix"].shape[1] * len(scan["entries"]) * _style_width(scan)
        for scan in scans
    )
    print(
        f"exit scan: setups={len(setups)} (from {SETUPS_FILE.name})  "
        f"styles={EXIT_STYLES}  "
        f"model_columns={sum(scan['matrix'].shape[1] for scan in scans)}  "
        f"evaluations={n_evals:,}  (device={device})"
    )

    n_tasks = len(scans) * len(EXIT_STYLES)
    t0 = time.time()
    task = 0

    def _task_progress(label: str):
        """Per-bar progress line for the stateful styles ([task i/n] bar x/y)."""

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
        if "band" in EXIT_STYLES:
            task += 1
            bt = time.time()
            styled.append(
                fast_scan(
                    scan["matrix"],
                    level,
                    entries=scan["entries"],
                    exit_band=scan["exits"],
                    cost_bps=TRANSACTION_COST_BPS,
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
        if "revert_frac" in EXIT_STYLES:
            task += 1
            bt = time.time()
            styled.append(
                stateful_exit_scan(
                    scan["matrix"],
                    level,
                    entries=scan["entries"],
                    exit_style="revert_frac",
                    exit_params=EXIT_REVERT_FRACS,
                    entry_allow=scan["entry_allow"],
                    cost_bps=TRANSACTION_COST_BPS,
                    combos=scan["combos"],
                    progress=_task_progress(f"{scan['entry_signal']} revert_frac"),
                ).with_columns(pl.lit("revert_frac").alias("exit_style"))
            )
            _done(f"{scan['entry_signal']} revert_frac", bt)
        if "half_life_frac" in EXIT_STYLES:
            task += 1
            bt = time.time()
            styled.append(
                stateful_exit_scan(
                    scan["matrix"],
                    level,
                    entries=scan["entries"],
                    exit_style="half_life_frac",
                    exit_params=EXIT_HALF_LIFE_FRACS,
                    half_life=scan["half_life"],
                    entry_allow=scan["entry_allow"],
                    cost_bps=TRANSACTION_COST_BPS,
                    combos=scan["combos"],
                    progress=_task_progress(f"{scan['entry_signal']} half_life_frac"),
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

    results.write_parquet(EXIT_RESULTS_FILE)

    valid = results.filter((pl.col("n_trades") > 0) & pl.col("sharpe").is_finite())

    exit_summary = (
        valid.group_by(
            "setup",
            "entry_signal",
            "beta_lb",
            "ou_lb",
            "entry_threshold",
            "predict_horizon",
            "gate",
            "gate_bucket",
            "exit_style",
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
        """Terminal-width view; full detail remains in results/MetricStore."""
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

    robust = valid.filter(pl.col("n_trades") >= EXIT_MIN_TRADES)
    leaderboard = robust if not robust.is_empty() else valid
    sample_rule = (
        f"n_trades >= {EXIT_MIN_TRADES}"
        if not robust.is_empty()
        else f"all rows; none reached n_trades >= {EXIT_MIN_TRADES}"
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
            "setup",
            "entry_signal",
            "beta_lb",
            "ou_lb",
            "entry_threshold",
            "predict_horizon",
            "gate",
            "gate_bucket",
            maintain_order=True,
        )
        .first()
    )
    winners = best_exit.sort("sharpe", descending=True).select(
        "setup", "entry_signal", "beta_lb", "ou_lb", "entry_threshold",
        "predict_horizon", "gate", "gate_bucket", "exit_style", "exit_threshold",
        "sharpe", "total_pnl_bps", "hit_rate", "pnl_per_trade_bps",
        "avg_hold_bars", "n_trades",
    )
    winners.write_parquet(EXITS_FILE)
    print(f"\nbest exit per setup ({sample_rule}), saved for --sweep -> {EXITS_FILE}:")
    utils.pdf(_compact_board(winners))

    print(f"\nsaved {len(results):,} exit rows -> {EXIT_RESULTS_FILE}")

    return {
        "data": data,
        "results": results,
        "exit_summary": exit_summary,
        "winners": winners,
    }


def _normalize_predict_signals(signals: list[str]) -> list[str]:
    """Canonicalize configured predict signals and reject silent no-ops."""
    aliases = {"ou": "ou_z"}
    normalized = [aliases.get(signal, signal) for signal in signals]
    unknown = sorted(set(normalized) - {"residual", "ou_z"})
    if unknown:
        raise ValueError(
            f"unknown PREDICT_ENTRY_SIGNALS values: {unknown}; "
            "expected 'residual', 'ou', or 'ou_z'"
        )
    return list(dict.fromkeys(normalized))


def predict(use_db: bool = True, device: str = "auto") -> dict:
    """GPU-friendly forward-horizon predictability scan with gate buckets."""
    entry_signals = _normalize_predict_signals(PREDICT_ENTRY_SIGNALS)
    raw_data = load_data() if use_db else synthetic_data()
    data = model_frame(raw_data)
    level = pipeline.trade_def.composite_series(data).to_numpy()

    scans = []

    if "residual" in entry_signals:
        resid, resid_combos, resid_conditions = signal_matrix(
            data[FEATURE],
            data[TARGET],
            PREDICT_BETA_LBS,
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
                "thresholds": PREDICT_RESID_THRESHOLDS_BPS,
            }
        )

    if "ou_z" in entry_signals:
        ou_z, ou_combos, ou_conditions = signal_matrix(
            data[FEATURE],
            data[TARGET],
            PREDICT_BETA_LBS,
            PREDICT_OU_LBS,
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
                "thresholds": PREDICT_OU_Z_THRESHOLDS,
            }
        )

    n_variants = (
        1 + len(scans[0]["conditions"]) * gate_variant_count(PREDICT_GATE_BUCKETS)
        if scans
        else 1
    )
    n_evals = sum(
        scan["matrix"].shape[1]
        * len(scan["thresholds"])
        * len(PREDICT_HORIZONS)
        * n_variants
        for scan in scans
    )
    print(
        f"predict scan: signals={entry_signals}  "
        f"horizons={PREDICT_HORIZONS}  "
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
                horizons=PREDICT_HORIZONS,
                combos=scan["combos"],
                gates=scan["conditions"],
                gate_buckets=PREDICT_GATE_BUCKETS,
                device=device,
                entry_col="entry_threshold",
            ).with_columns(
                pl.lit(scan["entry_signal"]).alias("entry_signal"),
                pl.lit(scan["units"]).alias("threshold_units"),
            )
            if scan["entry_signal"] == "residual":
                block = block.with_columns(pl.lit(None, dtype=pl.Int64).alias("ou_lb"))
            result_blocks.append(block)

        results = pl.concat(result_blocks, how="diagonal_relaxed").sort(
            "ic", descending=True, nulls_last=True
        )

    show = [
        "entry_signal",
        "beta_lb",
        "ou_lb",
        "entry_threshold",
        "threshold_units",
        "horizon",
        "gate",
        "gate_bucket",
        "ic",
        "hit_rate",
        "fire_rate",
        "n_obs",
    ]
    valid = results.filter(
        (pl.col("n_obs") >= PREDICT_MIN_OBS) & pl.col("ic").is_finite()
    )
    print(f"\ntop 20 setups by IC, gated or ungated (n_obs >= {PREDICT_MIN_OBS}):")
    utils.pdf(valid.select([c for c in show if c in valid.columns]).head(20))

    setups = _select_setups(valid)
    setups.write_parquet(SETUPS_FILE)
    print(
        f"\ntop {len(setups)} distinct setups by neighborhood IC "
        f"(>= {PREDICT_MIN_NEIGHBORS} neighbors), saved for --exit -> {SETUPS_FILE}:"
    )
    utils.pdf(setups)

    return {"data": data, "results": results, "setups": setups}


if __name__ == "__main__":
    args = set(sys.argv[1:])
    known = {
        "--synthetic",
        "--cpu",
        "--gpu",
        "--sweep",
        "--predict",
        "--exit",
        "--exits",
        "--fast",
    }
    unknown = args - known
    if unknown:
        sys.exit(
            f"unknown argument(s): {sorted(unknown)}\n"
            "modes: --predict | --exit | --sweep (default: single run)  "
            "flags: --synthetic --cpu --gpu"
        )
    use_db = "--synthetic" not in args
    device = "cpu" if "--cpu" in args else ("gpu" if "--gpu" in args else "auto")
    if "--sweep" in args:
        state = sweep(use_db=use_db)
    elif "--predict" in args:
        state = predict(use_db=use_db, device=device)
    elif args & {"--exit", "--exits", "--fast"}:  # --exits/--fast: deprecated aliases
        state = exit_scan(use_db=use_db, device=device)
    else:
        state = main(use_db=use_db)
