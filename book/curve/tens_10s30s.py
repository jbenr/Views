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
    python -m book.curve.tens_10s30s --predict    # which setups/gates predict (GPU)
    python -m book.curve.tens_10s30s --exit       # which exit bands trade well (GPU)
    python -m book.curve.tens_10s30s --sweep      # exact-engine backtest, gate-aware

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

import numpy as np
import polars as pl

import utils
from backtest import (
    BacktestConfig,
    Engine,
    MetricStore,
    ParamGrid,
    SignalConfig,
    SignalPipeline,
    TradeDef,
    add_predict_lift,
    fast_scan,
    gate_allow_mask,
    gate_variant_count,
    parse_gate,
    predict_scan,
    print_summary,
    profit_target,
    signal_matrix,
    stateful_exit_scan,
    sweep_strategy,
)
from stats import beta_cv, horizon_backtest, roll_lr_diff, roll_ou_features
from utils.market_data import align_columns, coverage_report, load_wide

# -- config -----------------------------------------------------------------

STRATEGY_FAMILY = "curve"
SIGNAL_NAME = "tens_10s30s"
MODULE = "book.curve.tens_10s30s"  # importable path, used by sweep workers

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


def _entry_filter(z_gate: float | None, half_life_min: float, half_life_max: float):
    """Quantile gate + half-life sanity + optional OU-z confirmation.

    z_gate=None drops the OU-z confirmation entirely so the quantile gate
    (params["gate"] -> gate_allow) carries the entry filtering on its own.
    """

    def fn(direction: int, bar: dict) -> bool:
        # gate_allow arrives as 1.0/0.0 (engine floats extras); absent = no gate
        gate = bar.get("gate_allow")
        if gate is not None and gate != 1.0:
            return False

        half_life = bar.get("half_life")
        if not _finite(half_life):
            return False
        if not (half_life_min <= float(half_life) <= half_life_max):
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
# The research funnel. Each step has its own variable block below; each step's
# winners seed the next step's (narrower) block:
#
#   1. --predict (PREDICT_*)  cast a wide net: which (lookbacks, entry signal,
#      threshold, horizon) cells show ANY forward predictability, with every
#      gate condition/bucket overlaid as an IC lift. No trading mechanics —
#      just IC / hit / fire rate. GPU-vectorized. This is also the gate
#      discovery layer: a gate that doesn't lift IC here isn't worth carrying.
#   2. --exit (EXIT_*)  approximate TRADE backtest of the shortlisted setups
#      across exit styles — threshold bands, percent-of-dislocation-reverted,
#      and half-life-scaled time stops: which exits pay, how long they hold,
#      at what hit rate and PnL per trade.
#   3. --sweep (SWEEP_*)  the survivors, through the exact row-by-row
#      engine with stops, time stops, costs, and quantile gates — the
#      live-simulation check. Expensive per combo: keep the grid to what
#      steps 1-2 earned.

# step 1 --predict: the setup search space. Wide on purpose — lookbacks x
# entry signal x threshold x horizon, plus every gate condition/bucket as an
# IC-lift overlay. Winners here define the setups steps 2-4 are allowed to use.
PREDICT_ENTRY_SIGNALS = ["residual"] # ["residual", "ou_z"]  # candidate entry signals
PREDICT_BETA_LBS = list(range(10, 501, 10))
PREDICT_OU_LBS = list(range(10, 501, 10))
PREDICT_HORIZONS = [5, 10, 20, 40, 60, 100]  # forward windows (days)
PREDICT_RESID_THRESHOLDS_BPS = list(range(11, 31, 2))
PREDICT_OU_Z_THRESHOLDS = np.arange(0.5, 3.1, 0.2).tolist()
PREDICT_GATE_BUCKETS = "regime"  # named quantile regimes per condition
PREDICT_MIN_OBS = 30  # ignore cells with fewer threshold-crossing events

# step 2 --exit: exit scan over the step-1 setups. Narrow the lookback/entry
# lists to the step-1 winners before running. Three exit styles:
#   band            flat when |signal| <= band. 0.0 = hold-until-reversal
#                   benchmark; bands only make sense below the entry threshold.
#   revert_frac     exit once this fraction of the point-in-time entry
#                   dislocation has reverted (1.0 = full reversion).
#   half_life_frac  time stop at frac x the residual half-life measured at
#                   entry — frac of the expected time to reversion.
EXIT_STYLES = ["band", "revert_frac", "half_life_frac"]
EXIT_ENTRY_SIGNALS = ["residual", "ou_z"]
EXIT_BETA_LBS = list(range(10, 501, 10)) # [20, 60, 120, 200, 400] 
EXIT_OU_LBS = list(range(10, 501, 10)) # [20, 60, 120, 200, 400, 420]
EXIT_RESID_ENTRIES_BPS = list(range(21, 31, 2))
EXIT_RESID_BANDS_BPS = [0.0, 2.5, 5.0, 10.0, 15.0, 20.0]  # flat when |resid| <= band
EXIT_OU_Z_ENTRIES = np.arange(0.5, 3.1, 0.5).tolist()
EXIT_OU_Z_BANDS = [0.0, 0.25, 0.5, 0.75, 1.0]  # flat when |ou_z| <= band
EXIT_REVERT_FRACS = np.arange(0.1, 1.6, 0.1).tolist() # [0.25, 0.5, 0.75, 1.0]  # frac of entry dislocation reverted
EXIT_HALF_LIFE_FRACS = np.arange(0.1, 2.1, 0.1).tolist()  # x point-in-time half-life
EXIT_MIN_TRADES = 30  # floor for the setup leaderboard

# step 3 --sweep: the exact-engine grid — full trade mechanics (stops, costs,
# trade logs). Structured like the --exit block: entry signal and exit style
# are first-class dimensions, and thresholds/exit params stay in each signal's
# own units (the sweep runs one sub-grid per signal x style so units never
# cross). Every list here should be a survivor of steps 1-2: setups and gates
# from --predict, exit styles/params from --exit.
SWEEP_ENTRY_SIGNALS = ["residual", "ou_z"]
SWEEP_BETA_LBS = [60, 120]  # list(range(10, 501, 10))
SWEEP_OU_LBS = [400, 420]  # list(range(10, 501, 10))
SWEEP_RESID_ENTRIES_BPS = list(range(21, 31, 2))
SWEEP_OU_Z_ENTRIES = [1.0, 1.5, 2.0, 2.5, 3.0]
SWEEP_EXIT_STYLES = ["band", "revert_frac", "half_life_frac"]
SWEEP_RESID_BANDS_BPS = [5.0, 10.0]  # flat when |resid| <= band
SWEEP_OU_Z_BANDS = [0.25, 0.5]  # flat when |ou_z| <= band
SWEEP_REVERT_FRACS = [0.25, 0.5, 0.75]  # frac of entry dislocation reverted
SWEEP_HALF_LIFE_FRACS = [1.0, 2.0, 3.0]  # x point-in-time half-life
SWEEP_STOP_LOSS_BPS = [15.0, 25.0, 40.0]  # hard stop overlay, always on
# gates that earned their IC lift in --predict (None = ungated baseline)
SWEEP_GATES = [
    None,
    ("r2", "low_25"),
    ("r2_mom10", "low_10"),
    ("beta_cv", "below_50"),
    ("beta_vol20", "low_10"),
]

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
        hl_ok = (
            _finite(half_life) and p["half_life_min"] <= half_life <= p["half_life_max"]
        )
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


def _sweep_grids() -> list[dict]:
    """One exact-engine sub-grid per (entry signal, exit style) so entry
    thresholds and exit params always stay in that signal/style's units."""
    entries = {"residual": SWEEP_RESID_ENTRIES_BPS, "ou_z": SWEEP_OU_Z_ENTRIES}
    bands = {"residual": SWEEP_RESID_BANDS_BPS, "ou_z": SWEEP_OU_Z_BANDS}
    grids = []
    for sig in SWEEP_ENTRY_SIGNALS:
        for style in SWEEP_EXIT_STYLES:
            exit_params = {
                "band": bands[sig],
                "revert_frac": SWEEP_REVERT_FRACS,
                "half_life_frac": SWEEP_HALF_LIFE_FRACS,
            }[style]
            grids.append({
                "entry_signal": [sig],
                "beta_lb": SWEEP_BETA_LBS,
                "ou_lb": SWEEP_OU_LBS,
                "entry_threshold": entries[sig],
                "exit_style": [style],
                "exit_param": exit_params,
                "stop_loss_bps": SWEEP_STOP_LOSS_BPS,
                "gate": SWEEP_GATES,
            })
    return grids


def sweep(use_db: bool = True, n_jobs: int | None = None) -> dict:
    """Exact-engine sweep: one full backtest (stops, costs, trade mechanics)
    per (entry signal x setup x exit style x stop x gate) combo, parallel
    across CPU cores, with live progress."""
    raw_data = load_data() if use_db else synthetic_data()
    data = model_frame(raw_data)
    source = "db" if use_db else "synthetic"

    grids = _sweep_grids()
    total = sum(len(ParamGrid(g)) for g in grids)
    print(
        f"sweep: {SIGNAL_NAME}  signals={SWEEP_ENTRY_SIGNALS}  "
        f"exit_styles={SWEEP_EXIT_STYLES}  gates={len(SWEEP_GATES)}  "
        f"sub-grids={len(grids)}  combos={total:,}  rows={len(data)}  "
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
            print(f"\nWARNING: {len(errors)} combos errored; first: {errors['error'][0]}")
        results = results.filter(pl.col("error").is_null())

    store = MetricStore()
    store.log(
        SIGNAL_NAME,
        results,
        meta={
            "engine": "exact",
            "source": source,
            "span": f"{data['ts'].min()}..{data['ts'].max()}",
        },
    )

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
    for label, by in [
        ("sharpe", ["sharpe"]),
        ("hit rate", ["hit_rate", "sharpe"]),
        ("total pnl", ["total_pnl_bps", "sharpe"]),
    ]:
        print(f"\ntop 10 by {label}:")
        board = results.sort(by, descending=[True] * len(by), nulls_last=True)
        utils.pdf(board.select([c for c in show if c in board.columns]).head(10))

    print("\ngate summary (across all combos sharing the gate; null = ungated):")
    utils.pdf(
        results.group_by("gate")
        .agg(
            pl.col("sharpe").median().alias("med_sharpe"),
            pl.col("sharpe").max().alias("best_sharpe"),
            pl.col("hit_rate").median().alias("med_hit_rate"),
            pl.col("n_trades").median().alias("med_n_trades"),
            pl.len().alias("n_combos"),
        )
        .sort("med_sharpe", descending=True, nulls_last=True)
    )

    print(f"\nlogged {len(results)} runs -> {store.path}")

    return {"data": data, "results": results, "store": store}


def exit_scan(use_db: bool = True, device: str = "auto") -> dict:
    """Vectorized exit scan: every (setup, entry, exit style, exit param) as
    an approximate trade backtest — which exits pay, how long they hold, at
    what hit rate. Bands run on device; the stateful styles (revert_frac,
    half_life_frac) are CPU."""
    raw_data = load_data() if use_db else synthetic_data()
    data = model_frame(raw_data)
    level = pipeline.trade_def.composite_series(data).to_numpy()

    scans = []

    if "residual" in EXIT_ENTRY_SIGNALS:
        resid, resid_combos, resid_conditions = signal_matrix(
            data[FEATURE],
            data[TARGET],
            EXIT_BETA_LBS,
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
                "entries": EXIT_RESID_ENTRIES_BPS,
                "exits": EXIT_RESID_BANDS_BPS,
                # point-in-time residual half-life, aligned column-for-column
                "half_life": resid_conditions["resid_half_life"],
            }
        )

    if "ou_z" in EXIT_ENTRY_SIGNALS:
        ou_z, ou_combos, ou_conditions = signal_matrix(
            data[FEATURE],
            data[TARGET],
            EXIT_BETA_LBS,
            EXIT_OU_LBS,
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
                "entries": EXIT_OU_Z_ENTRIES,
                "exits": EXIT_OU_Z_BANDS,
                "half_life": ou_conditions["resid_half_life"],
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
        f"exit scan: signals={EXIT_ENTRY_SIGNALS}  styles={EXIT_STYLES}  "
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
                    cost_bps=TRANSACTION_COST_BPS,
                    combos=scan["combos"],
                    progress=_task_progress(f"{scan['entry_signal']} half_life_frac"),
                ).with_columns(pl.lit("half_life_frac").alias("exit_style"))
            )
            _done(f"{scan['entry_signal']} half_life_frac", bt)
        block = pl.concat(styled, how="diagonal_relaxed").with_columns(
            pl.lit(scan["entry_signal"]).alias("entry_signal"),
            pl.lit(scan["units"]).alias("threshold_units"),
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

    store = MetricStore()
    store.log(
        SIGNAL_NAME,
        results,
        meta={
            "engine": "exit",
            "source": "db" if use_db else "synthetic",
            "span": f"{data['ts'].min()}..{data['ts'].max()}",
        },
    )

    valid = results.filter((pl.col("n_trades") > 0) & pl.col("sharpe").is_finite())

    exit_summary = (
        valid.group_by("entry_signal", "exit_style", "exit_threshold")
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
    show = [
        "entry_signal",
        "beta_lb",
        "ou_lb",
        "entry_threshold",
        "exit_style",
        "exit_threshold",
        "threshold_units",
        "sharpe",
        "total_pnl_bps",
        "pnl_per_trade_bps",
        "hit_rate",
        "avg_hold_bars",
        "n_trades",
    ]
    robust = valid.filter(pl.col("n_trades") >= EXIT_MIN_TRADES)
    for label, by in [
        ("sharpe", ["sharpe"]),
        ("hit rate", ["hit_rate", "sharpe"]),
        ("total pnl", ["total_pnl_bps", "sharpe"]),
    ]:
        print(
            f"\ntop 10 setups by {label} (n_trades >= {EXIT_MIN_TRADES}; "
            "approximate - verify via --sweep):"
        )
        board = robust.sort(by, descending=[True] * len(by), nulls_last=True)
        utils.pdf(board.select([c for c in show if c in board.columns]).head(10))

    # per setup (signal, lookbacks, entry), which exit takes the best sharpe
    best_exit = (
        valid.sort("sharpe", descending=True)
        .group_by("entry_signal", "beta_lb", "ou_lb", "entry_threshold", maintain_order=True)
        .first()
    )
    print("\nwhich exit wins (count of setups where the exit has the best sharpe):")
    utils.pdf(
        best_exit.group_by("entry_signal", "exit_style", "exit_threshold")
        .agg(
            pl.len().alias("n_setups_won"),
            pl.col("sharpe").median().alias("med_winning_sharpe"),
            pl.col("avg_hold_bars").median().alias("med_hold_bars"),
        )
        .sort(["entry_signal", "n_setups_won"], descending=[False, True])
    )
    print(f"\nlogged {len(results):,} exit rows -> {store.path}")

    return {
        "data": data,
        "results": results,
        "exit_summary": exit_summary,
        "store": store,
    }


def predict(use_db: bool = True, device: str = "auto") -> dict:
    """GPU-friendly forward-horizon predictability scan with gate buckets."""
    raw_data = load_data() if use_db else synthetic_data()
    data = model_frame(raw_data)
    level = pipeline.trade_def.composite_series(data).to_numpy()

    scans = []

    if "residual" in PREDICT_ENTRY_SIGNALS:
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

    if "ou_z" in PREDICT_ENTRY_SIGNALS:
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
        f"predict scan: signals={PREDICT_ENTRY_SIGNALS}  "
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
            result_blocks.append(add_predict_lift(block))

        results = pl.concat(result_blocks, how="diagonal_relaxed").sort(
            "ic", descending=True, nulls_last=True
        )

    store = MetricStore()
    store.log(
        SIGNAL_NAME,
        results,
        meta={
            "engine": "predict",
            "source": "db" if use_db else "synthetic",
            "span": f"{data['ts'].min()}..{data['ts'].max()}",
        },
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
    ungated = valid.filter(pl.col("gate") == "(none)")
    print(f"\ntop 10 ungated predictability rows by IC (n_obs >= {PREDICT_MIN_OBS}):")
    utils.pdf(ungated.select([c for c in show if c in ungated.columns]).head(10))

    gated = valid.filter(
        (pl.col("gate") != "(none)") & pl.col("ic_lift").is_finite()
    ).sort("ic_lift", descending=True, nulls_last=True)
    print(f"\ntop 10 gates by IC LIFT (n_obs >= {PREDICT_MIN_OBS}):")
    utils.pdf(
        gated.select([*show, "base_ic", "ic_lift", "base_hit_rate", "hit_lift"]).head(
            10
        )
    )

    gated_by_ic = valid.filter(pl.col("gate") != "(none)").sort(
        "ic", descending=True, nulls_last=True
    )
    print(f"\ntop 10 gated predictability rows by IC (n_obs >= {PREDICT_MIN_OBS}):")
    utils.pdf(gated_by_ic.select([*show, "base_ic", "ic_lift"]).head(10))

    print("\nwhich gates help IC most often:")
    utils.pdf(
        valid.filter((pl.col("gate") != "(none)") & pl.col("ic_lift").is_finite())
        .group_by("entry_signal", "horizon", "gate", "gate_bucket")
        .agg(
            pl.col("ic_lift").median().alias("med_ic_lift"),
            (pl.col("ic_lift") > 0).mean().alias("pct_combos_improved"),
            pl.col("ic").median().alias("med_ic"),
            pl.col("fire_rate").median().alias("med_fire_rate"),
            pl.len().alias("n_cells"),
        )
        .sort("med_ic_lift", descending=True)
        .head(25)
    )
    print(f"\nlogged {len(results):,} predict rows -> {store.path}")

    return {"data": data, "results": results, "store": store}


if __name__ == "__main__":
    args = set(sys.argv[1:])
    known = {"--synthetic", "--cpu", "--gpu", "--sweep", "--predict",
             "--exit", "--exits", "--fast"}
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
