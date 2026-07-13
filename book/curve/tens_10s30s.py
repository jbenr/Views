"""10s vs 10s30s - direction/curve interaction, scriptable standard output.

The current research thread (notes/notes.md): does the 10s30s slope explain
the level of 10Y, and is the leftover tradable? Model: changes-based rolling
OLS of d10Y on d10s30s (same construction as dig/beta_scan_10s_factors.py),
roll the residual into level space, and fade raw residual extremes. OU state
is used as a gate and for exits/time stops rather than as the primary entry
threshold.

Directions are in YIELD space: positive residual means 10Y yield is rich vs
the curve -> short 10s; negative residual means 10Y yield is cheap -> long 10s.

Backtest parameters live in DEFAULT_PARAMS. Override with main(params={...})
or search around them with the lab modes below.

    python -m book.curve.tens_10s30s              # single run, live DB
    python -m book.curve.tens_10s30s --synthetic  # single run, no DB
    python -m book.curve.tens_10s30s --sweep      # exact-engine grid
    python -m book.curve.tens_10s30s --fast       # vectorized coarse scan
    python -m book.curve.tens_10s30s --fast --gpu # GPU fast scan
    python -m book.curve.tens_10s30s --gates      # conditional edge buckets

Every mode returns a dict of state for interactive chaining: state = main().
"""

from __future__ import annotations

import datetime as dt
import math
import sys
from functools import partial

import numpy as np
import polars as pl

from backtest import (
    BacktestConfig,
    Engine,
    MetricStore,
    SignalConfig,
    SignalPipeline,
    TradeDef,
    add_gate_lift,
    fast_scan,
    gate_scan,
    half_drift_residual,
    print_summary,
    signal_matrix,
    sweep_strategy,
)
from stats import beta_cv, horizon_backtest, roll_lr_diff, roll_ou_features
import utils


# -- config -----------------------------------------------------------------

STRATEGY_FAMILY = "curve"
SIGNAL_NAME = "tens_10s30s"
MODULE = "book.curve.tens_10s30s"   # importable path, used by sweep workers

START = "2010-01-01"

TICKERS = {
    "10y": "USGG10YR Index",      # % -> scaled to bps at load
    "10s30s": "USYC1030 Index",   # already quoted in bps
}
BPS_COLS = ["10y"]
YIELD_COLS = ["10y", "10s30s"]



# -- helpers ----------------------------------------------------------------

def load_data(start: str = START) -> pl.DataFrame:
    """Load 10Y and 10s30s from md.index_eod (requires access to raptor)."""
    from utils.market_data import load_wide

    return (
        load_wide(TICKERS, start=start, bps_cols=BPS_COLS)
        .drop_nulls()
        .with_columns(pl.col(YIELD_COLS).round(2))
    )


def synthetic_data(n: int = 1500, seed: int = 21) -> pl.DataFrame:
    """Synthetic substitute: slope random walk plus OU residual."""
    rng = np.random.default_rng(seed)

    slope = 50.0 + np.cumsum(rng.normal(0.0, 1.5, n))

    resid = np.zeros(n)  # OU: half-life around 14d
    theta, sigma = 0.05, 2.0
    for i in range(1, n):
        resid[i] = resid[i - 1] * (1 - theta) + rng.normal(0.0, sigma)

    tens = 350.0 + 0.6 * slope + resid

    start_date = dt.date.fromisoformat(START)
    ts = pl.date_range(
        start_date, start_date + dt.timedelta(days=2 * n), interval="1d", eager=True
    )
    ts = ts.filter(ts.dt.weekday() <= 5)[:n]

    return (
        pl.DataFrame({"ts": ts, "10y": tens, "10s30s": slope})
        .with_columns(pl.col(YIELD_COLS).round(2))
    )


def _params(params: dict | None = None) -> dict:
    """Merge params and accept old z-score names as aliases."""
    raw = params or {}
    p = {**DEFAULT_PARAMS, **raw}
    if "z_lb" in raw and "ou_lb" not in raw:
        p["ou_lb"] = raw["z_lb"]
    if "entry_z" in raw and "entry_resid_bps" not in raw:
        p["entry_resid_bps"] = raw["entry_z"]
    return p


def _finite(value) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _entry_filter(z_gate: float, half_life_min: float, half_life_max: float):
    """Require OU z to confirm the raw residual entry direction."""

    def fn(direction: int, bar: dict) -> bool:
        ou_z = bar.get("ou_z")
        half_life = bar.get("half_life")
        if not (_finite(ou_z) and _finite(half_life)):
            return False
        if not (half_life_min <= float(half_life) <= half_life_max):
            return False

        # direction +1 is long 10s: residual should be cheap/negative.
        # direction -1 is short 10s: residual should be rich/positive.
        if direction == 1:
            return float(ou_z) <= -z_gate
        if direction == -1:
            return float(ou_z) >= z_gate
        return False

    return fn


def compute(data: pl.DataFrame, params: dict | None = None) -> pl.DataFrame:
    """Signal frame: raw beta-weighted 10Y-vs-curve residual in bps."""
    p = _params(params)
    reg = roll_lr_diff(data["10s30s"], data["10y"], lookback=p["beta_lb"])

    # roll_lr_diff drops one row (first diff) - pad back to len(data)
    null1 = pl.Series([None], dtype=pl.Float64)
    resid_roll = pl.concat([
        null1,
        reg["resid"].rolling_sum(p["beta_lb"], min_samples=p["beta_lb"]),
    ])
    beta = pl.concat([null1, reg["beta"]])
    r2 = pl.concat([null1, reg["r2"]])

    ou = roll_ou_features(resid_roll, lookback=p["ou_lb"])
    time_stop = (
        (ou["half_life"] * p["time_stop_mult"])
        .clip(p["time_stop_min"], p["time_stop_max"])
        .alias("time_stop")
    )

    return pl.DataFrame({
        "signal": resid_roll,
        "resid": resid_roll,
        "ou_z": ou["ou_z"],
        "ou_mean": ou["ou_mean"],
        "ou_sigma": ou["ou_sigma"],
        "ou_rho": ou["ou_rho"],
        "ou_theta": ou["ou_theta"],
        "expected_delta_1d": ou["expected_delta_1d"],
        "half_life": ou["half_life"],
        "time_stop": time_stop,
        "beta": beta,
        "r2": r2,
    })


def make_pipeline(params: dict | None = None) -> SignalPipeline:
    """Lab contract: build the pipeline for any param combo (sweeps call this)."""
    p = _params(params)
    return SignalPipeline(
        name=SIGNAL_NAME,
        trade_def=TradeDef.spread(SIGNAL_NAME, "10s30s", "10y"),
        compute_fn=compute if params is None else partial(compute, params=p),
        config=SignalConfig(
            entry_long=-p["entry_resid_bps"],
            entry_short=p["entry_resid_bps"],
            exit_long=None,
            exit_short=None,
            stop_loss_bps=p["stop_loss_bps"],
            time_stop_bars=None,
            exit_fn=half_drift_residual(p["exit_reversion_frac"]),
            entry_filter_fn=_entry_filter(
                p["z_gate"], p["half_life_min"], p["half_life_max"]
            ),
        ),
    )



# -- strategy parameters ----------------------------------------------------

# Backtest parameters live HERE - the first thing you touch when scripting.
DEFAULT_PARAMS = {
    "beta_lb": 252,              # hedge-ratio lookback
    "ou_lb": 252,                # OU-state lookback for z/mean/half-life
    "entry_resid_bps": 20.0,     # raw residual threshold; positive -> short 10s
    "z_gate": 0.5,               # OU z must confirm the residual direction
    "half_life_min": 3.0,        # block unstable / too-fast OU fits
    "half_life_max": 120.0,      # block slow drifts masquerading as mean reversion
    "exit_reversion_frac": 0.5,  # exit after this fraction of entry-to-OU-mean reverts
    "time_stop_mult": 2.0,       # dynamic time stop = half_life * multiplier
    "time_stop_min": 5.0,
    "time_stop_max": 120.0,
    "stop_loss_bps": 25.0,
}
TRANSACTION_COST_BPS = 0.1

SWEEP_GRID = {
    "beta_lb": [63, 126, 252],
    "ou_lb": [63, 126, 252],
    "entry_resid_bps": [15.0, 25.0, 40.0],
    "z_gate": [0.5, 1.0, 1.5],
}

FAST_BETA_LBS = list(range(21, 505, 21))
FAST_OU_LBS = list(range(21, 505, 21))
FAST_ENTRIES_BPS = [float(x) for x in range(5, 51, 2.5)]
FAST_EXIT_BAND_BPS = 5.0
FAST_GATE_BUCKETS = 5
FAST_MIN_TRADES = 30

pipeline = make_pipeline()


# -- modes ------------------------------------------------------------------

def main(use_db: bool = True, params: dict | None = None) -> dict:
    p = _params(params)

    data = load_data() if use_db else synthetic_data()
    utils.pdf(data.tail(5))
    print(
        f"rows={len(data)}  {data['ts'].min()} -> {data['ts'].max()}  "
        f"cols={data.columns}  (source={'db' if use_db else 'synthetic'})  params={p}"
    )

    sig_frame = compute(data, params=p)
    diag = horizon_backtest(sig_frame["resid"])
    print("\nresidual horizon backtest (IC / hit / Sharpe):")
    print(diag)

    engine = Engine(BacktestConfig(transaction_cost_bps=TRANSACTION_COST_BPS))
    result = engine.add_signal(make_pipeline(p)).run(data)
    print_summary(result)

    last = sig_frame.row(-1, named=True)
    resid = last["signal"]
    ou_z = last["ou_z"]
    half_life = last["half_life"]
    if resid is None or not _finite(resid):
        print("\nlatest signal: warmup - no signal yet")
    else:
        if (
            resid >= p["entry_resid_bps"]
            and _finite(ou_z)
            and ou_z >= p["z_gate"]
            and _finite(half_life)
            and p["half_life_min"] <= half_life <= p["half_life_max"]
        ):
            action = "SHORT 10s (yield rich vs curve)"
        elif (
            resid <= -p["entry_resid_bps"]
            and _finite(ou_z)
            and ou_z <= -p["z_gate"]
            and _finite(half_life)
            and p["half_life_min"] <= half_life <= p["half_life_max"]
        ):
            action = "LONG 10s (yield cheap vs curve)"
        else:
            action = "FLAT"
        print(
            f"\nlatest signal: ts={data['ts'][-1]}  resid={resid:+.1f}bps  "
            f"ou_z={ou_z:+.2f}  half_life={half_life:.1f}d  "
            f"beta={last['beta']:+.3f}  r2={last['r2']:.2f}  action={action}  "
            f"(abs(resid)>={p['entry_resid_bps']}bps, abs(ou_z)>={p['z_gate']})"
        )

    return {"data": data, "signals": sig_frame, "diag": diag, "result": result}


def sweep(use_db: bool = True, n_jobs: int | None = None) -> dict:
    """Exact-engine grid search over SWEEP_GRID, parallel across CPU cores."""
    data = load_data() if use_db else synthetic_data()
    source = "db" if use_db else "synthetic"
    print(f"sweep: {SIGNAL_NAME}  grid={SWEEP_GRID}  rows={len(data)}  (source={source})")

    with utils.timed("running exact-engine sweep"):
        results = sweep_strategy(
            MODULE, data, SWEEP_GRID,
            transaction_cost_bps=TRANSACTION_COST_BPS, n_jobs=n_jobs,
        )

    store = MetricStore()
    store.log(SIGNAL_NAME, results, meta={
        "engine": "exact", "source": source,
        "span": f"{data['ts'].min()}..{data['ts'].max()}",
    })

    show = [
        "beta_lb", "ou_lb", "entry_resid_bps", "z_gate", "sharpe",
        "total_pnl_bps", "hit_rate", "n_trades", "max_drawdown_bps",
    ]
    print("\nleaderboard (top 10 by sharpe):")
    utils.pdf(results.select([c for c in show if c in results.columns]).head(10))

    print("\nsharpe matrix - beta_lb (cols) x ou_lb (rows), best entry/gate per cell:")
    utils.pdf(store.matrix(
        x="beta_lb", y="ou_lb", metric="sharpe",
        strategy=SIGNAL_NAME, agg="max",
    ))
    print(f"\nlogged {len(results)} runs -> {store.path}")

    return {"data": data, "results": results, "store": store}


def fast(use_db: bool = True, device: str = "cpu") -> dict:
    """Vectorized coarse scan of raw-residual entries and quantile gates."""
    data = load_data() if use_db else synthetic_data()
    resid, combos, conditions = signal_matrix(
        data["10s30s"], data["10y"], FAST_BETA_LBS, FAST_OU_LBS,
        return_conditions=True,
        signal_kind="residual",
        lookback_name="ou_lb",
    )
    level = pipeline.trade_def.composite_series(data).to_numpy()
    n_variants = 1 + len(conditions) * FAST_GATE_BUCKETS
    n_evals = len(combos) * len(FAST_ENTRIES_BPS) * n_variants
    print(
        f"fast scan: {len(combos)} residual columns x {len(FAST_ENTRIES_BPS)} "
        f"raw entries x {n_variants} gate variants = {n_evals:,} evaluations  "
        f"(device={device})"
    )

    with utils.timed("running vectorized gated scan"):
        results = fast_scan(
            resid, level,
            entries=FAST_ENTRIES_BPS,
            exit_band=FAST_EXIT_BAND_BPS,
            cost_bps=TRANSACTION_COST_BPS,
            combos=combos,
            gates=conditions,
            gate_buckets=FAST_GATE_BUCKETS,
            device=device,
            entry_col="entry_resid_bps",
        )
        results = add_gate_lift(results)

    store = MetricStore()
    store.log(SIGNAL_NAME, results, meta={
        "engine": "fast", "source": "db" if use_db else "synthetic",
        "span": f"{data['ts'].min()}..{data['ts'].max()}",
    })

    show = [
        "beta_lb", "ou_lb", "entry_resid_bps", "gate", "gate_bucket",
        "sharpe", "total_pnl_bps", "hit_rate", "n_trades",
    ]
    valid = results.filter(
        (pl.col("n_trades") > 0) & pl.col("sharpe").is_finite()
    )
    ungated = valid.filter(pl.col("gate") == "(none)")
    print("\ntop 10 ungated by sharpe (approximate - verify via --sweep):")
    utils.pdf(ungated.select([c for c in show if c in ungated.columns]).head(10))

    gated = (
        valid.filter(
            (pl.col("gate") != "(none)")
            & (pl.col("n_trades") >= FAST_MIN_TRADES)
            & pl.col("sharpe_lift").is_finite()
        )
        .sort("sharpe_lift", descending=True, nulls_last=True)
    )
    print(f"\ntop 10 gates by sharpe LIFT vs same combo ungated (n_trades >= {FAST_MIN_TRADES}):")
    utils.pdf(gated.select([*show, "base_sharpe", "sharpe_lift", "hit_lift"]).head(10))

    strong = (
        gated.filter(pl.col("base_sharpe") > 0)
        .sort("sharpe", descending=True, nulls_last=True)
    )
    print("\ntop 10 gated setups by absolute sharpe (base already positive — "
          "gate improves a working combo):")
    print(strong.select([*show, "base_sharpe", "sharpe_lift"]).head(10))

    print("\nwhich gate helps most often (median lift across all combos):")
    utils.pdf(
        valid.filter(
            (pl.col("gate") != "(none)") & pl.col("sharpe_lift").is_finite()
        )
        .group_by("gate", "gate_bucket")
        .agg(
            pl.col("sharpe_lift").median().alias("med_sharpe_lift"),
            (pl.col("sharpe_lift") > 0).mean().alias("pct_combos_improved"),
            pl.len().alias("n_cells"),
        )
        .sort("med_sharpe_lift", descending=True)
    )
    print(f"\nlogged {len(results):,} runs -> {store.path}")

    return {"data": data, "results": results, "store": store}


def gates(use_db: bool = True, params: dict | None = None) -> dict:
    """Conditional edge scan for entry-time state variables."""
    p = _params(params)
    data = load_data() if use_db else synthetic_data()
    sig_frame = compute(data, params=p)
    level = pipeline.trade_def.composite_series(data)

    conditions = pl.DataFrame({
        "ou_z_abs": sig_frame["ou_z"].abs(),
        "ou_z_aligned": pl.Series(
            np.sign(sig_frame["resid"].to_numpy().astype(float))
            * sig_frame["ou_z"].to_numpy().astype(float),
            dtype=pl.Float64,
        ),
        "half_life": sig_frame["half_life"],
        "r2": sig_frame["r2"],
        "beta_cv": beta_cv(sig_frame["beta"], lookback=p["beta_lb"]),
        "resid_vol20": sig_frame["resid"].diff().rolling_std(20),
        "abs_beta": sig_frame["beta"].abs(),
    })

    horizon = int(round(np.clip(
        p["time_stop_mult"] * 14,
        p["time_stop_min"], p["time_stop_max"],
    )))
    table = gate_scan(
        sig_frame["signal"], level, conditions,
        entry_z=p["entry_resid_bps"], horizon=horizon,
    )
    print(
        f"gate scan: entry_resid_bps={p['entry_resid_bps']}  horizon={horizon}d  "
        f"conditions={conditions.columns}"
    )
    utils.pdf(table)
    return {"data": data, "signals": sig_frame, "gates": table}


if __name__ == "__main__":
    use_db = "--synthetic" not in sys.argv
    if "--sweep" in sys.argv:
        state = sweep(use_db=use_db)
    elif "--fast" in sys.argv:
        state = fast(use_db=use_db, device="gpu" if "--gpu" in sys.argv else "cpu")
    elif "--gates" in sys.argv:
        state = gates(use_db=use_db)
    else:
        state = main(use_db=use_db)
