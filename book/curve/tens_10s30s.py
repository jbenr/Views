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
    python -m book.curve.tens_10s30s --sweep      # exact-engine grid
    python -m book.curve.tens_10s30s --fast       # auto GPU if available, else CPU
    python -m book.curve.tens_10s30s --fast --cpu # force CPU
    python -m book.curve.tens_10s30s --fast --gpu # request GPU, fallback to CPU
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

from utils.market_data import align_columns, coverage_report, load_wide
from backtest import (
    BacktestConfig,
    Engine,
    MetricStore,
    SignalConfig,
    SignalPipeline,
    TradeDef,
    add_gate_lift,
    add_predict_lift,
    fast_scan,
    gate_allow_mask,
    gate_scan,
    gate_variant_count,
    half_drift_residual,
    parse_gate,
    predict_scan,
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
        # gate_allow arrives as 1.0/0.0 (engine floats extras); absent = no gate
        gate = bar.get("gate_allow")
        if gate is not None and gate != 1.0:
            return False

        ou_z = bar.get("ou_z")
        half_life = bar.get("half_life")
        if not (_finite(ou_z) and _finite(half_life)):
            return False
        if not (half_life_min <= float(half_life) <= half_life_max):
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
    """Signal frame: raw beta-weighted target-vs-feature residual in bps."""
    p = _params(params)

    y = data[TARGET]
    x = data[FEATURE]
    reg = roll_lr_diff(x, y, lookback=p["beta_lb"])

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

    frame = pl.DataFrame({
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
    if p.get("gate") is not None:
        allow = gate_allow_mask(_gate_condition(frame, p), p["gate"])
        frame = frame.with_columns(pl.Series("gate_allow", allow))
    return frame


def make_pipeline(params: dict | None = None) -> SignalPipeline:
    """Lab contract: build the pipeline for any param combo (sweeps call this)."""
    p = _params(params)
    return SignalPipeline(
        name=SIGNAL_NAME,
        trade_def=TradeDef.outright(SIGNAL_NAME, TARGET),
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
    "entry_resid_bps": 20.0,     # raw residual threshold; positive -> short curve
    "z_gate": 0.5,               # OU z must confirm the residual direction
    "half_life_min": 3.0,        # block unstable / too-fast OU fits
    "half_life_max": 120.0,      # block slow drifts masquerading as mean reversion
    "exit_reversion_frac": 0.5,  # exit after this fraction of entry-to-OU-mean reverts
    "time_stop_mult": 2.0,       # dynamic time stop = half_life * multiplier
    "time_stop_min": 5.0,
    "time_stop_max": 120.0,
    "stop_loss_bps": 25.0,
    "gate": None,                # (condition, bucket) tuple or dict — see lab.parse_gate
}
TRANSACTION_COST_BPS = 0.1

SWEEP_GRID = {
    "beta_lb": [60,120], # list(range(10, 501, 10)),
    "ou_lb": [400,420], # list(range(10, 501, 10)),
    "entry_resid_bps": list(range(21, 31, 2)),
    "z_gate": np.arange(0.5, 3.1, 0.5).tolist(),
    "exit_reversion_frac": [0.25, 0.5, 0.75],
    "time_stop_mult": [1.0, 2.0, 3.0],
    "stop_loss_bps": [15.0, 25.0, 40.0],
    "gate": [
        None,
        ("r2", "low_25"),
        ("r2_mom_10", "low_10"),
        ("beta_cv", "below_50"),
        ("beta_vol20", "low_10")
    ],
}

FAST_BETA_LBS = [20,60,120,200,400] # list(range(10, 501, 10))
FAST_OU_LBS = [20,60,120,200,400,420] # list(range(10, 501, 10))
FAST_ENTRY_SIGNALS = ["residual", "ou_z"]
FAST_RESID_ENTRIES_BPS = list(range(25, 31, 1))
FAST_RESID_EXIT_BANDS_BPS = [5.0, 10.0]
FAST_OU_Z_ENTRIES = np.arange(0.5, 3.1, 0.5).tolist()
FAST_OU_Z_EXIT_BANDS = [0.25, 0.5]
FAST_GATE_BUCKETS = "regime"
FAST_MIN_TRADES = 30

PREDICT_ENTRY_SIGNALS = ["residual", "ou_z"]
PREDICT_HORIZONS = [5, 10, 20, 40, 60]
PREDICT_RESID_THRESHOLDS_BPS = list(range(5, 31, 2))
PREDICT_OU_Z_THRESHOLDS = np.arange(0.5, 3.1, 0.5).tolist()
PREDICT_GATE_BUCKETS = "regime"
PREDICT_MIN_OBS = 30

pipeline = make_pipeline()


# -- modes ------------------------------------------------------------------

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
    resid = last["signal"]
    ou_z = last["ou_z"]
    half_life = last["half_life"]
    gate_ok = bool(last.get("gate_allow", True))
    if resid is None or not _finite(resid):
        print("\nlatest signal: warmup - no signal yet")
    else:
        if (
            gate_ok
            and resid >= p["entry_resid_bps"]
            and _finite(ou_z)
            and ou_z >= p["z_gate"]
            and _finite(half_life)
            and p["half_life_min"] <= half_life <= p["half_life_max"]
        ):
            action = f"SHORT {TARGET} (curve steep/rich vs 10Y)"
        elif (
            gate_ok
            and resid <= -p["entry_resid_bps"]
            and _finite(ou_z)
            and ou_z <= -p["z_gate"]
            and _finite(half_life)
            and p["half_life_min"] <= half_life <= p["half_life_max"]
        ):
            action = f"LONG {TARGET} (curve flat/cheap vs 10Y)"
        else:
            action = "FLAT"
        print(
            f"\nlatest signal: ts={data['ts'][-1]}  resid={resid:+.1f}bps  "
            f"ou_z={ou_z:+.2f}  half_life={half_life:.1f}d  "
            f"beta={last['beta']:+.3f}  r2={last['r2']:.2f}  action={action}  "
            f"(abs(resid)>={p['entry_resid_bps']}bps, abs(ou_z)>={p['z_gate']})"
        )

    return {
        "raw_data": raw_data,
        "coverage": coverage,
        "data": data,
        "signals": sig_frame,
        "diag": diag,
        "result": result,
    }


def sweep(use_db: bool = True, n_jobs: int | None = None) -> dict:
    """Exact-engine grid search over SWEEP_GRID, parallel across CPU cores."""
    raw_data = load_data() if use_db else synthetic_data()
    data = model_frame(raw_data)
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
        "beta_lb", "ou_lb", "entry_resid_bps", "z_gate", "gate", "sharpe",
        "total_pnl_bps", "hit_rate", "n_trades", "max_drawdown_bps",
    ]
    print("\nleaderboard (top 10 by sharpe):")
    utils.pdf(results.select([c for c in show if c in results.columns]).head(10))

    print("\nsharpe matrix - beta_lb (cols) x ou_lb (rows), best entry/gate per cell:")
    matrix = (
        results.pivot(
            index="ou_lb",
            on="beta_lb",
            values="sharpe",
            aggregate_function="max",
        )
        .sort("ou_lb")
    )
    beta_cols = [str(x) for x in sorted(SWEEP_GRID["beta_lb"]) if str(x) in matrix.columns]
    utils.pdf(matrix.select(["ou_lb", *beta_cols]))
    print(f"\nlogged {len(results)} runs -> {store.path}")

    return {"data": data, "results": results, "store": store}


def fast(use_db: bool = True, device: str = "auto") -> dict:
    """Vectorized coarse scan of residual/OU-z entries and quantile gates."""
    raw_data = load_data() if use_db else synthetic_data()
    data = model_frame(raw_data)
    level = pipeline.trade_def.composite_series(data).to_numpy()

    scans = []

    if "residual" in FAST_ENTRY_SIGNALS:
        resid, resid_combos, resid_conditions = signal_matrix(
            data[FEATURE], data[TARGET], FAST_BETA_LBS, [0],
            return_conditions=True,
            signal_kind="residual",
            lookback_name="ou_lb",
        )
        scans.append({
            "entry_signal": "residual",
            "units": "bps",
            "matrix": resid,
            "combos": resid_combos,
            "conditions": resid_conditions,
            "entries": FAST_RESID_ENTRIES_BPS,
            "exits": FAST_RESID_EXIT_BANDS_BPS,
        })

    if "ou_z" in FAST_ENTRY_SIGNALS:
        ou_z, ou_combos, ou_conditions = signal_matrix(
            data[FEATURE], data[TARGET], FAST_BETA_LBS, FAST_OU_LBS,
            return_conditions=True,
            signal_kind="ou_zscore",
            lookback_name="ou_lb",
        )
        scans.append({
            "entry_signal": "ou_z",
            "units": "z",
            "matrix": ou_z,
            "combos": ou_combos,
            "conditions": ou_conditions,
            "entries": FAST_OU_Z_ENTRIES,
            "exits": FAST_OU_Z_EXIT_BANDS,
        })

    n_variants = 1 + len(scans[0]["conditions"]) * gate_variant_count(FAST_GATE_BUCKETS) if scans else 1
    n_evals = sum(
        scan["matrix"].shape[1]
        * len(scan["entries"])
        * len(scan["exits"])
        * n_variants
        for scan in scans
    )
    print(
        f"fast scan: signals={FAST_ENTRY_SIGNALS}  "
        f"model_columns={sum(scan['matrix'].shape[1] for scan in scans)}  "
        f"gate variants={n_variants}  evaluations={n_evals:,}  "
        f"(device={device})"
    )

    with utils.timed("running vectorized gated scan"):
        result_blocks = []
        for scan in scans:
            block = fast_scan(
                scan["matrix"], level,
                entries=scan["entries"],
                exit_band=scan["exits"],
                cost_bps=TRANSACTION_COST_BPS,
                combos=scan["combos"],
                gates=scan["conditions"],
                gate_buckets=FAST_GATE_BUCKETS,
                device=device,
                entry_col="entry_threshold",
                exit_col="exit_threshold",
            ).with_columns(
                pl.lit(scan["entry_signal"]).alias("entry_signal"),
                pl.lit(scan["units"]).alias("threshold_units"),
            )
            if scan["entry_signal"] == "residual":
                block = block.with_columns(pl.lit(None, dtype=pl.Int64).alias("ou_lb"))
            result_blocks.append(add_gate_lift(block))

        results = pl.concat(result_blocks, how="diagonal_relaxed").sort(
            "sharpe", descending=True, nulls_last=True
        )

    store = MetricStore()
    store.log(SIGNAL_NAME, results, meta={
        "engine": "fast", "source": "db" if use_db else "synthetic",
        "span": f"{data['ts'].min()}..{data['ts'].max()}",
    })

    show = [
        "entry_signal", "beta_lb", "ou_lb", "entry_threshold", "exit_threshold",
        "threshold_units", "gate", "gate_bucket",
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
    utils.pdf(strong.select([*show, "base_sharpe", "sharpe_lift"]).head(10))

    print("\nwhich gate helps most often (median lift across all combos):")
    utils.pdf(
        valid.filter(
            (pl.col("gate") != "(none)") & pl.col("sharpe_lift").is_finite()
        )
        .group_by("entry_signal", "gate", "gate_bucket")
        .agg(
            pl.col("sharpe_lift").median().alias("med_sharpe_lift"),
            (pl.col("sharpe_lift") > 0).mean().alias("pct_combos_improved"),
            pl.len().alias("n_cells"),
        )
        .sort("med_sharpe_lift", descending=True)
    )
    print(f"\nlogged {len(results):,} runs -> {store.path}")

    return {"data": data, "results": results, "store": store}


def predict(use_db: bool = True, device: str = "auto") -> dict:
    """GPU-friendly forward-horizon predictability scan with gate buckets."""
    raw_data = load_data() if use_db else synthetic_data()
    data = model_frame(raw_data)
    level = pipeline.trade_def.composite_series(data).to_numpy()

    scans = []

    if "residual" in PREDICT_ENTRY_SIGNALS:
        resid, resid_combos, resid_conditions = signal_matrix(
            data[FEATURE], data[TARGET], FAST_BETA_LBS, [0],
            return_conditions=True,
            signal_kind="residual",
            lookback_name="ou_lb",
        )
        scans.append({
            "entry_signal": "residual",
            "units": "bps",
            "matrix": resid,
            "combos": resid_combos,
            "conditions": resid_conditions,
            "thresholds": PREDICT_RESID_THRESHOLDS_BPS,
        })

    if "ou_z" in PREDICT_ENTRY_SIGNALS:
        ou_z, ou_combos, ou_conditions = signal_matrix(
            data[FEATURE], data[TARGET], FAST_BETA_LBS, FAST_OU_LBS,
            return_conditions=True,
            signal_kind="ou_zscore",
            lookback_name="ou_lb",
        )
        scans.append({
            "entry_signal": "ou_z",
            "units": "z",
            "matrix": ou_z,
            "combos": ou_combos,
            "conditions": ou_conditions,
            "thresholds": PREDICT_OU_Z_THRESHOLDS,
        })

    n_variants = 1 + len(scans[0]["conditions"]) * gate_variant_count(PREDICT_GATE_BUCKETS) if scans else 1
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
                scan["matrix"], level,
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
    store.log(SIGNAL_NAME, results, meta={
        "engine": "predict", "source": "db" if use_db else "synthetic",
        "span": f"{data['ts'].min()}..{data['ts'].max()}",
    })

    show = [
        "entry_signal", "beta_lb", "ou_lb", "entry_threshold", "threshold_units",
        "horizon", "gate", "gate_bucket", "ic", "hit_rate", "fire_rate", "n_obs",
    ]
    valid = results.filter(
        (pl.col("n_obs") >= PREDICT_MIN_OBS) & pl.col("ic").is_finite()
    )
    ungated = valid.filter(pl.col("gate") == "(none)")
    print(f"\ntop 10 ungated predictability rows by IC (n_obs >= {PREDICT_MIN_OBS}):")
    utils.pdf(ungated.select([c for c in show if c in ungated.columns]).head(10))

    gated = (
        valid.filter(
            (pl.col("gate") != "(none)")
            & pl.col("ic_lift").is_finite()
        )
        .sort("ic_lift", descending=True, nulls_last=True)
    )
    print(f"\ntop 10 gates by IC LIFT (n_obs >= {PREDICT_MIN_OBS}):")
    utils.pdf(
        gated.select([*show, "base_ic", "ic_lift", "base_hit_rate", "hit_lift"])
        .head(10)
    )

    strong = (
        gated.filter(pl.col("base_ic") > 0)
        .sort("ic", descending=True, nulls_last=True)
    )
    print("\ntop 10 gated predictability rows (base IC already positive):")
    utils.pdf(
        strong.select([*show, "base_ic", "ic_lift"])
        .head(10)
    )

    print("\nwhich gates help IC most often:")
    utils.pdf(
        valid.filter(
            (pl.col("gate") != "(none)") & pl.col("ic_lift").is_finite()
        )
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


def gates(use_db: bool = True, params: dict | None = None) -> dict:
    """Conditional edge scan for entry-time state variables."""
    p = _params(params)
    raw_data = load_data() if use_db else synthetic_data()
    data = model_frame(raw_data)
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
        "beta_vol20": sig_frame["beta"].diff().rolling_std(20),
        "beta_mom10": sig_frame["beta"].diff(10),
        "r2_vol20": sig_frame["r2"].diff().rolling_std(20),
        "r2_mom10": sig_frame["r2"].diff(10),
        "resid_vol20": sig_frame["resid"].diff().rolling_std(20),
        "beta": sig_frame["beta"],
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
    device = "cpu" if "--cpu" in sys.argv else ("gpu" if "--gpu" in sys.argv else "auto")
    if "--sweep" in sys.argv:
        state = sweep(use_db=use_db)
    elif "--predict" in sys.argv:
        state = predict(use_db=use_db, device=device)
    elif "--fast" in sys.argv:
        state = fast(use_db=use_db, device=device)
    elif "--gates" in sys.argv:
        state = gates(use_db=use_db)
    else:
        state = main(use_db=use_db)
