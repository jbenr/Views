"""10s vs 10s30s — direction/curve interaction, scriptable standard output.

The current research thread (notes/notes.md): does the 10s30s slope explain
the level of 10Y, and is the leftover tradable? Model: changes-based rolling
OLS of Δ10Y on Δ10s30s (same construction as dig/beta_scan_10s_factors.py),
roll the residual into level space, fade OU z-score extremes. Directions are
in YIELD space: z >= +entry_z means 10Y yield rich vs the curve -> short 10s.

Backtest parameters live in the main section (DEFAULT_PARAMS) — override per
run via main(params={...}) or search around them with the lab modes below.

    python -m book.curve.tens_10s30s              # single run, live DB
    python -m book.curve.tens_10s30s --synthetic  # single run, no DB
    python -m book.curve.tens_10s30s --sweep      # exact-engine grid, all cores
    python -m book.curve.tens_10s30s --fast       # vectorized coarse scan (add --gpu on the tower)
    python -m book.curve.tens_10s30s --gates      # conditional edge buckets

Every mode returns a dict of state for interactive chaining: `state = main()`.
"""

from __future__ import annotations

import datetime as dt
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
    fast_scan,
    gate_scan,
    print_summary,
    signal_matrix,
    sweep_strategy,
)
from stats import beta_cv, horizon_backtest, roll_lr_diff, roll_ou_zscore
import utils

# ── config (identity — what this strategy is, not how it's tuned) ─────────────

STRATEGY_FAMILY = "curve"
SIGNAL_NAME = "tens_10s30s"
MODULE = "book.curve.tens_10s30s"   # importable path, used by sweep workers

START = "2010-01-01"

TICKERS = {
    "10y":    "USGG10YR Index",   # % -> scaled to bps at load
    "10s30s": "USYC1030 Index",   # already quoted in bps
}
BPS_COLS = ["10y"]
YIELD_COLS = ["10y", "10s30s"]


# ── helpers ───────────────────────────────────────────────────────────────────

def load_data(start: str = START) -> pl.DataFrame:
    """Load 10Y and 10s30s from md.index_eod (requires access to raptor)."""
    from utils.market_data import load_wide

    return (
        load_wide(TICKERS, start=start, bps_cols=BPS_COLS)
        .drop_nulls()
        .with_columns(pl.col(YIELD_COLS).round(2))
    )


def synthetic_data(n: int = 1500, seed: int = 21) -> pl.DataFrame:
    """Synthetic substitute: a slope random walk plus a 10Y that is beta-linked
    to it with an OU residual — the structure the signal fades."""
    rng = np.random.default_rng(seed)

    slope = 50.0 + np.cumsum(rng.normal(0.0, 1.5, n))     # 10s30s walk (bps)

    resid = np.zeros(n)                                   # OU: half-life ≈ 14d
    theta, sigma = 0.05, 2.0
    for i in range(1, n):
        resid[i] = resid[i - 1] * (1 - theta) + rng.normal(0.0, sigma)

    tens = 350.0 + 0.6 * slope + resid                    # 10Y yield (bps)

    start_date = dt.date.fromisoformat(START)
    ts = pl.date_range(
        start_date, start_date + dt.timedelta(days=2 * n), interval="1d", eager=True
    )
    ts = ts.filter(ts.dt.weekday() <= 5)[:n]

    return (
        pl.DataFrame({"ts": ts, "10y": tens, "10s30s": slope})
        .with_columns(pl.col(YIELD_COLS).round(2))
    )


def compute(data: pl.DataFrame, params: dict | None = None) -> pl.DataFrame:
    """Signal frame: OU z of the beta-weighted 10Y-vs-curve residual.

    Extra columns (resid, beta, r2) flow through the engine into the trade log.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    reg = roll_lr_diff(data["10s30s"], data["10y"], lookback=p["beta_lb"])

    # roll_lr_diff drops one row (first diff) — pad back to len(data)
    null1 = pl.Series([None], dtype=pl.Float64)
    resid_roll = pl.concat([
        null1,
        reg["resid"].rolling_sum(p["beta_lb"], min_samples=p["beta_lb"]),
    ])
    beta = pl.concat([null1, reg["beta"]])
    r2 = pl.concat([null1, reg["r2"]])

    z = roll_ou_zscore(resid_roll, lookback=p["z_lb"])
    return pl.DataFrame({"signal": z, "resid": resid_roll, "beta": beta, "r2": r2})


def make_pipeline(params: dict | None = None) -> SignalPipeline:
    """Lab contract: build the pipeline for any param combo (sweeps call this)."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    return SignalPipeline(
        name=SIGNAL_NAME,
        trade_def=TradeDef.spread(SIGNAL_NAME, "10s30s", "10y"),
        compute_fn=compute if params is None else partial(compute, params=p),
        config=SignalConfig(
            entry_long=-p["entry_z"],
            entry_short=p["entry_z"],
            exit_long=p["exit_z"],
            exit_short=-p["exit_z"],
            stop_loss_bps=p["stop_loss_bps"],
            time_stop_bars=p["time_stop_bars"],
        ),
    )


# ── main ──────────────────────────────────────────────────────────────────────
# Backtest parameters live HERE — the first thing you touch when scripting
# runs. sweep()/fast() search around them; main(params={...}) overrides ad hoc.

DEFAULT_PARAMS = {
    "beta_lb": 252,          # hedge-ratio lookback — mirrors beta_scan PLOT_LOOKBACK
    "z_lb": 252,             # z-score lookback
    "entry_z": 1.5,          # mirrors beta_scan ACTION_Z — re-derive before going live
    "exit_z": 0.0,           # z-cross exit level
    "stop_loss_bps": 25.0,
    "time_stop_bars": 40,    # ~2× residual half-life
}
TRANSACTION_COST_BPS = 0.1

SWEEP_GRID = {               # exact-engine sweep (--sweep)
    "beta_lb": [63, 126, 252],
    "z_lb": [63, 126, 252],
    "entry_z": [1.5, 2.0, 2.5],
}
FAST_BETA_LBS = list(range(21, 505, 21))
FAST_Z_LBS = list(range(21, 505, 21))
FAST_ENTRIES = [round(x / 10, 1) for x in range(10, 31)]

pipeline = make_pipeline()   # default-parameter pipeline (engine/tests import this)


def main(use_db: bool = True, params: dict | None = None) -> dict:
    p = {**DEFAULT_PARAMS, **(params or {})}

    # load data
    data = load_data() if use_db else synthetic_data()
    utils.pdf(data.tail(5))
    print(
        f"rows={len(data)}  {data['ts'].min()} → {data['ts'].max()}  "
        f"cols={data.columns}  (source={'db' if use_db else 'synthetic'})  params={p}"
    )

    # residual diagnostics before backtesting: does fading it even work?
    sig_frame = compute(data, params=p)
    diag = horizon_backtest(sig_frame["resid"])
    print("\nresidual horizon backtest (IC / hit / Sharpe):")
    print(diag)

    # backtest through the shared engine
    engine = Engine(BacktestConfig(transaction_cost_bps=TRANSACTION_COST_BPS))
    result = engine.add_signal(make_pipeline(params)).run(data)
    print_summary(result)

    # latest signal state — directions in yield space, matching beta_scan
    last = sig_frame.row(-1, named=True)
    z = last["signal"]
    if z is None:
        print("\nlatest signal: warmup — no signal yet")
    else:
        if z >= p["entry_z"]:
            action = "SHORT 10s (yield rich vs curve)"
        elif z <= -p["entry_z"]:
            action = "LONG 10s (yield cheap vs curve)"
        else:
            action = "FLAT"
        print(
            f"\nlatest signal: ts={data['ts'][-1]}  z={z:+.2f}  "
            f"resid={last['resid']:+.1f}bps  beta={last['beta']:+.3f}  "
            f"r2={last['r2']:.2f}  action={action} (|z|>{p['entry_z']})"
        )

    return {"data": data, "signals": sig_frame, "diag": diag, "result": result}


def sweep(use_db: bool = True, n_jobs: int | None = None) -> dict:
    """Exact-engine grid search over SWEEP_GRID, parallel across CPU cores.

    Logs every combo to the MetricStore and prints the leaderboard plus the
    beta_lb × z_lb sharpe matrix (best entry_z per cell).
    """
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

    show = ["beta_lb", "z_lb", "entry_z", "sharpe", "total_pnl_bps",
            "hit_rate", "n_trades", "max_drawdown_bps"]
    print("\nleaderboard (top 10 by sharpe):")
    print(results.select([c for c in show if c in results.columns]).head(10))

    print("\nsharpe matrix — beta_lb (cols) × z_lb (rows), best entry_z per cell:")
    print(store.matrix(x="beta_lb", y="z_lb", metric="sharpe",
                       strategy=SIGNAL_NAME, agg="max"))
    print(f"\nlogged {len(results)} runs → {store.path}")

    return {"data": data, "results": results, "store": store}


def fast(use_db: bool = True, device: str = "cpu") -> dict:
    """Vectorized coarse scan of the big grid — the GPU-ready funnel stage.

    Approximate mechanics (hysteresis exits, no stops) — shortlist here, then
    verify winners through sweep(). On the tower: --gpu (needs cupy-cuda12x).
    """
    data = load_data() if use_db else synthetic_data()
    z, combos = signal_matrix(data["10s30s"], data["10y"], FAST_BETA_LBS, FAST_Z_LBS)
    level = pipeline.trade_def.composite_series(data).to_numpy()
    n_evals = len(combos) * len(FAST_ENTRIES)
    print(f"fast scan: {len(combos)} signal columns × {len(FAST_ENTRIES)} entries "
          f"= {n_evals} evaluations  (device={device})")

    with utils.timed("running vectorized scan"):
        results = fast_scan(
            z, level, entries=FAST_ENTRIES, cost_bps=TRANSACTION_COST_BPS,
            combos=combos, device=device,
        )

    store = MetricStore()
    store.log(SIGNAL_NAME, results, meta={
        "engine": "fast", "source": "db" if use_db else "synthetic",
        "span": f"{data['ts'].min()}..{data['ts'].max()}",
    })

    print("\ntop 10 by sharpe (approximate — verify via --sweep):")
    print(results.head(10))
    return {"data": data, "results": results, "store": store}


def gates(use_db: bool = True, params: dict | None = None) -> dict:
    """Conditional edge scan: which state variables mark high-confidence entries?

    Buckets model-quality and regime variables at hypothetical entry bars and
    reports hit/PnL lift per bucket vs the unconditional baseline. Promising
    buckets graduate into SignalConfig.entry_filter_fn gates.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    data = load_data() if use_db else synthetic_data()
    sig_frame = compute(data, params=p)
    level = pipeline.trade_def.composite_series(data)

    conditions = pl.DataFrame({
        "r2": sig_frame["r2"],
        "beta_cv": beta_cv(sig_frame["beta"], lookback=p["beta_lb"]),
        "resid_vol20": sig_frame["resid"].diff().rolling_std(20),
        "abs_beta": sig_frame["beta"].abs(),
    })

    table = gate_scan(
        sig_frame["signal"], level, conditions,
        entry_z=p["entry_z"], horizon=p["time_stop_bars"] // 2,
    )
    print(f"gate scan: entry_z={p['entry_z']}  horizon={p['time_stop_bars'] // 2}d  "
          f"conditions={conditions.columns}")
    print(table)
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
