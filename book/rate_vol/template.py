"""Strategy template — the end-to-end pattern every new strategy follows.

This is a worked, runnable example of the repo's strategy lifecycle:

    tickers -> load data -> build features -> signal frame
            -> SignalPipeline -> backtest -> sweep/gates -> (ledger / execution)

It runs against the live DB:

    python -m book.rate_vol.template            # single backtest
    python -m book.rate_vol.template --sweep    # parameter grid through the lab

Layout rules the template demonstrates:
  * identity config (family, name, tickers) at the top of the module
  * BACKTEST PARAMETERS in the main section (DEFAULT_PARAMS) — override per
    run via main(params={...}); sweeps search around them
  * make_pipeline(params) — the lab contract used by backtest.lab sweeps
  * main() prints the standard four blocks and returns a state dict

The "alpha" here is deliberately generic (fade the OU z-score of a rolling
hedge-ratio residual between a target and an anchor). Copy this file into
your strategy family, swap in real tickers and a researched model, and keep
the structure. The real (stub) rate-vol strategy lives in strategy.py.
"""

from __future__ import annotations

import sys
from functools import partial

import polars as pl

from backtest import (
    BacktestConfig,
    Engine,
    MetricStore,
    SignalConfig,
    SignalPipeline,
    TradeDef,
    print_summary,
    sweep_strategy,
)
from stats import horizon_backtest, roll_lr_diff, roll_ou_zscore

# ── config (identity — what this strategy is, not how it's tuned) ─────────────

STRATEGY_FAMILY = "rates_vol"
SIGNAL_NAME = "template_target_vs_anchor"
MODULE = "book.rate_vol.template"     # importable path, used by sweep workers

START = "2015-01-01"

# 1. TICKERS — book alias -> Bloomberg ticker in md.index_eod.
#    Aliases are what your compute() and TradeDef reference.
TICKERS = {
    "target": "MOVE Index",   # what we model
    "anchor": "VIX Index",    # what we model it against
}


# ── helpers ───────────────────────────────────────────────────────────────────

def load_data(start: str = START) -> pl.DataFrame:
    """2a. Load research data from the DB (requires network access to raptor)."""
    from utils.market_data import load_wide

    return load_wide(TICKERS, start=start)


# 3. Features + 4. signal frame. compute() takes the wide data frame (and the
#    params it's tuned with) and returns a frame with a "signal" column; extra
#    columns (resid, beta, r2) pass through the engine into the trade log.
def compute(data: pl.DataFrame, params: dict | None = None) -> pl.DataFrame:
    p = {**DEFAULT_PARAMS, **(params or {})}
    reg = roll_lr_diff(data["anchor"], data["target"], lookback=p["beta_lb"])

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


# 5. make_pipeline — the lab contract. Sweeps import this module and call
#    make_pipeline(params) for every combo; the TradeDef legs reference the
#    TICKERS aliases.
def make_pipeline(params: dict | None = None) -> SignalPipeline:
    p = {**DEFAULT_PARAMS, **(params or {})}
    return SignalPipeline(
        name=SIGNAL_NAME,
        trade_def=TradeDef.spread(SIGNAL_NAME, "anchor", "target"),
        compute_fn=compute if params is None else partial(compute, params=p),
        config=SignalConfig(
            entry_long=-p["entry_z"],
            entry_short=p["entry_z"],
            exit_long=p["exit_z"],
            exit_short=-p["exit_z"],
            stop_loss_bps=None,
            time_stop_bars=p["time_stop_bars"],
        ),
    )


# ── main ──────────────────────────────────────────────────────────────────────
# Backtest parameters live HERE — the first thing you touch when scripting
# runs. sweep() searches around them; main(params={...}) overrides ad hoc.

DEFAULT_PARAMS = {
    "beta_lb": 63,           # hedge-ratio lookback (changes-based OLS)
    "z_lb": 63,              # z-score lookback
    "entry_z": 2.0,          # fade |z| beyond this
    "exit_z": 0.0,           # z-cross exit level
    "time_stop_bars": 40,    # ~2× residual half-life is the usual choice
}
TRANSACTION_COST_BPS = 0.5

SWEEP_GRID = {
    "beta_lb": [42, 63, 126],
    "z_lb": [42, 63, 126],
    "entry_z": [1.5, 2.0, 2.5],
}

pipeline = make_pipeline()   # default-parameter pipeline (engine/tests import this)


def main(params: dict | None = None) -> dict:
    p = {**DEFAULT_PARAMS, **(params or {})}

    data = load_data()
    print(f"rows={len(data)}  cols={data.columns}  params={p}")

    # residual diagnostics before backtesting: does fading it even work?
    sig_frame = compute(data, params=p)
    diag = horizon_backtest(sig_frame["resid"])
    print("\nresidual horizon backtest (IC / hit / Sharpe):")
    print(diag)

    # 6. backtest through the shared engine
    engine = Engine(BacktestConfig(transaction_cost_bps=TRANSACTION_COST_BPS))
    result = engine.add_signal(make_pipeline(params)).run(data)
    print_summary(result)

    # 7. ledger / execution integration (the intended live path):
    #    - log the latest signal row as a timestamped JSON record BEFORE the
    #      outcome is known (see notes/PLAN.md §7.2 for the schema)
    #    - map the signal to a TargetPosition and preview the order, e.g.:
    #        from execution import IBKRExecutor, TargetPosition
    #        executor = IBKRExecutor()   # dry-run by default
    #        executor.preview([TargetPosition(strategy=SIGNAL_NAME,
    #                                         instrument="10y", direction=1)])
    last_z = sig_frame["signal"][-1]
    if last_z is not None:
        print(f"\nlatest signal: ts={data['ts'][-1]}  z={last_z:+.2f}")

    return {"data": data, "signals": sig_frame, "diag": diag, "result": result}


def sweep(n_jobs: int | None = None) -> dict:
    """8. Parameter search through the lab: exact engine per combo, parallel
    across CPU cores, logged to the MetricStore for cross-strategy comparison.
    """
    data = load_data()
    print(f"sweep: {SIGNAL_NAME}  grid={SWEEP_GRID}  rows={len(data)}")

    results = sweep_strategy(
        MODULE, data, SWEEP_GRID,
        transaction_cost_bps=TRANSACTION_COST_BPS, n_jobs=n_jobs,
    )
    store = MetricStore()
    store.log(SIGNAL_NAME, results, meta={"engine": "exact", "source": "db"})

    print("\nleaderboard (top 10 by sharpe):")
    show = ["beta_lb", "z_lb", "entry_z", "sharpe", "total_pnl_bps", "hit_rate", "n_trades"]
    print(results.select([c for c in show if c in results.columns]).head(10))
    return {"data": data, "results": results, "store": store}


if __name__ == "__main__":
    if "--sweep" in sys.argv:
        state = sweep()
    else:
        state = main()
