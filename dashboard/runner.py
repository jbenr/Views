"""Pulls fresh data and runs the live analysis for a promoted signal.

Two distinct actions map to the dashboard's two buttons:

    pull_data(module)      fresh Strategy.load_data(), cached to disk.
                            No ledger write -- just refreshes the inputs.
    run_analysis(module)   Strategy.compute() on the cached data, checked
                            against the promoted entry threshold, ONE row
                            appended to the signal ledger.

compute_signal() is the shared, non-logging core both the dashboard's
chart rendering and run_analysis() use -- rendering a card must not itself
be an auditable event, only an explicit "re-run analysis" click is.
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import polars as pl

from .ledger import SignalLedger
from .params import params_from_row
from .registry import LiveRegistry, load_strategy


def _store_dir() -> Path:
    return Path(
        os.getenv("VIEWS_STORE_DIR", str(Path(__file__).resolve().parents[1] / "store"))
    )


def _data_cache_path(name: str) -> Path:
    return _store_dir() / "live_data" / f"{name}.parquet"


def _f(value):
    return float(value) if value is not None else None


def pull_data(module: str) -> dict:
    """Fresh Strategy.load_data(), cached to store/live_data/<name>.parquet."""
    strategy = load_strategy(module)
    data = strategy.model_frame(strategy.load_data())
    cache_path = _data_cache_path(strategy.name)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    data.write_parquet(cache_path)
    return {
        "data": data,
        "pulled_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "data_asof": data["ts"][-1],
    }


def cached_data(module: str) -> pl.DataFrame | None:
    strategy = load_strategy(module)
    cache_path = _data_cache_path(strategy.name)
    if not cache_path.exists():
        return None
    return pl.read_parquet(cache_path)


def compute_signal(module: str) -> dict:
    """Recompute the promoted signal on cached data. No ledger write --
    used to render charts on every card refresh, not just on demand."""
    strategy = load_strategy(module)
    data = cached_data(module)
    if data is None:
        raise RuntimeError(f"{module}: no cached data -- click 'Re-pull data' first")

    row = LiveRegistry().get(module)
    if row is None:
        raise RuntimeError(f"{module}: not promoted -- run dashboard.registry --promote first")
    params = params_from_row(row)

    sig = strategy.compute(data, params=params)
    last = sig.tail(1).to_dicts()[0]
    threshold = params["entry_threshold"]
    reading = last.get("signal")

    fired = "flat"
    if reading is not None:
        if reading <= -threshold:
            fired = "long"
        elif reading >= threshold:
            fired = "short"
    if fired != "flat" and last.get("gate_allow") is False:
        fired = "flat (gated)"

    return {
        "strategy": strategy,
        "data": data,
        "signal_frame": sig,
        "params": params,
        "last": last,
        "fired": fired,
        "data_asof": data["ts"][-1],
    }


def trade_history(module: str, state: dict | None = None) -> tuple[pl.DataFrame, dict | None]:
    """Re-run the exact promoted pipeline against cached data to recover the
    closed trade log -- always derived from the same `data` the charts
    render, so markers/table can never disagree with the chart. Costs one
    full Engine.run() (a Python loop over cached history) per call; not
    cached. The exact daily equity curve is attached to ``state`` for the
    dashboard PnL chart. Returns (closed_trades, open_entry) -- open_entry
    describes the live position if the promoted config is currently in a
    trade (else None).
    """
    from backtest.engine import BacktestConfig, Engine, trade_log

    state = state or compute_signal(module)
    strategy, data, params = state["strategy"], state["data"], state["params"]

    engine = Engine(BacktestConfig(transaction_cost_bps=strategy.transaction_cost_bps))
    result = engine.add_signal(strategy.make_pipeline(params)).run(data)
    state["equity_curve"] = result.equity_curve

    log = trade_log(result.closed_trades)
    if not log.is_empty():
        direction_sign = pl.when(pl.col("direction") == "long").then(1).otherwise(-1)
        log = log.with_columns(
            (pl.col("entry_level") - pl.col("entry_resid") + pl.col("entry_ou_mean"))
            .alias("target_lvl"),
            (direction_sign * (pl.col("entry_ou_mean") - pl.col("entry_resid")))
            .alias("expected_return_bps"),
        ).sort("exit_date", descending=True)

    open_entry = None
    if result.open_trades:  # max_positions=1 by default -- at most one
        pos = result.open_trades[0]
        current_level = float(data[strategy.target][-1])
        direction_sign = 1 if pos.direction == 1 else -1
        entry_resid = pos.entry_extras.get("resid")
        entry_ou_mean = pos.entry_extras.get("ou_mean")
        target_lvl = expected_return_bps = None
        if entry_resid is not None and entry_ou_mean is not None:
            target_lvl = pos.entry_level - entry_resid + entry_ou_mean
            expected_return_bps = direction_sign * (entry_ou_mean - entry_resid)
        open_entry = {
            "date": pos.entry_date,
            "level": pos.entry_level,
            "direction": "long" if pos.direction == 1 else "short",
            "entry_date": pos.entry_date,
            "exit_date": None,
            "entry_level": pos.entry_level,
            "exit_level": current_level,
            "target_lvl": target_lvl,
            "expected_return_bps": expected_return_bps,
            "pnl_bps": pos.unrealized_pnl(current_level),
            "entry_half_life": pos.entry_extras.get("half_life"),
            "bars_held": pos.bars_held,
            "exit_reason": "active",
        }

    return log, open_entry


def run_analysis(module: str) -> dict:
    """compute_signal() plus one auditable row appended to the ledger."""
    state = compute_signal(module)
    last, params = state["last"], state["params"]

    entry = {
        "module": module,
        "name": state["strategy"].name,
        "run_ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "data_asof_ts": str(state["data_asof"]),
        "level": _f(state["data"][state["strategy"].target][-1]),
        "signal": _f(last.get("signal")),
        "resid": _f(last.get("resid")),
        "ou_z": _f(last.get("ou_z")),
        "beta": _f(last.get("beta")),
        "r2": _f(last.get("r2")),
        "half_life": _f(last.get("half_life")),
        "entry_signal": params["entry_signal"],
        "entry_threshold": _f(params["entry_threshold"]),
        "gate_value": _f(last.get("gate_value")),
        "gate_percentile": _f(last.get("gate_percentile")),
        "gate_allow": last.get("gate_allow"),
        "fired": state["fired"],
    }
    SignalLedger().log(entry)
    return {**state, "ledger_entry": entry}
