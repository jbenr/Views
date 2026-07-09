"""Backtest engine smoke tests on synthetic data — no DB required."""

import datetime as dt

import numpy as np
import polars as pl

from backtest import (
    BacktestConfig,
    Engine,
    SignalConfig,
    SignalPipeline,
    TradeDef,
    generate_signals,
    summary_table,
)


def make_data(n=600, seed=11):
    """Two series whose spread oscillates — guaranteed entries and reversions."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    a = 100 + rng.normal(0, 0.1, n).cumsum()
    b = a + 5 * np.sin(t / 15) + rng.normal(0, 0.2, n)
    start = dt.date(2020, 1, 1)
    ts = [start + dt.timedelta(days=int(i)) for i in range(n)]
    return pl.DataFrame({"ts": ts, "a": a, "b": b})


def spread_zscore(data: pl.DataFrame) -> pl.Series:
    spread = data["b"] - data["a"]
    mu = spread.rolling_mean(60)
    sd = spread.rolling_std(60)
    return (spread - mu) / sd


def make_pipeline(**config_kwargs):
    return SignalPipeline(
        name="test_spread",
        trade_def=TradeDef.spread("ab", "a", "b"),
        compute_fn=spread_zscore,
        config=SignalConfig(entry_long=-1.5, entry_short=1.5, **config_kwargs),
    )


def test_generate_signals_thresholds():
    sig = pl.Series([0.0, -2.0, 0.5, 2.5, -0.5, None])
    out = generate_signals(sig, SignalConfig(entry_long=-1.5, entry_short=1.5))
    # 0.0 is not strictly above/below the exit thresholds -> no action
    assert out["action"].to_list() == [
        None, "enter_long", "exit_long", "enter_short", "exit_short", None,
    ]


def test_engine_smoke():
    result = Engine(BacktestConfig()).add_signal(make_pipeline()).run(make_data())

    assert len(result.closed_trades) > 3
    assert len(result.equity_curve) == 600
    # cumulative pnl is the cumsum of daily pnl
    eq = result.equity_curve
    assert abs(eq["cumulative_pnl"][-1] - eq["pnl_bps"].sum()) < 1e-9

    m = result.summary()
    for key in ["total_pnl_bps", "n_trades", "hit_rate", "sharpe", "max_drawdown_bps"]:
        assert key in m
    assert m["n_trades"] == len(result.closed_trades)
    # mean-reverting spread + z-score fade should be profitable
    assert m["hit_rate"] > 0.5

    assert summary_table(result).shape[0] == 1


def test_time_stop_exits():
    result = (
        Engine(BacktestConfig())
        .add_signal(make_pipeline(exit_long=None, exit_short=None, time_stop_bars=5))
        .run(make_data())
    )
    assert result.closed_trades
    assert all(t.exit_reason == "time_stop" for t in result.closed_trades)
    assert all(t.bars_held == 5 for t in result.closed_trades)


def test_transaction_costs_reduce_pnl():
    data = make_data()
    free = Engine(BacktestConfig()).add_signal(make_pipeline()).run(data)
    costly = Engine(BacktestConfig(transaction_cost_bps=2.0)).add_signal(make_pipeline()).run(data)
    assert costly.summary()["total_pnl_bps"] < free.summary()["total_pnl_bps"]


def test_date_filtering():
    result = (
        Engine(BacktestConfig(start_date="2020-06-01", end_date="2020-12-31"))
        .add_signal(make_pipeline())
        .run(make_data())
    )
    dates = result.equity_curve["ts"]
    assert dates.min() >= dt.date(2020, 6, 1)
    assert dates.max() <= dt.date(2020, 12, 31)
