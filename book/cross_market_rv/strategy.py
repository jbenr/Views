"""Cross-market RV strategy — USD-hedged gilt vs UST (stub).

Is a 10Y gilt hedged back to USD cheap vs the duration-matched Treasury?
Blocked on FX forwards and cross-currency basis data — see notes/TODO.md
("Research - Cross-Market RV").
"""

from __future__ import annotations

import polars as pl

from backtest import SignalPipeline, SignalConfig, TradeDef

STRATEGY_FAMILY = "cross_market_rv"
SIGNAL_NAME = "hedged_gilt_vs_ust"

TICKERS = {
    "ust10":   "USGG10YR Index",
    "gilt10":  "GUKG10 Index",
    "bund10":  "GDBR10 Index",
    "gbpusd":  "GBPUSD Curncy",
    "eurusd":  "EURUSD Curncy",
    # xccy basis and FX forwards not yet in DB — add when available
}


def compute(data: pl.DataFrame) -> pl.DataFrame:
    raise NotImplementedError(
        f"{SIGNAL_NAME}: model not built — blocked on FX forward / xccy basis data"
    )


pipeline = SignalPipeline(
    name=SIGNAL_NAME,
    trade_def=TradeDef.spread("gilt_ust", "ust10", "gilt10"),
    compute_fn=compute,
    config=SignalConfig(
        entry_long=-2.0,
        entry_short=2.0,
        exit_long=0.0,
        exit_short=0.0,
        stop_loss_bps=30.0,
        time_stop_bars=20,
    ),
)
