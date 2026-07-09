"""Event-driven strategy — post-CPI 10Y continuation (stub).

Does a hot CPI surprise with strong first-15-minute bearish price action
show continuation? Needs an event calendar with consensus/actual/surprise —
see notes/TODO.md ("Research - Event-Driven").
"""

from __future__ import annotations

import polars as pl

from backtest import SignalPipeline, SignalConfig, TradeDef

STRATEGY_FAMILY = "event_driven_macro"
SIGNAL_NAME = "post_cpi_10y_continuation"

TICKERS = {
    "10y":  "USGG10YR Index",
    "2y":   "USGG2YR Index",
    "spx":  "SPX Index",
    # event calendar (CPI surprises, FOMC) sourced separately
}


def compute(data: pl.DataFrame) -> pl.DataFrame:
    raise NotImplementedError(
        f"{SIGNAL_NAME}: model not built — needs event calendar data"
    )


pipeline = SignalPipeline(
    name=SIGNAL_NAME,
    trade_def=TradeDef.outright("10y_event", "10y"),
    compute_fn=compute,
    config=SignalConfig(
        entry_long=-1.5,
        entry_short=1.5,
        exit_long=0.0,
        exit_short=0.0,
        stop_loss_bps=20.0,
        time_stop_bars=5,
    ),
)
