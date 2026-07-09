"""Directional duration strategy — 10Y fair value vs macro anchors (stub).

Is 10Y rich or cheap vs macro fundamentals (breakevens, oil, DXY, equities)?
The residual model research lives in drill_to_the_core.py and spread_rv.py;
regime conditioning in signal_context.py and setups.py. compute() will be
filled in once a model wins the out-of-sample bake-off (see notes/TODO.md).
"""

from __future__ import annotations

import polars as pl

from backtest import SignalPipeline, SignalConfig, TradeDef

STRATEGY_FAMILY = "directional_duration"
SIGNAL_NAME = "10y_duration_fair_value"

TICKERS = {
    "10y":  "USGG10YR Index",
    "2y":   "USGG2YR Index",
    "5y":   "USGG5YR Index",
    "be10": "USGGBE10 Index",
    "spx":  "SPX Index",
    "oil":  "CO1 Comdty",
    "dxy":  "DXY Curncy",
}


def compute(data: pl.DataFrame) -> pl.DataFrame:
    raise NotImplementedError(
        f"{SIGNAL_NAME}: model not finalized — see book/duration research files"
    )


pipeline = SignalPipeline(
    name=SIGNAL_NAME,
    trade_def=TradeDef.outright("10y_dur", "10y"),
    compute_fn=compute,
    config=SignalConfig(
        entry_long=-2.0,
        entry_short=2.0,
        exit_long=0.0,
        exit_short=0.0,
        stop_loss_bps=50.0,
        time_stop_bars=20,
    ),
)
