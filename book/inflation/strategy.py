"""Inflation strategy — 10Y breakeven fair value (stub).

Are 10Y breakevens rich or cheap vs oil, dollar, and CPI trend? Trade the
TIPS/nominal pair. Model research not started — see notes/TODO.md
("Research - Inflation Signal") for the build plan.
"""

from __future__ import annotations

import polars as pl

from backtest import SignalPipeline, SignalConfig, TradeDef

STRATEGY_FAMILY = "inflation"
SIGNAL_NAME = "10y_breakeven_fair_value"

TICKERS = {
    "be10": "USGGBE10 Index",
    "be5":  "USGGBE05 Index",
    "10y":  "USGG10YR Index",
    "tips10": "USGGT10Y Index",
    "oil":  "CO1 Comdty",
    "dxy":  "DXY Curncy",
    "spx":  "SPX Index",
}


def compute(data: pl.DataFrame) -> pl.DataFrame:
    raise NotImplementedError(
        f"{SIGNAL_NAME}: model not built — see notes/TODO.md inflation research plan"
    )


pipeline = SignalPipeline(
    name=SIGNAL_NAME,
    trade_def=TradeDef.spread("10y_be", "tips10", "10y"),
    compute_fn=compute,
    config=SignalConfig(
        entry_long=-2.0,
        entry_short=2.0,
        exit_long=0.0,
        exit_short=0.0,
        stop_loss_bps=20.0,
        time_stop_bars=20,
    ),
)
