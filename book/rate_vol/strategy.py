"""Rates volatility strategy — 1m10y implied vs realized richness (stub).

Is implied vol compensated vs realized after accounting for event risk?
Blocked on swaption vol surface coverage in the DB (md.swaption_vol) — see
notes/TODO.md ("Research - Rates Vol Signal").

For the worked end-to-end pattern a new strategy should follow, see
book/rate_vol/template.py.

Note: strategy modules are named strategy.py (not signal.py) because a file
named signal.py shadows the stdlib 'signal' module for any script run from
its directory, breaking subprocess/polars/psycopg imports.
"""

from __future__ import annotations

import polars as pl

from backtest import SignalPipeline, SignalConfig, TradeDef

STRATEGY_FAMILY = "rates_vol"
SIGNAL_NAME = "1m10y_implied_realized_richness"

TICKERS = {
    "move":  "MOVE Index",
    "10y":   "USGG10YR Index",
    "spx":   "SPX Index",
    "vix":   "VIX Index",
    # swaption vol surface data not yet in DB — add when available
}


def compute(data: pl.DataFrame) -> pl.DataFrame:
    raise NotImplementedError(
        f"{SIGNAL_NAME}: model not built — blocked on swaption vol data coverage"
    )


pipeline = SignalPipeline(
    name=SIGNAL_NAME,
    trade_def=TradeDef.outright("vol_rv", "move"),
    compute_fn=compute,
    config=SignalConfig(
        entry_long=-2.0,
        entry_short=2.0,
        exit_long=0.0,
        exit_short=0.0,
        stop_loss_bps=30.0,
        time_stop_bars=15,
    ),
)
