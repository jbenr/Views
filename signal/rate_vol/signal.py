from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest import SignalPipeline, SignalConfig, TradeDef

SIGNAL_NAME = "1m10y_implied_realized_richness"

TICKERS = {
    "move":  "MOVE Index",
    "10y":   "USGG10YR Index",
    "spx":   "SPX Index",
    "vix":   "VIX Index",
    # swaption vol surface data not yet in DB — add when available
}


def compute(data: pl.DataFrame) -> pl.DataFrame:
    raise NotImplementedError


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
