from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest import SignalPipeline, SignalConfig, TradeDef

SIGNAL_NAME = "post_cpi_10y_continuation"

TICKERS = {
    "10y":  "USGG10YR Index",
    "2y":   "USGG2YR Index",
    "spx":  "SPX Index",
    # event calendar (CPI surprises, FOMC) sourced separately
}


def compute(data: pl.DataFrame) -> pl.DataFrame:
    raise NotImplementedError


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
