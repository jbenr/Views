from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest import SignalPipeline, SignalConfig, TradeDef

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
    raise NotImplementedError


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
