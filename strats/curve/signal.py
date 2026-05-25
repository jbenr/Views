from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest import SignalPipeline, SignalConfig, TradeDef

SIGNAL_NAME = "beta_weighted_5s30s"

TICKERS = {
    "5y":   "USGG5YR Index",
    "10y":  "USGG10YR Index",
    "30y":  "USGG30YR Index",
    "5s30s": "USYC5Y30 Index",
    "oil":  "CO1 Comdty",
    "dxy":  "DXY Curncy",
    "spx":  "SPX Index",
}


def compute(data: pl.DataFrame) -> pl.DataFrame:
    raise NotImplementedError


pipeline = SignalPipeline(
    name=SIGNAL_NAME,
    trade_def=TradeDef.spread("5s30s", "5y", "30y"),
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
