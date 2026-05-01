from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest import SignalPipeline, SignalConfig, TradeDef

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
    raise NotImplementedError


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
