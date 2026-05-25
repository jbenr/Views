from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest import SignalPipeline, SignalConfig, TradeDef

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
    raise NotImplementedError


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
