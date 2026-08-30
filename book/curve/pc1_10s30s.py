"""10s30s vs PC1 of the yield panel - the incumbent trade, hedged better.

Graduated from the xy_scan explorer: hedging 10s30s against the LEVEL FACTOR
(point-in-time PC1 of 2y/5y/10y/30y at a 126d window, sign-fixed, no
lookahead) scored materially better than hedging against 10y alone
(nbr_ic 0.74 vs 0.63). Same trade as tens_10s30s, cleaner direction strip.

Model:
    y = TARGET = 10s30s curve
    x = FEATURE = pc1_126 (rolling PC1 score of the yield panel, bps-scale)

All machinery lives in backtest.strategy.Strategy — this module is the
configuration plus the PC1 feature hook. Same funnel as every curve strategy:

    python -m book.curve.pc1_10s30s              # single run, live DB
    python -m book.curve.pc1_10s30s --predict    # setup search
    python -m book.curve.pc1_10s30s --exit       # exits per saved setup
    python -m book.curve.pc1_10s30s --sweep      # exact engine + trade logs
    python -m book.curve.pc1_10s30s --cook       # all three, in order
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from backtest.strategy import Strategy
from stats import roll_pc1_score
from utils.market_data import align_columns

STRATEGY_FAMILY = "curve"
SIGNAL_NAME = "pc1_10s30s"

PC1_COLS = ["2y", "5y", "10y", "30y"]
PC1_LB = 126  # xy_scan's winning PCA window for 10s30s

TICKERS = {
    "2y": "USGG2YR Index",
    "5y": "USGG5YR Index",
    "10y": "USGG10YR Index",
    "30y": "USGG30YR Index",
    "10s30s": "USYC1030 Index",  # already quoted in bps
}

TARGET = "10s30s"
FEATURE = f"pc1_{PC1_LB}"
FEATURES = [FEATURE]


def add_pc1(data: pl.DataFrame) -> pl.DataFrame:
    """Feature hook: point-in-time, sign-fixed PC1 score of the yield panel."""
    panel = align_columns(data, PC1_COLS)
    scores = panel.select(
        "ts",
        roll_pc1_score(panel.select(PC1_COLS), lookback=PC1_LB).alias(FEATURE),
    )
    return data.join(scores, on="ts", how="left")


STRATEGY = Strategy(
    name=SIGNAL_NAME,
    module="book.curve.pc1_10s30s",  # sweep workers import this
    path=Path(__file__),  # funnel artifacts live in data/pc1_10s30s/
    tickers=TICKERS,
    bps_cols=["2y", "5y", "10y", "30y"],
    target=TARGET,
    feature=FEATURE,
    family=STRATEGY_FAMILY,
    feature_fn=add_pc1,
)

# lab worker contract + interactive / app API
compute = STRATEGY.compute
make_pipeline = STRATEGY.make_pipeline
pipeline = STRATEGY.pipeline
load_data = STRATEGY.load_data
model_frame = STRATEGY.model_frame
main = STRATEGY.main
predict = STRATEGY.predict
exit_scan = STRATEGY.exit_scan
sweep = STRATEGY.sweep

TRADES_FILE = STRATEGY.trades_file

if __name__ == "__main__":
    state = STRATEGY.cli()
