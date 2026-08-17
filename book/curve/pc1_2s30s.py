"""2s30s vs PC1 of the yield panel - the curve hedged on the level factor.

Question: is the 2s30s curve rich or cheap versus the level factor of the
whole yield curve, rather than versus any single tenor?

Model:
    y = TARGET = 2s30s curve
    x = FEATURE = pc1_252 (rolling PC1 score of the yield panel, bps-scale)

PC1 is point-in-time and sign-fixed over a trailing 252-day window, so the
hedge carries no lookahead. Hedging on the level factor rather than one leg is
the cleaner strip when the curve's own short leg is itself a noisy proxy for
the level.

The PCA panel stays ['2y', '5y', '10y', '30y'] for every pair in this book. Widening it to
include the 20Y would truncate every PC1 series to 2020 onward, since the
panel is aligned on complete rows only.

PC1_LB provenance: xy_scan's winning PCA window for 2s30s. It is a tuning
parameter, not a finding - confirm it with this pair's own --predict before
leaning on it.

All machinery lives in backtest.strategy.Strategy - this module is the
configuration plus the PC1 feature hook. Same funnel as every curve strategy:

    python -m book.curve.pc1_2s30s              # single run, live DB
    python -m book.curve.pc1_2s30s --predict    # setup search
    python -m book.curve.pc1_2s30s --exit       # exits per saved setup
    python -m book.curve.pc1_2s30s --sweep      # exact engine + trade logs
    python -m book.curve.pc1_2s30s --cook       # all three, in order
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from backtest.strategy import Strategy
from stats import roll_pc1_score
from utils.market_data import align_columns

STRATEGY_FAMILY = "curve"
SIGNAL_NAME = "pc1_2s30s"

PC1_COLS = ['2y', '5y', '10y', '30y']
PC1_LB = 252  # xy_scan's winning PCA window for 2s30s

TICKERS = {
    "2y": "USGG2YR Index",
    "5y": "USGG5YR Index",
    "10y": "USGG10YR Index",
    "30y": "USGG30YR Index",
    "2s30s": "USYC2Y30 Index",  # already quoted in bps
}

TARGET = "2s30s"
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
    module="book.curve.pc1_2s30s",  # sweep workers import this
    path=Path(__file__),  # funnel artifacts live in data/pc1_2s30s/
    tickers=TICKERS,
    bps_cols=['2y', '5y', '10y', '30y'],
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
