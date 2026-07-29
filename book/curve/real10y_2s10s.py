"""2s10s vs real 10Y - the front curve against the real-rate level.

Graduated from the xy_scan explorer (best nbr_ic 0.83, the most-corroborated
pair on the board): real-rate shocks are Fed-path shocks, and the 2s10s slope
mean-reverts around its beta-hedged relationship to the 10Y real yield.

Model:
    y = TARGET = 2s10s curve
    x = FEATURE = 10Y real yield (USGGT10Y)

All machinery lives in backtest.strategy.Strategy — this module is the
configuration. Same funnel as every curve strategy:

    python -m book.curve.real10y_2s10s              # single run, live DB
    python -m book.curve.real10y_2s10s --synthetic  # single run, no DB
    python -m book.curve.real10y_2s10s --predict    # setup search
    python -m book.curve.real10y_2s10s --exit       # exits per saved setup
    python -m book.curve.real10y_2s10s --sweep      # exact engine + trade logs
    python -m book.curve.real10y_2s10s --cook       # all three, in order
"""

from __future__ import annotations

from pathlib import Path

from backtest.strategy import Strategy, synthetic_pair

STRATEGY_FAMILY = "curve"
SIGNAL_NAME = "real10y_2s10s"

TICKERS = {
    "real10y": "USGGT10Y Index",  # % -> scaled to bps at load
    "2s10s": "USYC2Y10 Index",  # already quoted in bps
}

TARGET = "2s10s"
FEATURE = "real10y"
FEATURES = [FEATURE]


def synthetic_data(n: int = 1500, seed: int = 11):
    """Synthetic substitute: 2s10s explained by real 10Y plus OU residual."""
    return synthetic_pair(
        TARGET, FEATURE, n=n, seed=seed, feature_level=150.0, target_base=80.0
    )


STRATEGY = Strategy(
    name=SIGNAL_NAME,
    module="book.curve.real10y_2s10s",  # sweep workers import this
    path=Path(__file__),  # funnel artifacts live in data/real10y_2s10s/
    tickers=TICKERS,
    bps_cols=["real10y"],
    target=TARGET,
    feature=FEATURE,
    family=STRATEGY_FAMILY,
    synthetic_fn=synthetic_data,
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
