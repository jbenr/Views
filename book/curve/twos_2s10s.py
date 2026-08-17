"""2y vs 2s10s - the curve hedged against its own short leg.

Question: is the 2s10s curve rich or cheap versus the level of 2Y?

Model:
    y = TARGET = 2s10s curve
    x = FEATURE = 2Y yield

Fit a changes-based rolling OLS of d(TARGET) on d(FEATURE), roll the daily
residuals into curve-bps level space, and fade raw residual extremes. Hedging
on the curve's own short leg strips the direction a naive spread carries (the
long leg moves more), so what is left is closer to pure curve. OU state gates
entries and drives exits/time stops rather than being the entry threshold.

Directions are in CURVE space: positive residual means 2s10s is steep/rich
vs 2Y -> short 2s10s; negative means flat/cheap -> long 2s10s.

Futures expression: TU (short leg) vs UXY (long leg).

All machinery lives in backtest.strategy.Strategy - this module is the
configuration. Tune by overriding Strategy fields below.

    python -m book.curve.twos_2s10s              # single run, live DB
    python -m book.curve.twos_2s10s --synthetic  # single run, no DB
    python -m book.curve.twos_2s10s --predict    # setup search -> setups parquet
    python -m book.curve.twos_2s10s --exit       # exits per setup -> exits parquet
    python -m book.curve.twos_2s10s --sweep      # exact engine + trade logs
    python -m book.curve.twos_2s10s --cook       # all three, in order

Every mode returns a dict of state for interactive chaining: state = main().
"""

from __future__ import annotations

from pathlib import Path

from backtest.strategy import Strategy, synthetic_pair

STRATEGY_FAMILY = "curve"
SIGNAL_NAME = "twos_2s10s"

TICKERS = {
    "2y": "USGG2YR Index",  # % -> scaled to bps at load
    "2s10s": "USYC2Y10 Index",  # already quoted in bps
}

TARGET = "2s10s"
FEATURE = "2y"
FEATURES = [FEATURE]


def synthetic_data(n: int = 1500, seed: int = 33):
    """Synthetic substitute: 2s10s explained by 2Y plus an OU residual."""
    return synthetic_pair(TARGET, FEATURE, n=n, seed=seed)


STRATEGY = Strategy(
    name=SIGNAL_NAME,
    module="book.curve.twos_2s10s",  # sweep workers import this
    path=Path(__file__),  # funnel artifacts live in data/twos_2s10s/
    tickers=TICKERS,
    bps_cols=["2y"],
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

TRADES_FILE = STRATEGY.trades_file  # the comparison app reads this

if __name__ == "__main__":
    state = STRATEGY.cli()
