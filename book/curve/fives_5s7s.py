"""5y vs 5s7s - the curve hedged against its own short leg.

Question: is the 5s7s curve rich or cheap versus the level of 5Y?

Model:
    y = TARGET = 5s7s curve
    x = FEATURE = 5Y yield

Fit a changes-based rolling OLS of d(TARGET) on d(FEATURE), roll the daily
residuals into curve-bps level space, and fade raw residual extremes. Hedging
on the curve's own short leg strips the direction a naive spread carries (the
long leg moves more), so what is left is closer to pure curve. OU state gates
entries and drives exits/time stops rather than being the entry threshold.

Directions are in CURVE space: positive residual means 5s7s is steep/rich
vs 5Y -> short 5s7s; negative means flat/cheap -> long 5s7s.

Futures expression: FV (short leg) vs TY (long leg).

All machinery lives in backtest.strategy.Strategy - this module is the
configuration. Tune by overriding Strategy fields below.

    python -m book.curve.fives_5s7s              # single run, live DB
    python -m book.curve.fives_5s7s --synthetic  # single run, no DB
    python -m book.curve.fives_5s7s --predict    # setup search -> setups parquet
    python -m book.curve.fives_5s7s --exit       # exits per setup -> exits parquet
    python -m book.curve.fives_5s7s --sweep      # exact engine + trade logs
    python -m book.curve.fives_5s7s --cook       # all three, in order

Every mode returns a dict of state for interactive chaining: state = main().
"""

from __future__ import annotations

from pathlib import Path

from backtest.strategy import Strategy, synthetic_pair

STRATEGY_FAMILY = "curve"
SIGNAL_NAME = "fives_5s7s"

TICKERS = {
    "5y": "USGG5YR Index",  # % -> scaled to bps at load
    "5s7s": "USYC5Y7Y Index",  # already quoted in bps
}

TARGET = "5s7s"
FEATURE = "5y"
FEATURES = [FEATURE]


def synthetic_data(n: int = 1500, seed: int = 35):
    """Synthetic substitute: 5s7s explained by 5Y plus an OU residual."""
    return synthetic_pair(TARGET, FEATURE, n=n, seed=seed)


STRATEGY = Strategy(
    name=SIGNAL_NAME,
    module="book.curve.fives_5s7s",  # sweep workers import this
    path=Path(__file__),  # funnel artifacts live in data/fives_5s7s/
    tickers=TICKERS,
    bps_cols=["5y"],
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
