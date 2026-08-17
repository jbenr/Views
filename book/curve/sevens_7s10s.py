"""7y vs 7s10s - the curve hedged against its own short leg.

Question: is the 7s10s curve rich or cheap versus the level of 7Y?

Model:
    y = TARGET = 7s10s curve
    x = FEATURE = 7Y yield

Fit a changes-based rolling OLS of d(TARGET) on d(FEATURE), roll the daily
residuals into curve-bps level space, and fade raw residual extremes. Hedging
on the curve's own short leg strips the direction a naive spread carries (the
long leg moves more), so what is left is closer to pure curve. OU state gates
entries and drives exits/time stops rather than being the entry threshold.

Directions are in CURVE space: positive residual means 7s10s is steep/rich
vs 7Y -> short 7s10s; negative means flat/cheap -> long 7s10s.

Futures expression: TY (short leg) vs UXY (long leg).

All machinery lives in backtest.strategy.Strategy - this module is the
configuration. Tune by overriding Strategy fields below.

    python -m book.curve.sevens_7s10s              # single run, live DB
    python -m book.curve.sevens_7s10s --synthetic  # single run, no DB
    python -m book.curve.sevens_7s10s --predict    # setup search -> setups parquet
    python -m book.curve.sevens_7s10s --exit       # exits per setup -> exits parquet
    python -m book.curve.sevens_7s10s --sweep      # exact engine + trade logs
    python -m book.curve.sevens_7s10s --cook       # all three, in order

Every mode returns a dict of state for interactive chaining: state = main().
"""

from __future__ import annotations

from pathlib import Path

from backtest.strategy import Strategy, synthetic_pair

STRATEGY_FAMILY = "curve"
SIGNAL_NAME = "sevens_7s10s"

TICKERS = {
    "7y": "USGG7YR Index",  # % -> scaled to bps at load
    "7s10s": "USYC7Y10 Index",  # already quoted in bps
}

TARGET = "7s10s"
FEATURE = "7y"
FEATURES = [FEATURE]


def synthetic_data(n: int = 1500, seed: int = 38):
    """Synthetic substitute: 7s10s explained by 7Y plus an OU residual."""
    return synthetic_pair(TARGET, FEATURE, n=n, seed=seed)


STRATEGY = Strategy(
    name=SIGNAL_NAME,
    module="book.curve.sevens_7s10s",  # sweep workers import this
    path=Path(__file__),  # funnel artifacts live in data/sevens_7s10s/
    tickers=TICKERS,
    bps_cols=["7y"],
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
