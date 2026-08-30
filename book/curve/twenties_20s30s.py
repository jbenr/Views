"""20y vs 20s30s - the curve hedged against its own short leg.

Question: is the 20s30s curve rich or cheap versus the level of 20Y?

Model:
    y = TARGET = 20s30s curve
    x = FEATURE = 20Y yield

Fit a changes-based rolling OLS of d(TARGET) on d(FEATURE), roll the daily
residuals into curve-bps level space, and fade raw residual extremes. Hedging
on the curve's own short leg strips the direction a naive spread carries (the
long leg moves more), so what is left is closer to pure curve. OU state gates
entries and drives exits/time stops rather than being the entry threshold.

Directions are in CURVE space: positive residual means 20s30s is steep/rich
vs 20Y -> short 20s30s; negative means flat/cheap -> long 20s30s.

Futures expression: US (short leg) vs WN (long leg).

History note: the 20Y was reintroduced in May 2020, so USGG20YR / USYC2030
start 2020-05-21 (~1,600 rows) against ~4,300 for every other pair. Era and
regime checks here rest on a much thinner sample.

All machinery lives in backtest.strategy.Strategy - this module is the
configuration. Tune by overriding Strategy fields below.

    python -m book.curve.twenties_20s30s              # single run, live DB
    python -m book.curve.twenties_20s30s --predict    # setup search -> setups parquet
    python -m book.curve.twenties_20s30s --exit       # exits per setup -> exits parquet
    python -m book.curve.twenties_20s30s --sweep      # exact engine + trade logs
    python -m book.curve.twenties_20s30s --cook       # all three, in order

Every mode returns a dict of state for interactive chaining: state = main().
"""

from __future__ import annotations

from pathlib import Path

from backtest.strategy import Strategy

STRATEGY_FAMILY = "curve"
SIGNAL_NAME = "twenties_20s30s"

TICKERS = {
    "20y": "USGG20YR Index",  # % -> scaled to bps at load
    "20s30s": "USYC2030 Index",  # already quoted in bps
}

TARGET = "20s30s"
FEATURE = "20y"
FEATURES = [FEATURE]


STRATEGY = Strategy(
    name=SIGNAL_NAME,
    module="book.curve.twenties_20s30s",  # sweep workers import this
    path=Path(__file__),  # funnel artifacts live in data/twenties_20s30s/
    tickers=TICKERS,
    bps_cols=["20y"],
    target=TARGET,
    feature=FEATURE,
    family=STRATEGY_FAMILY,
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
