"""10s vs 10s30s - direction/curve interaction, on the Strategy template.

Question: is the 10s30s curve rich or cheap versus the level of 10Y rates?

Model:
    y = TARGET = 10s30s curve
    x = FEATURE = 10Y yield

Fit a changes-based rolling OLS of d(TARGET) on d(FEATURE), roll the daily
residuals into curve-bps level space, and fade raw residual extremes. OU state
is used as a gate and for exits/time stops rather than as the primary entry
threshold.

Directions are in CURVE space: positive residual means 10s30s is steep/rich vs
10Y -> short 10s30s; negative residual means 10s30s is flat/cheap vs 10Y ->
long 10s30s.

All machinery lives in backtest.strategy.Strategy — this module is the
configuration. Tune by overriding Strategy fields below (grids, params,
costs); see the Strategy docstring for the funnel and every knob.

    python -m book.curve.tens_10s30s              # single run, live DB
    python -m book.curve.tens_10s30s --synthetic  # single run, no DB
    python -m book.curve.tens_10s30s --predict    # setup search -> setups parquet
    python -m book.curve.tens_10s30s --exit       # exits per setup -> exits parquet
    python -m book.curve.tens_10s30s --sweep      # exact engine + trade logs
    python -m book.curve.tens_10s30s --cook       # all three, in order
    python book/curve/app.py                      # compare winners' trades (Dash)

    --cpu / --gpu force the scan device for --predict/--exit (default: auto).

Every mode returns a dict of state for interactive chaining: state = main().
"""

from __future__ import annotations

from pathlib import Path

from backtest.strategy import Strategy, synthetic_pair

STRATEGY_FAMILY = "curve"
SIGNAL_NAME = "tens_10s30s"

TICKERS = {
    "10y": "USGG10YR Index",  # % -> scaled to bps at load
    "10s30s": "USYC1030 Index",  # already quoted in bps
}

TARGET = "10s30s"
FEATURE = "10y"
FEATURES = [FEATURE]


def synthetic_data(n: int = 1500, seed: int = 21):
    """Synthetic substitute: 10s30s target explained by 10Y plus OU residual."""
    return synthetic_pair(TARGET, FEATURE, n=n, seed=seed)


STRATEGY = Strategy(
    name=SIGNAL_NAME,
    module="book.curve.tens_10s30s",  # sweep workers import this
    path=Path(__file__),  # funnel artifacts live next to this file
    tickers=TICKERS,
    bps_cols=["10y"],
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
