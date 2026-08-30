"""2s vs 10s30s - front-end policy expectations against the long-end slope.

Question: is the 10s30s curve rich or cheap versus the level of the 2Y yield?
The 2Y concentrates expected policy-path repricing, while 10s30s expresses
long-end term-premium and convexity dynamics. The relationship is likely
regime-dependent, so the full causal gate/search funnel is part of the test
rather than an after-the-fact overlay.

Model:
    y = TARGET = 10s30s curve
    x = FEATURE = 2Y yield

Fit a changes-based rolling OLS of d(TARGET) on d(FEATURE), accumulate the
daily residuals into curve-bps space, and test both raw-residual and OU-z
entries. The traded expression is 10s30s; 2Y is the explanatory feature.

Directions are in curve space: a positive residual means 10s30s is steep/rich
versus the 2Y relationship -> short 10s30s. A negative residual means it is
flat/cheap -> long 10s30s.

    python -m book.curve.twos_10s30s              # live DB research candidate
    python -m book.curve.twos_10s30s --predict    # causal setup search
    python -m book.curve.twos_10s30s --exit       # exit search
    python -m book.curve.twos_10s30s --sweep      # exact engine + validation
    python -m book.curve.twos_10s30s --cook       # complete funnel
"""

from __future__ import annotations

from pathlib import Path

from backtest.strategy import DEFAULT_PARAMS, Strategy

STRATEGY_FAMILY = "curve"
SIGNAL_NAME = "twos_10s30s"

TICKERS = {
    "2y": "USGG2YR Index",  # % -> scaled to bps at load
    "10s30s": "USYC1030 Index",  # already quoted in bps
}

TARGET = "10s30s"
FEATURE = "2y"
FEATURES = [FEATURE]

# The broad cook favored a high-beta-volatility gate, but that gate has not
# admitted a new entry since 2021. Keep the module's runnable default on the
# nearby ungated neighborhood winner instead: it has more observations and
# continues to trade in the post-2022 regime. This is a research candidate,
# not a dashboard promotion.
RESEARCH_PARAMS = {
    "beta_lb": 90,
    "ou_lb": 470,
    "entry_signal": "ou_z",
    "entry_threshold": 0.9,
    "gate": None,
    "exit_style": "revert_frac",
    "exit_param": 1.0,
    "stop_loss_bps": 25.0,
}

# Named dashboard challengers. These deliberately remain ungated: the
# optimizer's higher-Sharpe beta-volatility gate stopped admitting entries
# after 2021. Variants are promoted independently and share this module's
# market-data cache and causal signal implementation.
DASHBOARD_VARIANTS = {
    "res50_e9": {
        "label": "RES50 · 9bps",
        "params": {
            **RESEARCH_PARAMS,
            "entry_signal": "residual",
            "beta_lb": 50,
            "ou_lb": 252,
            "entry_threshold": 9.0,
            "gate": None,
            "exit_style": "revert_frac",
            "exit_param": 0.25,
            "stop_loss_bps": 40.0,
        },
        "rationale": (
            "Best sample-rich raw-residual challenger: 125 trades, low "
            "best-trade concentration, and positive post-2022 performance."
        ),
    },
    "ou430_e09": {
        "label": "OU430 · 0.9z",
        "params": {
            **RESEARCH_PARAMS,
            "ou_lb": 430,
            "entry_threshold": 0.9,
        },
        "rationale": "Stronger post-2022 and trailing-one-year performance.",
    },
    "ou490_e07": {
        "label": "OU490 · 0.7z",
        "params": {
            **RESEARCH_PARAMS,
            "ou_lb": 490,
            "entry_threshold": 0.7,
        },
        "rationale": "Broader sample, smaller drawdown, flattest recent three-year path.",
    },
}


STRATEGY = Strategy(
    name=SIGNAL_NAME,
    module="book.curve.twos_10s30s",
    path=Path(__file__),
    tickers=TICKERS,
    bps_cols=["2y"],
    target=TARGET,
    feature=FEATURE,
    family=STRATEGY_FAMILY,
    default_params={**DEFAULT_PARAMS, **RESEARCH_PARAMS},
)

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
