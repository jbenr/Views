"""Curve strategy — beta-weighted 10s30s mean reversion.

Regress Δ30Y on Δ10Y to strip hidden direction from the naive spread, roll
the residual into level space, and fade OU z-score extremes. Parameter
choices are justified by the diagnostics in book/curve/research.py.
"""

from __future__ import annotations

import polars as pl

from backtest import SignalPipeline, SignalConfig, TradeDef
from stats.ols import roll_lr_diff
from stats.ou import roll_ou_zscore

STRATEGY_FAMILY = "curve"
SIGNAL_NAME = "beta_weighted_10s30s"

# Hedge ratio lookback. research.py diag 2: 63d σ=0.113 vs 21d σ=0.137.
# Stable enough, responsive enough to regime shifts.
BETA_LB = 63

# Z-score lookback matches BETA_LB — same horizon for hedge and signal.
ZSCORE_LB = 63

# research.py diag 4 finding: longs hit 73.5% at z=2.0 over 40d forward.
# Shorts hit 47.2% — below random. Curve has structural steepening bias
# (QE, fiscal, term premium) that makes short-curve entries unreliable.
# Until we find a robust short filter, we bias toward longs.
ENTRY_LONG_Z  = -2.0   # z < -2.0 → buy cheap spread (works: 73.5% hit rate)
ENTRY_SHORT_Z =  2.5   # z > +2.5 → sell rich spread (raise bar for weak side)

TICKERS = {
    "10y":  "USGG10YR Index",
    "30y":  "USGG30YR Index",
    "move": "MOVE Index",
}


def compute(data: pl.DataFrame) -> pl.DataFrame:
    """Beta-weighted 10s30s OU z-score.

    1. Changes-based OLS: regress Δ30Y on Δ10Y (BETA_LB window) → hedge ratio β
    2. Roll the daily residuals → level-space spread with direction stripped
    3. OU z-score of that rolling residual → signal

    Returns 'signal' (z-score) and 'resid_roll' (for exit_fn inspection).
    The leading null from roll_lr_diff's first-diff is padded so output
    length matches input.
    """
    reg = roll_lr_diff(data["10y"], data["30y"], lookback=BETA_LB)

    # roll_lr_diff drops one row (the first diff is null) — pad back to len(data)
    null1 = pl.Series([None], dtype=pl.Float64)
    resid_roll = pl.concat([
        null1,
        reg["resid"].rolling_sum(BETA_LB, min_samples=BETA_LB),
    ])

    z = roll_ou_zscore(resid_roll, lookback=ZSCORE_LB)
    return pl.DataFrame({"signal": z, "resid_roll": resid_roll})


pipeline = SignalPipeline(
    name=SIGNAL_NAME,
    trade_def=TradeDef.spread("10s30s", "10y", "30y"),
    compute_fn=compute,
    config=SignalConfig(
        entry_long=ENTRY_LONG_Z,
        entry_short=ENTRY_SHORT_Z,
        exit_long=0.0,
        exit_short=0.0,
        stop_loss_bps=25.0,
        # 2× rolling median half-life (19d from research.py diag 3)
        time_stop_bars=40,
    ),
)
