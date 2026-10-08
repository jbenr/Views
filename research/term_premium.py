"""Term premium: the 2s10s30s fly's residual against 2s10s, next to the NY Fed's ACM term premium.

The fly (2 x 10y - 2y - 30y) loads on the long end in a way slope alone
does not explain. Regressing it on 2s10s over a rolling window (only data
known each day) leaves a residual: how much the 10y is bid or offered
against its wings beyond what the curve's steepness implies -- a structural,
model-free read on the long end's compensation. The ACM term premium (Adrian,
Crump & Moench, published by the NY Fed) is the standard model-based
estimate; showing both says how far the cheap proxy tracks the model.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from research.fair_value import FairValueStudy
from research.panel import CATALOG, YIELDS
from stats import half_life
from utils.market_data import load_acm_term_premium, load_wide

FLY, CURVE = "2s10s30s", "2s10s"
TP_WINDOWS = [252, 504, 756, 1260]


def load_curve(start: str) -> pl.DataFrame:
    """2y, 10y and 30y yields and the fly and curve built from them, in bp."""
    legs = sorted({*CATALOG[FLY].legs, *CATALOG[CURVE].legs}, key=lambda t: int(t[:-1]))
    frame = load_wide({t: YIELDS[t] for t in legs}, start=start, bps_cols="all").drop_nulls().sort("ts")
    return frame.with_columns(CATALOG[FLY].composite_series(frame).alias(FLY),
                              CATALOG[CURVE].composite_series(frame).alias(CURVE))


def fly_residual(curve: pl.DataFrame, lookback: int) -> dict:
    """The fly regressed on 2s10s over a rolling window: fair value, residual (bp), z, beta, R2 and diagnostics.

    z is the residual over its own rolling standard deviation on the same
    window, so it is point-in-time too. R2 is the squared rolling correlation
    of the fly with 2s10s -- exact for one regressor; the rolling regression's
    own R2 accumulates residuals rather than refitting and can go negative.
    """
    study = FairValueStudy(target=FLY, factors=(CURVE,), lookback=lookback)
    out = study.research(curve.select("ts", FLY, CURVE))
    signals = out["signals"].with_columns(
        (pl.col("residual") / pl.col("residual").rolling_std(lookback, min_samples=lookback)).alias("z"),
        (pl.rolling_corr(pl.col(FLY), pl.col(CURVE), window_size=lookback) ** 2).alias("r2"))
    return {"signals": signals, "diagnostics": out["diagnostics"], "horizons": out["horizons"]}


def compare_to_acm(signals: pl.DataFrame, acm: pl.DataFrame, tenor: int = 10) -> dict:
    """How the fly residual lines up with the ACM term premium: matched on ACM's dates (month-ends when monthly).

    Returns the matched frame and correlations of levels and of changes
    between consecutive matched dates.
    """
    column = f"tp{tenor}"
    matched = (acm.select("ts", column).drop_nulls()
               .join_asof(signals.select("ts", "residual").drop_nulls().sort("ts"), on="ts", strategy="backward")
               .drop_nulls())
    if len(matched) < 3:
        return {"matched": matched, "level_corr": None, "change_corr": None}
    level = float(np.corrcoef(matched[column], matched["residual"])[0, 1])
    changes = matched.select(pl.col(column).diff(), pl.col("residual").diff()).drop_nulls()
    change = float(np.corrcoef(changes[column], changes["residual"])[0, 1]) if len(changes) > 2 else None
    return {"matched": matched, "level_corr": level, "change_corr": change}


def term_premium(start: str, lookback: int, tenor: int = 10) -> dict:
    """Everything the Term Premium tab shows."""
    curve = load_curve(start)
    model = fly_residual(curve, lookback)
    acm, frequency = load_acm_term_premium()
    acm = acm.filter(pl.col("ts") >= curve["ts"][0])
    tenor = tenor if f"tp{tenor}" in acm.columns else 10
    resid = model["signals"]["residual"].drop_nulls()
    return dict(curve=curve, signals=model["signals"], diagnostics=model["diagnostics"], acm=acm,
                acm_frequency=frequency, tenor=tenor, comparison=compare_to_acm(model["signals"], acm, tenor),
                half_life=float(half_life(resid)) if len(resid) > 20 else None)
