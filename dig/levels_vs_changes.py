"""What "fair" means under a levels OLS versus a changes OLS.

The same pair (10s30s modelled on 10Y) admits different model-implied fair
levels, and they disagree. This file draws them so the difference is visible
rather than theoretical:

  1. levels   -- roll_lr(x, y).yhat IS a fair level: alpha + beta*x. Beta is
                 fitted on levels, so it absorbs any shared trend between the
                 two series.
  2. changes, windowed -- roll_lr_diff(x, y) fits beta on daily changes (the
                 hedge ratio), which only states a daily move; a level is
                 rebuilt by accumulating the daily misses over a trailing
                 `lookback` window. Fair = y - rolling_sum(resid). This is what
                 backtest.strategy.Strategy.compute actually trades.

roll_lr_diff also offers `resid_cum`, the same daily residual accumulated from
the first bar of the sample. It is deliberately not modelled here: it is an
accumulation from an arbitrary anchor, so moving the sample start from 2010 to
2016 shifts today's "dislocation" by a constant ~56bp, and fading it earns a
Sharpe of roughly zero. Use a windowed accumulation, not a running one.

The question each chart answers: how far apart are these, does the residual
mean-revert under each definition, and does the answer depend on when your
sample happens to start?

    python -m dig.levels_vs_changes
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stats import half_life, horizon_backtest, hurst_exponent, roll_lr, roll_lr_diff
from utils.market_data import load_wide
from utils.viz import Viz

# ---- config -----------------------------------------------------------------

START = "2010-01-01"

TICKERS = {
    "10y": "USGG10YR Index",     # % -> scaled to bps at load
    "10s30s": "USYC1030 Index",  # already quoted in bps
}
BPS_COLS = ["10y"]

FEATURE = "10y"
TARGET = "10s30s"
LOOKBACK = 252

# sample starts used to test whether a definition of fair is anchor-dependent
ALT_STARTS = (START, "2013-01-01", "2016-01-01")

# label -> residual column, in the order the tables should read
RESID_COLS = {
    "levels (roll_lr)": "resid_levels",
    "changes, windowed": "resid_changes_window",
}
FAIR_COLS = ["fair_levels", "fair_changes_window"]


# ---- data -------------------------------------------------------------------

def load_panel(start: str = START) -> pd.DataFrame:
    """Feature/target pair on a common sample, yields in bps."""
    df = load_wide(TICKERS, start=start, bps_cols=BPS_COLS, to_pandas=True)
    df.index = pd.to_datetime(df.index)
    return df[[FEATURE, TARGET]].dropna()


# ---- the three fair values --------------------------------------------------

def fair_values(panel: pd.DataFrame, lookback: int = LOOKBACK) -> pd.DataFrame:
    """Model-implied fair levels for TARGET, with residuals and betas.

    A "fair level" is whatever the model says TARGET should be today; the
    residual is the observed level minus that. Under a levels fit the model
    states the level directly. Under a changes fit it only states the daily
    move, so a level has to be rebuilt by accumulating the daily misses over
    the trailing window.
    """
    x, y = panel[FEATURE], panel[TARGET]
    lv = roll_lr(x, y, lookback=lookback)
    ch = roll_lr_diff(x, y, lookback=lookback)  # one row shorter: first diff

    out = pd.DataFrame(index=panel.index)
    out[TARGET] = y

    # 1. levels fit — yhat is already a level
    out["fair_levels"] = lv["yhat"].to_numpy()
    out["resid_levels"] = lv["resid"].to_numpy()
    out["beta_levels"] = lv["beta"].to_numpy()

    # 2. changes fit, accumulated over a trailing window only
    pad = np.array([np.nan])
    resid_roll = np.concatenate(
        [pad, ch["resid"].rolling_sum(lookback, min_samples=lookback).to_numpy()]
    )
    out["resid_changes_window"] = resid_roll
    out["fair_changes_window"] = y - resid_roll

    out["beta_changes"] = np.concatenate([pad, ch["beta"].to_numpy()])
    return out


# ---- diagnostics ------------------------------------------------------------

def diagnose(fv: pd.DataFrame) -> pd.DataFrame:
    """Scale and mean-reversion character of each residual definition."""
    rows = []
    for label, col in RESID_COLS.items():
        r = fv[col].dropna()
        rows.append({
            "residual": label,
            "n": len(r),
            "mean": round(r.mean(), 2),
            "std": round(r.std(), 2),
            "min": round(r.min(), 1),
            "max": round(r.max(), 1),
            "last": round(r.iloc[-1], 1),
            "half_life_d": round(half_life(r), 1),
            "hurst": round(hurst_exponent(r), 3),
        })
    return pd.DataFrame(rows)


def fade_table(fv: pd.DataFrame, horizons: tuple[int, ...] = (5, 20, 60)) -> pd.DataFrame:
    """Fade each residual at several horizons: does the dislocation revert?"""
    frames = []
    for label, col in RESID_COLS.items():
        bt = horizon_backtest(fv[col].dropna(), horizons=horizons).to_pandas()
        bt.insert(0, "residual", label)
        frames.append(bt)
    return pd.concat(frames, ignore_index=True)


# ---- sample-start dependence ------------------------------------------------

def fair_by_start(
    starts: tuple[str, ...] = ALT_STARTS, lookback: int = LOOKBACK
) -> dict[str, pd.DataFrame]:
    """fair_values recomputed from each sample start date."""
    return {s: fair_values(load_panel(s), lookback=lookback) for s in starts}


def compare_starts(by_start: dict[str, pd.DataFrame], col: str) -> pd.DataFrame:
    """One residual definition across sample starts, on shared dates.

    A definition of fair that is genuinely a property of the market gives the
    same number today regardless of when the sample began. One that is an
    accumulation from an arbitrary anchor does not.
    """
    frame = pd.DataFrame({f"from {s[:4]}": fv[col] for s, fv in by_start.items()})
    return frame.dropna(how="all")


def start_spread(by_start: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """How much each residual definition shifts when the sample start moves."""
    rows = []
    for label, col in RESID_COLS.items():
        frame = compare_starts(by_start, col).dropna()
        gaps = frame.max(axis=1) - frame.min(axis=1)
        rows.append({
            "residual": label,
            "n_shared_days": len(frame),
            "mean_gap_bps": round(gaps.mean(), 2),
            "max_gap_bps": round(gaps.max(), 2),
        })
    return pd.DataFrame(rows)


# ---- main -------------------------------------------------------------------

def main():
    # plotly backend: same matplotlib charts, plus tables, served in a browser
    viz = Viz(backend="plotly")

    # the same model fitted from three sample starts; START is the base case
    by_start = fair_by_start()
    fv = by_start[START]

    # how big is each dislocation, and does it mean-revert?
    diagnostics = diagnose(fv)

    # is it tradable — does fading it pay, and over what horizon?
    fades = fade_table(fv)

    # sanity check: neither definition should depend on when the sample began
    anchor_sensitivity = start_spread(by_start)

    # 1. the headline: the fair lines against the actual curve
    viz.line(
        fv[[TARGET] + FAIR_COLS],
        title=f"{TARGET} vs model-implied fair levels ({FEATURE}, {LOOKBACK}d)",
        yaxis_title="bps",
    )

    # 2. the residuals those fair lines imply
    viz.line(
        fv[list(RESID_COLS.values())],
        title=f"{TARGET} dislocation under each definition of fair",
        yaxis_title="bps",
        hlines=[0],
    )

    # 3. levels beta absorbs the trend; changes beta is the hedge ratio
    viz.line(
        fv[["beta_levels", "beta_changes"]],
        title=f"hedge ratio: levels fit vs changes fit ({LOOKBACK}d)",
    )

    viz.table(diagnostics, title="residual scale and mean-reversion character")
    viz.table(fades, title="fade the residual — IC / hit / Sharpe by horizon")
    viz.table(anchor_sensitivity, title="how far each definition moves when the start date moves")

    return {
        "by_start": by_start,
        "fv": fv,
        "diagnostics": diagnostics,
        "fades": fades,
        "anchor_sensitivity": anchor_sensitivity,
    }


if __name__ == "__main__":
    state = main()
