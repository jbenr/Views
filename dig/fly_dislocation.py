"""Does 10s20s30s fade on its own, and does rate vol gate it?

dig.fly_vs_vol_scan killed vol as a regressor: nothing to residualize. This
tests the two weaker claims that survive that result.

  1. standalone -- the fly has a ~40d half-life and hurst < 0.5. Does fading
     its own dislocation actually earn anything, and at which horizon?
  2. gated      -- vol may not co-move with the fly but still say WHEN the
     fade works. Split the same signal by vol tercile and compare.

The dislocation is defined two ways, because the framework offers both and
they are not the same object:

  z_level  -- (fly - rolling mean) / rolling std. A level dislocation; this
              is what a ~40d half-life implies you should fade.
  d_daily  -- today's change in the fly. This is what DislocationStudy calls
              `dislocation` when features=(); a one-day object.

Scored with research.common.fade_scorecard so the numbers are comparable to
every other study in the book. Descriptive only -- no costs, no exits.

Two things this file is careful about, because a first pass got both wrong:

  * the vol tercile is cut on an EXPANDING quantile, not a full-sample one.
    A full-sample cut tells you today whether vol is low relative to levels
    it only reaches next year.
  * horizons overlap, so n_obs daily bars at horizon h carry roughly h times
    fewer independent observations. `n_indep` reports that, and any Sharpe
    backed by a single-digit n_indep is noise however large it looks.

    python -m dig.fly_dislocation
"""

from __future__ import annotations

import numpy as np
import polars as pl

import utils
from research.common import fade_scorecard

from .fly_vs_vol import load_data

LOOKBACK = 126
HORIZONS = (1, 5, 10, 20, 40, 60)
VOL_COL = "vol"
REGIME_MIN_OBS = 252  # burn-in before the expanding vol quantiles mean anything

# ---- helpers ----------------------------------------------------------------


def pit_regime(vol: pl.Series, min_obs: int = REGIME_MIN_OBS) -> pl.Series:
    """Point-in-time vol tercile: quantiles from prior bars only, never the future."""
    values = vol.to_numpy()
    labels: list[str | None] = []
    for i, v in enumerate(values):
        if i < min_obs:
            labels.append(None)
            continue
        past = values[:i]  # strictly prior bars
        q33, q66 = np.quantile(past, 1 / 3), np.quantile(past, 2 / 3)
        labels.append("low" if v <= q33 else "mid" if v <= q66 else "high")
    return pl.Series("vol_regime", labels, dtype=pl.Utf8)


def add_signals(data: pl.DataFrame, lookback: int = LOOKBACK) -> pl.DataFrame:
    """Attach both dislocation definitions and a point-in-time vol tercile."""
    frame = data.drop_nulls(subset=["fly", VOL_COL]).sort("ts")
    mean = frame["fly"].rolling_mean(lookback)
    std = frame["fly"].rolling_std(lookback)
    return frame.with_columns(
        ((frame["fly"] - mean) / std).alias("z_level"),
        frame["fly"].diff().alias("d_daily"),
        pit_regime(frame[VOL_COL]),
    )


def score(frame: pl.DataFrame, signal: str, label: str) -> pl.DataFrame:
    """Fade scorecard for one signal on one subsample, tagged and overlap-aware."""
    out = fade_scorecard(frame[signal], frame["fly"], HORIZONS)
    return out.with_columns(
        pl.lit(label).alias("subsample"),
        # overlapping bars: h-day horizons reuse the same underlying moves
        (pl.col("n_obs") // pl.col("horizon")).alias("n_indep"),
    ).select(
        "subsample", "horizon", "n_obs", "n_indep",
        *[c for c in out.columns if c not in ("horizon", "n_obs")],
    )


def episode_count(frame: pl.DataFrame) -> pl.DataFrame:
    """How many contiguous runs each regime is made of -- the real draw count."""
    return (
        frame.drop_nulls("vol_regime")
        .with_columns(
            (pl.col("vol_regime") != pl.col("vol_regime").shift()).cum_sum().alias("run")
        )
        .group_by("vol_regime")
        .agg(pl.col("run").n_unique().alias("n_episodes"), pl.len().alias("n_days"))
        .sort("vol_regime")
    )


def main() -> dict:
    # panel with both dislocation definitions and the point-in-time vol regime
    frame = add_signals(load_data())

    # 1: does the fly fade on its own, under either definition?
    standalone = pl.concat([
        score(frame, "z_level", "all / z_level"),
        score(frame, "d_daily", "all / d_daily"),
    ])

    # 2: does vol gate the level fade? same signal, three point-in-time regimes
    gated = pl.concat([
        score(frame.filter(pl.col("vol_regime") == r), "z_level", f"{r} vol / z_level")
        for r in ("low", "mid", "high")
    ])

    # a tercile made of three contiguous episodes is three draws, not 300
    episodes = episode_count(frame)

    print(f"sample {frame['ts'][0]} -> {frame['ts'][-1]}  rows={len(frame)}")
    print("\nstandalone fade of the fly's own dislocation:")
    utils.pdf(standalone)
    print("\nsame level signal, split by point-in-time 1Mo x 30y vol tercile:")
    utils.pdf(gated)
    print("\nhow many distinct episodes each regime is actually made of:")
    utils.pdf(episodes)
    return {"frame": frame, "standalone": standalone, "gated": gated,
            "episodes": episodes}


if __name__ == "__main__":
    state = main()
