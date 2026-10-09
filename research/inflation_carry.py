"""Breakeven inflation carry: what a long breakeven earns over a horizon if nothing moves.

A long breakeven (long TIPS, short nominals) accrues realized inflation on
the TIPS side and pays the breakeven rate on the nominal side; over a
horizon h that is (near-term inflation - breakeven) x h, divided by the
position's duration to express it in breakeven bp. With no CPI fixings in the
database, near-term inflation is the market's own: the 1y zero-coupon
inflation swap. Roll-down is the breakeven sliding along the curve as an
N-year becomes an (N-h)-year, read off the inflation swap curve because the
breakeven curve has only 5/10/30y points. Carry = accrual + roll-down: how
many bp the breakeven can fall over the horizon before a long loses money.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from utils.market_data import load_wide

BREAKEVENS = {5: "USGGBE05 Index", 10: "USGGBE10 Index", 30: "USGGBE30 Index"}
SWAP_TENORS = (1, 2, 3, 4, 5, 7, 10, 20, 30)
HORIZONS = {"1m": 1 / 12, "3m": 0.25, "1y": 1.0}
# One-day inflation-swap spikes larger than this (bp), that the next print undoes, are bad prints.
SPIKE_BP = {1: 40.0}  # the 1y moves most; longer tenors barely move day to day
SPIKE_BP_DEFAULT = 20.0


def drop_spikes(values: np.ndarray, jump: float) -> np.ndarray:
    """Replace isolated one-day spikes with the previous print.

    A spike jumps more than ``jump`` away from both neighbours, which agree
    with each other to within ``jump`` (e.g. the 1y swap at 366 -> 256 -> 357,
    or Good Friday 2024's +250 -> -212 -> +252). A real move, like late 2008's
    collapse, does not come straight back and stays. This looks at the next
    print, so it is data cleaning for display, not an input to a signal.
    """
    out = values.astype(float).copy()
    good = np.flatnonzero(np.isfinite(out))
    for a, b, c in zip(good[:-2], good[1:-1], good[2:]):
        before, now, after = out[a], out[b], out[c]
        if abs(now - before) > jump and abs(now - after) > jump and abs(after - before) <= jump \
                and np.sign(now - before) == np.sign(now - after):
            out[b] = before
    return out


def load_inflation(start: str) -> pl.DataFrame:
    """5/10/30y breakevens and the 1-30y zero-coupon inflation swap curve, in bp, one-day swap spikes removed."""
    tickers = {**{f"be{n}": t for n, t in BREAKEVENS.items()},
               **{f"is{n}": f"USSWIT{n} Curncy" for n in SWAP_TENORS}}
    frame = load_wide(tickers, start=start, bps_cols="all").sort("ts")
    frame = frame.with_columns(*[
        pl.Series(f"is{n}", drop_spikes(frame[f"is{n}"].to_numpy(), SPIKE_BP.get(n, SPIKE_BP_DEFAULT))).fill_nan(None)
        for n in SWAP_TENORS])
    return frame.drop_nulls([f"be{n}" for n in BREAKEVENS] + ["is1"])


def _swap_at(frame: pl.DataFrame, tenor: float) -> np.ndarray:
    """The inflation swap curve linearly interpolated at ``tenor`` years, day by day."""
    curve = frame.select([f"is{n}" for n in SWAP_TENORS]).to_numpy()
    xs = np.array(SWAP_TENORS, dtype=float)
    out = np.full(len(frame), np.nan)
    for i, row in enumerate(curve):
        ok = np.isfinite(row)
        if ok.sum() >= 2:
            out[i] = np.interp(tenor, xs[ok], row[ok])
    return out


def carry(frame: pl.DataFrame, horizon: str = "3m") -> pl.DataFrame:
    """Per breakeven tenor: accrual, roll-down and total carry over ``horizon``, in breakeven bp.

    accrual = (1y inflation swap - breakeven) x h / N   (N, the tenor, stands in for duration)
    roll    = swap(N) - swap(N - h)                      (the curve's slope over the horizon)
    Also each breakeven's realized vol over the horizon (daily changes over
    the last 63 days, scaled to h) and carry per unit of that vol.
    """
    h = HORIZONS[horizon]
    out = frame.select("ts")
    for n in BREAKEVENS:
        be = frame[f"be{n}"]
        accrual = (frame["is1"] - be) * h / n
        roll = pl.Series(_swap_at(frame, n) - _swap_at(frame, n - h)).fill_nan(None)
        vol = be.diff().rolling_std(63) * np.sqrt(h * 252)
        total = accrual + roll
        out = out.with_columns(be.alias(f"be{n}"), accrual.alias(f"accrual{n}"), roll.alias(f"roll{n}"),
                               total.alias(f"carry{n}"), vol.alias(f"vol{n}"), (total / vol).alias(f"carry_vol{n}"))
    return out.with_columns(frame["is1"].alias("is1"))


def carry_table(series: pl.DataFrame) -> pl.DataFrame:
    """Today's carry, one row per breakeven tenor."""
    now = series.drop_nulls("carry10").row(-1, named=True)
    return pl.DataFrame([{
        "breakeven": f"{n}y", "level_bp": now[f"be{n}"], "1y_infl_swap_bp": now["is1"],
        "accrual_bp": now[f"accrual{n}"], "roll_bp": now[f"roll{n}"], "carry_bp": now[f"carry{n}"],
        "vol_bp": now[f"vol{n}"], "carry_per_vol": now[f"carry_vol{n}"],
    } for n in BREAKEVENS])
