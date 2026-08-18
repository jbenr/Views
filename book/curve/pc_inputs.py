"""Which panel should the level factor be built from?

Every pc1_* module in this book hedges its curve against a PC1 fitted on
["2y", "5y", "10y", "30y"]. For 10s30s that panel contains both legs of the
target, so the explanatory variable is built partly out of the thing being
explained.

The leak is exact, not vague. Writing the two legs as their mean m and spread
s = long - short, the score's contribution from them is

    v_short*short + v_long*long = (v_short + v_long)*m + (v_long - v_short)/2 * s

so PC1 carries the target curve with coefficient (v_long - v_short)/2. Fit a
regression of s on that, and part of s explains itself: the residual shrinks
and looks more mean-reverting than it is. The term vanishes only when the two
legs load equally -- true by construction for an equal-weighted level, never
exactly true for a fitted PC1.

The fix is not obvious, though, which is why this compares rather than
asserts. Dropping the target's legs removes the leak but leaves a noisier
proxy for the level, and measurement error in x attenuates beta, which leaves
MORE direction in the residual. Those are opposite failures and only data says
which dominates.

Specs compared per target, all at the same lookback:
    full      2y/5y/10y/30y            the incumbent
    ex        full minus the target's own legs
    front     2y/5y/7y                 front-and-belly instrument, never
                                       contains a long-end target leg
    equal     equal-weighted full      zero leak by construction

    python -m book.curve.pc_inputs              # live DB
    python -m book.curve.pc_inputs --lookback 504
"""

from __future__ import annotations

import argparse
import sys

import polars as pl

import utils
from stats import PCSpec, horizon_backtest, roll_lr_diff
from utils.market_data import align_columns, load_wide

# -- config -----------------------------------------------------------------

START = "2010-01-01"

TICKERS = {
    "2y": "USGG2YR Index",
    "5y": "USGG5YR Index",
    "7y": "USGG7YR Index",
    "10y": "USGG10YR Index",
    "30y": "USGG30YR Index",
    "2s5s": "USYC2Y5Y Index",
    "2s7s": "USYC2Y7Y Index",
    "2s10s": "USYC2Y10 Index",
    "2s30s": "USYC2Y30 Index",
    "5s7s": "USYC5Y7Y Index",
    "5s10s": "USYC5Y10 Index",
    "5s30s": "USYC5Y30 Index",
    "7s10s": "USYC7Y10 Index",
    "7s30s": "USYC7Y30 Index",
    "10s30s": "USYC1030 Index",
}
BPS_COLS = ["2y", "5y", "7y", "10y", "30y"]

FULL_PANEL = ("2y", "5y", "10y", "30y")
FRONT_PANEL = ("2y", "5y", "7y")

# target -> (short leg, long leg)
TARGETS = {
    "2s5s": ("2y", "5y"),
    "2s7s": ("2y", "7y"),
    "2s10s": ("2y", "10y"),
    "2s30s": ("2y", "30y"),
    "5s7s": ("5y", "7y"),
    "5s10s": ("5y", "10y"),    
    "5s30s": ("5y", "30y"),
    "7s10s": ("7y", "10y"),
    "7s30s": ("7y", "30y"),
    "10s30s": ("10y", "30y"),
}

BETA_LB = 120  # hedge-ratio window; held fixed so only the panel varies
HORIZONS = (5, 20, 60)
# spec-independent stand-in for "the level", so dir_corr is comparable across
# panels rather than each spec being scored against its own factor
DIR_PROXY = ["2y", "5y", "10y", "30y"]


# -- helpers ----------------------------------------------------------------


def load_data(start: str = START) -> pl.DataFrame:
    """Full panel: yield levels (scaled to bps) plus the curve generics."""
    return load_wide(TICKERS, start=start, bps_cols=BPS_COLS)


def specs_for(short: str, long: str, lookback: int) -> dict[str, PCSpec]:
    """The four candidate level factors for one target curve."""
    ex_cols = tuple(c for c in FULL_PANEL if c not in (short, long))
    return {
        "full": PCSpec(FULL_PANEL, lookback, label="full"),
        "ex": PCSpec(ex_cols, lookback, label="ex"),
        "front": PCSpec(FRONT_PANEL, lookback, label="front"),
        "equal": PCSpec(FULL_PANEL, lookback, method="equal", label="full"),
    }


def evaluate(
    data: pl.DataFrame, target: str, short: str, long: str, spec: PCSpec
) -> dict:
    """One (target, spec) row: how much target the factor carries, and how
    well the resulting residual actually strips direction.

    dir_corr is the metric that decides this question. The entire job of the
    level-factor hedge is to remove direction from a curve trade, so the test
    is whether daily residual changes still move with the level. Nothing about
    the construction forces it to zero, so unlike IC it can distinguish one
    panel from another. Lower is better.
    """
    framed = align_columns(spec.attach(data), [target, spec.name, *DIR_PROXY])
    reg = roll_lr_diff(framed[spec.name], framed[target], lookback=BETA_LB)
    resid = reg["resid"]

    self_w = spec.self_weight(data, short, long)
    # correlation of the factor with the target curve: the leak's end effect,
    # whether it arrives directly through the loadings or through co-movement
    joint = framed.select(target, spec.name).drop_nulls()
    corr = joint.corr()[spec.name][0] if len(joint) > 2 else None

    # residual vs direction, on changes. roll_lr_diff drops the first row.
    level = framed.select(pl.mean_horizontal(DIR_PROXY).alias("lvl"))["lvl"]
    dir_frame = pl.DataFrame({
        "d_resid": resid, "d_lvl": level.diff().slice(len(level) - len(resid))
    }).drop_nulls()
    dir_corr = (
        abs(dir_frame.select(pl.corr("d_resid", "d_lvl")).item())
        if len(dir_frame) > 2 else None
    )

    horizons = horizon_backtest(resid, horizons=HORIZONS)
    row = {
        "target": target,
        "spec": spec.label if spec.method == "pca" else "equal",
        "panel": "+".join(spec.cols),
        "n": len(framed),
        "self_w": round(float(self_w.abs().mean()), 3) if len(self_w) else 0.0,
        "dir_corr": round(dir_corr, 3) if dir_corr is not None else None,
        "corr_x_y": round(corr, 3) if corr is not None else None,
        "r2": round(float(reg["r2"].mean()), 3),
    }
    for h, ic, sharpe in zip(horizons["h"], horizons["ic"], horizons["sharpe"]):
        row[f"ic{h}"] = round(float(ic), 3)
    return row


def compare(data: pl.DataFrame, lookback: int) -> pl.DataFrame:
    """Every target against every candidate panel."""
    rows = []
    for target, (short, long) in TARGETS.items():
        for spec in specs_for(short, long, lookback).values():
            rows.append(evaluate(data, target, short, long, spec))
    return pl.DataFrame(rows)


def leak_summary(table: pl.DataFrame) -> pl.DataFrame:
    """Per target: how much of itself the incumbent 'full' factor carries."""
    return (
        table.filter(pl.col("spec") == "full")
        .select("target", "self_w", "dir_corr", "corr_x_y", "r2", "ic20")
        .sort("self_w", descending=True)
    )


def best_panel(table: pl.DataFrame) -> pl.DataFrame:
    """Per target, the panel that strips the most direction."""
    return (
        table.sort("dir_corr")
        .group_by("target", maintain_order=True)
        .head(1)
        .select("target", pl.col("spec").alias("best"), "panel", "dir_corr", "self_w")
        .sort("target")
    )


# -- mode -------------------------------------------------------------------


def main(lookback: int = 252) -> dict:
    # data
    data = load_data()
    print(f"data  {len(data)} rows  {data['ts'][0]} -> {data['ts'][-1]}  "
          f"beta_lb={BETA_LB}  pc lookback={lookback}\n")

    # every (target, panel) combination
    table = compare(data, lookback)

    # how badly the incumbent panel leaks, per target
    print("incumbent panel (2y+5y+10y+30y): how much of itself it carries")
    utils.pdf(leak_summary(table))

    # the actual comparison, target by target
    print("\nresidual quality by panel choice:")
    utils.pdf(table.drop("n"))

    # does dropping the target's legs strip more direction, or less
    print("\nchange vs incumbent (negative d_dir = more direction stripped, better):")
    base = table.filter(pl.col("spec") == "full").select(
        "target", pl.col("dir_corr").alias("dir_full"), pl.col("ic20").alias("ic20_full")
    )
    delta = (
        table.filter(pl.col("spec") != "full")
        .join(base, on="target")
        .with_columns(
            (pl.col("dir_corr") - pl.col("dir_full")).round(3).alias("d_dir"),
            (pl.col("ic20") - pl.col("ic20_full")).round(3).alias("d_ic20"),
        )
        .select("target", "spec", "self_w", "dir_corr", "d_dir", "d_ic20")
        .sort("target", "spec")
    )
    utils.pdf(delta)

    # the verdict
    print("\nbest panel per target by direction stripped:")
    utils.pdf(best_panel(table))

    return {"data": data, "table": table, "delta": delta}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback", type=int, default=252,
                        help="rolling PCA window (default 252)")
    args = parser.parse_args()
    state = main(lookback=args.lookback)
