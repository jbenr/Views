"""Declared-factor and controlled-search fair-value research for 10s30s.

This is the slower, economic value path.  The initial declared model asks
whether 10s30s is rich or cheap versus three intentionally distinct factor
families: real-rate level, long-run inflation expectations, and rates
volatility.  It then tests whether the residual error-corrects.

The search is constrained to at most one factor per family.  It is a way to
explore economically related alternatives, not permission to brute-force all
available columns until one backtest looks attractive.

    python -m book.curve.fair_value_tens_10s30s
    python -m book.curve.fair_value_tens_10s30s --synthetic
"""

from __future__ import annotations

import sys

import polars as pl

import utils
from research import FairValueStudy

from .tens_10s30s_research_data import coverage, load_data, synthetic_data

NAME = "fair_value_tens_10s30s"
TARGET = "10s30s"
DECLARED_FACTORS = ("real10y", "5y5y_infl", "move")
LOOKBACK = 252
FACTOR_FAMILIES = {
    "real_rates": ("real10y",),
    "inflation": ("be10", "5y5y_infl"),
    "volatility": ("move",),
}

STUDY = FairValueStudy(TARGET, DECLARED_FACTORS, lookback=LOOKBACK)


def run(data: pl.DataFrame) -> dict:
    """Run declared-factor value, then retain every constrained search model."""
    state = STUDY.research(data)
    search = FairValueStudy.search(
        data,
        target=TARGET,
        factor_families=FACTOR_FAMILIES,
        lookback=LOOKBACK,
        max_factors=3,
    )
    return {
        "coverage": coverage(data, [TARGET, *DECLARED_FACTORS]),
        "factor_search": search,
        **state,
    }


def main(use_db: bool = True) -> dict:
    data = load_data() if use_db else synthetic_data()
    state = run(data)
    print(f"{NAME}  rows={len(data)}  {data['ts'][0]} -> {data['ts'][-1]}")
    print("\ncoverage of the declared model:")
    utils.pdf(state["coverage"])
    print("\ndeclared-model residual / error-correction diagnostics:")
    utils.pdf(state["diagnostics"])
    print("\ncorrection evidence by horizon:")
    utils.pdf(state["horizons"])
    print("\ncontrolled factor-family exploration (ranked by 20d IC):")
    utils.pdf(state["factor_search"].head(15))
    print("\nlatest declared fair-value state:")
    utils.pdf(
        state["signals"].tail(1).select(
            "ts", TARGET, "fair_value", "residual", "error_correction", "r2",
            "factor_condition_number",
            "raw_normal_eq_condition_number",
            *[f"beta_{factor}" for factor in DECLARED_FACTORS],
        )
    )
    return {"data": data, **state}


if __name__ == "__main__":
    args = set(sys.argv[1:])
    unknown = args - {"--synthetic"}
    if unknown:
        sys.exit(f"unknown argument(s): {sorted(unknown)}\nflags: --synthetic")
    main(use_db="--synthetic" not in args)
