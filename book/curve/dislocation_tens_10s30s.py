"""10s30s conditional-dislocation research versus 10Y direction.

Question: did 10s30s move too far -- or not far enough -- today given the
10Y move?  A rolling changes regression defines the dislocation in bps.  The
study then tests correction horizons, entry magnitudes, and the optional OU
description of expected correction and half-life.

This is research only.  It does not choose a winner or replace the incumbent
``tens_10s30s`` strategy/backtest.

    python -m book.curve.dislocation_tens_10s30s
    python -m book.curve.dislocation_tens_10s30s --synthetic
"""

from __future__ import annotations

import sys

import polars as pl

import utils
from research import DislocationStudy

from .tens_10s30s_research_data import coverage, load_data, synthetic_data

NAME = "dislocation_tens_10s30s"
TARGET = "10s30s"
FEATURES = ("10y",)
BETA_LOOKBACK = 126
NORMALIZATION_LOOKBACK = 63
OU_LOOKBACK = 126
HORIZONS = (1, 5, 10, 20)
RAW_THRESHOLDS_BPS = (5.0, 10.0, 15.0, 20.0)
OU_THRESHOLDS = (1.0, 1.5, 2.0, 2.5)

STUDY = DislocationStudy(
    target=TARGET,
    features=FEATURES,
    beta_lookback=BETA_LOOKBACK,
    normalization_lookback=NORMALIZATION_LOOKBACK,
    ou_lookback=OU_LOOKBACK,
)


def run(data: pl.DataFrame, metric: str = "dislocation") -> dict:
    """Run the fixed first-pass conditional-dislocation evidence table."""
    thresholds = OU_THRESHOLDS if metric == "dislocation_ou_z" else RAW_THRESHOLDS_BPS
    state = STUDY.research(
        data, horizons=HORIZONS, thresholds=thresholds, metric=metric
    )
    return {"coverage": coverage(data, [TARGET, *FEATURES]), **state}


def main(use_db: bool = True, metric: str = "dislocation") -> dict:
    """Print the research evidence and latest dislocation state."""
    data = load_data() if use_db else synthetic_data()
    state = run(data, metric=metric)
    print(f"{NAME}  metric={metric}  rows={len(data)}  {data['ts'][0]} -> {data['ts'][-1]}")
    print("\ncoverage:")
    utils.pdf(state["coverage"])
    print("\ncontinuous correction evidence:")
    utils.pdf(state["horizons"])
    print("\nfirst threshold-crossing evidence:")
    utils.pdf(state["events"])
    latest = state["signals"].tail(1).select(
        "ts", "10s30s", "10y", "target_move", "predicted_move", "dislocation",
        "dislocation_score", "dislocation_ou_z", "dislocation_expected_delta_1d",
        "dislocation_half_life", "beta_10y", "r2",
    )
    print("\nlatest conditional-dislocation state:")
    utils.pdf(latest)
    return {"data": data, **state}


if __name__ == "__main__":
    args = set(sys.argv[1:])
    known = {"--synthetic", "--ou"}
    unknown = args - known
    if unknown:
        sys.exit(f"unknown argument(s): {sorted(unknown)}\nflags: --synthetic --ou")
    main(use_db="--synthetic" not in args, metric="dislocation_ou_z" if "--ou" in args else "dislocation")
