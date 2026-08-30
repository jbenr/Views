"""10s30s versus 10Y beta-weighted relative-value research.

Question: does the 10s30s curve, hedged against its directional 10Y beta,
form a coherent convergence package?  This is a stronger claim than the
conditional-dislocation study: the object under examination is the weighted
10s30s-versus-10Y package itself.

The initial hedge is a rolling changes beta.  Before this becomes a trade,
research must establish hedge stability, residual stationarity/half-life,
remaining direction exposure, convergence across rates regimes, and carry/
roll for the final executable legs.

    python -m book.curve.rv_tens_10s30s
"""

from __future__ import annotations

import sys

import polars as pl

import utils
from research import PairRVStudy
from stats import beta_cv

from .tens_10s30s_research_data import coverage, load_data

NAME = "rv_tens_10s30s"
LEFT = "10s30s"
RIGHT = "10y"
BETA_LOOKBACK = 126
RESIDUAL_LOOKBACK = 126

STUDY = PairRVStudy(
    left=LEFT,
    right=RIGHT,
    weighting="beta",
    beta_lookback=BETA_LOOKBACK,
    z_lookback=RESIDUAL_LOOKBACK,
)


def _hedge_report(signals: pl.DataFrame) -> pl.DataFrame:
    """Report beta stability and the package's remaining directional link."""
    frame = signals.with_columns(
        beta_cv(signals["hedge_weight"], lookback=BETA_LOOKBACK).alias("hedge_beta_cv"),
        pl.col("rv_value").diff().alias("rv_change"),
        pl.col(RIGHT).diff().alias("ten_change"),
    ).drop_nulls(subset=["rv_change", "ten_change", "hedge_beta_cv"])
    if frame.is_empty():
        return pl.DataFrame()
    return frame.select(
        pl.len().alias("n_obs"),
        pl.col("hedge_weight").mean().alias("mean_hedge_beta"),
        pl.col("hedge_weight").std().alias("hedge_beta_std"),
        pl.col("hedge_beta_cv").mean().alias("mean_hedge_beta_cv"),
        pl.corr("rv_change", "ten_change").alias("remaining_direction_corr"),
    )


def run(data: pl.DataFrame) -> dict:
    """Run the initial beta-weighted 10s30s/10Y RV research pass."""
    state = STUDY.research(data)
    return {
        "coverage": coverage(data, [LEFT, RIGHT]),
        "hedge": _hedge_report(state["signals"]),
        **state,
    }


def main() -> dict:
    data = load_data()
    state = run(data)
    print(f"{NAME}  rows={len(data)}  {data['ts'][0]} -> {data['ts'][-1]}")
    print("\ncoverage:")
    utils.pdf(state["coverage"])
    print("\nresidual stationarity diagnostics:")
    utils.pdf(state["diagnostics"])
    print("\nhedge quality and remaining 10Y direction:")
    utils.pdf(state["hedge"])
    print("\nconvergence evidence by horizon:")
    utils.pdf(state["horizons"])
    print("\nlatest RV state:")
    utils.pdf(state["signals"].tail(1).select("ts", LEFT, RIGHT, "hedge_weight", "rv_value", "signal"))
    return {"data": data, **state}


if __name__ == "__main__":
    args = set(sys.argv[1:])
    if args:
        sys.exit(f"unknown argument(s): {sorted(args)}")
    main()
