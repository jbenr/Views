"""Hedge-beta contest: rolling changes OLS vs a daily-refit Kalman beta.

For each curve pair (a spread target hedged with its short leg) and each hedge
lookback, three changes hedge ratios are built with no lookahead: rolling OLS
(`roll_lr_diff`, what the oven uses), a Kalman beta with a fitted AR(1) pull to
the window's OLS beta, and a random-walk Kalman beta (phi = 1). Each is scored
on the hedge it would actually have held: yesterday's beta on today's moves.
Lower out-of-sample hedged variance wins; beta tails and turnover show whether
a method tames extreme hedge ratios and what it costs to rehedge.
"""

import time

import numpy as np
import polars as pl

from stats import kalman_beta, roll_lr_diff
from utils.market_data import load_wide

TICKERS = {"2y": "USGG2YR Index", "5y": "USGG5YR Index", "10y": "USGG10YR Index", "30y": "USGG30YR Index"}
START = "2000-01-01"
LOOKBACKS = [63, 126, 252]
# target name -> (target weights on the yield columns, hedge input)
PAIRS = {
    "2s10s vs 2y": ({"10y": 1, "2y": -1}, "2y"),
    "5s30s vs 5y": ({"30y": 1, "5y": -1}, "5y"),
    "10s30s vs 10y": ({"30y": 1, "10y": -1}, "10y"),
    "2s10s30s vs 10y": ({"2y": -1, "10y": 2, "30y": -1}, "10y"),
}


def load_yields() -> pl.DataFrame:
    return load_wide(TICKERS, start=START, bps_cols="all").drop_nulls()


def pair_series(data: pl.DataFrame, weights: dict, hedge: str) -> tuple[np.ndarray, np.ndarray]:
    y = sum(w * data[c].to_numpy().astype(float) for c, w in weights.items())
    return data[hedge].to_numpy().astype(float), y


def hedge_betas(x: np.ndarray, y: np.ndarray, lookback: int) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Each method's beta, aligned to changes (row i is the change into bar i+1), and its build time."""
    betas, secs = {}, {}
    for name, fn in (("ols", lambda: roll_lr_diff(pl.Series(x), pl.Series(y), lookback=lookback)),
                     ("kalman", lambda: kalman_beta(x, y, lookback)),
                     ("kalman_rw", lambda: kalman_beta(x, y, lookback, phi=1.0))):
        t0 = time.perf_counter()
        betas[name] = fn()["beta"].to_numpy().astype(float)
        secs[name] = time.perf_counter() - t0
    return betas, secs


def score(x: np.ndarray, y: np.ndarray, beta: np.ndarray) -> dict:
    """Score a beta on the hedge it held: yesterday's beta against today's moves."""
    dx, dy = np.diff(x), np.diff(y)
    hedged = dy[1:] - beta[:-1] * dx[1:]
    keep = np.isfinite(hedged)
    b = beta[np.isfinite(beta)]
    return {
        "hedged_var": float(np.var(hedged[keep])),
        "hedged_mad": float(np.median(np.abs(hedged[keep] - np.median(hedged[keep])))),
        "beta_mean": float(b.mean()),
        "beta_p01": float(np.percentile(b, 1)),
        "beta_p99": float(np.percentile(b, 99)),
        "beta_std": float(b.std()),
        "turnover": float(np.nanmean(np.abs(np.diff(beta)))),
    }


def contest(data: pl.DataFrame) -> pl.DataFrame:
    """One row per pair x lookback x method, on the common sample where all three have a beta."""
    rows = []
    for pair, (weights, hedge) in PAIRS.items():
        x, y = pair_series(data, weights, hedge)
        for lb in LOOKBACKS:
            betas, secs = hedge_betas(x, y, lb)
            common = np.all([np.isfinite(b) for b in betas.values()], axis=0)
            for method, beta in betas.items():
                rows.append({"pair": pair, "lookback": lb, "method": method, "secs": secs[method],
                             **score(x, y, np.where(common, beta, np.nan))})
    out = pl.DataFrame(rows)
    base = out.filter(pl.col("method") == "ols").select("pair", "lookback", pl.col("hedged_var").alias("ols_var"))
    return (out.join(base, on=["pair", "lookback"])
            .with_columns((pl.col("hedged_var") / pl.col("ols_var") - 1).alias("var_vs_ols")).drop("ols_var"))


def main() -> dict:
    # generic yields in bps, common sample
    data = load_yields()

    # warm the numba kernel so the timings are the steady-state cost
    kalman_beta(data["10y"].head(300), data["30y"].head(300), 63)

    # every pair x lookback x method, scored on yesterday's hedge
    results = contest(data)

    with pl.Config(tbl_rows=100, tbl_cols=20, float_precision=3, tbl_width_chars=220):
        print(f"{len(data)} bars, {data['ts'].min()} to {data['ts'].max()}")
        print(results)
    return {"data": data, "results": results}


if __name__ == "__main__":
    main()
