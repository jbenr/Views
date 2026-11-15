"""Does the basket model's delivery-option value lead the market's net basis?

Every day since 2010, for each Treasury futures contract, the basket study
(research.basis_basket) is rerun with only what was known that day: the 3pm bond
closes, the yield vols of the lookback before it, the bonds issued by then. That
gives two numbers in 32nds for the CTD: the market's net basis, and the model's
option value (what the basis is worth at delivery, averaged over the scenarios).

Three questions, each within a contract so no roll is crossed:

  level     does net - model (rich/cheap) predict the next h days' change in net
            basis, beyond what net basis alone predicts? Net basis decays to zero
            at delivery whatever a model says, so the test is the model's
            coefficient with net basis already in the regression.
  changes   does a change in the model value come before or after the market's
            change? corr(model change over the last h days, net change over the
            next h) against the same with the two swapped.
  trade     fading the rich/cheap: sell the basis when it is rich to the model,
            buy when cheap, hold h days. P&L in 32nds per unit.

    python -m research.basis_option_history                 # all six roots, cached
    python -m research.basis_option_history --roots TY,US --rebuild
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl
import statsmodels.api as sm

from research.basis_basket import (CURVES, ROOTS, curve_moves, delivery_odds, load_contracts, load_curve,
                                   with_when_issued)
from utils.basis import basket_basis, load_new_issues

START = "2010-01-01"
LOOKBACK = 252
HORIZONS = [5, 10, 20]
CACHE = Path(__file__).resolve().parents[1] / "store" / "basis_option_history.parquet"


def daily_values(root: str, start: str = START, lookback: int = LOOKBACK) -> pl.DataFrame:
    """One row per day for a root: the CTD's market net basis and model option value in 32nds,
    its probability of staying CTD, and the probability a bond not issued yet takes over."""
    curve = CURVES[root]
    priced = basket_basis([root], start=start).drop_nulls(["implied_repo", "net_basis"]).sort("ts", "maturity")
    history = load_curve(curve, dt.date.fromisoformat(start) - dt.timedelta(days=lookback * 2))
    issues, contracts = load_new_issues(), load_contracts(root)

    rows = []
    for (ts,), day in priced.group_by("ts", maintain_order=True):
        basket = with_when_issued(root, day, issues.filter(pl.col("issue_date") <= ts), contracts)
        first = basket.row(0, named=True)
        moves = curve_moves(curve, lookback, ts, max(int(np.busday_count(ts, first["delivery"])), 1), history)
        if moves["observations"] < lookback // 2:
            continue
        odds = delivery_odds(basket, curve, moves)
        ctd = int(basket["implied_repo"].fill_null(-np.inf).arg_max())
        upcoming = (basket["wi"] != "").to_numpy()
        rows.append(dict(
            ts=ts, root=root, contract=first["contract"], ctd=basket["cusip"][ctd], n_days=first["n_days"],
            bonds=int((~upcoming).sum()), net_32=basket["net_basis"][ctd] * 32, model_32=float(odds["option"][ctd]) * 32,
            p_ctd=float(odds["p_ctd"][ctd]), p_wi=float(odds["p_ctd"][upcoming].sum())))
    return pl.DataFrame(rows).with_columns((pl.col("net_32") - pl.col("model_32")).alias("rich_32"))


def with_changes(values: pl.DataFrame, horizon: int) -> pl.DataFrame:
    """Past and next `horizon`-day changes of net basis and model value, never across a contract."""
    by = ["root", "contract"]
    return values.sort("root", "ts").with_columns(
        (pl.col("net_32").shift(-horizon).over(by) - pl.col("net_32")).alias("net_next"),
        (pl.col("net_32") - pl.col("net_32").shift(horizon).over(by)).alias("net_past"),
        (pl.col("model_32").shift(-horizon).over(by) - pl.col("model_32")).alias("model_next"),
        (pl.col("model_32") - pl.col("model_32").shift(horizon).over(by)).alias("model_past"),
    )


def level_test(values: pl.DataFrame, horizon: int) -> dict:
    """Next-h change in net basis on net basis and model value, Newey-West errors for the overlap.
    A model that leads has a positive coefficient: the market moves toward it."""
    d = with_changes(values, horizon).drop_nulls(["net_next"])
    y = d["net_next"].to_numpy()
    alone = sm.OLS(y, sm.add_constant(d["net_32"].to_numpy())).fit()
    both = sm.OLS(y, sm.add_constant(d.select("net_32", "model_32").to_numpy())).fit(
        cov_type="HAC", cov_kwds={"maxlags": horizon})
    return dict(n=d.height, b_net=both.params[1], t_net=both.tvalues[1], b_model=both.params[2],
                t_model=both.tvalues[2], r2_net_only=alone.rsquared, r2_with_model=both.rsquared)


def change_test(values: pl.DataFrame, horizon: int) -> dict:
    """Which moves first: correlation of one series' past change with the other's next change."""
    d = with_changes(values, horizon)
    lead = d.drop_nulls(["model_past", "net_next"])
    lag = d.drop_nulls(["net_past", "model_next"])
    same = d.drop_nulls(["model_past", "net_past"])
    return dict(
        model_then_net=float(np.corrcoef(lead["model_past"], lead["net_next"])[0, 1]),
        net_then_model=float(np.corrcoef(lag["net_past"], lag["model_next"])[0, 1]),
        same_window=float(np.corrcoef(same["model_past"], same["net_past"])[0, 1]))


def fade_test(values: pl.DataFrame, horizon: int, band: float = 1.0) -> dict:
    """Sell the basis when net - model is above `band` 32nds, buy when below -band, hold `horizon` days.
    Entries are taken every `horizon` days so holding periods do not overlap."""
    d = with_changes(values, horizon).drop_nulls(["net_next"]).with_columns(
        pl.col("ts").cum_count().over("root", "contract").alias("k")).filter(pl.col("k") % horizon == 0)
    rich, cheap = d.filter(pl.col("rich_32") > band), d.filter(pl.col("rich_32") < -band)
    pnl = np.concatenate([-rich["net_next"].to_numpy(), cheap["net_next"].to_numpy()])
    if len(pnl) < 5:
        return dict(trades=len(pnl), sells=rich.height, buys=cheap.height, mean_32=None, hit=None, t=None)
    return dict(trades=len(pnl), sells=rich.height, buys=cheap.height, mean_32=float(pnl.mean()),
                hit=float((pnl > 0).mean()), t=float(pnl.mean() / pnl.std(ddof=1) * np.sqrt(len(pnl))))


def table(values: pl.DataFrame, test, horizons: list[int]) -> pl.DataFrame:
    """`test` for every root and horizon, one row each."""
    return pl.DataFrame([
        {"root": root, "h": h, **test(values.filter(pl.col("root") == root), h)}
        for root in values["root"].unique(maintain_order=True) for h in horizons])


def main(roots: list[str] | None = None, rebuild: bool = False) -> dict:
    roots = roots or ROOTS

    # the daily history of market net basis and model option value (slow: cached)
    cached = pl.read_parquet(CACHE) if CACHE.exists() and not rebuild else None
    have = set(cached["root"].unique()) if cached is not None else set()
    fresh = [daily_values(root) for root in roots if root not in have]
    values = pl.concat([f for f in [cached, *fresh] if f is not None]).sort("root", "ts")
    if fresh:
        CACHE.parent.mkdir(exist_ok=True)
        values.write_parquet(CACHE)
    values = values.filter(pl.col("root").is_in(roots))

    # what the two series look like
    summary = values.group_by("root", maintain_order=True).agg(
        pl.len().alias("days"), pl.col("contract").n_unique().alias("contracts"),
        pl.col("net_32").median().alias("net_median"), pl.col("model_32").median().alias("model_median"),
        pl.col("rich_32").median().alias("rich_median"), (pl.col("rich_32") > 0).mean().alias("share_rich"),
        pl.corr("net_32", "model_32").alias("corr_levels"), pl.col("p_ctd").mean().alias("p_ctd_mean"))

    # does the model lead the market?
    levels = table(values, level_test, HORIZONS)
    changes = table(values, change_test, HORIZONS)
    fades = table(values, fade_test, HORIZONS)

    with pl.Config(tbl_rows=40, tbl_cols=20, tbl_width_chars=220, float_precision=3):
        print("net basis and model option value, 32nds\n", summary)
        print("next-h change in net basis on net basis and model value (Newey-West t)\n", levels)
        print("which moves first: correlation of h-day changes\n", changes)
        print("fading net - model beyond 1/32, non-overlapping holds\n", fades)
    return {"values": values, "summary": summary, "levels": levels, "changes": changes, "fades": fades}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--roots", type=lambda s: s.split(","), default=None)
    parser.add_argument("--rebuild", action="store_true", help="recompute the daily history instead of using the cache")
    args = parser.parse_args()
    main(args.roots, args.rebuild)
