"""A Treasury future's deliverable basket: each bond's basis, and which becomes cheapest
to deliver as yields shift.

For one contract, every deliverable is priced against the future (gross basis, carry,
net basis, implied repo -- utils.basis) and given a yield and DV01 (utils.bonds). The
CTD matrix then asks what delivery looks like if yields move before it: each bond's
forward yield is shifted by a parallel move plus a curve move, the bond is repriced on
the delivery date, and the cheapest is the one with the lowest price / conversion factor.

    shift of bond i = level + curve x (T_i - T_short) / (T_long - T_short)

level is the change in the headline curve's short-leg yield, curve the change in the
curve itself (long minus short), T_i the bond's years to maturity at delivery. Their
joint distribution is a bivariate normal with the vols and correlation of the daily
changes over a lookback, scaled to the business days left to delivery. A bond's CTD
probability is the mass of the scenarios in which it is cheapest.

Not modelled: the end-of-month and timing options, repo specialness (financing is the
front SOFR future), and any yield move that is not level plus this one curve.

    python -m research.basis_basket TY
"""

from __future__ import annotations

import datetime as dt
import sys

import numpy as np
import polars as pl

from research.panel import YIELDS
from utils.basis import DELIVERABLE_ROOTS, basket_basis, load_baskets
from utils.bonds import clean_price, dv01, yield_to_maturity
from utils.helpers import query_db

ROOTS = DELIVERABLE_ROOTS
# the headline curve each basket sits on; the tab lets it be changed
CURVES = {"TU": "2s3s", "FV": "3s5s", "TY": "7s10s", "UXY": "7s10s", "US": "10s30s", "WN": "20s30s"}
CURVE_CHOICES = ["2s3s", "2s5s", "3s5s", "5s7s", "5s10s", "7s10s", "10s20s", "10s30s", "20s30s"]
LOOKBACKS = [63, 126, 252]
FACE = {"TU": 200_000}  # a contract's face in dollars; the rest are 100,000
SIGMAS = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])  # the displayed scenarios
FINE = np.linspace(-4.0, 4.0, 81)  # the scenarios probabilities are summed over, in sigmas


def curve_tenors(curve: str) -> tuple[int, int]:
    """'7s10s' -> (7, 10)."""
    short, long = curve.rstrip("s").split("s")
    return int(short), int(long)


def load_basket(root: str) -> pl.DataFrame:
    """The contract's deliverables on the latest day the whole basket is priced."""
    start = (dt.date.today() - dt.timedelta(days=21)).isoformat()
    priced = basket_basis([root], start=start).drop_nulls(["implied_repo", "net_basis"])
    counts = priced.group_by("ts").len().sort("ts")
    asof = counts.filter(pl.col("len") == counts["len"].max())["ts"][-1]  # today's bond closes can be partial
    return priced.filter(pl.col("ts") == asof).sort("maturity")


def add_risk(basket: pl.DataFrame) -> pl.DataFrame:
    """Spot yield and DV01, and the forward yield the delivery-date scenarios start from."""
    rows = []
    for b in basket.iter_rows(named=True):
        yld = yield_to_maturity(b["coupon"], b["maturity"], b["ts"], b["clean"])
        rows.append(dict(
            cusip=b["cusip"], yld=yld, dv01=dv01(b["coupon"], b["maturity"], b["ts"], yld),
            fwd_yld=yield_to_maturity(b["coupon"], b["maturity"], b["delivery"], b["forward_clean"]),
            years=(b["maturity"] - b["delivery"]).days / 365.25))
    return basket.join(pl.DataFrame(rows), on="cusip")


def curve_moves(curve: str, lookback: int, asof: dt.date, days: int) -> dict:
    """Vol of the level and curve factors in bp over `days` business days, and their correlation."""
    short, long = curve_tenors(curve)
    tickers = [YIELDS[f"{short}y"], YIELDS[f"{long}y"]]
    raw = pl.from_pandas(query_db(
        "SELECT ts, ticker, px_last::float * 100 AS bp FROM md.index_eod WHERE ticker = ANY(%s) AND ts <= %s "
        "AND ts >= %s ORDER BY ts", params=[tickers, asof, asof - dt.timedelta(days=lookback * 2)]))
    wide = raw.pivot(index="ts", on="ticker", values="bp").drop_nulls().sort("ts").tail(lookback + 1)
    level = np.diff(wide[tickers[0]].to_numpy())
    slope = np.diff(wide[tickers[1]].to_numpy() - wide[tickers[0]].to_numpy())
    return dict(level=float(level.std(ddof=1) * np.sqrt(days)), curve=float(slope.std(ddof=1) * np.sqrt(days)),
                corr=float(np.corrcoef(level, slope)[0, 1]), observations=len(level))


def converted_prices(basket: pl.DataFrame, curve: str, level: np.ndarray, slope: np.ndarray) -> np.ndarray:
    """Delivery-date price / conversion factor per bond, for each (curve, level) scenario in bp:
    shape (bonds, len(slope), len(level))."""
    short, long = curve_tenors(curve)
    out = []
    for b in basket.iter_rows(named=True):
        shift = level[None, :] + slope[:, None] * (b["years"] - short) / (long - short)
        out.append(clean_price(b["coupon"], b["maturity"], b["delivery"], b["fwd_yld"] + shift / 100.0) / b["cf"])
    return np.array(out)


def scenario_weights(moves: dict) -> np.ndarray:
    """Probability of each FINE x FINE scenario, rows curve and columns level, summing to 1."""
    z_curve, z_level = np.meshgrid(FINE, FINE, indexing="ij")
    rho = moves["corr"]
    density = np.exp(-(z_level ** 2 - 2 * rho * z_level * z_curve + z_curve ** 2) / (2 * (1 - rho ** 2)))
    return density / density.sum()


def study(root: str, curve: str | None = None, lookback: int = 252) -> dict:
    curve = curve or CURVES[root]
    basket = add_risk(load_basket(root))
    first = basket.row(0, named=True)
    days = max(int(np.busday_count(first["ts"], first["delivery"])), 1)
    moves = curve_moves(curve, lookback, first["ts"], days)

    # CTD probabilities: mass of the fine scenarios each bond is cheapest in
    weights = scenario_weights(moves)
    fine = converted_prices(basket, curve, FINE * moves["level"], FINE * moves["curve"])
    cheapest = fine.argmin(axis=0)
    prob = np.array([weights[cheapest == i].sum() for i in range(len(basket))])
    expected_future = float((fine.min(axis=0) * weights).sum())

    # the displayed matrix: each cell's CTD, and the mass of the fine scenarios nearest it
    level, slope = np.round(SIGMAS * moves["level"]), np.round(SIGMAS * moves["curve"])
    shown = converted_prices(basket, curve, level, slope).argmin(axis=0)
    nearest = np.abs(FINE[:, None] - SIGMAS[None, :]).argmin(axis=1)
    mass = np.zeros((len(SIGMAS), len(SIGMAS)))
    np.add.at(mass, (nearest[:, None], nearest[None, :]), weights)

    scale = FACE.get(root, 100_000) / 100.0  # price points per 100 face -> dollars per contract
    bonds = basket.with_columns(
        pl.Series("p_ctd", prob),
        (pl.col("gross_basis") * 32).alias("gross_32"), (pl.col("net_basis") * 32).alias("net_32"),
        ((pl.col("gross_basis") - pl.col("net_basis")) * 32).alias("carry_32"),
        ((pl.col("implied_repo") - pl.col("financing")) * 100).alias("irr_spread_bp"),
        (pl.col("dv01") * scale).alias("dv01_usd"), (pl.col("dv01") / pl.col("cf") * scale).alias("dv01_cf_usd"),
    ).sort("implied_repo", descending=True)
    ctd = bonds.row(0, named=True)
    return dict(
        root=root, curve=curve, lookback=lookback, bonds=bonds, ctd=ctd, moves=moves, days=days,
        asof=first["ts"], contract=first["contract"], delivery=first["delivery"], fut_px=first["fut_px"],
        financing=first["financing"], expected_future=expected_future,
        basket_size=load_baskets([root]).filter(pl.col("contract") == first["contract"]).height,
        blend_dv01=float((bonds["p_ctd"] * bonds["dv01_cf_usd"]).sum()),
        matrix=dict(level=level, curve=slope, ctd=shown, mass=mass, cusips=basket["cusip"].to_list()),
    )


def main(root: str = "TY") -> dict:
    # the basket, its basis measures and CTD probabilities
    state = study(root)
    bonds = state["bonds"]

    # what the tab shows, as text
    print(f"{state['contract']}  as of {state['asof']}  future {state['fut_px']:.4f}  delivery {state['delivery']}  "
          f"{state['days']} business days  financing {state['financing']:.3f}%")
    print(bonds.select("cusip", "coupon", "maturity", "cf", "clean", "yld", "gross_32", "carry_32", "net_32",
                       "implied_repo", "dv01_cf_usd", "p_ctd"))
    moves = state["moves"]
    print(f"{state['curve']}: level vol {moves['level']:.1f}bp, curve vol {moves['curve']:.1f}bp, "
          f"corr {moves['corr']:+.2f}  |  CTD DV01/CF ${state['ctd']['dv01_cf_usd']:.2f}  "
          f"probability-weighted ${state['blend_dv01']:.2f}")
    return state


if __name__ == "__main__":
    main(*sys.argv[1:2])
