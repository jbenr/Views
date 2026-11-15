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

Model option value. Holding a bond's basis to delivery pays its price less the future
times its factor, and the future settles at the cheapest converted price, so the payoff
in a scenario is factor x (the bond's converted price - the cheapest one): zero if the
bond is delivered, positive otherwise. Its probability-weighted average is what the
delivery option is worth in this model, in the same 32nds as the market's net basis.
Net minus model is the rich/cheap: positive means the market charges more for the
option than these scenarios say it is worth.

Bonds not issued yet. A basket keeps growing until delivery: each new 2-, 5-, 7-, 10-,
20- or 30-year auctioned before the last delivery day joins it. Those bonds are added
here as when-issued rows (utils.basis.upcoming_deliverables) so they compete in the
matrix. One has no price, so its yield is read off the basket -- the nearest priced
bond's yield plus the basket's yield slope times the maturity gap -- its coupon, until
Treasury sets it, is that yield rounded down to an eighth, and its conversion factor
is the CME formula at that coupon. The coupon is held fixed across the scenarios.

Not modelled: the end-of-month and timing options, repo specialness (financing is the
front SOFR future), and any yield move that is not level plus this one curve.

    python -m research.basis_basket TY
"""

from __future__ import annotations

import datetime as dt
import functools
import sys

import numpy as np
import polars as pl

from research.panel import YIELDS
from utils.basis import (DELIVERABLE_ROOTS, basket_basis, conversion_factor, load_baskets, load_new_issues,
                         net_basis, upcoming_deliverables)
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


def read_off(basket: pl.DataFrame, column: str, years: float) -> float:
    """A yield for a maturity the basket does not hold: the nearest bond's, moved along the
    basket's own yield slope (flat when it has fewer than three bonds to fit one)."""
    x, y = basket["years"].to_numpy(), basket[column].to_numpy()
    slope = float(np.polyfit(x, y, 1)[0]) if len(x) >= 3 else 0.0
    nearest = int(np.abs(x - years).argmin())
    return float(y[nearest] + slope * (years - x[nearest]))


def load_contracts(root: str) -> dict[str, tuple[dt.date, dt.date]]:
    """Each contract's delivery month and last delivery day."""
    raw = query_db("SELECT contract, contract_month::date AS month, last_delivery_date::date AS last "
                   "FROM sec.fut_contracts WHERE generic_ticker = %s AND last_delivery_date IS NOT NULL", params=[root])
    return {c: (m, d) for c, m, d in zip(raw["contract"], raw["month"], raw["last"])}


def when_issued(root: str, basket: pl.DataFrame, issues: pl.DataFrame | None = None,
                contracts: dict | None = None) -> pl.DataFrame:
    """Rows for the deliverables not issued by the basket's date, shaped like `basket`.

    The auction table can lag the market, so the projection starts from the last issue it
    knows about: a bond issued since then is modelled the same way and marked "est.".
    `issues` and `contracts` default to the database; a historical run passes its own,
    with `issues` cut to what had been issued by the basket's date."""
    first = basket.row(0, named=True)
    meta = (contracts or load_contracts(root)).get(first["contract"])
    if meta is None:  # a few contracts never had their delivery dates stored
        return pl.DataFrame()
    month, last = meta
    held = set(basket["maturity"].to_list())
    issues = load_new_issues() if issues is None else issues
    known_through = min(first["ts"], issues["issue_date"].max())
    rows = []
    for bond in upcoming_deliverables(root, month, last, known_through, issues):
        if bond["maturity"] in held:
            continue
        years = (bond["maturity"] - first["delivery"]).days / 365.25
        yld, fwd_yld = read_off(basket, "yld", years), read_off(basket, "fwd_yld", years)
        coupon = bond["coupon"] if bond["coupon"] is not None else max(np.floor(yld * 8) / 8, 0.125)
        cf = conversion_factor(root, coupon, bond["maturity"], month)
        forward = float(clean_price(coupon, bond["maturity"], first["delivery"], fwd_yld))
        rows.append(dict(
            ts=first["ts"], root=root, contract=first["contract"], delivery=first["delivery"],
            fut_px=first["fut_px"], financing=first["financing"],
            cusip=bond["cusip"] or f"WI {bond['term'] // 12}y {bond['issue_date']:%m/%d}",
            coupon=float(coupon), maturity=bond["maturity"], issue_date=bond["issue_date"], cf=cf,
            forward_clean=forward, net_basis=forward - first["fut_px"] * cf, yld=yld, fwd_yld=fwd_yld, years=years,
            dv01=dv01(coupon, bond["maturity"], first["delivery"], fwd_yld),
            wi="est." if bond["issue_date"] <= first["ts"] else "WI" if bond["coupon"] is not None else "WI est."))
    return pl.DataFrame(rows, schema_overrides={"ts": pl.Date, "delivery": pl.Date, "maturity": pl.Date,
                                                "issue_date": pl.Date}) if rows else pl.DataFrame()


@functools.lru_cache(maxsize=32)
def new_issue_history(root: str, start: str = "2010-01-01", asof: dt.date | None = None) -> dict:
    """How often a bond issued after its contract went front became the CTD.
    `asof` only keys the cache to the day."""
    nb = net_basis([root], start=start)
    issued = load_new_issues().select(pl.col("cusip").alias("ctd"), "issue_date")
    days = nb.join(issued, on="ctd", how="left").with_columns(
        pl.col("ts").min().over("contract").alias("front_from"),
        (pl.col("ts") == pl.col("ts").max().over("contract")).alias("last_day"),
    ).with_columns((pl.col("issue_date") > pl.col("front_from")).fill_null(False).alias("new"))
    per = days.group_by("contract").agg(
        pl.col("new").sum().alias("days_new"), (pl.col("new") & pl.col("last_day")).any().alias("new_at_end"))
    return dict(start=start, contracts=per.height, days=days.height, days_new=int(days["new"].sum()),
                ever=int((per["days_new"] > 0).sum()), week=int((per["days_new"] >= 5).sum()),
                at_end=int(per["new_at_end"].sum()))


def load_curve(curve: str, start: dt.date) -> pl.DataFrame:
    """Daily history of the two factors in bp: ts, level (the short leg's yield), curve (long minus short)."""
    short, long = curve_tenors(curve)
    tickers = [YIELDS[f"{short}y"], YIELDS[f"{long}y"]]
    raw = pl.from_pandas(query_db(
        "SELECT ts, ticker, px_last::float * 100 AS bp FROM md.index_eod WHERE ticker = ANY(%s) AND ts >= %s "
        "ORDER BY ts", params=[tickers, start]))
    wide = raw.pivot(index="ts", on="ticker", values="bp").drop_nulls().sort("ts")
    return wide.select(pl.col("ts").cast(pl.Date), pl.col(tickers[0]).alias("level"),
                       (pl.col(tickers[1]) - pl.col(tickers[0])).alias("curve"))


def curve_moves(curve: str, lookback: int, asof: dt.date, days: int, history: pl.DataFrame | None = None) -> dict:
    """Vol of the level and curve factors in bp over `days` business days, and their correlation,
    from the `lookback` daily changes up to `asof`. `history` (load_curve) saves the query."""
    if history is None:
        history = load_curve(curve, asof - dt.timedelta(days=lookback * 2))
    wide = history.filter(pl.col("ts") <= asof).tail(lookback + 1)
    level = np.diff(wide["level"].to_numpy())
    slope = np.diff(wide["curve"].to_numpy())
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


def with_when_issued(root: str, basket: pl.DataFrame, issues: pl.DataFrame | None = None,
                     contracts: dict | None = None) -> pl.DataFrame:
    """The priced basket with spot and forward yields, plus a row per bond not issued yet."""
    basket = add_risk(basket)
    upcoming = when_issued(root, basket, issues, contracts)
    basket = basket.with_columns(pl.lit("").alias("wi"))
    return basket if upcoming.is_empty() else pl.concat([basket, upcoming], how="diagonal")


def delivery_odds(basket: pl.DataFrame, curve: str, moves: dict) -> dict:
    """Over the fine scenarios: each bond's probability of being cheapest, what its basis is
    worth at delivery (price points), and the future's expected settlement."""
    weights = scenario_weights(moves)
    fine = converted_prices(basket, curve, FINE * moves["level"], FINE * moves["curve"])
    cheapest, floor = fine.argmin(axis=0), fine.min(axis=0)
    return dict(
        p_ctd=np.array([weights[cheapest == i].sum() for i in range(len(basket))]),
        option=basket["cf"].to_numpy() * ((fine - floor) * weights).sum(axis=(1, 2)),
        expected_future=float((floor * weights).sum()))


def study(root: str, curve: str | None = None, lookback: int = 252) -> dict:
    curve = curve or CURVES[root]
    basket = with_when_issued(root, load_basket(root))
    first = basket.row(0, named=True)
    days = max(int(np.busday_count(first["ts"], first["delivery"])), 1)
    moves = curve_moves(curve, lookback, first["ts"], days)

    # CTD probabilities and option values over the fine scenarios
    odds = delivery_odds(basket, curve, moves)
    prob, expected_future = odds["p_ctd"], odds["expected_future"]
    weights = scenario_weights(moves)

    # the displayed matrix: each cell's CTD, and the mass of the fine scenarios nearest it
    level, slope = np.round(SIGMAS * moves["level"]), np.round(SIGMAS * moves["curve"])
    shown = converted_prices(basket, curve, level, slope).argmin(axis=0)
    nearest = np.abs(FINE[:, None] - SIGMAS[None, :]).argmin(axis=1)
    mass = np.zeros((len(SIGMAS), len(SIGMAS)))
    np.add.at(mass, (nearest[:, None], nearest[None, :]), weights)

    scale = FACE.get(root, 100_000) / 100.0  # price points per 100 face -> dollars per contract
    bonds = basket.with_columns(
        pl.Series("p_ctd", prob), pl.Series("model_32", odds["option"] * 32),
        (pl.col("gross_basis") * 32).alias("gross_32"), (pl.col("net_basis") * 32).alias("net_32"),
        ((pl.col("gross_basis") - pl.col("net_basis")) * 32).alias("carry_32"),
        ((pl.col("implied_repo") - pl.col("financing")) * 100).alias("irr_spread_bp"),
        (pl.col("dv01") * scale).alias("dv01_usd"), (pl.col("dv01") / pl.col("cf") * scale).alias("dv01_cf_usd"),
    ).with_columns((pl.col("net_32") - pl.col("model_32")).alias("rich_32")
                   ).sort("implied_repo", descending=True, nulls_last=True)
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
    print(bonds.select("cusip", "wi", "coupon", "maturity", "cf", "clean", "yld", "gross_32", "carry_32", "net_32",
                       "model_32", "rich_32", "implied_repo", "dv01_cf_usd", "p_ctd"))
    h = new_issue_history(root, asof=state["asof"])
    print(f"since {h['start']}: a bond issued after its contract went front was CTD in {h['ever']} of "
          f"{h['contracts']} contracts ({h['week']} for 5+ days, {h['at_end']} on the last day measured), "
          f"{h['days_new']} of {h['days']} days")
    moves = state["moves"]
    print(f"{state['curve']}: level vol {moves['level']:.1f}bp, curve vol {moves['curve']:.1f}bp, "
          f"corr {moves['corr']:+.2f}  |  CTD DV01/CF ${state['ctd']['dv01_cf_usd']:.2f}  "
          f"probability-weighted ${state['blend_dv01']:.2f}")
    return {**state, "history": h}


if __name__ == "__main__":
    main(*sys.argv[1:2])
