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
switch option is worth in this model, in the same 32nds as the market's net basis.
The timing options are added to it, and net minus that total is the rich/cheap:
positive means the market charges more for the options than the model says they are
worth.

The options, named as in Burghardt's Treasury Bond Basis (book/basis/RedBook.pdf, ch. 3).
"Switch" and "quality" are two names for one option: any change of CTD while the future
still trades. It has three drivers, split here by adding them one at a time:

    yield levels    parallel moves, the curve moving only as its beta to the level says
    yield spreads   curve moves beyond that beta
    new issues      bonds not issued yet joining the basket

The timing options are the short's choice of delivery day:

    carry       deliver on the first day, not the last, when carry is negative by more
                than the switch value the month still holds. The future is then worth the
                lower of its last-day and first-day values; the option is the difference.
    wild card   the future settles at 2pm Chicago, delivery notices close hours later.
                A basis position is short CF futures per bond, so delivering leaves a tail
                of (1 - CF) bonds to trade at the evening's price against a fixed invoice:
                a gain of (1 - CF) x the price move, at the cost of the basis given up
                on the CF futures (Red Book p.72). One such option an evening, from two
                business days before the delivery month to the eve of the last trading day.

Both are worked out for the CTD on today's forwards (no distribution on repo or on
which bond is CTD by then) and passed to every bond through the futures price.

Bonds not issued yet. A basket keeps growing until delivery: each new 2-, 5-, 7-, 10-,
20- or 30-year auctioned before the last delivery day joins it. Those bonds are added
here as when-issued rows (utils.basis.upcoming_deliverables) so they compete in the
matrix. One has no price, so its yield is read off the basket -- the nearest priced
bond's yield plus the basket's yield slope times the maturity gap -- its coupon, until
Treasury sets it, is that yield rounded down to an eighth, and its conversion factor
is the CME formula at that coupon. The coupon is held fixed across the scenarios.

Contracts outside the US (utils.basis.GLOBAL_ROOTS: Schatz, Bobl, Bund, Buxl, OAT, BTP,
short BTP, long gilt, Canadian 2-, 5- and 10-year, JGB) run through the same study on
their own bonds, coupon frequency, settlement lag and overnight rate, with the curve
factors taken from that country's benchmark yields. Their basis is quoted in 100ths of
a price point, not 32nds, and DV01 is in the contract's currency. For them the model
value is the switch option alone: bonds not issued yet are not added to the basket,
and the timing options are not valued, which matters for the gilt and the Canadian
contracts when carry is negative (they deliver on any day of the month).

Not modelled: the end-of-month option as its own thing (the switch scenarios run to the
last delivery day as if the future traded until then), limit moves, repo specialness
(financing is the front SOFR future), and any yield move that is not level plus this
one curve.

    python -m research.basis_basket TY
"""

from __future__ import annotations

import datetime as dt
import functools
import sys

import numpy as np
import polars as pl
from scipy.stats import norm

from research.panel import YIELDS
from utils.basis import (DELIVERABLE_ROOTS, GLOBAL_ROOTS, PRICE_BOUNDS, basket_basis, conversion_factor, load_baskets, load_new_issues,
                         global_basket_basis, net_basis, upcoming_deliverables)
from utils.bonds import accrued, clean_price, dv01, yield_to_maturity
from utils.helpers import query_db

ROOTS = [*DELIVERABLE_ROOTS, *GLOBAL_ROOTS]
# the headline curve each basket sits on; the tab lets it be changed
CURVES = {"TU": "2s3s", "FV": "3s5s", "TY": "7s10s", "UXY": "7s10s", "US": "10s30s", "WN": "20s30s",
          "DU": "2s3s", "OE": "3s5s", "RX": "7s10s", "UB": "20s30s", "OAT": "7s10s", "IK": "7s10s", "BTS": "2s3s",
          "G ": "10s20s", "CV": "2s3s", "XQ": "3s5s", "CN": "7s10s", "JB": "7s10s"}
# the benchmark yield behind a tenor outside the US; Treasuries use research.panel.YIELDS
GENERICS = {**dict.fromkeys(["DU", "OE", "RX", "UB"], "GDBR{} Index"), "OAT": "GFRN{} Index",
            "IK": "GBTPGR{} Index", "BTS": "GBTPGR{} Index", "G ": "GUKG{} Index",
            **dict.fromkeys(["CV", "XQ", "CN"], "GCAN{}YR Index"), "JB": "GJGB{} Index"}
CURVE_CHOICES = ["2s3s", "2s5s", "3s5s", "5s7s", "5s10s", "7s10s", "10s20s", "10s30s", "20s30s"]
LOOKBACKS = [63, 126, 252]
FACE = {"TU": 200_000, "JB": 100_000_000}  # a contract's face in its currency; the rest are 100,000
SIGMAS = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])  # the displayed scenarios
FINE = np.linspace(-4.0, 4.0, 81)  # the scenarios probabilities are summed over, in sigmas
NOTICE_HOURS = 4.0    # futures settle at 2pm Chicago, delivery notices close at 6pm: the wild card's evening
MEASURED_HOURS = 2.0  # the after-close move that is stored: 3pm to 5pm New York bond closes


def curve_tenors(curve: str) -> tuple[int, int]:
    """'7s10s' -> (7, 10)."""
    short, long = curve.rstrip("s").split("s")
    return int(short), int(long)


def load_basket(root: str) -> pl.DataFrame:
    """The contract's deliverables on the latest day the whole basket is priced."""
    start = (dt.date.today() - dt.timedelta(days=21)).isoformat()
    pricer = global_basket_basis if root in GLOBAL_ROOTS else basket_basis
    priced = pricer([root], start=start).drop_nulls(["implied_repo", "net_basis"])
    counts = priced.group_by("ts").len().sort("ts")
    asof = counts.filter(pl.col("len") == counts["len"].max())["ts"][-1]  # today's bond closes can be partial
    return priced.filter(pl.col("ts") == asof).sort("maturity")


def add_risk(basket: pl.DataFrame) -> pl.DataFrame:
    """Spot yield and DV01, and the forward yield the delivery-date scenarios start from."""
    rows = []
    for b in basket.iter_rows(named=True):
        settle, freq = b.get("settle") or b["ts"], b.get("freq") or 2  # a Treasury is priced on its close date
        yld = yield_to_maturity(b["coupon"], b["maturity"], settle, b["clean"], freq)
        rows.append(dict(
            cusip=b["cusip"], yld=yld, dv01=dv01(b["coupon"], b["maturity"], settle, yld, freq),
            fwd_yld=yield_to_maturity(b["coupon"], b["maturity"], b["delivery"], b["forward_clean"], freq),
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


def load_curve(curve: str, start: dt.date, root: str | None = None) -> pl.DataFrame:
    """Daily history of the two factors in bp: ts, level (the short leg's yield), curve (long minus short).
    `root` picks the country's benchmark yields; without one they are Treasuries."""
    short, long = curve_tenors(curve)
    tickers = ([GENERICS[root].format(short), GENERICS[root].format(long)] if root in GENERICS
               else [YIELDS[f"{short}y"], YIELDS[f"{long}y"]])
    raw = pl.from_pandas(query_db(
        "SELECT ts, ticker, px_last::float * 100 AS bp FROM md.index_eod WHERE ticker = ANY(%s) AND ts >= %s "
        "ORDER BY ts", params=[tickers, start]))
    wide = raw.pivot(index="ts", on="ticker", values="bp").drop_nulls().sort("ts")
    return wide.select(pl.col("ts").cast(pl.Date), pl.col(tickers[0]).alias("level"),
                       (pl.col(tickers[1]) - pl.col(tickers[0])).alias("curve"))


def curve_moves(curve: str, lookback: int, asof: dt.date, days: int, history: pl.DataFrame | None = None,
                root: str | None = None) -> dict:
    """Vol of the level and curve factors in bp over `days` business days, and their correlation,
    from the `lookback` daily changes up to `asof`. `history` (load_curve) saves the query."""
    if history is None:
        history = load_curve(curve, asof - dt.timedelta(days=lookback * 2), root)
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
        out.append(clean_price(b["coupon"], b["maturity"], b["delivery"], b["fwd_yld"] + shift / 100.0,
                               b.get("freq") or 2) / b["cf"])
    return np.array(out)


def scenario_weights(moves: dict, spreads: bool = True) -> np.ndarray:
    """Probability of each FINE x FINE scenario, rows curve and columns level, summing to 1.
    Without `spreads` the curve is tied to the level, so only the diagonal has any mass."""
    if not spreads:
        density = np.exp(-FINE ** 2 / 2)
        return np.diag(density / density.sum())
    z_curve, z_level = np.meshgrid(FINE, FINE, indexing="ij")
    rho = moves["corr"]
    density = np.exp(-(z_level ** 2 - 2 * rho * z_level * z_curve + z_curve ** 2) / (2 * (1 - rho ** 2)))
    return density / density.sum()


def with_when_issued(root: str, basket: pl.DataFrame, issues: pl.DataFrame | None = None,
                     contracts: dict | None = None) -> pl.DataFrame:
    """The priced basket with spot and forward yields, plus a row per bond not issued yet."""
    basket = add_risk(basket)
    # the auction calendar that places unissued bonds is the Treasury's
    upcoming = when_issued(root, basket, issues, contracts) if root in DELIVERABLE_ROOTS else pl.DataFrame()
    basket = basket.with_columns(pl.lit("").alias("wi"))
    return basket if upcoming.is_empty() else pl.concat([basket, upcoming], how="diagonal")


def delivery_odds(basket: pl.DataFrame, curve: str, moves: dict, spreads: bool = True) -> dict:
    """Over the fine scenarios: each bond's probability of being cheapest, what its basis is
    worth at delivery (price points), and the future's expected settlement. `spreads=False`
    leaves only level moves, the curve moving with them as its beta to the level says."""
    weights = scenario_weights(moves, spreads)
    fine = converted_prices(basket, curve, FINE * moves["level"],
                            FINE * moves["curve"] * (1.0 if spreads else moves["corr"]))
    cheapest, floor = fine.argmin(axis=0), fine.min(axis=0)
    return dict(
        p_ctd=np.array([weights[cheapest == i].sum() for i in range(len(basket))]),
        option=basket["cf"].to_numpy() * ((fine - floor) * weights).sum(axis=(1, 2)),
        expected_future=float((floor * weights).sum()))


def load_calendar(contract: str) -> dict:
    """The contract's delivery-month dates: first_notice, first_delivery, last_trade."""
    raw = query_db("SELECT first_notice_date::date AS first_notice, first_delivery_date::date AS first_delivery, "
                   "last_trade_date::date AS last_trade FROM sec.fut_contracts WHERE contract = %s", params=[contract])
    return raw.iloc[0].to_dict()


def forward_clean(bond: dict, date: dt.date) -> float:
    """A priced bond's forward clean price for delivery on `date`, financed as utils.basis does."""
    rate = bond["financing"] / 100.0 / 360.0
    forward = bond["dirty"] * (1.0 + rate * (date - bond["ts"]).days)
    if bond["next_coupon"] <= date:
        forward -= bond["coupon"] / 2.0 * (1.0 + rate * (date - bond["next_coupon"]).days)
    return forward - accrued(bond["coupon"], bond["maturity"], date)


def delivered_on(basket: pl.DataFrame, date: dt.date) -> pl.DataFrame:
    """The basket as if delivery were on `date`: forward yields and years from there. A bond
    not issued yet keeps its modelled yield, and drops out if it is issued after `date`."""
    kept = basket.filter(pl.col("issue_date") <= date)
    yields = [b["fwd_yld"] if b["wi"] else yield_to_maturity(b["coupon"], b["maturity"], date, forward_clean(b, date))
              for b in kept.iter_rows(named=True)]
    return kept.with_columns(pl.lit(date).alias("delivery"), pl.Series("fwd_yld", yields),
                             ((pl.col("maturity") - pl.lit(date)).dt.total_days() / 365.25).alias("years"))


def after_close_vol(basket: pl.DataFrame, lookback: int) -> float:
    """Sd in bp of the yield move between the futures settlement and the delivery-notice deadline:
    the basket's 3pm-to-5pm price changes over DV01, pooled over the lookback, scaled by root time."""
    priced = basket.filter(pl.col("wi") == "")
    asof = priced["ts"][0]
    raw = pl.from_pandas(query_db(
        "SELECT left(a.cusip, 8) AS cusip, a.px_last::float - b.px_last::float AS move "
        "FROM md.ust_eod a JOIN md.ust_eod_3pm b USING (ts, cusip) "
        "WHERE left(a.cusip, 8) = ANY(%s) AND a.ts > %s AND a.ts <= %s "
        "AND a.px_last BETWEEN %s AND %s AND b.px_last BETWEEN %s AND %s",
        params=[priced["cusip"].to_list(), asof - dt.timedelta(days=lookback * 365 // 252), asof,
                *PRICE_BOUNDS, *PRICE_BOUNDS]))
    bp = raw.join(priced.select("cusip", "dv01"), on="cusip").select(pl.col("move") / pl.col("dv01"))["move"]
    return float(bp.std() * np.sqrt(NOTICE_HOURS / MEASURED_HOURS))


def wild_card(cf: float, tail_sd: float, given_up: np.ndarray) -> float:
    """What a run of wild card evenings is worth per bond, in price points. An evening pays
    (1 - cf) x the price move less cf x the basis given up, when that is positive; `tail_sd`
    is the sd of the first term, `given_up` the basis each evening. Playing one ends the rest."""
    strike = cf * np.maximum(given_up, 0.0)
    z = strike / max(tail_sd, 1e-12)
    worth = tail_sd * norm.pdf(z) - strike * norm.sf(z)
    unplayed = np.cumprod(np.r_[1.0, 1.0 - norm.sf(z)[:-1]])
    return float((worth * unplayed).sum())


def delivery_options(basket: pl.DataFrame, curve: str, moves: dict, odds: dict, days: int, lookback: int) -> dict:
    """The model value split the Red Book's way. `switch`, `timing` and `wild` are per bond in
    price points and add up to the model value; `ctd` is the CTD's switch value by driver;
    the rest is what the two timing options were worked out from."""
    priced = basket.filter(pl.col("wi") == "")  # the first rows of the basket, in the same order
    ctd = int(basket["implied_repo"].fill_null(-np.inf).arg_max())
    bond = basket.row(ctd, named=True)
    calendar = load_calendar(bond["contract"])

    # switch: levels alone, then spreads, then the bonds not issued yet
    level = float(delivery_odds(priced, curve, moves, spreads=False)["option"][ctd])
    no_new = float(delivery_odds(priced, curve, moves)["option"][ctd])
    switch = float(odds["option"][ctd])
    cf = basket["cf"].to_numpy()
    split = dict(level=level, spreads=no_new - level, new_issues=switch - no_new)
    if bond["root"] in GLOBAL_ROOTS:  # the timing options are valued for Treasuries only
        return dict(switch=odds["option"], timing=cf * 0.0, wild=cf * 0.0, ctd=split)

    # carry: the future is worth the lower of delivery on the last day and on the first
    early = max(calendar["first_delivery"], bond["ts"])
    shrink = np.sqrt(np.busday_count(bond["ts"], early) / days)
    early_moves = {**moves, "level": moves["level"] * shrink, "curve": moves["curve"] * shrink}
    early_future = delivery_odds(delivered_on(basket, early), curve, early_moves)["expected_future"]
    timing = max(odds["expected_future"] - early_future, 0.0)

    # wild card: an evening each business day from the first position day to the day before the last
    # trading day (after that the price is fixed for good: the end-of-month option), giving up the
    # carry left and the switch value left (shrunk by root time)
    start = max(np.busday_offset(calendar["first_notice"], -1), np.busday_offset(bond["ts"], 1))
    evenings = np.arange(start, np.datetime64(calendar["last_trade"]))
    evenings = evenings[np.is_busday(evenings)]
    carry_left = (bond["gross_basis"] - bond["net_basis"]) / bond["n_days"] * (
        np.datetime64(bond["delivery"]) - evenings).astype(int)
    given_up = carry_left + switch * np.sqrt(np.busday_count(evenings, bond["delivery"]) / days)
    close_sd = after_close_vol(basket, lookback)
    tail_sd = abs(1.0 - bond["cf"]) * bond["dv01"] * close_sd
    # early delivery already made means no evenings are left to play
    wild = wild_card(bond["cf"], tail_sd, given_up) / bond["cf"] if timing == 0.0 and len(evenings) else 0.0

    return dict(
        switch=odds["option"], timing=cf * timing, wild=cf * wild, ctd=split,
        early=early, early_future=early_future, late_future=odds["expected_future"],
        evenings=len(evenings), last_trade=calendar["last_trade"], close_sd=close_sd, tail_sd=tail_sd,
        given_up=given_up)


def study(root: str, curve: str | None = None, lookback: int = 252) -> dict:
    curve = curve or CURVES[root]
    basket = with_when_issued(root, load_basket(root))
    first = basket.row(0, named=True)
    days = max(int(np.busday_count(first["ts"], first["delivery"])), 1)
    moves = curve_moves(curve, lookback, first["ts"], days, root=root)
    unit = 32 if root in DELIVERABLE_ROOTS else 100  # the basis is quoted in 32nds for Treasuries, 100ths elsewhere

    # CTD probabilities and option values over the fine scenarios
    odds = delivery_odds(basket, curve, moves)
    prob, expected_future = odds["p_ctd"], odds["expected_future"]
    weights = scenario_weights(moves)

    # the switch value by driver, and the timing options on top of it
    options = delivery_options(basket, curve, moves, odds, days, lookback)

    # the displayed matrix: each cell's CTD, and the mass of the fine scenarios nearest it
    level, slope = np.round(SIGMAS * moves["level"]), np.round(SIGMAS * moves["curve"])
    shown = converted_prices(basket, curve, level, slope).argmin(axis=0)
    nearest = np.abs(FINE[:, None] - SIGMAS[None, :]).argmin(axis=1)
    mass = np.zeros((len(SIGMAS), len(SIGMAS)))
    np.add.at(mass, (nearest[:, None], nearest[None, :]), weights)

    scale = FACE.get(root, 100_000) / 100.0  # price points per 100 face -> dollars per contract
    bonds = basket.with_columns(
        pl.Series("p_ctd", prob), pl.Series("switch_32", options["switch"] * unit),
        pl.Series("timing_32", options["timing"] * unit), pl.Series("wild_32", options["wild"] * unit),
        (pl.col("gross_basis") * unit).alias("gross_32"), (pl.col("net_basis") * unit).alias("net_32"),
        ((pl.col("gross_basis") - pl.col("net_basis")) * unit).alias("carry_32"),
        ((pl.col("implied_repo") - pl.col("financing")) * 100).alias("irr_spread_bp"),
        (pl.col("dv01") * scale).alias("dv01_usd"), (pl.col("dv01") / pl.col("cf") * scale).alias("dv01_cf_usd"),
    ).with_columns((pl.col("switch_32") + pl.col("timing_32") + pl.col("wild_32")).alias("model_32")
                   ).with_columns((pl.col("net_32") - pl.col("model_32")).alias("rich_32")
                   ).sort("implied_repo", descending=True, nulls_last=True)
    ctd = bonds.row(0, named=True)
    return dict(
        root=root, curve=curve, lookback=lookback, bonds=bonds, ctd=ctd, moves=moves, days=days,
        unit=unit, crncy=GLOBAL_ROOTS.get(root, "USD"),
        asof=first["ts"], contract=first["contract"], delivery=first["delivery"], fut_px=first["fut_px"],
        financing=first["financing"], expected_future=expected_future, options=options,
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
                       "switch_32", "timing_32", "wild_32", "model_32", "rich_32", "implied_repo", "dv01_cf_usd",
                       "p_ctd"))
    o = state["options"]
    print(f"CTD switch value by driver, 1/{state['unit']}: "
          + ", ".join(f"{k} {v * state['unit']:.2f}" for k, v in o["ctd"].items()))
    moves = state["moves"]
    print(f"{state['curve']}: level vol {moves['level']:.1f}bp, curve vol {moves['curve']:.1f}bp, "
          f"corr {moves['corr']:+.2f}  |  CTD DV01/CF {state['crncy']} {state['ctd']['dv01_cf_usd']:,.2f}  "
          f"probability-weighted {state['blend_dv01']:,.2f}")
    if root in GLOBAL_ROOTS:
        return state
    print(f"carry: future {o['late_future']:.4f} delivering on the last day, {o['early_future']:.4f} on {o['early']}"
          f"  |  wild card: {o['evenings']} evenings to {o['last_trade']}, after-close sd {o['close_sd']:.2f}bp")
    h = new_issue_history(root, asof=state["asof"])
    print(f"since {h['start']}: a bond issued after its contract went front was CTD in {h['ever']} of "
          f"{h['contracts']} contracts ({h['week']} for 5+ days, {h['at_end']} on the last day measured), "
          f"{h['days_new']} of {h['days']} days")
    return {**state, "history": h}


if __name__ == "__main__":
    main(*sys.argv[1:2])
