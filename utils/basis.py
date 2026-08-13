"""Cash-versus-futures basis for the Treasury futures complex.

Everything needed to turn `md.fut_eod` + `sec.fut_contracts.dlv_basket` +
`md.ust_eod` into a daily net basis series per contract root. This is data
access and rate construction, so it lives in utils/ alongside market_data;
`book/basis/cash_futures.py` is the strategy that trades the output.

    from utils.basis import net_basis, basis_panel, futures_roll, futures_roll_panel

    nb = net_basis(["TY", "US"], start="2015-01-01")
    panel = basis_panel(nb)
    roll = futures_roll(["TY", "US"], start="2015-01-01")
    roll_panel = futures_roll_panel(roll)

The measure. For each date pick the front contract (rolling to the deferred
one once the front is within `min_days` of delivery), price every bond in its
deliverable basket forward to the delivery date at the financing rate, and
compare against the futures invoice price. Writing d for the dirty price, r
for financing, n for days to delivery, and c for a coupon paid nc days before
delivery (zero when none falls in the window):

    forward_clean = d(1 + r n/360) - c(1 + r nc/360) - ai_delivery
    net_basis     = forward_clean - futures_px * conversion_factor
    implied_repo  solves  d(1 + r n/360) - c(1 + r nc/360) = invoice
                  where   invoice = futures_px * cf + ai_delivery

The CTD is the bond with the highest implied repo — equivalently the lowest
net basis. Net basis is the delivery option premium: buying the basis (long
cash, short futures) pays it, so a rich net basis is a sell.

Which series to trade on. `net_basis_32` is the measure in 32nds and is the
signal series: its noise is roughly constant across the contract cycle
(median |value| moves only 1.7 -> 2.2 from 15 to 105 days to delivery).
`net_basis_bps` restates the same number as an implied-repo spread, which is
easier to read but divides by days-to-delivery and so amplifies a fixed price
error as delivery approaches — median |spread| runs 88bp inside 30 days
against 23bp beyond 105. Treat it as a diagnostic, not a signal.

Two approximations, both deliberate and both material enough to state:

  * There is no repo rate in the database. Financing is proxied by the front
    SOFR future (`SFR1`, 2018+) falling back to the front Fed Funds future
    (`FF1`, 2000+). Both are unsecured/GC-ish overnight expectations, while
    true term repo on CTD collateral trades special (below SOFR), so the
    measured net basis is biased low and special-collateral richness shows up
    inside the signal rather than being netted out of it.
  * Delivery is assumed to occur on the last business day of the contract
    month, which is what a short with a positive-carry bond would do. The
    timing option is therefore not priced separately.
"""

from __future__ import annotations

import datetime as dt

import polars as pl

from utils.helpers import query_db

DELIVERABLE_ROOTS = ["TU", "FV", "TY", "UXY", "US", "WN"]

# Bloomberg futures month codes, used to recover a contract month for the
# handful of contracts whose sec.fut_contracts metadata was never populated.
MONTH_CODES = {
    "F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
    "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12,
}

# Clean prices outside this band are quote errors rather than bonds. The
# common case is a when-issued row where px_last and yld_ytm_mid are
# transposed; the issue-date filter catches most of those and this catches
# the rest.
PRICE_BOUNDS = (20.0, 200.0)


# ── loaders ──────────────────────────────────────────────────────────────────

def load_futures(roots: list[str], start: str) -> pl.DataFrame:
    """Front and deferred generic futures prices: ts, root, rank, contract, fut_px."""
    generics = [f"{root}{rank}" for root in roots for rank in (1, 2)]
    raw = query_db(
        """
        SELECT ts, generic_ticker, contract, px_last::float AS fut_px
        FROM md.fut_eod
        WHERE generic_ticker = ANY(%s) AND ts >= %s AND px_last IS NOT NULL
        ORDER BY ts
        """,
        params=[generics, start],
    )
    return pl.from_pandas(raw).with_columns(
        pl.col("ts").cast(pl.Date),
        pl.col("generic_ticker").str.replace(r"\d+$", "").alias("root"),
        pl.col("generic_ticker").str.extract(r"(\d+)$").cast(pl.Int8).alias("rank"),
    ).drop("generic_ticker")


def load_baskets(roots: list[str]) -> pl.DataFrame:
    """Deliverable baskets: contract, cusip (8-char), conversion factor.

    dlv_basket stores 8-character CUSIPs while md.ust_eod and
    sec.auctioned_securities store the 9-character form, so every join
    against a basket goes through left(cusip, 8).
    """
    raw = query_db(
        """
        SELECT contract,
               jsonb_array_elements(dlv_basket->'deliverables')->>'cusip' AS cusip,
               (jsonb_array_elements(dlv_basket->'deliverables')->>'conversion_factor')::float AS cf
        FROM sec.fut_contracts
        WHERE generic_ticker = ANY(%s) AND dlv_basket IS NOT NULL
        """,
        params=[roots],
    )
    return pl.from_pandas(raw).filter(pl.col("cf") > 0)


def load_bond_terms(cusips: list[str]) -> pl.DataFrame:
    """Coupon, maturity and original issue date per deliverable CUSIP.

    Reopenings give a CUSIP several auction rows; the coupon and maturity are
    invariant across them but int_rate is null on some, hence the max(). The
    issue date is taken as the earliest, which is when the bond starts
    trading for real rather than when-issued.
    """
    raw = query_db(
        """
        SELECT left(cusip, 8)      AS cusip,
               max(int_rate)::float AS coupon,
               max(maturity_date)::date AS maturity,
               min(issue_date)::date    AS issue_date
        FROM sec.auctioned_securities
        WHERE left(cusip, 8) = ANY(%s) AND int_rate IS NOT NULL
        GROUP BY 1
        """,
        params=[cusips],
    )
    return pl.from_pandas(raw).with_columns(
        pl.col("maturity").cast(pl.Date), pl.col("issue_date").cast(pl.Date)
    )


def load_deliverable_prices(cusips: list[str], start: str) -> pl.DataFrame:
    """Clean prices for deliverable CUSIPs: ts, cusip, clean."""
    raw = query_db(
        """
        SELECT ts, left(cusip, 8) AS cusip, px_last::float AS clean
        FROM md.ust_eod
        WHERE left(cusip, 8) = ANY(%s) AND ts >= %s
          AND px_last BETWEEN %s AND %s
        ORDER BY ts
        """,
        params=[cusips, start, *PRICE_BOUNDS],
    )
    return pl.from_pandas(raw).with_columns(pl.col("ts").cast(pl.Date))


def load_financing(start: str) -> pl.DataFrame:
    """Financing proxy in percent: front SOFR future, falling back to Fed Funds.

    SFR1 only starts in 2018; FF1 goes back to 2000 and tracks it closely
    (2.44 vs 2.37 in mid-2019, 3.645 vs 3.633 in mid-2026), so the splice is
    not a regime break in the signal.
    """
    raw = query_db(
        """
        SELECT ts, generic_ticker, 100.0 - px_last::float AS rate
        FROM md.fut_eod
        WHERE generic_ticker IN ('SFR1', 'FF1') AND ts >= %s
          AND px_last IS NOT NULL
        """,
        params=[start],
    )
    wide = pl.from_pandas(raw).with_columns(pl.col("ts").cast(pl.Date)).pivot(
        index="ts", on="generic_ticker", values="rate"
    )
    for col in ("SFR1", "FF1"):
        if col not in wide.columns:
            wide = wide.with_columns(pl.lit(None, dtype=pl.Float64).alias(col))
    return wide.select(
        "ts", pl.coalesce("SFR1", "FF1").alias("financing")
    ).drop_nulls().sort("ts")


def delivery_dates(roots: list[str], observed_last: dict[str, dt.date]) -> dict[str, dt.date]:
    """Assumed delivery date per contract: last business day of the contract month.

    contract_month comes from sec.fut_contracts where it was populated. Twelve
    contracts (the H5/M5 cycle) never got their metadata pulled, so their
    month is recovered from the Bloomberg month code plus the year implied by
    the last date the contract traded.
    """
    raw = query_db(
        """
        SELECT contract, generic_ticker AS root, contract_month
        FROM sec.fut_contracts
        WHERE generic_ticker = ANY(%s)
        """,
        params=[roots],
    )
    known = pl.from_pandas(raw)

    out: dict[str, dt.date] = {}
    for row in known.iter_rows(named=True):
        contract, month = row["contract"], row["contract_month"]
        if month is None:
            month = _month_from_ticker(contract, row["root"], observed_last.get(contract))
        if month is not None:
            out[contract] = _last_business_day(month)
    return out


def _month_from_ticker(contract: str, root: str, last_traded: dt.date | None) -> dt.date | None:
    """Contract month from the ticker's month code, e.g. TYM5 -> June of its year.

    The year digit is ambiguous (TYM5 could be 2005, 2015 or 2025), so the
    year is taken as whichever candidate places the contract month at or just
    after the last date the contract actually traded.
    """
    if last_traded is None:
        return None
    code = contract[len(root):][:1]
    month = MONTH_CODES.get(code)
    if month is None:
        return None
    for year in (last_traded.year, last_traded.year + 1):
        candidate = dt.date(year, month, 1)
        if candidate >= dt.date(last_traded.year, last_traded.month, 1):
            return candidate
    return None


def _last_business_day(month_start: dt.date) -> dt.date:
    day = (month_start.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
    day -= dt.timedelta(days=1)
    while day.weekday() >= 5:
        day -= dt.timedelta(days=1)
    return day


# ── accrued interest ─────────────────────────────────────────────────────────

def _shift_back_months(date: pl.Expr, months: pl.Expr) -> pl.Expr:
    """`date` moved back `months` months, day clamped to the target month.

    Clamping is what makes end-of-month maturities work: a bond maturing
    31 Dec pays on 30 Jun, not an invalid 31 Jun.
    """
    total = date.dt.year() * 12 + date.dt.month() - 1 - months
    year = (total // 12).cast(pl.Int32)
    month = (total % 12 + 1).cast(pl.Int8)
    month_len = pl.date(year, month, 1).dt.month_end().dt.day()
    return pl.date(year, month, pl.min_horizontal(date.dt.day(), month_len))


def coupon_period(maturity: pl.Expr, on: pl.Expr) -> tuple[pl.Expr, pl.Expr]:
    """The semiannual coupon dates bracketing `on`, as (previous, next).

    Coupon dates are stepped back from maturity in six-month hops.
    """
    gap_months = (
        maturity.dt.year() * 12 + maturity.dt.month()
        - on.dt.year() * 12 - on.dt.month()
    )
    hops = gap_months // 6
    # A first guess of `hops` can land after `on` when the day of month falls
    # later than `on`'s; one more hop back always brackets it.
    guess = _shift_back_months(maturity, hops * 6)
    hops = hops + (guess > on).cast(pl.Int32)
    return (
        _shift_back_months(maturity, hops * 6),
        _shift_back_months(maturity, (hops - 1) * 6),
    )


def accrued_interest(coupon: pl.Expr, maturity: pl.Expr, on: pl.Expr) -> pl.Expr:
    """Act/act accrued interest per 100 face on semiannual Treasury coupons."""
    last, nxt = coupon_period(maturity, on)
    elapsed = (on - last).dt.total_days()
    period = (nxt - last).dt.total_days()
    return coupon / 2.0 * elapsed / period


# ── the basis ────────────────────────────────────────────────────────────────

def net_basis(
    roots: list[str] | None = None,
    start: str = "2010-01-01",
    min_days: int = 15,
) -> pl.DataFrame:
    """Daily CTD net basis per contract root.

    One row per (ts, root): the deliverable with the highest implied repo,
    its net basis in 32nds, and the implied-repo spread to financing in bps.

    `min_days` is the roll rule — once the front contract is within this many
    calendar days of its assumed delivery date the deferred contract takes
    over. It keeps the series continuous and keeps implied repo, which
    divides by days-to-delivery, away from its singularity.
    """
    roots = list(roots or DELIVERABLE_ROOTS)

    futures = load_futures(roots, start)
    baskets = load_baskets(roots).filter(
        pl.col("contract").is_in(futures["contract"].unique().implode())
    )
    cusips = baskets["cusip"].unique().to_list()
    terms = load_bond_terms(cusips)
    prices = load_deliverable_prices(cusips, start)
    financing = load_financing(start)

    observed_last = {
        row["contract"]: row["last_ts"]
        for row in futures.group_by("contract")
        .agg(pl.col("ts").max().alias("last_ts"))
        .iter_rows(named=True)
    }
    delivery = delivery_dates(roots, observed_last)
    futures = futures.with_columns(
        pl.col("contract").replace_strict(delivery, default=None).alias("delivery")
    ).drop_nulls("delivery")

    # Roll rule: front contract until it is inside min_days of delivery, then
    # the deferred one. Ranking by (days_to_delivery >= min_days, -rank)
    # prefers the nearest contract that still clears the threshold.
    candidates = futures.with_columns(
        (pl.col("delivery") - pl.col("ts")).dt.total_days().alias("n_days")
    ).filter(pl.col("n_days") >= min_days)
    chosen = (
        candidates.sort(["ts", "root", "rank"])
        .group_by(["ts", "root"], maintain_order=True)
        .first()
    )

    quotes = (
        chosen.join(baskets, on="contract")
        .join(prices, on=["ts", "cusip"])
        .join(terms, on="cusip")
        .join(financing, on="ts")
        # A bond quoted before it is issued is a when-issued quote, and
        # md.ust_eod transposes price and yield on those rows.
        .filter(pl.col("ts") >= pl.col("issue_date"))
    )

    priced = quotes.with_columns(
        accrued_interest(pl.col("coupon"), pl.col("maturity"), pl.col("ts")).alias("ai_now"),
        accrued_interest(pl.col("coupon"), pl.col("maturity"), pl.col("delivery")).alias("ai_dlv"),
        coupon_period(pl.col("maturity"), pl.col("ts"))[1].alias("next_coupon"),
    ).with_columns(
        (pl.col("clean") + pl.col("ai_now")).alias("dirty"),
        (pl.col("fut_px") * pl.col("cf")).alias("principal_invoice"),
        # A coupon falling between the quote date and delivery is cash in the
        # holder's hand for the rest of the window. Dropping it understates
        # the forward price by a whole coupon. A contract cycle is under six
        # months, so there is at most one.
        pl.when(pl.col("next_coupon") <= pl.col("delivery"))
        .then(pl.col("coupon") / 2.0)
        .otherwise(0.0)
        .alias("coupon_cash"),
        pl.when(pl.col("next_coupon") <= pl.col("delivery"))
        .then((pl.col("delivery") - pl.col("next_coupon")).dt.total_days())
        .otherwise(0)
        .alias("coupon_days"),
    ).with_columns(
        (pl.col("principal_invoice") + pl.col("ai_dlv")).alias("invoice"),
        (
            pl.col("dirty")
            * (1.0 + pl.col("financing") / 100.0 * pl.col("n_days") / 360.0)
            - pl.col("coupon_cash")
            * (1.0 + pl.col("financing") / 100.0 * pl.col("coupon_days") / 360.0)
            - pl.col("ai_dlv")
        ).alias("forward_clean"),
    ).with_columns(
        (pl.col("clean") - pl.col("principal_invoice")).alias("gross_basis"),
        (pl.col("forward_clean") - pl.col("principal_invoice")).alias("net_basis"),
        # The r that makes the purchase grow exactly into the invoice:
        #   dirty(1 + r n/360) - c(1 + r nc/360) = invoice
        (
            (pl.col("invoice") + pl.col("coupon_cash") - pl.col("dirty"))
            / (
                pl.col("dirty") * pl.col("n_days") / 360.0
                - pl.col("coupon_cash") * pl.col("coupon_days") / 360.0
            )
            * 100.0
        ).alias("implied_repo"),
    )

    ctd = (
        priced.drop_nulls(["implied_repo", "net_basis"])
        .sort(["ts", "root", "implied_repo"], descending=[False, False, True])
        .group_by(["ts", "root"], maintain_order=True)
        .first()
        .sort(["root", "ts"])
    )

    return ctd.select(
        "ts",
        "root",
        "contract",
        "delivery",
        "n_days",
        pl.col("cusip").alias("ctd"),
        "coupon",
        "maturity",
        "cf",
        "clean",
        "fut_px",
        "financing",
        "implied_repo",
        (pl.col("gross_basis") * 32.0).alias("gross_basis_32"),
        (pl.col("net_basis") * 32.0).alias("net_basis_32"),
        # The rate view of the same number: positive means the basis is rich,
        # i.e. implied repo sits below financing. Unlike the 32nds it is
        # roughly invariant to days-to-delivery and comparable across roots.
        ((pl.col("financing") - pl.col("implied_repo")) * 100.0).alias("net_basis_bps"),
    )


def basis_panel(nb: pl.DataFrame) -> pl.DataFrame:
    """Pivot `net_basis` output wide, one signal and one tradable level per root.

    Per root:
      {root}_nb_bps  implied-repo spread in bps — the signal series
      {root}_nb32    net basis in 32nds — the trader's quote
      {root}_level   roll-stitched net basis in 32nds — what the engine marks

    The level exists because net basis is quoted per contract and jumps at
    every roll. Stitching accumulates within-contract changes only, so the
    roll gap never lands in the P&L; the series is a running 32nds total for
    a position held continuously through rolls, which is the back-adjusted
    convention futures P&L is always marked on.
    """
    stitched = nb.sort(["root", "ts"]).with_columns(
        pl.when(pl.col("contract") == pl.col("contract").shift(1).over("root"))
        .then(pl.col("net_basis_32").diff().over("root"))
        .otherwise(0.0)
        .alias("d_nb32")
    ).with_columns(
        pl.col("d_nb32").fill_null(0.0).cum_sum().over("root").alias("level")
    )

    frames = []
    for root in stitched["root"].unique(maintain_order=True):
        part = stitched.filter(pl.col("root") == root).select(
            "ts",
            pl.col("net_basis_bps").alias(f"{root}_nb_bps"),
            pl.col("net_basis_32").alias(f"{root}_nb32"),
            pl.col("level").alias(f"{root}_level"),
        )
        frames.append(part)

    panel = frames[0]
    for part in frames[1:]:
        panel = panel.join(part, on="ts", how="full", coalesce=True)
    return panel.sort("ts")


# -- futures calendar roll ---------------------------------------------------

def futures_roll(
    roots: list[str] | None = None,
    start: str = "2010-01-01",
    min_days: int = 5,
) -> pl.DataFrame:
    """Daily front-minus-deferred futures roll per contract root.

    Returns one row per (ts, root), using Bloomberg generic rank 1 and rank 2
    futures. `roll` is front_px - deferred_px in futures price points. Positive
    means the front contract is rich to the deferred contract; buying the roll
    is long front / short deferred.

    `min_days` filters out the final days before the front contract delivery
    assumption. The roll is still a generic spread, so use futures_roll_panel()
    for a back-adjusted tradable level that does not book generic-roll gaps.
    """
    roots = list(roots or DELIVERABLE_ROOTS)
    futures = load_futures(roots, start)

    observed_last = {
        row["contract"]: row["last_ts"]
        for row in futures.group_by("contract")
        .agg(pl.col("ts").max().alias("last_ts"))
        .iter_rows(named=True)
    }
    delivery = delivery_dates(roots, observed_last)
    futures = futures.with_columns(
        pl.col("contract").replace_strict(delivery, default=None).alias("delivery")
    ).drop_nulls(["delivery", "fut_px"])

    front = futures.filter(pl.col("rank") == 1).select(
        "ts",
        "root",
        pl.col("contract").alias("front_contract"),
        pl.col("delivery").alias("front_delivery"),
        pl.col("fut_px").alias("front_px"),
    )
    deferred = futures.filter(pl.col("rank") == 2).select(
        "ts",
        "root",
        pl.col("contract").alias("deferred_contract"),
        pl.col("delivery").alias("deferred_delivery"),
        pl.col("fut_px").alias("deferred_px"),
    )

    return (
        front.join(deferred, on=["ts", "root"], how="inner")
        .with_columns(
            (pl.col("front_delivery") - pl.col("ts")).dt.total_days().alias("n_days"),
            (pl.col("deferred_delivery") - pl.col("ts")).dt.total_days().alias("deferred_n_days"),
            (pl.col("front_px") - pl.col("deferred_px")).alias("roll"),
        )
        .filter(pl.col("n_days") >= min_days)
        .sort(["root", "ts"])
    )


def futures_roll_panel(roll: pl.DataFrame) -> pl.DataFrame:
    """Pivot futures_roll() output wide with a stitched tradable level.

    Per root:
      {root}_roll   front-minus-deferred generic roll, in futures price points
      {root}_level  back-adjusted roll level, resetting roll gaps to zero PnL

    Generic futures rolls change contract identity. The level accumulates
    same-pair roll changes only, so a position marked through the generic roll
    date does not book the mechanical jump from one contract pair to the next.
    """
    if roll.is_empty():
        return pl.DataFrame(schema={"ts": pl.Date})

    required = {"ts", "root", "front_contract", "deferred_contract", "roll"}
    missing = required.difference(roll.columns)
    if missing:
        raise ValueError(f"futures roll frame missing columns: {sorted(missing)}")

    stitched = (
        roll.sort(["root", "ts"])
        .with_columns(
            pl.concat_str(
                [pl.col("front_contract"), pl.col("deferred_contract")],
                separator="/",
            ).alias("_pair")
        )
        .with_columns(
            pl.when(pl.col("_pair") == pl.col("_pair").shift(1).over("root"))
            .then(pl.col("roll").diff().over("root"))
            .otherwise(0.0)
            .alias("d_roll")
        )
        .with_columns(
            pl.col("d_roll").fill_null(0.0).cum_sum().over("root").alias("level")
        )
    )

    frames = []
    for root in stitched["root"].unique(maintain_order=True):
        part = stitched.filter(pl.col("root") == root).select(
            "ts",
            pl.col("roll").alias(f"{root}_roll"),
            pl.col("level").alias(f"{root}_level"),
        )
        frames.append(part)

    panel = frames[0]
    for part in frames[1:]:
        panel = panel.join(part, on="ts", how="full", coalesce=True)
    return panel.sort("ts")
