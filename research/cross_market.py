"""Cross-market bond RV: every government bond as a spread over USD SOFR.

Huggins & Schaller, Fixed Income Relative Value Analysis, ch. 15-17
(book/cross_market). A bond in any currency can be asset swapped into its own
overnight rate plus a spread, and that floating leg basis swapped into SOFR. What
is left is one number per bond, its USD asset swap spread, on a yardstick shared
by every market:

    USD ASW = (local ASW - CCBS) x BPV_local / BPV_USD

    local ASW   the bond's yield less the OIS swap rate of the same tenor in its
                own currency (CORRA, ESTR, SONIA; SOFR for a Treasury)
    CCBS        the cross-currency basis swap against SOFR, quoted as the spread on
                the foreign leg (negative when dollars are scarce)
    BPV ratio   a basis point a year in the foreign currency is not worth a basis
                point a year in dollars; the ratio of the two swap annuities converts

The 10-year Canada against the 10-year Treasury is then the difference of their
USD ASWs: the pick-up a dollar investor earns by owning the Canada, swapped and
basis swapped, over owning the Treasury swapped. Positive means the Canada is
cheap to the Treasury. Its z-score against its own history is the signal; futures
(CN against TY) are how a view on it would be expressed.

Simplifications, each against what the book does:
  * benchmark yields, not individual bonds, and the yield less the swap rate, not a
    par asset swap. The book shows the yield spread is distorted by coupon when the
    bond and swap curves are apart; benchmarks trade near par, which limits it.
  * no credit adjustment. The book subtracts an adjusted CDS so that only richness
    is left; here Italy's and France's credit sits inside their spread.
  * no specialness, haircut or benchmark-premium regressors (ch. 8, 18).
  * closes are local: a day's change mixes time zones, the level does not.
Japan is left out until its SOFR basis swap is stored.

    python -m research.cross_market 10
"""

from __future__ import annotations

import sys

import polars as pl

from stats import half_life, ou_zscore
from utils.helpers import query_db
from utils.rates import annual_yield, swap_annuity, usd_asset_swap_spread

START = "2021-03-01"  # SOFR-based cross-currency basis swaps start in 2021 (CAD in mid-2022)
TENORS = [2, 5, 10, 30]
LOOKBACKS = [63, 126, 252]
# market -> name, currency, benchmark yield ticker, and whether that yield compounds semiannually
MARKETS = {
    "US": ("Treasury", "USD", "USGG{}YR Index", True),
    "CA": ("Canada", "CAD", "GCAN{}YR Index", True),
    "DE": ("Bund", "EUR", "GDBR{} Index", False),
    "FR": ("OAT", "EUR", "GFRN{} Index", False),
    "IT": ("BTP", "EUR", "GBTPGR{} Index", False),
    "GB": ("Gilt", "GBP", "GUKG{} Index", True),
}
OIS = {"USD": "USOSFR{} Curncy", "CAD": "CDSO{} Curncy", "EUR": "EESWE{} Curncy", "GBP": "BPSWS{} Curncy"}
CCBS = {"CAD": "CDXOQQ{} Curncy", "EUR": "EUXOQQ{} Curncy", "GBP": "BPXOQQ{} Curncy"}
FUTURES = {"US": {2: "TU", 5: "FV", 10: "TY", 30: "US"}, "CA": {2: "CV", 5: "XQ", 10: "CN"},
           "DE": {2: "DU", 5: "OE", 10: "RX", 30: "UB"}, "FR": {10: "OAT"}, "IT": {2: "BTS", 10: "IK"}, "GB": {10: "G"}}


def load_quotes(tenor: int, start: str = START) -> pl.DataFrame:
    """Daily closes for one tenor, wide: ts, then per ticker a column in the units stored
    (yields and swap rates in percent, basis swaps in bp)."""
    tickers = ([m[2].format(tenor) for m in MARKETS.values()] + [t.format(tenor) for t in OIS.values()]
               + [t.format(tenor) for t in CCBS.values()])
    raw = pl.from_pandas(query_db(
        "SELECT ts, ticker, px_last::float AS px FROM md.index_eod WHERE ticker = ANY(%s) AND ts >= %s ORDER BY ts",
        params=[tickers, start]))
    return raw.with_columns(pl.col("ts").cast(pl.Date)).pivot(index="ts", on="ticker", values="px").sort("ts")


def usd_spreads(quotes: pl.DataFrame, tenor: int) -> pl.DataFrame:
    """Per market and day, in bp: yield (annual), OIS, local ASW, CCBS, USD ASW, and the pick-up
    over the Treasury. Long: ts, market, and those columns plus the BPV ratio."""
    usd_ois = pl.col(OIS["USD"].format(tenor))
    frames = []
    for market, (_, crncy, ticker, semi) in MARKETS.items():
        yld = pl.col(ticker.format(tenor))
        ois = pl.col(OIS[crncy].format(tenor))
        ccbs = pl.col(CCBS[crncy].format(tenor)) if crncy in CCBS else pl.lit(0.0)
        frames.append(quotes.select(
            "ts", pl.lit(market).alias("market"),
            ((annual_yield(yld) if semi else yld) * 100).alias("yld"), (ois * 100).alias("ois"), ccbs.alias("ccbs"),
            (swap_annuity(ois, tenor) / swap_annuity(usd_ois, tenor)).alias("bpv_ratio"),
        ).drop_nulls().with_columns((pl.col("yld") - pl.col("ois")).alias("local_asw")).with_columns(
            usd_asset_swap_spread(pl.col("local_asw"), pl.col("ccbs"), pl.col("bpv_ratio")).alias("usd_asw")))
    spreads = pl.concat(frames)
    treasury = spreads.filter(pl.col("market") == "US").select("ts", pl.col("usd_asw").alias("ust_asw"))
    return spreads.join(treasury, on="ts").with_columns(
        (pl.col("usd_asw") - pl.col("ust_asw")).alias("pickup")).sort("market", "ts")


def with_signal(spreads: pl.DataFrame, lookback: int) -> pl.DataFrame:
    """The pick-up's rolling mean, sd and z-score per market."""
    return spreads.with_columns(
        pl.col("pickup").rolling_mean(lookback).over("market").alias("mean"),
        pl.col("pickup").rolling_std(lookback).over("market").alias("sd"),
        pl.col("pickup").map_batches(lambda s: ou_zscore(s, lookback)).over("market").alias("z"))


def snapshot(signal: pl.DataFrame, tenor: int) -> pl.DataFrame:
    """One row per market on its latest day, with the pick-up's half-life over the sample."""
    rows = []
    for (market,), part in signal.group_by("market", maintain_order=True):
        last = part.row(-1, named=True)
        rows.append(dict(
            market=market, bond=MARKETS[market][0], future=FUTURES[market].get(tenor, ""), asof=last["ts"],
            yld=last["yld"], ois=last["ois"], local_asw=last["local_asw"], ccbs=last["ccbs"],
            bpv_ratio=last["bpv_ratio"], usd_asw=last["usd_asw"], pickup=last["pickup"],
            change_20d=last["pickup"] - part["pickup"][-21] if part.height > 21 else None, z=last["z"],
            half_life=half_life(part["pickup"]) if market != "US" else None, days=part.height))
    return pl.DataFrame(rows).sort("usd_asw")


def study(tenor: int = 10, lookback: int = 252, start: str = START) -> dict:
    # closes for the tenor, then every market on the SOFR yardstick
    quotes = load_quotes(tenor, start)
    spreads = usd_spreads(quotes, tenor)

    # the pick-up over the Treasury against its own range
    signal = with_signal(spreads, lookback)
    today = snapshot(signal, tenor)
    return dict(tenor=tenor, lookback=lookback, signal=signal, today=today)


def main(tenor: int = 10) -> dict:
    # every market's benchmark as a spread over SOFR, and its pick-up over the Treasury
    state = study(int(tenor))

    with pl.Config(tbl_rows=20, tbl_cols=20, tbl_width_chars=220, float_precision=2):
        print(f"{state['tenor']}y benchmarks as a spread over USD SOFR, bp (richest first)")
        print(state["today"])
    return state


if __name__ == "__main__":
    main(*sys.argv[1:2])
