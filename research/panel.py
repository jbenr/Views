"""The target/feature catalog and panel builder behind the research app.

This module owns data only: what a target IS, what features are available,
how they are loaded, and how healthy the resulting panel is. It does not
decide how anything is researched -- the three study classes do that, and the
app only draws them. Keeping the recipe here means the app, a notebook and a
future book module all build the same panel.

A target is a weighted basket of generic yields, expressed as a
``backtest.engine.TradeDef``. That is deliberate reuse: the object the app
explores and the object the backtest engine trades are the same type, so a
target cannot mean one thing in research and another in production.

Sign conventions, chosen to match the Bloomberg series they mirror:

    curve  a s b s   = b - a          (USYC{a}{b}: positive is steeper)
    fly    a s b s c = 2*b - a - c    (BF{a}{b}{c}: positive is belly cheap)

Verified against the database, not assumed: BF102030 Index regresses on
2*20y - 10y - 30y with beta 0.9923 and R2 0.9992 over 1,638 shared bars.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from backtest.engine import TradeDef
from stats import roll_mlr_diff
from utils.market_data import (
    align_columns,
    coverage_report,
    load_swaption_wide,
    load_wide,
    swaption_point,
)

# ---- the universe -----------------------------------------------------------

START = "2000-01-01"
BETA_LOOKBACK = 126

# generic on-the-run yields
HEADLINE = (2, 3, 5, 7, 10, 20, 30)
YIELDS = {f"{n}y": f"USGG{n}YR Index" for n in HEADLINE}

# curated exogenous features: alias -> (ticker, quoted in percent?)
# `percent=True` is scaled x100 at load so the whole panel is natively bps.
# Swap spreads and the MOVE-style vol surface are already bps and must not be.
EXO: dict[str, tuple[str, bool]] = {
    "real5y":  ("USGGT05Y Index", True),
    "real10y": ("USGGT10Y Index", True),
    "real30y": ("USGGT30Y Index", True),
    "be5":     ("USGGBE05 Index", True),
    "be10":    ("USGGBE10 Index", True),
    "be30":    ("USGGBE30 Index", True),
    "sofr2":   ("USOSFR2 Curncy", True),
    "sofr10":  ("USOSFR10 Curncy", True),
    "sofr30":  ("USOSFR30 Curncy", True),
    "swsp2":   ("USSFCT02 Curncy", False),  # swap spreads: already bps
    "swsp5":   ("USSFCT05 Curncy", False),
    "swsp10":  ("USSFCT10 Curncy", False),
    "swsp20":  ("USSFCT20 Curncy", False),
    "swsp30":  ("USSFCT30 Curncy", False),
    "mtg_cc":  ("MTGEFNCL Index", True),   # Fannie 30y current coupon yield
    "dxy":     ("DXY Curncy", False),
    "gold":    ("XAU Curncy", False),
    "spx":     ("SPX Index", False),
    "oil":     ("CL1 Comdty", False),
}

# These are not just explanatory features: each is an executable Treasury
# versus same-tenor SOFR-swap expression, so it is eligible as a target too.
SWAP_SPREADS = {
    name: ticker for name, (ticker, _percent) in EXO.items()
    if name.startswith("swsp")
}

# A custom target may use only legs with a clear executable expression.  Macro,
# vol, and other explanatory inputs intentionally stay out of this set.
TRADEABLE_LEGS = {**YIELDS, **SWAP_SPREADS}

# the ATM swaption grid, as (expiry, tenor) -> "vol_1Mo_30" style aliases
SWAPTION_EXPIRIES = ("1Mo", "3Mo", "6Mo", "1Y", "2Y", "5Y")
SWAPTION_TENORS = (2, 5, 10, 20, 30)
VOLS = {
    swaption_point(e, t): (e, t)
    for e in SWAPTION_EXPIRIES
    for t in SWAPTION_TENORS
}


# ---- the target catalog -----------------------------------------------------


def catalog() -> dict[str, TradeDef]:
    """Every prebuilt target: Treasury outrights/curves/flies and swap spreads."""
    out: dict[str, TradeDef] = {}
    for n in HEADLINE:
        out[f"{n}y"] = TradeDef.outright(f"{n}y", f"{n}y")
    for a in HEADLINE:
        for b in HEADLINE:
            if a < b:
                out[f"{a}s{b}s"] = TradeDef.spread(
                    f"{a}s{b}s", f"{a}y", f"{b}y", weights=(-1.0, 1.0)
                )
    for a in HEADLINE:
        for b in HEADLINE:
            for c in HEADLINE:
                if a < b < c:
                    out[f"{a}s{b}s{c}s"] = TradeDef.butterfly(
                        f"{a}s{b}s{c}s", f"{a}y", f"{b}y", f"{c}y",
                        weights=(-1.0, 2.0, -1.0),  # positive = belly cheap
                    )
    for name in SWAP_SPREADS:
        out[name] = TradeDef.outright(name, name)
    return out


CATALOG = catalog()


def parse_weights(text: str, name: str = "custom") -> TradeDef:
    """Build a TradeDef from a free-form '20y:2, 10y:-1, 30y:-1' string.

    Legs must be executable aliases (Treasury yields or Treasury swap spreads);
    anything else is a typo, not a silently-dropped leg.
    """
    legs: dict[str, float] = {}
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"leg {part!r} must look like '20y:2'")
        leg, weight = part.split(":", 1)
        leg = leg.strip()
        if leg not in TRADEABLE_LEGS:
            raise ValueError(
                f"unknown leg {leg!r}; choose from {sorted(TRADEABLE_LEGS)}"
            )
        try:
            legs[leg] = float(weight)
        except ValueError as exc:
            raise ValueError(f"weight for {leg!r} is not a number: {weight!r}") from exc
    if not legs:
        raise ValueError("no legs given")
    return TradeDef(name=name, legs=legs)


def resolve_target(selection: str, custom: str | None = None) -> TradeDef:
    """Catalog name, or 'custom' plus a weights string."""
    if selection == "custom":
        if not custom:
            raise ValueError("custom target selected but no weights given")
        return parse_weights(custom)
    if selection not in CATALOG:
        raise ValueError(f"unknown target {selection!r}")
    return CATALOG[selection]


# ---- panel construction -----------------------------------------------------


@dataclass(frozen=True)
class Panel:
    """A built research panel plus the metadata needed to interpret it."""

    data: pl.DataFrame
    target: TradeDef
    features: tuple[str, ...]
    weighting: str = "fixed"
    beta_lookback: int = BETA_LOOKBACK
    beta_dependent: str | None = None

    @property
    def columns(self) -> list[str]:
        return [self.target.name, *self.features]

    @property
    def weight_cols(self) -> list[str]:
        """Per-leg hedge weights used by a beta-weighted target."""
        if self.weighting != "beta":
            return []
        return self.beta_diagnostic_cols

    @property
    def beta_diagnostic_cols(self) -> list[str]:
        """Fitted rolling betas retained for display, including fixed baskets."""
        try:
            expected = [
                f"w_{leg}"
                for leg in hedge_legs(self.target, self.beta_dependent)
            ]
        except ValueError:
            return []
        return [col for col in expected if col in self.data.columns]


def dependent_leg(trade: TradeDef, override: str | None = None) -> str:
    """The leg the others hedge: the belly of a fly, the long leg of a curve.

    Under the catalog's sign conventions this is the unique largest positive
    weight -- +2 for a fly, +1 for a curve. Custom baskets can be ambiguous,
    in which case callers must nominate it rather than accept a guess.
    """
    if override is not None:
        if override not in trade.legs:
            raise ValueError(
                f"beta dependent leg {override!r} is not in {trade.name!r}: "
                f"{list(trade.legs)}"
            )
        return override

    largest = max(trade.legs.values())
    candidates = [leg for leg, weight in trade.legs.items() if weight == largest]
    if len(candidates) != 1:
        raise ValueError(
            f"cannot infer one beta dependent leg for {trade.name!r}; "
            f"choose one of {list(trade.legs)} explicitly"
        )
    return candidates[0]


def hedge_legs(trade: TradeDef, dependent: str | None = None) -> list[str]:
    """Every leg except the dependent one, in the target's own leg order."""
    dep = dependent_leg(trade, dependent)
    return [leg for leg in trade.legs if leg != dep]


def beta_weighted(
    data: pl.DataFrame,
    trade: TradeDef,
    lookback: int = BETA_LOOKBACK,
    ts_col: str = "ts",
    dependent: str | None = None,
) -> pl.DataFrame:
    """Hedge the dependent leg against the others with rolling changes betas.

    Fixed weights assume the wings hedge the belly in a constant ratio -- for
    a fly, half each. That is an assumption, not a measurement, and it decays
    as the curve reprices. Here the ratio is fitted on daily changes over a
    trailing window instead:

        level = w_dep * ( dep - SUM_i beta_i * hedge_i )

    Scaling by the dependent leg's fixed weight makes the two constructions
    directly comparable: a fly whose fitted betas are both 0.5 reproduces
    2*belly - wing1 - wing2 exactly, and a curve with beta 1.0 reproduces
    long - short. So the fixed target is the special case where the
    regression finds textbook weights.

    Betas come from changes, not levels, because only a changes beta is a
    hedge ratio -- a levels beta absorbs whatever trend the legs share.

    Returns ts, the weighted level under the target's name, and one w_<leg>
    column per hedge leg. Rows before the regression warms up are null.
    """
    dep = dependent_leg(trade, dependent)
    hedges = hedge_legs(trade, dep)
    if not hedges:
        raise ValueError(
            f"{trade.name!r} has one leg; there is nothing to hedge it against. "
            "Beta weighting needs a spread, fly or wider package."
        )

    frame = align_columns(data, [dep, *hedges], date_col=ts_col).sort(ts_col)
    reg = roll_mlr_diff(frame.select(hedges), frame[dep], lookback=lookback)

    # roll_mlr_diff drops the first differencing row; align_columns already
    # guarantees no interior nulls, so the shortfall is exactly that warmup.
    pad = len(frame) - len(reg)
    def padded(name: str) -> pl.Series:
        return pl.concat([pl.Series(name, [None] * pad, dtype=pl.Float64), reg[name]])

    betas = {leg: padded(f"beta_{leg}") for leg in hedges}
    scale = float(trade.legs[dep])
    level = frame[dep].cast(pl.Float64)
    for leg, beta in betas.items():
        level = level - beta * frame[leg].cast(pl.Float64)

    return frame.select(ts_col).with_columns(
        (level * scale).alias(trade.name),
        *[series.alias(f"w_{leg}") for leg, series in betas.items()],
    )


def _needed_sources(
    target: TradeDef, features: list[str]
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Split requested names into yield legs, exogenous, vol points, composites."""
    yields = {leg for leg in target.legs if leg in YIELDS}
    target_exo = [leg for leg in target.legs if leg in EXO]
    unknown_legs = set(target.legs) - yields - set(target_exo)
    if unknown_legs:
        raise ValueError(f"unknown target leg(s): {sorted(unknown_legs)}")
    exo, vols, composites = [], [], []
    for name in features:
        if name in YIELDS:
            yields.add(name)
        elif name in EXO:
            exo.append(name)
        elif name in VOLS:
            vols.append(name)
        elif name in CATALOG:
            composites.append(name)
            yields.update(CATALOG[name].legs)
        else:
            raise ValueError(f"unknown feature {name!r}")
    return sorted(yields), list(dict.fromkeys([*target_exo, *exo])), vols, composites


def build_panel(
    target: TradeDef,
    features: list[str] | None = None,
    start: str = START,
    end: str | None = None,
    weighting: str = "fixed",
    beta_lookback: int = BETA_LOOKBACK,
    beta_dependent: str | None = None,
) -> Panel:
    """Load every source the target and features need and assemble one frame.

    Series are joined on date with a full outer join, never an inner one: an
    exogenous series that starts in 2021 must not silently truncate a target
    with history back to 2000. Trimming to a common sample is the study's job
    (align_columns), and `diagnostics` reports what that would cost.
    """
    features = list(features or [])
    yields, exo, vols, composites = _needed_sources(target, features)

    tickers = {a: YIELDS[a] for a in yields}
    tickers.update({a: EXO[a][0] for a in exo})
    bps_cols = [a for a in yields] + [a for a in exo if EXO[a][1]]

    frame = load_wide(tickers, start=start, end=end, bps_cols=bps_cols)
    if vols:
        vol_frame = load_swaption_wide(
            [VOLS[v] for v in vols], start=start, end=end
        )
        frame = frame.join(vol_frame, on="ts", how="full", coalesce=True)

    # Sort BEFORE building composites. composite_series returns a bare Series
    # that with_columns attaches positionally, and a full outer join appends
    # right-only dates at the end rather than in date order -- so computing on
    # the unsorted frame and attaching to the sorted one silently pairs each
    # target value with the wrong day.
    frame = frame.sort("ts")
    frame = frame.with_columns(
        target.composite_series(frame).alias(target.name),
        *[CATALOG[c].composite_series(frame).alias(c) for c in composites],
    )

    if weighting not in {"fixed", "beta"}:
        raise ValueError("weighting must be 'fixed' or 'beta'")
    beta_cols: list[str] = []
    leg_cols: list[str] = []
    if len(target.legs) > 1:
        try:
            fitted = beta_weighted(
                frame, target, lookback=beta_lookback, dependent=beta_dependent
            )
        except ValueError:
            # A fixed custom basket may not nominate one dependent leg. That
            # makes beta comparison ambiguous, but does not invalidate its
            # explicitly fixed arithmetic.
            if weighting == "beta":
                raise
            fitted = None
        if fitted is not None:
            beta_cols = [c for c in fitted.columns if c.startswith("w_")]
            if weighting == "beta":
                # the fitted level replaces the fixed one under the same name
                frame = frame.drop(target.name).join(fitted, on="ts", how="left")
                # Held-PnL diagnostics require the original legs.
                leg_cols = list(target.legs)
            else:
                # Fixed basket remains untouched; these weights are a visual
                # comparison against the entered ratios, never model inputs.
                frame = frame.join(fitted.select("ts", *beta_cols), on="ts", how="left")

    # a name can appear as both target and feature; keep first occurrence only
    keep = list(dict.fromkeys(
        ["ts", target.name, *features, *beta_cols, *leg_cols]
    ))
    return Panel(
        data=frame.select(keep),
        target=target,
        features=tuple(features),
        weighting=weighting,
        beta_lookback=beta_lookback,
        beta_dependent=beta_dependent,
    )


# ---- panel health -----------------------------------------------------------


def gap_report(data: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """Calendar gaps in the common non-null sample a study would actually see.

    The study classes drop incomplete rows and then difference, so a dropped
    bar turns the next day's '1-day move' into a multi-day move without
    saying so. This counts how often that happens and how bad the worst case
    is. Weekends are expected; anything longer is a holiday or a missing pull.
    """
    aligned = align_columns(data, columns).sort("ts")
    if len(aligned) < 2:
        return pl.DataFrame({"n_rows": [len(aligned)]})
    days = aligned["ts"].to_numpy().astype("datetime64[D]")
    step = (days[1:] - days[:-1]).astype(int)
    # a normal step is 1 business day; 3 spans a weekend
    return pl.DataFrame({
        "n_rows": [len(aligned)],
        "rows_dropped_by_alignment": [len(data) - len(aligned)],
        "median_step_days": [float(np.median(step))],
        "gaps_over_3d": [int((step > 3).sum())],
        "gaps_over_5d": [int((step > 5).sum())],
        "largest_gap_days": [int(step.max())],
        "pct_bars_after_a_gap": [float((step > 3).mean())],
    })


def units_report(data: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """Min/median/max/daily-move per series, so a percent-vs-bps slip is visible.

    A rates series that belongs in bps but reads 4.7 instead of 470 is the
    single easiest way to make every downstream beta wrong by 100x. There is
    deliberately no automatic flag: a spread or fly can sit legitimately near
    zero, so any threshold that catches a mis-scaled yield also cries wolf on
    every curve. Read `median` against what the series is supposed to be.
    """
    rows = []
    for col in columns:
        s = data[col].drop_nulls()
        rows.append({
            "series": col,
            "n": len(s),
            "min": s.min() if len(s) else None,
            "median": s.median() if len(s) else None,
            "max": s.max() if len(s) else None,
            "daily_chg_std": s.diff().std() if len(s) > 2 else None,
        })
    return pl.DataFrame(rows)


def stale_report(data: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """Runs of a repeated value -- a series that stopped updating but kept printing.

    Found the hard way: md.swaption_vol opens with eleven consecutive bars at
    exactly 82.3 before a 28-day hole. A stale run is worse than a hole,
    because a hole shows up as missing while a repeat looks like real data
    that simply did not move -- and it drags any correlation toward zero.
    """
    rows = []
    for col in columns:
        s = data.select("ts", col).drop_nulls(col)[col]
        if len(s) < 2:
            rows.append({"series": col, "n": len(s), "pct_unchanged": None,
                         "longest_repeat_run": None})
            continue
        same = (s.diff() == 0).fill_null(False).to_numpy()
        longest = best = 0
        for flag in same:
            best = best + 1 if flag else 0
            longest = max(longest, best)
        rows.append({
            "series": col,
            "n": len(s),
            "pct_unchanged": float(same.mean()),
            "longest_repeat_run": int(longest) + 1 if longest else 1,
        })
    return pl.DataFrame(rows)


def remark_report(panel: Panel, horizons: tuple[int, ...] = (5, 20, 60)) -> pl.DataFrame:
    """How much of a beta-weighted target's move is the hedge being rewritten.

    A beta-weighted level is marked with today's betas. A position opened h
    days ago holds the betas it was opened with, so part of the level's change
    is the hedge ratio moving, not P&L. Auditing PairRVStudy put that at 61%
    of scored variance on a 10s30s package -- large enough that a scorecard
    reading the level's forward change is not measuring a tradeable return.

    Reported here so the cost of beta weighting is visible when you choose it,
    rather than discovered in a backtest later. `held` freezes the entry
    betas; `remark` is the remainder.
    """
    if panel.weighting != "beta":
        return pl.DataFrame()

    trade = panel.target
    dep = dependent_leg(trade, panel.beta_dependent)
    hedges = hedge_legs(trade, dep)
    need = [trade.name, dep, *hedges, *panel.weight_cols]
    frame = align_columns(panel.data, need).sort("ts") if all(
        c in panel.data.columns for c in need
    ) else None
    if frame is None or len(frame) < max(horizons) + 2:
        return pl.DataFrame()

    scale = float(trade.legs[dep])
    level = frame[trade.name].to_numpy()
    dep_lvl = frame[dep].to_numpy()
    hedge_lvl = {leg: frame[leg].to_numpy() for leg in hedges}
    betas = {leg: frame[f"w_{leg}"].to_numpy() for leg in hedges}

    rows = []
    for h in horizons:
        scored = level[h:] - level[:-h]
        held = dep_lvl[h:] - dep_lvl[:-h]
        for leg in hedges:
            held = held - betas[leg][:-h] * (hedge_lvl[leg][h:] - hedge_lvl[leg][:-h])
        held = held * scale
        remark = scored - held
        rows.append({
            "horizon": h,
            "n_obs": len(scored),
            "std_scored": float(np.std(scored)),
            "std_held_pnl": float(np.std(held)),
            "std_remark": float(np.std(remark)),
            "remark_share_of_var": float(np.var(remark) / np.var(scored))
            if np.var(scored) > 0 else None,
            "corr_scored_vs_held": float(np.corrcoef(scored, held)[0, 1]),
        })
    return pl.DataFrame(rows)


def weight_report(panel: Panel) -> pl.DataFrame:
    """Stability of the fitted hedge weights, against their fixed-weight prior.

    A hedge ratio that swings or flips sign is not a hedge. `fixed_prior` is
    what the equal-weight construction assumes the beta is, so the two are
    directly comparable.
    """
    if panel.weighting != "beta":
        return pl.DataFrame()
    trade = panel.target
    dep = dependent_leg(trade, panel.beta_dependent)
    scale = float(trade.legs[dep])
    hedges = hedge_legs(trade, dep)
    rows = []
    for leg in hedges:
        s = panel.data[f"w_{leg}"].drop_nulls()
        if not len(s):
            continue
        rows.append({
            "leg": leg,
            "n": len(s),
            "fixed_prior": -float(trade.legs[leg]) / scale,
            "median": float(s.median()),
            "std": float(s.std()),
            "min": float(s.min()),
            "max": float(s.max()),
            "pct_negative": float((s < 0).mean()),
        })
    if not rows:
        return pl.DataFrame()

    # Hedge weights summing to 1 cancel a one-for-one parallel yield shift.
    # This is a yield-space proxy only, not DV01 neutrality: actual DV01 needs
    # security-level durations and contract sizing, neither of which this
    # generic-yield panel has.
    # ignore_nulls=False so warmup bars stay null instead of summing to zero
    total = panel.data.select(
        pl.sum_horizontal(
            [pl.col(f"w_{leg}") for leg in hedges], ignore_nulls=False
        ).alias("s")
    )["s"].drop_nulls()
    rows.append({
        "leg": "SUM (1.0 = parallel-shift neutral)",
        "n": len(total),
        "fixed_prior": sum(-float(trade.legs[leg]) / scale for leg in hedges),
        "median": float(total.median()),
        "std": float(total.std()),
        "min": float(total.min()),
        "max": float(total.max()),
        "pct_negative": float((total < 0).mean()),
    })
    return pl.DataFrame(rows)


def diagnostics(panel: Panel) -> dict[str, pl.DataFrame]:
    """The checks that decide whether a panel is safe to research.

    The last two are empty for a fixed-weight target and only populate when
    beta weighting introduces the problems they measure.
    """
    cols = panel.columns
    return {
        "coverage": coverage_report(panel.data, cols),
        "gaps": gap_report(panel.data, cols),
        "units": units_report(panel.data, cols),
        "stale": stale_report(panel.data, cols),
        "weights": weight_report(panel),
        "remark": remark_report(panel),
    }
