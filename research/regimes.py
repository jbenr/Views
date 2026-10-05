"""Macro regimes: causal states of the world that discovery can later gate on.

Each regime is a daily series built only from data known that day, plus a
named state ("hiking", "high vol", "trades with oil"). States are
confirmed causally: a new state counts only once it has held for
``confirm`` days in a row, so a value hovering at a threshold does not flicker.
An "episode" is one unbroken run of a state; regimes are long and few, so the
episode count says how much independent history a gate really has.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from backtest.lab import gate_percentile_rank
from utils.market_data import load_futures_wide, load_swaption_wide, load_wide, swaption_point

INPUTS = {"10y": "USGG10YR Index", "oil": "CL1 Comdty", "infl5": "USSWIT5 Curncy"}
BPS_INPUTS = ["10y", "infl5"]
# Fed funds futures: FF1 is the current month, FF7 the month six months out; 100 - price = expected average rate.
FED_FUNDS = {"ff_now": "FF1", "ff_6m": "FF7"}
REALIZED_DAYS = 126  # six months of trading days
IMPLIED_POINT = ("1Mo", 10)  # 1m10y ATM swaption, normal vol in bp/yr
IMPLIED = swaption_point(*IMPLIED_POINT)


@dataclass(frozen=True)
class RegimeParams:
    cycle_threshold_bp: float = 25.0     # |realized 6m + priced 6m| beyond this is hiking / cutting
    vol_window: int = 20                 # realized-vol window (days)
    vol_rank_window: int = 756           # trailing days the vol percentile is ranked against
    vol_low: float = 0.25                # percentile at or below: low vol
    vol_high: float = 0.75               # percentile at or above: high vol
    oil_window: int = 126                # rolling correlation window (days)
    oil_high: float = 0.30               # correlation at or above: rates trade with oil
    oil_low: float = 0.0                 # correlation below: none / inverse
    inflation_low_bp: float = 225.0      # 5y inflation swap below this: low inflation
    inflation_high_bp: float = 275.0     # at or above this: high inflation
    confirm: int = 5                     # days a new state must hold before it counts


# Display order and colour of each regime's states (first = "low" side).
STATES = {
    "policy": ["cutting", "on hold", "hiking"],
    "vol": ["low vol", "normal vol", "high vol"],
    "implied": ["low vol", "normal vol", "high vol"],
    "oil": ["none / inverse", "weak", "trades with oil"],
    "inflation": ["low inflation", "anchored", "high inflation"],
}


def load_inputs(start: str) -> pl.DataFrame:
    """The 10y and 5y inflation swap in bp, front-month WTI in $, fed funds futures, and 1m10y implied vol, by date."""
    frame = load_wide(INPUTS, start=start, bps_cols=BPS_INPUTS)
    fed_funds = load_futures_wide(FED_FUNDS, start=start)
    for alias in FED_FUNDS:
        if alias not in fed_funds.columns:
            fed_funds = fed_funds.with_columns(pl.lit(None, dtype=pl.Float64).alias(alias))
    frame = frame.join(fed_funds.select("ts", *FED_FUNDS), on="ts", how="full", coalesce=True)
    implied = load_swaption_wide([IMPLIED_POINT], start=start)
    if IMPLIED in implied.columns:
        frame = frame.join(implied.select("ts", IMPLIED), on="ts", how="full", coalesce=True)
    else:
        frame = frame.with_columns(pl.lit(None, dtype=pl.Float64).alias(IMPLIED))
    return frame.sort("ts")


def confirm_states(raw: list, days: int) -> list:
    """A new state is adopted only after it has held ``days`` in a row (causal); before that the old one stays."""
    out, current, run, last = [], None, 0, None
    for state in raw:
        run = run + 1 if state == last and state is not None else (1 if state is not None else 0)
        last = state
        if state is not None and (current is None or (state != current and run >= days)):
            current = state
        out.append(current)
    return out


def _bucket(values: np.ndarray, low: float, high: float, names: list[str]) -> list:
    """names[0] below ``low``, names[2] at or above ``high``, names[1] between; None where missing."""
    return [None if not np.isfinite(v) else names[0] if v < low else names[2] if v >= high else names[1]
            for v in values]


def policy_cycle(frame: pl.DataFrame, p: RegimeParams) -> pl.DataFrame:
    """Fed policy moves over the year around today, from fed funds futures (bp).

    realized = the front-month rate now minus six months ago (moves already
    made); priced = the contract six months out minus the front month
    (moves the market expects). Their sum spans the past six months and the
    next six, using only what is known today, so a cycle shows up as soon as
    it starts being priced and stays on while it is being delivered.
    """
    data = frame.select("ts", "ff_now", "ff_6m").drop_nulls()
    rate = (100.0 - data["ff_now"]) * 100.0
    realized = rate - rate.shift(REALIZED_DAYS)
    priced = (data["ff_now"] - data["ff_6m"]) * 100.0
    value = (realized + priced).alias("value")
    raw = _bucket(value.fill_null(np.nan).to_numpy(), -p.cycle_threshold_bp, p.cycle_threshold_bp, STATES["policy"])
    return data.select("ts").with_columns(value, realized.alias("realized"), priced.alias("priced"),
                                          rate.alias("policy_rate"),
                                          pl.Series("state", confirm_states(raw, p.confirm)))


def inflation_regime(frame: pl.DataFrame, p: RegimeParams) -> pl.DataFrame:
    """5y zero-coupon inflation swap (bp) against fixed bands around the 2% target.

    The 5y swap rather than the 1y: the 1y has bad prints (e.g. -514bp in
    2008, -212bp in 2024). Swaps sit a little above breakevens, so 2.25-2.75%
    is roughly anchored.
    """
    data = frame.select("ts", "infl5").drop_nulls()
    raw = _bucket(data["infl5"].to_numpy(), p.inflation_low_bp, p.inflation_high_bp, STATES["inflation"])
    return data.select("ts").with_columns(data["infl5"].alias("value"),
                                          pl.Series("state", confirm_states(raw, p.confirm)))


def _ranked_vol(data: pl.DataFrame, vol: pl.Series, p: RegimeParams) -> pl.DataFrame:
    """Low / normal / high vol by the vol's causal percentile against its own trailing history."""
    rank = gate_percentile_rank(vol.cast(pl.Float64).fill_null(np.nan).to_numpy(),
                                min_history=min(252, p.vol_rank_window), window=p.vol_rank_window)
    raw = _bucket(rank, p.vol_low, p.vol_high, STATES["vol"])
    return data.select("ts").with_columns(vol.alias("value"), pl.Series("percentile", rank).fill_nan(None),
                                          pl.Series("state", confirm_states(raw, p.confirm)))


def rate_vol(frame: pl.DataFrame, p: RegimeParams) -> pl.DataFrame:
    """Realized vol of daily 10y changes (bp, annualized), ranked against its own trailing history."""
    data = frame.select("ts", "10y").drop_nulls()
    return _ranked_vol(data, (data["10y"].diff().rolling_std(p.vol_window) * np.sqrt(252)).fill_nan(None), p)


def implied_rate_vol(frame: pl.DataFrame, p: RegimeParams) -> pl.DataFrame:
    """1m10y ATM swaption normal vol (bp/yr), ranked against its own trailing history.

    The swaption surface is clean only from September 2021, so with a year
    of ranking history its states start around September 2022.
    """
    data = frame.select("ts", IMPLIED).drop_nulls()
    return _ranked_vol(data, data[IMPLIED], p)


def oil_sensitivity(frame: pl.DataFrame, p: RegimeParams) -> pl.DataFrame:
    """Rolling correlation of daily 10y changes with daily oil changes.

    Oil changes are in dollars, not percent: front-month WTI went negative in
    April 2020, where percent changes break. Roll days add some noise.
    """
    data = frame.select("ts", "10y", "oil").drop_nulls()
    corr = data.select(pl.rolling_corr(pl.col("10y").diff(), pl.col("oil").diff(), window_size=p.oil_window)
                       .fill_nan(None))["10y"]
    raw = _bucket(corr.cast(pl.Float64).fill_null(np.nan).to_numpy(), p.oil_low, p.oil_high, STATES["oil"])
    return data.select("ts").with_columns(corr.alias("value"), pl.Series("state", confirm_states(raw, p.confirm)))


REGIMES = {
    "policy": dict(title="Policy cycle", build=policy_cycle, units="bp",
                   what="Fed moves over the year around today, from fed funds futures: the front-month rate's "
                        "change over the last six months plus the change priced over the next six."),
    "vol": dict(title="Rate vol", build=rate_vol, units="bp / yr",
                what="Realized volatility of daily 10y changes, ranked against its own trailing history."),
    "implied": dict(title="Implied rate vol · 1m10y", build=implied_rate_vol, units="bp / yr",
                    what="1m10y ATM swaption normal vol, ranked against its own trailing history (data from "
                         "September 2021, so states start around September 2022)."),
    "oil": dict(title="Oil sensitivity", build=oil_sensitivity, units="correlation",
                what="Rolling correlation of daily 10y changes with daily oil changes."),
    "inflation": dict(title="Inflation · 5y inflation swap", build=inflation_regime, units="bp",
                      what="5y zero-coupon inflation swap against fixed bands: below 2.25% low, 2.25-2.75% "
                           "anchored, 2.75% and above high."),
}


def episodes(series: pl.DataFrame, states: list[str]) -> pl.DataFrame:
    """Per state: share of days, number of episodes (unbroken runs), and their typical length."""
    data = series.drop_nulls("state")
    runs = (data.with_columns((pl.col("state") != pl.col("state").shift()).fill_null(True).cum_sum().alias("run"))
            .group_by("run", "state", maintain_order=True).agg(pl.len().alias("days"), pl.col("ts").min().alias("start")))
    rows = []
    for state in states:
        mine = runs.filter(pl.col("state") == state)
        rows.append({"state": state, "share_of_days": (mine["days"].sum() / len(data)) if len(data) else None,
                     "episodes": len(mine), "median_days": mine["days"].median() if len(mine) else None,
                     "longest_days": mine["days"].max() if len(mine) else None,
                     "last_started": mine["start"].max() if len(mine) else None})
    return pl.DataFrame(rows, infer_schema_length=None)


def current_state(series: pl.DataFrame) -> dict:
    """Today's state, its value, and when the current run began."""
    data = series.drop_nulls("state")
    if not len(data):
        return {}
    state = data["state"][-1]
    changed = data.with_row_index().filter(pl.col("state") != state)
    since = data["ts"][int(changed["index"][-1]) + 1] if len(changed) else data["ts"][0]
    return {"state": state, "value": data["value"][-1], "since": since, "as_of": data["ts"][-1]}


def regimes(start: str, params: RegimeParams | None = None) -> dict:
    """Every regime: its daily series, today's state and its episode table."""
    params = params or RegimeParams()
    frame = load_inputs(start)
    out = {}
    for name, spec in REGIMES.items():
        series = spec["build"](frame, params)
        out[name] = dict(spec, series=series, current=current_state(series),
                         episodes=episodes(series, STATES[name]))
    return out


# ---- regimes as discovery gates ---------------------------------------------

REGIME_PREFIX = "regime_"  # a frame column holding one regime's daily state, e.g. regime_policy
HISTORY_START = "2000-01-01"  # regimes need years of history before any sample they gate (ranks, 6m changes)


def regime_column(name: str) -> str:
    return f"{REGIME_PREFIX}{name}"


def with_regimes(frame: pl.DataFrame, names, params: RegimeParams | None = None) -> pl.DataFrame:
    """``frame`` plus one ``regime_<name>`` state column per regime, matched by date.

    Regimes are computed on their full history (from 2000) and then joined,
    so a sample's first day already has a properly warmed-up state. Days a
    regime has no state yet (e.g. implied vol before 2022) are null.
    """
    names = [n for n in names if n in REGIMES]
    if not names:
        return frame
    params = params or RegimeParams()
    inputs = load_inputs(HISTORY_START)
    out = frame.sort("ts")
    for name in names:
        states = REGIMES[name]["build"](inputs, params).select(
            pl.col("ts").cast(frame.schema["ts"]), pl.col("state").alias(regime_column(name)))
        # A day the regime's inputs did not print (a futures holiday, say) keeps yesterday's state:
        # known that day, and it stops a one-day gap from splitting an episode in two.
        out = (out.drop(regime_column(name), strict=False).join(states, on="ts", how="left")
               .with_columns(pl.col(regime_column(name)).forward_fill()))
    return out


def count_episodes(states, state: str) -> int:
    """Unbroken runs of ``state`` in a sequence of daily states; a missing day neither ends nor starts one."""
    count, previous = 0, None
    for value in states:
        if value is None:
            continue
        if value == state and previous != state:
            count += 1
        previous = value
    return count
