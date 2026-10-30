"""Oven: bake signals for one target against a boatload of inputs.

The question is which relationships produce the most profitable, most
trustworthy TRADES. For every (input, signal lookback, hedge lookback) it fits two rolling
regressions of the target y on the input x, each with its own job:

  signal  levels OLS     y = a + bL * x      "is y rich/cheap vs where x says it should be?"
  hedge   changes OLS   dy = c + bC * dx     "what ratio of x neutralises y's daily moves?"

The trade is the PORTFOLIO y hedged by x: hold y and -bC x, with bC frozen at
entry. An input that isn't itself tradable (vol, macro) can't be held as a
hedge, so its trade is y alone.

The entry signal has two bases, both trading that same portfolio:

  levels     y - (a + bL x)              rich/cheap vs the levels fit
  portfolio  (y - bC x) - its mean       the traded portfolio's own dislocation

Entries fade the signal's residual when it first crosses a threshold, either
normalised (residual / its std) or raw (bps). From the entry day the MODEL IS
FROZEN: the residual's intercept, slope and std keep their entry values, so an
exit fires only when prices move, never because the rolling fit re-centred on
the trade. On the portfolio basis the frozen residual IS the position, so its
reversion and the trade's P&L are the same move. Exits, combined per setup:
the frozen residual recovers a share of its entry value (revert; 0 = off), it
widens past a stop, the trade's own P&L, marked on the close, falls to
-pnl_stop bps (0 = off), a time stop at a multiple of the signal's half-life
at entry (hl_mult; 0 = off), or the max-hold cap. Revert 0 with a half-life
multiple is a pure half-life time exit. Every exit is decided on end-of-day
data -- no intraday high/low, which a beta-weighted portfolio doesn't have --
and fills are one bar after every decision; one position at a time per setup.

What the frozen model EXPECTS of a trade, at entry (exp_return, exp_bars): the
residual reverts like an OU process with the half-life measured at entry, so
after T bars it has recovered |r0| (1 - 2^(-T/hl)). T is when the revert target
is reached, in half-lives: hl log2(1 / (1 - revert)), so revert 0.5 is one
half-life and 0.75 two; revert 1 (all the way back) never quite arrives, so it
counts 90% of the way (3.3 half-lives); revert 0 (time exits only) runs to the
time stop. Either way T is capped at the half-life time stop and max hold. exp_return is that recovery net of cost, in bps; exp_risk is the
hedged portfolio's daily vol times sqrt(T). Trades can be sized from these
(SIZINGS): the board then scores the sized P&L.

Every setup is scored as a strategy: Sharpe from daily mark-to-market P&L
(flat days count as zero), total P&L, trades per year, hit rate, average and
median trade, profit factor, max drawdown, Calmar, time in market, Sharpe in
each half of the sample and the best trade's share of the total -- so one
lucky trade or one good era can't pass for confidence. Every trade also
records the state of its fit window at entry (R2, betas and their gap, beta
drift, residual half-life and vol, input and target vol).

    run = bake(panel, inputs)              # board + trades, in memory
    oven_job(target, inputs, settings)     # the Oven app's background job (python -m oven)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import polars as pl
from numba import njit

from research.panel import CATALOG, TRADEABLE_LEGS, build_panel
from stats import roll_half_life, roll_lr, roll_lr_diff

OVEN_RUNS = Path(__file__).parent / "data"

LOOKBACKS = [63, 126, 252, 504]
HEDGE_LOOKBACK = 126  # the hedge lookback older bakes used, before it was swept
HEDGE_LOOKBACKS = [63, 126, 252]  # changes-OLS lookbacks for the hedge ratio, swept like the signal's
Z_ENTRIES = [1.5, 2.0, 2.5]
RAW_ENTRIES = [5.0, 10.0, 15.0, 20.0]
REVERTS = [0.0, 0.5, 1.0]  # share of the entry residual recovered; 1.0 = back to fair; 0 = no reversion exit
MAX_HOLDS = [20, 60]  # the hard cap on bars held
STOPS = [0.0, 1.0]  # entry-sigmas the frozen residual may widen before stopping; 0 = none
HL_MULTS = [0.0, 1.0]  # time stop at this multiple of the signal's half-life at entry; 0 = none
PNL_STOPS = [0.0]  # stop loss: bps the trade may lose, marked on the close, before it exits; 0 = none
# how each trade is sized from what the frozen model expects of it at entry (see expected())
#   flat         every trade the same size
#   exp_return   in proportion to its expected return, bps net of cost
#   exp_sharpe   in proportion to expected return / expected risk over the expected hold
#   bps_per_day  in proportion to expected return / expected days held: capital efficiency
# A trade the model expects to lose on (or can't say: no half-life) is sized to nothing,
# i.e. not taken. Sizes are scaled to average 1 over a setup's trades, a constant that
# leaves Sharpe untouched and keeps P&L comparable to flat sizing.
SIZINGS = ("flat", "exp_return", "exp_sharpe", "bps_per_day")
SIZING_DEFAULT = ["flat"]
COST = 0.0  # bps per round trip
MIN_BARS = 500  # an input needs this much overlap with the target beyond its lookback
PERIODS = 252

SCALES = ("z", "raw")
# "open": still held at the last bar; "pnl_stop": the stop loss in bps of trade P&L
EXIT_REASONS = ("revert", "stop", "time", "half_life", "open", "pnl_stop")
FEATURES = ("entry_z", "r2", "beta_levels", "beta_changes", "hedge_gap", "beta_drift20",
            "half_life", "resid_sigma", "input_vol60", "target_vol60")
SETUP = ["input", "lookback", "hedge_lb", "basis", "scale", "entry", "revert", "max_hold", "stop", "hl_mult",
         "pnl_stop", "sizing"]
# What the entry signal measures (both trade the same hedged portfolio):
#   levels     y - (a + bL x): rich/cheap vs where the levels fit says y should be
#   portfolio  (y - bC x) - its rolling mean: the traded portfolio's own dislocation
BASES = ("levels", "portfolio")
RANKS = {
    "sharpe": "Sharpe",
    "worst_half_sharpe": "worst-half Sharpe",
    "trade_t": "trade t-stat",
    "total_pnl": "total P&L",
    "calmar": "Calmar",
}
# per-config daily sufficient statistics written by _walk
_STATS = ("sum", "sumsq", "days", "max_dd", "held", "sum1", "sumsq1", "days1", "sum2", "sumsq2", "days2")


# ---- model arrays -----------------------------------------------------------


def hedgeable(name: str) -> bool:
    """An input can be held as a hedge only if it is itself an executable series."""
    return name in CATALOG or name in TRADEABLE_LEGS


def _front_pad(values: np.ndarray, n: int) -> np.ndarray:
    return np.r_[np.full(n - len(values), np.nan), values]


def model_arrays(y: np.ndarray, x: np.ndarray, lookback: int, hedge_lookback: int) -> dict[str, np.ndarray]:
    """Rolling levels fit (signal), rolling changes beta (hedge) and the entry-state
    features, all fitted through each bar. y and x are aligned with no gaps."""
    n = len(y)
    lv = roll_lr(pl.Series(x), pl.Series(y), lookback=lookback)
    resid = lv["resid"].to_numpy().astype(float)
    sigma = pl.Series(resid).rolling_std(lookback, min_samples=lookback).to_numpy().astype(float)
    beta_l = lv["beta"].to_numpy().astype(float)
    beta_c = _front_pad(roll_lr_diff(pl.Series(x), pl.Series(y), lookback=hedge_lookback)["beta"]
                        .to_numpy().astype(float), n)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = resid / sigma
    dx, dy = np.diff(x, prepend=np.nan), np.diff(y, prepend=np.nan)

    # The portfolio basis: the traded portfolio y - bC x itself, against its own mean
    # over the lookback -- i.e. a levels fit whose slope is pinned to the hedge ratio.
    # Freezing it at entry freezes exactly the position the trade holds.
    xs, ys = pl.Series(x), pl.Series(y)
    mx, my = xs.rolling_mean(lookback).to_numpy(), ys.rolling_mean(lookback).to_numpy()
    vx = xs.rolling_var(lookback, ddof=0).to_numpy()
    vy = ys.rolling_var(lookback, ddof=0).to_numpy()
    cxy = (xs * ys).rolling_mean(lookback).to_numpy() - mx * my
    p_alpha = my - beta_c * mx
    p_resid = y - p_alpha - beta_c * x
    p_sigma = np.sqrt(np.clip(vy + beta_c ** 2 * vx - 2 * beta_c * cxy, 0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        p_z = p_resid / p_sigma
    return {
        "p_alpha": p_alpha,
        "p_resid": p_resid,
        "p_sigma": p_sigma,
        "p_z": p_z,
        "p_half_life": roll_half_life(pl.Series(p_resid), lookback=lookback).to_numpy().astype(float),
        "alpha": lv["alpha"].to_numpy().astype(float),
        "beta_levels": beta_l,
        "beta_changes": beta_c,
        "resid": resid,
        "resid_sigma": sigma,
        "entry_z": z,
        "r2": lv["r2"].to_numpy().astype(float),
        "hedge_gap": beta_l - beta_c,
        "beta_drift20": beta_l - np.r_[np.full(20, np.nan), beta_l[:-20]],
        "half_life": roll_half_life(pl.Series(resid), lookback=lookback).to_numpy().astype(float),
        "input_vol60": pl.Series(dx).rolling_std(60).to_numpy().astype(float),
        "target_vol60": pl.Series(dy).rolling_std(60).to_numpy().astype(float),
        # daily vol of the hedged portfolio y - bC x (yesterday's hedge): the trade's expected risk
        "port_vol60": pl.Series(dy - np.r_[np.nan, beta_c[:-1]] * dx).rolling_std(60).to_numpy().astype(float),
    }


def model_frame(data: pl.DataFrame, target: str, name: str, lookback: int, hedge_lookback: int) -> pl.DataFrame:
    """One model on a panel's data: ts, the target and the input under their own names
    (aligned, no gaps), every model array, the hedged portfolio's level and the 126d
    correlation of daily moves. What the Oven app and a live oven signal draw from."""
    aligned = data.select("ts", target, name).drop_nulls()
    y = aligned[target].to_numpy().astype(float)
    x = aligned[name].to_numpy().astype(float)
    arrays = model_arrays(y, x, lookback, hedge_lookback)
    package = held_package(y, x, arrays["beta_changes"], 1.0 if hedgeable(name) else 0.0)
    corr = (pl.DataFrame({"dy": np.diff(y, prepend=np.nan), "dx": np.diff(x, prepend=np.nan)})
            .select(pl.rolling_corr("dy", "dx", window_size=126)).to_series())
    return aligned.with_columns(*[pl.Series(k, v) for k, v in arrays.items()], pl.Series("portfolio", package),
                                corr.alias("corr126"))


def held_package(y: np.ndarray, x: np.ndarray, beta_c: np.ndarray, hedge: float = 1.0) -> np.ndarray:
    """The portfolio y hedged by x as a level: the cumulative daily P&L of holding y
    against yesterday's changes beta (zero before the beta exists). Display only --
    trades freeze their beta at entry rather than re-hedging daily."""
    move = np.diff(y, prepend=np.nan) - hedge * np.r_[np.nan, beta_c[:-1]] * np.diff(x, prepend=np.nan)
    return np.nancumsum(np.where(np.isfinite(move), move, 0.0))


def config_grid(z_entries=Z_ENTRIES, raw_entries=RAW_ENTRIES, reverts=REVERTS,
                max_holds=MAX_HOLDS, stops=STOPS, hl_mults=HL_MULTS, pnl_stops=PNL_STOPS,
                sizings=SIZING_DEFAULT) -> pl.DataFrame:
    """Every (scale, entry, revert, hold, stop, half-life multiple, P&L stop, sizing) cell a
    model is walked through. A cell with no reversion exit and no half-life stop is a pure
    max-hold exit."""
    unknown = [s for s in sizings if s not in SIZINGS]
    if unknown:
        raise ValueError(f"unknown sizing {unknown}; expected some of {SIZINGS}")
    rows = [("z", e) for e in z_entries] + [("raw", e) for e in raw_entries]
    # cast every value: the app sends whole numbers as ints, and a row-built frame typed
    # from its first rows would otherwise truncate a later 2.5 to 2
    cells = [(s, float(e), float(r), int(h), float(st), float(m), float(p), z)
             for (s, e), r, h, st, m, p, z in product(rows, reverts, max_holds, stops, hl_mults, pnl_stops, sizings)]
    return pl.DataFrame(cells, schema=["scale", "entry", "revert", "max_hold", "stop", "hl_mult", "pnl_stop",
                                       "sizing"], orient="row")


# ---- the walk ---------------------------------------------------------------


@njit(cache=True)
def _horizon(hold, hl_mult, half_life):
    """(bars to the time stop, its exit reason): max_hold, or sooner at hl_mult x the
    half-life measured at entry."""
    horizon, timeout = hold, 2
    if hl_mult > 0:
        by_half_life = max(1, int(np.ceil(hl_mult * half_life)))
        if by_half_life < horizon:
            horizon, timeout = by_half_life, 3
    return horizon, timeout


@njit(cache=True)
def _expect(abs_r0, hl, revert, horizon, cost):
    """(exp_return, exp_bars) the frozen model expects of an entry residual |r0| (module
    docstring): OU recovery with half-life hl by the expected exit, net of cost."""
    exp_bars = float(horizon)
    exp_return = np.nan
    if np.isfinite(hl) and hl > 0:
        if 0.0 < revert < 1.0:  # revert 0.5: one half-life; 0.75: two
            exp_bars = min(exp_bars, hl * np.log(1.0 / (1.0 - revert)) / np.log(2.0))
        elif revert >= 1.0:  # all the way back never arrives: count 90% of it (3.3 half-lives)
            exp_bars = min(exp_bars, hl * np.log(10.0) / np.log(2.0))
        exp_bars = max(exp_bars, 1.0)
        exp_return = abs_r0 * (1.0 - 2.0 ** (-exp_bars / hl)) - cost
    return exp_return, exp_bars


def expected_now(resid: float, half_life: float, port_vol: float, setup: dict, cost: float = COST) -> dict:
    """What the frozen model would expect of an entry at residual ``resid`` today, with
    the setup's exits: exp_return (bps, net of cost), exp_bars, bps_per_day and exp_risk
    (the portfolio's daily vol over the expected hold). The walk's own arithmetic."""
    hl = float(half_life) if half_life is not None else np.nan
    horizon, _ = _horizon(int(setup["max_hold"]), float(setup["hl_mult"]), hl)
    exp_return, exp_bars = _expect(abs(float(resid)), hl, float(setup["revert"]), horizon, float(cost))
    return dict(exp_return=exp_return, exp_bars=exp_bars, bps_per_day=exp_return / exp_bars,
                exp_risk=float(port_vol) * np.sqrt(exp_bars) if port_vol is not None else np.nan)


@njit(cache=True)
def _walk(y, x, alpha, beta_l, beta_c, sigma, half_life, port_vol, signals, hedge, scale, entry, revert, hold, stop,
          hl_mult, pnl_stop, sizing, cost, split, trades, stats):
    """Every config's trades and daily P&L statistics for one model.

    trades rows: (config, decision bar, exit fill bar, side, pnl, reason, size,
    exp_return, exp_bars), pnl already sized.
    stats rows:  _STATS, from daily marks of the held portfolio y - hedge*beta_c[entry]*x
    times the trade's size, over every bar from the first usable signal (flat days
    are zero), with the two halves split at bar ``split``. Entries are first
    crossings of |signal| over the threshold; the model is frozen at the decision bar."""
    n = y.shape[0]
    count = 0
    for c in range(scale.shape[0]):
        first = count
        s = signals[scale[c]]
        start = 1
        while start < n and not np.isfinite(s[start]):
            start += 1
        thr = entry[c]
        total = 0.0
        total_sq = 0.0
        s1 = 0.0
        ss1 = 0.0
        s2 = 0.0
        ss2 = 0.0
        cum = 0.0
        peak = 0.0
        dd = 0.0
        held = 0
        t = start + 1
        while t < n - 1:
            ok = (np.isfinite(s[t]) and np.isfinite(s[t - 1]) and abs(s[t]) > thr and abs(s[t - 1]) <= thr
                  and np.isfinite(alpha[t]) and np.isfinite(beta_l[t]) and np.isfinite(beta_c[t])
                  and sigma[t] > 0)
            if not ok:
                t += 1
                continue
            # a half-life time stop needs a half-life at entry; without one there is no such trade
            if hl_mult[c] > 0 and not (np.isfinite(half_life[t]) and half_life[t] > 0):
                t += 1
                continue
            r0 = y[t] - alpha[t] - beta_l[t] * x[t]
            sign0 = 1.0 if r0 > 0 else -1.0
            goal = (1.0 - revert[c]) * abs(r0)
            limit = abs(r0) + stop[c] * sigma[t]
            horizon, timeout = _horizon(hold[c], hl_mult[c], half_life[t])
            exp_return, exp_bars = _expect(abs(r0), half_life[t], revert[c], horizon, cost)
            size = 1.0
            if sizing[c] == 1:
                size = exp_return
            elif sizing[c] == 2:
                size = exp_return / (port_vol[t] * np.sqrt(exp_bars)) if port_vol[t] > 0 else np.nan
            elif sizing[c] == 3:
                size = exp_return / exp_bars
            if not (np.isfinite(size) and size > 0):  # expected to lose, or can't say: not taken
                t += 1
                continue
            f0 = t + 1
            side = -sign0
            ratio = hedge * beta_c[t]
            exit_u = -1
            reason = timeout
            for k in range(1, horizon + 1):
                u = t + k
                if u >= n - 1:
                    break
                frozen = sign0 * (y[u] - alpha[t] - beta_l[t] * x[u])
                if revert[c] > 0 and frozen <= goal:  # revert 0 = no reversion exit, time only
                    exit_u, reason = u, 0
                    break
                if stop[c] > 0 and frozen >= limit:
                    exit_u, reason = u, 1
                    break
                # the stop loss: the trade's P&L since its entry fill, marked on this close
                if pnl_stop[c] > 0 and side * ((y[u] - y[f0]) - ratio * (x[u] - x[f0])) <= -pnl_stop[c]:
                    exit_u, reason = u, 5
                    break
                if k == horizon:
                    exit_u, reason = u, timeout
            # runs off the end of the sample: still open, marked to the last bar; it is
            # recorded as an "open" trade but kept out of the stats, which score closed trades
            still_open = exit_u < 0
            if still_open:
                exit_u, reason = n - 2, 4
            f1 = exit_u + 1
            pnl = 0.0
            for d in range(f0 + 1, f1 + 1):  # daily marks of the held portfolio, at the trade's size
                mark = size * side * ((y[d] - y[d - 1]) - ratio * (x[d] - x[d - 1]))
                if still_open:
                    pnl += mark
                    continue
                if d == f1:
                    mark -= size * cost
                pnl += mark
                total += mark
                total_sq += mark * mark
                if d < split:
                    s1 += mark
                    ss1 += mark * mark
                else:
                    s2 += mark
                    ss2 += mark * mark
                cum += mark
                if cum > peak:
                    peak = cum
                if cum - peak < dd:
                    dd = cum - peak
                held += 1
            trades[count, 0] = c
            trades[count, 1] = t
            trades[count, 2] = f1
            trades[count, 3] = side
            trades[count, 4] = pnl
            trades[count, 5] = reason
            trades[count, 6] = size
            trades[count, 7] = exp_return
            trades[count, 8] = exp_bars
            count += 1
            if still_open:
                break
            t = f1
        # scale sizes to average 1 over this config's trades: Sharpe is unchanged by a
        # constant, and P&L reads on the same scale as flat sizing
        if count > first:
            norm = trades[first:count, 6].mean()
            trades[first:count, 4] /= norm
            trades[first:count, 6] /= norm
            total /= norm
            total_sq /= norm * norm
            s1 /= norm
            ss1 /= norm * norm
            s2 /= norm
            ss2 /= norm * norm
            dd /= norm
        days = max(n - 1 - start, 0)
        days1 = min(max(split - start - 1, 0), days)
        stats[c, 0] = total
        stats[c, 1] = total_sq
        stats[c, 2] = days
        stats[c, 3] = dd
        stats[c, 4] = held
        stats[c, 5] = s1
        stats[c, 6] = ss1
        stats[c, 7] = days1
        stats[c, 8] = s2
        stats[c, 9] = ss2
        stats[c, 10] = days - days1
    return count


def basis_arrays(arrays: dict, basis: str) -> dict[str, np.ndarray]:
    """The signal model of one basis: alpha and slope (frozen at entry), the residual,
    its scale, its z and its half-life."""
    if basis == "levels":
        return dict(alpha=arrays["alpha"], beta=arrays["beta_levels"], resid=arrays["resid"],
                    sigma=arrays["resid_sigma"], z=arrays["entry_z"], half_life=arrays["half_life"])
    if basis == "portfolio":
        return dict(alpha=arrays["p_alpha"], beta=arrays["beta_changes"], resid=arrays["p_resid"],
                    sigma=arrays["p_sigma"], z=arrays["p_z"], half_life=arrays["p_half_life"])
    raise ValueError(f"unknown signal basis {basis!r}; expected one of {BASES}")


def walk_model(y: np.ndarray, x: np.ndarray, arrays: dict, configs: pl.DataFrame, hedge: float = 1.0,
               cost: float = COST, split: int | None = None, basis: str = "levels") -> tuple[np.ndarray, np.ndarray]:
    """(trades, stats) for every config of one model: the raw arrays _walk fills.
    Configs without a pnl_stop or sizing column (hand-built ones) have no stop loss
    and flat sizing."""
    n = len(y)
    m = basis_arrays(arrays, basis)
    signals = np.ascontiguousarray(np.vstack([m["z"], m["resid"]]))
    scale = np.array([SCALES.index(s) for s in configs["scale"]], dtype=np.int64)
    sizing = np.array([SIZINGS.index(s) for s in configs["sizing"]] if "sizing" in configs.columns
                      else np.zeros(len(configs)), dtype=np.int64)
    # the trade's risk: the hedged portfolio's vol, or the target's own when it trades alone
    vol = arrays.get("port_vol60") if hedge else arrays.get("target_vol60")
    vol = np.full(n, np.nan) if vol is None else vol
    trades = np.empty((len(configs) * (n // 2 + 1), 9))
    stats = np.zeros((len(configs), len(_STATS)))
    count = _walk(np.ascontiguousarray(y, dtype=np.float64), np.ascontiguousarray(x, dtype=np.float64),
                  np.ascontiguousarray(m["alpha"]), np.ascontiguousarray(m["beta"]), arrays["beta_changes"],
                  np.ascontiguousarray(m["sigma"]), np.ascontiguousarray(m["half_life"], dtype=np.float64),
                  np.ascontiguousarray(vol, dtype=np.float64),
                  signals, float(hedge), scale, configs["entry"].to_numpy().astype(np.float64),
                  configs["revert"].to_numpy().astype(np.float64), configs["max_hold"].to_numpy().astype(np.int64),
                  configs["stop"].to_numpy().astype(np.float64), configs["hl_mult"].to_numpy().astype(np.float64),
                  (configs["pnl_stop"].to_numpy().astype(np.float64) if "pnl_stop" in configs.columns
                   else np.zeros(len(configs))), sizing,
                  float(cost), int(n if split is None else split), trades, stats)
    return trades[:count], stats


def walk_setup(frame: pl.DataFrame, target: str, setup: dict, cost: float = COST, pnl_stops=None, split=None):
    """Walk one setup (a dict of SETUP keys) on a model_frame -- at its own stop loss, or
    at each of ``pnl_stops``: (raw trades, stats, configs, model arrays, hedge)."""
    name = setup["input"]
    y, x = frame[target].to_numpy().astype(float), frame[name].to_numpy().astype(float)
    arrays = {c: frame[c].to_numpy().astype(float) for c in frame.columns
              if c not in ("ts", target, name, "portfolio", "corr126")}
    stops = [float(setup.get("pnl_stop", 0.0))] if pnl_stops is None else [float(p) for p in pnl_stops]
    configs = pl.DataFrame({"scale": setup["scale"], "entry": float(setup["entry"]), "revert": float(setup["revert"]),
                            "max_hold": int(setup["max_hold"]), "stop": float(setup["stop"]),
                            "hl_mult": float(setup["hl_mult"]), "pnl_stop": stops,
                            "sizing": setup.get("sizing", "flat")})
    hedge = 1.0 if hedgeable(name) else 0.0
    raw, stats = walk_model(y, x, arrays, configs, hedge, cost, split, setup["basis"])
    return raw, stats, configs, arrays, hedge


def setup_trades(frame: pl.DataFrame, target: str, setup: dict, cost: float = COST) -> pl.DataFrame:
    """Every trade of one setup on a model_frame, closed and (last, reason "open") any
    position still held, with entry state and expectations; empty if it never traded."""
    raw, _, configs, arrays, hedge = walk_setup(frame, target, setup, cost)
    if not len(raw):
        return pl.DataFrame()
    lb = setup["lookback"]
    return trade_frame(raw, setup["input"], lb, setup["hedge_lb"], setup["basis"], frame["ts"], arrays, {lb: arrays},
                       [lb], configs, hedge)


def _sharpe(total, total_sq, days):
    mean = total / days
    var = total_sq / days - mean ** 2
    return pl.when((days > 1) & (var > 1e-12)).then(mean / var.sqrt() * np.sqrt(PERIODS)).otherwise(None)


# ---- bake: the whole search -------------------------------------------------


def bake(panel, inputs: list[str], lookbacks=LOOKBACKS, hedge_lookbacks=HEDGE_LOOKBACKS,
         configs: pl.DataFrame | None = None, cost: float = COST, progress=None, bases=BASES) -> dict:
    """Walk every (input, signal lookback, hedge lookback, signal basis, config) for the
    panel's target.

    Returns ``trades`` (one row per trade, with its entry-state features),
    ``board`` (one row per setup, with its strategy stats) and ``skipped``
    (inputs without enough overlap, with the reason). Each model's halves
    split its own usable sample, so an input with a short history still gets
    two halves; ``split_date`` on the board says where."""
    configs = config_grid() if configs is None else configs
    target = panel.target.name
    data = panel.data
    trade_frames, stat_frames, skipped = [], [], []
    names = [i for i in dict.fromkeys(inputs) if i != target]
    total, k = len(names) * len(lookbacks) * len(hedge_lookbacks), 0
    for name, hedge_lookback in product(names, hedge_lookbacks):
        if name not in data.columns:
            skipped += [dict(input=name, lookback=lb, reason="not in panel") for lb in lookbacks]
            k += len(lookbacks)
            continue
        aligned = data.select("ts", target, name).drop_nulls()
        y = aligned[target].to_numpy().astype(float)
        x = aligned[name].to_numpy().astype(float)
        ts = aligned["ts"]
        hedge = 1.0 if hedgeable(name) else 0.0
        # every horizon's fit for this input first, so each trade can record what the other
        # horizons said at its entry (z_lb<L>): the multi-resolution view
        fits = {lb: model_arrays(y, x, lb, hedge_lookback) for lb in lookbacks if len(aligned) >= lb + MIN_BARS}
        skipped += [dict(input=name, lookback=lb, reason=f"only {len(aligned)} shared bars")
                    for lb in lookbacks if lb not in fits]
        k += len(lookbacks) - len(fits)
        for lookback, arrays in fits.items():
            if progress:
                progress(k, total, f"Baking {target} vs {name} · lookback {lookback} · hedge {hedge_lookback}")
            k += 1
            usable = np.flatnonzero(np.isfinite(arrays["entry_z"]))
            split = int(usable[0] + (len(y) - usable[0]) // 2) if len(usable) else len(y)
            # with no hedge the "portfolio" is the target alone, the same for every input: skip it
            for basis in [b for b in bases if hedge or b == "levels"]:
                _walk_basis(name, lookback, hedge_lookback, basis, y, x, ts, arrays, fits, lookbacks, configs, hedge,
                            cost, split, stat_frames, trade_frames)
    if progress:
        progress(total, total, f"Scoring {sum(len(f) for f in trade_frames):,} trades")
    trades = pl.concat(trade_frames) if trade_frames else pl.DataFrame()
    stats = pl.concat(stat_frames) if stat_frames else pl.DataFrame()
    return {"trades": trades, "board": board(trades, stats), "skipped": pl.DataFrame(skipped)}


def _walk_basis(name, lookback, hedge_lookback, basis, y, x, ts, arrays, fits, lookbacks, configs, hedge, cost, split,
                stat_frames, trade_frames) -> None:
    """One model on one signal basis: its setups' stats and its trades, appended.

    Each trade also records the same input's signal z, on the same basis and hedge,
    at every signal lookback of the bake (``z_lb<L>``; null where that horizon had
    too little history) -- what the other horizons said when this one fired."""
    raw, stats = walk_model(y, x, arrays, configs, hedge, cost, split, basis)
    stat_frames.append(pl.concat([
        pl.DataFrame({"input": [name] * len(configs), "lookback": [lookback] * len(configs),
                      "hedge_lb": [hedge_lookback] * len(configs), "basis": [basis] * len(configs),
                      "split_date": [ts[min(split, len(ts) - 1)]] * len(configs)}), configs,
        pl.DataFrame(stats, schema=list(_STATS), orient="row")], how="horizontal"))
    if len(raw):
        trade_frames.append(trade_frame(raw, name, lookback, hedge_lookback, basis, ts, arrays, fits, lookbacks,
                                        configs, hedge))


def trade_frame(raw, name, lookback, hedge_lookback, basis, ts, arrays, fits, lookbacks, configs,
                hedge) -> pl.DataFrame:
    """_walk's raw trade rows as a frame, with each trade's entry state and what the
    other horizons said at entry."""
    decided = raw[:, 1].astype(np.int64)
    filled = raw[:, 2].astype(np.int64)
    cfg = configs[raw[:, 0].astype(np.int64)]
    m = basis_arrays(arrays, basis)
    entry_state = {**arrays, "entry_z": m["z"], "resid_sigma": m["sigma"], "half_life": m["half_life"]}
    horizons = {f"z_lb{lb}": (basis_arrays(fits[lb], basis)["z"][decided].astype(np.float32) if lb in fits
                              else np.full(len(decided), np.nan, dtype=np.float32)) for lb in lookbacks}
    return pl.DataFrame({
        "input": name, "lookback": lookback, "hedge_lb": hedge_lookback, "basis": basis,
        **{c: cfg[c] for c in configs.columns},
        "entry_ts": ts.gather(decided + 1), "exit_ts": ts.gather(filled),
        "bars": filled - decided - 1, "side": raw[:, 3],
        "hedge_ratio": hedge * arrays["beta_changes"][decided], "pnl": raw[:, 4],
        "reason": [EXIT_REASONS[int(r)] for r in raw[:, 5]],
        "size": raw[:, 6].astype(np.float32), "exp_return": raw[:, 7].astype(np.float32),
        "exp_bars": raw[:, 8].astype(np.float32),
        **{f: entry_state[f][decided].astype(np.float32) for f in FEATURES},
        **horizons,
    })


def board(trades: pl.DataFrame, stats: pl.DataFrame) -> pl.DataFrame:
    """One row per setup with its strategy stats (see the module docstring).
    Setups that never traded are dropped."""
    if trades.is_empty() or stats.is_empty():
        return pl.DataFrame()
    trades = trades.filter(pl.col("reason") != "open")  # the board scores closed trades
    p = pl.col("pnl")
    per_trade = trades.group_by(SETUP).agg(
        pl.len().alias("n_trades"), p.mean().alias("avg_trade"), p.median().alias("median_trade"),
        (p > 0).mean().alias("hit_rate"), p.max().alias("best_trade"), p.min().alias("worst_trade"),
        (p.mean() / p.std() * pl.len().sqrt()).alias("trade_t"),
        # no losing trade leaves it undefined, not infinite
        pl.when((p < 0).any()).then(p.filter(p > 0).sum() / -p.filter(p < 0).sum()).otherwise(None)
        .alias("profit_factor"),
        pl.col("bars").mean().alias("avg_bars"), pl.col("hedge_ratio").mean().alias("avg_hedge_ratio"))
    years = pl.col("days") / PERIODS
    return (stats.join(per_trade, on=SETUP, how="inner").with_columns(
        _sharpe(pl.col("sum"), pl.col("sumsq"), pl.col("days")).alias("sharpe"),
        _sharpe(pl.col("sum1"), pl.col("sumsq1"), pl.col("days1")).alias("sharpe_h1"),
        _sharpe(pl.col("sum2"), pl.col("sumsq2"), pl.col("days2")).alias("sharpe_h2"),
        pl.col("sum").alias("total_pnl"),
        (pl.col("sum") / years).alias("pnl_per_year"),
        pl.col("max_dd").alias("max_drawdown"),
        (pl.col("n_trades") / years).alias("trades_per_year"),
        (pl.col("held") / pl.col("days")).alias("time_in_market"),
        (pl.col("best_trade") / pl.col("sum")).alias("best_trade_share"),
        pl.col("input").is_in(list(CATALOG)).or_(pl.col("input").is_in(list(TRADEABLE_LEGS))).alias("hedged"),
    ).with_columns(
        # null unless both halves have a Sharpe: one half alone is not consistency
        pl.when(pl.col("sharpe_h1").is_not_null() & pl.col("sharpe_h2").is_not_null())
        .then(pl.min_horizontal("sharpe_h1", "sharpe_h2")).otherwise(None).alias("worst_half_sharpe"),
        pl.when(pl.col("max_drawdown") < 0).then(pl.col("pnl_per_year") / -pl.col("max_drawdown"))
        .otherwise(None).alias("calmar"),
    ).drop(*_STATS))


def rank(board_df: pl.DataFrame, by: str = "sharpe", min_trades: int = 20, top: int | None = 50) -> pl.DataFrame:
    """Setups with at least ``min_trades``, best ``by`` first (a key of RANKS)."""
    if board_df.is_empty():
        return board_df
    out = (board_df.filter(pl.col("n_trades") >= min_trades).with_columns(pl.col(by).alias("score"))
           .sort("score", descending=True, nulls_last=True))
    return out if top is None else out.head(top)


def daily_marks(ts: pl.Series, y: np.ndarray, x: np.ndarray, trades: pl.DataFrame, cost: float = COST) -> np.ndarray:
    """One setup's daily P&L on the aligned dates ``ts``: each trade's held
    portfolio marked from its entry fill to its exit fill, the same marks the
    board's Sharpe is built from, at each trade's size (1 for bakes from before sizing)."""
    out = np.zeros(len(ts))
    index = {d: i for i, d in enumerate(ts.to_list())}
    dy, dx = np.diff(y, prepend=np.nan), np.diff(x, prepend=np.nan)
    for t in trades.iter_rows(named=True):
        i0, i1 = index[t["entry_ts"]], index[t["exit_ts"]]
        size = t.get("size") or 1.0
        out[i0 + 1:i1 + 1] += size * t["side"] * (dy[i0 + 1:i1 + 1] - t["hedge_ratio"] * dx[i0 + 1:i1 + 1])
        out[i1] -= size * cost
    return out


def characteristics(trades: pl.DataFrame, setup: dict) -> pl.DataFrame:
    """For one setup's trades: per entry-state feature, mean P&L and hit rate in
    each tercile of that feature. Which lookback states did the winners come from?"""
    t = trades.filter(pl.all_horizontal([pl.col(k) == v for k, v in setup.items() if k in SETUP]))
    rows = []
    for f in FEATURES:
        v = t.select(f, "pnl").drop_nulls().filter(pl.col(f).is_finite())
        if len(v) < 6:
            continue
        lo, hi = v[f].quantile(1 / 3), v[f].quantile(2 / 3)
        for label, part in (("low", v.filter(pl.col(f) <= lo)), ("mid", v.filter((pl.col(f) > lo) & (pl.col(f) <= hi))),
                            ("high", v.filter(pl.col(f) > hi))):
            rows.append(dict(feature=f, tercile=label, trades=len(part), range_lo=part[f].min(),
                             range_hi=part[f].max(), avg_trade=part["pnl"].mean(), hit_rate=(part["pnl"] > 0).mean()))
    return pl.DataFrame(rows)


# ---- saved runs and the app's job --------------------------------------------


def save(run: dict, target: str, inputs: list[str], settings: dict, root: Path = OVEN_RUNS) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = root / f"{stamp}_{target}"
    path.mkdir(parents=True, exist_ok=True)
    run["trades"].write_parquet(path / "trades.parquet")
    run["board"].write_parquet(path / "board.parquet")
    if not run["skipped"].is_empty():
        run["skipped"].write_parquet(path / "skipped.parquet")
    (path / "meta.json").write_text(json.dumps(dict(
        target=target, inputs=inputs, settings=settings, created_at=stamp,
        n_trades=len(run["trades"]), n_setups=len(run["board"])), indent=2), encoding="utf-8")
    return path


def load(path: Path) -> dict:
    path = Path(path)
    return dict(meta=json.loads((path / "meta.json").read_text(encoding="utf-8")),
                trades=pl.read_parquet(path / "trades.parquet"), board=pl.read_parquet(path / "board.parquet"),
                skipped=pl.read_parquet(path / "skipped.parquet") if (path / "skipped.parquet").is_file()
                else pl.DataFrame())


def list_runs(root: Path = OVEN_RUNS) -> list[Path]:
    """Saved bakes, newest first; bakes from before the board carried strategy stats are left out."""
    runs = []
    for p in sorted(root.glob("*"), reverse=True):
        if (p / "meta.json").is_file() and "sharpe" in pl.read_parquet_schema(p / "board.parquet"):
            runs.append(p)
    return runs


def oven_job(target: str, inputs: list[str], settings: dict) -> dict:
    """Background job: build the panel, bake, save. Returns the saved run's path."""
    from research import progress as work

    inputs = [i for i in dict.fromkeys(inputs) if i != target]

    def progress(done, total, message):
        work.update("jobs", "oven", message, done, total)

    progress(0, 0, f"Loading {target} and {len(inputs)} inputs")
    panel = build_panel(CATALOG[target], features=inputs, progress=lambda m: progress(0, 0, m))
    configs = config_grid(settings.get("z_entries", Z_ENTRIES), settings.get("raw_entries", RAW_ENTRIES),
                          settings.get("reverts", REVERTS), settings.get("max_holds", MAX_HOLDS),
                          settings.get("stops", STOPS), settings.get("hl_mults", HL_MULTS),
                          settings.get("pnl_stops", PNL_STOPS), settings.get("sizings", SIZING_DEFAULT))
    run = bake(panel, inputs, lookbacks=settings.get("lookbacks", LOOKBACKS),
               hedge_lookbacks=[int(h) for h in settings.get("hedge_lookbacks", HEDGE_LOOKBACKS)],
               configs=configs,
               cost=float(settings.get("cost", COST)), progress=progress,
               bases=tuple(settings.get("bases", BASES)))
    path = save(run, target, inputs, settings, Path(settings.get("runs_dir", OVEN_RUNS)))
    progress(1, 1, f"Completed · {len(run['board']):,} setups, {len(run['trades']):,} trades")
    return {"run_path": str(path), "summary": f"{len(run['board']):,} setups"}
