"""The backtest engine: the one trade walk every strategy runs on.

A strategy hands the engine a MODEL -- per-bar arrays it fitted itself -- and a
grid of CONFIGS (entry, exits, sizing, fill, re-arm, gate). The engine walks
each config through time, one position at a time, and returns its trades and
the daily P&L statistics it is scored on. ``walk`` is the fast path (numba,
every config of a model at once); ``walk_exact`` is the same rules written out
bar by bar in plain Python, the sense check a promoted strategy is held to,
trade for trade.

The model (all arrays aligned, length n):

    y, x          the target and the input, in bps
    signal_z      the live entry signal in z units ...
    signal_raw    ... and in bps; a config picks one by its scale
    alpha, beta   the fit's intercept and slope, so the residual is y - alpha - beta x
    center        a mean the residual reverts to (0 for a plain fit; an OU mean)
    sigma         the residual's scale (bps per z)
    half_life     the residual's half-life in bars
    hedge_beta    the hedge ratio a trade holds x at (y - hedge * hedge_beta x)
    port_vol      the held portfolio's daily vol, for risk-based sizing

The rules:

  Entry      on a crossing: the live signal moves beyond +/-entry while ARMED.
             A crossing is spent whether or not it is taken (gated out, no
             measured half-life, expected to lose under sizing). An entry needs
             a finite, positive half-life: without one the residual shows no
             reversion to trade (need_half_life; off only for older setups
             that ran without it, and then only a half-life stop needs one).
  Re-arm     after a crossing, the live signal must come back before another
             entry can fire (REARMS):
               threshold   back inside +/-entry
               half        back inside +/-entry/2
               mean        back through its mean (a new dislocation)
               by_exit     threshold after a revert or band exit; mean after a
                           stop, stop loss or time exit (the thesis failed)
             Re-arming reads the LIVE model, from the exit's decision bar on.
  Holding    the model is FROZEN at the entry decision: the residual is
             y - alpha[t] - beta[t] x - center[t] with entry-day values, so a
             trade exits only when prices move, never because the fit moved.
  Exits      checked each bar after entry, in this order (EXIT_REASONS):
               revert     the frozen residual recovers that share of its entry value
               band       it comes back within `band` of the mean (signal units)
               stop       it widens `stop` entry-sigmas past its entry value
               pnl_stop   the trade's P&L since its fill is down `pnl_stop` bps
               time       max_hold bars, or half_life at hl_mult x the entry half-life
  Fill       "same": decided and filled on the same close; "next": filled on
             the next bar's close. Applies to entries and exits alike.
  Gate       a boolean allow-mask per gate (gate_mask); a config's gate must
             allow the crossing bar. -1 = no gate.
  P&L        marked daily from the entry fill to the exit fill on the held
             portfolio y - ratio x (ratio = hedge * hedge_beta frozen at entry),
             times the trade's size; one round-trip cost at the exit. A trade
             still held at the last bar is reported as "open" and kept out of
             the statistics, which score closed trades.
  Sizing     (SIZINGS) from what the frozen model expects at entry: an OU
             recovery with the entry half-life by the expected exit, net of cost.
             Sizes are scaled to average 1 per config.
  Episodes   trades are grouped into dislocations: a trade starts a new episode
             unless the previous one was on the same side and the live signal
             never crossed its mean in between. Re-entering one dislocation adds
             trades, not episodes.
"""

from __future__ import annotations

import os

import numba
import numpy as np
import polars as pl
from numba import njit, prange

from backtest.lab import gate_allow_mask, parse_gate

SCALES = ("z", "raw")
FILLS = ("same", "next")
REARMS = ("threshold", "half", "mean", "by_exit")
SIZINGS = ("flat", "exp_return", "exp_sharpe", "bps_per_day")
# "open": still held at the last bar
EXIT_REASONS = ("revert", "stop", "time", "half_life", "open", "pnl_stop", "band")
GATE_MODES = ("full", "causal")  # raw edges from full-sample quantiles, or point-in-time percentiles
PERIODS = 252

# the config columns walk reads, with the value a missing column means
CONFIG_DEFAULTS = {
    "scale": "z", "entry": None, "revert": 0.0, "band": np.nan, "max_hold": None, "stop": 0.0, "hl_mult": 0.0,
    "pnl_stop": 0.0, "sizing": "flat", "fill": "same", "rearm": "threshold", "gate": -1, "need_half_life": True,
}
MODEL_KEYS = ("y", "x", "signal_z", "signal_raw", "alpha", "beta", "center", "sigma", "half_life", "hedge_beta",
              "port_vol")

# A walk of at least PARALLEL_CONFIGS configs (a bake's grid) runs its configs across
# WALK_THREADS threads: half the machine's logical cores, so the dashboard, the apps and
# whatever else is running keep the rest. Smaller walks (a setup re-walked for a page)
# stay on one thread, where starting threads would cost more than it saves.
WALK_THREADS = max(1, min(numba.config.NUMBA_NUM_THREADS, (os.cpu_count() or 2) // 2))
PARALLEL_CONFIGS = 256

# one row per trade
TRADE_COLS = ("config", "decided", "entry_fill", "exit_fill", "side", "pnl", "reason", "size", "exp_return",
              "exp_bars", "hedge_ratio", "new_episode")
# per config: daily-mark sufficient statistics (whole sample and each half), then closed-trade aggregates
STATS = ("sum", "sumsq", "days", "max_dd", "held", "sum1", "sumsq1", "days1", "sum2", "sumsq2", "days2",
         "n_trades", "trade_sum", "trade_sumsq", "wins", "gross_win", "gross_loss", "best_trade", "worst_trade",
         "bars_sum", "hedge_sum", "episodes")

_REVERT, _STOP, _TIME, _HALF_LIFE, _OPEN, _PNL_STOP, _BAND = range(7)


# ---- shared arithmetic ------------------------------------------------------


@njit(cache=True)
def horizon(hold, hl_mult, half_life):
    """(bars to the time stop, its exit reason): max_hold, or sooner at hl_mult x the
    half-life measured at entry."""
    bars, reason = hold, _TIME
    if hl_mult > 0:
        by_half_life = max(1, int(np.ceil(hl_mult * half_life)))
        if by_half_life < bars:
            bars, reason = by_half_life, _HALF_LIFE
    return bars, reason


@njit(cache=True)
def expect(abs_r0, hl, revert, bars, cost):
    """(exp_return, exp_bars) the frozen model expects of an entry residual |r0|: OU
    recovery with half-life hl by the expected exit, net of cost. A revert target is
    reached after hl log2(1 / (1 - revert)) bars; all the way back (revert 1) never
    quite arrives, so it counts 90% of the way. Capped at the time stop."""
    exp_bars = float(bars)
    exp_return = np.nan
    if np.isfinite(hl) and hl > 0:
        if 0.0 < revert < 1.0:
            exp_bars = min(exp_bars, hl * np.log(1.0 / (1.0 - revert)) / np.log(2.0))
        elif revert >= 1.0:
            exp_bars = min(exp_bars, hl * np.log(10.0) / np.log(2.0))
        exp_bars = max(exp_bars, 1.0)
        exp_return = abs_r0 * (1.0 - 2.0 ** (-exp_bars / hl)) - cost
    return exp_return, exp_bars


@njit(cache=True)
def _size(sizing, exp_return, exp_bars, port_vol):
    if sizing == 1:
        return exp_return
    if sizing == 2:
        return exp_return / (port_vol * np.sqrt(exp_bars)) if port_vol > 0 else np.nan
    if sizing == 3:
        return exp_return / exp_bars
    return 1.0


@njit(cache=True)
def _armed(rule, s, thr, last_sign):
    """Has the live signal come back far enough to allow the next crossing?"""
    if rule == 1:
        return abs(s) <= thr / 2.0
    if rule == 2:
        return last_sign == 0.0 or last_sign * s <= 0.0
    return abs(s) <= thr


@njit(cache=True)
def _rule_after(rearm, reason):
    """The re-arm rule in force after an exit (0 threshold, 1 half, 2 mean)."""
    if rearm == 3:  # by_exit
        return 0 if (reason == _REVERT or reason == _BAND) else 2
    return rearm


# ---- the fast walk -----------------------------------------------------------


@njit(cache=True)
def _walk_one(c, y, x, alpha, beta, center, sigma, half_life, port_vol, hedge_beta, signals, gates, hedge,
              scale, entry, revert, band, hold, stop, hl_mult, pnl_stop, sizing, fill, rearm, gate, need_hl,
              cost, split, keep, first, room, trades, stats):
    """Config ``c``: its STATS row, and -- if kept -- its trades (rows of TRADE_COLS) written
    from row ``first``, at most ``room`` of them. Returns how many it wrote, or -1 if it
    had more than ``room``. Touches only its own rows, so configs can run in parallel."""
    n = y.shape[0]
    count = first
    if True:  # the body of one config's walk
        s = signals[scale[c]]
        thr = entry[c]
        f = fill[c]
        last = n - 1 - f  # the last bar a decision can still be filled on
        start = 1
        while start < n and not np.isfinite(s[start]):
            start += 1
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
        n_trades = 0
        t_sum = 0.0
        t_sq = 0.0
        wins = 0
        gross_win = 0.0
        gross_loss = 0.0
        best = -np.inf
        worst = np.inf
        bars_sum = 0.0
        hedge_sum = 0.0
        episodes = 0
        size_sum = 0.0
        size_n = 0
        traded = False
        armed = False
        rule = 0  # the first entry needs the signal inside the threshold first
        last_sign = 0.0
        last_side = 0.0
        crossed_mean = True
        t = start
        while t <= last:
            st = s[t]
            if not np.isfinite(st):
                t += 1
                continue
            if last_sign != 0.0 and last_sign * st <= 0.0:
                crossed_mean = True
            if not armed:
                armed = _armed(rule, st, thr, last_sign)
            if not (armed and abs(st) > thr):
                t += 1
                continue
            armed = False  # the crossing is spent, taken or not
            rule = 1 if rearm[c] == 1 else 0  # one not taken: the next needs a fresh crossing
            if not (np.isfinite(alpha[t]) and np.isfinite(beta[t]) and np.isfinite(center[t])
                    and np.isfinite(hedge_beta[t]) and sigma[t] > 0):
                t += 1
                continue
            if gate[c] >= 0 and not gates[gate[c], t]:
                t += 1
                continue
            if (need_hl[c] or hl_mult[c] > 0) and not (np.isfinite(half_life[t]) and half_life[t] > 0):
                t += 1
                continue
            r0 = y[t] - alpha[t] - beta[t] * x[t] - center[t]
            sign0 = 1.0 if r0 > 0 else -1.0
            bars, timeout = horizon(hold[c], hl_mult[c], half_life[t])
            exp_return, exp_bars = expect(abs(r0), half_life[t], revert[c], bars, cost)
            size = _size(sizing[c], exp_return, exp_bars, port_vol[t])
            if not (np.isfinite(size) and size > 0):
                t += 1
                continue
            goal = (1.0 - revert[c]) * abs(r0)
            band_level = band[c] * (sigma[t] if scale[c] == 0 else 1.0)
            limit = abs(r0) + stop[c] * sigma[t]
            side = -sign0
            ratio = hedge * hedge_beta[t]
            f0 = t + f
            new_episode = (not traded) or crossed_mean or side != last_side
            traded = True
            size_sum += size
            size_n += 1
            last_sign = 1.0 if st > 0 else -1.0
            last_side = side
            crossed_mean = False
            exit_u = -1
            reason = timeout
            for k in range(1, bars + 1):
                u = t + k
                if u > last:
                    break
                if np.isfinite(s[u]) and last_sign * s[u] <= 0.0:
                    crossed_mean = True
                frozen = sign0 * (y[u] - alpha[t] - beta[t] * x[u] - center[t])
                if revert[c] > 0 and frozen <= goal:
                    exit_u, reason = u, _REVERT
                    break
                if np.isfinite(band_level) and frozen <= band_level:
                    exit_u, reason = u, _BAND
                    break
                if stop[c] > 0 and frozen >= limit:
                    exit_u, reason = u, _STOP
                    break
                if pnl_stop[c] > 0 and side * ((y[u] - y[f0]) - ratio * (x[u] - x[f0])) <= -pnl_stop[c]:
                    exit_u, reason = u, _PNL_STOP
                    break
                if k == bars:
                    exit_u, reason = u, timeout
            still_open = exit_u < 0
            if still_open:
                exit_u, reason = last, _OPEN
            f1 = exit_u + f
            pnl = 0.0
            for d in range(f0 + 1, f1 + 1):
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
            if not still_open:
                n_trades += 1
                t_sum += pnl
                t_sq += pnl * pnl
                if pnl > 0:
                    wins += 1
                    gross_win += pnl
                elif pnl < 0:
                    gross_loss -= pnl
                best = max(best, pnl)
                worst = min(worst, pnl)
                bars_sum += f1 - f0
                hedge_sum += ratio
                if new_episode:
                    episodes += 1
            if keep[c]:
                if count - first >= room:
                    return -1  # more trades than the room set aside for it: walk() raises
                trades[count, 0] = c
                trades[count, 1] = t
                trades[count, 2] = f0
                trades[count, 3] = f1
                trades[count, 4] = side
                trades[count, 5] = pnl
                trades[count, 6] = reason
                trades[count, 7] = size
                trades[count, 8] = exp_return
                trades[count, 9] = exp_bars
                trades[count, 10] = ratio
                trades[count, 11] = 1.0 if new_episode else 0.0
                count += 1
            if still_open:
                break
            # re-arming reads the live signal from the exit's decision bar on
            rule = _rule_after(rearm[c], reason)
            t = exit_u
        # sizes average 1 over the config's trades: Sharpe is unchanged, P&L reads like flat sizing
        norm = size_sum / size_n if size_n else 1.0
        stats[c, 0] = total / norm
        stats[c, 1] = total_sq / (norm * norm)
        days = max(n - 1 - start, 0)
        days1 = min(max(split - start - 1, 0), days)
        stats[c, 2] = days
        stats[c, 3] = dd / norm
        stats[c, 4] = held
        stats[c, 5] = s1 / norm
        stats[c, 6] = ss1 / (norm * norm)
        stats[c, 7] = days1
        stats[c, 8] = s2 / norm
        stats[c, 9] = ss2 / (norm * norm)
        stats[c, 10] = days - days1
        stats[c, 11] = n_trades
        stats[c, 12] = t_sum / norm
        stats[c, 13] = t_sq / (norm * norm)
        stats[c, 14] = wins
        stats[c, 15] = gross_win / norm
        stats[c, 16] = gross_loss / norm
        stats[c, 17] = best / norm if n_trades else np.nan
        stats[c, 18] = worst / norm if n_trades else np.nan
        stats[c, 19] = bars_sum
        stats[c, 20] = hedge_sum
        stats[c, 21] = episodes
        if keep[c] and count > first:
            trades[first:count, 5] /= norm
            trades[first:count, 7] /= norm
    return count - first


@njit(cache=True)
def _walk_serial(y, x, alpha, beta, center, sigma, half_life, port_vol, hedge_beta, signals, gates, hedge,
                 scale, entry, revert, band, hold, stop, hl_mult, pnl_stop, sizing, fill, rearm, gate, need_hl,
                 cost, split, keep, firsts, rooms, written, trades, stats):
    """Every config one after another (small walks: no threads to start)."""
    for c in range(scale.shape[0]):
        written[c] = _walk_one(c, y, x, alpha, beta, center, sigma, half_life, port_vol, hedge_beta, signals,
                               gates, hedge, scale, entry, revert, band, hold, stop, hl_mult, pnl_stop, sizing,
                               fill, rearm, gate, need_hl, cost, split, keep, firsts[c], rooms[c], trades, stats)


@njit(cache=True, parallel=True)
def _walk_parallel(y, x, alpha, beta, center, sigma, half_life, port_vol, hedge_beta, signals, gates, hedge,
                   scale, entry, revert, band, hold, stop, hl_mult, pnl_stop, sizing, fill, rearm, gate, need_hl,
                   cost, split, keep, firsts, rooms, written, trades, stats):
    """Every config across WALK_THREADS threads: each writes only its own rows."""
    for c in prange(scale.shape[0]):
        written[c] = _walk_one(c, y, x, alpha, beta, center, sigma, half_life, port_vol, hedge_beta, signals,
                               gates, hedge, scale, entry, revert, band, hold, stop, hl_mult, pnl_stop, sizing,
                               fill, rearm, gate, need_hl, cost, split, keep, firsts[c], rooms[c], trades, stats)


# ---- configs and the fast path ----------------------------------------------


def configs_frame(configs: pl.DataFrame | dict | list[dict]) -> pl.DataFrame:
    """Configs with every CONFIG_DEFAULTS column, missing ones at their defaults
    (entry and max_hold have none and are required)."""
    frame = configs if isinstance(configs, pl.DataFrame) else pl.DataFrame(
        configs if isinstance(configs, list) else [configs])
    for col, default in CONFIG_DEFAULTS.items():
        if col not in frame.columns:
            if default is None:
                raise ValueError(f"configs need a {col!r} column")
            frame = frame.with_columns(pl.lit(default).alias(col))
    for col, allowed in (("scale", SCALES), ("fill", FILLS), ("rearm", REARMS), ("sizing", SIZINGS)):
        unknown = set(frame[col].unique().to_list()) - set(allowed)
        if unknown:
            raise ValueError(f"unknown {col} {sorted(unknown)}; expected some of {allowed}")
    return frame


def _crossing_runs(s: np.ndarray, thr: float) -> int:
    """How many separate runs the signal spends beyond +/-thr: at most one entry per run,
    since re-arming needs the signal back inside (or through its mean: a new run)."""
    side = np.where(np.isfinite(s), np.sign(s) * (np.abs(s) > thr), 0)
    return int(((side != 0) & (side != np.r_[0, side[:-1]])).sum())


def walk(model: dict, configs: pl.DataFrame, hedge: float = 1.0, cost: float = 0.0, split: int | None = None,
         gates: np.ndarray | None = None, keep=True) -> tuple[np.ndarray, np.ndarray]:
    """Walk every config through one model: (trades, stats), rows of TRADE_COLS and
    STATS. ``gates`` is a (n_gates, n) bool array the configs' ``gate`` column indexes;
    ``keep`` (a bool, or one per config) says whose trades to return -- every config's
    stats come back either way."""
    configs = configs_frame(configs)
    arr = {k: np.ascontiguousarray(model[k], dtype=np.float64) for k in MODEL_KEYS}
    n = len(arr["y"])
    signals = np.ascontiguousarray(np.vstack([arr["signal_z"], arr["signal_raw"]]))
    # labels to their indices, vectorised: a bake's grid is millions of configs
    index = lambda col, names: configs[col].replace_strict({v: i for i, v in enumerate(names)},  # noqa: E731
                                                            return_dtype=pl.Int64).to_numpy()
    scale = index("scale", SCALES)
    entry = configs["entry"].to_numpy().astype(np.float64)
    keep = np.full(len(configs), bool(keep)) if np.isscalar(keep) else np.asarray(keep, dtype=bool)
    # room for every kept config's trades: at most one per run beyond its threshold, plus one;
    # each config gets its own block of rows (firsts, rooms) so configs can walk in parallel
    rooms = np.zeros(len(configs), dtype=np.int64)
    if keep.any():
        pairs, which = np.unique(np.c_[scale, entry], axis=0, return_inverse=True)
        runs = np.array([_crossing_runs(signals[int(sc)], e) for sc, e in pairs], dtype=np.int64)
        rooms = np.where(keep, runs[which.ravel()] + 1, 0).astype(np.int64)
    firsts = np.r_[0, np.cumsum(rooms)[:-1]].astype(np.int64)
    written = np.zeros(len(configs), dtype=np.int64)
    trades = np.empty((int(rooms.sum()), len(TRADE_COLS)))
    stats = np.zeros((len(configs), len(STATS)))
    gates = np.zeros((1, n), dtype=np.bool_) if gates is None else np.ascontiguousarray(gates, dtype=np.bool_)
    args = (arr["y"], arr["x"], arr["alpha"], arr["beta"], arr["center"], arr["sigma"], arr["half_life"],
            arr["port_vol"], arr["hedge_beta"], signals, gates, float(hedge), scale, entry,
            configs["revert"].to_numpy().astype(np.float64), configs["band"].to_numpy().astype(np.float64),
            configs["max_hold"].to_numpy().astype(np.int64), configs["stop"].to_numpy().astype(np.float64),
            configs["hl_mult"].to_numpy().astype(np.float64), configs["pnl_stop"].to_numpy().astype(np.float64),
            index("sizing", SIZINGS), index("fill", FILLS), index("rearm", REARMS),
            configs["gate"].to_numpy().astype(np.int64), configs["need_half_life"].to_numpy().astype(np.bool_),
            float(cost), int(n if split is None else split), keep, firsts, rooms, written, trades, stats)
    if len(configs) >= PARALLEL_CONFIGS and WALK_THREADS > 1:
        numba.set_num_threads(WALK_THREADS)
        _walk_parallel(*args)
    else:
        _walk_serial(*args)
    if (written < 0).any():
        raise RuntimeError("a config produced more trades than the walk set aside room for")
    # each config's trades, in config order: the same rows, in the same order, as walking them one by one
    used = np.repeat(firsts, written) + (np.arange(written.sum()) - np.repeat(np.cumsum(written) - written, written))
    return trades[used], stats


# ---- the exact walk: the same rules, bar by bar ------------------------------


def walk_exact(model: dict, config: dict, hedge: float = 1.0, cost: float = 0.0,
               gate: np.ndarray | None = None) -> tuple[list[dict], np.ndarray]:
    """One config walked one bar at a time in plain Python: (trades, daily marks).

    The sense check for ``walk``. It shares only the expectation arithmetic with the
    fast path -- it is a bar-by-bar state machine where ``walk`` jumps trade to
    trade -- so the two agreeing trade for trade is evidence both implement the
    module's rules. Trades are dicts with TRADE_COLS keys (``reason`` by name);
    marks are the closed trades' daily P&L, which every statistic is built from."""
    cfg = configs_frame(config).row(0, named=True)
    m = {k: np.asarray(model[k], dtype=float) for k in MODEL_KEYS}
    y, x, n = m["y"], m["x"], len(m["y"])
    s = m["signal_z"] if cfg["scale"] == "z" else m["signal_raw"]
    thr, f = cfg["entry"], FILLS.index(cfg["fill"])
    last = n - 1 - f
    rearm = REARMS.index(cfg["rearm"])
    allow = np.ones(n, dtype=bool) if gate is None else np.asarray(gate, dtype=bool)

    trades, pos = [], None  # pos: the trade being held, until its exit is decided
    armed, rule = False, 0  # rule: 0 back inside entry, 1 inside half of it, 2 through the mean
    last_sign, last_side, crossed_mean = 0.0, 0.0, True
    first = 1
    while first < n and not np.isfinite(s[first]):
        first += 1
    for t in range(first, last + 1):
        st = s[t]
        live = np.isfinite(st)
        if live and last_sign != 0.0 and last_sign * st <= 0.0:
            crossed_mean = True

        if pos is not None:  # holding: is today the exit?
            frozen = pos["sign0"] * (y[t] - pos["alpha"] - pos["beta"] * x[t] - pos["center"])
            f0 = pos["entry_fill"]
            pnl_now = pos["side"] * ((y[t] - y[f0]) - pos["hedge_ratio"] * (x[t] - x[f0]))
            reason = None
            if cfg["revert"] > 0 and frozen <= pos["goal"]:
                reason = "revert"
            elif np.isfinite(pos["band"]) and frozen <= pos["band"]:
                reason = "band"
            elif cfg["stop"] > 0 and frozen >= pos["limit"]:
                reason = "stop"
            elif cfg["pnl_stop"] > 0 and pnl_now <= -cfg["pnl_stop"]:
                reason = "pnl_stop"
            elif t - pos["decided"] == pos["bars"]:
                reason = pos["timeout"]
            if reason is None:
                continue
            pos.update(exit_fill=t + f, reason=reason)
            trades.append(pos)
            pos = None
            rule = _rule_after(rearm, EXIT_REASONS.index(reason))
            armed = False

        if not live:
            continue
        if not armed:
            armed = bool(_armed(rule, st, thr, last_sign))
        if not (armed and abs(st) > thr):
            continue
        armed, rule = False, (1 if rearm == 1 else 0)
        if not (np.isfinite(m["alpha"][t]) and np.isfinite(m["beta"][t]) and np.isfinite(m["center"][t])
                and np.isfinite(m["hedge_beta"][t]) and m["sigma"][t] > 0):
            continue
        if cfg["gate"] >= 0 and not allow[t]:
            continue
        hl = m["half_life"][t]
        if (cfg["need_half_life"] or cfg["hl_mult"] > 0) and not (np.isfinite(hl) and hl > 0):
            continue
        r0 = y[t] - m["alpha"][t] - m["beta"][t] * x[t] - m["center"][t]
        bars, timeout = horizon(int(cfg["max_hold"]), float(cfg["hl_mult"]), hl)
        exp_return, exp_bars = expect(abs(r0), hl, float(cfg["revert"]), bars, float(cost))
        size = _size(SIZINGS.index(cfg["sizing"]), exp_return, exp_bars, m["port_vol"][t])
        if not (np.isfinite(size) and size > 0):
            continue
        sign0 = 1.0 if r0 > 0 else -1.0
        side = -sign0
        new_episode = not trades or crossed_mean or side != last_side
        last_sign, last_side, crossed_mean = (1.0 if st > 0 else -1.0), side, False
        band = cfg["band"] * (m["sigma"][t] if cfg["scale"] == "z" else 1.0)
        pos = dict(config=0, decided=t, entry_fill=t + f, side=side, size=size, exp_return=exp_return,
                   exp_bars=exp_bars, hedge_ratio=hedge * m["hedge_beta"][t], new_episode=new_episode,
                   alpha=m["alpha"][t], beta=m["beta"][t], center=m["center"][t], sign0=sign0,
                   goal=(1.0 - cfg["revert"]) * abs(r0), limit=abs(r0) + cfg["stop"] * m["sigma"][t],
                   band=band, bars=bars, timeout=EXIT_REASONS[timeout])
    if pos is not None:  # still held: marked to the last bar, kept out of the statistics
        pos.update(exit_fill=n - 1, reason="open")
        trades.append(pos)

    dy, dx = np.diff(y, prepend=np.nan), np.diff(x, prepend=np.nan)
    marks = np.zeros(n)
    for tr in trades:
        span = np.arange(tr["entry_fill"] + 1, tr["exit_fill"] + 1)
        daily = tr["size"] * tr["side"] * (dy[span] - tr["hedge_ratio"] * dx[span])
        if tr["reason"] != "open":
            daily[-1] -= tr["size"] * cost
            marks[span] += daily
        tr["pnl"] = float(daily.sum())
    norm = float(np.mean([tr["size"] for tr in trades])) if trades else 1.0
    marks /= norm
    out = []
    for tr in trades:
        tr["pnl"] /= norm
        tr["size"] /= norm
        out.append({k: v for k, v in tr.items() if k in TRADE_COLS})
    return out, marks


def exact_stats(trades: list[dict], marks: np.ndarray, start: int, split: int | None = None) -> dict:
    """STATS from walk_exact's output, computed the plain way: what ``walk`` reports
    for the same config. ``start`` is the first bar the signal exists."""
    n = len(marks)
    split = n if split is None else split
    scored = marks[start + 1:]
    first = marks[start + 1:max(split, start + 1)]
    second = marks[max(split, start + 1):]
    cum = np.cumsum(scored)
    closed = [t for t in trades if t["reason"] != "open"]
    pnl = np.array([t["pnl"] for t in closed])
    days = max(n - 1 - start, 0)
    days1 = min(max(split - start - 1, 0), days)
    held = sum(t["exit_fill"] - t["entry_fill"] for t in closed)
    return dict(
        sum=scored.sum(), sumsq=(scored ** 2).sum(), days=days,
        max_dd=float(min(0.0, (cum - np.maximum(np.maximum.accumulate(cum), 0.0)).min())) if len(cum) else 0.0,
        held=held, sum1=first.sum(), sumsq1=(first ** 2).sum(), days1=days1,
        sum2=second.sum(), sumsq2=(second ** 2).sum(), days2=days - days1,
        n_trades=len(closed), trade_sum=pnl.sum(), trade_sumsq=(pnl ** 2).sum(), wins=int((pnl > 0).sum()),
        gross_win=pnl[pnl > 0].sum(), gross_loss=-pnl[pnl < 0].sum(),
        best_trade=pnl.max() if len(pnl) else np.nan, worst_trade=pnl.min() if len(pnl) else np.nan,
        bars_sum=held, hedge_sum=sum(t["hedge_ratio"] for t in closed),
        episodes=sum(bool(t["new_episode"]) for t in closed),
    )


# ---- scoring ----------------------------------------------------------------


def _sharpe(total: pl.Expr, total_sq: pl.Expr, days: pl.Expr) -> pl.Expr:
    """Annualised Sharpe of daily marks from their sum, sum of squares and count."""
    mean = total / days
    var = total_sq / days - mean ** 2
    return pl.when((days > 1) & (var > 1e-12)).then(mean / var.sqrt() * np.sqrt(PERIODS)).otherwise(None)


def score(stats: np.ndarray | pl.DataFrame) -> pl.DataFrame:
    """Each config's strategy statistics from walk's STATS, one row per config:

      sharpe, sharpe_h1/h2, worst_half_sharpe   daily mark-to-market, flat days count as zero
      total_pnl, pnl_per_year, max_drawdown, calmar, time_in_market
      n_trades, trades_per_year, n_episodes, hit_rate, avg_trade, best/worst_trade,
      best_trade_share, trade_t, profit_factor, avg_bars, avg_hedge_ratio

    Closed trades only. (A median needs the trades themselves, so it isn't here.)"""
    s = stats if isinstance(stats, pl.DataFrame) else pl.DataFrame(stats, schema=list(STATS), orient="row")
    n = pl.col("n_trades")
    years = pl.col("days") / PERIODS
    mean = pl.col("trade_sum") / n
    var = (pl.col("trade_sumsq") - n * mean ** 2) / (n - 1)
    return s.select(
        _sharpe(pl.col("sum"), pl.col("sumsq"), pl.col("days")).alias("sharpe"),
        _sharpe(pl.col("sum1"), pl.col("sumsq1"), pl.col("days1")).alias("sharpe_h1"),
        _sharpe(pl.col("sum2"), pl.col("sumsq2"), pl.col("days2")).alias("sharpe_h2"),
        pl.col("sum").alias("total_pnl"),
        (pl.col("sum") / years).alias("pnl_per_year"),
        pl.col("max_dd").alias("max_drawdown"),
        n.cast(pl.Int64).alias("n_trades"),
        (n / years).alias("trades_per_year"),
        pl.col("episodes").cast(pl.Int64).alias("n_episodes"),
        (pl.col("wins") / n).alias("hit_rate"),
        mean.alias("avg_trade"),
        pl.col("best_trade"), pl.col("worst_trade"),
        (pl.col("best_trade") / pl.col("sum")).alias("best_trade_share"),
        pl.when((n > 1) & (var > 0)).then(mean / var.sqrt() * n.sqrt()).otherwise(None).alias("trade_t"),
        # no losing trade leaves it undefined, not infinite
        pl.when(pl.col("gross_loss") > 0).then(pl.col("gross_win") / pl.col("gross_loss")).otherwise(None)
        .alias("profit_factor"),
        (pl.col("bars_sum") / n).alias("avg_bars"),
        (pl.col("hedge_sum") / n).alias("avg_hedge_ratio"),
        (pl.col("held") / pl.col("days")).alias("time_in_market"),
    ).with_columns(
        # null unless both halves have a Sharpe: one half alone is not consistency
        pl.when(pl.col("sharpe_h1").is_not_null() & pl.col("sharpe_h2").is_not_null())
        .then(pl.min_horizontal("sharpe_h1", "sharpe_h2")).otherwise(None).alias("worst_half_sharpe"),
        pl.when(pl.col("max_drawdown") < 0).then(pl.col("pnl_per_year") / -pl.col("max_drawdown"))
        .otherwise(None).alias("calmar"),
    )


# ---- gates ------------------------------------------------------------------


def condition_arrays(beta, r2, resid, lookback: int) -> dict[str, np.ndarray]:
    """The gate conditions of one fit (backtest.lab.CONDITION_NAMES), each aligned to its
    inputs: the fit's quality and stability (r2, beta_cv, beta), their 20-bar vol and
    10-bar momentum, and the residual's AR(1) state, half-life, vol and momentum. Every
    strategy's gates read these, whatever model produced the fit."""
    from stats.diagnostics import beta_cv
    from stats.ou import roll_ou_features

    beta, r2, resid = (pl.Series(v, dtype=pl.Float64) if not isinstance(v, pl.Series) else v.cast(pl.Float64)
                       for v in (beta, r2, resid))
    beta, r2, resid = (v.fill_nan(None) for v in (beta, r2, resid))
    ou = roll_ou_features(resid, lookback=lookback)
    conds = {
        "r2": r2,
        "beta_cv": beta_cv(beta, lookback=lookback),
        "beta": beta,
        "beta_vol20": beta.diff().rolling_std(20),
        "beta_mom10": beta.diff(10),
        "r2_vol20": r2.diff().rolling_std(20),
        "r2_mom10": r2.diff(10),
        "resid_phi": ou["ou_rho"] - 1.0,
        "resid_half_life": ou["half_life"],
        "resid_vol20": resid.diff().rolling_std(20),
        "resid_mom10": resid.diff(10),
    }
    return {k: pl.Series(v).cast(pl.Float64).to_numpy().astype(float) for k, v in conds.items()}


def gate_edges(values: np.ndarray, spec) -> tuple[str, tuple[float, ...]]:
    """A gate's bucket as raw values of its condition: (kind, edges), the bucket's
    quantiles over the whole sample. Frozen with a promoted signal, so live trading
    applies the same raw cut the backtest did."""
    _, kind, qs = parse_gate(spec)
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return kind, tuple(float(np.quantile(v, q)) for q in qs)


def mask_from_edges(values: np.ndarray, kind: str, edges: tuple[float, ...]) -> np.ndarray:
    """Entry-allow mask of a raw bucket; a missing value is never allowed."""
    v = np.asarray(values, dtype=float)
    finite = np.isfinite(v)
    with np.errstate(invalid="ignore"):
        if kind == "below":
            return finite & (v <= edges[0])
        if kind == "above":
            return finite & (v >= edges[0])
        if kind == "between":
            return finite & (v > edges[0]) & (v < edges[1])
        return finite & ((v <= edges[0]) | (v >= edges[1]))


def gate_mask(values: np.ndarray, spec, mode: str = "full", min_history: int = 252,
              window: int | None = None) -> np.ndarray:
    """Entry-allow mask of one gate (a backtest.lab gate spec, e.g. ("r2", "high_75")):

      full     raw edges at the whole sample's quantiles (gate_edges): one fixed cut
               for every bar, placed with hindsight
      causal   each bar ranked only against the history up to it (point in time, no
               look-ahead): at least ``min_history`` bars, over a trailing ``window``
               if given
    """
    if mode == "full":
        return mask_from_edges(values, *gate_edges(values, spec))
    if mode == "causal":
        return gate_allow_mask(np.asarray(values, dtype=float), spec, min_history=min_history, window=window)
    raise ValueError(f"unknown gate mode {mode!r}; expected one of {GATE_MODES}")
