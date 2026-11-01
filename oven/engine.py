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
from backtest import engine as core
from backtest.engine import EXIT_REASONS, FILLS, REARMS, SCALES, SIZINGS, STATS, TRADE_COLS, expect, horizon
from backtest.lab import CONDITION_NAMES, REGIME_GATE_BUCKETS
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
SIZING_DEFAULT = ["flat"]
COST = 0.0  # bps per round trip
MIN_BARS = 500  # an input needs this much overlap with the target beyond its lookback
PERIODS = 252

FEATURES = ("entry_z", "r2", "beta_levels", "beta_changes", "hedge_gap", "beta_drift20",
            "half_life", "resid_sigma", "input_vol60", "target_vol60")
SETUP = ["input", "lookback", "hedge_lb", "basis", "scale", "entry", "revert", "max_hold", "stop", "hl_mult",
         "pnl_stop", "sizing", "fill", "rearm", "gate", "gate_mode", "need_half_life"]
# fills, re-arms and gates (backtest.engine): a gate is "<condition>:<bucket>" or none
FILLS_DEFAULT = ["same"]
REARMS_DEFAULT = ["by_exit"]
GATE_CONDITIONS = CONDITION_NAMES
GATE_BUCKETS = tuple(bucket[0] for bucket in REGIME_GATE_BUCKETS)
GATE_BUCKETS_DEFAULT = ["low_25", "high_75", "mid_25_75"]
NO_GATE = "none"
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
# what a setup from before these were knobs ran on (older bakes, setups promoted from them):
# filled a bar after each decision, re-armed back inside the entry band, no gate, and could
# enter with no measured half-life
LEGACY_RULES = {"fill": "next", "rearm": "threshold", "gate": NO_GATE, "gate_mode": "full", "need_half_life": False}


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
                sizings=SIZING_DEFAULT, fills=FILLS_DEFAULT, rearms=REARMS_DEFAULT, gates=(NO_GATE,),
                gate_mode: str = "full") -> pl.DataFrame:
    """Every (scale, entry, revert, hold, stop, half-life multiple, P&L stop, sizing, fill,
    re-arm, gate) cell a model is walked through. A cell with no reversion exit and no
    half-life stop is a pure max-hold exit. ``gates`` are labels (gate_label) or none;
    ``gate_mode`` (backtest.engine.GATE_MODES) is the bake's, the same for every gate."""
    for name, values, allowed in (("sizing", sizings, SIZINGS), ("fill", fills, FILLS), ("rearm", rearms, REARMS)):
        unknown = [v for v in values if v not in allowed]
        if unknown:
            raise ValueError(f"unknown {name} {unknown}; expected some of {allowed}")
    rows = [("z", e) for e in z_entries] + [("raw", e) for e in raw_entries]
    # cast every value: the app sends whole numbers as ints, and a row-built frame typed
    # from its first rows would otherwise truncate a later 2.5 to 2
    cells = [(s, float(e), float(r), int(h), float(st), float(m), float(p), z, f, ra, g, gate_mode, True)
             for (s, e), r, h, st, m, p, z, f, ra, g in product(rows, reverts, max_holds, stops, hl_mults, pnl_stops,
                                                                sizings, fills, rearms, gates)]
    return pl.DataFrame(cells, schema=["scale", "entry", "revert", "max_hold", "stop", "hl_mult", "pnl_stop",
                                       "sizing", "fill", "rearm", "gate", "gate_mode", "need_half_life"],
                        orient="row")


def gate_label(condition: str, bucket: str) -> str:
    return f"{condition}:{bucket}"


def gate_grid(conditions, buckets) -> list[str]:
    """No gate, then every condition in every bucket."""
    return [NO_GATE] + [gate_label(c, b) for c in conditions for b in buckets]


# ---- the walk (backtest.engine) ---------------------------------------------


def expected_now(resid: float, half_life: float, port_vol: float, setup: dict, cost: float = COST) -> dict:
    """What the frozen model would expect of an entry at residual ``resid`` today, with
    the setup's exits: exp_return (bps, net of cost), exp_bars, bps_per_day and exp_risk
    (the portfolio's daily vol over the expected hold). The walk's own arithmetic."""
    hl = float(half_life) if half_life is not None else np.nan
    bars, _ = horizon(int(setup["max_hold"]), float(setup["hl_mult"]), hl)
    exp_return, exp_bars = expect(abs(float(resid)), hl, float(setup["revert"]), bars, float(cost))
    return dict(exp_return=exp_return, exp_bars=exp_bars, bps_per_day=exp_return / exp_bars,
                exp_risk=float(port_vol) * np.sqrt(exp_bars) if port_vol is not None else np.nan)


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


def _with_legacy(configs: pl.DataFrame) -> pl.DataFrame:
    """Configs from before fills, re-arms and gates were knobs run on LEGACY_RULES."""
    if "fill" in configs.columns:  # a current config: every rule is set, and entries need a half-life
        return configs if "need_half_life" in configs.columns else configs.with_columns(need_half_life=pl.lit(True))
    return configs.with_columns(*[pl.lit(v).alias(k) for k, v in LEGACY_RULES.items() if k not in configs.columns])


def gate_conditions(arrays: dict, basis: str, lookback: int) -> dict[str, np.ndarray]:
    """The gate conditions of one model's signal fit (backtest.engine.condition_arrays)."""
    m = basis_arrays(arrays, basis)
    return core.condition_arrays(m["beta"], arrays["r2"], m["resid"], lookback)


def gate_masks(arrays: dict, basis: str, lookback: int, configs: pl.DataFrame,
               gate_edges: dict | None = None) -> tuple[np.ndarray | None, np.ndarray]:
    """(allow-masks, each config's row in them or -1): one mask per (gate, mode) the configs
    use. ``gate_edges`` ({label: [kind, edges]}, freeze_gate) replaces a full-sample gate's
    cut with one frozen elsewhere."""
    keys = list(zip(configs["gate"].to_list(), configs["gate_mode"].to_list()))
    wanted = list(dict.fromkeys(k for k in keys if k[0] != NO_GATE))
    if not wanted:
        return None, np.full(len(configs), -1, dtype=np.int64)
    conds = gate_conditions(arrays, basis, lookback)
    masks = []
    for label, mode in wanted:
        condition, bucket = label.split(":")
        if gate_edges and label in gate_edges and mode == "full":
            kind, edges = gate_edges[label]
            masks.append(core.mask_from_edges(conds[condition], kind, tuple(edges)))
        else:
            masks.append(core.gate_mask(conds[condition], (condition, bucket), mode=mode))
    row = {k: r for r, k in enumerate(wanted)}
    return np.vstack(masks), np.array([row.get(k, -1) for k in keys], dtype=np.int64)


def walk_model(y: np.ndarray, x: np.ndarray, arrays: dict, configs: pl.DataFrame, hedge: float = 1.0,
               cost: float = COST, split: int | None = None, basis: str = "levels", lookback: int | None = None,
               keep=True, gate_edges: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(trades, stats) for every config of one model, from backtest.engine.walk: rows of
    its TRADE_COLS and STATS. ``lookback`` is the model's (its gates are computed over
    it); ``keep`` says whose trades to return. Configs without the newer columns walk on
    LEGACY_RULES, with no stop loss and flat sizing."""
    configs = _with_legacy(configs)
    masks, rows = gate_masks(arrays, basis, lookback, configs, gate_edges)
    return core.walk(engine_model(y, x, arrays, hedge, basis), configs.with_columns(pl.Series("gate", rows)),
                     hedge=hedge, cost=cost, split=split, gates=masks, keep=keep)


def engine_model(y: np.ndarray, x: np.ndarray, arrays: dict, hedge: float, basis: str) -> dict:
    """One Oven model on one signal basis, as backtest.engine's model arrays."""
    m = basis_arrays(arrays, basis)
    # the trade's risk: the hedged portfolio's vol, or the target's own when it trades alone
    vol = arrays.get("port_vol60") if hedge else arrays.get("target_vol60")
    return dict(y=y, x=x, signal_z=m["z"], signal_raw=m["resid"], alpha=m["alpha"], beta=m["beta"],
                center=np.zeros(len(y)), sigma=m["sigma"], half_life=m["half_life"],
                hedge_beta=arrays["beta_changes"], port_vol=np.full(len(y), np.nan) if vol is None else vol)


def _frame_arrays(frame: pl.DataFrame, target: str, name: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """A model_frame's (y, x, model arrays)."""
    return (frame[target].to_numpy().astype(float), frame[name].to_numpy().astype(float),
            {c: frame[c].to_numpy().astype(float) for c in frame.columns
             if c not in ("ts", target, name, "portfolio", "corr126")})


def _setup_configs(setup: dict, pnl_stops=None) -> pl.DataFrame:
    """One setup as engine configs: one row, or one per stop loss in ``pnl_stops``. A rule
    the setup doesn't name is the one it was baked under before it existed (LEGACY_RULES)."""
    stops = [float(setup.get("pnl_stop", 0.0))] if pnl_stops is None else [float(p) for p in pnl_stops]
    rules = {k: setup.get(k, v) for k, v in LEGACY_RULES.items()}
    return pl.DataFrame({"scale": setup["scale"], "entry": float(setup["entry"]), "revert": float(setup["revert"]),
                         "max_hold": int(setup["max_hold"]), "stop": float(setup["stop"]),
                         "hl_mult": float(setup["hl_mult"]), "pnl_stop": stops,
                         "sizing": setup.get("sizing", "flat"), **rules})


def freeze_gate(frame: pl.DataFrame, target: str, setup: dict) -> dict | None:
    """A full-sample gate's raw cut on ``frame`` ({label: [kind, edges]}), to apply unchanged
    on other data -- today's, or live. None when the setup has no such gate."""
    label = setup.get("gate", NO_GATE)
    if label == NO_GATE or setup.get("gate_mode", "full") != "full":
        return None
    condition, bucket = label.split(":")
    _, _, arrays = _frame_arrays(frame, target, setup["input"])
    kind, edges = core.gate_edges(gate_conditions(arrays, setup["basis"], setup["lookback"])[condition],
                                  (condition, bucket))
    return {label: [kind, list(edges)]}


def setup_model(frame: pl.DataFrame, target: str, setup: dict, gate_edges: dict | None = None):
    """One setup on a model_frame, ready for backtest.engine: (model, config, hedge, gate
    mask or None). What a promoted oven signal's exact check walks."""
    name = setup["input"]
    y, x, arrays = _frame_arrays(frame, target, name)
    hedge = 1.0 if hedgeable(name) else 0.0
    configs = _setup_configs(setup)
    masks, rows = gate_masks(arrays, setup["basis"], setup["lookback"], configs, gate_edges)
    config = {**configs.row(0, named=True), "gate": int(rows[0])}
    return (engine_model(y, x, arrays, hedge, setup["basis"]), config, hedge,
            masks[0] if masks is not None else None)


def walk_setup(frame: pl.DataFrame, target: str, setup: dict, cost: float = COST, pnl_stops=None, split=None,
               gate_edges: dict | None = None):
    """Walk one setup (a dict of SETUP keys) on a model_frame -- at its own stop loss, or
    at each of ``pnl_stops``: (raw trades, stats, configs, model arrays, hedge)."""
    name = setup["input"]
    y, x, arrays = _frame_arrays(frame, target, name)
    configs = _setup_configs(setup, pnl_stops)
    hedge = 1.0 if hedgeable(name) else 0.0
    raw, stats = walk_model(y, x, arrays, configs, hedge, cost, split, setup["basis"], setup["lookback"],
                            gate_edges=gate_edges)
    return raw, stats, configs, arrays, hedge


def setup_trades(frame: pl.DataFrame, target: str, setup: dict, cost: float = COST, horizons: dict | None = None,
                 gate_edges: dict | None = None) -> pl.DataFrame:
    """Every trade of one setup on a model_frame, closed and (last, reason "open") any
    position still held, with entry state and expectations; empty if it never traded.
    ``horizons`` ({lookback: model arrays} on the same rows) adds what the other signal
    lookbacks said at each entry (z_lb<L>)."""
    raw, _, configs, arrays, hedge = walk_setup(frame, target, setup, cost, gate_edges=gate_edges)
    if not len(raw):
        return pl.DataFrame()
    lb = setup["lookback"]
    fits = {**(horizons or {}), lb: arrays}
    return trade_frame(raw, setup["input"], lb, setup["hedge_lb"], setup["basis"], frame["ts"], arrays, fits,
                       sorted(fits), configs, hedge)


# ---- bake: the whole search -------------------------------------------------


def bake(panel, inputs: list[str], lookbacks=LOOKBACKS, hedge_lookbacks=HEDGE_LOOKBACKS,
         configs: pl.DataFrame | None = None, cost: float = COST, progress=None, bases=BASES,
         keep_trades: bool = False) -> dict:
    """Walk every (input, signal lookback, hedge lookback, signal basis, config) for the
    panel's target.

    Returns ``board`` (one row per setup that traded, with its strategy stats),
    ``skipped`` (inputs without enough overlap, with the reason), ``data_end`` (the
    last bar baked) and, only with ``keep_trades``, ``trades`` (one row per trade,
    with its entry state). A bake keeps no trades by default: the board is scored
    from the engine's per-setup totals, and one setup's trades are re-walked when
    it is opened (setup_trades). Each model's halves split its own usable sample,
    so an input with a short history still gets two halves; ``split_date`` on the
    board says where."""
    configs = config_grid() if configs is None else configs
    target = panel.target.name
    data = panel.data
    trade_frames, board_frames, skipped = [], [], []
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
                            cost, split, board_frames, trade_frames if keep_trades else None)
    return {"board": pl.concat(board_frames) if board_frames else pl.DataFrame(),
            "trades": pl.concat(trade_frames) if trade_frames else pl.DataFrame(),
            "skipped": pl.DataFrame(skipped), "data_end": data["ts"].max()}


def _walk_basis(name, lookback, hedge_lookback, basis, y, x, ts, arrays, fits, lookbacks, configs, hedge, cost, split,
                board_frames, trade_frames=None) -> None:
    """One model on one signal basis: its setups' board rows, appended -- and, when
    ``trade_frames`` is given, its trades.

    Each trade also records the same input's signal z, on the same basis and hedge,
    at every signal lookback of the bake (``z_lb<L>``; null where that horizon had
    too little history) -- what the other horizons said when this one fired."""
    raw, stats = walk_model(y, x, arrays, configs, hedge, cost, split, basis, lookback, keep=trade_frames is not None)
    scored = core.score(stats)
    board_frames.append(pl.concat([
        pl.DataFrame({"input": [name] * len(configs), "lookback": [lookback] * len(configs),
                      "hedge_lb": [hedge_lookback] * len(configs), "basis": [basis] * len(configs),
                      "split_date": [ts[min(split, len(ts) - 1)]] * len(configs)}), configs,
        scored.with_columns(pl.selectors.float().cast(pl.Float32)),
        pl.DataFrame({"hedged": [bool(hedge)] * len(configs)})], how="horizontal").filter(pl.col("n_trades") > 0))
    if trade_frames is not None and len(raw):
        trade_frames.append(trade_frame(raw, name, lookback, hedge_lookback, basis, ts, arrays, fits, lookbacks,
                                        configs, hedge))


def trade_frame(raw, name, lookback, hedge_lookback, basis, ts, arrays, fits, lookbacks, configs,
                hedge) -> pl.DataFrame:
    """The engine's raw trade rows as a frame, with each trade's entry state and what
    the other horizons said at entry."""
    col = {c: raw[:, i] for i, c in enumerate(TRADE_COLS)}
    decided = col["decided"].astype(np.int64)
    entered, filled = col["entry_fill"].astype(np.int64), col["exit_fill"].astype(np.int64)
    cfg = configs[col["config"].astype(np.int64)]
    m = basis_arrays(arrays, basis)
    entry_state = {**arrays, "entry_z": m["z"], "resid_sigma": m["sigma"], "half_life": m["half_life"]}
    horizons = {f"z_lb{lb}": (basis_arrays(fits[lb], basis)["z"][decided].astype(np.float32) if lb in fits
                              else np.full(len(decided), np.nan, dtype=np.float32)) for lb in lookbacks}
    return pl.DataFrame({
        "input": name, "lookback": lookback, "hedge_lb": hedge_lookback, "basis": basis,
        **{c: cfg[c] for c in configs.columns},
        "entry_ts": ts.gather(entered), "exit_ts": ts.gather(filled),
        "bars": filled - entered, "side": col["side"],
        "hedge_ratio": col["hedge_ratio"], "pnl": col["pnl"],
        "reason": [EXIT_REASONS[int(r)] for r in col["reason"]],
        "size": col["size"].astype(np.float32), "exp_return": col["exp_return"].astype(np.float32),
        "exp_bars": col["exp_bars"].astype(np.float32),
        **{f: entry_state[f][decided].astype(np.float32) for f in FEATURES},
        **horizons,
    })


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
    if not run["trades"].is_empty():
        run["trades"].write_parquet(path / "trades.parquet")
    run["board"].write_parquet(path / "board.parquet")
    if not run["skipped"].is_empty():
        run["skipped"].write_parquet(path / "skipped.parquet")
    n_trades = int(run["board"]["n_trades"].sum()) if not run["board"].is_empty() else 0
    (path / "meta.json").write_text(json.dumps(dict(
        target=target, inputs=inputs, settings=settings, created_at=stamp, n_trades=n_trades,
        n_setups=len(run["board"]), data_end=str(run.get("data_end")), trades_saved=not run["trades"].is_empty()),
        indent=2), encoding="utf-8")
    return path


def load(path: Path) -> dict:
    path = Path(path)
    return dict(meta=json.loads((path / "meta.json").read_text(encoding="utf-8")),
                trades=pl.read_parquet(path / "trades.parquet") if (path / "trades.parquet").is_file()
                else pl.DataFrame(), board=pl.read_parquet(path / "board.parquet"),
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
                          settings.get("pnl_stops", PNL_STOPS), settings.get("sizings", SIZING_DEFAULT),
                          settings.get("fills", FILLS_DEFAULT), settings.get("rearms", REARMS_DEFAULT),
                          gate_grid(settings.get("gate_conditions", []), settings.get("gate_buckets", [])),
                          settings.get("gate_mode", "full"))
    run = bake(panel, inputs, lookbacks=settings.get("lookbacks", LOOKBACKS),
               hedge_lookbacks=[int(h) for h in settings.get("hedge_lookbacks", HEDGE_LOOKBACKS)],
               configs=configs,
               cost=float(settings.get("cost", COST)), progress=progress,
               bases=tuple(settings.get("bases", BASES)))
    path = save(run, target, inputs, settings, Path(settings.get("runs_dir", OVEN_RUNS)))
    n_trades = int(run["board"]["n_trades"].sum()) if not run["board"].is_empty() else 0
    progress(1, 1, f"Completed · {len(run['board']):,} setups, {n_trades:,} trades")
    return {"run_path": str(path), "summary": f"{len(run['board']):,} setups"}
