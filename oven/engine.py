"""Oven: bake signals for one target against a boatload of inputs.

The question is which relationships produce the most profitable, most
trustworthy TRADES. For every (input, lookback) the Oven fits two rolling
regressions of the target y on the input x, each with its own job:

  signal  levels OLS     y = a + bL * x      "is y rich/cheap vs where x says it should be?"
  hedge   changes OLS   dy = c + bC * dx     "what ratio of x neutralises y's daily moves?"

The trade is the PORTFOLIO y hedged by x: hold y and -bC x, with bC frozen at
entry. An input that isn't itself tradable (vol, macro) can't be held as a
hedge, so its trade is y alone.

Entries fade the levels residual when it first crosses a threshold, either
normalised (residual / its rolling std) or raw (bps). From the entry day the
MODEL IS FROZEN: alpha, bL and the residual's std keep their entry values, so
an exit fires only when prices move, never because the rolling fit re-centred
on the trade. Exits: the frozen residual recovers a share of its entry value,
it widens past a stop, or a time stop. Fills are one bar after every decision;
one position at a time per setup.

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
HEDGE_LOOKBACK = 126
Z_ENTRIES = [1.5, 2.0, 2.5]
RAW_ENTRIES = [5.0, 10.0, 15.0, 20.0]
REVERTS = [0.5, 1.0]  # share of the entry residual recovered; 1.0 = back to fair
MAX_HOLDS = [20, 60]
STOPS = [0.0, 1.0]  # entry-sigmas the frozen residual may widen before stopping; 0 = none
COST = 0.0  # bps per round trip
MIN_BARS = 500  # an input needs this much overlap with the target beyond its lookback
PERIODS = 252

SCALES = ("z", "raw")
EXIT_REASONS = ("revert", "stop", "time")
FEATURES = ("entry_z", "r2", "beta_levels", "beta_changes", "hedge_gap", "beta_drift20",
            "half_life", "resid_sigma", "input_vol60", "target_vol60")
SETUP = ["input", "lookback", "scale", "entry", "revert", "max_hold", "stop"]
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
    return {
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
    }


def held_package(y: np.ndarray, x: np.ndarray, beta_c: np.ndarray, hedge: float = 1.0) -> np.ndarray:
    """The portfolio y hedged by x as a level: the cumulative daily P&L of holding y
    against yesterday's changes beta (zero before the beta exists). Display only --
    trades freeze their beta at entry rather than re-hedging daily."""
    move = np.diff(y, prepend=np.nan) - hedge * np.r_[np.nan, beta_c[:-1]] * np.diff(x, prepend=np.nan)
    return np.nancumsum(np.where(np.isfinite(move), move, 0.0))


def config_grid(z_entries=Z_ENTRIES, raw_entries=RAW_ENTRIES, reverts=REVERTS,
                max_holds=MAX_HOLDS, stops=STOPS) -> pl.DataFrame:
    """Every (scale, entry, revert, hold, stop) cell a model is walked through."""
    rows = [("z", e) for e in z_entries] + [("raw", e) for e in raw_entries]
    cells = [(s, e, r, h, st) for (s, e), r, h, st in product(rows, reverts, max_holds, stops)]
    return pl.DataFrame(cells, schema=["scale", "entry", "revert", "max_hold", "stop"], orient="row")


# ---- the walk ---------------------------------------------------------------


@njit(cache=True)
def _walk(y, x, alpha, beta_l, beta_c, sigma, signals, hedge, scale, entry, revert, hold, stop, cost, split,
          trades, stats):
    """Every config's trades and daily P&L statistics for one model.

    trades rows: (config, decision bar, exit fill bar, side, pnl, reason).
    stats rows:  _STATS, from daily marks of the held portfolio y - hedge*beta_c[entry]*x
    over every bar from the first usable signal (flat days are zero), with the
    two halves split at bar ``split``. Entries are first crossings of |signal|
    over the threshold; the model is frozen at the decision bar."""
    n = y.shape[0]
    count = 0
    for c in range(scale.shape[0]):
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
            r0 = y[t] - alpha[t] - beta_l[t] * x[t]
            sign0 = 1.0 if r0 > 0 else -1.0
            goal = (1.0 - revert[c]) * abs(r0)
            limit = abs(r0) + stop[c] * sigma[t]
            exit_u = -1
            reason = 2
            for k in range(1, hold[c] + 1):
                u = t + k
                if u >= n - 1:
                    break
                frozen = sign0 * (y[u] - alpha[t] - beta_l[t] * x[u])
                if frozen <= goal:
                    exit_u, reason = u, 0
                    break
                if stop[c] > 0 and frozen >= limit:
                    exit_u, reason = u, 1
                    break
                if k == hold[c]:
                    exit_u, reason = u, 2
            if exit_u < 0:
                break  # runs off the end of the sample: no completed trade
            f0, f1 = t + 1, exit_u + 1
            side = -sign0
            ratio = hedge * beta_c[t]
            pnl = 0.0
            for d in range(f0 + 1, f1 + 1):  # daily marks of the held portfolio
                mark = side * ((y[d] - y[d - 1]) - ratio * (x[d] - x[d - 1]))
                if d == f1:
                    mark -= cost
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
            count += 1
            t = f1
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


def walk_model(y: np.ndarray, x: np.ndarray, arrays: dict, configs: pl.DataFrame, hedge: float = 1.0,
               cost: float = COST, split: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(trades, stats) for every config of one model: the raw arrays _walk fills."""
    n = len(y)
    signals = np.ascontiguousarray(np.vstack([arrays["entry_z"], arrays["resid"]]))
    scale = np.array([SCALES.index(s) for s in configs["scale"]], dtype=np.int64)
    trades = np.empty((len(configs) * (n // 2 + 1), 6))
    stats = np.zeros((len(configs), len(_STATS)))
    count = _walk(np.ascontiguousarray(y, dtype=np.float64), np.ascontiguousarray(x, dtype=np.float64),
                  arrays["alpha"], arrays["beta_levels"], arrays["beta_changes"], arrays["resid_sigma"],
                  signals, float(hedge), scale, configs["entry"].to_numpy().astype(np.float64),
                  configs["revert"].to_numpy().astype(np.float64), configs["max_hold"].to_numpy().astype(np.int64),
                  configs["stop"].to_numpy().astype(np.float64), float(cost), int(n if split is None else split),
                  trades, stats)
    return trades[:count], stats


def _sharpe(total, total_sq, days):
    mean = total / days
    var = total_sq / days - mean ** 2
    return pl.when((days > 1) & (var > 1e-12)).then(mean / var.sqrt() * np.sqrt(PERIODS)).otherwise(None)


# ---- bake: the whole search -------------------------------------------------


def bake(panel, inputs: list[str], lookbacks=LOOKBACKS, hedge_lookback: int = HEDGE_LOOKBACK,
         configs: pl.DataFrame | None = None, cost: float = COST, progress=None) -> dict:
    """Walk every (input, lookback, config) for the panel's target.

    Returns ``trades`` (one row per trade, with its entry-state features),
    ``board`` (one row per setup, with its strategy stats) and ``skipped``
    (inputs without enough overlap, with the reason). Each model's halves
    split its own usable sample, so an input with a short history still gets
    two halves; ``split_date`` on the board says where."""
    configs = config_grid() if configs is None else configs
    target = panel.target.name
    data = panel.data
    trade_frames, stat_frames, skipped = [], [], []
    models = [(i, lb) for i in dict.fromkeys(inputs) if i != target for lb in lookbacks]
    for k, (name, lookback) in enumerate(models):
        if progress:
            progress(k, len(models), f"Baking {target} vs {name} · lookback {lookback}")
        if name not in data.columns:
            skipped.append(dict(input=name, lookback=lookback, reason="not in panel"))
            continue
        aligned = data.select("ts", target, name).drop_nulls()
        if len(aligned) < lookback + MIN_BARS:
            skipped.append(dict(input=name, lookback=lookback, reason=f"only {len(aligned)} shared bars"))
            continue
        y = aligned[target].to_numpy().astype(float)
        x = aligned[name].to_numpy().astype(float)
        ts = aligned["ts"]
        arrays = model_arrays(y, x, lookback, hedge_lookback)
        hedge = 1.0 if hedgeable(name) else 0.0
        usable = np.flatnonzero(np.isfinite(arrays["entry_z"]))
        split = int(usable[0] + (len(y) - usable[0]) // 2) if len(usable) else len(y)
        raw, stats = walk_model(y, x, arrays, configs, hedge, cost, split)
        stat_frames.append(pl.concat([
            pl.DataFrame({"input": [name] * len(configs), "lookback": [lookback] * len(configs),
                          "split_date": [ts[min(split, len(ts) - 1)]] * len(configs)}), configs,
            pl.DataFrame(stats, schema=list(_STATS), orient="row")], how="horizontal"))
        if not len(raw):
            continue
        decided = raw[:, 1].astype(np.int64)
        filled = raw[:, 2].astype(np.int64)
        cfg = configs[raw[:, 0].astype(np.int64)]
        trade_frames.append(pl.DataFrame({
            "input": name, "lookback": lookback,
            **{c: cfg[c] for c in configs.columns},
            "entry_ts": ts.gather(decided + 1), "exit_ts": ts.gather(filled),
            "bars": filled - decided - 1, "side": raw[:, 3],
            "hedge_ratio": hedge * arrays["beta_changes"][decided], "pnl": raw[:, 4],
            "reason": [EXIT_REASONS[int(r)] for r in raw[:, 5]],
            **{f: arrays[f][decided].astype(np.float32) for f in FEATURES},
        }))
    if progress:
        progress(len(models), len(models), f"Scoring {sum(len(f) for f in trade_frames):,} trades")
    trades = pl.concat(trade_frames) if trade_frames else pl.DataFrame()
    stats = pl.concat(stat_frames) if stat_frames else pl.DataFrame()
    return {"trades": trades, "board": board(trades, stats), "skipped": pl.DataFrame(skipped)}


def board(trades: pl.DataFrame, stats: pl.DataFrame) -> pl.DataFrame:
    """One row per setup with its strategy stats (see the module docstring).
    Setups that never traded are dropped."""
    if trades.is_empty() or stats.is_empty():
        return pl.DataFrame()
    p = pl.col("pnl")
    per_trade = trades.group_by(SETUP).agg(
        pl.len().alias("n_trades"), p.mean().alias("avg_trade"), p.median().alias("median_trade"),
        (p > 0).mean().alias("hit_rate"), p.max().alias("best_trade"), p.min().alias("worst_trade"),
        (p.mean() / p.std() * pl.len().sqrt()).alias("trade_t"),
        (p.filter(p > 0).sum() / -p.filter(p < 0).sum()).alias("profit_factor"),
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
    board's Sharpe is built from."""
    out = np.zeros(len(ts))
    index = {d: i for i, d in enumerate(ts.to_list())}
    dy, dx = np.diff(y, prepend=np.nan), np.diff(x, prepend=np.nan)
    for t in trades.iter_rows(named=True):
        i0, i1 = index[t["entry_ts"]], index[t["exit_ts"]]
        out[i0 + 1:i1 + 1] += t["side"] * (dy[i0 + 1:i1 + 1] - t["hedge_ratio"] * dx[i0 + 1:i1 + 1])
        out[i1] -= cost
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
                          settings.get("stops", STOPS))
    run = bake(panel, inputs, lookbacks=settings.get("lookbacks", LOOKBACKS),
               hedge_lookback=int(settings.get("hedge_lookback", HEDGE_LOOKBACK)), configs=configs,
               cost=float(settings.get("cost", COST)), progress=progress)
    path = save(run, target, inputs, settings, Path(settings.get("runs_dir", OVEN_RUNS)))
    progress(1, 1, f"Completed · {len(run['board']):,} setups, {len(run['trades']):,} trades")
    return {"run_path": str(path), "summary": f"{len(run['board']):,} setups"}
