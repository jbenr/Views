"""Config-vectorised single-position backtester with exact ``Engine`` semantics.

``Engine`` walks one strategy through time in Python. Research grids differ
only in parameters, not data: every config reads the same target legs and one
of a handful of signal columns. So the data stays small and only per-config
*state* (direction, entry level, bars held, ...) scales with the grid. Each
config is an independent state machine run over the shared arrays inside one
numba kernel, parallel across configs.

Semantics mirror ``research.dislocation_backtest.make_pipeline`` driven by
``Engine(max_total_positions=1)``: first-crossing entries (optionally gated),
exits checked before entries, same-bar re-entry, full round-trip cost charged
at exit, entry weights frozen and legs re-marked daily, bars with a missing
signal or level skipped without ageing the position. ``tests/test_vector.py``
holds the engine to that, trade for trade.

Output is aggregate, not per-day: per-config totals plus per-fold sums and
sums of squares, from which the Sharpe of any union of folds is exact. Daily
P&L is kept only for configs explicitly requested.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from numba import njit, prange

EXIT_STYLES = {"time": 0, "band": 1, "revert_frac": 2, "half_life_frac": 3}


@dataclass(frozen=True)
class VectorResult:
    """Per-config metrics plus per-fold P&L sufficient statistics.

    ``fold_sum`` / ``fold_sumsq`` / ``fold_trades`` / ``fold_wins`` are
    (configs, folds); ``fold_days`` is (folds,) because every config is
    marked on every bar. ``pnl`` is (bars, kept configs) or ``None``.
    """

    metrics: pl.DataFrame
    folds: np.ndarray
    fold_days: np.ndarray
    fold_sum: np.ndarray
    fold_sumsq: np.ndarray
    fold_trades: np.ndarray
    fold_wins: np.ndarray
    pnl: np.ndarray | None

    def sharpe(self, fold_mask: np.ndarray | None = None) -> np.ndarray:
        """Annualised daily Sharpe (population std, as ``Engine``) over chosen folds."""
        mask = np.ones(len(self.folds), dtype=bool) if fold_mask is None else np.asarray(fold_mask)
        return _sharpe(self.fold_days[mask].sum(), self.fold_sum[:, mask].sum(axis=1),
                       self.fold_sumsq[:, mask].sum(axis=1))


def _trade_summary(n_trades, n_wins, gross_profit, gross_loss, total, summary_dd) -> list[pl.Series]:
    """Engine.summary's trade ratios, with its conventions: 0 with no trades, inf on a zero divisor."""
    has = n_trades > 0
    n_loss = n_trades - n_wins
    with np.errstate(invalid="ignore", divide="ignore"):
        avg_win = np.where(n_wins > 0, gross_profit / np.maximum(n_wins, 1), 0.0)
        avg_loss = np.where(n_loss > 0, gross_loss / np.maximum(n_loss, 1), 0.0)
        profit_factor = np.where(gross_loss != 0, np.abs(gross_profit / gross_loss), np.inf)
        calmar = np.where(summary_dd != 0, total / np.abs(summary_dd), np.inf)
        ratio = np.where(avg_loss != 0, np.abs(avg_win / avg_loss), np.inf)
    return [pl.Series("avg_win_bps", avg_win), pl.Series("avg_loss_bps", avg_loss),
            pl.Series("profit_factor", np.where(has, profit_factor, 0.0)),
            pl.Series("calmar", np.where(has, calmar, 0.0)),
            pl.Series("win_loss_ratio", np.where(has, ratio, 0.0))]


def _sharpe(n: int, total: np.ndarray, total_sq: np.ndarray) -> np.ndarray:
    mean = total / n
    var = np.maximum(total_sq / n - mean * mean, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(var > 0, mean / np.sqrt(var) * np.sqrt(252.0), 0.0)


@njit(parallel=True, cache=True)
def _kernel(legs, fixed_w, signals, half_life, gates, entry_w, use_entry_w,
            model, gate, entry, style, param, stop, sig_stop, cap, cost, lag, fold, n_fold,
            keep, pnl_out, fold_sum, fold_sumsq, fold_trades, fold_wins,
            total, closed_pnl, n_trades, n_wins, max_dd, days_held,
            gross_profit, gross_loss, closed_bars, open_end, summary_dd, held_w):
    n_bars, n_legs = legs.shape
    level = np.zeros(n_bars)
    for i in range(n_bars):
        acc = 0.0
        for leg in range(n_legs):
            acc += legs[i, leg] * fixed_w[leg]
        level[i] = acc

    for k in prange(model.shape[0]):
        m, g, e, st, p, sl = model[k], gate[k], entry[k], style[k], param[k], stop[k]
        ss, cp = sig_stop[k], cap[k]
        time_stop = int(np.round(p)) if st == 0 else 0
        # Frozen entry weights live in this config's row of held_w, indexed in
        # place. Allocating (or slicing) an array here makes numba hoist one
        # buffer out of the parallel loop and share it, refcount included,
        # across threads: a data race whose refcounting corrupted memory and
        # crashed the research server natively.
        pos = 0
        bars = 0
        dyn = 0
        entry_level = 0.0
        entry_sig = 0.0
        prev_unreal = 0.0
        cum = 0.0
        peak = 0.0
        worst = 0.0
        # Engine's summary drawdown (behind calmar) peaks from the first bar, not zero.
        summary_peak = -np.inf
        summary_worst = 0.0
        for i in range(n_bars):
            j = i - lag
            s = signals[m, j] if j >= 0 else np.nan
            daily = 0.0
            if not (np.isnan(s) or np.isnan(level[i])):
                realized = 0.0
                if pos != 0:
                    bars += 1
                    held = 0.0
                    for leg in range(n_legs):
                        held += held_w[k, leg] * legs[i, leg]
                    cur = pos * (held - entry_level)
                    out = False
                    if st == 1:
                        out = (pos == 1 and s > -p) or (pos == -1 and s < p)
                    if sl > 0 and cur < -sl:
                        out = True
                    if ss > 0 and ((pos == 1 and s < entry_sig - ss) or (pos == -1 and s > entry_sig + ss)):
                        out = True
                    if st == 0 and time_stop > 0 and bars >= time_stop:
                        out = True
                    # half-life exits and half-life caps on band/reversion exits
                    if dyn > 0 and bars >= dyn:
                        out = True
                    if not out and st == 2:
                        target = entry_sig * (1.0 - p)
                        out = (pos == 1 and s >= target) or (pos == -1 and s <= target)
                    if out:
                        trade = cur - cost
                        realized = trade
                        f = fold[i]
                        fold_trades[k, f] += 1
                        n_trades[k] += 1
                        closed_pnl[k] += trade
                        closed_bars[k] += bars
                        if trade > 0:
                            fold_wins[k, f] += 1
                            n_wins[k] += 1
                            gross_profit[k] += trade
                        else:
                            gross_loss[k] += trade
                        pos = 0
                if pos == 0 and j >= 0:
                    now = signals[m, j]
                    before = signals[m, j - 1] if j >= 1 else np.nan
                    allowed = g < 0 or gates[g, j]
                    if st == 3:
                        hl = half_life[m, j]
                        allowed = allowed and np.isfinite(hl) and hl > 0
                    direction = 0
                    if allowed:
                        if now <= -e and not (before <= -e):
                            direction = 1
                        elif now >= e and not (before >= e):
                            direction = -1
                    if direction != 0:
                        if use_entry_w:
                            for leg in range(n_legs):
                                held_w[k, leg] = entry_w[j, leg]
                                if not np.isfinite(held_w[k, leg]):
                                    direction = 0
                        else:
                            for leg in range(n_legs):
                                held_w[k, leg] = fixed_w[leg]
                    if direction != 0:
                        if use_entry_w:
                            acc = 0.0
                            for leg in range(n_legs):
                                acc += held_w[k, leg] * legs[i, leg]
                            entry_level = acc
                        else:
                            entry_level = level[i]
                        pos = direction
                        bars = 0
                        entry_sig = s
                        dyn = 0
                        if st == 3:
                            dyn = max(1, int(np.round(np.ceil(half_life[m, j] * p))))
                        elif cp > 0 and (st == 1 or st == 2):
                            hl = half_life[m, j]
                            if np.isfinite(hl) and hl > 0:
                                dyn = max(1, int(np.round(np.ceil(hl * cp))))
                unreal = 0.0
                if pos != 0:
                    held = 0.0
                    for leg in range(n_legs):
                        held += held_w[k, leg] * legs[i, leg]
                    unreal = pos * (held - entry_level)
                daily = realized + unreal - prev_unreal
                prev_unreal = unreal
            if pos != 0:
                days_held[k] += 1
            f = fold[i]
            fold_sum[k, f] += daily
            fold_sumsq[k, f] += daily * daily
            total[k] += daily
            cum += daily
            peak = max(peak, cum)
            worst = min(worst, cum - peak)
            summary_peak = max(summary_peak, cum)
            summary_worst = min(summary_worst, cum - summary_peak)
            if keep[k] >= 0:
                pnl_out[i, keep[k]] = daily
        max_dd[k] = worst
        summary_dd[k] = summary_worst
        open_end[k] = pos != 0


def run_vector(
    legs: np.ndarray,
    leg_weights: np.ndarray,
    signals: np.ndarray,
    configs: pl.DataFrame,
    *,
    half_life: np.ndarray | None = None,
    gates: np.ndarray | None = None,
    entry_weights: np.ndarray | None = None,
    cost: float = 0.0,
    lag: int = 1,
    folds: np.ndarray | None = None,
    keep: np.ndarray | list[int] | None = None,
) -> VectorResult:
    """Backtest every row of ``configs`` against shared market arrays.

    legs          (bars, legs) executable leg levels, in P&L units (bps).
    leg_weights   (legs,) fixed weights; their sum defines the traded level.
    signals       (models, bars) unshifted signal per model.
    configs       columns ``model`` (row of ``signals``), ``entry``,
                  ``exit_style`` (a key of EXIT_STYLES), ``exit_param``, and
                  optional ``gate`` (row of ``gates``; -1/null = ungated),
                  ``stop`` (P&L stop in bps; 0/null = none), ``signal_stop``
                  (exit once the signal moves this far beyond its entry value,
                  away from zero, in signal units; 0/null = none) and ``cap``
                  (band/revert_frac only: exit after ceil(entry half-life x
                  cap) bars when that half-life is positive; 0/null = none).
    half_life     (models, bars), required by half_life_frac configs.
    gates         (gate masks, bars) boolean entry-allow masks, unshifted.
    entry_weights (bars, legs) weights frozen at entry, replacing
                  ``leg_weights``; unshifted like the signal.
    lag           bars between signal and fill (0 or 1), as ``execution_lag``.
    folds         (bars,) integer labels (e.g. years); defaults to one fold.
    keep          config row indices whose daily P&L should be returned.
    """
    legs = np.ascontiguousarray(legs, dtype=np.float64)
    if legs.ndim == 1:
        legs = legs[:, None]
    n_bars = legs.shape[0]
    signals = np.ascontiguousarray(np.atleast_2d(signals), dtype=np.float64)
    if signals.shape[1] != n_bars:
        raise ValueError(f"signals have {signals.shape[1]} bars, legs have {n_bars}")
    if lag not in (0, 1):
        raise ValueError("lag must be 0 or 1")
    styles = configs["exit_style"].to_list()
    unknown = sorted(set(styles) - set(EXIT_STYLES))
    if unknown:
        raise ValueError(f"unknown exit styles {unknown}; expected {sorted(EXIT_STYLES)}")
    style = np.array([EXIT_STYLES[s] for s in styles], dtype=np.int64)
    model = configs["model"].cast(pl.Int64).to_numpy()
    if model.min(initial=0) < 0 or model.max(initial=0) >= len(signals):
        raise ValueError("config model index outside signals")
    gate = (configs["gate"].fill_null(-1).cast(pl.Int64).to_numpy()
            if "gate" in configs.columns else np.full(len(configs), -1, dtype=np.int64))
    if gates is None:
        if (gate >= 0).any():
            raise ValueError("configs reference gates but no gate masks were given")
        gates = np.ones((1, n_bars), dtype=np.bool_)
    gates = np.ascontiguousarray(gates, dtype=np.bool_)
    if gates.shape[1] != n_bars or gate.max(initial=-1) >= len(gates):
        raise ValueError("gate masks must be (gates, bars) and cover every config gate index")
    stop = (configs["stop"].fill_null(0.0).cast(pl.Float64).to_numpy()
            if "stop" in configs.columns else np.zeros(len(configs)))
    sig_stop = (configs["signal_stop"].fill_null(0.0).cast(pl.Float64).to_numpy()
                if "signal_stop" in configs.columns else np.zeros(len(configs)))
    cap = (configs["cap"].fill_null(0.0).cast(pl.Float64).to_numpy()
           if "cap" in configs.columns else np.zeros(len(configs)))
    if half_life is None:
        if (style == EXIT_STYLES["half_life_frac"]).any() or (cap > 0).any():
            raise ValueError("half_life_frac configs and half-life caps need half_life")
        half_life = np.full((1, 1), np.nan)
    half_life = np.ascontiguousarray(np.atleast_2d(half_life), dtype=np.float64)
    use_w = entry_weights is not None
    entry_w = np.ascontiguousarray(entry_weights if use_w else np.zeros((1, legs.shape[1])), dtype=np.float64)
    if use_w and entry_w.shape != legs.shape:
        raise ValueError("entry_weights must match legs' shape")

    labels, fold = (np.array([0]), np.zeros(n_bars, dtype=np.int64)) if folds is None \
        else np.unique(np.asarray(folds), return_inverse=True)
    fold = fold.astype(np.int64)
    n_cfg, n_fold = len(configs), len(labels)
    keep_idx = np.full(n_cfg, -1, dtype=np.int64)
    kept = [] if keep is None else list(keep)
    keep_idx[kept] = np.arange(len(kept))
    pnl = np.zeros((n_bars, len(kept)))
    fold_sum, fold_sumsq = np.zeros((n_cfg, n_fold)), np.zeros((n_cfg, n_fold))
    fold_trades, fold_wins = np.zeros((n_cfg, n_fold), np.int64), np.zeros((n_cfg, n_fold), np.int64)
    total, closed = np.zeros(n_cfg), np.zeros(n_cfg)
    n_trades, n_wins = np.zeros(n_cfg, np.int64), np.zeros(n_cfg, np.int64)
    max_dd, days_held = np.zeros(n_cfg), np.zeros(n_cfg, np.int64)
    gross_profit, gross_loss = np.zeros(n_cfg), np.zeros(n_cfg)
    closed_bars, open_end = np.zeros(n_cfg, np.int64), np.zeros(n_cfg, np.bool_)
    summary_dd = np.zeros(n_cfg)
    held_w = np.zeros((n_cfg, legs.shape[1]))

    _kernel(legs, np.asarray(leg_weights, dtype=np.float64), signals, half_life, gates, entry_w, use_w,
            model, gate, configs["entry"].cast(pl.Float64).to_numpy(), style,
            configs["exit_param"].cast(pl.Float64).to_numpy(), stop, sig_stop, cap, float(cost), int(lag),
            fold, n_fold, keep_idx, pnl, fold_sum, fold_sumsq, fold_trades, fold_wins,
            total, closed, n_trades, n_wins, max_dd, days_held,
            gross_profit, gross_loss, closed_bars, open_end, summary_dd, held_w)

    fold_days = np.bincount(fold, minlength=n_fold)
    metrics = configs.with_columns(
        pl.Series("total_pnl_bps", total), pl.Series("closed_pnl_bps", closed),
        pl.Series("n_trades", n_trades), pl.Series("trade_win_rate", np.where(n_trades > 0, n_wins / np.maximum(n_trades, 1), 0.0)),
        pl.Series("sharpe", _sharpe(n_bars, fold_sum.sum(axis=1), fold_sumsq.sum(axis=1))),
        pl.Series("max_drawdown_bps", max_dd),
        pl.Series("time_in_market_pct", days_held / n_bars),
        pl.Series("avg_pnl_per_trade_bps", np.where(n_trades > 0, closed / np.maximum(n_trades, 1), 0.0)),
        # Engine reports zero, not NaN, when nothing has closed.
        pl.Series("avg_holding_days", np.where(n_trades > 0, closed_bars / np.maximum(n_trades, 1), 0.0)),
        pl.Series("gross_profit_bps", gross_profit), pl.Series("gross_loss_bps", gross_loss),
        pl.Series("open_trades", open_end.astype(np.int64)),
        *_trade_summary(n_trades, n_wins, gross_profit, gross_loss, total, summary_dd),
    )
    return VectorResult(metrics, labels, fold_days, fold_sum, fold_sumsq, fold_trades, fold_wins,
                        pnl if kept else None)
