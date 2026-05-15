"""OU-aware exit policies for duration setups.

Two policies, both pair-aware and per-trade adaptive:

  ou_time_stop:   max_hold = round(k * max(half_life across components))
                  half-life is computed at entry from a rolling AR(1) on each
                  component's residual_cum series.

  ou_level_target: exit when the residual has closed by `fraction` of its
                   entry value (any-component or all-components). Falls back
                   to a half-life-scaled hard cap.

Both also log the implied 10y move to target (informational only) so trades
remain interpretable in yield space.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from stats.ou import roll_half_life

try:  # package import (when 'signal.duration' is importable)
    from .setups import Setup
except ImportError:  # flat import (notebook adds 'signal/duration' to sys.path)
    from setups import Setup  # type: ignore


def components_from_pair(setup: Setup, all_setups: list[Setup]) -> list[Setup]:
    """Look up component setups for a pair name like 'A & B'."""
    if " & " not in setup.name:
        return [setup]
    by_name = {s.name: s for s in all_setups}
    parts = [p.strip() for p in setup.name.split(" & ")]
    out = []
    for p in parts:
        if p not in by_name:
            raise KeyError(f"component {p!r} not found in setup catalog")
        out.append(by_name[p])
    return out


def _half_life_at(residual: pd.Series, hl_lookback: int) -> pd.Series:
    """Rolling AR(1) half-life on the residual, indexed like residual."""
    pl_hl = roll_half_life(residual, lookback=hl_lookback).to_pandas()
    if len(pl_hl) == len(residual):
        out = pd.Series(pl_hl.values, index=residual.index, name="half_life")
    else:
        # roll_half_life pads internally; align defensively
        out = pd.Series(pl_hl.values, index=residual.index[-len(pl_hl):], name="half_life")
    return out.reindex(residual.index)


def _component_state(comp: Setup, hl_lookback: int) -> Optional[dict]:
    """Per-component lookup tables: residual, score (z), rolling std of resid, half-life.
    Returns None for components without a mean-reverting residual (e.g. trend/RSI/momentum).
    """
    if comp.residual is None:
        return None
    resid = comp.residual.astype(float)
    z = comp.score.astype(float) if comp.score is not None else None
    # rolling std of residual_cum at the same window the score was z-scored over
    z_lookback = comp.params.get("lookback", 252)
    resid_std = resid.rolling(z_lookback).std()
    hl = _half_life_at(resid, hl_lookback=hl_lookback)
    return {"resid": resid, "z": z, "resid_std": resid_std, "half_life": hl, "z_lookback": z_lookback}


def _trade_metrics(pnl: pd.Series) -> dict:
    pnl = pnl.dropna()
    if len(pnl) < 2 or pnl.std() == 0:
        return {"n": int(len(pnl)), "mean": np.nan, "std": np.nan,
                "sharpe_pertrade": np.nan, "hit_rate": np.nan, "avg_hold": np.nan}
    return {
        "n": int(len(pnl)),
        "mean": float(pnl.mean()),
        "std": float(pnl.std(ddof=1)),
        "sharpe_pertrade": float(pnl.mean() / pnl.std(ddof=1)),
        "hit_rate": float((pnl > 0).mean()),
    }


def _walk_trades(
    trig: pd.Series,
    y: pd.Series,
    direction: int,
    *,
    decide_exit,
    min_hold: int = 1,
):
    """Generic episodic trade walker.

    decide_exit(entry_i, exit_i, held) -> (should_exit: bool, reason: str) | None
    Returns trade rows; non-overlapping with same-episode dedupe.
    """
    n = len(y)
    rows = []
    i = 0
    trade_id = 0
    while i < n:
        if not bool(trig.iloc[i]) or pd.isna(y.iloc[i]):
            i += 1
            continue
        entry_i = i
        exit_i = entry_i
        exit_reason = "end_of_data"
        while exit_i + 1 < n:
            next_i = exit_i + 1
            held = next_i - entry_i
            decision = decide_exit(entry_i, next_i, held)
            if decision is not None and held >= min_hold:
                should_exit, reason = decision
                if should_exit:
                    exit_i = next_i
                    exit_reason = reason
                    break
            exit_i = next_i

        if exit_i > entry_i and pd.notna(y.iloc[exit_i]):
            change = float(y.iloc[exit_i] - y.iloc[entry_i])
            rows.append({
                "trade_id": trade_id,
                "entry_date": y.index[entry_i],
                "exit_date": y.index[exit_i],
                "entry_idx": entry_i,
                "exit_idx": exit_i,
                "holding_period": exit_i - entry_i,
                "direction": direction,
                "entry_yield_bps": float(y.iloc[entry_i]),
                "exit_yield_bps": float(y.iloc[exit_i]),
                "yield_change_bps": change,
                "pnl_bps": float(direction * change),
                "exit_reason": exit_reason,
            })
            trade_id += 1
        # Skip the remainder of this signal episode
        i = max(exit_i + 1, entry_i + 1)
        while i < n and bool(trig.iloc[i]):
            i += 1
    return pd.DataFrame(rows)


def ou_time_stop_log(
    setup: Setup,
    y: pd.Series,
    components: list[Setup],
    *,
    k: float = 1.5,
    hl_lookback: int = 252,
    hard_cap: int = 90,
    floor: int = 3,
) -> pd.DataFrame:
    """Adaptive max-hold = round(k * max half-life across components at entry).

    Falls back to `hard_cap` if half-life is non-positive/NaN. `floor` prevents
    1-day exits when the residual is decaying fast.
    """
    trig = setup.trigger.reindex(y.index).fillna(False).astype(bool)
    raw = [(c, _component_state(c, hl_lookback=hl_lookback)) for c in components]
    pairs = [(c, s) for c, s in raw if s is not None]
    if not pairs:
        return pd.DataFrame()
    use_components = [c for c, _ in pairs]
    states = [s for _, s in pairs]

    # cache aligned to y.index
    hls = [s["half_life"].reindex(y.index) for s in states]

    def decide(entry_i, next_i, held):
        # max half-life across components, evaluated at entry
        hl_vals = []
        for s in hls:
            v = s.iloc[entry_i]
            if pd.notna(v) and v > 0:
                hl_vals.append(float(v))
        if hl_vals:
            target = int(round(k * max(hl_vals)))
            target = max(floor, min(target, hard_cap))
        else:
            target = hard_cap
        if held >= target:
            return (True, f"ou_time_stop_{target}d")
        return (False, "")

    log = _walk_trades(trig, y, setup.direction, decide_exit=decide)
    # annotate with the entry-time half-lives
    if not log.empty:
        for j, s in enumerate(states):
            col_name = f"{use_components[j].name}__half_life_at_entry"
            log[col_name] = log["entry_idx"].map(lambda i: float(s["half_life"].iloc[i]) if pd.notna(s["half_life"].iloc[i]) else np.nan)
    return log


def ou_level_target_log(
    setup: Setup,
    y: pd.Series,
    components: list[Setup],
    *,
    fraction: float = 0.5,
    require: str = "any",         # "any" or "all"
    hl_lookback: int = 252,
    k_cap: float = 2.0,
    hard_cap: int = 90,
    floor: int = 3,
) -> pd.DataFrame:
    """Exit when residual has closed by `fraction` of entry value.

    For each component, log entry residual r0; exit triggers when current
    residual r_t satisfies sign(r0)*r_t <= sign(r0)*r0*(1-fraction).
    `require="any"` exits on the first leg to revert; "all" waits for both.
    Time fallback = round(k_cap * max half-life at entry), capped at hard_cap.
    """
    if require not in {"any", "all"}:
        raise ValueError("require must be 'any' or 'all'")

    trig = setup.trigger.reindex(y.index).fillna(False).astype(bool)
    raw = [(c, _component_state(c, hl_lookback=hl_lookback)) for c in components]
    pairs = [(c, s) for c, s in raw if s is not None]
    if not pairs:
        return pd.DataFrame()
    use_components = [c for c, _ in pairs]
    states = [s for _, s in pairs]

    # entry-time captures
    entry_residuals: dict[int, list[float]] = {}
    entry_half_lives: dict[int, list[float]] = {}
    entry_stds: dict[int, list[float]] = {}
    entry_zs: dict[int, list[float]] = {}

    def decide(entry_i, next_i, held):
        if entry_i not in entry_residuals:
            entry_residuals[entry_i] = [float(s["resid"].iloc[entry_i]) if pd.notna(s["resid"].iloc[entry_i]) else np.nan for s in states]
            entry_half_lives[entry_i] = [float(s["half_life"].iloc[entry_i]) if pd.notna(s["half_life"].iloc[entry_i]) else np.nan for s in states]
            entry_stds[entry_i] = [float(s["resid_std"].iloc[entry_i]) if pd.notna(s["resid_std"].iloc[entry_i]) else np.nan for s in states]
            entry_zs[entry_i] = [float(s["z"].iloc[entry_i]) if (s["z"] is not None and pd.notna(s["z"].iloc[entry_i])) else np.nan for s in states]

        r0s = entry_residuals[entry_i]
        # per-leg: has residual closed by fraction?
        hits = []
        for s, r0 in zip(states, r0s):
            r_t = s["resid"].iloc[next_i]
            if pd.isna(r_t) or pd.isna(r0) or r0 == 0:
                hits.append(False)
                continue
            target = (1.0 - fraction) * r0
            # closed = current residual is on the same side of target as zero
            if r0 > 0:
                hits.append(r_t <= target)
            else:
                hits.append(r_t >= target)

        triggered = any(hits) if require == "any" else all(hits)
        if triggered:
            return (True, f"ou_level_{require}_{int(fraction*100)}p")

        # time fallback
        hl_vals = [v for v in entry_half_lives[entry_i] if pd.notna(v) and v > 0]
        if hl_vals:
            cap = int(round(k_cap * max(hl_vals)))
            cap = max(floor, min(cap, hard_cap))
        else:
            cap = hard_cap
        if held >= cap:
            return (True, f"ou_level_time_cap_{cap}d")
        return (False, "")

    log = _walk_trades(trig, y, setup.direction, decide_exit=decide)
    if log.empty:
        return log

    for j, comp in enumerate(use_components):
        log[f"{comp.name}__entry_resid"] = log["entry_idx"].map(lambda i: entry_residuals.get(i, [np.nan]*len(use_components))[j])
        log[f"{comp.name}__entry_z"] = log["entry_idx"].map(lambda i: entry_zs.get(i, [np.nan]*len(use_components))[j])
        log[f"{comp.name}__half_life_at_entry"] = log["entry_idx"].map(lambda i: entry_half_lives.get(i, [np.nan]*len(use_components))[j])
        # implied 10y move to target (in bps), purely informational
        # entry residual is in y-units (resid_cum from regression of dy on dx)
        # closing by fraction → y moves by fraction * entry_resid in opposite direction
        log[f"{comp.name}__implied_y_move_bps_to_target"] = log[f"{comp.name}__entry_resid"].apply(
            lambda r0: float(-fraction * r0) if pd.notna(r0) else np.nan
        )
    return log


def summarize_log(log: pd.DataFrame, name: str, exit_policy: str) -> dict:
    if log.empty:
        return {"name": name, "exit_policy": exit_policy, "n_trades": 0,
                "avg_hold": np.nan, "median_hold": np.nan,
                "mean_pnl_bps": np.nan, "std_pnl_bps": np.nan,
                "sharpe_pertrade": np.nan, "sharpe_ann": np.nan, "hit_rate": np.nan}
    m = _trade_metrics(log["pnl_bps"])
    avg_hold = float(log["holding_period"].mean())
    # Annualized Sharpe under the assumption you can re-run the bet back-to-back.
    # Apples-to-apples across policies with different hold horizons.
    sharpe_ann = (m["sharpe_pertrade"] * np.sqrt(252.0 / avg_hold)) if avg_hold > 0 else np.nan
    return {
        "name": name,
        "exit_policy": exit_policy,
        "n_trades": m["n"],
        "avg_hold": avg_hold,
        "median_hold": float(log["holding_period"].median()),
        "mean_pnl_bps": m["mean"],
        "std_pnl_bps": m["std"],
        "sharpe_pertrade": m["sharpe_pertrade"],
        "sharpe_ann": sharpe_ann,
        "hit_rate": m["hit_rate"],
    }


def compare_ou_exits(
    setup: Setup,
    y: pd.Series,
    all_setups: list[Setup],
    *,
    ks: Iterable[float] = (0.5, 1.0, 1.5, 2.0),
    fractions: Iterable[float] = (0.5, 0.75, 1.0),
    hl_lookback: int = 252,
    hard_cap: int = 90,
) -> pd.DataFrame:
    """Run all OU exit variants on one setup, return a tidy summary frame."""
    components = components_from_pair(setup, all_setups)
    rows = []
    for k in ks:
        log = ou_time_stop_log(setup, y, components, k=k, hl_lookback=hl_lookback, hard_cap=hard_cap)
        rows.append(summarize_log(log, setup.name, f"ou_time_stop_k{k}"))
    for f in fractions:
        for req in ("any", "all"):
            log = ou_level_target_log(setup, y, components, fraction=f, require=req,
                                       hl_lookback=hl_lookback, hard_cap=hard_cap)
            rows.append(summarize_log(log, setup.name, f"ou_level_{req}_f{int(f*100)}p"))
    return pd.DataFrame(rows)
