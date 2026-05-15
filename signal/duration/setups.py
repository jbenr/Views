"""Setup battery for systematic signal evaluation.

A `Setup` is a (boolean trigger, direction) pair declaring "when these
conditions fire, bet that yields move in this direction." Mean-reversion and
momentum setups are structurally identical — they only differ in `direction`.

The battery evaluates each setup at multiple forward horizons against:
  - per-trade PnL (direction × forward yield change)
  - hit rate, profit factor, t-stat, annualized Sharpe
  - shuffled-trigger null distribution (significance vs random firing)
  - Spearman IC (if a continuous `score` is supplied)

Compose setups via boolean AND (`combine_pairs`) — only same-direction
pairs are combined. Lift-over-components is reported so you can see whether
a combo actually adds information vs just selecting a subset of one component.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import combinations, product
from typing import Optional

import numpy as np
import pandas as pd
from utils.rates import with_synthetic_real_rates

try:
    import polars as pl
except ImportError:  # keep the notebook usable if polars is not installed
    pl = None


# ── Core types ─────────────────────────────────────────────────────────

@dataclass
class Setup:
    """Atomic trade hypothesis: when `trigger` fires, bet `direction`.

    direction = +1 → bet 10y yield rises (short duration)
    direction = -1 → bet 10y yield falls (long duration)
    """
    name: str
    trigger: pd.Series       # boolean, indexed by date
    direction: int
    family: str = ""
    score: Optional[pd.Series] = None
    residual: Optional[pd.Series] = None  # raw mean-reverting series under the score (for OU fits)
    params: dict = field(default_factory=dict)


@dataclass
class SetupResult:
    name: str
    direction: int
    family: str
    horizon: int
    n_triggers: int
    mean_pnl_bps: float
    std_pnl_bps: float
    sharpe: float            # annualized: per-trade Sharpe × sqrt(252/horizon)
    hit_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    t_stat: float
    null_pct: float          # percentile of real per-trade Sharpe vs shuffled
    ic: Optional[float] = None


@dataclass
class PositionRule:
    """Rules for turning a signal state into actual trades."""
    exit_on_signal_false: bool = True
    max_holding_period: Optional[int] = None
    min_holding_period: int = 1
    enter_on_flip: bool = True
    exit_score_level: Optional[float] = None
    exit_score_operator: Optional[str] = None
    exit_on_any_component_score: bool = True
    name: str = "signal_false"

    def canonical_name(self) -> str:
        if self.exit_on_signal_false and self.exit_score_operator is None and self.max_holding_period is None:
            return "signal_false"
        if not self.exit_on_signal_false and self.exit_score_operator is None and self.max_holding_period is not None:
            return f"fixed_{self.max_holding_period}d"
        if self.exit_score_operator in {"abs<=", "abs<"} and self.exit_score_level is not None:
            any_all = "any" if self.exit_on_any_component_score else "all"
            level = _fmt_num(abs(float(self.exit_score_level)))
            suffix = f"_or_{self.max_holding_period}d" if self.max_holding_period is not None else ""
            return f"score_{any_all}_abs_le_{level}{suffix}"
        return self.name

    def with_canonical_name(self) -> "PositionRule":
        return replace(self, name=self.canonical_name())


# ── Helpers ────────────────────────────────────────────────────────────

def fwd_change(y: pd.Series, horizon: int) -> pd.Series:
    """Forward N-bar yield change in bps. Positive ⇒ yields rose."""
    return y.diff(horizon).shift(-horizon)


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython

        shell = get_ipython()
        return shell is not None and "zmqshell" in type(shell).__module__
    except Exception:
        return False


def _progress(iterable, total=None, desc: Optional[str] = None, enabled: Optional[bool] = None):
    """Notebook-friendly progress wrapper with a no-dependency fallback."""
    use_progress = _in_notebook() if enabled is None else enabled
    if not use_progress:
        return iterable
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, total=total, desc=desc, leave=False)
    except Exception:
        return iterable


def _rows_frame(
    rows: list[dict],
    sort_by: Optional[list[str]] = None,
    descending: Optional[list[bool]] = None,
) -> pd.DataFrame:
    """Build/sort result tables via Polars when available, then return pandas."""
    if not rows:
        return pd.DataFrame()

    if pl is not None:
        try:
            frame = pl.DataFrame(rows)
            if sort_by:
                frame = frame.sort(sort_by, descending=descending or False)
            return frame.to_pandas()
        except Exception:
            pass

    frame = pd.DataFrame(rows)
    if sort_by:
        desc = descending or [False] * len(sort_by)
        frame = frame.sort_values(sort_by, ascending=[not x for x in desc])
    return frame


def _trade_metrics(pnl: pd.Series) -> dict:
    pnl = pnl.dropna()
    if len(pnl) < 2 or pnl.std() == 0:
        return None
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    pf = wins.sum() / abs(losses.sum()) if len(losses) and losses.sum() != 0 else float("inf")
    return {
        "mean":            pnl.mean(),
        "std":             pnl.std(),
        "sharpe_pertrade": pnl.mean() / pnl.std(),
        "hit_rate":        (pnl > 0).mean(),
        "avg_win":         wins.mean() if len(wins) else 0.0,
        "avg_loss":        losses.mean() if len(losses) else 0.0,
        "profit_factor":   pf,
        "t_stat":          pnl.mean() / (pnl.std() / np.sqrt(len(pnl))),
    }


def performance_stats(trades: pd.DataFrame) -> dict:
    """Detailed performance stats from a position/trade log."""
    if trades is None or trades.empty or "pnl_bps" not in trades:
        return {}

    pnl = trades["pnl_bps"].dropna()
    if pnl.empty:
        return {}

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    cum = pnl.cumsum()
    peak = cum.cummax()
    dd = cum - peak
    hold = trades["holding_period"].dropna() if "holding_period" in trades else pd.Series(dtype=float)

    gross_win = wins.sum()
    gross_loss = losses.sum()
    total = pnl.sum()
    avg_win = wins.mean() if len(wins) else 0.0
    avg_loss = losses.mean() if len(losses) else 0.0
    std = pnl.std()
    downside = pnl[pnl < 0].std()
    pf = gross_win / abs(gross_loss) if gross_loss != 0 else np.inf
    payoff = avg_win / abs(avg_loss) if avg_loss != 0 else np.inf
    expectancy = pnl.mean()

    streak_win = streak_loss = max_win_streak = max_loss_streak = 0
    for x in pnl:
        if x > 0:
            streak_win += 1
            streak_loss = 0
        elif x < 0:
            streak_loss += 1
            streak_win = 0
        else:
            streak_win = 0
            streak_loss = 0
        max_win_streak = max(max_win_streak, streak_win)
        max_loss_streak = max(max_loss_streak, streak_loss)

    if "entry_date" in trades and "exit_date" in trades and len(trades):
        start = pd.to_datetime(trades["entry_date"]).min()
        end = pd.to_datetime(trades["exit_date"]).max()
        years = max((end - start).days / 365.25, np.nan)
    else:
        start = end = pd.NaT
        years = np.nan

    avg_hold = float(hold.mean()) if len(hold) else np.nan
    sharpe_pt = float(expectancy / std) if std and pd.notna(std) else np.nan
    # Annualized Sharpe assumes back-to-back deployment of capital at the
    # avg holding period — apples-to-apples across policies with different
    # hold horizons (a 2-day trade can fire ~25x more often than a 50-day one).
    sharpe_ann = (
        sharpe_pt * np.sqrt(252.0 / avg_hold)
        if pd.notna(sharpe_pt) and pd.notna(avg_hold) and avg_hold > 0
        else np.nan
    )
    return {
        "n_trades": int(len(pnl)),
        "wins": int(len(wins)),
        "losses": int(len(losses)),
        "hit_rate": float((pnl > 0).mean()),
        "total_pnl_bps": float(total),
        "mean_pnl_bps": float(expectancy),
        "median_pnl_bps": float(pnl.median()),
        "std_pnl_bps": float(std) if pd.notna(std) else np.nan,
        "best_trade_bps": float(pnl.max()),
        "worst_trade_bps": float(pnl.min()),
        "avg_win_bps": float(avg_win),
        "avg_loss_bps": float(avg_loss),
        "gross_win_bps": float(gross_win),
        "gross_loss_bps": float(gross_loss),
        "profit_factor": float(pf),
        "payoff_ratio": float(payoff),
        "expectancy_bps": float(expectancy),
        "sharpe_pertrade": sharpe_pt,
        "sharpe_ann": sharpe_ann,
        "sortino_pertrade": float(expectancy / downside) if downside and pd.notna(downside) else np.nan,
        "t_stat": float(expectancy / (std / np.sqrt(len(pnl)))) if std and pd.notna(std) else np.nan,
        "max_drawdown_bps": float(dd.min()),
        "max_drawdown_pct_of_gross_win": float(abs(dd.min()) / gross_win) if gross_win != 0 else np.nan,
        "max_win_streak": int(max_win_streak),
        "max_loss_streak": int(max_loss_streak),
        "avg_hold": avg_hold,
        "median_hold": float(hold.median()) if len(hold) else np.nan,
        "max_hold": float(hold.max()) if len(hold) else np.nan,
        "trades_per_year": float(len(pnl) / years) if years and pd.notna(years) and years > 0 else np.nan,
        "start": start,
        "end": end,
    }


def performance_stats_frame(trades: pd.DataFrame) -> pd.DataFrame:
    """One-column DataFrame for display in notebooks."""
    stats = performance_stats(trades)
    if not stats:
        return pd.DataFrame(columns=["metric", "value"])
    return pd.DataFrame({"metric": list(stats.keys()), "value": list(stats.values())})


# ── Evaluation ─────────────────────────────────────────────────────────

def evaluate(
    setup: Setup,
    y: pd.Series,
    horizon: int = 20,
    n_shuffles: int = 500,
    min_triggers: int = 20,
    rng: Optional[np.random.Generator] = None,
) -> Optional[SetupResult]:
    """Evaluate a single setup at one horizon.

    Returns None if fewer than `min_triggers` valid trigger events.
    """
    rng = rng if rng is not None else np.random.default_rng(0)

    fc = fwd_change(y, horizon)
    pnl_universe = setup.direction * fc

    trig = setup.trigger.reindex(y.index).fillna(False).astype(bool)
    triggered = pnl_universe[trig].dropna()

    if len(triggered) < min_triggers:
        return None

    m = _trade_metrics(triggered)
    if m is None:
        return None

    sharpe_ann = m["sharpe_pertrade"] * np.sqrt(252.0 / horizon)

    # Shuffled-trigger null: sample equal-N random timestamps from the universe
    pool = pnl_universe.dropna().to_numpy()
    n_trig = len(triggered)
    null_sharpes = np.empty(n_shuffles)
    for i in range(n_shuffles):
        sub = rng.choice(pool, size=n_trig, replace=False)
        s = sub.std()
        null_sharpes[i] = sub.mean() / s if s > 0 else 0.0
    null_pct = float((null_sharpes < m["sharpe_pertrade"]).mean() * 100)

    # Spearman IC if continuous score present
    ic = None
    if setup.score is not None:
        score_aligned = setup.score.reindex(y.index)
        joint = pd.DataFrame({"s": score_aligned, "r": pnl_universe}).dropna()
        if len(joint) > 30:
            ic = float(joint["s"].corr(joint["r"], method="spearman"))

    return SetupResult(
        name=setup.name,
        direction=setup.direction,
        family=setup.family,
        horizon=horizon,
        n_triggers=n_trig,
        mean_pnl_bps=m["mean"],
        std_pnl_bps=m["std"],
        sharpe=sharpe_ann,
        hit_rate=m["hit_rate"],
        avg_win=m["avg_win"],
        avg_loss=m["avg_loss"],
        profit_factor=m["profit_factor"],
        t_stat=m["t_stat"],
        null_pct=null_pct,
        ic=ic,
    )


def evaluate_battery(
    setups: list[Setup],
    y: pd.Series,
    horizons: list[int] = (5, 20, 60),
    n_shuffles: int = 500,
    min_triggers: int = 20,
    show_progress: Optional[bool] = None,
) -> pd.DataFrame:
    rows = []
    tasks = product(setups, horizons)
    total = len(setups) * len(horizons)
    for s, h in _progress(tasks, total=total, desc="evaluate battery", enabled=show_progress):
        r = evaluate(s, y, horizon=h, n_shuffles=n_shuffles, min_triggers=min_triggers)
        if r is not None:
            rows.append(vars(r))
    return _rows_frame(rows)


def find_setup(setups: list[Setup], name: str) -> Setup:
    """Find a setup by exact name."""
    for s in setups:
        if s.name == name:
            return s
    raise KeyError(f"Setup not found: {name}")


def component_names(setup: Setup) -> list[str]:
    """Return component setup names for an AND-combo, else the setup itself."""
    if setup.params.get("a") and setup.params.get("b"):
        return [setup.params["a"], setup.params["b"]]
    if " & " in setup.name:
        return setup.name.split(" & ")
    return [setup.name]


def describe_setup(setup: Setup, setup_lookup: Optional[dict[str, Setup]] = None) -> str:
    """Plain-English description of a setup or pair setup."""
    if " & " in setup.name:
        parts = []
        if setup_lookup is not None:
            for nm in component_names(setup):
                child = setup_lookup.get(nm)
                parts.append(describe_setup(child, setup_lookup=None) if child is not None else nm)
        else:
            parts = component_names(setup)
        direction = "10y yield rises" if setup.direction == 1 else "10y yield falls"
        return (
            f"{setup.name}: AND-combo. Trigger only when both components fire. "
            f"Trade direction: bet {direction}. Components: " + " | ".join(parts)
        )

    p = setup.params
    direction = "10y yield rises" if setup.direction == 1 else "10y yield falls"

    if setup.name.startswith("momentum_move_") or setup.name.startswith("reversal_move_"):
        mode = p.get("mode", setup.name.split("_", 1)[0])
        raw_dir = "up" if p.get("direction") == 1 else "down"
        return (
            f"{setup.name}: {mode} after a large move. Trigger when 10y moved "
            f"{raw_dir} by at least {p.get('min_move_bps')} bps over "
            f"{p.get('lookback')} trading days. Trade direction: bet {direction}."
        )

    if setup.name.startswith("val_"):
        side = p.get("side")
        z = p.get("z_thresh")
        lb = p.get("lookback")
        tail = "high/rich" if side == "high" else "low/cheap"
        return (
            f"{setup.name}: residual valuation. Rolling regression of 10y changes "
            f"against the named curve anchor over {lb} days; cumulative residual is "
            f"OU-zscored. Trigger when residual z is {tail} beyond +/-{z}. "
            f"Trade direction: bet {direction}."
        )

    if setup.name.startswith("spread_"):
        side = p.get("side")
        z = p.get("z_thresh")
        win = p.get("win")
        tail = "high" if side == "high" else "low"
        return (
            f"{setup.name}: curve-spread z-score valuation. Trigger when spread "
            f"is {tail} versus its {win}d rolling z-score beyond +/-{z}. "
            f"Trade direction: bet {direction}."
        )

    if setup.name.startswith("mom_"):
        return f"{setup.name}: consecutive-day momentum. Trade direction: bet {direction}."
    if setup.name.startswith("mr_at_"):
        return (
            f"{setup.name}: yield near trailing {p.get('win')}d {p.get('side')} "
            f"within {p.get('bps_tol')} bps. Trade direction: bet {direction}."
        )
    if setup.name.startswith("mr_bb_") or setup.name.startswith("mr_ma_dist_"):
        return (
            f"{setup.name}: z-score mean reversion. Trigger on {p.get('side')} "
            f"tail beyond +/-{p.get('z_thresh')} using {p.get('win')}d window. "
            f"Trade direction: bet {direction}."
        )
    if setup.name.startswith("mr_rsi_"):
        return (
            f"{setup.name}: RSI mean reversion on 10y changes using {p.get('win')}d "
            f"window. Trade direction: bet {direction}."
        )
    if setup.name.startswith("breakout_"):
        return (
            f"{setup.name}: trailing range breakout over {p.get('win')}d with "
            f"{p.get('bps_buffer')} bps buffer. Trade direction: bet {direction}."
        )
    if setup.name.startswith("trend_ma_"):
        return (
            f"{setup.name}: moving-average trend state, fast={p.get('fast')}d "
            f"versus slow={p.get('slow')}d. Trade direction: bet {direction}."
        )

    return f"{setup.name}: family={setup.family}. Trade direction: bet {direction}."


def setup_signal_frame(
    setup: Setup,
    y: pd.Series,
    components: Optional[list[Setup]] = None,
) -> pd.DataFrame:
    """Build a date-indexed signal frame for charting and inspection."""
    out = pd.DataFrame(index=y.index)
    out["y"] = y
    out["trigger"] = setup.trigger.reindex(y.index).fillna(False).astype(bool)
    out["direction"] = np.where(out["trigger"], setup.direction, np.nan)
    if setup.score is not None:
        out["score"] = setup.score.reindex(y.index)

    for child in components or []:
        prefix = child.name
        out[f"{prefix}__trigger"] = child.trigger.reindex(y.index).fillna(False).astype(bool)
        if child.score is not None:
            out[f"{prefix}__score"] = child.score.reindex(y.index)
    return out


def trade_log(
    setup: Setup,
    y: pd.Series,
    horizon: int = 20,
    components: Optional[list[Setup]] = None,
) -> pd.DataFrame:
    """Return one row per triggered trade with forward return and diagnostics."""
    fc = fwd_change(y, horizon)
    trig = setup.trigger.reindex(y.index).fillna(False).astype(bool)
    entries = y[trig].index
    base_cols = [
        "trigger_date", "exit_date", "horizon", "direction",
        "entry_yield_bps", "exit_yield_bps", "fwd_change_bps", "pnl_bps",
    ]
    extra_cols = []
    if setup.score is not None:
        extra_cols.append("score")
    for child in components or []:
        extra_cols.append(f"{child.name}__trigger")
        if child.score is not None:
            extra_cols.append(f"{child.name}__score")

    rows = []
    pos = pd.Series(np.arange(len(y)), index=y.index)
    for dt in entries:
        i = int(pos.loc[dt])
        if i + horizon >= len(y) or pd.isna(fc.loc[dt]):
            continue
        exit_dt = y.index[i + horizon]
        row = {
            "trigger_date": dt,
            "exit_date": exit_dt,
            "horizon": horizon,
            "direction": setup.direction,
            "entry_yield_bps": float(y.loc[dt]),
            "exit_yield_bps": float(y.loc[exit_dt]),
            "fwd_change_bps": float(fc.loc[dt]),
            "pnl_bps": float(setup.direction * fc.loc[dt]),
        }
        if setup.score is not None:
            row["score"] = float(setup.score.reindex(y.index).loc[dt])
        for child in components or []:
            row[f"{child.name}__trigger"] = bool(child.trigger.reindex(y.index).fillna(False).loc[dt])
            if child.score is not None:
                val = child.score.reindex(y.index).loc[dt]
                row[f"{child.name}__score"] = float(val) if pd.notna(val) else np.nan
        rows.append(row)

    return pd.DataFrame(rows, columns=base_cols + extra_cols)


def _episode_entries(
    trigger: pd.Series,
    y: pd.Series,
    horizon: int,
    cooldown: Optional[int] = None,
) -> pd.Index:
    """First trigger date of each contiguous trigger episode.

    cooldown defaults to `horizon`, so separate episodes still cannot create
    overlapping forward-return trades.
    """
    trig = trigger.reindex(y.index).fillna(False).astype(bool)
    starts = trig & ~trig.shift(1, fill_value=False)
    candidates = list(y.index[starts])
    if not candidates:
        return pd.Index([], name=y.index.name)

    gap = horizon if cooldown is None else cooldown
    pos = pd.Series(np.arange(len(y)), index=y.index)
    selected = []
    last_i = -10**9
    for dt in candidates:
        i = int(pos.loc[dt])
        if i - last_i < gap:
            continue
        selected.append(dt)
        last_i = i
    return pd.Index(selected, name=y.index.name)


def episode_trade_log(
    setup: Setup,
    y: pd.Series,
    horizon: int = 20,
    components: Optional[list[Setup]] = None,
    cooldown: Optional[int] = None,
) -> pd.DataFrame:
    """One trade per signal episode, not one trade per triggered date.

    This is the safer interpretation for research because many setup triggers
    persist for days/weeks and forward returns overlap heavily.
    """
    entries = _episode_entries(setup.trigger, y, horizon=horizon, cooldown=cooldown)
    episode_trigger = pd.Series(False, index=y.index)
    episode_trigger.loc[entries] = True
    episode_setup = Setup(
        name=f"{setup.name}__episodes",
        trigger=episode_trigger,
        direction=setup.direction,
        family=setup.family,
        score=setup.score,
        params=setup.params,
    )
    out = trade_log(episode_setup, y, horizon=horizon, components=components)
    if len(out):
        out.insert(0, "episode_id", np.arange(len(out)))
    elif "episode_id" not in out.columns:
        out.insert(0, "episode_id", pd.Series(dtype=int))
    return out


def position_trade_log(
    setup: Setup,
    y: pd.Series,
    rule: Optional[PositionRule] = None,
    components: Optional[list[Setup]] = None,
) -> pd.DataFrame:
    """Convert signal state into non-overlapping trades.

    Entry is when the signal turns true. If the signal remains true tomorrow,
    that is still the same trade. By default, exit when the signal turns false.
    Set `max_holding_period` to force exits after N bars even if signal stays on.
    """
    rule = (rule or PositionRule()).with_canonical_name()
    trig = setup.trigger.reindex(y.index).fillna(False).astype(bool)
    setup_score = setup.score.reindex(y.index) if setup.score is not None else None
    component_scores = [
        child.score.reindex(y.index)
        for child in components or []
        if child.score is not None
    ]
    base_cols = [
        "trade_id", "entry_date", "exit_date", "holding_period",
        "direction", "entry_yield_bps", "exit_yield_bps",
        "yield_change_bps", "pnl_bps", "entry_reason", "exit_reason", "exit_policy",
    ]
    extra_cols = []
    if setup.score is not None:
        extra_cols.extend(["entry_score", "exit_score"])
    for child in components or []:
        extra_cols.append(f"{child.name}__entry_trigger")
        extra_cols.append(f"{child.name}__exit_trigger")
        if child.score is not None:
            extra_cols.append(f"{child.name}__entry_score")
            extra_cols.append(f"{child.name}__exit_score")

    rows = []
    n = len(y)
    i = 0
    trade_id = 0
    while i < n:
        if not trig.iloc[i] or pd.isna(y.iloc[i]):
            i += 1
            continue

        entry_i = i
        exit_i = entry_i
        exit_reason = "end_of_data"
        while exit_i + 1 < n:
            next_i = exit_i + 1
            held = next_i - entry_i
            score_hits = []
            if setup_score is not None:
                score_hits.append(_score_exit_hit(
                    setup_score.iloc[next_i],
                    rule.exit_score_operator,
                    rule.exit_score_level,
                ))
            for score in component_scores:
                score_hits.append(_score_exit_hit(
                    score.iloc[next_i],
                    rule.exit_score_operator,
                    rule.exit_score_level,
                ))
            if score_hits and held >= rule.min_holding_period:
                score_exit = any(score_hits) if rule.exit_on_any_component_score else all(score_hits)
                if score_exit:
                    exit_i = next_i
                    exit_reason = "score_exit"
                    break
            if rule.max_holding_period is not None and held >= rule.max_holding_period:
                exit_i = next_i
                exit_reason = "max_holding_period"
                break
            if (
                rule.exit_on_signal_false
                and held >= rule.min_holding_period
                and not bool(trig.iloc[next_i])
            ):
                exit_i = next_i
                exit_reason = "signal_false"
                break
            exit_i = next_i

        if exit_i > entry_i and pd.notna(y.iloc[exit_i]):
            change = float(y.iloc[exit_i] - y.iloc[entry_i])
            row = {
                "trade_id": trade_id,
                "entry_date": y.index[entry_i],
                "exit_date": y.index[exit_i],
                "holding_period": exit_i - entry_i,
                "direction": setup.direction,
                "entry_yield_bps": float(y.iloc[entry_i]),
                "exit_yield_bps": float(y.iloc[exit_i]),
                "yield_change_bps": change,
                "pnl_bps": float(setup.direction * change),
                "entry_reason": "signal_true",
                "exit_reason": exit_reason,
                "exit_policy": rule.name,
            }
            if setup.score is not None:
                score = setup.score.reindex(y.index)
                row["entry_score"] = float(score.iloc[entry_i]) if pd.notna(score.iloc[entry_i]) else np.nan
                row["exit_score"] = float(score.iloc[exit_i]) if pd.notna(score.iloc[exit_i]) else np.nan
            for child in components or []:
                child_trig = child.trigger.reindex(y.index).fillna(False)
                row[f"{child.name}__entry_trigger"] = bool(child_trig.iloc[entry_i])
                row[f"{child.name}__exit_trigger"] = bool(child_trig.iloc[exit_i])
                if child.score is not None:
                    child_score = child.score.reindex(y.index)
                    entry_val = child_score.iloc[entry_i]
                    exit_val = child_score.iloc[exit_i]
                    row[f"{child.name}__entry_score"] = float(entry_val) if pd.notna(entry_val) else np.nan
                    row[f"{child.name}__exit_score"] = float(exit_val) if pd.notna(exit_val) else np.nan
            rows.append(row)
            trade_id += 1

        # Do not re-enter while the same signal episode is still true.
        i = max(exit_i + 1, entry_i + 1)
        while i < n and bool(trig.iloc[i]):
            i += 1

    return pd.DataFrame(rows, columns=base_cols + extra_cols)


def evaluate_trade_log(
    setup: Setup,
    y: pd.Series,
    horizon: int = 20,
    mode: str = "daily",
    min_triggers: int = 20,
    cooldown: Optional[int] = None,
) -> Optional[SetupResult]:
    """Evaluate from daily trigger trades or de-clustered episode trades."""
    if mode == "positions":
        log = position_trade_log(
            setup,
            y,
            rule=PositionRule(max_holding_period=horizon),
        )
    elif mode == "episodes":
        log = episode_trade_log(setup, y, horizon=horizon, cooldown=cooldown)
    elif mode == "daily":
        log = trade_log(setup, y, horizon=horizon)
    else:
        raise ValueError("mode must be 'daily', 'episodes', or 'positions'")

    if len(log) < min_triggers:
        return None
    m = _trade_metrics(log["pnl_bps"])
    if m is None:
        return None
    sharpe_ann = m["sharpe_pertrade"] * np.sqrt(252.0 / horizon)
    return SetupResult(
        name=setup.name,
        direction=setup.direction,
        family=setup.family,
        horizon=horizon,
        n_triggers=len(log),
        mean_pnl_bps=m["mean"],
        std_pnl_bps=m["std"],
        sharpe=sharpe_ann,
        hit_rate=m["hit_rate"],
        avg_win=m["avg_win"],
        avg_loss=m["avg_loss"],
        profit_factor=m["profit_factor"],
        t_stat=m["t_stat"],
        null_pct=np.nan,
        ic=None,
    )


def default_exit_policies(setup: Setup, horizons: list[int] = (5, 10, 20, 40, 60)) -> list[PositionRule]:
    """Candidate deterministic exit policies for one setup.

    These are research candidates, not recommendations. Compare them out of
    sample before trusting one.
    """
    policies = [
        PositionRule(exit_on_signal_false=True, name="signal_false"),
    ]
    for h in horizons:
        policies.append(PositionRule(
            exit_on_signal_false=False,
            max_holding_period=h,
            name=f"fixed_{h}d",
        ))

    has_score = setup.score is not None or " & " in setup.name
    if has_score:
        # For z-score-like signals, exit when edge has mostly normalized.
        policies.extend([
            PositionRule(
                exit_on_signal_false=False,
                exit_score_operator="abs<=",
                exit_score_level=0.5,
                exit_on_any_component_score=True,
                max_holding_period=60,
                name="score_any_abs_le_0p5_or_60d",
            ),
            PositionRule(
                exit_on_signal_false=False,
                exit_score_operator="abs<=",
                exit_score_level=0.5,
                exit_on_any_component_score=False,
                max_holding_period=60,
                name="score_all_abs_le_0p5_or_60d",
            ),
            PositionRule(
                exit_on_signal_false=False,
                exit_score_operator="abs<=",
                exit_score_level=1.0,
                exit_on_any_component_score=True,
                max_holding_period=40,
                name="score_any_abs_le_1_or_40d",
            ),
        ])
    return policies


def compare_exit_policies(
    setup: Setup,
    y: pd.Series,
    policies: list[PositionRule],
    components: Optional[list[Setup]] = None,
    min_trades: int = 5,
    show_progress: Optional[bool] = None,
) -> pd.DataFrame:
    """Evaluate a setup under multiple position exit policies."""
    rows = []
    for rule in _progress(
        policies,
        total=len(policies),
        desc=f"exit policies: {setup.name}",
        enabled=show_progress,
    ):
        log = position_trade_log(setup, y, rule=rule, components=components)
        if len(log) < min_trades:
            continue
        m = _trade_metrics(log["pnl_bps"])
        if m is None:
            continue
        avg_hold = float(log["holding_period"].mean())
        sharpe_ann = (
            m["sharpe_pertrade"] * np.sqrt(252.0 / avg_hold)
            if pd.notna(m["sharpe_pertrade"]) and avg_hold > 0
            else np.nan
        )
        rows.append({
            "exit_policy": rule.name,
            "n_trades": len(log),
            "avg_hold": avg_hold,
            "median_hold": log["holding_period"].median(),
            "mean_pnl_bps": m["mean"],
            "std_pnl_bps": m["std"],
            "sharpe_pertrade": m["sharpe_pertrade"],
            "sharpe_ann": sharpe_ann,
            "hit_rate": m["hit_rate"],
            "profit_factor": m["profit_factor"],
            "t_stat": m["t_stat"],
        })
    return _rows_frame(
        rows,
        sort_by=["sharpe_ann", "sharpe_pertrade", "mean_pnl_bps"],
        descending=[True, True, True],
    )


def exit_policy_report(
    setup_name: str,
    setups: list[Setup],
    y: pd.Series,
    horizons: list[int] = (5, 10, 20, 40, 60),
    min_trades: int = 5,
    show_progress: Optional[bool] = None,
) -> pd.DataFrame:
    """Find setup/components and compare default exit candidates."""
    lookup = {s.name: s for s in setups}
    setup = find_setup(setups, setup_name)
    components = [lookup[nm] for nm in component_names(setup) if nm in lookup and nm != setup.name]
    return compare_exit_policies(
        setup,
        y,
        default_exit_policies(setup, horizons=horizons),
        components=components,
        min_trades=min_trades,
        show_progress=show_progress,
    )


def evaluate_battery_from_trades(
    setups: list[Setup],
    y: pd.Series,
    horizons: list[int] = (5, 20, 60),
    mode: str = "positions",
    min_triggers: int = 10,
    cooldown: Optional[int] = None,
    show_progress: Optional[bool] = None,
) -> pd.DataFrame:
    """Evaluate a battery using daily or de-clustered trade logs."""
    rows = []
    tasks = product(setups, horizons)
    total = len(setups) * len(horizons)
    for s, h in _progress(tasks, total=total, desc="trade-log battery", enabled=show_progress):
        r = evaluate_trade_log(
            s, y, horizon=h, mode=mode,
            min_triggers=min_triggers, cooldown=cooldown,
        )
        if r is not None:
            row = vars(r)
            row["trade_mode"] = mode
            rows.append(row)
    return _rows_frame(rows)


def position_scoreboard(
    setups: list[Setup],
    y: pd.Series,
    policies: Optional[list[PositionRule]] = None,
    mode: str = "positions",
    min_trades: int = 5,
    show_progress: Optional[bool] = None,
) -> pd.DataFrame:
    """Rank setups by actual trades, not trigger-days.

    For mode='positions', each policy produces a row per setup. For other modes,
    use evaluate_battery_from_trades directly.
    """
    if mode != "positions":
        return evaluate_battery_from_trades(
            setups,
            y,
            mode=mode,
            min_triggers=min_trades,
            show_progress=show_progress,
        )

    policies = policies or [
        PositionRule(exit_on_signal_false=True, name="signal_false"),
        PositionRule(exit_on_signal_false=False, max_holding_period=5, name="fixed_5d"),
        PositionRule(exit_on_signal_false=False, max_holding_period=10, name="fixed_10d"),
        PositionRule(exit_on_signal_false=False, max_holding_period=20, name="fixed_20d"),
        PositionRule(exit_on_signal_false=False, max_holding_period=40, name="fixed_40d"),
        PositionRule(exit_on_signal_false=False, max_holding_period=60, name="fixed_60d"),
    ]

    rows = []
    tasks = product(setups, policies)
    total = len(setups) * len(policies)
    total_days = int(y.dropna().shape[0])
    for s, rule in _progress(tasks, total=total, desc="position scoreboard", enabled=show_progress):
        log = position_trade_log(s, y, rule=rule)
        stats = performance_stats(log)
        if not stats or stats["n_trades"] < min_trades:
            continue
        days_in = float(log["holding_period"].sum()) if "holding_period" in log else 0.0
        stats["time_in_market"] = days_in / total_days if total_days else np.nan
        rows.append({
            "name": s.name,
            "family": s.family,
            "direction": s.direction,
            "exit_policy": rule.name,
            **stats,
        })

    return _rows_frame(
        rows,
        sort_by=["sharpe_ann", "sharpe_pertrade", "mean_pnl_bps", "n_trades"],
        descending=[True, True, True, True],
    )


def rank_engagement(
    scoreboard: pd.DataFrame,
    *,
    min_hold: float = 5.0,
    max_hold: float = 30.0,
    min_time_in_market: float = 0.30,
    min_trades: int = 20,
    min_payoff: float = 1.0,
) -> pd.DataFrame:
    """Filter scoreboard for 'engaged, asymmetric-payoff' strategies and rank them.

    Targets:
      - medium holding period (min_hold ≤ avg_hold ≤ max_hold)
      - actually in the market (time_in_market ≥ floor)
      - meaningful sample size (n_trades ≥ floor)
      - asymmetric payoff (payoff_ratio ≥ floor, i.e. avg_win ≥ avg_loss)

    Composite score = payoff_ratio · √n_trades · time_in_market — rewards big
    wins / small losses, scaled by both how often it fires and how invested it
    keeps you. Hit rate is *not* part of the rank (deliberately).
    """
    if scoreboard is None or scoreboard.empty:
        return scoreboard
    df = scoreboard.copy()
    needed = {"avg_hold", "time_in_market", "n_trades", "payoff_ratio"}
    missing = needed - set(df.columns)
    if missing:
        raise KeyError(f"scoreboard missing columns: {sorted(missing)}")

    mask = (
        df["avg_hold"].between(min_hold, max_hold)
        & (df["time_in_market"] >= min_time_in_market)
        & (df["n_trades"] >= min_trades)
        & (df["payoff_ratio"] >= min_payoff)
    )
    df = df.loc[mask].copy()
    if df.empty:
        return df

    df["engagement_score"] = (
        df["payoff_ratio"] * np.sqrt(df["n_trades"]) * df["time_in_market"]
    )
    return df.sort_values("engagement_score", ascending=False).reset_index(drop=True)


def threshold_lines(setup: Setup) -> list[float]:
    """Threshold levels to draw on score charts when available."""
    p = setup.params
    if "z_thresh" in p:
        z = abs(float(p["z_thresh"]))
        side = p.get("side")
        if side == "high":
            return [z]
        if side == "low":
            return [-z]
        return [-z, z]
    if setup.name.startswith("mr_rsi_"):
        side = p.get("side")
        return [float(p.get("high", 70.0))] if side == "high" else [float(p.get("low", 30.0))]
    if setup.name.startswith("momentum_move_") or setup.name.startswith("reversal_move_"):
        move = float(p.get("min_move_bps", 0.0))
        return [move if p.get("direction") == 1 else -move]
    return []


def plot_setup_diagnostics(
    setup: Setup,
    y: pd.Series,
    horizon: int = 20,
    components: Optional[list[Setup]] = None,
    trades: Optional[pd.DataFrame] = None,
    start=None,
    end=None,
    title: Optional[str] = None,
    interactive: bool = True,
):
    """Plot yield with trade arrows, plus score/threshold panels when available."""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from utils.viz import Viz, figure_copy_html

    def _in_notebook() -> bool:
        try:
            from IPython import get_ipython
            shell = get_ipython()
            return shell is not None and "zmqshell" in type(shell).__module__
        except Exception:
            return False

    components = components or []
    frame = setup_signal_frame(setup, y, components=components)
    if start is not None or end is not None:
        frame = frame.loc[start:end]
    full = frame.copy()

    score_cols = []
    if "score" in frame:
        score_cols.append(("score", setup))
    for child in components:
        col = f"{child.name}__score"
        if col in frame:
            score_cols.append((col, child))

    def _score_shade(ax_i, source_setup, y_min, y_max):
        if source_setup.name.lower().startswith("trend_ma_"):
            ax_i.axhline(0, color="#666", linestyle=":", linewidth=1.0, label="crossover 0")
            return
        for lvl in threshold_lines(source_setup):
            if lvl > 0:
                ax_i.axhspan(lvl, y_max, color="#c0392b", alpha=0.08)
            else:
                ax_i.axhspan(y_min, lvl, color="#18864b", alpha=0.08)

    def _score_is_bar(source_setup: Setup) -> bool:
        return source_setup.name.startswith("momentum_move_") or source_setup.name.startswith("reversal_move_")

    def _score_is_trend(source_setup: Setup) -> bool:
        return source_setup.name.lower().startswith("trend_ma_")

    chart_title = title or f"{setup.name} | horizon={horizon}d"
    viz = Viz()

    def _style_panel(ax_i, yaxis_title: str, panel_title: Optional[str] = None):
        viz._style_ax(ax_i, title=panel_title, yaxis_title=yaxis_title)
        viz._legend(ax_i)

    def _render(window: pd.DataFrame, rebase_pnl: bool = True, include_title: bool = False):
        pnl_panel = trades is not None and not trades.empty and {"exit_date", "pnl_bps"}.issubset(trades.columns)
        n_panels = 1 + len(score_cols) + int(pnl_panel)
        fig, axes = plt.subplots(
            n_panels,
            1,
            figsize=(13, 4.1 + 2.65 * max(0, n_panels - 1)),
            sharex=True,
            gridspec_kw={"height_ratios": [2.2] + [1.0] * (n_panels - 1)},
        )
        if n_panels == 1:
            axes = [axes]
        fig.patch.set_facecolor("white")
        for ax_i in axes:
            ax_i.set_facecolor(viz.BG_COLOR)

        ax = axes[0]
        ax.plot(window.index, window["y"], color=viz.colors[0], linewidth=1.5, label="10y")
        hits = window[window["trigger"]]
        if setup.direction == 1:
            ax.scatter(hits.index, hits["y"], marker="^", color="#27AE60", s=55, label="bet yield up", zorder=4)
        else:
            ax.scatter(hits.index, hits["y"], marker="v", color="#C0392B", s=55, label="bet yield down", zorder=4)
        if include_title:
            viz._style_ax(ax, title=chart_title, yaxis_title="10y yield, bps")
            viz._legend(ax)
        else:
            _style_panel(ax, "10y yield, bps", "10y yield and trade signals")

        next_ax = 1
        for ax_i, (col, source_setup) in zip(axes[next_ax:], score_cols):
            if _score_is_bar(source_setup):
                vals = window[col].fillna(0.0)
                colors = np.where(vals >= 0, "#27AE60", "#C0392B")
                ax_i.bar(window.index, vals, color=colors, alpha=0.70, width=1.0, label=col.replace("__score", " score"))
            else:
                ax_i.plot(window.index, window[col], color=viz.colors[3], linewidth=1.5, label=col.replace("__score", " score"))
            y_min = window[col].min()
            y_max = window[col].max()
            for lvl in threshold_lines(source_setup):
                y_min = min(y_min, lvl)
                y_max = max(y_max, lvl)
            pad = max((y_max - y_min) * 0.08, 0.25) if pd.notna(y_min) and pd.notna(y_max) else 0.25
            y_min, y_max = y_min - pad, y_max + pad
            _score_shade(ax_i, source_setup, y_min, y_max)
            for lvl in threshold_lines(source_setup):
                ax_i.axhline(lvl, color="#C0392B", linestyle="--", linewidth=1.0, label=f"threshold {lvl:g}")
            trig_col = "trigger" if col == "score" else col.replace("__score", "__trigger")
            if trig_col in window:
                trig_points = window[window[trig_col]]
                if _score_is_trend(source_setup):
                    fill_color = "#27AE60" if source_setup.direction == 1 else "#C0392B"
                    ax_i.fill_between(
                        window.index,
                        0,
                        window[col],
                        where=window[trig_col].to_numpy(dtype=bool),
                        color=fill_color,
                        alpha=0.12,
                        interpolate=True,
                        label="signal region",
                    )
                else:
                    ax_i.fill_between(
                        window.index,
                        y_min,
                        y_max,
                        where=window[trig_col].to_numpy(dtype=bool),
                        color="#D35400",
                        alpha=0.06,
                        step="mid",
                        label="trigger active",
                    )
                    ax_i.scatter(trig_points.index, trig_points[col], color="#D35400", s=14, zorder=3)
            ax_i.set_ylim(y_min, y_max)
            score_title = col.replace("__score", " score")
            _style_panel(ax_i, "score", score_title)
            next_ax += 1

        if pnl_panel:
            ax_pnl = axes[next_ax]
            pnl = pd.Series(0.0, index=full.index)
            pnl_by_exit = (
                trades.assign(exit_date=pd.to_datetime(trades["exit_date"]))
                .groupby("exit_date")["pnl_bps"]
                .sum()
            )
            common_exit_dates = pnl_by_exit.index.intersection(full.index)
            if len(common_exit_dates):
                pnl.loc[common_exit_dates] = pnl_by_exit.loc[common_exit_dates].to_numpy()
            cum = pnl.cumsum()
            cum_window = cum.reindex(window.index).ffill().fillna(0.0)
            if rebase_pnl and len(cum_window):
                cum_window = cum_window - cum_window.iloc[0]
            ax_pnl.plot(cum_window.index, cum_window, color="#16A085", linewidth=1.5, label="cumulative pnl")
            ax_pnl.axhline(0, color="#666", linestyle=":", linewidth=1.0)
            _style_panel(ax_pnl, "cum pnl, bps", "cumulative trade pnl")

        for ax_i in axes:
            viz._format_dates(ax_i, window.index.min(), window.index.max())
        # Only the bottom panel shows x-tick labels (sharex). Drop its legend
        # below those labels so they don't collide.
        bottom_leg = axes[-1].get_legend()
        if bottom_leg is not None:
            bottom_leg.set_bbox_to_anchor((0, -0.42))
        fig.subplots_adjust(bottom=0.12, hspace=0.62)
        return fig

    if not interactive or not _in_notebook():
        return _render(frame, include_title=True)

    import ipywidgets as widgets
    from IPython.display import HTML, display, clear_output

    out = widgets.Output()
    copy_out = widgets.Output(layout=widgets.Layout(width="260px", height="32px"))
    date_layout = widgets.Layout(width="185px")
    start_picker = widgets.DatePicker(description="Start:", value=full.index.min().date(), layout=date_layout)
    end_picker = widgets.DatePicker(description="End:", value=full.index.max().date(), layout=date_layout)

    button_defs = [
        ("1M", pd.DateOffset(months=1)),
        ("3M", pd.DateOffset(months=3)),
        ("6M", pd.DateOffset(months=6)),
        ("YTD", "ytd"),
        ("1Y", pd.DateOffset(years=1)),
        ("2Y", pd.DateOffset(years=2)),
        ("5Y", pd.DateOffset(years=5)),
        ("10Y", pd.DateOffset(years=10)),
        ("ALL", "all"),
    ]
    buttons = [widgets.Button(description=label, layout=widgets.Layout(width="52px", height="30px")) for label, _ in button_defs]
    buttons[-1].button_style = "warning"
    active_range = {"label": "ALL Window"}
    suppress_picker = {"value": False}

    def _window_title() -> str:
        label = active_range.get("label")
        return f"{chart_title} ({label})" if label else chart_title

    def _refresh_title():
        title_widget.value = f'<b style="font-size:13px; color:#333;">{_window_title().upper()}</b>'

    def _highlight_active(label: Optional[str]):
        active_range["label"] = f"{label} Window" if label else None
        for button in buttons:
            button.button_style = "warning" if button.description == label else ""

    def _slice_and_draw(start_dt, end_dt):
        if start_dt is None:
            start_dt = full.index.min()
        else:
            start_dt = pd.Timestamp(start_dt)
        if end_dt is None:
            end_dt = full.index.max()
        else:
            end_dt = pd.Timestamp(end_dt)
        window = full.loc[(full.index >= start_dt) & (full.index <= end_dt)]
        if window.empty:
            window = full.iloc[-1:]
        _refresh_title()
        fig = _render(window, rebase_pnl=True)
        with out:
            clear_output(wait=True)
            display(fig)
        with copy_out:
            clear_output(wait=True)
            display(HTML(figure_copy_html(fig, title=_window_title())))
        plt.close(fig)
        return window

    def _set_range(label, kind):
        end_dt = full.index.max()
        if kind == "all":
            start_dt = full.index.min()
        elif kind == "ytd":
            start_dt = pd.Timestamp(year=end_dt.year, month=1, day=1)
        else:
            start_dt = end_dt - kind
        suppress_picker["value"] = True
        try:
            start_picker.value = max(start_dt, full.index.min()).date()
            end_picker.value = end_dt.date()
        finally:
            suppress_picker["value"] = False
        _highlight_active(label)
        _slice_and_draw(start_picker.value, end_picker.value)

    for button, (label, kind) in zip(buttons, button_defs):
        button.on_click(lambda _b, lbl=label, k=kind: _set_range(lbl, k))

    def _picker_changed(_change):
        if suppress_picker["value"]:
            return
        _highlight_active(None)
        if start_picker.value and end_picker.value:
            active_range["label"] = f"{start_picker.value} to {end_picker.value}"
        _slice_and_draw(start_picker.value, end_picker.value)

    start_picker.observe(_picker_changed, names="value")
    end_picker.observe(_picker_changed, names="value")

    title_widget = widgets.HTML(
        value=f'<b style="font-size:13px; color:#333;">{_window_title().upper()}</b>'
    )
    title_spacer = widgets.Box(layout=widgets.Layout(flex="1 1 auto"))
    title_row = widgets.HBox(
        [title_widget, title_spacer, copy_out],
        layout=widgets.Layout(width="100%", align_items="center", margin="0 0 4px 0"),
    )
    date_controls = widgets.HBox(
        [start_picker, end_picker],
        layout=widgets.Layout(gap="6px", flex_flow="row wrap", align_items="center"),
    )
    range_controls = widgets.HBox(
        buttons,
        layout=widgets.Layout(gap="3px", flex_flow="row wrap", align_items="center"),
    )
    controls = widgets.HBox(
        [date_controls, range_controls],
        layout=widgets.Layout(gap="8px", align_items="center", flex_flow="row wrap", margin="0 0 8px 0"),
    )
    box = widgets.VBox([title_row, controls, out])
    display(box)
    _slice_and_draw(start_picker.value, end_picker.value)
    return box


def setup_report(
    setup_name: str,
    setups: list[Setup],
    y: pd.Series,
    horizon: int = 20,
    trade_mode: str = "positions",
    position_rule: Optional[PositionRule] = None,
    cooldown: Optional[int] = None,
    plot: bool = True,
    interactive_plot: bool = True,
    start=None,
    end=None,
) -> dict:
    """Fetch description, signal frame, trade log, and optional diagnostic plot."""
    lookup = {s.name: s for s in setups}
    setup = find_setup(setups, setup_name)
    components = [lookup[nm] for nm in component_names(setup) if nm in lookup and nm != setup.name]
    if trade_mode == "positions":
        log = position_trade_log(setup, y, rule=position_rule, components=components)
    elif trade_mode == "episodes":
        log = episode_trade_log(setup, y, horizon=horizon, components=components, cooldown=cooldown)
    elif trade_mode == "daily":
        log = trade_log(setup, y, horizon=horizon, components=components)
    else:
        raise ValueError("trade_mode must be 'positions', 'episodes', or 'daily'")
    report = {
        "setup": setup,
        "description": describe_setup(setup, lookup),
        "signals": setup_signal_frame(setup, y, components=components),
        "trades": log,
        "trade_mode": trade_mode,
        "summary": _trade_metrics(log["pnl_bps"]) if len(log) else None,
        "stats": performance_stats(log),
        "stats_frame": performance_stats_frame(log),
    }
    if plot:
        report["figure"] = plot_setup_diagnostics(
            setup,
            y,
            horizon=horizon,
            components=components,
            trades=log,
            start=start,
            end=end,
            interactive=interactive_plot,
        )
    return report


# ── Composition ────────────────────────────────────────────────────────

def combine_pairs(
    setups: list[Setup],
    min_triggers: int = 20,
    show_progress: Optional[bool] = None,
) -> list[Setup]:
    """Pairwise AND composition. Only same-direction pairs are combined."""
    pairs: list[Setup] = []
    total = len(setups) * (len(setups) - 1) // 2
    for a, b in _progress(
        combinations(setups, 2),
        total=total,
        desc="combine pairs",
        enabled=show_progress,
    ):
        if a.direction != b.direction:
            continue
        combined = (a.trigger.fillna(False) & b.trigger.fillna(False))
        if combined.sum() < min_triggers:
            continue
        family = a.family if a.family == b.family else f"{a.family}+{b.family}"
        pairs.append(Setup(
            name=f"{a.name} & {b.name}",
            trigger=combined,
            direction=a.direction,
            family=family,
            params={"a": a.name, "b": b.name},
        ))
    return pairs


def add_lift_vs_components(combo_df: pd.DataFrame, single_df: pd.DataFrame) -> pd.DataFrame:
    """Add `lift_vs_components` = combo Sharpe minus max(component Sharpes)."""
    out = combo_df.copy()
    keyed = single_df.set_index(["name", "horizon"])["sharpe"].to_dict()

    def lift(row):
        try:
            a, b = row["name"].split(" & ", 1)
        except ValueError:
            return np.nan
        sa = keyed.get((a, row["horizon"]), np.nan)
        sb = keyed.get((b, row["horizon"]), np.nan)
        return row["sharpe"] - max(sa, sb)

    out["lift_vs_components"] = out.apply(lift, axis=1)
    return out


def _zscore(s: pd.Series, win: int) -> pd.Series:
    ma = s.rolling(win).mean()
    sd = s.rolling(win).std()
    return (s - ma) / sd.replace(0, np.nan)


def _clean_name(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "")
        .replace("/", "")
        .replace("-", "")
        .replace("_index", "")
    )


# Family → trade-style mapping. mr = bets that overshoots will retrace.
# mom = bets that current direction continues.
_MR_FAMILIES = {"mean_reversion", "valuation"}
_MOM_FAMILIES = {"momentum", "trend", "breakout"}


def classify_type(family: str) -> str:
    """Return 'mr', 'mom', or 'mixed' for a setup family string.

    Pair families look like 'valuation+momentum'; classify as 'mixed' when
    the legs disagree on style.
    """
    if not family:
        return "mixed"
    parts = [p.strip() for p in str(family).split("+")]
    is_mr = all(p in _MR_FAMILIES for p in parts)
    is_mom = all(p in _MOM_FAMILIES for p in parts)
    if is_mr:
        return "mr"
    if is_mom:
        return "mom"
    return "mixed"


def trade_side(direction: int) -> str:
    """+1 (bet yields rise) → 'short' duration. -1 (bet yields fall) → 'long'."""
    if direction == -1:
        return "long"
    if direction == +1:
        return "short"
    return "?"


def _fmt_num(x: float) -> str:
    return f"{x:g}".replace(".", "p")


def _score_exit_hit(value: float, operator: Optional[str], level: Optional[float]) -> bool:
    if operator is None or level is None or pd.isna(value):
        return False
    if operator == ">":
        return value > level
    if operator == ">=":
        return value >= level
    if operator == "<":
        return value < level
    if operator == "<=":
        return value <= level
    if operator == "abs<=":
        return abs(value) <= abs(level)
    if operator == "abs<":
        return abs(value) < abs(level)
    raise ValueError(f"Unsupported score exit operator: {operator}")


# ── Setup builders ─────────────────────────────────────────────────────

def s_n_consecutive(y: pd.Series, n: int, direction: int) -> Setup:
    """direction=+1: n consecutive up-days, expect continuation up.
       direction=-1: n consecutive down-days, expect continuation down."""
    diffs = y.diff()
    is_dir = (diffs > 0) if direction == 1 else (diffs < 0)
    trig = is_dir.rolling(n).sum() == n
    suffix = "up" if direction == 1 else "down"
    return Setup(
        name=f"mom_{n}{suffix}",
        trigger=trig.fillna(False),
        direction=direction,
        family="momentum",
        params={"n": n},
    )


def s_at_extreme(y: pd.Series, win: int, side: str, bps_tol: float = 5.0) -> Setup:
    """side='high': within bps_tol of N-day high → bet reversion (direction=-1).
       side='low':  within bps_tol of N-day low  → bet reversion (direction=+1)."""
    if side == "high":
        ref = y.shift(1).rolling(win).max()
        trig = (ref - y).abs() <= bps_tol
        direction = -1
    else:
        ref = y.shift(1).rolling(win).min()
        trig = (y - ref).abs() <= bps_tol
        direction = +1
    return Setup(
        name=f"mr_at_{side}_{win}d",
        trigger=trig.fillna(False),
        direction=direction,
        family="mean_reversion",
        params={"win": win, "side": side, "bps_tol": bps_tol},
    )


def s_bb_extreme(y: pd.Series, win: int, side: str, z_thresh: float = 2.0) -> Setup:
    """Bollinger z-score extreme. Continuous score available."""
    thresh = abs(z_thresh)
    bb_z = (y - y.rolling(win).mean()) / y.rolling(win).std()
    if side == "high":
        trig = bb_z > thresh
        direction = -1
    else:
        trig = bb_z < -thresh
        direction = +1
    return Setup(
        name=f"mr_bb_{side}_{win}d",
        trigger=trig.fillna(False),
        direction=direction,
        family="mean_reversion",
        score=bb_z,
        params={"win": win, "side": side, "z_thresh": thresh},
    )


def s_resid_extreme(
    y: pd.Series,
    x: pd.Series,
    lookback: int,
    side: str,
    z_thresh: float = 2.0,
    name_prefix: str = "val",
) -> Setup:
    """Regress y on x in changes; OU-z the cumulative residual.
       side='high': z > thresh (y rich vs x), bet y falls. direction=-1.
       side='low':  z < -thresh (y cheap vs x), bet y rises. direction=+1."""
    from stats import roll_lr_diff, ou_zscore
    thresh = abs(z_thresh)

    pair = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    if len(pair) < lookback + 10:
        # Return a setup with empty trigger; evaluate() will skip it.
        empty = pd.Series(False, index=y.index)
        return Setup(
            name=f"{name_prefix}_{side}_lb{lookback}_INSUFFICIENT_DATA",
            trigger=empty, direction=-1 if side == "high" else +1,
            family="valuation", params={"lookback": lookback, "side": side},
        )

    reg = roll_lr_diff(pair["x"], pair["y"], lookback=lookback).to_pandas()
    reg.index = pair.index[1:]  # first row dropped to take diffs
    resid_cum = reg["resid_cum"].reindex(y.index)
    z = ou_zscore(reg["resid_cum"], lookback=lookback).to_pandas()
    z.index = reg.index
    z = z.reindex(y.index)

    if side == "high":
        trig = z > thresh
        direction = -1
    else:
        trig = z < -thresh
        direction = +1

    return Setup(
        name=f"{name_prefix}_{side}_lb{lookback}",
        trigger=trig.fillna(False),
        direction=direction,
        family="valuation",
        score=z,
        residual=resid_cum,
        params={"lookback": lookback, "side": side, "z_thresh": thresh},
    )


def s_large_move(
    y: pd.Series,
    lookback: int,
    direction: int,
    min_move_bps: float,
    mode: str = "momentum",
) -> Setup:
    """Large yield move trigger.

    mode='momentum' expects continuation in the same direction.
    mode='reversal' expects mean reversion after the move.
    """
    move = y.diff(lookback)
    if direction == 1:
        trig = move >= min_move_bps
        suffix = "up"
    else:
        trig = move <= -min_move_bps
        suffix = "down"

    out_dir = direction if mode == "momentum" else -direction
    family = "momentum" if mode == "momentum" else "mean_reversion"
    return Setup(
        name=f"{mode}_move_{lookback}d_{suffix}_{min_move_bps:g}bp",
        trigger=trig.fillna(False),
        direction=out_dir,
        family=family,
        score=move,
        params={
            "lookback": lookback,
            "direction": direction,
            "min_move_bps": min_move_bps,
            "mode": mode,
        },
    )


def s_ma_cross(y: pd.Series, fast: int, slow: int, direction: int) -> Setup:
    """Moving-average trend state: fast MA above/below slow MA."""
    fast_ma = y.rolling(fast).mean()
    slow_ma = y.rolling(slow).mean()
    spread = fast_ma - slow_ma
    if direction == 1:
        trig = spread > 0
        suffix = "bull"
    else:
        trig = spread < 0
        suffix = "bear"
    return Setup(
        name=f"trend_ma_{fast}_{slow}_{suffix}",
        trigger=trig.fillna(False),
        direction=direction,
        family="trend",
        score=spread,
        params={"fast": fast, "slow": slow},
    )


def s_ma_distance_extreme(
    y: pd.Series,
    win: int,
    side: str,
    z_thresh: float = 2.0,
) -> Setup:
    """Yield far from its moving average; bet mean reversion."""
    thresh = abs(z_thresh)
    z = _zscore(y, win)
    if side == "high":
        trig = z > thresh
        direction = -1
    else:
        trig = z < -thresh
        direction = +1
    return Setup(
        name=f"mr_ma_dist_{side}_{win}d_z{thresh:g}",
        trigger=trig.fillna(False),
        direction=direction,
        family="mean_reversion",
        score=z,
        params={"win": win, "side": side, "z_thresh": thresh},
    )


def s_rsi_extreme(
    y: pd.Series,
    win: int,
    side: str,
    low: float = 30.0,
    high: float = 70.0,
) -> Setup:
    """RSI-style extreme on yield changes; bet reversion."""
    chg = y.diff()
    gain = chg.clip(lower=0).rolling(win).mean()
    loss = (-chg.clip(upper=0)).rolling(win).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    if side == "high":
        trig = rsi >= high
        direction = -1
        level = high
    else:
        trig = rsi <= low
        direction = +1
        level = low
    return Setup(
        name=f"mr_rsi_{side}_{win}d_{level:g}",
        trigger=trig.fillna(False),
        direction=direction,
        family="mean_reversion",
        score=rsi,
        params={"win": win, "side": side, "low": low, "high": high},
    )


def s_range_breakout(
    y: pd.Series,
    win: int,
    side: str,
    bps_buffer: float = 0.0,
) -> Setup:
    """Break above/below a trailing range; bet continuation."""
    if side == "high":
        ref = y.shift(1).rolling(win).max()
        trig = y > ref + bps_buffer
        direction = +1
    else:
        ref = y.shift(1).rolling(win).min()
        trig = y < ref - bps_buffer
        direction = -1
    return Setup(
        name=f"breakout_{side}_{win}d_buf{bps_buffer:g}",
        trigger=trig.fillna(False),
        direction=direction,
        family="breakout",
        params={"win": win, "side": side, "bps_buffer": bps_buffer},
    )


def s_spread_z_extreme(
    spread: pd.Series,
    win: int,
    side: str,
    z_thresh: float = 2.0,
    target_direction: Optional[int] = None,
    name_prefix: Optional[str] = None,
) -> Setup:
    """Spread z-score valuation signal mapped to a 10y direction.

    If target_direction is omitted, high spread bets 10y yield rises and low
    spread bets 10y yield falls. Pass target_direction to override this mapping.
    """
    thresh = abs(z_thresh)
    z = _zscore(spread, win)
    label = name_prefix or f"spread_{_clean_name(spread.name or 'x')}"
    if side == "high":
        trig = z > thresh
        direction = +1 if target_direction is None else target_direction
    else:
        trig = z < -thresh
        direction = -1 if target_direction is None else target_direction
    return Setup(
        name=f"{label}_{side}_{win}d_z{thresh:g}",
        trigger=trig.fillna(False),
        direction=direction,
        family="valuation",
        score=z,
        params={"win": win, "side": side, "z_thresh": thresh},
    )


def build_duration_setup_catalog(
    curve: pd.DataFrame,
    y_col: str = "10y",
    profile: str = "wide",
    real_rate_columns: Optional[dict[str, str]] = None,
) -> list[Setup]:
    """Build a broad starter catalog of duration technical and valuation setups.

    The catalog only uses columns present in `curve`. profile='wide' is meant
    for research sweeps, not a final production signal set.
    """
    curve = with_synthetic_real_rates(curve, columns=real_rate_columns)
    y = curve[y_col]
    setups: list[Setup] = []

    for n in [2, 3, 4, 5]:
        setups.extend([s_n_consecutive(y, n=n, direction=+1), s_n_consecutive(y, n=n, direction=-1)])

    for lookback in [3, 5, 10, 20]:
        for move in [10.0, 20.0, 35.0]:
            setups.extend([
                s_large_move(y, lookback, +1, move, mode="momentum"),
                s_large_move(y, lookback, -1, move, mode="momentum"),
                s_large_move(y, lookback, +1, move, mode="reversal"),
                s_large_move(y, lookback, -1, move, mode="reversal"),
            ])

    for win in [20, 60, 120, 252]:
        for side in ["high", "low"]:
            setups.append(s_at_extreme(y, win=win, side=side, bps_tol=5.0))
            setups.append(s_ma_distance_extreme(y, win=win, side=side, z_thresh=1.5))
            setups.append(s_ma_distance_extreme(y, win=win, side=side, z_thresh=2.0))
            setups.append(s_range_breakout(y, win=win, side=side, bps_buffer=0.0))

    for win in [20, 60, 120]:
        for zt in [1.5, 2.0, 2.5]:
            setups.extend([
                s_bb_extreme(y, win=win, side="high", z_thresh=zt),
                s_bb_extreme(y, win=win, side="low", z_thresh=zt),
            ])

    for fast, slow in [(5, 20), (10, 60), (20, 120), (60, 252)]:
        setups.extend([s_ma_cross(y, fast, slow, +1), s_ma_cross(y, fast, slow, -1)])

    for win in [7, 14, 28]:
        setups.extend([
            s_rsi_extreme(y, win=win, side="high", high=70.0),
            s_rsi_extreme(y, win=win, side="low", low=30.0),
            s_rsi_extreme(y, win=win, side="high", high=80.0),
            s_rsi_extreme(y, win=win, side="low", low=20.0),
        ])

    valuation_targets = {
        "2y": curve["2y"] if "2y" in curve else None,
        "30y": curve["30y"] if "30y" in curve else None,
        "2s10s": curve["2s10s"] if "2s10s" in curve else None,
        "10s30s": curve["10s30s"] if "10s30s" in curve else None,
        "real_5y": curve["real_5y"] if "real_5y" in curve else None,
        "real_10y": curve["real_10y"] if "real_10y" in curve else None,
        "real_5y5y": curve["real_5y5y"] if "real_5y5y" in curve else None,
        "5y5y_sfr": curve["5y5y_sfr"] if "5y5y_sfr" in curve else None,
        "5y5y_zc": curve["5y5y_zc"] if "5y5y_zc" in curve else None,
    }
    for name, x in valuation_targets.items():
        if x is None:
            continue
        for lookback in [60, 120, 252]:
            for zt in [2.0, 1.5]:
                base_prefix = f"val_{_clean_name(name)}"
                prefix = base_prefix if zt == 2.0 else f"{base_prefix}_z{_fmt_num(zt)}"
                setups.extend([
                    s_resid_extreme(y, x, lookback=lookback, side="high", z_thresh=zt, name_prefix=prefix),
                    s_resid_extreme(y, x, lookback=lookback, side="low", z_thresh=zt, name_prefix=prefix),
                ])

    for spread_name in ["2s10s", "10s30s"]:
        if spread_name not in curve:
            continue
        spread = curve[spread_name].rename(spread_name)
        for win in [60, 120, 252]:
            for zt in [2.0, 1.5]:
                base_prefix = f"spread_{_clean_name(spread_name)}"
                prefix = base_prefix if zt == 2.0 else f"{base_prefix}_z{_fmt_num(zt)}"
                setups.extend([
                    s_spread_z_extreme(spread, win, side="high", z_thresh=zt, name_prefix=prefix),
                    s_spread_z_extreme(spread, win, side="low", z_thresh=zt, name_prefix=prefix),
                ])

    deduped: list[Setup] = []
    seen: set[str] = set()
    for s in setups:
        if s.name in seen:
            continue
        seen.add(s.name)
        deduped.append(s)

    if profile == "compact":
        return [
            s for s in deduped
            if s.family in {"trend", "valuation"} or s.params.get("lookback") in {20, 60, 120}
        ]
    return deduped
