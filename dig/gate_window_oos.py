"""Does a rolling gate percentile generalize better than an expanding one?

The gate percentile decides what "the 25th percentile of half-life" is measured
against. Today that is all history since 2011 (expanding); the alternative is a
trailing window that re-centres on the current regime.

Expanding never needs a lookback chosen, but its denominator grows without
bound, so a persistent regime shift is absorbed rather than re-based and a gate
can stay shut for years. Rolling re-centres, but the lookback is itself a fitted
choice. Which wins is an empirical question, and the only honest way to ask it
is out of sample.

Method — walk-forward gate selection, run identically per window type:

    for each fold (train through year T, test the following two years):
        rank all 132 gate variants by TRAIN gross Sharpe
        take the best one and record what it earned in TEST

A window type is better if the gates it selects hold up in the test years. The
ungated baseline runs through the same folds as the hurdle: a gate is only worth
having if it beats not gating at all.

PnL is gross daily re-marking, position[t-1] x d(level) -- the same convention
backtest.strategy._robustness and book/curve/app.py use for comparative work.
Entry/exit params stay frozen at each signal's promoted values so this measures
the gate, not a re-search of the whole space.

    mamba run -n 2s10s python -m dig.gate_window_oos
"""

from __future__ import annotations

import numpy as np
import polars as pl

from backtest.lab import REGIME_GATE_BUCKETS, gate_allow_mask
from dashboard import runner
from dashboard.registry import LiveRegistry

# gate percentile lookbacks to compare, in bars (None = expanding, today's rule)
WINDOWS = [None, 504, 1260]
WINDOW_LABELS = {None: "expanding", 504: "roll 2y", 1260: "roll 5y"}

# every condition Strategy._gate_condition can build
CONDITIONS = [
    "r2", "beta_cv", "beta", "beta_vol20", "beta_mom10", "r2_vol20",
    "r2_mom10", "resid_phi", "resid_half_life", "resid_vol20", "resid_mom10",
]
BUCKETS = [spec[0] for spec in REGIME_GATE_BUCKETS]

# fold f trains on everything through TRAIN_END[f] and tests the next two years
TRAIN_ENDS = [2017, 2019, 2021, 2023]
TEST_SPAN = 2


def load_signals() -> list[dict]:
    """Every promoted signal, with its frozen params and cached data."""
    out = []
    for row in LiveRegistry().list().to_dicts():
        signal_id = row["signal_id"]
        state = runner.compute_signal(signal_id)
        out.append({
            "signal_id": signal_id,
            "strategy": state["strategy"],
            "data": state["data"],
            "params": state["params"],
            "frame": state["signal_frame"],
        })
    return out


def positions(frame: pl.DataFrame, params: dict) -> np.ndarray:
    """Daily +1/-1/0 from the promoted entry threshold and a band exit.

    A vectorized stand-in for the exact engine: enter when |signal| crosses the
    threshold, hold until it reverts inside half of it. Identical across every
    gate and window, so differences are attributable to the gate alone.
    """
    signal = frame["signal"].to_numpy().astype(float)
    entry = float(params["entry_threshold"])
    exit_band = entry * 0.5
    pos = np.zeros(len(signal))
    state = 0.0
    for i, s in enumerate(signal):
        if not np.isfinite(s):
            state = 0.0
        elif state == 0.0 and abs(s) >= entry:
            state = -np.sign(s)  # positive signal = target rich = short it
        elif state != 0.0 and abs(s) <= exit_band:
            state = 0.0
        pos[i] = state
    return pos


def gated_positions(base: np.ndarray, allow: np.ndarray) -> np.ndarray:
    """Base positions with entries blocked while the gate is shut.

    A trade already open is allowed to run to its own exit; the gate filters
    entries, exactly as entry_filter_fn does in the exact engine.
    """
    out = np.zeros(len(base))
    state = 0.0
    for i in range(len(base)):
        want = base[i]
        if want == 0.0:
            state = 0.0
        elif state == 0.0:
            state = want if allow[i] else 0.0
        elif want != state:  # base flipped side: treat as a fresh entry
            state = want if allow[i] else 0.0
        out[i] = state
    return out


def sharpe(pnl: np.ndarray) -> float:
    sd = float(pnl.std())
    return float(pnl.mean()) / sd * np.sqrt(252.0) if sd > 0 else 0.0


def condition_series(strategy, frame: pl.DataFrame, params: dict, name: str):
    """One gate condition column, via the strategy's own builder."""
    return strategy._gate_condition(frame, {**params, "gate": (name, "below_50")})


def evaluate(signal: dict) -> pl.DataFrame:
    """One row per (window, condition, bucket, fold) with train/test sharpe."""
    strategy, frame, params = signal["strategy"], signal["frame"], signal["params"]
    years = np.array([t.year for t in signal["data"]["ts"].to_list()])
    level = signal["data"][strategy.target].to_numpy().astype(float)
    dlevel = np.concatenate([[0.0], np.diff(level)])

    base = positions(frame, params)
    conditions = {name: condition_series(strategy, frame, params, name)
                  for name in CONDITIONS}

    def pnl_of(pos: np.ndarray) -> np.ndarray:
        return np.concatenate([[0.0], pos[:-1]]) * dlevel

    rows = []
    for window in WINDOWS:
        masks = {}
        for name, cond in conditions.items():
            for bucket in BUCKETS:
                masks[(name, bucket)] = gate_allow_mask(
                    cond, (name, bucket),
                    min_history=strategy.gate_min_history, window=window,
                )
        for train_end in TRAIN_ENDS:
            train = years <= train_end
            test = (years > train_end) & (years <= train_end + TEST_SPAN)
            if test.sum() < 100:
                continue
            for (name, bucket), allow in masks.items():
                pnl = pnl_of(gated_positions(base, allow))
                rows.append({
                    "signal_id": signal["signal_id"],
                    "window": WINDOW_LABELS[window],
                    "condition": name,
                    "bucket": bucket,
                    "train_end": train_end,
                    "train_sharpe": sharpe(pnl[train]),
                    "test_sharpe": sharpe(pnl[test]),
                    "test_open_share": float(allow[test].mean()),
                })
            ungated = pnl_of(base)
            rows.append({
                "signal_id": signal["signal_id"],
                "window": WINDOW_LABELS[window],
                "condition": "(ungated)",
                "bucket": "(none)",
                "train_end": train_end,
                "train_sharpe": sharpe(ungated[train]),
                "test_sharpe": sharpe(ungated[test]),
                "test_open_share": 1.0,
            })
    return pl.DataFrame(rows)


def selection_table(results: pl.DataFrame) -> pl.DataFrame:
    """Per (window, signal, fold): pick the train winner, keep its test score."""
    gated = results.filter(pl.col("condition") != "(ungated)")
    picked = (
        gated.sort("train_sharpe", descending=True)
        .group_by("window", "signal_id", "train_end", maintain_order=True)
        .first()
    )
    baseline = results.filter(pl.col("condition") == "(ungated)").select(
        "window", "signal_id", "train_end",
        pl.col("test_sharpe").alias("ungated_test_sharpe"),
    )
    return picked.join(baseline, on=["window", "signal_id", "train_end"])


def summarize(picked: pl.DataFrame) -> pl.DataFrame:
    """Does the gate selected in training pay in testing, per window type?"""
    return (
        picked.group_by("window")
        .agg(
            pl.col("train_sharpe").mean().round(2).alias("mean_train_sr"),
            pl.col("test_sharpe").mean().round(2).alias("mean_test_sr"),
            (pl.col("train_sharpe") - pl.col("test_sharpe")).mean().round(2)
            .alias("mean_degradation"),
            pl.col("ungated_test_sharpe").mean().round(2).alias("ungated_test_sr"),
            (pl.col("test_sharpe") > pl.col("ungated_test_sharpe")).mean().round(2)
            .alias("beat_ungated"),
            (pl.col("test_sharpe") > 0).mean().round(2).alias("test_positive"),
            pl.col("test_open_share").mean().round(2).alias("mean_open_share"),
            pl.len().alias("n"),
        )
        .sort("mean_test_sr", descending=True)
    )


def stability(results: pl.DataFrame) -> pl.DataFrame:
    """How often does the train-winning gate stay the winner next fold?"""
    gated = results.filter(pl.col("condition") != "(ungated)")
    winners = (
        gated.sort("train_sharpe", descending=True)
        .group_by("window", "signal_id", "train_end", maintain_order=True)
        .first()
        .with_columns((pl.col("condition") + "/" + pl.col("bucket")).alias("pick"))
        .sort("window", "signal_id", "train_end")
    )
    return (
        winners.with_columns(
            (pl.col("pick") == pl.col("pick").shift(1))
            .over("window", "signal_id")
            .alias("same_as_prior")
        )
        .drop_nulls("same_as_prior")
        .group_by("window")
        .agg(pl.col("same_as_prior").mean().round(2).alias("pick_persistence"),
             pl.len().alias("n"))
        .sort("window")
    )


def main() -> dict:
    signals = load_signals()
    print(f"signals: {len(signals)}  gate variants: "
          f"{len(CONDITIONS) * len(BUCKETS)}  windows: {len(WINDOWS)}  "
          f"folds: {len(TRAIN_ENDS)}")

    results = pl.concat([evaluate(s) for s in signals], how="diagonal_relaxed")
    picked = selection_table(results)
    summary = summarize(picked)
    persistence = stability(results)

    print("\nwalk-forward gate selection -- train winner, measured out of sample")
    print(summary)
    print("\nhow often the winning gate survives into the next fold")
    print(persistence)
    print("\nselected gate per fold (expanding vs rolling)")
    print(
        picked.select("window", "signal_id", "train_end", "condition", "bucket",
                      "train_sharpe", "test_sharpe", "ungated_test_sharpe")
        .with_columns(pl.col("^.*sharpe$").round(2))
        .sort("signal_id", "train_end", "window")
    )
    return {"results": results, "picked": picked, "summary": summary,
            "persistence": persistence}


if __name__ == "__main__":
    state = main()
