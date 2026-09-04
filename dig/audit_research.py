"""Audit the three research pillars against what their docstrings claim.

The studies in research/ were not written by hand and have not all been run.
Before building an exploration app on top of them we check the claims a
research result silently depends on. Reading the code is not enough; every
test below is executed against the live 10s20s30s panel.

  1. lookahead   -- perturb one future bar. Nothing before it may move. This
     is the only test that actually proves "no lookahead bias".
  2. alignment   -- every study pads a rolling regression back to panel
     length. Off by one row and the signal silently reads tomorrow.
  3. interior nulls -- roll_lr_diff drops nulls JOINTLY and then diffs, so a
     single-series holiday makes a 2-day move look like a 1-day move. Do the
     studies protect against that, or pass it through?
  4. tradability -- fade_scorecard measures the forward change in whatever
     series it is handed. If that series is re-marked by a drifting hedge
     ratio, its forward change is not the P&L of any position you can hold.
  5. r2 semantics -- roll_lr accumulates each bar's own residual rather than
     refitting the window. Report how far that is from in-window R2.

    python -m dig.audit_research
"""

from __future__ import annotations

import numpy as np
import polars as pl

import utils
from research import DislocationStudy, FairValueStudy, PairRVStudy
from stats import roll_lr

from .fly_vs_vol import load_data

TARGET = "fly"
INPUTS = (TARGET, "10y", "vol")
PERTURB_BPS = 100.0  # large enough that any leak is unmistakable
PROBE_FRAC = 0.7
LOOKBACK = 126

# ---- the panel and the three studies under audit ----------------------------


def studies() -> dict:
    """One instance of each pillar, all pointed at the 10s20s30s fly."""
    return {
        "dislocation": DislocationStudy(target=TARGET, features=("10y", "vol")),
        "relative_value": PairRVStudy(left=TARGET, right="10y"),
        "fair_value": FairValueStudy(target=TARGET, factors=("10y", "vol")),
    }


def audit_panel(data: pl.DataFrame) -> pl.DataFrame:
    """The common non-null sample every probe runs on."""
    return data.drop_nulls(subset=list(INPUTS)).sort("ts")


# ---- 1. lookahead -----------------------------------------------------------


def lookahead_probe(study, data: pl.DataFrame, frac: float = PROBE_FRAC) -> pl.DataFrame:
    """Bump one bar of every input; report any output that changed BEFORE it."""
    panel = audit_panel(data)
    cut = panel["ts"][int(len(panel) * frac)]

    base = study.compute(panel)
    bumped = panel.with_columns([
        pl.when(pl.col("ts") == cut)
        .then(pl.col(c) + PERTURB_BPS)
        .otherwise(pl.col(c))
        .alias(c)
        for c in INPUTS
    ])
    after = study.compute(bumped)

    numeric = [
        c for c, dt in zip(base.columns, base.dtypes)
        if dt.is_numeric() and c in after.columns
    ]
    joined = base.filter(pl.col("ts") < cut).select("ts", *numeric).join(
        after.filter(pl.col("ts") < cut).select("ts", *numeric),
        on="ts", suffix="__after",
    )
    rows = []
    for col in numeric:
        a, b = joined[col].to_numpy(), joined[f"{col}__after"].to_numpy()
        both = np.isfinite(a) & np.isfinite(b)
        leak = float(np.max(np.abs(a[both] - b[both]))) if both.any() else 0.0
        rows.append({
            "column": col,
            "n_compared": int(both.sum()),
            "max_abs_change_before_bump": leak,
            "leaks": bool(leak > 1e-9),
        })
    return pl.DataFrame(rows).sort("max_abs_change_before_bump", descending=True)


# ---- 2. alignment -----------------------------------------------------------


def alignment_probe(study, data: pl.DataFrame) -> dict:
    """Does compute() return one row per input row, on the same dates?"""
    panel = audit_panel(data)
    out = study.compute(panel)
    matched = len(out) == len(panel) and bool((out["ts"] == panel["ts"]).all())
    return {"in_rows": len(panel), "out_rows": len(out), "dates_match": matched}


def manual_residual_check(data: pl.DataFrame, lookback: int = LOOKBACK) -> dict:
    """Rebuild the live bar's dislocation by hand from a trailing window only."""
    panel = data.drop_nulls(subset=[TARGET, "10y"]).sort("ts")
    study = DislocationStudy(target=TARGET, features=("10y",), beta_lookback=lookback)
    out = study.compute(panel)

    i = len(panel) - 1  # the live bar, the one a signal would actually trade
    dy = panel[TARGET].diff().to_numpy()
    dx = panel["10y"].diff().to_numpy()
    win_y, win_x = dy[i - lookback + 1: i + 1], dx[i - lookback + 1: i + 1]
    beta, alpha = np.polyfit(win_x, win_y, 1)
    manual = float(win_y[-1] - (alpha + beta * win_x[-1]))
    return {
        "framework_dislocation": float(out["dislocation"][i]),
        "manual_trailing_only": manual,
        "abs_diff": abs(float(out["dislocation"][i]) - manual),
        "framework_beta": float(out["beta_10y"][i]),
        "manual_beta": float(beta),
    }


# ---- 3. interior nulls ------------------------------------------------------


def interior_null_probe(data: pl.DataFrame) -> pl.DataFrame:
    """Blank one feature bar; see whether a study drops it or diffs across it."""
    panel = audit_panel(data)
    k = int(len(panel) * 0.5)
    holed = panel.with_columns(
        pl.when(pl.arange(0, pl.len()) == k)
        .then(None)
        .otherwise(pl.col("10y"))
        .alias("10y")
    )
    rows = []
    for name, study in studies().items():
        clean_n = len(study.compute(panel))
        holed_n = len(study.compute(holed))
        rows.append({
            "study": name,
            "rows_clean": clean_n,
            "rows_with_1_null": holed_n,
            "rows_lost": clean_n - holed_n,
            "diffs_across_the_gap": clean_n - holed_n == 1,
        })
    return pl.DataFrame(rows)


# ---- 4. tradability of what fade_scorecard scores ---------------------------


def rv_tradability(data: pl.DataFrame, horizon: int = 20) -> pl.DataFrame:
    """Split the forward change in rv_value into held P&L and hedge re-marking.

    A position entered at t holds beta_t. rv_value at t+h is marked with
    beta_{t+h}. That difference is not P&L -- it is the hedge being rewritten
    underneath the trade, and fade_scorecard counts all of it as return.
    """
    panel = data.drop_nulls(subset=[TARGET, "10y"]).sort("ts")
    sig = PairRVStudy(left=TARGET, right="10y").compute(panel).drop_nulls(
        subset=["rv_value", "hedge_weight"]
    )
    left, right = sig[TARGET].to_numpy(), sig["10y"].to_numpy()
    beta, rv = sig["hedge_weight"].to_numpy(), sig["rv_value"].to_numpy()

    h = horizon
    scored = rv[h:] - rv[:-h]                                             # scored
    held = (left[h:] - left[:-h]) - beta[:-h] * (right[h:] - right[:-h])  # real P&L
    remark = -(beta[h:] - beta[:-h]) * right[h:]                          # rewritten
    return pl.DataFrame({
        "horizon": [h],
        "n_obs": [len(scored)],
        "std_scored_change": [float(np.std(scored))],
        "std_true_held_pnl": [float(np.std(held))],
        "std_beta_remark": [float(np.std(remark))],
        "remark_share_of_var": [float(np.var(remark) / np.var(scored))],
        "corr_scored_vs_held": [float(np.corrcoef(scored, held)[0, 1])],
        "identity_residual": [float(np.max(np.abs(scored - held - remark)))],
    })


# ---- 5. r2 semantics --------------------------------------------------------


def r2_semantics(data: pl.DataFrame, lookback: int = LOOKBACK) -> pl.DataFrame:
    """Compare roll_lr's r2 to a true refit-the-window in-sample R2."""
    panel = data.drop_nulls(subset=[TARGET, "10y"]).sort("ts")
    dy = panel[TARGET].diff().drop_nulls()
    dx = panel["10y"].diff().drop_nulls()
    reported = roll_lr(dx, dy, lookback=lookback)["r2"].to_numpy()

    y, x = dy.to_numpy(), dx.to_numpy()
    true = np.full(len(y), np.nan)
    for i in range(lookback - 1, len(y)):
        wy, wx = y[i - lookback + 1: i + 1], x[i - lookback + 1: i + 1]
        b, a = np.polyfit(wx, wy, 1)
        resid = wy - (a + b * wx)
        true[i] = 1.0 - resid.var() / wy.var()

    both = np.isfinite(reported) & np.isfinite(true)
    return pl.DataFrame({
        "n_windows": [int(both.sum())],
        "reported_r2_median": [float(np.median(reported[both]))],
        "true_inwindow_r2_median": [float(np.median(true[both]))],
        "median_abs_gap": [float(np.median(np.abs(reported[both] - true[both])))],
        "max_abs_gap": [float(np.max(np.abs(reported[both] - true[both])))],
    })


def main() -> dict:
    data = load_data()
    panel = audit_panel(data)
    print(f"audit panel {panel['ts'][0]} -> {panel['ts'][-1]}  rows={len(panel)}\n")

    # 1: the lookahead probe, per pillar
    leaks = {}
    for name, study in studies().items():
        probe = lookahead_probe(study, data)
        leaks[name] = probe
        n_leaking = int(probe["leaks"].sum())
        print(f"[1] lookahead / {name}: {n_leaking} of {len(probe)} columns "
              f"changed before the bumped bar")
        if n_leaking:
            utils.pdf(probe.filter(pl.col("leaks")))

    # 2: alignment, then a by-hand reconstruction of the live bar
    align = pl.DataFrame([
        {"study": name, **alignment_probe(study, data)}
        for name, study in studies().items()
    ])
    manual = manual_residual_check(data)

    # 3: what one missing feature bar does
    nulls = interior_null_probe(data)

    # 4: is the thing rel-val scores actually holdable
    trade = rv_tradability(data)

    # 5: what roll_lr's r2 actually measures
    r2 = r2_semantics(data)

    print("\n[2] alignment -- one output row per input row?")
    utils.pdf(align)
    print("\n[2] live-bar dislocation rebuilt by hand from the trailing window only:")
    utils.pdf(pl.DataFrame([manual]))
    print("\n[3] effect of a single interior null in one feature:")
    utils.pdf(nulls)
    print("\n[4] what fade_scorecard actually scores for PairRVStudy (20d):")
    utils.pdf(trade)
    print("\n[5] roll_lr r2 versus a true refit-the-window R2:")
    utils.pdf(r2)
    return {"leaks": leaks, "alignment": align, "manual": manual,
            "nulls": nulls, "tradability": trade, "r2": r2}


if __name__ == "__main__":
    state = main()
