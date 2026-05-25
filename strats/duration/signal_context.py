"""Signal context features and conditional edge analysis for OU z-score signals.

The core idea: at each signal bar the OU z-score is the entry trigger, but the
*context* around that bar (beta stability, residual momentum, vol regime, model
fit quality, range position) may predict whether the signal will work or fail.

Three building blocks
---------------------
1. build_signal_features(model)
       Per-bar context DataFrame from a single-anchor model.

2. conditional_ic_table(features, y, ...)
       Spearman IC of ou_zscore vs forward yield change, broken out by
       quantile bins of each context feature.  Positive IC in a bin means
       the signal has edge when the feature is in that range.

3. oos_edge_test(features, y, filter_col, ...)
       Walk-forward: for each OOS window, identify which feature bins had
       positive in-sample IC, then apply that filter OOS.  Compares filtered
       vs unfiltered signal performance so you can see whether the context
       actually adds out-of-sample value.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm


# ─── feature engineering ──────────────────────────────────────────────────────

def build_signal_features(
    model: pd.DataFrame,
    *,
    hi_lo_win: int = 30,
    beta_trend_win: int = 20,
    r2_trend_win: int = 20,
    vol_fast: int = 20,
    vol_slow: int = 60,
    zscore_mom_wins: tuple[int, ...] = (5, 10),
) -> pd.DataFrame:
    """Compute context features at each bar for a single-anchor OU model.

    `model` is the per-anchor DataFrame from single_factor_10y_table:
    columns x, y, alpha, beta, yhat, resid, r2, ou_zscore.

    Feature groups
    --------------
    Model quality   r2, r2_trend
    Beta dynamics   beta, beta_trend, beta_vol (rolling std of beta)
    Range position  resid_pct_hi_Nd — where is the residual in its Nd range?
                    1.0 = at the trailing high, 0.0 = at the trailing low
    Signal momentum zscore_mom_Nd — is the z-score building (+) or fading (-)?
    Vol regime      resid_vol_fast, resid_vol_slow, vol_ratio (>1 = vol expanding)
    Signal duration zscore_streak — bars |z| has been above 1
    """
    m = model.copy()
    resid = m["resid"]
    z     = m["ou_zscore"]
    beta  = m["beta"]
    r2    = m["r2"]

    f = pd.DataFrame(index=m.index)
    f["ou_zscore"] = z

    # --- model quality ---
    f["r2"]       = r2
    f["r2_trend"] = r2 - r2.shift(r2_trend_win)

    # --- beta dynamics ---
    f["beta"]      = beta
    f["beta_trend"] = beta - beta.shift(beta_trend_win)
    f["beta_vol"]   = beta.rolling(beta_trend_win).std()

    # --- residual range position ---
    # Use shift(1) to avoid lookahead: prior-day high/low
    hi  = resid.shift(1).rolling(hi_lo_win).max()
    lo  = resid.shift(1).rolling(hi_lo_win).min()
    rng = (hi - lo).replace(0, np.nan)
    f[f"resid_pct_hi_{hi_lo_win}d"] = (resid - lo) / rng  # 1=at hi, 0=at lo
    f[f"resid_dist_hi_{hi_lo_win}d"] = resid - hi           # ≤0 = below trailing high
    f[f"resid_dist_lo_{hi_lo_win}d"] = resid - lo           # ≥0 = above trailing low

    # --- signal momentum ---
    for w in zscore_mom_wins:
        f[f"zscore_mom_{w}d"] = z - z.shift(w)

    # --- volatility regime ---
    f[f"resid_vol_{vol_fast}d"] = resid.rolling(vol_fast).std()
    f[f"resid_vol_{vol_slow}d"] = resid.rolling(vol_slow).std()
    slow_vol = f[f"resid_vol_{vol_slow}d"].replace(0, np.nan)
    f["vol_ratio"] = f[f"resid_vol_{vol_fast}d"] / slow_vol

    # --- signal duration (how long has |z| been elevated) ---
    above = (z.abs() > 1).astype(int)
    # streak = consecutive bars above threshold ending at each bar
    groups = (above == 0).cumsum()
    f["zscore_streak"] = above.groupby(groups).cumcount()

    return f.dropna(how="all")


# ─── conditional IC analysis ─────────────────────────────────────────────────

def conditional_ic_table(
    features: pd.DataFrame,
    y: pd.Series,
    *,
    horizons: tuple[int, ...] = (5, 20, 60),
    n_bins: int = 5,
    feature_cols: Optional[list[str]] = None,
    min_obs_per_bin: int = 20,
) -> pd.DataFrame:
    """Spearman IC of ou_zscore vs -fwd_yield_change, per feature quintile.

    For each feature column and each horizon H, this bins the feature into
    `n_bins` quantile groups and reports the IC within each group.

    Interpretation
    --------------
    Positive IC in a bin → the signal has directional edge when the feature
    is in that range.  Compare bins within a feature to see which context
    regime is most predictive.

    Returns a DataFrame with columns:
        feature, bin (1=lowest … n_bins=highest), n, ic_5d, ic_20d, ic_60d
    sorted by feature then bin.
    """
    if feature_cols is None:
        feature_cols = [c for c in features.columns if c != "ou_zscore"]

    rows = []
    for feat_col in feature_cols:
        fwd_cols: dict[int, pd.Series] = {
            h: y.diff(h).shift(-h) for h in horizons
        }
        base = pd.concat(
            [features["ou_zscore"].rename("z"),
             features[feat_col].rename("feat"),
             *[v.rename(f"fwd_{h}") for h, v in fwd_cols.items()]],
            axis=1,
        ).dropna()
        base = base[base["z"] != 0]
        if len(base) < n_bins * min_obs_per_bin:
            continue

        try:
            base = base.copy()
            base["bin"] = pd.qcut(base["feat"], q=n_bins, labels=False, duplicates="drop")
        except Exception:
            continue

        for b, grp in base.groupby("bin"):
            if len(grp) < min_obs_per_bin:
                continue
            row: dict = {
                "feature": feat_col,
                "bin": int(b) + 1,
                "n": len(grp),
            }
            for h in horizons:
                row[f"ic_{h}d"] = float(grp["z"].corr(-grp[f"fwd_{h}"], method="spearman"))
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    return (pd.DataFrame(rows)
              .sort_values(["feature", "bin"])
              .reset_index(drop=True))


# ─── vectorized OOS edge test ────────────────────────────────────────────────

def _simple_stats(pnl: pd.Series, horizon: int) -> dict:
    pnl = pnl.dropna()
    if len(pnl) < 5 or pnl.std() == 0:
        return {"n": len(pnl)}
    wins   = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    pf     = wins.sum() / abs(losses.sum()) if len(losses) and losses.sum() != 0 else np.inf
    spt    = pnl.mean() / pnl.std()
    return {
        "n":              len(pnl),
        "mean_pnl":       float(pnl.mean()),
        "hit_rate":       float((pnl > 0).mean()),
        "sharpe_pertrade": float(spt),
        "sharpe_ann":     float(spt * np.sqrt(252.0 / horizon)),
        "profit_factor":  float(pf),
        "t_stat":         float(pnl.mean() / (pnl.std() / np.sqrt(len(pnl)))),
    }


def oos_edge_test_fast(
    features: pd.DataFrame,
    y: pd.Series,
    filter_col: str,
    *,
    horizon: int = 20,
    train_window: int = 504,
) -> dict:
    """Vectorized OOS filter — no Python loop over windows.

    At each bar t, uses data [t-train_window, t-1] to estimate whether the
    feature directionally predicts signal quality (rolling Pearson IC vs realized
    PnL).  Takes the signal when the feature is on the IC-predicted good side of
    the rolling median.  Bars before the first full training window are excluded.

    Returns same keys as oos_edge_test: unfiltered, filtered, lift, filter_mask.
    """
    fwd = y.diff(horizon).shift(-horizon)
    combined = pd.concat([
        features["ou_zscore"].rename("z"),
        features[filter_col].rename("feat"),
        fwd.rename("fwd"),
    ], axis=1).dropna()
    combined = combined[combined["z"] != 0].copy()
    combined["pnl"] = np.sign(combined["z"]) * (-combined["fwd"])

    feat = combined["feat"]
    pnl  = combined["pnl"]
    min_periods = max(30, train_window // 4)

    # shift(1) so training window ends at t-1 — no lookahead
    rolling_ic  = feat.shift(1).rolling(train_window, min_periods=min_periods).corr(pnl.shift(1))
    rolling_med = feat.shift(1).rolling(train_window, min_periods=min_periods).median()

    above_med   = feat > rolling_med
    ic_valid    = rolling_ic.notna()
    filter_mask = ic_valid & ((rolling_ic > 0) & above_med | (rolling_ic <= 0) & ~above_med)

    unfiltered = _simple_stats(pnl, horizon)
    filtered   = _simple_stats(pnl[filter_mask], horizon)
    lift = filtered.get("sharpe_ann", np.nan) - unfiltered.get("sharpe_ann", np.nan)

    return {
        "unfiltered":  unfiltered,
        "filtered":    filtered,
        "lift":        lift,
        "filter_mask": filter_mask,
    }


def oos_edge_summary_fast(
    features: pd.DataFrame,
    y: pd.Series,
    *,
    horizon: int = 20,
    train_window: int = 504,
    feature_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Run oos_edge_test_fast over all (or selected) feature columns.

    Fully vectorized — suitable for param sweeps.  Returns same format as
    oos_edge_summary, sorted by lift descending.
    """
    if feature_cols is None:
        feature_cols = [c for c in features.columns if c != "ou_zscore"]

    rows = []
    for col in feature_cols:
        try:
            result = oos_edge_test_fast(features, y, col, horizon=horizon, train_window=train_window)
        except Exception:
            continue
        u  = result["unfiltered"]
        f_ = result["filtered"]
        n_u = u.get("n", 0)
        rows.append({
            "feature":           col,
            "n_unfiltered":      n_u,
            "n_filtered":        f_.get("n", np.nan),
            "pct_kept":          f_.get("n", np.nan) / n_u if n_u else np.nan,
            "sharpe_unfiltered": u.get("sharpe_ann", np.nan),
            "sharpe_filtered":   f_.get("sharpe_ann", np.nan),
            "lift":              result["lift"],
            "hit_unfiltered":    u.get("hit_rate", np.nan),
            "hit_filtered":      f_.get("hit_rate", np.nan),
        })

    if not rows:
        return pd.DataFrame()

    return (pd.DataFrame(rows)
              .sort_values("lift", ascending=False)
              .reset_index(drop=True))


# ─── walk-forward OOS edge test (reference implementation) ───────────────────

def oos_edge_test(
    features: pd.DataFrame,
    y: pd.Series,
    filter_col: str,
    *,
    horizon: int = 20,
    train_window: int = 504,
    step: int = 63,
    n_bins: int = 5,
    min_obs_per_bin: int = 15,
) -> dict:
    """Walk-forward OOS filter: does the context feature add edge?

    For each OOS window:
    1. Fit the prior `train_window` bars — compute per-bin IC for `filter_col`.
    2. Identify bins with positive IC (the "good" context).
    3. In the OOS window, only take signals that fall in the good bins.
    4. Roll forward by `step` bars and repeat.

    Returns
    -------
    dict with keys:
        unfiltered   stats dict for all signal bars (baseline)
        filtered     stats dict for OOS-filtered signal bars
        lift         filtered sharpe_ann - unfiltered sharpe_ann
        filter_log   DataFrame: one row per OOS window with good_bins and in-sample ICs
        filter_mask  boolean Series aligned to features.index
    """
    fwd = y.diff(horizon).shift(-horizon)
    combined = pd.concat([
        features["ou_zscore"].rename("z"),
        features[filter_col].rename("feat"),
        fwd.rename("fwd"),
    ], axis=1).dropna()
    combined = combined[combined["z"] != 0].copy()
    combined["pnl"] = np.sign(combined["z"]) * (-combined["fwd"])

    dates = combined.index
    n     = len(dates)
    filter_mask = pd.Series(False, index=combined.index)
    log_rows: list[dict] = []

    for start_i in range(train_window, n, step):
        train = combined.iloc[start_i - train_window : start_i]
        test  = combined.iloc[start_i : min(start_i + step, n)]
        if test.empty:
            break

        # In-sample: find which feature bins have positive IC
        try:
            train = train.copy()
            train["bin"], edges = pd.qcut(
                train["feat"], q=n_bins, labels=False,
                retbins=True, duplicates="drop",
            )
        except Exception:
            continue

        bin_ic: dict[int, float] = {}
        for b, grp in train.groupby("bin"):
            if len(grp) < min_obs_per_bin:
                continue
            ic = float(grp["z"].corr(-grp["fwd"], method="spearman"))
            bin_ic[int(b)] = ic

        good_bins = {b for b, ic in bin_ic.items() if ic > 0}
        if not good_bins:
            continue

        # Apply to OOS using in-sample bin edges
        edges[0]  = -np.inf
        edges[-1] = np.inf
        oos_bins = pd.cut(test["feat"], bins=edges, labels=False)
        passes   = oos_bins.isin(good_bins)
        filter_mask.loc[passes[passes].index] = True

        log_rows.append({
            "train_end":  dates[start_i - 1],
            "test_start": dates[start_i],
            "test_end":   dates[min(start_i + step, n) - 1],
            "good_bins":  sorted(good_bins),
            "n_good_bins": len(good_bins),
            "avg_ic_good": float(np.mean([ic for b, ic in bin_ic.items() if b in good_bins])),
            **{f"bin{b}_ic": ic for b, ic in bin_ic.items()},
        })

    unfiltered_pnl = combined["pnl"]
    filtered_pnl   = combined.loc[filter_mask, "pnl"]

    unfiltered = _simple_stats(unfiltered_pnl, horizon)
    filtered   = _simple_stats(filtered_pnl,   horizon)

    lift = (
        filtered.get("sharpe_ann", np.nan) - unfiltered.get("sharpe_ann", np.nan)
    )

    return {
        "unfiltered":  unfiltered,
        "filtered":    filtered,
        "lift":        lift,
        "filter_log":  pd.DataFrame(log_rows),
        "filter_mask": filter_mask,
    }


def filtered_sharpe_summary(
    models: dict,
    y: pd.Series,
    *,
    horizons: tuple[int, ...] = (5, 20, 60),
    train_window: int = 504,
) -> pd.DataFrame:
    """Best OOS-filtered sharpe per anchor across all features and horizons.

    For each anchor × horizon, runs oos_edge_summary and records the top feature
    by sharpe_filtered. Returns one row per anchor (best horizon wins), sorted by
    sharpe_filtered descending.
    """
    rows = []
    for anchor, model in tqdm(models.items(), desc="filtered sharpe scan", unit="anchor"):
        feats = build_signal_features(model)
        if feats.empty or feats["ou_zscore"].dropna().empty:
            continue

        best: dict | None = None
        for h in horizons:
            summary = oos_edge_summary_fast(feats, y, horizon=h, train_window=train_window)
            if summary.empty:
                continue
            top = summary.iloc[0]  # sorted by lift desc — take the best filter
            sharpe_f = top["sharpe_filtered"]
            if not np.isfinite(sharpe_f):
                continue
            if best is None or sharpe_f > best["sharpe_filtered"]:
                best = {
                    "anchor":            anchor,
                    "horizon":           h,
                    "best_feature":      top["feature"],
                    "sharpe_unfiltered": top["sharpe_unfiltered"],
                    "sharpe_filtered":   sharpe_f,
                    "lift":              top["lift"],
                    "pct_kept":          top["pct_kept"],
                    "hit_filtered":      top["hit_filtered"],
                }

        if best is not None:
            rows.append(best)

    if not rows:
        return pd.DataFrame()

    return (pd.DataFrame(rows)
              .sort_values("sharpe_filtered", ascending=False)
              .reset_index(drop=True))


def oos_edge_summary(
    features: pd.DataFrame,
    y: pd.Series,
    *,
    horizon: int = 20,
    train_window: int = 504,
    step: int = 63,
    n_bins: int = 5,
    feature_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Run oos_edge_test over all (or selected) feature columns.

    Returns a summary DataFrame sorted by OOS Sharpe lift, so you can
    quickly see which context features add the most out-of-sample value.
    """
    if feature_cols is None:
        feature_cols = [c for c in features.columns if c != "ou_zscore"]

    rows = []
    for col in feature_cols:
        try:
            result = oos_edge_test(
                features, y, col,
                horizon=horizon,
                train_window=train_window,
                step=step,
                n_bins=n_bins,
            )
        except Exception:
            continue
        u = result["unfiltered"]
        f_ = result["filtered"]
        rows.append({
            "feature":             col,
            "n_unfiltered":        u.get("n", np.nan),
            "n_filtered":          f_.get("n", np.nan),
            "pct_kept":            f_.get("n", np.nan) / u.get("n", 1),
            "sharpe_unfiltered":   u.get("sharpe_ann", np.nan),
            "sharpe_filtered":     f_.get("sharpe_ann", np.nan),
            "lift":                result["lift"],
            "hit_unfiltered":      u.get("hit_rate", np.nan),
            "hit_filtered":        f_.get("hit_rate", np.nan),
        })

    if not rows:
        return pd.DataFrame()

    return (pd.DataFrame(rows)
              .sort_values("lift", ascending=False)
              .reset_index(drop=True))
