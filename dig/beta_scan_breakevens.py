from __future__ import annotations

import numpy as np
import pandas as pd

from stats import roll_lr_diff, ou_zscore
from utils.market_data import load_wide, pick_ticker
from utils.viz import Viz

START = "2010-01-01"
LOOKBACKS = [60, 120, 252, 504]
PLOT_LOOKBACK = 252
TO_BPS = True
MIN_R2_CORR = 0.02
MAX_BETA_CV = 1.50
BETA_CV_CAP = 2.00
ACTION_Z = 1.50

BE_TICKERS = {
    "5y_be": "USGGBE05Y Index",
    "10y_be": "USGGBE10Y Index",
    "30y_be": "USGGBE30Y Index",
}

FACTOR_CANDIDATES = {
    "crude": ["CL1 Comdty"],
    "5y_dur": ["USGG5YR Index"],
    "10y_dur": ["USGG10YR Index"],
    "30y_dur": ["USGG30YR Index"],
}


def pull_px(tickers: list[str], start: str) -> pd.DataFrame:
    return load_wide(tickers, start=start, to_pandas=True)


def fit_pair(px: pd.DataFrame, y_col: str, x_col: str, lookback: int) -> pd.DataFrame | None:
    pair = px[[y_col, x_col]].dropna().copy()
    if len(pair) < lookback + 10:
        return None

    raw = roll_lr_diff(pair[x_col], pair[y_col], lookback=lookback).to_pandas()
    raw.index = pair.index[1:]

    reg = pd.DataFrame(index=raw.index)
    reg["y"] = raw["y"]
    reg["x"] = raw["x"]
    reg["alpha"] = raw["alpha"]
    reg["beta"] = raw["beta"]
    reg["resid"] = raw["resid_cum"]

    z = ou_zscore(reg["resid"], lookback=lookback).to_pandas()
    z.index = reg.index
    reg["ou_z"] = z

    reg["r2_corr"] = raw["r2"].clip(lower=0.0, upper=1.0)

    beta_abs_mean = reg["beta"].abs().rolling(lookback).mean()
    reg["beta_cv"] = (reg["beta"].rolling(lookback).std() / beta_abs_mean).replace(
        [np.inf, -np.inf], np.nan
    )

    reg["fair"] = reg["y"] - reg["resid"]
    return reg


def build_composite(
    results: dict[tuple[str, int], pd.DataFrame],
    factor_names: list[str],
    lookback: int,
) -> pd.DataFrame:
    comp = pd.DataFrame()

    for factor_name in factor_names:
        reg = results.get((factor_name, lookback))
        if reg is None:
            continue

        block = pd.DataFrame(index=reg.index)
        block[f"{factor_name}_z"] = reg["ou_z"]
        block[f"{factor_name}_r2"] = reg["r2_corr"]
        block[f"{factor_name}_beta_cv"] = reg["beta_cv"]

        gate = (block[f"{factor_name}_r2"] >= MIN_R2_CORR) & (
            block[f"{factor_name}_beta_cv"] <= MAX_BETA_CV
        )
        qual = (
            block[f"{factor_name}_r2"]
            * (1.0 - (block[f"{factor_name}_beta_cv"] / BETA_CV_CAP).clip(0.0, 1.0))
        ).clip(lower=0.0, upper=1.0)

        block[f"{factor_name}_gate"] = gate
        block[f"{factor_name}_w"] = qual.where(gate, 0.0)
        block[f"{factor_name}_zw"] = block[f"{factor_name}_z"] * block[f"{factor_name}_w"]

        comp = block if comp.empty else comp.join(block, how="outer")

    if comp.empty:
        return comp

    w_cols = [f"{f}_w" for f in factor_names if f"{f}_w" in comp.columns]
    zw_cols = [f"{f}_zw" for f in factor_names if f"{f}_zw" in comp.columns]
    g_cols = [f"{f}_gate" for f in factor_names if f"{f}_gate" in comp.columns]

    comp["w_sum"] = comp[w_cols].sum(axis=1, min_count=1)
    comp["composite_z"] = comp[zw_cols].sum(axis=1, min_count=1) / comp["w_sum"]
    comp.loc[comp["w_sum"] <= 0.0, "composite_z"] = np.nan
    comp["n_active"] = comp[g_cols].sum(axis=1, min_count=1)
    return comp


def scan_be(be_name: str, be_ticker: str, px: pd.DataFrame, resolved_factors: dict[str, str], v: Viz):
    print(f"\n{'='*60}")
    print(f"  {be_name} ({be_ticker})")
    print(f"{'='*60}")

    if be_ticker not in px.columns:
        print(f"  [skip] {be_ticker} not in data")
        return

    results: dict[tuple[str, int], pd.DataFrame] = {}
    rows = []

    for factor_name, factor_ticker in resolved_factors.items():
        for lb in LOOKBACKS:
            reg = fit_pair(px, be_ticker, factor_ticker, lb)
            if reg is None:
                continue
            results[(factor_name, lb)] = reg

            tail = reg.dropna(subset=["beta", "r2_corr", "beta_cv", "resid", "ou_z"])
            if tail.empty:
                continue
            last = tail.iloc[-1]
            active = bool(
                (float(last["r2_corr"]) >= MIN_R2_CORR)
                and (float(last["beta_cv"]) <= MAX_BETA_CV)
            )

            rows.append(
                {
                    "factor": factor_name,
                    "lookback": lb,
                    "beta": float(last["beta"]),
                    "r2_corr": float(last["r2_corr"]),
                    "beta_cv": float(last["beta_cv"]),
                    "resid": float(last["resid"]),
                    "ou_z": float(last["ou_z"]),
                    "active": active,
                    "date": tail.index[-1].date(),
                }
            )

    if not rows:
        print("  No valid regressions.")
        return

    summary = pd.DataFrame(rows).sort_values(["factor", "lookback"])
    print(f"\nLatest beta-weighted stats for {be_name}:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

    comp = build_composite(results, list(resolved_factors.keys()), PLOT_LOOKBACK)

    if not comp.empty:
        live = comp.dropna(subset=["composite_z"])
        if not live.empty:
            last_idx = live.index[-1]
            last_row = live.iloc[-1]
            z_now = float(last_row["composite_z"])

            if z_now >= ACTION_Z:
                action = f"SHORT {be_name} (rich vs factors)"
            elif z_now <= -ACTION_Z:
                action = f"LONG {be_name} (cheap vs factors)"
            else:
                action = "FLAT"

            print(
                f"\nComposite ({PLOT_LOOKBACK}d): date={last_idx.date()} "
                f"z={z_now:+.3f} "
                f"active={int(last_row['n_active'])}/{len(resolved_factors)} "
                f"action={action} (|z|>{ACTION_Z})"
            )

            contrib = []
            w_sum = float(last_row["w_sum"]) if pd.notna(last_row["w_sum"]) else np.nan
            for factor_name in resolved_factors.keys():
                z_col = f"{factor_name}_z"
                w_col = f"{factor_name}_w"
                g_col = f"{factor_name}_gate"
                if z_col not in live.columns or w_col not in live.columns:
                    continue

                fz = float(live.loc[last_idx, z_col])
                fw = float(live.loc[last_idx, w_col])
                fg = bool(live.loc[last_idx, g_col]) if g_col in live.columns else False
                fctr = (fz * fw / w_sum) if (w_sum > 0 and np.isfinite(w_sum)) else np.nan
                contrib.append(
                    {
                        "factor": factor_name,
                        "z": fz,
                        "weight": fw,
                        "active": fg,
                        "z_contrib": fctr,
                    }
                )

            if contrib:
                print("Composite contributions:")
                print(
                    pd.DataFrame(contrib)
                    .sort_values("weight", ascending=False)
                    .to_string(index=False, float_format=lambda x: f"{x:,.4f}")
                )

            v.line(
                live[["composite_z"]].dropna(),
                title=f"Composite OU z-score: {be_name} ({PLOT_LOOKBACK}d)",
                residual=True,
            )

    # Per-factor plots
    for factor_name in resolved_factors.keys():
        resid_cmp = pd.DataFrame()
        for lb in LOOKBACKS:
            reg = results.get((factor_name, lb))
            if reg is not None:
                resid_cmp[f"resid_{lb}d"] = reg["resid"]

        if not resid_cmp.empty:
            v.line(
                resid_cmp.dropna(how="all"),
                title=f"Residuals across lookbacks: {be_name} ~ {factor_name}",
                residual=False,
            )

        reg = results.get((factor_name, PLOT_LOOKBACK))
        if reg is None:
            continue

        panel = pd.DataFrame(index=reg.index)
        panel[be_name] = reg["y"]
        panel["fair"] = reg["fair"]
        panel["beta"] = reg["beta"]
        panel["resid"] = reg["resid"]
        panel["ou_z"] = reg["ou_z"]
        panel["r2_corr"] = reg["r2_corr"]
        panel["beta_cv"] = reg["beta_cv"]

        v.line(
            panel[[be_name, "fair"]].dropna(),
            title=f"{be_name} vs beta-weighted fair ({factor_name}, {PLOT_LOOKBACK}d)",
        )
        v.line(
            panel[["resid"]].dropna(),
            title=f"Residual: {be_name} ~ {factor_name} ({PLOT_LOOKBACK}d)",
            residual=True,
        )
        v.line(
            panel[["ou_z"]].dropna(),
            title=f"OU z-score: {be_name} ~ {factor_name} ({PLOT_LOOKBACK}d)",
            residual=True,
        )
        v.line(
            panel[["beta"]].dropna(),
            title=f"Rolling beta: {be_name} vs {factor_name} ({PLOT_LOOKBACK}d)",
        )


def main() -> None:
    # Resolve factor tickers
    resolved_factors: dict[str, str] = {}
    for name, candidates in FACTOR_CANDIDATES.items():
        t = pick_ticker(candidates, START)
        if t is None:
            print(f"[skip] {name}: no ticker found in {candidates}")
            continue
        resolved_factors[name] = t

    if not resolved_factors:
        raise RuntimeError("No factor tickers resolved.")

    # Pull all data in one shot
    all_tickers = (
        list(BE_TICKERS.values())
        + sorted(set(resolved_factors.values()))
    )
    px = pull_px(all_tickers, START)

    if px.empty:
        raise RuntimeError("No data returned from md.index_eod.")

    if TO_BPS:
        px = px * 100.0

    last_date = px.index.max().date()
    print("Resolved factors:")
    for k, t in resolved_factors.items():
        print(f"  {k:10s} -> {t}")
    print(f"Data through: {last_date}")
    if pd.Timestamp(last_date) < pd.Timestamp.today().normalize() - pd.Timedelta(days=3):
        print("WARNING: data looks stale vs today.")

    v = Viz()

    for be_name, be_ticker in BE_TICKERS.items():
        scan_be(be_name, be_ticker, px, resolved_factors, v)


if __name__ == "__main__":
    main()
