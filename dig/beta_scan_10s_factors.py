from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from stats import roll_lr, ou_zscore
from utils.viz import Viz

DB_DSN = os.getenv("DB_DSN", "postgresql://benjils:snickers@raptor:5432/markets")
START = "2010-01-01"
LOOKBACKS = [60, 120, 252, 504]
PLOT_LOOKBACK = 252
TO_BPS = True
MIN_R2_CORR = 0.02
MAX_BETA_CV = 1.50
BETA_CV_CAP = 2.00
ACTION_Z = 1.50

BASE_TICKER = "USGG10YR Index"

FACTOR_CANDIDATES = {
    "10s30s": ["USYC1030 Index"],
    "10y_swaps": ["USSWIT10 Curncy", "USSFCT10 Curncy"],
    "5s10s30s": ["BF051030 Index"],
}


def query_df(conn, sql: str, params: list | tuple | None = None) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
        cols = [d.name for d in cur.description]
    return pd.DataFrame(rows, columns=cols)


def pick_ticker(conn, candidates: list[str], start: str) -> str | None:
    q = """
    SELECT ticker, COUNT(*) AS n
    FROM md.index_eod
    WHERE ticker = ANY(%s)
      AND ts >= %s
    GROUP BY ticker
    ORDER BY n DESC
    LIMIT 1
    """
    out = query_df(conn, q, params=(candidates, start))
    if out.empty:
        return None
    return str(out.loc[0, "ticker"])


def pull_px(conn, tickers: list[str], start: str) -> pd.DataFrame:
    q = """
    SELECT ts, ticker, px_last
    FROM md.index_eod
    WHERE ticker = ANY(%s)
      AND ts >= %s
    ORDER BY ts
    """
    df = query_df(conn, q, params=(tickers, start))
    if df.empty:
        return pd.DataFrame()
    df["ts"] = pd.to_datetime(df["ts"])
    return df.pivot(index="ts", columns="ticker", values="px_last").sort_index()


def fit_pair(px: pd.DataFrame, y_col: str, x_col: str, lookback: int) -> pd.DataFrame | None:
    pair = px[[y_col, x_col]].dropna().copy()
    if len(pair) < lookback + 10:
        return None

    reg = roll_lr(pair[x_col], pair[y_col], lookback=lookback).to_pandas()
    reg.index = pair.index

    z = ou_zscore(reg["resid"], lookback=lookback).to_pandas()
    z.index = reg.index
    reg["ou_z"] = z

    corr = pair[y_col].rolling(lookback).corr(pair[x_col])
    reg["r2_corr"] = (corr ** 2).clip(lower=0.0, upper=1.0)

    beta_abs_mean = reg["beta"].abs().rolling(lookback).mean()
    reg["beta_cv"] = (reg["beta"].rolling(lookback).std() / beta_abs_mean).replace(
        [np.inf, -np.inf], np.nan
    )

    reg["fair"] = reg["alpha"] + reg["beta"] * pair[x_col]
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


def main() -> None:
    with psycopg.connect(DB_DSN) as conn:
        resolved: dict[str, str] = {}
        for name, candidates in FACTOR_CANDIDATES.items():
            t = pick_ticker(conn, candidates, START)
            if t is None:
                print(f"[skip] {name}: no ticker found in {candidates}")
                continue
            resolved[name] = t

        if not resolved:
            raise RuntimeError("No factor tickers resolved. Update FACTOR_CANDIDATES.")

        all_tickers = [BASE_TICKER] + sorted(set(resolved.values()))
        px = pull_px(conn, all_tickers, START)

    if px.empty:
        raise RuntimeError("No data returned from md.index_eod.")

    if BASE_TICKER not in px.columns:
        raise RuntimeError(f"Missing base series: {BASE_TICKER}")

    if TO_BPS:
        px = px * 100.0

    last_date = px.index.max().date()
    print("\nResolved tickers:")
    for k, v in resolved.items():
        print(f"  {k:10s} -> {v}")
    print(f"Data through: {last_date}")
    if pd.Timestamp(last_date) < pd.Timestamp.today().normalize() - pd.Timedelta(days=3):
        print("WARNING: data looks stale vs today.")

    results: dict[tuple[str, int], pd.DataFrame] = {}
    rows = []

    for factor_name, factor_ticker in resolved.items():
        for lb in LOOKBACKS:
            reg = fit_pair(px, BASE_TICKER, factor_ticker, lb)
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
        raise RuntimeError("No valid regressions. Check lookbacks/data coverage.")

    summary = pd.DataFrame(rows).sort_values(["factor", "lookback"])
    print("\nLatest beta-weighted stats:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

    comp = build_composite(results, list(resolved.keys()), PLOT_LOOKBACK)

    v = Viz()

    if not comp.empty:
        live = comp.dropna(subset=["composite_z"])
        if not live.empty:
            last_idx = live.index[-1]
            last_row = live.iloc[-1]
            z_now = float(last_row["composite_z"])

            if z_now >= ACTION_Z:
                action = "SHORT 10s"
            elif z_now <= -ACTION_Z:
                action = "LONG 10s"
            else:
                action = "FLAT"

            print(
                f"\nComposite ({PLOT_LOOKBACK}d): date={last_idx.date()} "
                f"z={z_now:+.3f} "
                f"active={int(last_row['n_active'])}/{len(resolved)} "
                f"action={action} (|z|>{ACTION_Z})"
            )

            contrib = []
            w_sum = float(last_row["w_sum"]) if pd.notna(last_row["w_sum"]) else np.nan
            for factor_name in resolved.keys():
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
                title=f"Composite OU z-score ({PLOT_LOOKBACK}d)",
                residual=True,
            )

    for factor_name in resolved.keys():
        resid_cmp = pd.DataFrame()
        for lb in LOOKBACKS:
            reg = results.get((factor_name, lb))
            if reg is not None:
                resid_cmp[f"resid_{lb}d"] = reg["resid"]

        if not resid_cmp.empty:
            v.line(
                resid_cmp.dropna(how="all"),
                title=f"Residuals across lookbacks: 10s ~ {factor_name}",
                residual=False,
            )

        reg = results.get((factor_name, PLOT_LOOKBACK))
        if reg is None:
            continue

        panel = pd.DataFrame(index=reg.index)
        panel["10s"] = reg["y"]
        panel["fair"] = reg["fair"]
        panel["beta"] = reg["beta"]
        panel["resid"] = reg["resid"]
        panel["ou_z"] = reg["ou_z"]
        panel["r2_corr"] = reg["r2_corr"]
        panel["beta_cv"] = reg["beta_cv"]

        v.line(
            panel[["10s", "fair"]].dropna(),
            title=f"10s vs beta-weighted fair ({factor_name}, {PLOT_LOOKBACK}d)",
        )
        v.line(
            panel[["resid"]].dropna(),
            title=f"Residual: 10s ~ {factor_name} ({PLOT_LOOKBACK}d)",
            residual=True,
        )
        v.line(
            panel[["ou_z"]].dropna(),
            title=f"OU z-score: 10s ~ {factor_name} ({PLOT_LOOKBACK}d)",
            residual=True,
        )
        v.line(
            panel[["beta"]].dropna(),
            title=f"Rolling beta: 10s vs {factor_name} ({PLOT_LOOKBACK}d)",
        )
        v.line(
            panel[["r2_corr"]].dropna(),
            title=f"Rolling corr^2: 10s vs {factor_name} ({PLOT_LOOKBACK}d)",
        )
        v.line(
            panel[["beta_cv"]].dropna(),
            title=f"Rolling beta CV: 10s vs {factor_name} ({PLOT_LOOKBACK}d)",
        )


if __name__ == "__main__":
    main()
