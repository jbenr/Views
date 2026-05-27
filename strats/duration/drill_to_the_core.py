"""Drill to the core — recursive regression diagnostics for rates research.

Pattern: pick (x, y). Rolling-regress y on x. Inspect the residual. Peel a
known factor z out of it. Repeat. The story of the analysis lives in main();
the helpers above are reusable building blocks.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
import polars as pl
import psycopg
from IPython.display import display
from statsmodels.tsa.stattools import acf, adfuller

_HERE = Path(__file__).resolve().parent
for _p in [_HERE, *_HERE.parents]:
    if (_p / "stats").exists():
        ROOT = _p
        break
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest import (
    BacktestConfig,
    Engine,
    SignalConfig,
    SignalPipeline,
    TradeDef,
    half_drift_residual,
)
from backtest.metrics import trade_log as engine_trade_log
from stats import half_life, ou_params, roll_half_life, roll_lr, roll_ou_zscore
from strats.duration.signal_context import (
    build_signal_features,
    conditional_ic_table,
    filtered_sharpe_summary,
    oos_edge_summary,
    oos_edge_test_fast,
)
from utils.rates import linear_5y5y_forward
from utils.viz import Viz

warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", 200)


# ─── config ────────────────────────────────────────────────────────────────

DB_DSN = os.getenv("DB_DSN", "postgresql://benjils:snickers@raptor:5432/markets")
START = "2000-01-01"

TICKERS = {
    "2y": "USGG2YR Index",
    "5y": "USGG5YR Index",
    "10y": "USGG10YR Index",
    "30y": "USGG30YR Index",
    "be5": "USGGBE05 Index",
    "be10": "USGGBE10 Index",
    "spx": "SPX Index",
    "oil": "CO1 Comdty",
    "dxy": "DXY Curncy",
    "move": "MOVE Index",
    "mtg_cc": "MTGEFNCL Index",
    # components for 5y5y forwards
    "zc5": "USSWIT5 Curncy",
    "zc10": "USSWIT10 Curncy",
    "sofr5": "USOSFR5 Curncy",
    "sofr10": "USOSFR10 Curncy",
}

BPS_COLS = (
    "2y",
    "5y",
    "10y",
    "30y",
    "be5",
    "be10",
    "mtg_cc",
    "zc5",
    "zc10",
    "sofr5",
    "sofr10",
)


# ─── data ──────────────────────────────────────────────────────────────────


def _query_to_pl(sql: str) -> pl.DataFrame:
    with psycopg.connect(DB_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            cols = [d.name for d in cur.description]
            rows = cur.fetchall()
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame({c: [r[i] for r in rows] for i, c in enumerate(cols)})


def load_basket(tickers: dict[str, str] = TICKERS, start: str = START) -> pd.DataFrame:
    """Pull `tickers` from md.index_eod, pivot wide. Yields scaled to bps."""
    tlist = ", ".join(f"'{t}'" for t in tickers.values())
    raw = _query_to_pl(f"""
        SELECT ts, ticker, px_last::float AS px
        FROM md.index_eod
        WHERE ticker IN ({tlist}) AND ts >= '{start}'
        ORDER BY ts
    """)
    wide = (
        raw.to_pandas()
        .pivot(index="ts", columns="ticker", values="px")
        .sort_index()
        .rename(columns={v: k for k, v in tickers.items()})
    )
    for c in BPS_COLS:
        if c in wide:
            wide[c] = wide[c] * 100.0
    # derived: mortgage basis = current coupon yield - 10y (both already in bps)
    if "mtg_cc" in wide.columns and "10y" in wide.columns:
        wide["mtg_basis"] = wide["mtg_cc"] - wide["10y"]
    # derived: 5y5y forwards via linear approx (2×10y − 5y)
    if "zc5" in wide.columns and "zc10" in wide.columns:
        wide["5y5y_ifs"] = linear_5y5y_forward(wide["zc5"], wide["zc10"])
    if "sofr5" in wide.columns and "sofr10" in wide.columns:
        wide["5y5y_sfr"] = linear_5y5y_forward(wide["sofr5"], wide["sofr10"])
    return wide.dropna(how="all")


# ─── residual diagnostics ──────────────────────────────────────────────────


def residual_fingerprint(resid: pd.Series) -> dict:
    """Compact stats describing what a residual looks like."""
    r = resid.dropna()
    if len(r) < 60:
        return {"n": len(r)}
    ac = acf(r, nlags=5, fft=True)
    try:
        adf_p = float(adfuller(r, autolag="AIC")[1])
    except Exception:
        adf_p = np.nan
    return {
        "n": len(r),
        "std": float(r.std()),
        "kurt": float(r.kurt()),
        "skew": float(r.skew()),
        "acf_lag1": float(ac[1]),
        "acf_lag5": float(ac[5]),
        "half_life": float(half_life(r)),
        "adf_p": adf_p,
    }


def _print_fingerprint(label: str, fp: dict) -> None:
    print(f"residual fingerprint  ({label}):")
    for k, val in fp.items():
        if isinstance(val, float):
            print(f"  {k:<10s}  {val:+.3f}")
        else:
            print(f"  {k:<10s}  {val}")


def drill(
    x: pd.Series,
    y: pd.Series,
    *,
    lookback: int = 60,
    x_name: str = "x",
    y_name: str = "y",
    viz: Viz | None = None,
) -> pd.DataFrame:
    """Rolling OLS of y on x. Plots inputs, rolling β/R², and the residual.

    Returns pandas DataFrame indexed by date with columns
    [x, y, alpha, beta, yhat, resid, r2].
    """
    pair = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    res = roll_lr(pair["x"], pair["y"], lookback=lookback).to_pandas()
    res.index = pair.index

    if viz is not None:
        viz.line(
            pair.rename(columns={"x": x_name, "y": y_name}),
            title=f"{y_name}  vs  {x_name}",
            yaxis_title="level",
        )
        viz.line(
            res[["beta", "r2"]],
            left=["r2"],
            title=f"rolling β and R²  (lookback={lookback})",
            yaxis_title="β",
            yaxis_right_title="R²",
        )
        viz.line(
            res[["resid"]].rename(columns={"resid": f"resid({y_name} | {x_name})"}),
            title=f"residual:  {y_name} − ({lookback}d β·{x_name} + α)",
            yaxis_title="resid",
            residual=True,
            hlines=[(0, None, "solid")],
        )

    _print_fingerprint(f"{y_name} | {x_name}", residual_fingerprint(res["resid"]))
    return res


def corr_scan(resid: pd.Series, candidates: pd.DataFrame) -> pd.DataFrame:
    """Corr of resid against each candidate column (level and change). Sorted by |corr_level|."""
    rows = []
    for col in candidates.columns:
        joint = pd.concat(
            [resid.rename("r"), candidates[col].rename("z")], axis=1
        ).dropna()
        if len(joint) < 100:
            continue
        rows.append(
            {
                "candidate": col,
                "corr_level": joint["r"].corr(joint["z"]),
                "corr_change": joint["r"].diff().corr(joint["z"].diff()),
                "n": len(joint),
            }
        )
    return (
        pd.DataFrame(rows)
        .assign(abs_corr=lambda d: d["corr_level"].abs())
        .sort_values("abs_corr", ascending=False)
        .drop(columns="abs_corr")
        .reset_index(drop=True)
    )


def predict_scan(
    resid: pd.Series,
    candidates: pd.DataFrame,
    horizons: tuple[int, ...] = (5, 20, 60),
) -> pd.DataFrame:
    """Spearman IC of resid_t vs fwd_change(candidate, h) for each h. Sorted by |ic|."""
    rows = []
    for col in candidates.columns:
        for h in horizons:
            fwd = candidates[col].diff(h).shift(-h)
            joint = pd.concat([resid.rename("r"), fwd.rename("f")], axis=1).dropna()
            if len(joint) < 100:
                continue
            rows.append(
                {
                    "target": col,
                    "horizon": h,
                    "ic": joint["r"].corr(joint["f"], method="spearman"),
                    "n": len(joint),
                }
            )
    return (
        pd.DataFrame(rows)
        .assign(abs_ic=lambda d: d["ic"].abs())
        .sort_values("abs_ic", ascending=False)
        .drop(columns="abs_ic")
        .reset_index(drop=True)
    )


def peel(
    resid: pd.Series,
    z: pd.Series,
    *,
    lookback: int = 60,
    resid_name: str = "resid",
    z_name: str = "z",
    viz: Viz | None = None,
) -> pd.DataFrame:
    """Strip the part of `resid` explained by `z`. Returns drill output for the new layer."""
    return drill(z, resid, lookback=lookback, x_name=z_name, y_name=resid_name, viz=viz)


def _latest_ou_metrics(resid: pd.Series, lookback: int = 252) -> dict:
    """Latest OU stats on a trailing residual window."""
    r = resid.dropna()
    if len(r) < max(20, lookback // 4):
        return {
            "ou_mean_resid": np.nan,
            "ou_zscore": np.nan,
            "half_life_d": np.nan,
            "ou_window_n": len(r),
        }

    window = r.tail(lookback)
    params = ou_params(window)
    sigma = float(window.std())
    mu = float(params["mu"])
    current = float(window.iloc[-1])
    z = (current - mu) / sigma if sigma > 0 and np.isfinite(mu) else np.nan
    return {
        "ou_mean_resid": mu,
        "ou_zscore": z,
        "half_life_d": float(params["half_life"]),
        "ou_window_n": len(window),
    }


def single_factor_10y_table(
    df: pd.DataFrame,
    *,
    target: str = "10y",
    lookback: int = 60,
    ou_lookback: int = 252,
    exclude: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Regress 10y against each candidate one at a time and summarize latest OU residual state."""
    rows = []
    models: dict[str, pd.DataFrame] = {}
    candidates = [c for c in df.columns if c != target and c not in exclude]

    for anchor in candidates:
        pair = df[[anchor, target]].dropna()
        if len(pair) < lookback:
            continue

        out = roll_lr(pair[anchor], pair[target], lookback=lookback).to_pandas()
        out.index = pair.index
        out["ou_zscore"] = roll_ou_zscore(out["resid"], lookback=ou_lookback).to_numpy()
        models[anchor] = out

        valid = out.dropna(subset=["y", "yhat", "resid"])
        if valid.empty:
            continue

        last = valid.iloc[-1]
        ou = _latest_ou_metrics(out["resid"], lookback=ou_lookback)
        ou_mean = ou["ou_mean_resid"]
        rows.append(
            {
                "anchor": anchor,
                "date": valid.index[-1],
                "actual_10y": float(last["y"]),
                "model_10y": float(last["yhat"]),
                "residual": float(last["resid"]),
                "ou_mean_resid": ou_mean,
                "ou_fair_10y": float(last["yhat"] + ou_mean)
                if np.isfinite(ou_mean)
                else np.nan,
                "ou_zscore": ou["ou_zscore"],
                "half_life_d": ou["half_life_d"],
                "beta": float(last["beta"]),
                "r2": float(last["r2"]),
                "ou_window_n": ou["ou_window_n"],
            }
        )

    table = pd.DataFrame(rows)
    if not table.empty:
        table = (
            table.assign(abs_ou_zscore=lambda d: d["ou_zscore"].abs())
            .sort_values("abs_ou_zscore", ascending=False)
            .drop(columns="abs_ou_zscore")
            .reset_index(drop=True)
        )
    return table, models


def single_factor_model(
    df: pd.DataFrame,
    *,
    anchor: str,
    target: str = "10y",
    lookback: int = 60,
    ou_lookback: int = 252,
) -> pd.DataFrame:
    """Build one rolling 10y residual model for parameter sweeps."""
    pair = df[[anchor, target]].dropna()
    if len(pair) < lookback:
        return pd.DataFrame()

    out = roll_lr(pair[anchor], pair[target], lookback=lookback).to_pandas()
    out.index = pair.index
    out["ou_zscore"] = roll_ou_zscore(out["resid"], lookback=ou_lookback).to_numpy()
    return out


# ─── backtest ──────────────────────────────────────────────────────────────


def _profit_factor(pnl: pd.Series) -> float:
    """Gross profit divided by gross loss; zeros do not affect either side."""
    wins = pnl[pnl > 0].sum()
    losses = pnl[pnl < 0].sum()
    if losses < 0:
        return float(wins / abs(losses))
    return float("inf") if wins > 0 else np.nan


def ou_signal_forward_diagnostics(
    models: dict[str, pd.DataFrame],
    *,
    horizons: tuple[int, ...] = (5, 20, 60),
) -> pd.DataFrame:
    """Spearman IC, hit rate, and 1d Sharpe for each model's OU z-score signal.

    Signal convention: ou_zscore > 0 → 10y is rich → expect yield to fall.
    Correct prediction: sign(ou_zscore_t) == sign(-Δy_{t+H}).
    """
    rows = []
    for anchor, model in models.items():
        row: dict = {"anchor": anchor}
        signal = model["ou_zscore"]
        y = model["y"]

        fwd_1d = y.diff(1).shift(-1)
        j1 = pd.concat([signal.rename("s"), fwd_1d.rename("f")], axis=1).dropna()
        j1 = j1[j1["s"] != 0]
        if len(j1) >= 30:
            pnl = np.sign(j1["s"]) * (-j1["f"])
            row["sharpe_1d"] = (
                float(pnl.mean() / pnl.std() * np.sqrt(252))
                if pnl.std() > 0
                else np.nan
            )
            row["pf_1d"] = _profit_factor(pnl)
        else:
            row["sharpe_1d"] = np.nan
            row["pf_1d"] = np.nan

        for h in horizons:
            fwd = y.diff(h).shift(-h)
            jh = pd.concat([signal.rename("s"), fwd.rename("f")], axis=1).dropna()
            jh = jh[jh["s"] != 0]
            if len(jh) < 30:
                row[f"ic_{h}d"] = np.nan
                row[f"hit_{h}d"] = np.nan
                row[f"sharpe_{h}d"] = np.nan
                row[f"pf_{h}d"] = np.nan
                continue
            pnl = np.sign(jh["s"]) * (-jh["f"])
            row[f"ic_{h}d"] = float(jh["s"].corr(-jh["f"], method="spearman"))
            row[f"hit_{h}d"] = float((np.sign(jh["s"]) == np.sign(-jh["f"])).mean())
            row[f"sharpe_{h}d"] = (
                float(pnl.mean() / pnl.std() * np.sqrt(252.0 / h))
                if pnl.std() > 0
                else np.nan
            )
            row[f"pf_{h}d"] = _profit_factor(pnl)

        rows.append(row)

    col_order = [
        "anchor",
        "ic_5d",
        "ic_20d",
        "ic_60d",
        "hit_5d",
        "hit_20d",
        "hit_60d",
        "sharpe_1d",
        "sharpe_5d",
        "sharpe_20d",
        "sharpe_60d",
        "pf_1d",
        "pf_5d",
        "pf_20d",
        "pf_60d",
    ]
    bt = pd.DataFrame(rows)
    return bt[[c for c in col_order if c in bt.columns]]


# ─── engine backtest + deep-dive charts ─────────────────────────────────


def _engine_data(model: pd.DataFrame) -> pl.DataFrame:
    """Build the minimal wide frame expected by the shared backtest engine."""
    return pl.DataFrame(
        {
            "ts": pd.to_datetime(model.index).date,
            "10y": model["y"].astype(float).to_numpy(),
        }
    )


def _gated_entry_signal(
    model: pd.DataFrame,
    filter_mask: pd.Series,
    *,
    entry_z: float = 1.0,
) -> pd.Series:
    """Gate entries with the OOS filter while preserving z-score exits.

    The shared SignalConfig uses thresholds for both entries and exits. When
    the filter is off, cap extreme z-scores just inside the entry bands so new
    positions cannot open, but mean-crossing exits still fire.
    """
    z = model["ou_zscore"].reindex(model.index).astype(float).copy()
    active = filter_mask.reindex(model.index, fill_value=False).astype(bool)
    eps = 1e-6
    z.loc[~active & (z >= entry_z)] = entry_z - eps
    z.loc[~active & (z <= -entry_z)] = -entry_z + eps
    return z


def _dynamic_time_stop(model: pd.DataFrame, lookback: int = 252) -> pd.Series:
    """Rolling half-life in bars, falling back to 20 when the fit is invalid."""
    hl = roll_half_life(model["resid"], lookback=lookback).to_pandas()
    hl.index = model.index
    hl = hl.where(np.isfinite(hl) & (hl > 0), 20.0)
    return hl.round().clip(lower=1)


def run_engine_backtest(
    model: pd.DataFrame,
    filter_mask: pd.Series,
    *,
    entry_z: float = 1.0,
    ou_lookback: int = 252,
    residual_target_frac: float = 0.5,
    mom_ser: pd.Series | None = None,
    entry_filter_fn=None,
) -> tuple:
    """Run the OU signal through the shared backtest engine.

    Internal engine convention: direction -1 profits when 10y yield falls.
    Display convention: direction -1 is BUY rates; direction +1 is SELL rates.

    mom_ser: optional momentum series aligned to model.index; included as
             "mom" extra column so entry_filter_fn can gate on direction.
    entry_filter_fn: optional callable (direction, bar) -> bool passed to
                     SignalConfig to gate entries. See SignalConfig docs.
    """
    signal = _gated_entry_signal(model, filter_mask, entry_z=entry_z)
    time_stop = _dynamic_time_stop(model, lookback=ou_lookback)

    # Rolling OU mean — the equilibrium the residual reverts to.
    ou_mean = model["resid"].rolling(ou_lookback, min_periods=ou_lookback // 4).mean()

    mom_aligned = mom_ser.reindex(model.index).fillna(0.0) if mom_ser is not None else None

    def compute(data: pl.DataFrame) -> pl.DataFrame:
        cols = {
            "signal": signal.to_numpy(),
            "time_stop": time_stop.to_numpy(),
            "resid": model["resid"].to_numpy(),
            "ou_mean": ou_mean.to_numpy(),
        }
        if mom_aligned is not None:
            cols["mom"] = mom_aligned.to_numpy()
        return pl.DataFrame(cols)

    pipeline = SignalPipeline(
        name="10y_ou_filtered",
        trade_def=TradeDef.outright("10y", "10y"),
        compute_fn=compute,
        config=SignalConfig(
            entry_long=-entry_z,
            entry_short=entry_z,
            exit_long=None,  # signal exits disabled — half_drift_residual owns this
            exit_short=None,
            exit_fn=half_drift_residual(residual_target_frac),
            entry_filter_fn=entry_filter_fn,
            max_positions=1,
        ),
    )
    result = (
        Engine(BacktestConfig(max_total_positions=1))
        .add_signal(pipeline)
        .run(_engine_data(model))
    )
    return result, signal, time_stop


def engine_metrics_frame(result) -> pd.DataFrame:
    """Compact display table from shared backtest metrics."""
    metrics = result.summary()
    keep = [
        "total_pnl_bps",
        "n_trades",
        "hit_rate",
        "avg_win_bps",
        "avg_loss_bps",
        "profit_factor",
        "sharpe",
        "max_drawdown_bps",
        "calmar",
        "avg_holding_days",
    ]
    return pd.DataFrame([{k: metrics.get(k, np.nan) for k in keep}])


def engine_trades_frame(result) -> pd.DataFrame:
    """Trade log with buy/sell naming layered on top of engine accounting."""
    trades = engine_trade_log(result.closed_trades).to_pandas()
    if trades.empty:
        return trades
    trades = trades.rename(
        columns={
            "entry_date": "entry",
            "exit_date": "exit",
            "entry_level": "entry_10y",
            "exit_level": "exit_10y",
        }
    )
    trades["entry"] = pd.to_datetime(trades["entry"])
    trades["exit"] = pd.to_datetime(trades["exit"])
    trades.insert(3, "side", np.where(trades["direction"] == "short", "BUY", "SELL"))
    return trades


def select_filtered_candidate(
    filt: pd.DataFrame,
    *,
    anchor: str | None = None,
    horizon: int | None = None,
    filter_col: str | None = None,
    filter_rule: str | None = None,
    filter_quantile: float | None = None,
) -> pd.Series | None:
    """Pick a row from the filtered summary, defaulting to the top-ranked row."""
    if filt.empty:
        return None

    candidates = filt.copy()
    if anchor is not None:
        candidates = candidates[candidates["anchor"] == anchor]
    if horizon is not None:
        candidates = candidates[candidates["horizon"] == horizon]
    if filter_col is not None:
        if "filter_col" in candidates.columns:
            candidates = candidates[candidates["filter_col"] == filter_col]
        else:
            candidates = candidates[candidates["best_feature"].str.startswith(filter_col)]
    if filter_rule is not None and "filter_rule" in candidates.columns:
        candidates = candidates[candidates["filter_rule"] == filter_rule]
    if filter_quantile is not None and "filter_quantile" in candidates.columns:
        candidates = candidates[np.isclose(candidates["filter_quantile"], filter_quantile)]

    if candidates.empty:
        return None
    return candidates.sort_values("sharpe_filtered", ascending=False).iloc[0]


def parameter_robustness(
    df: pd.DataFrame,
    *,
    anchor: str,
    filter_col: str,
    horizon: int,
    filter_cutoffs: dict[str, tuple[float, ...]] | None = None,
    reg_lookbacks: tuple[int, ...] = (40, 60, 90, 126),
    ou_lookbacks: tuple[int, ...] = (126, 252, 504),
    entry_zs: tuple[float, ...] = (1.5, 2.0, 2.5),
    residual_target_fracs: tuple[float, ...] = (0.25, 0.5, 0.75),
    train_window: int = 504,
) -> pd.DataFrame:
    """Small engine-scored robustness grid for one pre-selected signal idea."""
    if filter_cutoffs is None:
        filter_cutoffs = {
            "low": (0.25, 0.33, 0.50),
            "high": (0.50, 0.67, 0.75),
        }
    rows = []
    for reg_lb in reg_lookbacks:
        for ou_lb in ou_lookbacks:
            model = single_factor_model(
                df,
                anchor=anchor,
                lookback=reg_lb,
                ou_lookback=ou_lb,
            )
            if model.empty:
                continue

            feats = build_signal_features(model)
            if feats.empty or filter_col not in feats.columns:
                continue

            filter_specs = [
                (rule, q)
                for rule in ("low", "high")
                for q in filter_cutoffs.get(rule, ())
            ]
            for rule, q in filter_specs:
                result = oos_edge_test_fast(
                    feats,
                    df["10y"],
                    filter_col,
                    horizon=horizon,
                    train_window=train_window,
                    filter_rule=rule,
                    filter_quantile=q,
                )
                mask = result["filter_mask"]
                fwd = result["filtered"]
                for entry_z in entry_zs:
                    for target_frac in residual_target_fracs:
                        bt_result, _, _ = run_engine_backtest(
                            model,
                            mask,
                            entry_z=entry_z,
                            ou_lookback=ou_lb,
                            residual_target_frac=target_frac,
                        )
                        metrics = bt_result.summary()
                        rows.append(
                            {
                                "anchor": anchor,
                                "horizon": horizon,
                                "filter_col": filter_col,
                                "filter_rule": rule,
                                "filter_quantile": q,
                                "reg_lookback": reg_lb,
                                "ou_lookback": ou_lb,
                                "entry_z": entry_z,
                                "residual_target_frac": target_frac,
                                "n_trades": metrics.get("n_trades", 0),
                                "total_pnl_bps": metrics.get("total_pnl_bps", np.nan),
                                "sharpe": metrics.get("sharpe", np.nan),
                                "profit_factor": metrics.get("profit_factor", np.nan),
                                "hit_rate": metrics.get("hit_rate", np.nan),
                                "max_drawdown_bps": metrics.get("max_drawdown_bps", np.nan),
                                "avg_holding_days": metrics.get("avg_holding_days", np.nan),
                                "fwd_sharpe_filtered": fwd.get("sharpe_ann", np.nan),
                                "pct_kept": (
                                    fwd.get("n", np.nan) / result["unfiltered"].get("n", np.nan)
                                    if result["unfiltered"].get("n", 0)
                                    else np.nan
                                ),
                            }
                        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return (
        out.sort_values(["sharpe", "profit_factor", "n_trades"], ascending=False)
        .reset_index(drop=True)
    )


def _best_signal_drill(
    best_row: pd.Series,
    models: dict[str, pd.DataFrame],
    df: pd.DataFrame,
    viz,
    *,
    train_window: int = 504,
    entry_z: float = 1.0,
    ou_lookback: int = 252,
    residual_target_frac: float = 0.5,
    directional_mom_filter: bool = True,
) -> pd.DataFrame:
    """Three deep-dive charts + trade table for the best filtered signal."""
    anchor = best_row["anchor"]
    h = int(best_row["horizon"])
    filter_name = best_row["best_feature"]
    feat_col = best_row.get("filter_col", filter_name)
    filter_rule = best_row.get("filter_rule", "learned")
    filter_quantile = float(best_row.get("filter_quantile", 0.5) or 0.5)
    if not np.isfinite(filter_quantile):
        filter_quantile = 0.5

    model = models[anchor]
    feats = build_signal_features(model)
    result = oos_edge_test_fast(
        feats,
        df["10y"],
        feat_col,
        horizon=h,
        train_window=train_window,
        by_direction=directional_mom_filter,
        filter_rule=filter_rule,
        filter_quantile=float(filter_quantile),
    )

    mask = result["filter_mask"]
    filter_med = result["rolling_med"].reindex(model.index)
    feat_ser = feats[feat_col].reindex(model.index)

    bt_result, _, time_stop_ser = run_engine_backtest(
        model, mask,
        entry_z=entry_z,
        ou_lookback=ou_lookback,
        residual_target_frac=residual_target_frac,
        mom_ser=feat_ser,
        entry_filter_fn=None,
    )
    trades = engine_trades_frame(bt_result)
    ou_z = model["ou_zscore"].copy()
    y_10 = model["y"].copy()
    ou_z.index = pd.to_datetime(ou_z.index)
    y_10.index = pd.to_datetime(y_10.index)
    feat_ser.index = pd.to_datetime(feat_ser.index)
    filter_med.index = pd.to_datetime(filter_med.index)
    mask.index = pd.to_datetime(mask.index)
    sfx = f"10y | {anchor}  ·  filter: {filter_name}  ·  h={h}d"

    # ── Chart 1: signal conditions — two stacked panels, shared x-axis ────────
    # Top: OU z-score + entry markers.  Bottom: selected filter feature.
    def _render_conditions(fig, axes, start, end):
        ax_z, ax_f = axes

        sub_z = ou_z.loc[start:end]
        sub_f = feat_ser.loc[start:end]
        sub_med = filter_med.loc[start:end]
        sub_m = mask.reindex(sub_z.index, fill_value=False)

        # ── top: OU z-score ────────────────────────────────────────────────
        yz = max(float(sub_z.abs().max(skipna=True)) * 1.2, entry_z * 2.5)
        buy_on = (sub_m & (sub_z >= entry_z)).values.astype(bool)
        sell_on = (sub_m & (sub_z <= -entry_z)).values.astype(bool)

        ax_z.fill_between(sub_z.index, -yz, yz, where=buy_on,
                          color="#27AE60", alpha=0.13, zorder=1, label="BUY entry allowed")
        ax_z.fill_between(sub_z.index, -yz, yz, where=sell_on,
                          color="#C0392B", alpha=0.13, zorder=1, label="SELL entry allowed")
        ax_z.plot(sub_z.index, sub_z.values, color="#2980B9", linewidth=1.5,
                  label="OU z-score", zorder=3)
        ax_z.axhline(entry_z, color="#27AE60", linestyle="--", linewidth=0.9, alpha=0.8,
                     label=f"+{entry_z:.1f} entry")
        ax_z.axhline(-entry_z, color="#C0392B", linestyle="--", linewidth=0.9, alpha=0.8,
                     label=f"−{entry_z:.1f} entry")
        ax_z.axhline(0, color="#666", linestyle="-", linewidth=0.8)
        ax_z.set_ylim(-yz, yz)

        if not trades.empty:
            sub_t = trades[(trades["entry"] >= start) & (trades["entry"] <= end)]
            for _, t in sub_t.iterrows():
                d = t["entry"]
                if d in sub_z.index:
                    zv = sub_z.loc[d]
                    c = "#27AE60" if t["side"] == "BUY" else "#C0392B"
                    mk = "v" if t["side"] == "BUY" else "^"
                    tip_off = np.sqrt(90) / 2 / 72
                    dy = tip_off if mk == "v" else -tip_off
                    t_mk = ax_z.transData + mtransforms.ScaledTranslation(0, dy, fig.dpi_scale_trans)
                    ax_z.scatter([d], [zv], marker=mk, color=c, s=90, zorder=6,
                                 edgecolors="white", linewidths=0.5, transform=t_mk)

        viz._style_ax(ax_z, yaxis_title="z-score")
        ax_z.legend(
            loc="upper left",
            bbox_to_anchor=(0.01, 0.99),
            ncol=3,
            fontsize=8,
            frameon=False,
        )

        # ── bottom: selected filter feature ───────────────────────────────
        finite_f = pd.concat([sub_f, sub_med], axis=1).stack().dropna()
        if finite_f.empty:
            ylo, yhi = -1.0, 1.0
        else:
            fmin = min(float(finite_f.min()), 0.0)
            fmax = max(float(finite_f.max()), 0.0)
            pad = (fmax - fmin) * 0.15
            if pad == 0:
                pad = max(abs(fmax) * 0.15, 0.01)
            ylo, yhi = fmin - pad, fmax + pad
        pass_mask = sub_m.reindex(sub_f.index, fill_value=False).values.astype(bool)
        ax_f.fill_between(sub_f.index, ylo, yhi, where=pass_mask,
                          color="#2ECC71", alpha=0.10, zorder=1, label="filter pass")
        ax_f.plot(sub_f.index, sub_f.values, color="#F39C12", linewidth=1.2,
                  label=feat_col, zorder=3)
        med_label = (
            "zero line"
            if filter_rule == "directional"
            else f"rolling q{int(round(float(filter_quantile) * 100))}"
        )
        ax_f.plot(sub_med.index, sub_med.values, color="#555", linewidth=0.9,
                  linestyle="--", label=med_label, zorder=3)
        ax_f.axhline(0, color="#666", linestyle="-", linewidth=0.8)
        ax_f.set_ylim(ylo, yhi)

        viz._style_ax(ax_f, yaxis_title=feat_col.upper())
        viz._format_dates(ax_f, start, end)
        viz._legend(ax_f)

        fig.subplots_adjust(hspace=0.08)

    viz._make_time_nav(
        pd.DataFrame(index=ou_z.dropna().index),
        _render_conditions,
        title=f"signal conditions  ·  {sfx}",
        nrows=2,
        height_ratios=[3, 2],
    )

    # ── Chart 2: 10y yield with entry / exit markers ────────────────────────
    def _render_yield(fig, ax, start, end):
        sub_y = y_10.loc[start:end].dropna()
        ax.plot(
            sub_y.index,
            sub_y.values,
            color="#2C3E50",
            linewidth=1.5,
            label="10y",
            zorder=3,
        )

        if not trades.empty:
            sub_t = trades[(trades["entry"] >= start) & (trades["entry"] <= end)]

            def _y_at(ds):
                return [float(y_10.loc[d]) if d in y_10.index else np.nan for d in ds]

            buys = sub_t[sub_t["side"] == "BUY"]
            sells = sub_t[sub_t["side"] == "SELL"]
            tip_offset = np.sqrt(180) / 2 / 72  # inches: half marker height in points
            if not buys.empty:
                # 'v' tip points down — shift center up so tip lands on data
                t_buy = ax.transData + mtransforms.ScaledTranslation(0, tip_offset, ax.figure.dpi_scale_trans)
                ax.scatter(
                    pd.DatetimeIndex(buys["entry"]),
                    _y_at(buys["entry"]),
                    marker="v",
                    color="#27AE60",
                    s=180,
                    zorder=6,
                    label="buy ▼",
                    edgecolors="white",
                    linewidths=0.8,
                    transform=t_buy,
                )
            if not sells.empty:
                # '^' tip points up — shift center down so tip lands on data
                t_sell = ax.transData + mtransforms.ScaledTranslation(0, -tip_offset, ax.figure.dpi_scale_trans)
                ax.scatter(
                    pd.DatetimeIndex(sells["entry"]),
                    _y_at(sells["entry"]),
                    marker="^",
                    color="#C0392B",
                    s=180,
                    zorder=6,
                    label="sell ▲",
                    edgecolors="white",
                    linewidths=0.8,
                    transform=t_sell,
                )

            # exit circles
            ax.scatter(
                pd.DatetimeIndex(sub_t["exit"]),
                _y_at(sub_t["exit"]),
                marker="o",
                color="#7F8C8D",
                s=60,
                zorder=5,
                label="exit",
                edgecolors="white",
                linewidths=0.8,
            )

            # entry→exit dotted connector per trade
            for _, t in sub_t.iterrows():
                if t["entry"] in y_10.index and t["exit"] in y_10.index:
                    c = "#27AE60" if t["side"] == "BUY" else "#C0392B"
                    ax.plot(
                        [t["entry"], t["exit"]],
                        [y_10.loc[t["entry"]], y_10.loc[t["exit"]]],
                        color=c,
                        linewidth=1.0,
                        linestyle=":",
                        alpha=0.5,
                        zorder=4,
                    )

        viz._style_ax(ax, yaxis_title="yield, bps")
        viz._format_dates(ax, start, end)
        viz._legend(ax)

    viz._make_time_nav(
        pd.DataFrame(index=y_10.dropna().index),
        _render_yield,
        title=f"10y yield  ·  entries & exits  ·  {sfx}",
    )

    # ── Chart 3: cumulative PnL + per-trade bars ────────────────────────────
    def _render_pnl(fig, ax, start, end):
        if trades.empty:
            return
        all_t = trades.sort_values("exit").copy()
        all_t["cum_pnl_bps"] = all_t["pnl_bps"].cumsum()

        prior = all_t[all_t["exit"] < start]
        baseline = float(prior["cum_pnl_bps"].iloc[-1]) if not prior.empty else 0.0
        window_t = all_t[(all_t["exit"] >= start) & (all_t["exit"] <= end)].copy()
        cum = pd.concat(
            [
                pd.Series([baseline], index=pd.DatetimeIndex([start])),
                window_t.set_index("exit")["cum_pnl_bps"],
                pd.Series(
                    [float(window_t["cum_pnl_bps"].iloc[-1]) if not window_t.empty else baseline],
                    index=pd.DatetimeIndex([end]),
                ),
            ]
        )
        ax.step(
            cum.index,
            cum.values,
            where="post",
            color="#2980B9",
            linewidth=2.0,
            label="cum PnL",
            zorder=3,
        )
        ax.axhline(0, color="#666", linestyle="-", linewidth=0.8)
        ax.fill_between(
            cum.index,
            0,
            cum.values,
            where=(cum.values >= 0),
            color="#27AE60",
            alpha=0.12,
            step="post",
        )
        ax.fill_between(
            cum.index,
            0,
            cum.values,
            where=(cum.values < 0),
            color="#C0392B",
            alpha=0.12,
            step="post",
        )

        vis_t = window_t
        if not vis_t.empty:
            ax2 = ax.twinx()
            bc = ["#27AE60" if v >= 0 else "#C0392B" for v in vis_t["pnl_bps"]]
            ax2.bar(
                pd.DatetimeIndex(vis_t["exit"]),
                vis_t["pnl_bps"].values,
                color=bc,
                width=3,
                alpha=0.4,
                zorder=2,
                label="trade PnL",
            )
            ax2.axhline(0, color="#666", linestyle="-", linewidth=0.5, alpha=0.5)
            ax2.grid(False)
            ax2.yaxis.tick_left()
            ax2.yaxis.set_label_position("left")
            ax2.set_ylabel("TRADE PNL, BPS", fontsize=8)
            for sp in ("top", "right", "bottom"):
                ax2.spines[sp].set_visible(False)
            ax2.tick_params(axis="y", labelsize=8)

        viz._style_ax(ax, yaxis_title="cum pnl, bps")
        viz._format_dates(ax, start, end)
        ax.set_xlim(start, end)
        viz._legend(ax)

    viz._make_time_nav(
        pd.DataFrame(index=y_10.dropna().index),
        _render_pnl,
        title=f"PnL  ·  {sfx}",
    )

    # trade table
    if not trades.empty:
        metrics = engine_metrics_frame(bt_result)
        display(metrics.round(3))
        viz.table(metrics.round(3), title=f"engine backtest metrics  ·  {sfx}")
        ts = time_stop_ser.copy()
        ts.index = pd.to_datetime(ts.index)
        resid_ser = model["resid"].copy()
        resid_ser.index = pd.to_datetime(resid_ser.index)
        ou_mean_ser = model["resid"].rolling(
            ou_lookback,
            min_periods=ou_lookback // 4,
        ).mean()
        ou_mean_ser.index = pd.to_datetime(ou_mean_ser.index)
        display_trades = (
            trades.drop(columns=["direction", "pnl_dollar"], errors="ignore")
            .rename(columns={"entry_signal": "entry_z", "exit_signal": "exit_z"})
            .copy()
        )
        entry_dates = pd.to_datetime(display_trades["entry"])
        exit_dates = pd.to_datetime(display_trades["exit"])
        display_trades["entry_resid_bps"] = entry_dates.map(resid_ser)
        display_trades["exit_resid_bps"] = exit_dates.map(resid_ser)
        display_trades["entry_ou_mean_bps"] = entry_dates.map(ou_mean_ser)
        display_trades["target_resid_bps"] = (
            display_trades["entry_ou_mean_bps"]
            + (display_trades["entry_resid_bps"] - display_trades["entry_ou_mean_bps"])
            * (1.0 - residual_target_frac)
        )
        display_trades["target_move_bps"] = (
            display_trades["target_resid_bps"] - display_trades["entry_resid_bps"]
        )
        display_trades["half_life_days"] = entry_dates.map(ts).round().astype("Int64")
        display_trades["pnl_bps"] = display_trades["pnl_bps"].round(1)
        display_trades["entry_z"] = display_trades["entry_z"].round(3)
        display_trades["exit_z"] = display_trades["exit_z"].round(3)
        residual_cols = [
            "entry_resid_bps",
            "exit_resid_bps",
            "entry_ou_mean_bps",
            "target_resid_bps",
            "target_move_bps",
        ]
        display_trades[residual_cols] = display_trades[residual_cols].round(1)
        preferred_cols = [
            "trade_name",
            "entry",
            "exit",
            "side",
            "entry_10y",
            "exit_10y",
            "entry_z",
            "exit_z",
            "entry_resid_bps",
            "exit_resid_bps",
            "entry_ou_mean_bps",
            "target_resid_bps",
            "target_move_bps",
            "half_life_days",
            "bars_held",
            "exit_reason",
            "pnl_bps",
        ]
        display_trades = display_trades[
            [c for c in preferred_cols if c in display_trades.columns]
            + [c for c in display_trades.columns if c not in preferred_cols]
        ]
        display_trades = display_trades.sort_values("exit", ascending=False)
        display(display_trades)
        viz.table(display_trades, title=f"trade log  ·  {sfx}")

    return bt_result


# ─── main ──────────────────────────────────────────────────────────────────


def main() -> dict:
    viz = Viz(backend="plotly")

    df = load_basket()
    print(f"basket: {len(df)} obs · {df.index.min().date()} → {df.index.max().date()}")
    print(f"        cols: {list(df.columns)}")

    # ── parameters ────────────────────────────────────────────────────────────
    reg_lookback = 60    # rolling OLS window for single-factor models
    ou_lookback  = 252   # OU estimation window (mean, half-life, time stop)
    entry_z      = 1.5   # |ou_zscore| threshold to enter a trade
    # Exit: residual reverts halfway to OU mean (half_drift_residual),
    #       OR ou_lookback-derived half-life bars elapse — whichever first.
    directional_mom_filter = True  # z-score momentum filters use sign-aligned regimes
    drilldown_override = None
    # drilldown_override = {
    #     "anchor": "30y",
    #     "horizon": 20,
    #     "filter_col": "beta",
    #     "filter_rule": None,  # None means choose the better beta_low/beta_high row.
    # }
    # drilldown_override = {
    #     "anchor": "sofr10",
    #     "horizon": 5,
    #     "filter_col": "zscore_streak",
    #     "filter_rule": None,
    # }
    robustness_config = {
        "anchor": "30y",
        "horizon": 20,
        "filter_col": "beta",
        "filter_cutoffs": {
            "low": (0.25, 0.33, 0.50),
            "high": (0.50, 0.67, 0.75),
        },
        "reg_lookbacks": (40, 60, 90, 126),
        "ou_lookbacks": (126, 252, 504),
        "entry_zs": (1.5, 2.0, 2.5),
        "residual_target_fracs": (0.25, 0.50, 0.75),
    }

    # Single-factor 10y models: each candidate gets its own rolling regression.
    ten_single, ten_single_models = single_factor_10y_table(
        df,
        target="10y",
        lookback=reg_lookback,
        ou_lookback=ou_lookback,
    )
    print("\n--- 10y single-factor rolling regression residual OU table ---")
    display_cols = [
        "anchor",
        "date",
        "actual_10y",
        "model_10y",
        "residual",
        "ou_mean_resid",
        "ou_fair_10y",
        "ou_zscore",
        "half_life_d",
        "beta",
        "r2",
    ]
    if ten_single.empty:
        print("No single-factor models had enough overlapping data.")
    else:
        display(ten_single[display_cols].round(3))
        viz.table(
            ten_single[display_cols].round(3),
            title="10y single-factor rolling regression residual OU table",
        )

    bt = ou_signal_forward_diagnostics(ten_single_models)
    if not ten_single.empty and not bt.empty:
        anchor_order = ten_single["anchor"].tolist()
        bt = bt.set_index("anchor").reindex(anchor_order).reset_index()
        display(bt.round(3))
        viz.table(
            bt.round(3),
            title="10y single-factor OU signal diagnostics  (IC + hit rate + sharpe)",
        )

    # Filtered sharpe: best OOS feature filter per anchor across all horizons.
    print("\n--- filtered sharpe summary (best feature×horizon per anchor) ---")
    filt = filtered_sharpe_summary(
        ten_single_models,
        df["10y"],
        by_direction=directional_mom_filter,
    )
    if not filt.empty:
        display(filt.round(3))
        viz.table(filt.round(3), title="best regime-filtered sharpe per anchor")

    # Robustness grid: hold the idea fixed, then vary only implementation parameters.
    robust = parameter_robustness(df, **robustness_config)
    if not robust.empty:
        robust_display = robust.copy()
        numeric_cols = robust_display.select_dtypes(include=[np.number]).columns
        robust_display[numeric_cols] = robust_display[numeric_cols].round(3)
        display(robust_display)
        viz.table(
            robust_display,
            title=(
                "engine robustness grid"
                f"  ·  {robustness_config['anchor']}"
                f" / {robustness_config['filter_col']}"
                f" / h={robustness_config['horizon']}d"
            ),
            max_rows=80,
        )

    # Deep-dive charts for the selected filtered signal.
    best_bt = None
    if not filt.empty:
        if drilldown_override is None:
            drill_row = filt.iloc[0]
        else:
            drill_row = select_filtered_candidate(filt, **drilldown_override)
            if drill_row is None:
                print(f"No filtered candidate matched override: {drilldown_override}")
                drill_row = filt.iloc[0]
        print(
            "\n--- engine drilldown candidate ---\n"
            f"{drill_row['anchor']} · h={int(drill_row['horizon'])}d · "
            f"{drill_row['best_feature']} · sharpe={drill_row['sharpe_filtered']:.3f}"
        )
        best_bt = _best_signal_drill(
            drill_row,
            ten_single_models,
            df,
            viz,
            entry_z=entry_z,
            ou_lookback=ou_lookback,
            directional_mom_filter=directional_mom_filter,
        )

    return {
        "df": df,
        "ten_single": ten_single,
        "ten_single_models": ten_single_models,
        "bt": bt,
        "filt": filt,
        "robust": robust,
        "best_bt": best_bt,
    }


if __name__ == "__main__":
    state = main()
