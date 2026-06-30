"""Drill to the core — recursive regression diagnostics for rates research.

Pattern: pick (x, y). Rolling-regress y on x. Inspect the residual. Peel a
known factor z out of it. Repeat. The story of the analysis lives in main();
the helpers above are reusable building blocks.
"""

from __future__ import annotations

import argparse
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
from book.duration.signal_context import (
    build_signal_features,
    conditional_ic_table,
    filtered_sharpe_summary,
    oos_edge_summary,
    oos_edge_test_fast,
)
from utils.helpers import timed
from utils.rates import linear_5y5y_forward
from utils.viz import Viz

warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", 200)


# ─── config ────────────────────────────────────────────────────────────────

DB_DSN = os.getenv("DB_DSN", "postgresql://benjils:snickers@raptor:5432/markets?connect_timeout=10")
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


_BASKET_CACHE = _HERE / ".basket_cache.parquet"


def load_basket(tickers: dict[str, str] = TICKERS, start: str = START, cache: bool = False) -> pd.DataFrame:
    """Pull `tickers` from md.index_eod, pivot wide. Yields scaled to bps."""
    if cache and _BASKET_CACHE.exists():
        print(f"  (cache hit: {_BASKET_CACHE})")
        return pd.read_parquet(_BASKET_CACHE)
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
    wide = wide.dropna(how="all")
    if cache:
        wide.to_parquet(_BASKET_CACHE)
        print(f"  (cached to {_BASKET_CACHE})")
    return wide


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
    signal_col: str = "ou_zscore",
    entry_threshold: float = 1.0,
) -> pd.Series:
    """Gate entries with the OOS filter while preserving signal sign.

    The shared SignalConfig uses thresholds for both entries and exits. When
    the filter is off, cap extreme signal values just inside the entry bands so new
    positions cannot open, but mean-crossing exits still fire.
    """
    z = model[signal_col].reindex(model.index).astype(float).copy()
    active = filter_mask.reindex(model.index, fill_value=False).astype(bool)
    eps = 1e-6
    z.loc[~active & (z >= entry_threshold)] = entry_threshold - eps
    z.loc[~active & (z <= -entry_threshold)] = -entry_threshold + eps
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
    entry_signal_col: str = "ou_zscore",
    entry_threshold: float = 1.0,
    ou_lookback: int = 252,
    residual_target_frac: float = 0.5,
    stop_loss_bps: float | None = None,
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
    signal = _gated_entry_signal(
        model,
        filter_mask,
        signal_col=entry_signal_col,
        entry_threshold=entry_threshold,
    )
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
                entry_long=-entry_threshold,
                entry_short=entry_threshold,
            exit_long=None,  # signal exits disabled — half_drift_residual owns this
            exit_short=None,
            stop_loss_bps=stop_loss_bps,
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
    entry_signal_col: str,
    horizon: int,
    filter_cutoffs: dict[str, tuple[float, ...]] | None = None,
    reg_lookbacks: tuple[int, ...] = (40, 60, 90, 126),
    ou_lookbacks: tuple[int, ...] = (126, 252, 504),
    entry_thresholds: tuple[float, ...] = (1.5, 2.0, 2.5),
    residual_target_fracs: tuple[float, ...] = (0.25, 0.5, 0.75),
    stop_loss_bpss: tuple[float | None, ...] = (None,),
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
                    signal_col=entry_signal_col,
                    horizon=horizon,
                    train_window=train_window,
                    filter_rule=rule,
                    filter_quantile=q,
                )
                mask = result["filter_mask"]
                fwd = result["filtered"]
                for entry_threshold in entry_thresholds:
                    for target_frac in residual_target_fracs:
                        for stop_loss in stop_loss_bpss:
                            bt_result, _, _ = run_engine_backtest(
                                model,
                                mask,
                                entry_signal_col=entry_signal_col,
                                entry_threshold=entry_threshold,
                                ou_lookback=ou_lb,
                                residual_target_frac=target_frac,
                                stop_loss_bps=stop_loss,
                            )
                            metrics = bt_result.summary()
                            rows.append(
                                {
                                    "anchor": anchor,
                                    "horizon": horizon,
                                    "filter_col": filter_col,
                                    "entry_signal_col": entry_signal_col,
                                    "filter_rule": rule,
                                    "filter_quantile": q,
                                    "reg_lookback": reg_lb,
                                    "ou_lookback": ou_lb,
                                    "entry_threshold": entry_threshold,
                                    "residual_target_frac": target_frac,
                                    "stop_loss_bps": stop_loss if stop_loss is not None else np.nan,
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


def _default_param_value(values, fallback):
    return values[0] if values else fallback


def _selected_or_default(raw, values, dtype, default):
    val = _control_value(raw, dtype)
    if val is None:
        return default
    return val


def parameter_robustness_slices(
    df: pd.DataFrame,
    *,
    anchor: str,
    filter_col: str,
    entry_signal_col: str,
    horizon: int,
    selected: dict[str, str],
    filter_cutoffs: dict[str, tuple[float, ...]],
    reg_lookbacks: tuple[int, ...],
    ou_lookbacks: tuple[int, ...],
    entry_thresholds: tuple[float, ...],
    residual_target_fracs: tuple[float, ...],
    stop_loss_bpss: tuple[float | None, ...],
    train_window: int = 504,
    progress_fn=None,
) -> pd.DataFrame:
    """Evaluate focused 2D slices instead of a full Cartesian hypercube."""
    filter_specs = [
        (rule, q)
        for rule in ("high", "low")
        for q in filter_cutoffs.get(rule, ())
    ]
    default_filter = filter_specs[0]
    default_reg = _default_param_value(reg_lookbacks, 60)
    default_ou = _default_param_value(ou_lookbacks, 252)
    default_entry = _default_param_value(entry_thresholds, 5.0)
    default_target = 0.5 if 0.5 in residual_target_fracs else _default_param_value(residual_target_fracs, 0.5)
    default_stop = None

    fixed = {
        "reg_lookback": _selected_or_default(selected.get("reg_lookback"), reg_lookbacks, int, default_reg),
        "ou_lookback": _selected_or_default(selected.get("ou_lookback"), ou_lookbacks, int, default_ou),
        "filter_rule": _selected_or_default(selected.get("filter_rule"), (), str, default_filter[0]),
        "filter_quantile": _selected_or_default(selected.get("filter_quantile"), (), float, default_filter[1]),
        "entry_threshold": _selected_or_default(selected.get("entry_threshold"), entry_thresholds, float, default_entry),
        "residual_target_frac": _selected_or_default(selected.get("residual_target_frac"), residual_target_fracs, float, default_target),
        "stop_loss_bps": _selected_or_default(selected.get("stop_loss_bps"), stop_loss_bpss, float, default_stop),
    }
    if pd.isna(fixed["stop_loss_bps"]):
        fixed["stop_loss_bps"] = None

    combos = {}

    def add_combo(source, **overrides):
        combo = {**fixed, **overrides}
        key = (
            combo["reg_lookback"],
            combo["ou_lookback"],
            combo["filter_rule"],
            float(combo["filter_quantile"]),
            float(combo["entry_threshold"]),
            float(combo["residual_target_frac"]),
            None if combo["stop_loss_bps"] is None or pd.isna(combo["stop_loss_bps"]) else float(combo["stop_loss_bps"]),
        )
        combo["slice"] = source
        combos.setdefault(key, combo)

    for rule, q in filter_specs:
        for entry in entry_thresholds:
            add_combo(
                "entry_x_filter",
                filter_rule=rule,
                filter_quantile=q,
                entry_threshold=entry,
            )
    for target in residual_target_fracs:
        for stop in stop_loss_bpss:
            add_combo("target_x_stop", residual_target_frac=target, stop_loss_bps=stop)
    for reg in reg_lookbacks:
        for ou in ou_lookbacks:
            add_combo("reg_x_ou", reg_lookback=reg, ou_lookback=ou)
    add_combo("selected")

    model_cache: dict[tuple[int, int], tuple[pd.DataFrame, pd.DataFrame]] = {}
    filter_cache: dict[tuple[int, int, str, float], tuple[pd.Series, dict]] = {}
    rows = []
    total = len(combos)
    if progress_fn is not None:
        progress_fn(done=0, total=total, message=f"Starting {total:,} focused engine backtests...")
    for i, combo in enumerate(combos.values(), start=1):
        reg_lb = int(combo["reg_lookback"])
        ou_lb = int(combo["ou_lookback"])
        model_key = (reg_lb, ou_lb)
        if model_key not in model_cache:
            model = single_factor_model(df, anchor=anchor, lookback=reg_lb, ou_lookback=ou_lb)
            feats = build_signal_features(model) if not model.empty else pd.DataFrame()
            model_cache[model_key] = (model, feats)
        model, feats = model_cache[model_key]
        if model.empty or feats.empty or filter_col not in feats.columns:
            continue

        rule = combo["filter_rule"]
        q = float(combo["filter_quantile"])
        filter_key = (reg_lb, ou_lb, rule, q)
        if filter_key not in filter_cache:
            result = oos_edge_test_fast(
                feats,
                df["10y"],
                filter_col,
                signal_col=entry_signal_col,
                horizon=horizon,
                train_window=train_window,
                filter_rule=rule,
                filter_quantile=q,
            )
            filter_cache[filter_key] = (result["filter_mask"], result)
        mask, result = filter_cache[filter_key]
        bt_result, _, _ = run_engine_backtest(
            model,
            mask,
            entry_signal_col=entry_signal_col,
            entry_threshold=float(combo["entry_threshold"]),
            ou_lookback=ou_lb,
            residual_target_frac=float(combo["residual_target_frac"]),
            stop_loss_bps=combo["stop_loss_bps"],
        )
        metrics = bt_result.summary()
        fwd = result["filtered"]
        rows.append({
            "slice": combo["slice"],
            "anchor": anchor,
            "horizon": horizon,
            "filter_col": filter_col,
            "entry_signal_col": entry_signal_col,
            "filter_rule": rule,
            "filter_quantile": q,
            "reg_lookback": reg_lb,
            "ou_lookback": ou_lb,
            "entry_threshold": float(combo["entry_threshold"]),
            "residual_target_frac": float(combo["residual_target_frac"]),
            "stop_loss_bps": combo["stop_loss_bps"] if combo["stop_loss_bps"] is not None else np.nan,
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
        })
        if progress_fn is not None:
            progress_fn(
                done=i,
                total=total,
                message=(
                    f"Running focused robustness slices: {i:,} / {total:,} "
                    f"({combo['slice']})"
                ),
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["sharpe", "profit_factor", "n_trades"], ascending=False).reset_index(drop=True)


def _control_value(value: str, dtype):
    """Parse dropdown values; '__best__' means leave that parameter unconstrained."""
    if value in (None, "__best__"):
        return None
    if value == "__none__":
        return np.nan
    if dtype is int:
        return int(float(value))
    if dtype is float:
        return float(value)
    return str(value)


def _select_robust_row(robust: pd.DataFrame, controls: dict[str, object], metric: str = "sharpe") -> pd.Series:
    """Pick the best row after applying explicit dropdown constraints.

    Prefers the 'selected' slice (the exact param intersection) when it
    matches all constraints; otherwise returns the best-by-metric match.
    """
    specs = {
        "reg_lookback": int,
        "ou_lookback": int,
        "filter_rule": str,
        "filter_quantile": float,
        "entry_threshold": float,
        "residual_target_frac": float,
        "stop_loss_bps": float,
    }

    def _filter(df):
        out = df.copy()
        for col, dtype in specs.items():
            val = _control_value(controls.get(col), dtype)
            if val is None:
                continue
            if dtype is float and pd.isna(val):
                out = out[out[col].isna()]
            elif dtype is float:
                out = out[np.isclose(out[col].astype(float), val)]
            else:
                out = out[out[col] == val]
        return out

    matched = _filter(robust)
    if matched.empty:
        matched = robust

    sort_col = metric if metric in matched.columns else "sharpe"
    # Prefer the "selected" slice — it's the exact param intersection point
    selected_slice = matched[matched["slice"] == "selected"] if "slice" in matched.columns else pd.DataFrame()
    if not selected_slice.empty:
        return selected_slice.sort_values(sort_col, ascending=False).iloc[0]
    return matched.sort_values(sort_col, ascending=False).iloc[0]


def _filter_threshold_label(row: pd.Series) -> str:
    side = "above" if row["filter_rule"] == "high" else "below"
    return f"{side} q{int(round(float(row['filter_quantile']) * 100))}"


def _filter_threshold_order(labels: pd.Index) -> list[str]:
    def _key(label: str) -> tuple[int, int]:
        side, q = label.split(" q")
        qv = int(q)
        return (0, -qv) if side == "above" else (1, -qv)
    return sorted(labels, key=_key)


def _metric_heatmap(
    viz,
    data: pd.DataFrame,
    *,
    metric: str,
    x_col: str,
    y_col: str,
    title: str,
) -> None:
    """Render a small static robustness heatmap."""
    if data.empty or metric not in data:
        return
    hm = (
        data.groupby([y_col, x_col], dropna=False)[metric]
        .mean()
        .unstack(x_col)
        .sort_index()
    )
    if hm.empty:
        return

    def _render(fig, ax):
        values = hm.to_numpy(dtype=float)
        im = ax.imshow(values, aspect="auto", cmap="RdYlGn")
        ax.set_xticks(np.arange(len(hm.columns)))
        ax.set_xticklabels([str(c) for c in hm.columns], rotation=0)
        ax.set_yticks(np.arange(len(hm.index)))
        ax.set_yticklabels([str(i) for i in hm.index])
        ax.set_xlabel(x_col.replace("_", " ").upper(), fontsize=9)
        ax.set_ylabel(y_col.replace("_", " ").upper(), fontsize=9)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                v = values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
        cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        cbar.ax.set_ylabel(metric.replace("_", " ").upper(), fontsize=8)
        ax.tick_params(axis="both", labelsize=8)

    viz.static_chart(_render, title=title, fig_height=3.4)


def _metric_heatmap_panel(
    viz,
    specs: list[tuple[pd.DataFrame, str, str, str]],
    *,
    metric: str,
    title: str,
) -> None:
    """Render multiple robustness heatmaps side by side."""
    heatmaps = []
    for data, x_col, y_col, panel_title in specs:
        if data.empty or metric not in data:
            continue
        hm = (
            data.groupby([y_col, x_col], dropna=False)[metric]
            .mean()
            .unstack(x_col)
            .sort_index()
        )
        if hm.empty:
            continue
        if y_col == "filter_threshold":
            hm = hm.reindex(_filter_threshold_order(hm.index))
        if y_col == "stop_loss_label":
            hm = hm.reindex(sorted(
                hm.index,
                key=lambda x: -1 if str(x) == "none" else float(x),
            ))
        heatmaps.append((hm, x_col, y_col, panel_title))
    if not heatmaps:
        return

    def _render(fig, axes):
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        vals = np.concatenate([
            hm.to_numpy(dtype=float).ravel()
            for hm, *_ in heatmaps
        ])
        vals = vals[np.isfinite(vals)]
        vmax = max(abs(float(np.nanmin(vals))), abs(float(np.nanmax(vals)))) if len(vals) else 1.0
        vmin = -vmax if metric in {"sharpe", "total_pnl_bps", "fwd_sharpe_filtered"} else None
        cmap = "RdYlGn"
        last_im = None
        for ax, (hm, x_col, y_col, panel_title) in zip(axes, heatmaps):
            arr = hm.to_numpy(dtype=float)
            last_im = ax.imshow(arr, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(panel_title.upper(), fontsize=9, fontweight="bold", loc="left")
            ax.set_xticks(np.arange(len(hm.columns)))
            ax.set_xticklabels([str(c) for c in hm.columns], rotation=0, fontsize=8)
            ax.set_yticks(np.arange(len(hm.index)))
            ax.set_yticklabels([str(i) for i in hm.index], fontsize=8)
            ax.set_xlabel(x_col.replace("_", " ").upper(), fontsize=8)
            ax.set_ylabel(y_col.replace("_", " ").upper(), fontsize=8)
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    v = arr[i, j]
                    if np.isfinite(v):
                        ax.text(
                            j,
                            i,
                            f"{v:.2f}",
                            ha="center",
                            va="center",
                            fontsize=7,
                            color="#111",
                        )
        for ax in axes[len(heatmaps):]:
            ax.axis("off")
        if last_im is not None:
            cbar = fig.colorbar(last_im, ax=axes[:len(heatmaps)], fraction=0.02, pad=0.015)
            cbar.ax.set_ylabel(metric.replace("_", " ").upper(), fontsize=8)

    viz.static_chart(_render, title=title, fig_height=4.4, ncols=len(heatmaps))


def _residual_distribution(viz, model: pd.DataFrame, *, title: str) -> None:
    resid = model["resid"].dropna()
    if resid.empty:
        return

    def _render(fig, ax):
        ax.hist(resid, bins=50, color="#2980B9", alpha=0.72, edgecolor="white")
        for x, label, color in [
            (resid.mean(), "mean", "#333"),
            (resid.quantile(0.10), "q10", "#C0392B"),
            (resid.quantile(0.90), "q90", "#27AE60"),
        ]:
            ax.axvline(x, color=color, linestyle="--", linewidth=1.0, label=label)
        ax.set_xlabel("RESIDUAL, BPS", fontsize=9)
        ax.set_ylabel("OBS", fontsize=9)
        ax.legend(frameon=False, fontsize=8)

    viz.static_chart(_render, title=title, fig_height=3.2)


def _best_signal_drill(
    best_row: pd.Series,
    models: dict[str, pd.DataFrame],
    df: pd.DataFrame,
    viz,
    *,
    train_window: int = 504,
    entry_signal_col: str = "ou_zscore",
    entry_threshold: float = 1.0,
    ou_lookback: int = 252,
    residual_target_frac: float = 0.5,
    stop_loss_bps: float | None = None,
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
        signal_col=entry_signal_col,
        horizon=h,
        train_window=train_window,
        by_direction=directional_mom_filter,
        filter_rule=filter_rule,
        filter_quantile=float(filter_quantile),
    )

    mask = result["filter_mask"]
    filter_med = result["rolling_med"].reindex(model.index)
    feat_ser = feats[feat_col].reindex(model.index)
    sfx = f"10y | {anchor}  ·  filter: {filter_name}  ·  h={h}d"

    bt_result, _, time_stop_ser = run_engine_backtest(
        model, mask,
        entry_signal_col=entry_signal_col,
        entry_threshold=entry_threshold,
        ou_lookback=ou_lookback,
        residual_target_frac=residual_target_frac,
        stop_loss_bps=stop_loss_bps,
        mom_ser=feat_ser,
        entry_filter_fn=None,
    )
    trades = engine_trades_frame(bt_result)
    if not trades.empty:
        metrics = engine_metrics_frame(bt_result)
        display(metrics.round(3))
        viz.table(metrics.round(3), title=f"engine backtest metrics  ·  {sfx}")
    entry_signal = model[entry_signal_col].copy()
    y_10 = model["y"].copy()
    entry_signal.index = pd.to_datetime(entry_signal.index)
    y_10.index = pd.to_datetime(y_10.index)
    feat_ser.index = pd.to_datetime(feat_ser.index)
    filter_med.index = pd.to_datetime(filter_med.index)
    mask.index = pd.to_datetime(mask.index)

    # ── Chart 1: signal conditions — two stacked panels, shared x-axis ────────
    # Top: OU z-score + entry markers.  Bottom: selected filter feature.
    def _render_conditions(fig, axes, start, end):
        ax_z, ax_f = axes

        sub_z = entry_signal.loc[start:end]
        sub_f = feat_ser.loc[start:end]
        sub_med = filter_med.loc[start:end]
        sub_m = mask.reindex(sub_z.index, fill_value=False)

        # ── top: OU z-score ────────────────────────────────────────────────
        yz = max(float(sub_z.abs().max(skipna=True)) * 1.2, entry_threshold * 1.2)
        buy_on = (sub_m & (sub_z >= entry_threshold)).values.astype(bool)
        sell_on = (sub_m & (sub_z <= -entry_threshold)).values.astype(bool)

        ax_z.fill_between(sub_z.index, -yz, yz, where=buy_on,
                          color="#27AE60", alpha=0.13, zorder=1, label="BUY entry allowed")
        ax_z.fill_between(sub_z.index, -yz, yz, where=sell_on,
                          color="#C0392B", alpha=0.13, zorder=1, label="SELL entry allowed")
        signal_label = "OU z-score" if entry_signal_col == "ou_zscore" else "raw residual"
        ax_z.plot(sub_z.index, sub_z.values, color="#2980B9", linewidth=1.5,
                  label=signal_label, zorder=3)
        ax_z.axhline(entry_threshold, color="#27AE60", linestyle="--", linewidth=0.9, alpha=0.8,
                     label=f"+{entry_threshold:.1f} entry")
        ax_z.axhline(-entry_threshold, color="#C0392B", linestyle="--", linewidth=0.9, alpha=0.8,
                     label=f"−{entry_threshold:.1f} entry")
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

        yaxis_title = "z-score" if entry_signal_col == "ou_zscore" else "resid, bps"
        viz._style_ax(ax_z, yaxis_title=yaxis_title)
        viz._format_dates(ax_z, start, end)
        ax_z.tick_params(axis="x", which="both", labelbottom=True)
        ax_z.legend(
            loc="upper left",
            bbox_to_anchor=(0.0, -0.14),
            ncol=3,
            fontsize=8,
            frameon=False,
            borderaxespad=0,
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

        fig.subplots_adjust(hspace=0.36)

    viz._make_time_nav(
        pd.DataFrame(index=entry_signal.dropna().index),
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

    # ── Chart 3: cumulative PnL ─────────────────────────────────────────────
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
            .rename(
                columns={
                    "entry_signal": f"entry_{entry_signal_col}",
                    "exit_signal": f"exit_{entry_signal_col}",
                }
            )
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
        entry_signal_name = f"entry_{entry_signal_col}"
        exit_signal_name = f"exit_{entry_signal_col}"
        display_trades[entry_signal_name] = display_trades[entry_signal_name].round(3)
        display_trades[exit_signal_name] = display_trades[exit_signal_name].round(3)
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
            entry_signal_name,
            exit_signal_name,
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
        viz.table(
            display_trades,
            title=f"trade log  ·  {sfx}",
            max_rows=len(display_trades),
            max_height="420px",
        )

    return bt_result


# ─── main ──────────────────────────────────────────────────────────────────


def main(cache: bool = False) -> dict:
    viz = Viz(backend="plotly")

    with timed("loading basket"):
        df = load_basket(cache=cache)
    print(f"  {len(df)} obs · {df.index.min().date()} → {df.index.max().date()}")
    print(f"  cols: {list(df.columns)}")

    # ── parameters ────────────────────────────────────────────────────────────
    reg_lookback = 60    # rolling OLS window for single-factor models
    ou_lookback  = 252   # OU estimation window (mean, half-life, time stop)
    entry_signal_col = "resid"  # "resid" trades raw residual bps; "ou_zscore" trades OU z.
    # Exit: residual reverts halfway to OU mean (half_drift_residual),
    #       OR ou_lookback-derived half-life bars elapse — whichever first.
    directional_mom_filter = True  # z-score momentum filters use sign-aligned regimes

    # Robustness sweep config — `anchor` is the *initial* dropdown value; the
    # explorer rebuilds the grid (and the drilldown charts) when a new anchor
    # is picked. filter_col and horizon are held fixed here.
    robustness_config = {
        "initial_anchor": "sofr5",
        "horizon": 5,
        "filter_col": "resid_vol_60d",
        "entry_signal_col": entry_signal_col,
        "filter_cutoffs": {
            "low": tuple(np.round(np.arange(0.5, 0.0, -0.1), 1)),
            "high": tuple(np.round(np.arange(0.9, 0.4, -0.1), 1)),
        },
        "reg_lookbacks": (40, 60, 90, 126),
        "ou_lookbacks": (126, 252, 504),
        "entry_thresholds": tuple(np.round(np.arange(2.5, 20.1, 2.5), 1)),
        "residual_target_fracs": tuple(np.round(np.arange(0.1, 1.0, 0.1), 1)),
        "stop_loss_bpss": (None, 10.0, 20.0, 30.0, 40.0, 50.0),
    }
    # Single-factor 10y models: each candidate gets its own rolling regression.
    with timed("fitting single-factor models"):
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

    # Filtered sharpe: best OOS feature filter per anchor across all horizons.
    print("\n--- filtered sharpe summary (best feature×horizon per anchor) ---")
    filt = filtered_sharpe_summary(
        ten_single_models,
        df["10y"],
        signal_col=entry_signal_col,
        by_direction=directional_mom_filter,
    )
    if not filt.empty:
        display(filt.round(3))
        viz.table(filt.round(3), title="best regime-filtered sharpe per anchor")

    # ── interactive robustness explorer ───────────────────────────────────────
    # Anchor dropdown + REF button. Each refresh recomputes the engine
    # robustness grid for that anchor and re-runs the deep-dive charts using
    # the top-sharpe parameter row from the grid.
    sweep_kwargs = {k: v for k, v in robustness_config.items()
                    if k not in ("initial_anchor",)}
    anchors = sorted(ten_single_models.keys())
    initial_anchor = robustness_config["initial_anchor"]
    if initial_anchor not in anchors and anchors:
        initial_anchor = anchors[0]
    n_filter_specs = sum(len(v) for v in robustness_config["filter_cutoffs"].values())
    n_engine_runs = (
        n_filter_specs * len(robustness_config["entry_thresholds"])
        + len(robustness_config["residual_target_fracs"]) * len(robustness_config["stop_loss_bpss"])
        + len(robustness_config["reg_lookbacks"]) * len(robustness_config["ou_lookbacks"])
        + 1
    )
    print(
        "\n--- robustness explorer ---\n"
        f"initial render is lazy; press REF to run ~{n_engine_runs:,} "
        "focused engine backtests for the selected anchor"
    )

    def build_for_anchor(
        scoped_viz,
        *,
        anchor: str,
        metric: str = "sharpe",
        reg_lookback: str = "__best__",
        ou_lookback: str = "__best__",
        filter_rule: str = "__best__",
        filter_quantile: str = "__best__",
        entry_threshold: str = "__best__",
        residual_target_frac: str = "__best__",
        stop_loss_bps: str = "__best__",
    ) -> None:
        controls = {
            "reg_lookback": reg_lookback,
            "ou_lookback": ou_lookback,
            "filter_rule": filter_rule,
            "filter_quantile": filter_quantile,
            "entry_threshold": entry_threshold,
            "residual_target_frac": residual_target_frac,
            "stop_loss_bps": stop_loss_bps,
        }
        robust = parameter_robustness_slices(
            df,
            anchor=anchor,
            selected=controls,
            progress_fn=scoped_viz.progress,
            **sweep_kwargs,
        )
        if robust.empty:
            scoped_viz.table(
                pd.DataFrame({"msg": [f"no robustness data for {anchor}"]}),
                title=f"robustness grid · {anchor}",
            )
            return
        scoped_viz.progress(message="Rendering robustness table and heatmaps...")

        metric = metric if metric in robust.columns else "sharpe"
        best = _select_robust_row(robust, controls, metric=metric)

        # Show the params that will drive the drill-down charts.
        param_cols = ["reg_lookback", "ou_lookback", "filter_rule", "filter_quantile",
                      "entry_threshold", "residual_target_frac", "stop_loss_bps", metric]
        param_cols = [c for c in param_cols if c in best.index]
        scoped_viz.table(
            best[param_cols].to_frame().T.round(3),
            title="selected params  ·  driving the drill-down charts below",
        )

        robust_display = robust.sort_values(metric, ascending=False).copy()
        numeric_cols = robust_display.select_dtypes(include=[np.number]).columns
        robust_display[numeric_cols] = robust_display[numeric_cols].round(3)
        robust_display = robust_display.rename(columns={metric: f"{metric} ↑"})
        scoped_viz.table(
            robust_display,
            title=(
                "engine robustness grid"
                f"  ·  {anchor} / {sweep_kwargs['filter_col']}"
                f" / h={sweep_kwargs['horizon']}d"
            ),
            max_rows=len(robust_display),
            max_height="360px",
        )

        heat = robust.copy()
        heat["filter_threshold"] = heat.apply(_filter_threshold_label, axis=1)
        heat["stop_loss_label"] = heat["stop_loss_bps"].map(
            lambda x: "none" if pd.isna(x) else f"{float(x):.0f}"
        )
        fixed_1 = heat[heat["slice"].isin(["entry_x_filter", "selected"])].copy()
        for col, dtype in {
            "reg_lookback": int,
            "ou_lookback": int,
            "residual_target_frac": float,
            "stop_loss_bps": float,
        }.items():
            val = _control_value(controls.get(col), dtype)
            if val is not None:
                fixed_1 = (
                    fixed_1[fixed_1[col].isna()]
                    if pd.isna(val)
                    else fixed_1[np.isclose(fixed_1[col].astype(float), float(val))]
                )

        fixed_2 = heat[heat["slice"].isin(["target_x_stop", "selected"])].copy()
        for col, dtype in {
            "reg_lookback": int,
            "filter_rule": str,
            "filter_quantile": float,
            "entry_threshold": float,
        }.items():
            val = _control_value(controls.get(col), dtype)
            if val is None:
                continue
            if dtype is str:
                fixed_2 = fixed_2[fixed_2[col] == val]
            else:
                fixed_2 = fixed_2[np.isclose(fixed_2[col].astype(float), float(val))]

        fixed_3 = heat[heat["slice"].isin(["reg_x_ou", "selected"])].copy()
        for col, dtype in {
            "filter_rule": str,
            "filter_quantile": float,
            "entry_threshold": float,
            "residual_target_frac": float,
            "stop_loss_bps": float,
        }.items():
            val = _control_value(controls.get(col), dtype)
            if val is None:
                continue
            if dtype is str:
                fixed_3 = fixed_3[fixed_3[col] == val]
            elif pd.isna(val):
                fixed_3 = fixed_3[fixed_3[col].isna()]
            else:
                fixed_3 = fixed_3[np.isclose(fixed_3[col].astype(float), float(val))]

        _metric_heatmap_panel(
            scoped_viz,
            metric=metric,
            title=f"{metric} robustness heatmaps",
            specs=[
                (
                    fixed_1,
                    "entry_threshold",
                    "filter_threshold",
                    f"entry residual vs {sweep_kwargs['filter_col']} threshold",
                ),
                (
                    fixed_2,
                    "residual_target_frac",
                    "stop_loss_label",
                    "target frac vs stop loss",
                ),
                (
                    fixed_3,
                    "reg_lookback",
                    "ou_lookback",
                    "reg lookback vs OU lookback",
                ),
            ],
        )

        reg_lb = int(best["reg_lookback"])
        ou_lb = int(best["ou_lookback"])
        model = single_factor_model(
            df, anchor=anchor, lookback=reg_lb, ou_lookback=ou_lb,
        )
        if model.empty:
            return
        _residual_distribution(
            scoped_viz,
            model,
            title=f"residual distribution  ·  {anchor} / reg={reg_lb} / ou={ou_lb}",
        )
        scoped_viz.progress(message="Rendering selected strategy charts and trade log...")

        drill_row = pd.Series({
            "anchor": anchor,
            "horizon": int(best["horizon"]),
            "best_feature": best["filter_col"],
            "filter_col": best["filter_col"],
            "filter_rule": best["filter_rule"],
            "filter_quantile": float(best["filter_quantile"]),
            "sharpe_filtered": float(best.get("sharpe", np.nan)),
        })
        _best_signal_drill(
            drill_row,
            {anchor: model},
            df,
            scoped_viz,
            entry_signal_col=entry_signal_col,
            entry_threshold=float(best["entry_threshold"]),
            ou_lookback=ou_lb,
            residual_target_frac=float(best["residual_target_frac"]),
            stop_loss_bps=None if pd.isna(best.get("stop_loss_bps", np.nan)) else float(best["stop_loss_bps"]),
            directional_mom_filter=directional_mom_filter,
        )

    viz.robustness_explorer(
        anchors=anchors,
        initial_anchor=initial_anchor,
        build_fn=build_for_anchor,
        title="robustness explorer",
        lazy_initial=True,
        progress_text=(
            f"Running focused robustness slices for the selected anchor "
            f"(about {n_engine_runs:,} engine backtests)."
        ),
        controls=[
            {
                "name": "metric",
                "label": "metric",
                "options": [
                    "sharpe",
                    "total_pnl_bps",
                    "profit_factor",
                    "hit_rate",
                    "max_drawdown_bps",
                    "n_trades",
                    "avg_holding_days",
                    "fwd_sharpe_filtered",
                ],
                "value": "sharpe",
                "group": "metric",
                "width": "220px",
            },
            {
                "name": "reg_lookback",
                "label": "reg lb",
                "options": [{"label": "best", "value": "__best__"}, *map(str, robustness_config["reg_lookbacks"])],
                "value": "__best__",
                "group": "params",
                "width": "120px",
                "subgroup": "model",
            },
            {
                "name": "ou_lookback",
                "label": "ou lb",
                "options": [{"label": "best", "value": "__best__"}, *map(str, robustness_config["ou_lookbacks"])],
                "value": "__best__",
                "group": "params",
                "width": "120px",
                "subgroup": "model",
            },
            {
                "name": "filter_rule",
                "label": "filter side",
                "options": [{"label": "best", "value": "__best__"}, "low", "high"],
                "value": "__best__",
                "group": "params",
                "width": "150px",
                "subgroup": "filter",
            },
            {
                "name": "filter_quantile",
                "label": "filter q",
                "options": [
                    {"label": "best", "value": "__best__"},
                    *map(str, sorted(set(
                        robustness_config["filter_cutoffs"]["low"]
                        + robustness_config["filter_cutoffs"]["high"]
                    ))),
                ],
                "value": "__best__",
                "group": "params",
                "width": "150px",
                "subgroup": "filter",
            },
            {
                "name": "entry_threshold",
                "label": "entry resid",
                "options": [{"label": "best", "value": "__best__"}, *map(str, robustness_config["entry_thresholds"])],
                "value": "__best__",
                "group": "params",
                "width": "150px",
                "subgroup": "entry",
            },
            {
                "name": "residual_target_frac",
                "label": "target frac",
                "options": [{"label": "best", "value": "__best__"}, *map(str, robustness_config["residual_target_fracs"])],
                "value": "__best__",
                "group": "params",
                "width": "130px",
                "subgroup": "exit",
            },
            {
                "name": "stop_loss_bps",
                "label": "stop loss",
                "options": [
                    {"label": "best", "value": "__best__"},
                    {"label": "none", "value": "__none__"},
                    *map(str, [x for x in robustness_config["stop_loss_bpss"] if x is not None]),
                ],
                "value": "__best__",
                "group": "params",
                "width": "130px",
                "subgroup": "exit",
            },
        ],
    )

    return {
        "df": df,
        "ten_single": ten_single,
        "ten_single_models": ten_single_models,
        "filt": filt,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", action="store_true", help="load basket from local cache; fetch+save if cache missing")
    args = ap.parse_args()
    state = main(cache=args.cache)
