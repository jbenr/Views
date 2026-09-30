"""Exact, same-day-entry backtests for a frozen dislocation candidate.

This is intentionally an adapter over :mod:`backtest.engine`, not another
simulator.  Discovery owns the relationship search; this module freezes one
row and varies only trade mechanics such as entry magnitude and exit style.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import polars as pl

from backtest.engine import (
    BacktestConfig,
    BooleanSignalPipeline,
    Engine,
    SignalConfig,
    TradeDef,
    profit_target,
    trade_log,
)
from backtest.lab import gate_allow_mask
from backtest.vector import run_vector
from stats import roll_lr, roll_lr_diff, roll_ou_features
from stats.diagnostics import beta_cv, quality_weight
from utils.market_data import align_columns


def _window_r2(x: pl.Series, y: pl.Series, lookback: int) -> pl.Series:
    """Exact trailing one-factor R² (squared current-window correlation)."""
    m = pl.DataFrame({"x": x, "y": y}).with_columns(
        pl.col("x").rolling_sum(lookback, min_samples=lookback).alias("sx"),
        pl.col("y").rolling_sum(lookback, min_samples=lookback).alias("sy"),
        (pl.col("x") * pl.col("x")).rolling_sum(
            lookback, min_samples=lookback
        ).alias("sxx"),
        (pl.col("y") * pl.col("y")).rolling_sum(
            lookback, min_samples=lookback
        ).alias("syy"),
        (pl.col("x") * pl.col("y")).rolling_sum(
            lookback, min_samples=lookback
        ).alias("sxy"),
        pl.col("x").is_not_null().cast(pl.Float64).rolling_sum(
            lookback, min_samples=lookback
        ).alias("n"),
    ).with_columns(
        (pl.col("n") * pl.col("sxx") - pl.col("sx") ** 2).alias("xx"),
        (pl.col("n") * pl.col("syy") - pl.col("sy") ** 2).alias("yy"),
        (pl.col("n") * pl.col("sxy") - pl.col("sx") * pl.col("sy")).alias("xy"),
    )
    return m.select(
        pl.when((pl.col("xx") > 0) & (pl.col("yy") > 0))
        .then(pl.col("xy") ** 2 / (pl.col("xx") * pl.col("yy")))
        .otherwise(None)
        .alias("r2")
    )["r2"]


def signal_frame(
    data: pl.DataFrame,
    *,
    target: str,
    feature: str,
    fit_on: str,
    beta_lb: int,
    residual_lb: int | None,
    norm_lb: int,
    signal_kind: str = "normalized",
) -> pl.DataFrame:
    """One candidate's causal signal and gate-condition frame.

    The changes branch is the exact windowed accumulated residual used in
    discovery.  Levels has no residual window because its regression residual
    is already the level dislocation.
    """
    if fit_on not in {"changes", "levels"}:
        raise ValueError("fit_on must be 'changes' or 'levels'")
    frame = align_columns(data, [target, feature]).sort("ts")
    x, y = frame[feature].cast(pl.Float64), frame[target].cast(pl.Float64)
    if fit_on == "changes":
        if residual_lb is None:
            raise ValueError("changes backtest needs a residual lookback")
        reg = roll_lr_diff(x, y, lookback=beta_lb)
        pad = len(frame) - len(reg)
        innovation = pl.concat([
            pl.Series([None] * pad, dtype=pl.Float64), reg["resid"],
        ])
        beta = pl.concat([pl.Series([None] * pad, dtype=pl.Float64), reg["beta"]])
        resid = innovation.rolling_sum(residual_lb, min_samples=residual_lb)
        r2 = _window_r2(x.diff(), y.diff(), beta_lb)
    else:
        reg = roll_lr(x, y, lookback=beta_lb)
        resid, beta = reg["resid"], reg["beta"]
        r2 = _window_r2(x, y, beta_lb)

    ou = roll_ou_features(resid, lookback=norm_lb)
    if signal_kind not in {"normalized", "ou_z", "raw"}:
        raise ValueError("signal_kind must be normalized, ou_z or raw")
    signal = resid if signal_kind == "raw" else ou["ou_z"] if signal_kind == "ou_z" else resid / resid.rolling_std(norm_lb, min_samples=norm_lb)
    stability = beta_cv(beta, lookback=beta_lb)
    conditions = {
        "feature_level": x,
        "feature_move20": x.diff(20),
        "feature_vol20": x.diff().rolling_std(20),
        "target_level": y,
        "target_move20": y.diff(20),
        "target_vol20": y.diff().rolling_std(20),
        "r2": r2,
        "beta": beta,
        "beta_cv": stability,
        "model_quality": quality_weight(r2, stability),
        "beta_vol20": beta.diff().rolling_std(20),
        "beta_mom10": beta.diff(10),
        "r2_vol20": r2.diff().rolling_std(20),
        "r2_mom10": r2.diff(10),
        "resid_vol20": resid.diff().rolling_std(20),
        "resid_vol60": resid.diff().rolling_std(60),
        "resid_mom10": resid.diff(10),
        "resid_phi": ou["ou_rho"] - 1,
        "resid_half_life": ou["half_life"],
    }
    return frame.select("ts").with_columns(
        signal.alias("signal"), resid.alias("resid"), beta.alias("beta"), r2.alias("r2"),
        ou["half_life"].alias("half_life"),
        *[value.alias(f"gate_{name}") for name, value in conditions.items()],
    )


def _entry_gate(state: pl.DataFrame, candidate: dict) -> np.ndarray:
    gate = candidate.get("gate", "(none)")
    if gate in (None, "(none)"):
        return np.ones(len(state), dtype=bool)
    column = f"gate_{gate}"
    if column not in state.columns:
        raise ValueError(f"unknown candidate gate: {gate!r}")
    return gate_allow_mask(
        state[column],
        (gate, candidate["gate_bucket"]),
        min_history=126,
        window=int(candidate["gate_window"]),
    )


def make_pipeline(
    data: pl.DataFrame,
    trade: TradeDef,
    candidate: dict,
    *,
    entry_z: float,
    exit_style: str,
    exit_param: float,
    stop_loss_bps: float | None,
    state: pl.DataFrame | None = None,
    execution_lag: int = 0,
) -> BooleanSignalPipeline:
    """Build one exact Engine pipeline with same-day first-crossing entries."""
    if state is None:
        state = signal_frame(
            data,
            target=candidate["target"], feature=candidate["feature"],
            fit_on=candidate["fit_on"], beta_lb=int(candidate["beta_lb"]),
            residual_lb=(
                None if candidate.get("residual_lb") is None
                else int(candidate["residual_lb"])
            ),
            norm_lb=int(candidate["norm_lb"]),
            signal_kind=candidate.get("signal_kind", "normalized"),
        )
    signal = state["signal"].to_numpy().astype(float)
    previous = np.concatenate([[np.nan], signal[:-1]])
    allowed = _entry_gate(state, candidate)
    enter_long = (signal <= -entry_z) & ~(previous <= -entry_z) & allowed
    enter_short = (signal >= entry_z) & ~(previous >= entry_z) & allowed

    exit_long = np.zeros(len(state), dtype=bool)
    exit_short = np.zeros(len(state), dtype=bool)
    time_stop = None
    exit_fn = None
    if exit_style == "time":
        time_stop = int(round(exit_param))
    elif exit_style == "band":
        exit_long = signal > -exit_param
        exit_short = signal < exit_param
    elif exit_style == "revert_frac":
        exit_fn = profit_target(float(exit_param))
    elif exit_style == "half_life_frac":
        valid = np.isfinite(state["half_life"].to_numpy()) & (state["half_life"].to_numpy() > 0)
        enter_long &= valid
        enter_short &= valid
    else:
        raise ValueError(f"unknown exit style: {exit_style!r}")

    def compute(_data: pl.DataFrame) -> pl.DataFrame:
        frame = state.select("signal", "resid", "beta", "r2", "half_life").with_columns(
            pl.Series("enter_long", enter_long),
            pl.Series("enter_short", enter_short),
            pl.Series("exit_long", exit_long),
            pl.Series("exit_short", exit_short),
        )
        if exit_style == "half_life_frac":
            frame = frame.with_columns((pl.col("half_life") * exit_param).ceil().alias("time_stop"))
        if candidate.get("weight_columns"):
            frame = frame.with_columns(*[data[c] for c in candidate["weight_columns"].values()])
        return frame.shift(execution_lag)

    return BooleanSignalPipeline(
        name="dislocation",
        trade_def=trade,
        compute_fn=compute,
        config=SignalConfig(
            stop_loss_bps=stop_loss_bps,
            time_stop_bars=time_stop,
            exit_fn=exit_fn,
            max_positions=1,
            entry_weight_columns=candidate.get("weight_columns"),
        ),
    )


def _metrics(result) -> dict:
    metrics = dict(result.summary())
    trades = trade_log(result.closed_trades)
    n = int(metrics["n_trades"])
    metrics["avg_pnl_per_trade_bps"] = float(trades["pnl_bps"].mean()) if n else 0.0
    metrics["trade_win_rate"] = metrics.pop("hit_rate")
    # Equity includes open marks even when no trade has closed yet.
    pnl = result.equity_curve["pnl_bps"].to_numpy()
    equity = np.cumsum(pnl)
    metrics["total_pnl_bps"] = float(pnl.sum())
    metrics["closed_pnl_bps"] = float(trades["pnl_bps"].sum()) if n else 0.0
    metrics["sharpe"] = float(pnl.mean()/pnl.std()*np.sqrt(252)) if len(pnl) and pnl.std() > 0 else 0.0
    metrics["max_drawdown_bps"] = float(np.min(equity - np.maximum.accumulate(np.r_[0, equity])[1:])) if len(equity) else 0.0
    metrics["time_in_market_pct"] = float(
        (result.daily_pnl["position_count"] > 0).mean()
    ) if len(result.daily_pnl) else 0.0
    metrics["open_trades"] = len(result.open_trades)
    if trades.is_empty():
        metrics.update(gross_profit_bps=0.0, gross_loss_bps=0.0, median_pnl_bps=0.0)
    else:
        metrics.update(
            gross_profit_bps=float(trades.filter(pl.col("pnl_bps") > 0)["pnl_bps"].sum() or 0.0),
            gross_loss_bps=float(trades.filter(pl.col("pnl_bps") <= 0)["pnl_bps"].sum() or 0.0),
            median_pnl_bps=float(trades["pnl_bps"].median()),
        )
    return metrics


def _grid_inputs(data, trade, candidate, entry_zs, exit_params, stop_losses,
                 stop_loss_bps, execution_lag) -> tuple[pl.DataFrame, pl.DataFrame, list[tuple]]:
    """Aligned data, the frozen candidate's signal frame and the valid grid cells."""
    # Engine and signal frame must share precisely the same bar index.  The
    # feature can have a shorter history than the executable target legs, so
    # trim once here rather than silently attaching a shorter signal by row.
    # An outright target's name is one of its own legs; dedupe or align_columns
    # sees the same column requested twice and polars refuses the projection.
    data = align_columns(
        data, list(dict.fromkeys([candidate["target"], candidate["feature"], *trade.legs,
                                 *candidate.get("weight_columns", {}).values()]))
    ).sort("ts")
    if execution_lag not in (0, 1):
        raise ValueError("execution_lag must be 0 or 1")
    # The relationship and its gate are invariant across entry and exit
    # mechanics.  Build them once; only the exact position state machine is
    # repeated for each requested grid cell.
    state = signal_frame(
        data,
        target=candidate["target"], feature=candidate["feature"],
        fit_on=candidate["fit_on"], beta_lb=int(candidate["beta_lb"]),
        residual_lb=(
            None if candidate.get("residual_lb") is None
            else int(candidate["residual_lb"])
        ),
        norm_lb=int(candidate["norm_lb"]),
        signal_kind=candidate.get("signal_kind", "normalized"),
    )
    cells = [
        (float(entry), style, float(param), stop)
        for entry in entry_zs
        for style, params in exit_params.items()
        for param in params
        for stop in (list(stop_losses) if stop_losses is not None else [stop_loss_bps])
        if style != "band" or float(param) < float(entry)
    ]
    return data, state, cells


def run_detail(
    data: pl.DataFrame,
    trade: TradeDef,
    candidate: dict,
    *,
    state: pl.DataFrame,
    config_id: str,
    entry_z: float,
    exit_style: str,
    exit_param: float,
    stop_loss_bps: float | None,
    round_trip_cost_bps: float,
    execution_lag: int,
) -> dict:
    """One cell through the exact Engine: metrics row, trades with MAE/MFE, equity, years.

    ``data`` must already be aligned to ``state`` (see ``_grid_inputs``).
    """
    pipeline = make_pipeline(
        data, trade, candidate, entry_z=entry_z, exit_style=exit_style,
        exit_param=exit_param, stop_loss_bps=stop_loss_bps or None,
        state=state, execution_lag=execution_lag,
    )
    result = Engine(BacktestConfig(
        transaction_cost_bps=round_trip_cost_bps,
        max_total_positions=1,
    )).add_signal(pipeline).run(data)
    row = {
        "config_id": config_id, "stop_loss_bps": stop_loss_bps,
        "execution_lag": execution_lag,
        "signal_kind": candidate.get("signal_kind", "normalized"),
        "signal_units": "target units" if candidate.get("signal_kind") == "raw" else "standard deviations",
        "entry_z": entry_z, "exit_style": exit_style,
        "exit_param": exit_param, **_metrics(result),
    }
    if candidate.get("split_date"):
        for label, predicate in (
            ("earlier", pl.col("ts").cast(pl.Utf8) < candidate["split_date"]),
            ("later", pl.col("ts").cast(pl.Utf8) >= candidate["split_date"]),
        ):
            pnl = result.daily_pnl.filter(predicate)["pnl_bps"]
            std = float(pnl.std() or 0)
            row[f"{label}_sharpe"] = float(pnl.mean() or 0) / std * np.sqrt(252) if std else 0.0
            row[f"{label}_pnl_bps"] = float(pnl.sum() or 0)
    trades = trade_log(result.closed_trades).to_dicts()
    for item in trades:
        path = data.filter(pl.col("ts").is_between(item["entry_date"], item["exit_date"]))
        held_trade = TradeDef(trade.name, {leg: item[f"entry_{col}"] for leg, col in candidate["weight_columns"].items()}) if candidate.get("weight_columns") else trade
        moves = (held_trade.composite_series(path) - item["entry_level"]) * (1 if item["direction"] == "long" else -1)
        item["mae_bps"] = min(0.0, float(moves.min() or 0))
        item["mfe_bps"] = max(0.0, float(moves.max() or 0))
    periods = result.daily_pnl.with_columns(pl.col("ts").dt.year().alias("year")).group_by("year").agg(
        pl.col("pnl_bps").sum().alias("pnl_bps"),
        (pl.col("position_count") > 0).sum().alias("active_days"),
    ).sort("year").to_dicts()
    return {"metrics": row, "trades": trades,
            "equity": result.equity_curve.to_dicts(), "periods": periods}


def detail_for(
    data: pl.DataFrame,
    trade: TradeDef,
    candidate: dict,
    *,
    config_id: str,
    entry_z: float,
    exit_style: str,
    exit_param: float,
    stop_loss_bps: float | None,
    round_trip_cost_bps: float,
    execution_lag: int,
) -> dict:
    """Exact Engine detail for one grid cell, on the same alignment as the grid."""
    data, state, _ = _grid_inputs(data, trade, candidate, [], {}, None, None, execution_lag)
    return run_detail(
        data, trade, candidate, state=state, config_id=config_id, entry_z=entry_z,
        exit_style=exit_style, exit_param=exit_param, stop_loss_bps=stop_loss_bps,
        round_trip_cost_bps=round_trip_cost_bps, execution_lag=execution_lag,
    )


def run_grid(
    data: pl.DataFrame,
    trade: TradeDef,
    candidate: dict,
    *,
    entry_zs: Iterable[float],
    exit_params: dict[str, Iterable[float]],
    stop_loss_bps: float | None = None,
    round_trip_cost_bps: float = 0.0,
    progress: Callable[[int, int], None] | None = None,
    stop_losses: Iterable[float | None] | None = None,
    execution_lag: int = 0,
) -> tuple[pl.DataFrame, dict]:
    """Run every requested entry/exit cell through the exact Engine."""
    data, state, cells = _grid_inputs(data, trade, candidate, entry_zs, exit_params,
                                      stop_losses, stop_loss_bps, execution_lag)
    rows: list[dict] = []
    selected: dict = {}
    runs: dict = {}
    rank_metric = "earlier_sharpe" if candidate.get("split_date") else "sharpe"
    if not cells:
        raise ValueError("No valid exit combinations: exit bands must be below entry thresholds")
    for done, (entry_z, style, exit_param, stop) in enumerate(cells, 1):
        run = run_detail(
            data, trade, candidate, state=state, config_id=str(done), entry_z=entry_z,
            exit_style=style, exit_param=exit_param, stop_loss_bps=stop,
            round_trip_cost_bps=round_trip_cost_bps, execution_lag=execution_lag,
        )
        row = run["metrics"]
        rows.append(row)
        runs[str(done)] = run
        # Preserve one fully inspectable run for the current best Sharpe.
        if not selected or (np.isfinite(row[rank_metric]) and
                            (not np.isfinite(selected["metrics"][rank_metric]) or row[rank_metric] > selected["metrics"][rank_metric])):
            selected = dict(runs[str(done)])
        if progress is not None:
            progress(done, len(cells))
    selected["runs"] = runs
    selected["rank_metric"] = rank_metric
    return pl.DataFrame(rows), selected


def run_vector_grid(
    data: pl.DataFrame,
    trade: TradeDef,
    candidate: dict,
    *,
    entry_zs: Iterable[float],
    exit_params: dict[str, Iterable[float]],
    stop_losses: Iterable[float | None] = (None,),
    round_trip_cost_bps: float = 0.0,
    execution_lag: int = 0,
) -> tuple[pl.DataFrame, dict, pl.DataFrame, pl.DataFrame]:
    """``run_grid``'s table from the vectorised engine; exact detail for the best cell only.

    Same cells, candidate, alignment and metrics as ``run_grid`` (the parity
    tests hold ``backtest.vector`` to ``Engine``), except the median trade,
    which needs every trade and so is left to ``run_detail``. Returns the
    grid, the selected best run (with its full detail under ``runs``), the
    per-config yearly P&L and the aligned data the grid was run on.
    """
    data, state, cells = _grid_inputs(data, trade, candidate, entry_zs, exit_params,
                                      stop_losses, None, execution_lag)
    if not cells:
        raise ValueError("No valid exit combinations: exit bands must be below entry thresholds")
    gated = candidate.get("gate", "(none)") not in (None, "(none)")
    configs = pl.DataFrame({
        "model": [0] * len(cells), "gate": [0 if gated else -1] * len(cells),
        "entry": [c[0] for c in cells], "exit_style": [c[1] for c in cells],
        "exit_param": [c[2] for c in cells], "stop": [float(c[3] or 0.0) for c in cells],
    })
    weights = candidate.get("weight_columns")
    split = candidate.get("split_date")
    later = (data["ts"].cast(pl.Utf8) >= split).to_numpy() if split else np.zeros(len(data), dtype=bool)
    # One fold per (year, side of the split): both views come from the same sums.
    result = run_vector(
        data.select(list(trade.legs)).to_numpy(), np.array(list(trade.legs.values()), dtype=float),
        state["signal"].to_numpy()[None, :], configs,
        half_life=state["half_life"].to_numpy()[None, :],
        gates=_entry_gate(state, candidate)[None, :] if gated else None,
        entry_weights=data.select([weights[leg] for leg in trade.legs]).to_numpy() if weights else None,
        cost=round_trip_cost_bps, lag=execution_lag,
        folds=data["ts"].dt.year().to_numpy() * 2 + later,
    )
    m = result.metrics
    grid = pl.DataFrame({
        "config_id": [str(i) for i in range(1, len(cells) + 1)],
        "stop_loss_bps": [c[3] for c in cells],
        "execution_lag": [execution_lag] * len(cells),
        "signal_kind": [candidate.get("signal_kind", "normalized")] * len(cells),
        "signal_units": ["target units" if candidate.get("signal_kind") == "raw" else "standard deviations"] * len(cells),
        "entry_z": m["entry"], "exit_style": m["exit_style"], "exit_param": m["exit_param"],
    }).with_columns(m.select(
        "total_pnl_bps", "n_trades", "trade_win_rate", "sharpe", "max_drawdown_bps", "avg_holding_days",
        "avg_pnl_per_trade_bps", "closed_pnl_bps", "time_in_market_pct", "open_trades",
        "gross_profit_bps", "gross_loss_bps",
        "avg_win_bps", "avg_loss_bps", "profit_factor", "calmar", "win_loss_ratio",
    ))
    if split:
        is_later = result.folds % 2 == 1
        for label, mask in (("earlier", ~is_later), ("later", is_later)):
            # run_grid uses polars' sample std (ddof=1) for the split Sharpes.
            n = result.fold_days[mask].sum()
            total = result.fold_sum[:, mask].sum(axis=1)
            sq = result.fold_sumsq[:, mask].sum(axis=1)
            var = (sq - total * total / n) / (n - 1) if n > 1 else np.zeros(len(cells))
            with np.errstate(invalid="ignore", divide="ignore"):
                sharpe = np.where(var > 0, total / n / np.sqrt(np.maximum(var, 0)) * np.sqrt(252), 0.0)
            grid = grid.with_columns(pl.Series(f"{label}_sharpe", sharpe), pl.Series(f"{label}_pnl_bps", total))
    years = np.unique(result.folds // 2)
    by_year = {y: result.folds // 2 == y for y in years}
    yearly = pl.DataFrame({
        "config_id": np.repeat(grid["config_id"].to_numpy(), len(years)),
        "year": np.tile(years, len(cells)),
        "pnl_bps": np.column_stack([result.fold_sum[:, by_year[y]].sum(axis=1) for y in years]).ravel(),
        "trades": np.column_stack([result.fold_trades[:, by_year[y]].sum(axis=1) for y in years]).ravel(),
    })
    rank_metric = "earlier_sharpe" if split else "sharpe"
    best = grid.row(int(np.argmax(grid[rank_metric].to_numpy())), named=True)
    detail = run_detail(
        data, trade, candidate, state=state, config_id=best["config_id"], entry_z=best["entry_z"],
        exit_style=best["exit_style"], exit_param=best["exit_param"], stop_loss_bps=best["stop_loss_bps"],
        round_trip_cost_bps=round_trip_cost_bps, execution_lag=execution_lag,
    )
    return grid, {**detail, "runs": {best["config_id"]: detail}, "rank_metric": rank_metric}, yearly, data
