"""Performance analytics — Sharpe, drawdown, hit rate, PnL decomposition."""

from __future__ import annotations
import polars as pl
import numpy as np


def compute_metrics(result) -> dict:
    """Compute comprehensive performance metrics from a BacktestResult."""
    eq = result.equity_curve
    trades = result.closed_trades

    if eq.is_empty() or not trades:
        return _empty_metrics()

    daily_pnl = eq["pnl_bps"].to_numpy().astype(float)
    cum_pnl = eq["cumulative_pnl"].to_numpy().astype(float)

    total_pnl = float(cum_pnl[-1]) if len(cum_pnl) > 0 else 0.0
    n_trades = len(trades)
    winners = [t for t in trades if t.pnl_bps > 0]
    losers = [t for t in trades if t.pnl_bps <= 0]

    hit_rate = len(winners) / n_trades if n_trades > 0 else 0.0
    avg_win = float(np.mean([t.pnl_bps for t in winners])) if winners else 0.0
    avg_loss = float(np.mean([t.pnl_bps for t in losers])) if losers else 0.0

    total_wins = sum(t.pnl_bps for t in winners)
    total_losses = sum(t.pnl_bps for t in losers)
    profit_factor = (
        abs(total_wins / total_losses) if total_losses != 0 else float("inf")
    )

    # Sharpe (annualized, daily data)
    daily_std = float(np.std(daily_pnl)) if len(daily_pnl) > 1 else 1.0
    daily_mean = float(np.mean(daily_pnl))
    sharpe = (daily_mean / daily_std) * np.sqrt(252) if daily_std > 0 else 0.0

    # Drawdown
    running_max = np.maximum.accumulate(cum_pnl)
    drawdown = cum_pnl - running_max
    max_drawdown = float(drawdown.min())

    # Calmar
    calmar = total_pnl / abs(max_drawdown) if max_drawdown != 0 else float("inf")

    # Average holding period
    avg_bars = float(np.mean([t.bars_held for t in trades]))

    return {
        "total_pnl_bps": total_pnl,
        "n_trades": n_trades,
        "hit_rate": hit_rate,
        "avg_win_bps": avg_win,
        "avg_loss_bps": avg_loss,
        "profit_factor": profit_factor,
        "sharpe": float(sharpe),
        "max_drawdown_bps": max_drawdown,
        "calmar": float(calmar),
        "avg_holding_days": avg_bars,
        "win_loss_ratio": abs(avg_win / avg_loss) if avg_loss != 0 else float("inf"),
    }


def drawdown_series(equity_curve: pl.DataFrame) -> pl.DataFrame:
    """Compute rolling drawdown from equity curve."""
    return equity_curve.with_columns(
        pl.col("cumulative_pnl").cum_max().alias("peak"),
    ).with_columns(
        (pl.col("cumulative_pnl") - pl.col("peak")).alias("drawdown"),
    )


def trade_log(closed_trades: list) -> pl.DataFrame:
    """Convert closed trades to a polars DataFrame for analysis."""
    if not closed_trades:
        return pl.DataFrame()

    records = []
    for t in closed_trades:
        record = {
            "trade_name": t.trade_def.name,
            "entry_date": t.entry_date,
            "exit_date": t.exit_date,
            "direction": "long" if t.direction == 1 else "short",
            "size": t.size,
            "entry_level": t.entry_level,
            "exit_level": t.exit_level,
            "pnl_bps": t.pnl_bps,
            "pnl_dollar": t.pnl_dollar,
            "bars_held": t.bars_held,
            "exit_reason": t.exit_reason,
            "entry_signal": t.entry_signal,
            "exit_signal": t.exit_signal,
        }
        for key, value in getattr(t, "entry_extras", {}).items():
            record[f"entry_{key}"] = value
        for key, value in getattr(t, "exit_extras", {}).items():
            record[f"exit_{key}"] = value
        records.append(record)
    return pl.DataFrame(records)


def _empty_metrics() -> dict:
    return {
        k: 0.0
        for k in [
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
            "win_loss_ratio",
        ]
    }
