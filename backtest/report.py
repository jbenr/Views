"""Output bridge — summary tables, equity curves, pandas conversion for Viz."""

from __future__ import annotations
import polars as pl

from .metrics import trade_log


def summary_table(result) -> pl.DataFrame:
    """One-row summary table."""
    return pl.DataFrame([result.summary()])


def equity_curve_pd(result):
    """Convert equity curve to pandas for visualization with Viz class."""
    return result.equity_curve.to_pandas().set_index("ts")


def trade_log_pd(result):
    """Convert trade log to pandas for display."""
    return trade_log(result.closed_trades).to_pandas()


def print_summary(result) -> None:
    """Pretty-print the backtest summary."""
    m = result.summary()
    print("=" * 60)
    print("  BACKTEST SUMMARY")
    print("=" * 60)
    print(f"  Total PnL:         {m['total_pnl_bps']:>10.1f} bps")
    print(f"  # Trades:          {m['n_trades']:>10d}")
    print(f"  Hit Rate:          {m['hit_rate']:>10.1%}")
    print(f"  Avg Win:           {m['avg_win_bps']:>10.1f} bps")
    print(f"  Avg Loss:          {m['avg_loss_bps']:>10.1f} bps")
    print(f"  Profit Factor:     {m['profit_factor']:>10.2f}")
    print(f"  Sharpe:            {m['sharpe']:>10.2f}")
    print(f"  Max Drawdown:      {m['max_drawdown_bps']:>10.1f} bps")
    print(f"  Calmar:            {m['calmar']:>10.2f}")
    print(f"  Avg Holding:       {m['avg_holding_days']:>10.1f} days")
    print(f"  Win/Loss Ratio:    {m['win_loss_ratio']:>10.2f}")
    print("=" * 60)

    if result.open_trades:
        print(f"\n  Open Positions: {len(result.open_trades)}")
        for pos in result.open_trades:
            d = "LONG" if pos.direction == 1 else "SHORT"
            print(f"    {pos.trade_def.name} {d} @ {pos.entry_level:.4f} ({pos.bars_held}d)")
