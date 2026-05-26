"""Core backtest engine.

Signals are pre-computed vectorially (polars). Position management is a
lightweight row-by-row state machine over the pre-computed arrays.

Usage:
    engine = Engine(BacktestConfig(...))
    engine.add_signal(pipeline1)
    engine.add_signal(pipeline2)
    result = engine.run(data)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import polars as pl
import numpy as np

from .trades import TradeDef, Position, ClosedPosition
from .signals import SignalPipeline
from .sizing import DV01Map, size_dv01_neutral


@dataclass
class BacktestConfig:
    """Global backtest configuration."""

    start_date: Optional[str] = None
    end_date: Optional[str] = None
    transaction_cost_bps: float = 0.0
    slippage_bps: float = 0.0
    max_total_positions: int = 10
    dv01_map: Optional[DV01Map] = None


@dataclass
class BacktestResult:
    """Complete backtest output."""

    daily_pnl: pl.DataFrame          # ts, signal_name, pnl_bps, position_count
    equity_curve: pl.DataFrame       # ts, pnl_bps, cumulative_pnl
    signals_ts: pl.DataFrame         # ts, signal_name, signal_value
    closed_trades: list[ClosedPosition]
    open_trades: list[Position]

    _metrics: Optional[dict] = field(default=None, repr=False)

    def summary(self) -> dict:
        from .metrics import compute_metrics

        if self._metrics is None:
            self._metrics = compute_metrics(self)
        return self._metrics


class Engine:
    """Core backtest engine with fluent API for adding signals."""

    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.pipelines: list[SignalPipeline] = []

    def add_signal(self, pipeline: SignalPipeline) -> Engine:
        self.pipelines.append(pipeline)
        return self

    def run(self, data: pl.DataFrame, dates_col: str = "ts") -> BacktestResult:
        """Run the backtest.

        Args:
            data: Wide-format polars DataFrame with a date column and
                  instrument columns (e.g. [ts, 2Y, 5Y, 10Y, 30Y]).
            dates_col: Name of the date column.
        """
        # Date filtering
        df = data
        if self.config.start_date:
            df = df.filter(pl.col(dates_col) >= self.config.start_date)
        if self.config.end_date:
            df = df.filter(pl.col(dates_col) <= self.config.end_date)

        # 1. Pre-compute all signals (vectorized, polars)
        signal_frames: dict[str, pl.DataFrame] = {}
        for pipeline in self.pipelines:
            signal_frames[pipeline.name] = pipeline.run(df)

        # 2. Pre-compute composite trade levels
        composite_levels: dict[str, pl.Series] = {}
        for pipeline in self.pipelines:
            composite_levels[pipeline.name] = pipeline.trade_def.composite_series(df)

        # 3. Extract numpy arrays for fast row-by-row access
        dates = df[dates_col].to_list()
        n = len(dates)

        sig_arrays: dict[str, np.ndarray] = {}
        act_arrays: dict[str, list] = {}
        lvl_arrays: dict[str, np.ndarray] = {}
        ts_arrays: dict[str, Optional[np.ndarray]] = {}  # dynamic time stops
        sz_arrays: dict[str, Optional[np.ndarray]] = {}  # dynamic position sizes

        # Extra columns passed through from compute_fn (beyond signal/action/time_stop/size)
        extras_arrays: dict[str, dict[str, np.ndarray]] = {}

        for pipeline in self.pipelines:
            name = pipeline.name
            sf = signal_frames[name]
            sig_arrays[name] = sf["signal"].to_numpy()
            act_arrays[name] = sf["action"].to_list()
            lvl_arrays[name] = composite_levels[name].to_numpy()
            ts_arrays[name] = sf["time_stop"].to_numpy() if "time_stop" in sf.columns else None
            sz_arrays[name] = sf["size"].to_numpy() if "size" in sf.columns else None
            reserved = {"signal", "action", "time_stop", "size"}
            extras_arrays[name] = {
                c: sf[c].to_numpy()
                for c in sf.columns
                if c not in reserved
            }

        # 4. State machine — position management
        all_closed: list[ClosedPosition] = []
        active: dict[str, list[Position]] = {p.name: [] for p in self.pipelines}
        prev_unrealized: dict[str, float] = {p.name: 0.0 for p in self.pipelines}
        daily_records: list[dict] = []

        tc = self.config.transaction_cost_bps
        slip = self.config.slippage_bps

        for i in range(n):
            date = dates[i]

            for pipeline in self.pipelines:
                name = pipeline.name
                config = pipeline.config
                signal_val = sig_arrays[name][i]
                action = act_arrays[name][i]
                level = lvl_arrays[name][i]

                if np.isnan(signal_val) or np.isnan(level):
                    daily_records.append({
                        "ts": date,
                        "signal_name": name,
                        "signal_value": None,
                        "pnl_bps": 0.0,
                        "position_count": len(active[name]),
                    })
                    continue

                positions = active[name]

                # --- Exits first ---
                remaining = []
                realized_today = 0.0
                for pos in positions:
                    pos.bars_held += 1
                    current_pnl = pos.unrealized_pnl(level)
                    pos.peak_pnl = max(pos.peak_pnl, current_pnl)

                    exit_reason = None

                    # Signal-based exit
                    if pos.direction == 1 and action == "exit_long":
                        exit_reason = "signal"
                    elif pos.direction == -1 and action == "exit_short":
                        exit_reason = "signal"

                    # Stop loss
                    if config.stop_loss_bps and current_pnl < -config.stop_loss_bps:
                        exit_reason = "stop_loss"

                    # Time stop — per-trade dynamic value takes priority over global config
                    effective_ts = pos.dynamic_time_stop or config.time_stop_bars
                    if effective_ts and pos.bars_held >= effective_ts:
                        exit_reason = "time_stop"

                    # Custom exit function — fires after time_stop, before trailing_stop
                    if exit_reason is None and config.exit_fn is not None:
                        bar_data = {"signal": signal_val, "level": level}
                        bar_data.update({
                            c: float(arr[i])
                            for c, arr in extras_arrays[name].items()
                        })
                        fn_reason = config.exit_fn(pos, bar_data)
                        if fn_reason:
                            exit_reason = fn_reason

                    # Trailing stop
                    if (
                        config.trailing_stop_bps
                        and pos.peak_pnl - current_pnl > config.trailing_stop_bps
                    ):
                        exit_reason = "trailing_stop"

                    if exit_reason:
                        pnl_bps = current_pnl - tc
                        realized_today += pnl_bps
                        all_closed.append(
                            ClosedPosition(
                                trade_def=pos.trade_def,
                                entry_date=pos.entry_date,
                                exit_date=date,
                                entry_level=pos.entry_level,
                                exit_level=level,
                                direction=pos.direction,
                                size=pos.size,
                                pnl_bps=pnl_bps,
                                pnl_dollar=self._to_dollar_pnl(pnl_bps, pos),
                                bars_held=pos.bars_held,
                                exit_reason=exit_reason,
                                entry_signal=pos.entry_signal,
                                exit_signal=float(signal_val),
                                leg_sizes=pos.leg_sizes,
                            )
                        )
                    else:
                        remaining.append(pos)

                active[name] = remaining

                # --- Entries ---
                total_active = sum(len(v) for v in active.values())
                if (
                    len(active[name]) < config.max_positions
                    and total_active < self.config.max_total_positions
                ):
                    direction = None
                    if action == "enter_long":
                        direction = 1
                    elif action == "enter_short":
                        direction = -1

                    if direction is not None and config.entry_filter_fn is not None:
                        filter_bar = {"signal": signal_val, "level": level}
                        filter_bar.update({c: float(arr[i]) for c, arr in extras_arrays[name].items()})
                        if not config.entry_filter_fn(direction, filter_bar):
                            direction = None

                    if direction is not None:
                        leg_sizes = {}
                        if self.config.dv01_map:
                            leg_sizes = size_dv01_neutral(
                                pipeline.trade_def, self.config.dv01_map
                            )

                        # Capture rolling time stop at entry (if provided by signal)
                        dyn_ts = None
                        if ts_arrays[name] is not None:
                            raw = ts_arrays[name][i]
                            if not np.isnan(raw):
                                dyn_ts = max(1, int(round(raw)))

                        # Capture dynamic size at entry (if provided by signal)
                        entry_size = 1.0
                        if sz_arrays[name] is not None:
                            raw_sz = sz_arrays[name][i]
                            if not np.isnan(raw_sz) and raw_sz > 0:
                                entry_size = float(raw_sz)

                        entry_extras = {
                            c: float(arr[i])
                            for c, arr in extras_arrays[name].items()
                        }

                        pos = Position(
                            trade_def=pipeline.trade_def,
                            entry_date=date,
                            entry_level=level + (slip * direction),
                            direction=direction,
                            size=entry_size,
                            entry_signal=float(signal_val),
                            leg_sizes=leg_sizes,
                            dynamic_time_stop=dyn_ts,
                            entry_extras=entry_extras,
                        )
                        active[name].append(pos)

                # --- Daily PnL (daily change in mark-to-market) ---
                current_unrealized = sum(p.unrealized_pnl(level) for p in active[name])
                daily_pnl_change = realized_today + current_unrealized - prev_unrealized[name]
                prev_unrealized[name] = current_unrealized

                daily_records.append({
                    "ts": date,
                    "signal_name": name,
                    "signal_value": float(signal_val),
                    "pnl_bps": daily_pnl_change,
                    "position_count": len(active[name]),
                })

        # Collect open positions
        all_open = [pos for positions in active.values() for pos in positions]

        # Build output — explicit schema to avoid type inference issues
        daily_pnl = pl.DataFrame(
            daily_records,
            schema={
                "ts": pl.Date,
                "signal_name": pl.Utf8,
                "signal_value": pl.Float64,
                "pnl_bps": pl.Float64,
                "position_count": pl.Int64,
            },
        )

        equity_curve = (
            daily_pnl.group_by("ts", maintain_order=True)
            .agg(pl.col("pnl_bps").sum())
            .sort("ts")
            .with_columns(pl.col("pnl_bps").cum_sum().alias("cumulative_pnl"))
        )

        signals_ts = daily_pnl.select("ts", "signal_name", "signal_value")

        return BacktestResult(
            daily_pnl=daily_pnl,
            equity_curve=equity_curve,
            signals_ts=signals_ts,
            closed_trades=all_closed,
            open_trades=all_open,
        )

    def _to_dollar_pnl(self, pnl_bps: float, pos: Position) -> float:
        if not self.config.dv01_map or not pos.leg_sizes:
            return pnl_bps
        total = 0.0
        for instrument, size in pos.leg_sizes.items():
            dv01 = self.config.dv01_map.dv01.get(instrument, 1.0)
            total += abs(size) * dv01 * pnl_bps / 100
        return total
