"""The exact backtest simulator — everything about running one true backtest.

Signals are pre-computed vectorially (polars); position management is a
lightweight row-by-row state machine over the pre-computed arrays.

Sections:
    trade definitions    TradeDef / Position / ClosedPosition
    signal interface     SignalConfig / SignalPipeline / exit recipes
    sizing               DV01Map / size_dv01_neutral / size_beta_weighted
    the engine           BacktestConfig / BacktestResult / Engine
    metrics & reporting  compute_metrics / trade_log / print_summary
    spread book          SpreadBook — cross-sectional multi-spread allocation

Usage:
    engine = Engine(BacktestConfig(...))
    engine.add_signal(pipeline1)
    engine.add_signal(pipeline2)
    result = engine.run(data)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional
import numpy as np
import polars as pl


# ── trade definitions ────────────────────────────────────────────────────────
#
# Everything (outrights, spreads, butterflies, N-leg custom) is just a dict of
# {instrument: weight}. TradeDef is the immutable 'what', Position is the live
# 'when/how much', ClosedPosition is the archive.


class TradeType(Enum):
    OUTRIGHT = "outright"
    SPREAD = "spread"
    BUTTERFLY = "butterfly"
    CUSTOM = "custom"


@dataclass(frozen=True)
class TradeDef:
    """Immutable trade definition — the 'what' of a trade."""

    name: str
    legs: dict[str, float]  # instrument -> weight
    trade_type: TradeType = TradeType.CUSTOM

    @classmethod
    def outright(cls, name: str, instrument: str, direction: float = 1.0) -> TradeDef:
        return cls(name=name, legs={instrument: direction}, trade_type=TradeType.OUTRIGHT)

    @classmethod
    def spread(
        cls,
        name: str,
        short_leg: str,
        long_leg: str,
        weights: tuple[float, float] = (-1.0, 1.0),
    ) -> TradeDef:
        return cls(
            name=name,
            legs={short_leg: weights[0], long_leg: weights[1]},
            trade_type=TradeType.SPREAD,
        )

    @classmethod
    def butterfly(
        cls,
        name: str,
        wing1: str,
        belly: str,
        wing2: str,
        weights: tuple[float, float, float] = (1.0, -2.0, 1.0),
    ) -> TradeDef:
        return cls(
            name=name,
            legs={wing1: weights[0], belly: weights[1], wing2: weights[2]},
            trade_type=TradeType.BUTTERFLY,
        )

    def composite_series(self, data: pl.DataFrame) -> pl.Series:
        """Compute weighted combination from wide-format yields DataFrame.

        E.g. for 2s10s with legs {"2Y": -1, "10Y": 1}:
            result = -1 * data["2Y"] + 1 * data["10Y"]
        """
        expr = None
        for col, wt in self.legs.items():
            term = pl.col(col) * wt
            expr = term if expr is None else expr + term
        return data.select(expr.alias(self.name))[self.name]


@dataclass
class Position:
    """A live position — tracks state from entry through exit."""

    trade_def: TradeDef
    entry_date: object
    entry_level: float
    direction: int  # +1 long, -1 short
    size: float = 1.0
    entry_signal: float = 0.0
    leg_sizes: dict[str, float] = field(default_factory=dict)

    # Arbitrary per-bar data captured at entry (e.g. resid, ou_mean for exit_fn)
    entry_extras: dict = field(default_factory=dict)

    # Mutable tracking state
    peak_pnl: float = 0.0
    bars_held: int = 0
    dynamic_time_stop: Optional[int] = None  # per-trade time stop (e.g. rolling half-life at entry)

    def unrealized_pnl(self, current_level: float) -> float:
        """PnL in composite units (bps for yield spreads)."""
        return self.direction * (current_level - self.entry_level) * self.size


@dataclass
class ClosedPosition:
    """Archived position with full PnL record."""

    trade_def: TradeDef
    entry_date: object
    exit_date: object
    entry_level: float
    exit_level: float
    direction: int
    size: float
    pnl_bps: float
    pnl_dollar: float
    bars_held: int
    exit_reason: str  # "signal", "exit_fn", "stop_loss", "time_stop", "trailing_stop"
    entry_signal: float
    exit_signal: float
    leg_sizes: dict[str, float] = field(default_factory=dict)
    entry_extras: dict = field(default_factory=dict)
    exit_extras: dict = field(default_factory=dict)


# ── signal interface ─────────────────────────────────────────────────────────
#
# Signals are just polars Series of floats (z-scores, residuals, etc.).
# SignalConfig defines the thresholds. SignalPipeline ties a signal
# computation function to a TradeDef and config.


@dataclass
class SignalConfig:
    """Entry/exit mechanics for a signal.

    Entry: go long when signal < entry_long, short when signal > entry_short.
    Exit:  close long when signal > exit_long, close short when signal < exit_short.
           Set to None to disable signal-based exits (use stops only).
    """

    # Entry thresholds
    entry_long: float = -2.0
    entry_short: float = 2.0

    # Exit thresholds — None means signal exits disabled
    exit_long: Optional[float] = 0.0
    exit_short: Optional[float] = 0.0

    # Scale-in: list of (threshold, additional_size_fraction)
    scale_in_levels: list[tuple[float, float]] = field(default_factory=list)

    # Stops
    stop_loss_bps: Optional[float] = None
    time_stop_bars: Optional[int] = None
    trailing_stop_bps: Optional[float] = None

    # Custom exit function — called each bar for every open position.
    # Signature: (pos: Position, bar: dict) -> Optional[str]
    # Return a reason string to exit, None to stay in.
    # Runs after time_stop; trailing_stop still overrides it.
    # bar contains: "signal", "level", plus any extras the compute_fn passed through.
    exit_fn: Optional[Callable] = None

    # Entry gate — called before opening each new position.
    # Signature: (direction: int, bar: dict) -> bool
    # Return False to block the entry. direction is +1 or -1.
    # bar contains: "signal", "level", plus any extras from compute_fn.
    entry_filter_fn: Optional[Callable] = None

    # Position limits per signal
    max_positions: int = 1


def generate_signals(signal_series: pl.Series, config: SignalConfig) -> pl.DataFrame:
    """Convert continuous signal into discrete entry/exit actions.

    Returns DataFrame with columns: signal, action.
    Actions: "enter_long", "enter_short", "exit_long", "exit_short", or null.
    """
    sig = pl.col("signal")

    # Start with entries
    expr = (
        pl.when(sig < config.entry_long).then(pl.lit("enter_long"))
        .when(sig > config.entry_short).then(pl.lit("enter_short"))
    )

    # Add exits only if thresholds are set
    if config.exit_long is not None:
        expr = expr.when(
            (sig > config.exit_long) & (sig < config.entry_short)
        ).then(pl.lit("exit_long"))

    if config.exit_short is not None:
        expr = expr.when(
            (sig < config.exit_short) & (sig > config.entry_long)
        ).then(pl.lit("exit_short"))

    expr = expr.otherwise(pl.lit(None)).alias("action")

    return pl.DataFrame({"signal": signal_series}).with_columns(expr)


def generate_boolean_actions(signal_frame: pl.DataFrame) -> pl.DataFrame:
    """Convert explicit boolean signal columns into engine actions.

    Required columns:
      - signal
      - enter_long, enter_short, exit_long, exit_short

    Exits take priority over entries. That keeps the engine conservative:
    it will close an existing trade before considering a fresh entry on a
    later bar, instead of same-bar reversing.
    """
    required = {"signal", "enter_long", "enter_short", "exit_long", "exit_short"}
    missing = required.difference(signal_frame.columns)
    if missing:
        raise ValueError(f"boolean signal frame missing columns: {sorted(missing)}")

    action = (
        pl.when(pl.col("exit_long")).then(pl.lit("exit_long"))
        .when(pl.col("exit_short")).then(pl.lit("exit_short"))
        .when(pl.col("enter_long")).then(pl.lit("enter_long"))
        .when(pl.col("enter_short")).then(pl.lit("enter_short"))
        .otherwise(pl.lit(None))
        .alias("action")
    )
    return signal_frame.with_columns(action)


@dataclass
class SignalPipeline:
    """Wraps a signal computation function + its TradeDef + config.

    The compute_fn takes a polars DataFrame (wide-format data) and returns
    a polars Series of signal values (z-scores, residuals, etc.).

    Example:
        def my_signal(data: pl.DataFrame) -> pl.Series:
            reg = roll_lr(data["10Y"], data["2Y"], lookback=100)
            return ou_zscore(reg["resid"], lookback=100)

        pipeline = SignalPipeline(
            name="2s10s_ou",
            trade_def=TradeDef.spread("2s10s", "2Y", "10Y"),
            compute_fn=my_signal,
            config=SignalConfig(entry_long=-2.0, entry_short=2.0),
        )
    """

    name: str
    trade_def: TradeDef
    compute_fn: Callable[[pl.DataFrame], pl.Series]
    config: SignalConfig = field(default_factory=SignalConfig)

    def run(self, data: pl.DataFrame) -> pl.DataFrame:
        """Compute signal and generate entry/exit actions.

        compute_fn may return:
          - pl.Series: just the signal values
          - pl.DataFrame: must contain "signal" column, may contain "time_stop"
        """
        result = self.compute_fn(data)

        if isinstance(result, pl.DataFrame):
            signal = result["signal"]
            extras = {c: result[c] for c in result.columns if c != "signal"}
        else:
            signal = result
            extras = {}

        actions = generate_signals(signal, self.config)

        # Attach extra columns (e.g. time_stop) from compute_fn
        for col_name, col_series in extras.items():
            actions = actions.with_columns(col_series.alias(col_name))

        return actions


@dataclass
class BooleanSignalPipeline:
    """Pipeline for strategies that already produce boolean entry/exit arrays."""

    name: str
    trade_def: TradeDef
    compute_fn: Callable[[pl.DataFrame], pl.DataFrame]
    config: SignalConfig = field(default_factory=SignalConfig)

    def run(self, data: pl.DataFrame) -> pl.DataFrame:
        return generate_boolean_actions(self.compute_fn(data))


# ── standard exit recipes ────────────────────────────────────────────────────

def profit_target(pct: float = 0.5) -> Callable:
    """Exit when signal has reverted `pct` of its entry value back toward zero.

    Works on any z-score signal. For an entry at z=1.5 with pct=0.5,
    exits when z drops to 0.75.
    """
    def fn(pos: Position, bar: dict) -> Optional[str]:
        target = pos.entry_signal * (1.0 - pct)
        if pos.direction == 1 and bar["signal"] >= target:
            return "profit_target"
        if pos.direction == -1 and bar["signal"] <= target:
            return "profit_target"
    return fn


def half_drift_residual(frac: float = 0.5) -> Callable:
    """Exit when residual has reverted `frac` of the way from entry to OU mean.

    Requires compute_fn to pass 'resid' and 'ou_mean' as extra columns so
    they are captured in pos.entry_extras at entry and available in bar at exit.

    frac=0.5 is the half-drift target. Lower values take profit earlier.
    """
    def fn(pos: Position, bar: dict) -> Optional[str]:
        if "resid" not in bar or "resid" not in pos.entry_extras:
            return None
        mu          = pos.entry_extras.get("ou_mean", 0.0)
        entry_resid = pos.entry_extras["resid"]
        target      = mu + (entry_resid - mu) * (1.0 - frac)
        if pos.direction == -1 and bar["resid"] <= target:
            return f"resid_target_{frac:.2f}"
        if pos.direction == 1 and bar["resid"] >= target:
            return f"resid_target_{frac:.2f}"
    return fn


# ── sizing ───────────────────────────────────────────────────────────────────
#
# All sizing functions return dict[str, float] mapping instrument -> notional size.


@dataclass
class DV01Map:
    """DV01 (dollar value of a basis point) per instrument.

    For quick approximation, use .from_tenors() which estimates
    DV01 ~ modified_duration * 100 (per $1M notional).
    For actual bonds, populate from Bloomberg or database.
    """

    dv01: dict[str, float]

    @classmethod
    def from_tenors(cls, tenors: dict[str, float]) -> DV01Map:
        """Quick approx: DV01 ~ tenor_years * 100 ($ per bp per $1M notional).

        Args:
            tenors: e.g. {"2Y": 2.0, "5Y": 5.0, "10Y": 10.0, "30Y": 30.0}
        """
        return cls({k: v * 100 for k, v in tenors.items()})


def size_dv01_neutral(
    trade_def: TradeDef,
    dv01_map: DV01Map,
    target_risk_bps: float = 100.0,
) -> dict[str, float]:
    """Scale per-leg sizes so net DV01 exposure is zero.

    For outrights, just scale to target_risk_bps.
    For spreads/flies, sizes are inversely proportional to DV01.
    """
    legs = trade_def.legs
    instruments = list(legs.keys())
    weights = np.array([legs[k] for k in instruments])
    dv01s = np.array([dv01_map.dv01.get(k, 1.0) for k in instruments])

    if len(instruments) == 1:
        return {instruments[0]: target_risk_bps / dv01s[0]}

    # Size inversely proportional to DV01 per leg
    risk_per_leg = np.abs(weights) * dv01s
    base_size = 1.0 / risk_per_leg
    base_size = base_size / base_size[0]

    total_risk_per_unit = np.sum(np.abs(weights) * dv01s * base_size)
    scale = target_risk_bps / total_risk_per_unit if total_risk_per_unit > 0 else 1.0

    return {k: base_size[i] * scale for i, k in enumerate(instruments)}


def size_beta_weighted(
    trade_def: TradeDef,
    beta: float,
    dv01_map: DV01Map,
    target_risk_bps: float = 100.0,
) -> dict[str, float]:
    """Beta-weighted sizing: adjust one leg by regression beta.

    For a spread y ~ beta*x, the x-leg weight is scaled by |beta|
    to match the empirical relationship before DV01 neutralization.
    """
    legs = trade_def.legs
    instruments = list(legs.keys())

    if len(instruments) != 2:
        raise ValueError("Beta-weighted sizing requires exactly 2 legs")

    adjusted_legs = {
        instruments[0]: legs[instruments[0]] * abs(beta),
        instruments[1]: legs[instruments[1]],
    }

    adjusted_trade = TradeDef(
        name=trade_def.name + "_beta",
        legs=adjusted_legs,
        trade_type=trade_def.trade_type,
    )

    return size_dv01_neutral(adjusted_trade, dv01_map, target_risk_bps)


def size_custom(
    custom_weights: dict[str, float],
    target_risk_bps: float = 100.0,
) -> dict[str, float]:
    """Pass-through for fully custom sizing, normalized to target risk."""
    total = sum(abs(v) for v in custom_weights.values())
    scale = target_risk_bps / total if total > 0 else 1.0
    return {k: v * scale for k, v in custom_weights.items()}


# ── the engine ───────────────────────────────────────────────────────────────


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
        # Date filtering — parse ISO strings so they compare against Date columns
        def _as_date(d):
            import datetime as _dt
            return _dt.date.fromisoformat(d) if isinstance(d, str) else d

        df = data
        if self.config.start_date:
            df = df.filter(pl.col(dates_col) >= _as_date(self.config.start_date))
        if self.config.end_date:
            df = df.filter(pl.col(dates_col) <= _as_date(self.config.end_date))

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
        bool_arrays: dict[str, dict[str, np.ndarray]] = {}
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
            bool_cols = {"enter_long", "enter_short", "exit_long", "exit_short"}
            bool_arrays[name] = {
                c: sf[c].to_numpy()
                for c in bool_cols
                if c in sf.columns
            }
            for c in bool_cols:
                bool_arrays[name].setdefault(c, np.zeros(len(sf), dtype=bool))
            lvl_arrays[name] = composite_levels[name].to_numpy()
            ts_arrays[name] = sf["time_stop"].to_numpy() if "time_stop" in sf.columns else None
            sz_arrays[name] = sf["size"].to_numpy() if "size" in sf.columns else None
            reserved = {"signal", "action", "time_stop", "size", *bool_cols}
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
                bools = bool_arrays[name]

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
                    if pos.direction == 1 and (action == "exit_long" or bools["exit_long"][i]):
                        exit_reason = "signal"
                    elif pos.direction == -1 and (action == "exit_short" or bools["exit_short"][i]):
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
                        exit_extras = {
                            c: float(arr[i])
                            for c, arr in extras_arrays[name].items()
                        }
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
                                entry_extras=pos.entry_extras,
                                exit_extras=exit_extras,
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
                    if action == "enter_long" or bools["enter_long"][i]:
                        direction = 1
                    elif action == "enter_short" or bools["enter_short"][i]:
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


# ── metrics & reporting ──────────────────────────────────────────────────────
#
# Performance analytics — Sharpe, drawdown, hit rate, PnL decomposition.


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


# ── spread book ──────────────────────────────────────────────────────────────
#
# Cross-sectional spread portfolio — rank signals, allocate risk. The big
# Sharpe multiplier: instead of trading one spread, score ALL spreads each
# day by |z| × confidence, allocate to top-K with risk parity (inverse-vol
# sizing). Diversification is mechanical Sharpe improvement.
#
# Usage:
#     book = SpreadBook(BookConfig(max_spreads=3))
#     book.add_spread("2s10s",  gen_x="gen_2Y",  gen_y="gen_2s10s",
#                      otr_short="otr_2Y", otr_long="otr_10Y")
#     book.add_spread("10s30s", gen_x="gen_10Y", gen_y="gen_1030",
#                      otr_short="otr_10Y", otr_long="otr_30Y")
#     result = book.run(data, lookback=50, z_entry=2.0)


@dataclass
class SpreadDef:
    """One spread to include in the cross-sectional book."""

    name: str
    gen_x: str       # generic column for regression x (e.g. "gen_10Y")
    gen_y: str       # generic column for regression y (e.g. "gen_1030")
    otr_short: str   # OTR column for short leg PnL (e.g. "otr_10Y")
    otr_long: str    # OTR column for long leg PnL  (e.g. "otr_30Y")
    dv01_short: float = 1.0
    dv01_long: float = 1.0


@dataclass
class BookConfig:
    """Configuration for the cross-sectional spread book."""

    max_spreads: int = 3
    allocation: str = "risk_parity"   # "risk_parity", "equal_weight", "rank_weighted"
    total_risk_budget: float = 1.0
    transaction_cost_bps: float = 0.1
    slippage_bps: float = 0.0


def _compute_spread_signals(
    data: pl.DataFrame, gen_x: str, gen_y: str,
    lookback: int, z_entry: float,
) -> pl.DataFrame:
    """Compute signal, time_stop, size, confidence, vol for one spread."""
    from stats import roll_lr, ou_zscore, roll_half_life

    reg = roll_lr(data[gen_x], data[gen_y], lookback=lookback)
    z = ou_zscore(reg["resid"], lookback=lookback)
    rhl = roll_half_life(reg["resid"], lookback=lookback)

    # Confidence: R² × beta stability
    r2 = reg["r2"].fill_null(0).clip(0, 1)
    beta_cv = (
        reg["beta"].rolling_std(lookback)
        / reg["beta"].rolling_mean(lookback).abs()
    ).fill_null(2.0).clip(0, 2)
    conf = (r2 * (1 - beta_cv / 2)).clip(0.05, 1.0)

    # Vol scaling
    resid_vol = reg["resid"].diff().rolling_std(20)
    vol_target = resid_vol.rolling_mean(252)
    vol_scale = (vol_target / resid_vol).fill_null(1.0).clip(0.2, 5.0)

    # Continuous sizing
    raw_size = z.abs() / z_entry
    size = (raw_size * vol_scale * conf).clip(0.1, 3.0)

    out = pl.DataFrame({
        "signal": z, "time_stop": rhl, "size": size,
        "confidence": conf, "vol": resid_vol,
    })

    # Suppress when not mean-reverting
    return out.with_columns(
        pl.when(pl.col("time_stop").is_not_null())
        .then(pl.col("signal"))
        .otherwise(None)
        .alias("signal")
    )


class SpreadBook:
    """Cross-sectional spread portfolio with ranking and allocation."""

    def __init__(self, config: BookConfig = None):
        self.config = config or BookConfig()
        self.spreads: list[SpreadDef] = []

    def add_spread(self, name: str, gen_x: str, gen_y: str,
                   otr_short: str, otr_long: str,
                   dv01_short: float = 1.0,
                   dv01_long: float = 1.0) -> SpreadBook:
        self.spreads.append(SpreadDef(
            name=name, gen_x=gen_x, gen_y=gen_y,
            otr_short=otr_short, otr_long=otr_long,
            dv01_short=dv01_short, dv01_long=dv01_long,
        ))
        return self

    def run(self, data: pl.DataFrame, lookback: int = 50,
            z_entry: float = 2.0) -> BacktestResult:
        """Run cross-sectional backtest.

        1. Compute signals for all spreads
        2. Cross-sectional rank by |z| × confidence
        3. Allocate top-K with risk parity
        4. Feed modified signals into Engine
        """
        names = [s.name for s in self.spreads]
        n = data.shape[0]

        # ── 1. Pre-compute raw signals for every spread ───────────────
        raw_frames: dict[str, pl.DataFrame] = {}
        for s in self.spreads:
            raw_frames[s.name] = _compute_spread_signals(
                data, s.gen_x, s.gen_y, lookback, z_entry,
            )

        # ── 2. Extract to mutable numpy for cross-sectional ranking ───
        sig = {nm: raw_frames[nm]["signal"].to_numpy() for nm in names}
        sz = {nm: raw_frames[nm]["size"].to_numpy() for nm in names}
        conf = {nm: raw_frames[nm]["confidence"].to_numpy() for nm in names}
        vol = {nm: raw_frames[nm]["vol"].to_numpy() for nm in names}

        # ── 3. Row-by-row cross-sectional ranking ─────────────────────
        for i in range(n):
            # Score each spread: |z| × confidence
            scores = {}
            for nm in names:
                s_val = sig[nm][i]
                c_val = conf[nm][i]
                if np.isnan(s_val) or np.isnan(c_val):
                    continue
                scores[nm] = abs(s_val) * c_val

            if not scores:
                continue

            # Top-K selection
            ranked = sorted(scores, key=scores.get, reverse=True)
            top_k = set(ranked[:self.config.max_spreads])

            # Allocate risk budget
            if self.config.allocation == "risk_parity":
                vols = {}
                for nm in top_k:
                    v = vol[nm][i]
                    vols[nm] = v if not np.isnan(v) and v > 0 else 1.0
                inv_vol_sum = sum(1.0 / v for v in vols.values())
                weights = {nm: (1.0 / vols[nm]) / inv_vol_sum for nm in top_k}
            elif self.config.allocation == "rank_weighted":
                total_score = sum(scores[nm] for nm in top_k)
                weights = {nm: scores[nm] / total_score for nm in top_k} if total_score > 0 else {nm: 1.0 / len(top_k) for nm in top_k}
            else:  # equal_weight
                weights = {nm: 1.0 / len(top_k) for nm in top_k}

            budget = self.config.total_risk_budget

            # Apply: scale top-K sizes, suppress non-top-K signals
            for nm in names:
                if nm in top_k:
                    if not np.isnan(sz[nm][i]):
                        sz[nm][i] *= weights[nm] * budget
                else:
                    sig[nm][i] = np.nan  # engine skips NaN signals

        # ── 4. Rebuild modified signal frames ─────────────────────────
        modified_frames: dict[str, pl.DataFrame] = {}
        for nm in names:
            rf = raw_frames[nm]
            modified_frames[nm] = rf.with_columns(
                pl.Series("signal", sig[nm]),
                pl.Series("size", sz[nm]),
            )

        # ── 5. Build Engine with wrapper pipelines ────────────────────
        dv01_map = {}
        for s in self.spreads:
            dv01_map[s.otr_short] = s.dv01_short
            dv01_map[s.otr_long] = s.dv01_long

        engine = Engine(BacktestConfig(
            transaction_cost_bps=self.config.transaction_cost_bps,
            slippage_bps=self.config.slippage_bps,
            max_total_positions=self.config.max_spreads * 2,
            dv01_map=DV01Map(dv01=dv01_map) if dv01_map else None,
        ))

        # Create pipelines whose compute_fn returns the pre-computed frame.
        # Pipeline.run() will extract "signal", run generate_signals(), then
        # re-attach extras (time_stop, size, etc.). This is correct because
        # we modified the signal column to suppress non-top-K entries.
        for s in self.spreads:
            frame = modified_frames[s.name]

            def _make_fn(f):
                def fn(d):
                    return f
                return fn

            trade_def = TradeDef.spread(s.name, s.otr_short, s.otr_long)
            pipeline = SignalPipeline(
                name=s.name,
                trade_def=trade_def,
                compute_fn=_make_fn(frame),
                config=SignalConfig(
                    entry_long=-z_entry,
                    entry_short=z_entry,
                    exit_long=None,
                    exit_short=None,
                    time_stop_bars=None,
                    max_positions=1,
                ),
            )
            engine.add_signal(pipeline)

        return engine.run(data)
