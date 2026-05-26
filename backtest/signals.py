"""Signal interface — wraps continuous signals into discrete entry/exit actions.

Signals are just polars Series of floats (z-scores, residuals, etc.).
SignalConfig defines the thresholds. SignalPipeline ties a signal
computation function to a TradeDef and config.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Callable
import polars as pl

from .trades import TradeDef, Position


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


def half_drift_residual() -> Callable:
    """Exit when the residual has reverted halfway from entry level to the OU mean.

    Requires compute_fn to pass 'resid' and 'ou_mean' as extra columns so
    they are captured in pos.entry_extras at entry and available in bar at exit.

    Grounded in OU theory: at t = half-life, E[resid] = mu + (entry_resid - mu)/2.
    """
    def fn(pos: Position, bar: dict) -> Optional[str]:
        if "resid" not in bar or "resid" not in pos.entry_extras:
            return None
        mu          = pos.entry_extras.get("ou_mean", 0.0)
        entry_resid = pos.entry_extras["resid"]
        target      = mu + (entry_resid - mu) * 0.5
        if pos.direction == -1 and bar["resid"] <= target:
            return "half_drift"
        if pos.direction == 1 and bar["resid"] >= target:
            return "half_drift"
    return fn
