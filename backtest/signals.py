"""Signal interface — wraps continuous signals into discrete entry/exit actions.

Signals are just polars Series of floats (z-scores, residuals, etc.).
SignalConfig defines the thresholds. SignalPipeline ties a signal
computation function to a TradeDef and config.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Callable
import polars as pl

from .trades import TradeDef


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
