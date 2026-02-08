"""Trade representation — weight vectors over instruments.

Everything (outrights, spreads, butterflies, N-leg custom) is just a
dict of {instrument: weight}. TradeDef is the immutable 'what',
Position is the live 'when/how much', ClosedPosition is the archive.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import polars as pl


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
    exit_reason: str  # "signal", "stop_loss", "time_stop", "trailing_stop"
    entry_signal: float
    exit_signal: float
    leg_sizes: dict[str, float] = field(default_factory=dict)
