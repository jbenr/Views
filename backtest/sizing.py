"""Position sizing — DV01-neutral, beta-weighted, custom.

All sizing functions return dict[str, float] mapping instrument -> notional size.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .trades import TradeDef


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
