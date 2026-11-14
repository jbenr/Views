"""A signal's trade as Treasury futures: yield legs, then whole contracts per leg.

A signal trades a package in yield space -- its target, less beta of its input when
the setup holds the input as a hedge -- sized in dollars per bp of that package. Each
tenor leg becomes the future that tracks it, sized by that contract's DV01.
Preview only: nothing here talks to the broker.
"""
from __future__ import annotations

from research import basis_basket
from research.panel import CATALOG

# Which future stands in for each yield tenor. The classic bond (US) is the nearest
# thing to 20y; TY's deliverables sit near 7y, so 10y goes to the ultra 10.
ROOTS = {"2y": "TU", "5y": "FV", "7y": "TY", "10y": "UXY", "20y": "US", "30y": "WN"}


def yield_legs(target: str, side: int, hedge: str | None = None, beta: float = 0.0) -> dict[str, float]:
    """Bp of exposure to each tenor's yield per bp of the package, signed: positive
    gains when that yield rises. side is +1 long the package, -1 short."""
    legs = dict(CATALOG[target].legs)
    if hedge is not None:
        for tenor, weight in CATALOG[hedge].legs.items():
            legs[tenor] = legs.get(tenor, 0.0) - beta * weight
    return {tenor: side * weight for tenor, weight in legs.items() if weight}


def futures_dv01() -> dict[str, dict]:
    """Per root: its contract and that contract's DV01 in dollars, each deliverable's
    DV01 / conversion factor weighted by its probability of being cheapest to deliver
    (research.basis_basket). A root whose basket cannot be priced is left out."""
    out = {}
    for root in ROOTS.values():
        try:
            basket = basis_basket.study(root)
        except Exception:
            continue
        out[root] = dict(contract=basket["contract"], asof=basket["asof"], dv01=basket["blend_dv01"])
    return out


def contract_legs(legs: dict[str, float], usd_per_bp: float, dv01: dict[str, dict]) -> list[dict]:
    """Each yield leg as a futures order: buy contracts to be short the yield, sell to be
    long it. residual is the dollars per bp of that leg left over by whole contracts."""
    rows = []
    for tenor, weight in legs.items():
        root = ROOTS.get(tenor)
        risk = weight * usd_per_bp  # dollars gained per bp rise in this yield
        row = dict(tenor=tenor, root=root, risk=risk, contract=None, dv01=None, exact=None, quantity=None,
                   action=None, residual=None)
        if root in dv01:
            each = dv01[root]["dv01"]
            exact = -risk / each
            quantity = round(exact)
            row.update(contract=dv01[root]["contract"], dv01=each, exact=exact, quantity=abs(quantity),
                       action="BUY" if quantity > 0 else "SELL" if quantity < 0 else None,
                       residual=risk + quantity * each)
        rows.append(row)
    return rows
