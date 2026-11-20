"""A signal's trade as Treasury futures: yield legs, then whole contracts per leg.

A signal trades a package in yield space -- its target, less beta of its input when
the setup holds the input as a hedge -- sized in dollars per bp of that package. Each
tenor leg becomes the future that tracks it, sized by that contract's DV01.
Preview only: nothing here talks to the broker.
"""
from __future__ import annotations

import itertools

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
    long it. risk is the dollars per bp the model wants on that leg, held what the whole
    contracts carry, residual the difference; signed is contracts, positive long."""
    rows = []
    for tenor, weight in legs.items():
        root = ROOTS.get(tenor)
        risk = weight * usd_per_bp  # dollars gained per bp rise in this yield
        row = dict(tenor=tenor, weight=weight, root=root, risk=risk, contract=None, dv01=None, exact=None,
                   quantity=None, signed=None, action=None, held=None, residual=None)
        if root in dv01:
            each = dv01[root]["dv01"]
            exact = -risk / each
            quantity = round(exact)
            row.update(contract=dv01[root]["contract"], dv01=each, exact=exact, quantity=abs(quantity),
                       signed=quantity, action="BUY" if quantity > 0 else "SELL" if quantity < 0 else None,
                       held=-quantity * each, residual=risk + quantity * each)
        rows.append(row)
    return rows


def spread_orders(signed: dict[str, int], dv01: dict[str, float],
                  listed: dict[str, tuple[str, str, int, int]]) -> tuple[list[dict], dict[str, int]]:
    """Cover the legs' contracts with whole exchange-listed spreads, leaving the least DV01
    to trade outright (the legging risk), then the fewest outright contracts, then the
    fewest spread orders. signed is contracts per root, positive long; listed is name ->
    (front root, back root, front ratio, back ratio), and buying one spread buys the front
    ratio and sells the back. A spread is used only where the package trades its two legs
    opposite ways and never for more than a leg holds, so the package stays exactly as
    sized. Returns the spread orders -- name, action, quantity and parts, each part a root
    and its signed contracts per spread -- and the contracts left to trade outright."""
    candidates = []
    for name, (front, back, front_ratio, back_ratio) in listed.items():
        if signed.get(front, 0) * signed.get(back, 0) < 0:
            side = 1 if signed[front] > 0 else -1
            most = min(abs(signed[front]) // front_ratio, abs(signed[back]) // back_ratio)
            if most:
                candidates.append((name, side, [(front, side * front_ratio), (back, -side * back_ratio)], most))
    best = None
    for units in itertools.product(*[range(most + 1) for *_, most in candidates]):
        left = dict(signed)
        for count, (_, _, parts, _) in zip(units, candidates):
            for root, per in parts:
                left[root] -= count * per
        if any(left[root] * signed[root] < 0 for root in left):
            continue  # a shared leg overdrawn by two spreads
        key = (sum(abs(n) * dv01[root] for root, n in left.items()), sum(abs(n) for n in left.values()),
               sum(1 for count in units if count))
        if best is None or key < best[0]:
            best = key, units, left
    if best is None:
        return [], dict(signed)
    _, units, left = best
    orders = [dict(name=name, action="BUY" if side > 0 else "SELL", quantity=count, parts=parts)
              for count, (name, side, parts, _) in zip(units, candidates) if count]
    return orders, left


def package_dv01(rows: list[dict]) -> float | None:
    """Dollars per bp of the package the whole contracts carry: the held leg risks projected
    onto the package's leg weights (least squares). None when a leg has no DV01."""
    if any(row["held"] is None for row in rows):
        return None
    return sum(row["held"] * row["weight"] for row in rows) / sum(row["weight"] ** 2 for row in rows)
