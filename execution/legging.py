"""Contingent legging: a package of futures orders worked as one lead and its hedges.

The lead -- the leg hardest to fill -- rests as a limit order. Nothing else trades until
it fills; then each hedge leg goes out at once, in proportion to the lead's fills so far,
as a marketable limit (the far touch plus a few ticks of slippage, never past the hedge's
cap). A lead that never fills leaves nothing on. These are the pure rules; the session
watches the fills and places the orders.
"""
from __future__ import annotations

import math

SLIP_TICKS = 2  # ticks through the far touch a hedge may pay to be sure of filling


def hedge_due(lead_filled: float, lead_total: float, hedge_total: int, sent: int) -> int:
    """Contracts of one hedge leg to send now: its share of the lead's fills so far
    (half-contracts round up), less what has already gone. Exact once the lead is done."""
    target = hedge_total if lead_filled >= lead_total else math.floor(hedge_total * lead_filled / lead_total + 0.5)
    return max(0, min(target, hedge_total) - sent)


def hedge_price(action: str, bid: float | None, ask: float | None, tick: float, cap: float) -> float:
    """A hedge's limit: through the far touch by SLIP_TICKS, never worse than its cap.
    With no touch to work from, the cap itself."""
    touch = ask if action == "BUY" else bid
    if touch is None:
        return cap
    through = touch + SLIP_TICKS * tick if action == "BUY" else touch - SLIP_TICKS * tick
    return min(through, cap) if action == "BUY" else max(through, cap)


def far_size(leg: dict) -> float | None:
    """What is showing at the touch a leg would trade into: the ask to buy, the bid to sell."""
    return leg.get("ask_size") if leg["action"] == "BUY" else leg.get("bid_size")


def pick_lead(legs: list[dict]) -> int:
    """The leg to lead with: the one whose size is the largest share of what is showing
    at its touch -- the hardest to fill, so the others are hedged into deeper books.
    A leg with no size showing counts as hardest."""
    def strain(leg):
        size = far_size(leg)
        return math.inf if not size else leg["quantity"] / size
    return max(range(len(legs)), key=lambda i: strain(legs[i]))


def thin_hedges(legs: list[dict], lead: int) -> list[str]:
    """A warning per hedge leg larger than the size showing at its touch: it may need more
    than one price level to fill."""
    out = []
    for i, leg in enumerate(legs):
        size = far_size(leg)
        if i != lead and size is not None and leg["quantity"] > size:
            side = "ask" if leg["action"] == "BUY" else "bid"
            out.append(f"{leg['contract']}: {leg['quantity']:,} to hedge vs {size:,.0f} at the {side}")
    return out
