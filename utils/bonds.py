"""Treasury coupon-bond math: price, yield and DV01 on the street convention.

Coupons stepped back from maturity, act/act accrued, yield compounded at the coupon
frequency with a fractional first period. Prices are clean, per 100 face; yields and
coupons are in percent. `freq` is coupons a year: 2 for Treasuries (the default),
gilts, Canadas, BTPs and JGBs, 1 for Bunds and OATs.

    clean_price(4.25, date(2034, 11, 15), date(2026, 11, 12), 4.31)
    yield_to_maturity(4.25, date(2034, 11, 15), date(2026, 11, 12), 99.59)
    dv01(4.25, date(2034, 11, 15), date(2026, 11, 12), 4.31)   # price points per bp
    bond_label(4.5, date(2036, 11, 15))                        # '4.5 11/15/36'
"""

from __future__ import annotations

import calendar
import datetime as dt

import numpy as np


def bond_label(coupon: float, maturity: dt.date) -> str:
    """The street name of a Treasury, coupon then maturity: '4 11/15/36', '4.5 2/15/36'."""
    return f"{coupon:g} {maturity.month}/{maturity.day}/{maturity:%y}"


def _months_back(date: dt.date, months: int) -> dt.date:
    """`date` moved back `months` months, the day clamped to the target month's length."""
    total = date.year * 12 + date.month - 1 - months
    year, month = divmod(total, 12)
    return dt.date(year, month + 1, min(date.day, calendar.monthrange(year, month + 1)[1]))


def coupon_schedule(maturity: dt.date, settle: dt.date, freq: int = 2) -> tuple[dt.date, list[dt.date]]:
    """(the coupon date on or before settle, the coupon dates after it through maturity)."""
    ahead, hops = [], 0
    while (date := _months_back(maturity, 12 // freq * hops)) > settle:
        ahead.append(date)
        hops += 1
    return date, ahead[::-1]


def accrued(coupon: float, maturity: dt.date, settle: dt.date, freq: int = 2) -> float:
    last, ahead = coupon_schedule(maturity, settle, freq)
    return coupon / freq * (settle - last).days / (ahead[0] - last).days


def clean_price(coupon: float, maturity: dt.date, settle: dt.date, yld, freq: int = 2):
    """Clean price at a yield, or an array of prices at an array of yields."""
    last, ahead = coupon_schedule(maturity, settle, freq)
    first = (ahead[0] - settle).days / (ahead[0] - last).days  # fraction of a period to the next coupon
    periods = first + np.arange(len(ahead))
    flows = np.full(len(ahead), coupon / freq)
    flows[-1] += 100.0
    discount = (1.0 + np.asarray(yld, dtype=float)[..., None] / (100.0 * freq)) ** -periods
    return (discount * flows).sum(axis=-1) - coupon / freq * (1.0 - first)


def yield_to_maturity(coupon: float, maturity: dt.date, settle: dt.date, clean: float, freq: int = 2) -> float:
    """The yield that reprices the bond to `clean`, by Newton steps from the coupon."""
    yld = coupon or 1.0
    for _ in range(50):
        price = float(clean_price(coupon, maturity, settle, yld, freq))
        slope = (float(clean_price(coupon, maturity, settle, yld + 0.001, freq)) - price) / 0.001
        step = (price - clean) / slope
        yld -= step
        if abs(step) < 1e-10:
            break
    return yld


def dv01(coupon: float, maturity: dt.date, settle: dt.date, yld: float, freq: int = 2) -> float:
    """Price points gained per 100 face for a 1bp fall in yield (central difference)."""
    up, down = clean_price(coupon, maturity, settle, np.array([yld + 0.01, yld - 0.01]), freq)
    return float(down - up) / 2.0
