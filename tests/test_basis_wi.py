"""Deliverable grades, conversion factors, bonds not issued yet, and how a bond is named."""

import datetime as dt

import polars as pl
import pytest

from research import basis_basket as bb
from utils import basis
from utils.bonds import bond_label

D = dt.date


def test_bond_label_is_coupon_then_maturity():
    assert bond_label(4.0, D(2036, 11, 15)) == "4 11/15/36"
    assert bond_label(4.5, D(2036, 2, 15)) == "4.5 2/15/36"
    assert bond_label(3.875, D(2033, 8, 15)) == "3.875 8/15/33"
    assert bond_label(0.625, D(2030, 5, 15)) == "0.625 5/15/30"


def test_conversion_factor_matches_the_exchange():
    # TYZ6 and FVZ6 factors as Bloomberg publishes them
    assert basis.conversion_factor("TY", 3.875, D(2033, 8, 15), D(2026, 12, 1)) == 0.8870
    assert basis.conversion_factor("TY", 4.5, D(2033, 8, 31), D(2026, 12, 1)) == 0.9202
    assert basis.conversion_factor("FV", 3.875, D(2031, 3, 31), D(2026, 12, 1)) == 0.9212
    assert basis.conversion_factor("US", 6.0, D(2046, 12, 1), D(2026, 12, 1)) == 1.0  # a 6% bond is par at 6%


def test_deliverable_grades():
    month = D(2026, 12, 1)
    assert basis.is_deliverable("TY", D(2033, 6, 30), 84, month)          # 6y7m left
    assert not basis.is_deliverable("TY", D(2033, 5, 31), 84, month)      # under 6y6m
    assert not basis.is_deliverable("TY", D(2035, 2, 15), 120, month)     # over 8y since Sep 2023
    assert basis.is_deliverable("TY", D(2031, 11, 15), 120, D(2022, 12, 1))   # 10y cap before that
    assert basis.is_deliverable("UXY", D(2036, 5, 15), 120, month) and not basis.is_deliverable("UXY", D(2036, 2, 15), 120, month)
    assert basis.is_deliverable("TU", D(2028, 12, 31), 24, month) and not basis.is_deliverable("TU", D(2029, 1, 31), 24, month)
    assert not basis.is_deliverable("FV", D(2031, 8, 31), 84, month)      # an old 7-year is not a 5-year note
    assert basis.is_deliverable("WN", D(2056, 11, 15), 360, month) and basis.is_deliverable("US", D(2046, 8, 15), 240, month)


def _issues(rows):
    return pl.DataFrame(rows, schema={"cusip": pl.String, "term": pl.Int32, "issue_date": pl.Date,
                                      "maturity": pl.Date, "coupon": pl.Float64}, orient="row").sort("issue_date")


def test_upcoming_deliverables_steps_each_line_forward_and_keeps_month_ends():
    issues = _issues([
        ("7a", 84, D(2026, 8, 31), D(2033, 8, 31), 4.5), ("7b", 84, D(2026, 9, 30), D(2033, 9, 30), 5.0),
        ("10a", 120, D(2026, 5, 15), D(2036, 5, 15), 4.375), ("10b", 120, D(2026, 8, 17), D(2036, 8, 15), 4.625),
    ])
    got = basis.upcoming_deliverables("TY", D(2026, 12, 1), D(2026, 12, 31), D(2026, 10, 1), issues)
    # the October and November 7-years; December's is issued on the last delivery day, too late.
    # the 10-year line is too long for TY
    assert [(b["term"], b["issue_date"], b["maturity"]) for b in got] == [
        (84, D(2026, 10, 31), D(2033, 10, 31)), (84, D(2026, 11, 30), D(2033, 11, 30))]
    assert all(b["coupon"] is None and b["cusip"] is None for b in got)
    ultra = basis.upcoming_deliverables("UXY", D(2026, 12, 1), D(2026, 12, 31), D(2026, 10, 1), issues)
    assert [(b["issue_date"], b["maturity"]) for b in ultra] == [(D(2026, 11, 17), D(2036, 11, 15))]


def test_an_announced_bond_keeps_its_own_cusip_and_coupon():
    issues = _issues([
        ("7a", 84, D(2026, 8, 31), D(2033, 8, 31), 4.5), ("7b", 84, D(2026, 9, 30), D(2033, 9, 30), 5.0),
        ("7c", 84, D(2026, 11, 2), D(2033, 10, 31), 4.875),   # auctioned, settles after asof
    ])
    got = basis.upcoming_deliverables("TY", D(2026, 12, 1), D(2026, 12, 31), D(2026, 10, 28), issues)
    assert [(b["cusip"], b["coupon"], b["maturity"]) for b in got] == [
        ("7c", 4.875, D(2033, 10, 31)), (None, None, D(2033, 11, 30))]


def test_a_missing_yield_is_read_off_the_basket_slope():
    basket = pl.DataFrame({"years": [6.5, 7.0, 7.5], "yld": [5.00, 5.02, 5.04]})
    assert bb.read_off(basket, "yld", 8.0) == pytest.approx(5.06)
    assert bb.read_off(basket.head(2), "yld", 8.0) == pytest.approx(5.02)  # too few bonds for a slope: flat


def test_option_value_is_zero_for_a_bond_always_delivered_and_grows_with_the_chance_of_a_switch():
    delivery = D(2026, 12, 31)

    def basket(second_yield):
        rows = []
        for cusip, coupon, maturity, yld in (("near", 6.0, D(2034, 2, 15), 6.0), ("far", 6.0, D(2036, 11, 15), second_yield)):
            rows.append(dict(cusip=cusip, coupon=coupon, maturity=maturity, delivery=delivery, fwd_yld=yld,
                             cf=basis.conversion_factor("TY", coupon, maturity, D(2026, 12, 1)),
                             years=(maturity - delivery).days / 365.25))
        return pl.DataFrame(rows)

    moves = dict(level=30.0, curve=5.0, corr=-0.4)
    # the far bond 200bp rich: it is never cheapest, so the near bond's basis is worth nothing
    safe = bb.delivery_odds(basket(4.0), "7s10s", moves)
    assert safe["p_ctd"][0] == pytest.approx(1.0) and safe["option"][0] == pytest.approx(0.0, abs=1e-9)
    assert safe["option"][1] > 5.0
    # both at 6%, the crossover: either can be delivered, so both bases are worth something
    close = bb.delivery_odds(basket(6.0), "7s10s", moves)
    assert 0.2 < close["p_ctd"][0] < 0.8 and close["option"][0] > 0.05 and close["option"][1] > 0.05
    assert close["p_ctd"].sum() == pytest.approx(1.0)
