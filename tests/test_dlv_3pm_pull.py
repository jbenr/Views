"""Tests for which bonds and dates the 3pm deliverable pull asks Bloomberg for."""

import datetime as dt
import sys
from pathlib import Path

import polars as pl

DATA_PULL = Path(__file__).resolve().parents[1] / "data_pull"
sys.path.insert(0, str(DATA_PULL))

import pull_dlv_3pm as dlv

D = dt.date


def _nb(rows):
    return pl.DataFrame(rows, schema={"ts": pl.Date, "contract": pl.String}, orient="row")


def _stored(rows):
    return pl.DataFrame(rows, schema={"cusip": pl.String, "have_hi": pl.Date}, orient="row")


def test_a_deliverable_is_wanted_for_the_whole_life_of_every_contract_it_is_in():
    # A is in both baskets, B only in TYH6
    nb = _nb([(D(2026, 1, 5), "TYH6"), (D(2026, 2, 27), "TYH6"), (D(2026, 3, 2), "TYM6"), (D(2026, 5, 29), "TYM6")])
    baskets = pl.DataFrame({"contract": ["TYH6", "TYH6", "TYM6"], "cusip": ["A", "B", "A"], "cf": [0.9, 0.9, 0.9]})
    windows = {r["cusip"]: (r["lo"], r["hi"]) for r in dlv.basket_windows(nb, baskets).iter_rows(named=True)}
    assert windows == {"A": (D(2026, 1, 5), D(2026, 5, 29)), "B": (D(2026, 1, 5), D(2026, 2, 27))}


def test_pulls_cover_new_bonds_fully_and_stored_bonds_only_forward():
    windows = pl.DataFrame({"cusip": ["new", "open", "done", "short"],
                            "lo": [D(2025, 1, 2), D(2026, 3, 2), D(2024, 1, 2), D(2024, 1, 2)],
                            "hi": [D(2025, 3, 31), D(2026, 5, 29), D(2024, 3, 28), D(2024, 3, 28)]})
    # "done" is missing its first days (a new issue) and its last two: neither is a gap worth filling
    stored = _stored([("open", D(2026, 5, 28)), ("done", D(2024, 3, 26)), ("short", D(2024, 2, 1))])
    todo = {r["cusip"]: (r["start"], r["end"])
            for r in dlv.pulls_needed(windows, stored, today=D(2026, 6, 1)).iter_rows(named=True)}
    assert todo == {
        "new": (D(2025, 1, 2), D(2025, 3, 31)),     # never stored: the whole window
        "open": (D(2026, 5, 28), D(2026, 6, 1)),    # still deliverable: last stored day through today
        "short": (D(2024, 2, 1), D(2024, 3, 28)),   # stopped early: resume from the last stored day
    }
