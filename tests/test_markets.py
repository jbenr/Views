"""Overview headline: yields, curves and flies with their moves, and signal sections."""

import datetime as dt

import polars as pl

from dashboard import markets


def _frame():
    days = [dt.date(2025, 12, 31), dt.date(2026, 10, 5), dt.date(2026, 11, 4), dt.date(2026, 11, 5)]
    return pl.DataFrame({
        "ts": days,
        "7y": [400.0, 410.0, 420.0, 421.0],
        "10y": [420.0, 430.0, 440.0, 442.0],
        "30y": [450.0, 470.0, 480.0, 481.0],
        "10s30s": [30.0, 40.0, 40.0, 39.0],
    })


def test_board_reads_last_level_and_moves_since_the_last_bar_a_month_and_year_end():
    lines = markets.board(_frame())
    ten = next(r for r in lines["yields"] if r["name"] == "10y")
    assert ten["last"] == 4.42  # yields shown in percent
    assert (ten["d1"], ten["m1"], ten["ytd"]) == (2.0, 12.0, 22.0)  # moves in bps

    curve = next(r for r in lines["curves"] if r["name"] == "10s30s")
    assert (curve["last"], curve["d1"], curve["ytd"], curve["held"]) == (39.0, -1.0, 9.0, "long")

    fly = next(r for r in lines["flies"] if r["name"] == "7s10s30s")
    assert fly["last"] == 2 * 442 - 421 - 481  # 2*belly - wings


def test_overview_sections_follow_the_targets_shape(monkeypatch, tmp_path):
    monkeypatch.setenv("VIEWS_STORE_DIR", str(tmp_path))
    from dashboard import app

    assert app._structure({"target": "10y", "family": "duration"}) == "outright"
    assert app._structure({"target": "10s30s", "family": "curve"}) == "curve"
    assert app._structure({"target": "7s10s30s", "family": "curve"}) == "fly"
    assert app._structure({"target": "bei10", "family": "inflation"}) == "inflation"
