"""Futures-roll basis strategy, synthetic-data only."""

import datetime as dt
import importlib

import polars as pl

from tests.synthetic import roll_panel
from utils.basis import futures_roll_panel

view = importlib.import_module("book.basis.fut_roll")


def test_futures_roll_panel_stitches_contract_pair_gaps():
    dates = [dt.date(2024, 1, d) for d in range(2, 7)]
    roll = pl.DataFrame(
        {
            "ts": dates,
            "root": ["TY"] * 5,
            "front_contract": ["TYH4", "TYH4", "TYH4", "TYM4", "TYM4"],
            "deferred_contract": ["TYM4", "TYM4", "TYM4", "TYU4", "TYU4"],
            "roll": [-1.0, -0.5, -0.25, 2.0, 1.0],
        }
    )

    panel = futures_roll_panel(roll)

    assert panel["TY_roll"].to_list() == [-1.0, -0.5, -0.25, 2.0, 1.0]
    assert panel["TY_level"].to_list() == [0.0, 0.5, 0.75, 0.75, -0.25]


def test_synthetic_roll_and_compute_schema():
    roll = roll_panel(["TY", "US"])
    panel = futures_roll_panel(roll)
    sig = view.compute(panel, "TY", params={"ou_lb": 60})

    assert {"ts", "TY_roll", "TY_level", "US_roll", "US_level"} <= set(panel.columns)
    assert set(roll["root"].unique()) == {"TY", "US"}
    assert len(sig) == len(panel)
    assert {"signal", "roll", "ou_mean", "ou_sigma", "half_life"} <= set(sig.columns)
    assert sig["signal"].drop_nulls().len() > 0


def test_pipeline_wiring():
    pipeline = view.make_pipeline("TY")

    assert pipeline.name == "fut_roll_TY"
    assert pipeline.trade_def.legs == {"TY_level": 1.0}


def test_end_to_end_synthetic(monkeypatch):
    roll = roll_panel(["TY", "US"])
    monkeypatch.setattr(
        view, "load_data", lambda *a, **k: (roll, futures_roll_panel(roll))
    )
    state = view.main(
        roots=["TY", "US"],
        params={"ou_lb": 60, "entry_z": 1.0, "exit_z": 0.2},
    )

    assert set(state) == {
        "roll",
        "panel",
        "coverage",
        "reversion",
        "result",
        "trades",
        "latest",
    }
    assert len(state["result"].closed_trades) > 0
    assert state["latest"].height == 2
