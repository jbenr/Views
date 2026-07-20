"""xy_scan explorer: pair loop, PC1 features, selection - no DB required."""

import importlib

import polars as pl
import pytest

xy = importlib.import_module("book.curve.xy_scan")


def test_synthetic_panel_has_all_columns():
    data = xy.synthetic_data(n=600)
    assert set(xy.TICKERS) <= set(data.columns)


def test_add_features_builds_derived_xs():
    data = xy.add_features(xy.synthetic_data(n=900))
    for col in ["5y5y", "5y5y_infl", *[f"pc1_{lb}" for lb in xy.PC1_LBS]]:
        assert col in data.columns, col
    # 5y5y nominal is the trader shortcut 2*10y - 5y
    row = data.row(500, named=True)
    assert row["5y5y"] == pytest.approx(2 * row["10y"] - row["5y"])
    # pc1 warms up after its lookback, then populates
    assert data[f"pc1_{min(xy.PC1_LBS)}"].slice(min(xy.PC1_LBS)).null_count() == 0


def test_xy_scan_end_to_end_synthetic(monkeypatch, tmp_path):
    monkeypatch.setattr(xy, "XY_SETUPS_FILE", tmp_path / "xy_setups.parquet")
    monkeypatch.setattr(xy, "YS", ["2s10s", "10s30s"])
    monkeypatch.setattr(xy, "XS", ["10y", "pc1_126"])
    monkeypatch.setattr(xy, "PC1_LBS", [126])
    monkeypatch.setattr(xy, "XY_BETA_LBS", [60, 120])
    monkeypatch.setattr(xy, "XY_OU_LBS", [120, 180])
    monkeypatch.setattr(xy, "XY_HORIZONS", [20, 40])
    monkeypatch.setattr(xy, "XY_MIN_ROWS", 400)

    state = xy.main(use_db=False, device="cpu")

    results = state["results"]
    assert set(results["x"]) == {"10y", "pc1_126"}
    assert set(results["y"]) == {"2s10s", "10s30s"}
    assert {"ic", "hit_rate", "fire_rate", "n_obs", "gate", "gate_bucket",
            "entry_signal"} <= set(results.columns)

    setups = state["setups"]
    assert setups.equals(pl.read_parquet(tmp_path / "xy_setups.parquet"))
    if not setups.is_empty():
        # at most TOP_N per pair, names carry the pair identity
        per_pair = setups.group_by("x", "y").agg(pl.len().alias("n"))
        assert (per_pair["n"] <= xy.XY_TOP_N_PER_PAIR).all()
        assert all("~" in n for n in setups["name"])
