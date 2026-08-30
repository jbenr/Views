"""Exogenous gate columns: panel series offered to the gate search.

Covers the path a non-regression input (market vol, auction demand, issuance)
takes from the loaded panel through model_frame, the predict/exit scans, and
compute()'s gate evaluation -- including the case that motivates it, an input
whose history starts long after the model's.
"""

import numpy as np
import polars as pl
import pytest

from backtest.lab import CONDITION_NAMES, signal_matrix
from backtest.strategy import Strategy
from tests.synthetic import synthetic_pair
from utils.market_data import align_columns

TARGET, FEATURE, EXO = "10s30s", "10y", "vol"


def _panel(n: int = 900, exo_start: int = 0, seed: int = 3) -> pl.DataFrame:
    """Synthetic pair plus an exogenous column that is null before exo_start."""
    data = synthetic_pair(TARGET, FEATURE, n=n, seed=seed)
    rng = np.random.default_rng(seed)
    exo = np.abs(np.cumsum(rng.normal(0.0, 1.0, n))) + 5.0
    values = [None] * exo_start + exo[exo_start:].tolist()
    return data.with_columns(pl.Series(EXO, values, dtype=pl.Float64))


def _strategy(tmp_path, gate_columns=(EXO,), **kw) -> Strategy:
    return Strategy(
        name="exo_test",
        module="tests.test_exogenous_gates",
        path=tmp_path / "mod.py",
        tickers={FEATURE: "X Index", TARGET: "Y Index"},
        target=TARGET,
        feature=FEATURE,
        gate_columns=list(gate_columns),
        **kw,
    )


# -- align_columns -----------------------------------------------------------

def test_optional_columns_are_carried_without_truncating_the_sample():
    data = _panel(n=500, exo_start=400)
    frame = align_columns(data, [TARGET, FEATURE], optional=[EXO])
    assert EXO in frame.columns
    # the exogenous column is null for 400 bars and must not cost us those bars
    assert len(frame) == 500
    assert frame[EXO].null_count() == 400


def test_optional_columns_must_still_exist():
    data = _panel(n=200)
    with pytest.raises(ValueError, match="missing columns"):
        align_columns(data, [TARGET, FEATURE], optional=["nope"])


# -- signal_matrix -----------------------------------------------------------

def test_extra_conditions_broadcast_across_every_signal_column():
    data = _panel(n=400)
    extra = {EXO: data[EXO]}
    z, combos, conditions = signal_matrix(
        data[FEATURE], data[TARGET], [30, 60], [40, 80],
        return_conditions=True, extra_conditions=extra,
    )
    assert EXO in conditions
    assert conditions[EXO].shape == z.shape
    # exogenous by definition: identical down every (beta_lb, ou_lb) column
    assert np.array_equal(conditions[EXO][:, 0], conditions[EXO][:, -1])
    # and the built-in menu is untouched
    assert set(CONDITION_NAMES) <= set(conditions)


def test_extra_condition_may_not_shadow_a_builtin():
    data = _panel(n=300)
    with pytest.raises(ValueError, match="shadows a built-in"):
        signal_matrix(
            data[FEATURE], data[TARGET], [30], [40],
            return_conditions=True, extra_conditions={"r2": data[EXO]},
        )


def test_extra_condition_length_is_checked():
    data = _panel(n=300)
    with pytest.raises(ValueError, match="expected a 1-D series"):
        signal_matrix(
            data[FEATURE], data[TARGET], [30], [40],
            return_conditions=True,
            extra_conditions={EXO: np.ones(7)},
        )


# -- Strategy wiring ---------------------------------------------------------

def test_shadowing_gate_column_is_refused_at_construction(tmp_path):
    with pytest.raises(ValueError, match="shadow built-in"):
        _strategy(tmp_path, gate_columns=["r2"])


def test_model_or_feature_column_may_not_be_a_gate_column(tmp_path):
    with pytest.raises(ValueError, match="may not contain the model's own"):
        _strategy(tmp_path, gate_columns=[TARGET])


def test_model_frame_carries_gate_columns(tmp_path):
    strategy = _strategy(tmp_path)
    frame = strategy.model_frame(_panel(n=400, exo_start=300))
    assert frame.columns == ["ts", TARGET, FEATURE, EXO]
    assert len(frame) == 400


def test_compute_gates_on_an_exogenous_column(tmp_path):
    strategy = _strategy(tmp_path)
    data = strategy.model_frame(_panel(n=900))
    sig = strategy.compute(
        data,
        params={
            "entry_signal": "ou_z", "beta_lb": 60, "ou_lb": 120,
            "gate": (EXO, "high_75"), "gate_window": None,
        },
    )
    assert {"gate_value", "gate_percentile", "gate_allow"} <= set(sig.columns)
    assert sig["gate_value"].equals(data[EXO].rename("gate_value"))
    # a real gate both opens and closes over the sample
    allow = sig["gate_allow"].to_numpy()
    assert allow.any() and not allow.all()


def test_short_history_exogenous_gate_stays_shut_until_its_data_starts(tmp_path):
    """The case that motivates optional columns: swaption vol starts 2021 but
    the model starts 2010. The gate must be closed early, not truncate."""
    strategy = _strategy(tmp_path)
    data = strategy.model_frame(_panel(n=900, exo_start=600))
    assert len(data) == 900  # sample intact
    sig = strategy.compute(
        data,
        params={
            "entry_signal": "ou_z", "beta_lb": 60, "ou_lb": 120,
            "gate": (EXO, "high_75"), "gate_window": None,
        },
    )
    allow = sig["gate_allow"].to_numpy()
    assert not allow[:600].any()  # never fires before the series exists


def test_unknown_gate_names_report_both_menus(tmp_path):
    strategy = _strategy(tmp_path)
    data = strategy.model_frame(_panel(n=400))
    with pytest.raises(ValueError, match="unknown gate condition"):
        strategy.compute(data, params={"gate": ("nope", "high_75")})


def test_exogenous_conditions_tolerate_a_frame_missing_the_column(tmp_path):
    """A cached panel predating the gate_columns change must not hard-fail."""
    strategy = _strategy(tmp_path)
    stale = synthetic_pair(TARGET, FEATURE, n=300)
    assert strategy._exogenous_conditions(stale) == {}


def test_predict_scan_searches_exogenous_gates(tmp_path, monkeypatch):
    """End to end: the exogenous column competes as a gate in --predict."""
    panel = _panel(n=1200)
    strategy = _strategy(
        tmp_path,
        predict_entry_signals=["ou"],
        predict_beta_lbs=[40, 60],
        predict_ou_lbs=[120, 180],
        predict_horizons=[10, 20],
        predict_ou_z_thresholds=[1.5, 2.0],
        gate_windows=[252, None],
        gate_min_history=126,
    )
    monkeypatch.setattr(strategy, "load_data", lambda *a, **k: panel)
    state = strategy.predict(device="cpu")
    gates = set(state["results"]["gate"].unique().to_list())
    assert EXO in gates, f"exogenous gate absent from scan; saw {sorted(gates)}"
