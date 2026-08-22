"""The three generalized research paths work on arbitrary named panels."""

import numpy as np
import polars as pl

from research import DislocationStudy, FairValueStudy, PCRelativeValueStudy, PairRVStudy


def _panel(n: int = 900, seed: int = 12) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    factor_a = np.cumsum(rng.normal(0, 1.0, n))
    factor_b = np.cumsum(rng.normal(0, 0.7, n))
    shock = rng.normal(0, 0.5, n)
    target = 25.0 + 0.4 * factor_a - 0.3 * factor_b + np.cumsum(shock)
    other = target + rng.normal(0, 0.8, n)
    return pl.DataFrame(
        {
            "ts": pl.date_range(pl.date(2020, 1, 1), pl.date(2024, 12, 31), "1d", eager=True)[:n],
            "target": target,
            "factor_a": factor_a,
            "factor_b": factor_b,
            "other": other,
        }
    )


def test_dislocation_accepts_zero_one_and_many_features():
    data = _panel()
    for features in [(), ("factor_a",), ("factor_a", "factor_b")]:
        state = DislocationStudy("target", features, beta_lookback=60, z_lookback=30).research(data)
        assert {"signals", "horizons", "events"} == set(state)
        assert "signal" in state["signals"].columns
        assert len(state["horizons"]) == 4


def test_pair_and_pc_rv_studies_are_general_named_panel_tools():
    data = _panel()
    pair = PairRVStudy("target", "other", beta_lookback=60, z_lookback=30).research(data)
    pc = PCRelativeValueStudy(
        "target", ("target", "factor_a", "factor_b"), pca_lookback=60, beta_lookback=40, z_lookback=30
    ).research(data)
    assert "rv_value" in pair["signals"].columns
    assert "pc1" in pc["signals"].columns
    assert {"half_life", "hurst"} <= set(pair["diagnostics"].columns)


def test_fair_value_supports_declared_factors_and_family_search():
    data = _panel()
    study = FairValueStudy("target", ("factor_a", "factor_b"), lookback=80)
    state = study.research(data)
    assert {"fair_value", "residual", "error_correction"} <= set(state["signals"].columns)
    search = FairValueStudy.search(
        data,
        target="target",
        factor_families={"direction": ["factor_a"], "macro": ["factor_b"]},
        lookback=80,
    )
    assert len(search) == 3
    assert {"factors", "n_factors", "ic", "latest_error_correction"} <= set(search.columns)


def test_research_signals_are_prefix_invariant():
    """Future observations cannot rewrite an earlier research signal."""
    data = _panel()
    cutoff = 700
    studies = [
        DislocationStudy("target", ("factor_a", "factor_b"), beta_lookback=60, z_lookback=30),
        PairRVStudy("target", "other", beta_lookback=60, z_lookback=30),
        FairValueStudy("target", ("factor_a", "factor_b"), lookback=80),
    ]
    for study in studies:
        full = study.compute(data).head(cutoff)
        prefix = study.compute(data.head(cutoff))
        assert full.columns == prefix.columns
        for column in full.columns:
            if full[column].dtype in (pl.Date, pl.Datetime):
                assert full[column].to_list() == prefix[column].to_list()
            else:
                np.testing.assert_allclose(
                    full[column].to_numpy(), prefix[column].to_numpy(), equal_nan=True
                )
