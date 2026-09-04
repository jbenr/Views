"""Pure unit tests for beta-weighted research targets (no database required)."""

import datetime as dt

import numpy as np
import polars as pl
import pytest

from backtest.engine import TradeDef
from research.panel import (
    CATALOG,
    Panel,
    _needed_sources,
    beta_weighted,
    dependent_leg,
    parse_weights,
    remark_report,
)


def _exact_fly_data(n: int = 300) -> pl.DataFrame:
    """A 20Y whose daily changes are exactly half each wing's changes."""
    rng = np.random.default_rng(7)
    left = 100.0 + np.cumsum(rng.normal(size=n))
    right = 200.0 + np.cumsum(rng.normal(size=n))
    middle = 25.0 + 0.5 * left + 0.5 * right
    return pl.DataFrame(
        {
            "ts": [dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(n)],
            "left": left,
            "middle": middle,
            "right": right,
        }
    )


def test_beta_weighted_matches_fixed_fly_when_betas_are_half_each():
    data = _exact_fly_data()
    trade = TradeDef.butterfly(
        "fly", "left", "middle", "right", weights=(-1.0, 2.0, -1.0)
    )

    fitted = beta_weighted(data, trade, lookback=63).drop_nulls()
    fixed = data.select(
        "ts", (2 * pl.col("middle") - pl.col("left") - pl.col("right")).alias("fixed")
    )
    checked = fitted.join(fixed, on="ts").drop_nulls()

    assert checked["w_left"].to_list() == pytest.approx([0.5] * len(checked))
    assert checked["w_right"].to_list() == pytest.approx([0.5] * len(checked))
    assert checked["fly"].to_list() == pytest.approx(checked["fixed"].to_list())


def test_ambiguous_custom_target_requires_a_dependent_leg_override():
    trade = TradeDef("custom", {"left": 1.0, "middle": -2.0, "right": 1.0})

    with pytest.raises(ValueError, match="cannot infer"):
        dependent_leg(trade)
    assert dependent_leg(trade, "middle") == "middle"


def test_remark_report_is_populated_for_a_beta_weighted_panel():
    n = 12
    left = np.arange(n, dtype=float)
    middle = 10 + 0.4 * left
    right = 20 + 0.8 * left
    w_left = np.linspace(0.2, 0.5, n)
    w_right = np.linspace(0.8, 0.5, n)
    target = 2 * (middle - w_left * left - w_right * right)
    trade = TradeDef.butterfly(
        "fly", "left", "middle", "right", weights=(-1.0, 2.0, -1.0)
    )
    panel = Panel(
        data=pl.DataFrame(
            {
                "ts": [dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(n)],
                "fly": target,
                "left": left,
                "middle": middle,
                "right": right,
                "w_left": w_left,
                "w_right": w_right,
            }
        ),
        target=trade,
        features=(),
        weighting="beta",
    )

    report = remark_report(panel, horizons=(1,))

    assert report.shape == (1, 7)
    assert report["std_remark"][0] > 0
    assert report["remark_share_of_var"][0] > 0


def test_swap_spreads_are_executable_target_legs_and_custom_basket_legs():
    """Swap spreads are targetable; macro/vol remain explanatory-only."""
    assert CATALOG["swsp20"].legs == {"swsp20": 1.0}

    trade = parse_weights("swsp10:1, swsp20:-1", name="10s20s swap spread")
    yields, exo, vols, composites = _needed_sources(trade, [])

    assert yields == []
    assert exo == ["swsp10", "swsp20"]
    assert vols == []
    assert composites == []
    with pytest.raises(ValueError, match="unknown leg"):
        parse_weights("dxy:1")
