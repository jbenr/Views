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
    held_residual,
    parse_derived,
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


def test_parse_derived_canonicalises_and_rejects_unknown_series():
    assert parse_derived(" 10y ~ 2y ").name == "10y~2y@126"
    fly = parse_derived("20y ~ 10y + 30y @ 63")
    assert (fly.dependent, fly.hedges, fly.lookback) == ("20y", ("10y", "30y"), 63)
    with pytest.raises(ValueError, match="unknown series"):
        parse_derived("10y ~ nope")
    with pytest.raises(ValueError, match="must look like"):
        parse_derived("10y")


def test_held_residual_is_the_unhedged_move_with_yesterdays_beta():
    rng = np.random.default_rng(3)
    n = 400
    hedge = 400 + np.cumsum(rng.normal(size=n))
    noise = rng.normal(scale=0.1, size=n)
    dep = 0.7 * hedge + np.cumsum(noise)
    data = pl.DataFrame({
        "ts": [dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(n)],
        "10y": dep, "2y": hedge,
    })
    feature = held_residual(data, parse_derived("10y ~ 2y @ 63"))["10y~2y@63"].to_numpy()

    moves = np.diff(feature)[100:]
    # beta is estimated with noise, so moves track the true residual closely
    assert np.corrcoef(moves, noise[101:])[0, 1] > 0.95
    assert np.isnan(feature[:62]).all() and feature[63] == 0.0
    assert held_residual(data, parse_derived("10y ~ 2y @ 63"))["10y~2y@63"].null_count() == 63


def test_held_residual_has_no_phantom_move_when_beta_changes():
    # dep moves exactly beta_t * hedge move with a drifting beta; the held
    # residual stays near zero, while a re-marked level would jump each day.
    n = 300
    hedge_moves = np.where(np.arange(n) % 2, 1.0, -1.0)
    hedge = 400 + np.cumsum(hedge_moves)
    beta = np.linspace(0.5, 0.9, n)
    dep = 100 + np.cumsum(beta * hedge_moves)
    data = pl.DataFrame({
        "ts": [dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(n)],
        "10y": dep, "2y": hedge,
    })
    feature = held_residual(data, parse_derived("10y ~ 2y @ 20"))["10y~2y@20"].to_numpy()
    assert np.nanmax(np.abs(np.diff(feature[25:]))) < 0.05


def test_needed_sources_loads_derived_components_not_the_derived_name():
    yields, exo, vols, composites = _needed_sources(
        CATALOG["10s30s"], ["swsp10~10y@126", "20y~10y+30y@126"]
    )
    assert set(yields) == {"10y", "20y", "30y"}
    assert exo == ["swsp10"] and vols == [] and composites == []


def test_single_instrument_target_refuses_beta_weighting_clearly(monkeypatch):
    import research.panel as panel_module
    monkeypatch.setattr(panel_module, "load_wide", lambda *a, **k: pytest.fail("must refuse before loading data"))
    with pytest.raises(ValueError, match=r"single series with no second leg"):
        panel_module.build_panel(CATALOG["10y"], ["2y"], weighting="beta")


def test_beta_weighted_swap_spread_is_built_from_swap_and_treasury_legs(monkeypatch):
    import research.panel as panel_module
    n = 300
    rng = np.random.default_rng(9)
    tsy = 400 + np.cumsum(rng.normal(size=n))
    swap = tsy - 40 + 0.05 * (tsy - 400) + np.cumsum(rng.normal(scale=0.2, size=n))
    frame = pl.DataFrame({"ts": [dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(n)],
                          "sofr10": swap, "10y": tsy})
    monkeypatch.setattr(panel_module, "load_wide", lambda tickers, **kw: frame.select("ts", *tickers))
    panel = panel_module.build_panel(CATALOG["swsp10"], [], weighting="beta", beta_lookback=126)
    assert panel.target.name == "swsp10_legs" and panel.target.legs == {"sofr10": 1.0, "10y": -1.0}
    assert dependent_leg(panel.target) == "sofr10" and "w_10y" in panel.data.columns
    assert panel.data["w_10y"].drop_nulls().median() == pytest.approx(1.05, abs=0.05)
    fixed = panel_module.beta_package(CATALOG["10s30s"])
    assert fixed is CATALOG["10s30s"]  # packages pass through unchanged


def test_target_also_chosen_as_a_feature_is_listed_once():
    panel = Panel(data=pl.DataFrame({"ts": [dt.date(2020, 1, 1)], "swsp10": [1.0]}),
                  target=CATALOG["swsp10"], features=("swsp10",))
    assert panel.columns == ["swsp10"]


def test_swap_spread_can_be_built_from_its_legs_and_beta_weighted():
    trade = parse_weights("sofr10:1, 10y:-1")
    assert dependent_leg(trade) == "sofr10"
    n = 300
    rng = np.random.default_rng(4)
    tsy = 400 + np.cumsum(rng.normal(size=n))
    swap = tsy - 40 + np.cumsum(rng.normal(scale=0.2, size=n)) + 0.1 * (tsy - 400)  # swaps move ~1.1x Treasuries
    data = pl.DataFrame({"ts": [dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(n)], "sofr10": swap, "10y": tsy})
    fitted = beta_weighted(data, trade, lookback=126).drop_nulls()
    assert fitted["w_10y"].median() == pytest.approx(1.1, abs=0.05)
    # each day's move is the swap move net of yesterday's beta x Treasury move
    joined = data.join(fitted, on="ts")
    held = np.diff(joined["sofr10"].to_numpy()) - joined["w_10y"].to_numpy()[:-1] * np.diff(joined["10y"].to_numpy())
    np.testing.assert_allclose(np.diff(fitted["custom"].to_numpy()), held)


def test_beta_weighted_target_has_no_phantom_move_when_beta_changes():
    # dep moves exactly beta_t x hedge move with a drifting beta on a 400bp
    # hedge: a re-marked level would jump ~0.5bp a day, the held target stays flat.
    n = 300
    hedge_moves = np.where(np.arange(n) % 2, 1.0, -1.0)
    beta = np.linspace(0.5, 0.9, n)
    data = pl.DataFrame({
        "ts": [dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(n)],
        "sofr10": 100 + np.cumsum(beta * hedge_moves), "10y": 400 + np.cumsum(hedge_moves),
    })
    fitted = beta_weighted(data, parse_weights("sofr10:1, 10y:-1"), lookback=20)
    level = fitted["custom"].to_numpy()
    assert np.nanmax(np.abs(np.diff(level[25:]))) < 0.05
    remarked = data["sofr10"].to_numpy() - fitted["w_10y"].to_numpy() * data["10y"].to_numpy()
    assert np.nanmax(np.abs(np.diff(remarked[25:]))) > 0.3


def test_fit_lr_recovers_a_line_and_ignores_missing_rows():
    from stats import fit_lr
    x = np.arange(50, dtype=float)
    y = 2.0 + 0.5 * x
    y[3] = np.nan
    fit = fit_lr(x, y)
    assert fit["beta"] == pytest.approx(0.5) and fit["alpha"] == pytest.approx(2.0)
    assert fit["r2"] == pytest.approx(1.0) and fit["n"] == 49


def test_regression_scatter_fits_latest_window_and_deeper_past_separately():
    import json
    from plotly.utils import PlotlyJSONEncoder
    from research.app import regression_scatter, scatter_fits
    n = 300
    x = np.linspace(0, 10, n)
    y = np.where(np.arange(n) < n - 50, 1.0 * x, 3.0 * x)  # the relationship steepens in the last 50 days
    data = pl.DataFrame({"ts": [dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(n)], "y": y, "x": x})
    fits = scatter_fits(data, "y", "x", "levels", 50, "previous")
    assert fits["fits"]["latest"]["beta"] == pytest.approx(3.0) and fits["fits"]["past"]["beta"] == pytest.approx(1.0)
    assert len(fits["latest"]) == len(fits["deeper"]) == 50
    assert len(scatter_fits(data, "y", "x", "changes", 50, "all")["deeper"]) == n - 1 - 50
    view = json.dumps(regression_scatter(data, "y", "x", "levels", 50, "previous"), cls=PlotlyJSONEncoder, ensure_ascii=False)
    assert "data:image/png;base64," in view and "β moved 1.000 → 3.000" in view
    alone = json.dumps(regression_scatter(data, "y", "x", "changes", 50, "none"), cls=PlotlyJSONEncoder, ensure_ascii=False)
    assert "β moved" not in alone
