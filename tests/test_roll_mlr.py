"""Rolling multiple regression: correctness against a brute-force reference.

roll_mlr solves one normal-equation system per bar in a single batched call,
so the thing worth testing is that the vectorization agrees with the obvious
slow loop, that it degrades to roll_lr in the single-regressor case, and that
near-collinear regressors are reported rather than silently fitted.
"""

import numpy as np
import polars as pl
import pytest

from stats.ols import roll_lr, roll_lr_diff, roll_mlr, roll_mlr_diff


def _panel(n=400, k=2, seed=7, collinear=False):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (n, k)).cumsum(axis=0)
    if collinear:
        x[:, 1] = x[:, 0] + rng.normal(0, 1e-7, n)
    y = 0.4 * x[:, 0] - 0.9 * x[:, -1] + rng.normal(0, 0.5, n) + 3.0
    X = pl.DataFrame({f"x{i}": x[:, i] for i in range(k)})
    return X, pl.Series("y", y)


def _reference(X: pl.DataFrame, y: pl.Series, lookback: int):
    """Brute-force per-bar lstsq — the definition roll_mlr must reproduce."""
    xa, ya = X.to_numpy(), y.to_numpy()
    n, k = xa.shape
    out = np.full((n, k + 1), np.nan)
    for t in range(lookback - 1, n):
        w = slice(t - lookback + 1, t + 1)
        design = np.column_stack([np.ones(lookback), xa[w]])
        out[t] = np.linalg.lstsq(design, ya[w], rcond=None)[0]
    return out


def test_matches_brute_force_reference():
    X, y = _panel(n=300, k=3)
    got = roll_mlr(X, y, lookback=60)
    want = _reference(X, y, 60)
    cols = ["alpha"] + [f"beta_{c}" for c in X.columns]
    np.testing.assert_allclose(
        got.select(cols).to_numpy(), want, rtol=1e-6, atol=1e-6
    )


def test_single_regressor_reduces_to_roll_lr():
    X, y = _panel(n=300, k=1)
    multi = roll_mlr(X, y, lookback=50)
    uni = roll_lr(X["x0"], y, lookback=50)
    np.testing.assert_allclose(
        multi["beta_x0"].to_numpy(), uni["beta"].to_numpy(),
        rtol=1e-8, atol=1e-8, equal_nan=True,
    )
    for col in ("alpha", "yhat", "resid", "r2"):
        np.testing.assert_allclose(
            multi[col].to_numpy(), uni[col].to_numpy(),
            rtol=1e-8, atol=1e-8, equal_nan=True,
        )


def test_diff_variant_reduces_to_roll_lr_diff():
    X, y = _panel(n=300, k=1)
    multi = roll_mlr_diff(X, y, lookback=50)
    uni = roll_lr_diff(X["x0"], y, lookback=50)
    assert len(multi) == len(uni)
    for got, want in (("beta_x0", "beta"), ("resid", "resid"),
                      ("resid_cum", "resid_cum"), ("r2", "r2")):
        np.testing.assert_allclose(
            multi[got].to_numpy(), uni[want].to_numpy(),
            rtol=1e-8, atol=1e-8, equal_nan=True,
        )


def test_warmup_is_null_and_nothing_leaks_backwards():
    X, y = _panel(n=200, k=2)
    got = roll_mlr(X, y, lookback=60)
    assert got["beta_x0"][:59].null_count() == 59
    assert got["beta_x0"][59] is not None
    # prefix stability: appending future rows cannot change an emitted bar
    longer = roll_mlr(X, y, lookback=60)
    trimmed = roll_mlr(X.head(150), y.head(150), lookback=60)
    np.testing.assert_allclose(
        longer["beta_x0"].to_numpy()[:150], trimmed["beta_x0"].to_numpy(),
        rtol=1e-9, atol=1e-9, equal_nan=True,
    )


def test_recovers_known_coefficients():
    rng = np.random.default_rng(11)
    n = 3000
    x = rng.normal(0, 1, (n, 2))
    y = 1.5 * x[:, 0] - 0.75 * x[:, 1] + 2.0 + rng.normal(0, 0.05, n)
    X = pl.DataFrame({"a": x[:, 0], "b": x[:, 1]})
    got = roll_mlr(X, pl.Series("y", y), lookback=1000).tail(1)
    assert got["beta_a"][0] == pytest.approx(1.5, abs=0.02)
    assert got["beta_b"][0] == pytest.approx(-0.75, abs=0.02)
    assert got["alpha"][0] == pytest.approx(2.0, abs=0.02)
    assert got["r2"][0] > 0.99


def test_collinear_regressors_are_nulled_not_fitted():
    X, y = _panel(n=300, k=2, collinear=True)
    got = roll_mlr(X, y, lookback=60)
    fitted = got["beta_x0"].drop_nulls()
    assert got["cond"].drop_nulls().max() > 1e10
    # the point: no confidently-wrong betas survive an unidentified system
    assert len(fitted) == 0


def test_factor_condition_number_is_unit_invariant():
    """Collinearity is about factor overlap, not whether a column uses bps."""
    X, y = _panel(n=300, k=2)
    scaled = X.with_columns((pl.col("x0") * 1_000_000).alias("x0"))
    base = roll_mlr(X, y, lookback=60)
    changed_units = roll_mlr(scaled, y, lookback=60)
    assert changed_units["cond"].drop_nulls()[-1] > base["cond"].drop_nulls()[-1] * 1e8
    np.testing.assert_allclose(
        base["factor_cond"].drop_nulls().to_numpy(),
        changed_units["factor_cond"].drop_nulls().to_numpy(),
        rtol=1e-10,
    )


def test_raising_max_cond_returns_fits_but_only_their_sum_is_identified():
    """With two near-identical regressors the individual loadings are
    arbitrary -- only their combined effect is pinned down. Raising max_cond
    yields numbers instead of nulls, and this is what those numbers are worth.
    """
    X, y = _panel(n=300, k=2, collinear=True)
    permissive = roll_mlr(X, y, lookback=60, max_cond=np.inf)
    b0 = permissive["beta_x0"].drop_nulls().to_numpy()
    b1 = permissive["beta_x1"].drop_nulls().to_numpy()
    assert len(b0) > 0
    # y was built with 0.4*x0 - 0.9*x1 and x1 == x0, so the net loading is -0.5
    assert (b0 + b1).mean() == pytest.approx(-0.5, abs=0.03)
    # neither individual coefficient recovers its generating value
    assert abs(b0.mean() - 0.4) > 0.2


def test_underdetermined_window_is_refused():
    X, y = _panel(n=100, k=3)
    with pytest.raises(ValueError, match="underdetermined"):
        roll_mlr(X, y, lookback=4, min_periods=3)


def test_empty_regressor_frame_is_refused():
    _, y = _panel(n=50, k=1)
    with pytest.raises(ValueError, match="at least one regressor"):
        roll_mlr(pl.DataFrame(), y, lookback=10)


def test_accepts_pandas_input():
    X, y = _panel(n=200, k=2)
    got = roll_mlr(X.to_pandas(), y.to_pandas(), lookback=50)
    want = roll_mlr(X, y, lookback=50)
    np.testing.assert_allclose(
        got["beta_x1"].to_numpy(), want["beta_x1"].to_numpy(),
        rtol=1e-9, atol=1e-9, equal_nan=True,
    )
