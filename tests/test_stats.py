"""Core stats helpers on synthetic data — no DB required."""

import numpy as np
import polars as pl
import pytest

from stats import (
    beta_cv,
    fit_pca,
    half_life,
    horizon_backtest,
    hurst_exponent,
    ou_params,
    quality_weight,
    roll_lr,
    roll_lr_diff,
    roll_ou_features,
    roll_ou_zscore,
    roll_pc1_score,
)


def test_roll_pc1_score_tracks_common_level_factor():
    rng = np.random.default_rng(5)
    n, lb = 1200, 252
    level = np.cumsum(rng.normal(0.0, 2.0, n))
    panel = pl.DataFrame({
        "2y": 150 + level + np.cumsum(rng.normal(0, 0.5, n)),
        "5y": 250 + level + np.cumsum(rng.normal(0, 0.5, n)),
        "10y": 350 + level + np.cumsum(rng.normal(0, 0.5, n)),
        "30y": 400 + level + np.cumsum(rng.normal(0, 0.5, n)),
    })
    score = roll_pc1_score(panel, lookback=lb)

    assert len(score) == n
    assert score[: lb - 1].is_null().all()  # warmup
    assert score.slice(lb).is_not_null().all()

    # sign-fixed PC1 of a yield panel is the level factor: its changes must
    # track the average yield change POSITIVELY (no window-to-window flips)
    avg_chg = np.diff(panel.mean_horizontal().to_numpy())[lb:]
    pc1_chg = np.diff(score.to_numpy())[lb:]
    corr = np.corrcoef(avg_chg, pc1_chg)[0, 1]
    assert corr > 0.95


def make_ou(n=2000, theta=0.05, sigma=1.0, mu=0.0, seed=1):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = x[i - 1] + theta * (mu - x[i - 1]) + rng.normal(0, sigma)
    return pl.Series(x)


def test_roll_lr_recovers_known_relationship():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, 1000).cumsum()
    y = 3.0 + 2.0 * x + rng.normal(0, 0.01, 1000)
    reg = roll_lr(pl.Series(x), pl.Series(y), lookback=100)

    tail = reg.drop_nulls()
    assert tail["beta"].mean() == pytest.approx(2.0, abs=0.01)
    assert tail["alpha"].mean() == pytest.approx(3.0, abs=0.1)
    assert tail["r2"].min() > 0.99
    assert abs(tail["resid"].mean()) < 0.01


def test_roll_lr_diff_hedge_ratio():
    rng = np.random.default_rng(3)
    dx = rng.normal(0, 1, 1500)
    x = dx.cumsum()
    y = 0.7 * x + rng.normal(0, 0.05, 1500).cumsum()
    reg = roll_lr_diff(pl.Series(x), pl.Series(y), lookback=100)

    assert reg.columns == ["x", "y", "dx", "dy", "alpha", "beta", "yhat", "resid", "resid_cum", "r2"]
    assert reg.drop_nulls()["beta"].mean() == pytest.approx(0.7, abs=0.05)


def test_half_life_and_ou_params_on_simulated_ou():
    s = make_ou(theta=0.05)
    hl = half_life(s)
    # theoretical half-life = ln(2)/theta ≈ 13.9 days
    assert 8 < hl < 25

    p = ou_params(s)
    assert p["theta"] == pytest.approx(0.05, abs=0.03)
    assert p["half_life"] == pytest.approx(hl)


def test_half_life_useless_on_trending_series():
    # a pure trend has no mean reversion: half_life must be nan or non-tradable
    hl = half_life(pl.Series(np.linspace(0, 100, 500)))
    assert not (0 < hl < 1000)


def test_roll_ou_zscore_flags_dislocations():
    s = make_ou()
    z = roll_ou_zscore(s, lookback=252)
    z_clean = z.drop_nulls()
    assert len(z_clean) > 1000
    # a stationary series should spend most time inside ±3
    assert (z_clean.abs() < 3).mean() > 0.95




def test_roll_ou_features_returns_aligned_state_frame():
    s = make_ou(n=800, theta=0.08, sigma=0.7)
    out = roll_ou_features(s, lookback=160)

    assert out.columns == [
        "ou_z",
        "ou_mean",
        "ou_sigma",
        "ou_rho",
        "ou_theta",
        "expected_delta_1d",
        "half_life",
    ]
    assert len(out) == len(s)

    tail = out.tail(200)
    z = tail["ou_z"].to_numpy()
    half_life = tail["half_life"].to_numpy()
    theta = tail["ou_theta"].to_numpy()
    expected = tail["expected_delta_1d"].to_numpy()
    level_minus_mean = s.tail(200).to_numpy() - tail["ou_mean"].to_numpy()

    assert np.isfinite(z).mean() > 0.9
    assert np.isfinite(half_life).mean() > 0.9

    mask = (
        np.isfinite(theta)
        & np.isfinite(expected)
        & np.isfinite(level_minus_mean)
        & (np.abs(level_minus_mean) > 1e-9)
    )
    assert mask.sum() > 100
    assert (
        np.sign(expected[mask]) == -np.sign(level_minus_mean[mask])
    ).mean() > 0.95
def test_hurst_exponent_regimes():
    rng = np.random.default_rng(4)
    mean_reverting = make_ou(theta=0.2)

    # trending = positively autocorrelated increments (AR(1) momentum)
    dx = np.zeros(2000)
    shocks = rng.normal(0, 1, 2000)
    for i in range(1, 2000):
        dx[i] = 0.6 * dx[i - 1] + shocks[i]
    trending = pl.Series(np.cumsum(dx))

    assert hurst_exponent(mean_reverting) < 0.5
    assert hurst_exponent(trending) > 0.5


def test_hurst_exponent_short_series_is_nan():
    assert np.isnan(hurst_exponent(pl.Series([1.0, 2.0, 3.0])))


def test_fit_pca_level_factor_dominates():
    rng = np.random.default_rng(5)
    level = rng.normal(0, 1, 800).cumsum()
    slope = rng.normal(0, 0.2, 800).cumsum()
    curve = pl.DataFrame({
        "2y": level - slope + rng.normal(0, 0.01, 800),
        "10y": level + rng.normal(0, 0.01, 800),
        "30y": level + slope + rng.normal(0, 0.01, 800),
    })
    result = fit_pca(curve, n_components=3)
    assert result["explained_variance"][0] > 0.7
    # PC1 loadings all same sign (level factor)
    pc1 = result["loadings"]["PC1"].to_numpy()
    assert np.all(pc1 > 0) or np.all(pc1 < 0)


def test_horizon_backtest_positive_on_mean_reverting_residual():
    resid = make_ou(theta=0.1)
    out = horizon_backtest(resid, horizons=(5, 20))
    assert out.columns == ["h", "n", "ic", "hit", "sharpe"]
    assert len(out) == 2
    # fading a true OU residual must show positive IC and >50% hit
    assert (out["ic"] > 0).all()
    assert (out["hit"] > 0.5).all()


def test_horizon_backtest_too_few_obs():
    out = horizon_backtest(pl.Series([1.0, -1.0, 2.0]), horizons=(5,))
    assert out["ic"][0] is None


def test_beta_cv_and_quality_weight():
    stable = pl.Series(np.full(300, 0.8) + np.random.default_rng(6).normal(0, 0.01, 300))
    noisy = pl.Series(np.random.default_rng(7).normal(0.0, 1.0, 300))

    cv_stable = beta_cv(stable, lookback=60).drop_nulls()
    cv_noisy = beta_cv(noisy, lookback=60).drop_nulls()
    assert cv_stable.max() < 0.1
    assert cv_noisy.mean() > cv_stable.mean()
    assert cv_noisy.max() <= 2.0  # capped

    r2 = pl.Series(np.full(300, 0.5))
    w_stable = quality_weight(r2, beta_cv(stable, lookback=60)).drop_nulls()
    w_noisy = quality_weight(r2, beta_cv(noisy, lookback=60)).drop_nulls()
    assert w_stable.mean() > w_noisy.mean()
    assert w_stable.max() <= 1.0
