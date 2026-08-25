"""Rolling OLS — no lookahead bias. Polars-native vectorized implementation."""

from __future__ import annotations
import numpy as np
import polars as pl
import pandas as pd
from typing import Union
from utils.helpers import to_pl_df, to_pl_series


def roll_lr(
    x: Union[pl.Series, pd.Series, pl.DataFrame, pd.DataFrame],
    y: Union[pl.Series, pd.Series],
    lookback: int = 100,
    min_periods: int = None,
) -> pl.DataFrame:
    """y = alpha + beta*x, rolled over trailing `lookback` rows.

    Fully vectorized via rolling sums — no Python loop.
    Returns polars DataFrame with columns: x, y, alpha, beta, yhat, resid, r2.
    """
    # Normalize inputs
    if isinstance(x, (pd.DataFrame, pl.DataFrame)):
        x = x.to_series(0) if isinstance(x, pl.DataFrame) else pl.from_pandas(x.iloc[:, 0])
    else:
        x = to_pl_series(x)
    y = to_pl_series(y)

    if min_periods is None:
        min_periods = lookback

    # Align and drop nulls
    df = pl.DataFrame({"x": x, "y": y}).drop_nulls()

    # Rolling moments
    df = df.with_columns(
        pl.col("x").rolling_sum(lookback, min_periods=min_periods).alias("sum_x"),
        pl.col("y").rolling_sum(lookback, min_periods=min_periods).alias("sum_y"),
        (pl.col("x") * pl.col("x")).rolling_sum(lookback, min_periods=min_periods).alias("sum_xx"),
        (pl.col("x") * pl.col("y")).rolling_sum(lookback, min_periods=min_periods).alias("sum_xy"),
        (pl.col("y") * pl.col("y")).rolling_sum(lookback, min_periods=min_periods).alias("sum_yy"),
        pl.col("x").is_not_null().cast(pl.Int32).rolling_sum(lookback, min_periods=min_periods).cast(pl.Float64).alias("n"),
    )

    # Beta and alpha from normal equations
    df = df.with_columns(
        (
            (pl.col("n") * pl.col("sum_xy") - pl.col("sum_x") * pl.col("sum_y"))
            / (pl.col("n") * pl.col("sum_xx") - pl.col("sum_x") ** 2)
        ).alias("beta"),
    )
    df = df.with_columns(
        ((pl.col("sum_y") - pl.col("beta") * pl.col("sum_x")) / pl.col("n")).alias("alpha"),
    )

    # Fitted values and residuals
    df = df.with_columns(
        (pl.col("alpha") + pl.col("beta") * pl.col("x")).alias("yhat"),
    )
    df = df.with_columns(
        (pl.col("y") - pl.col("yhat")).alias("resid"),
    )

    # R-squared: 1 - SS_res / SS_tot
    # SS_tot = sum_yy - sum_y^2 / n  (rolling variance of y * n)
    # SS_res = rolling sum of resid^2
    df = df.with_columns(
        (pl.col("resid") ** 2).rolling_sum(lookback, min_periods=min_periods).alias("ss_res"),
        (pl.col("sum_yy") - pl.col("sum_y") ** 2 / pl.col("n")).alias("ss_tot"),
    )
    df = df.with_columns(
        pl.when(pl.col("ss_tot") > 0)
        .then(1.0 - pl.col("ss_res") / pl.col("ss_tot"))
        .otherwise(None)
        .alias("r2"),
    )

    return df.select("x", "y", "alpha", "beta", "yhat", "resid", "r2")


def roll_lr_diff(
    x: Union[pl.Series, pd.Series, pl.DataFrame, pd.DataFrame],
    y: Union[pl.Series, pd.Series],
    lookback: int = 100,
    min_periods: int = None,
) -> pl.DataFrame:
    """Changes-based rolling OLS: dy = alpha + beta*dx.

    Returns polars DataFrame with columns:
        x, y, dx, dy, alpha, beta, yhat, resid, resid_cum, r2

    - beta is the hedge ratio on daily changes (correct for beta-weighting).
    - resid is the single-period changes residual.
    - resid_cum is the cumulative residual in level space (for z-scoring).
    """
    if isinstance(x, (pd.DataFrame, pl.DataFrame)):
        x = x.to_series(0) if isinstance(x, pl.DataFrame) else pl.from_pandas(x.iloc[:, 0])
    else:
        x = to_pl_series(x)
    y = to_pl_series(y)

    levels = pl.DataFrame({"x": x, "y": y}).drop_nulls()
    levels = levels.with_columns(
        pl.col("x").diff().alias("dx"),
        pl.col("y").diff().alias("dy"),
    )

    # drop first row (null from diff) to match roll_lr output length
    levels = levels.slice(1)

    reg = roll_lr(levels["dx"], levels["dy"], lookback=lookback, min_periods=min_periods)

    out = levels.select("x", "y").with_columns(
        reg["x"].alias("dx"),
        reg["y"].alias("dy"),
        reg["alpha"],
        reg["beta"],
        reg["yhat"],
        reg["resid"],
        reg["resid"].cum_sum().alias("resid_cum"),
        reg["r2"],
    )
    return out


def _roll_sum(a: np.ndarray, lookback: int, min_periods: int) -> np.ndarray:
    """Trailing-window sum of each column of `a`, aligned to the window's last
    row, NaN where the window holds fewer than `min_periods` finite values.

    Uses a cumulative-sum difference rather than a per-bar reduction, so the
    cost is one pass regardless of lookback. Non-finite entries are skipped
    rather than propagated: a plain cumsum turns a single NaN into NaN for
    every later row, which silently voids a whole series when the input is a
    warmup-padded column like the residual. Counting finite entries separately
    reproduces polars' rolling_sum(min_samples=...) semantics, so this matches
    roll_lr column for column.
    """
    n_rows, n_cols = a.shape
    finite = np.isfinite(a)
    cum_sum = np.cumsum(np.where(finite, a, 0.0), axis=0)
    cum_count = np.cumsum(finite, axis=0)

    zero = np.zeros((1, n_cols))
    padded_sum = np.concatenate([zero, cum_sum], axis=0)
    padded_count = np.concatenate([zero, cum_count], axis=0)

    end = np.arange(n_rows) + 1
    start = np.maximum(end - lookback, 0)
    window_sum = padded_sum[end] - padded_sum[start]
    window_count = padded_count[end] - padded_count[start]
    return np.where(window_count >= min_periods, window_sum, np.nan)


def roll_mlr(
    X: Union[pl.DataFrame, pd.DataFrame],
    y: Union[pl.Series, pd.Series],
    lookback: int = 100,
    min_periods: int = None,
    max_cond: float = 1e10,
) -> pl.DataFrame:
    """y = alpha + B.X for K regressors, rolled over trailing `lookback` rows.

    The multivariate counterpart to roll_lr. Rolling cross-product sums build
    one (K+1, K+1) normal-equation system per bar, solved as a single batched
    linear solve -- no Python loop over bars.

    Returns a polars DataFrame with columns:
        y, alpha, beta_<name> (one per regressor), yhat, resid, r2, cond,
        factor_cond

    `cond` is the 2-norm condition number of that bar's raw normal-equation
    matrix, chiefly a numerical-solver diagnostic.  `factor_cond` is the
    scale-free 2-norm condition number of the factor correlation matrix and
    is the diagnostic to use for multicollinearity.  Rates regressors are near-collinear
    (10y vs 30y runs above 0.95), which makes X'X ill-conditioned and sends
    individual betas swinging to large offsetting values even while the fit
    looks fine. Bars where cond exceeds `max_cond` are returned as null rather
    than as a confidently-wrong number; a persistently high cond means the
    regressors are not separately identified over that window, and the honest
    fixes are to drop a regressor or orthogonalize (see stats.pca.fit_pca),
    not to raise the ceiling.
    """
    X = to_pl_df(X)
    y = to_pl_series(y)
    names = list(X.columns)
    if not names:
        raise ValueError("roll_mlr needs at least one regressor column")

    if min_periods is None:
        min_periods = lookback
    if min_periods <= len(names):
        raise ValueError(
            f"min_periods={min_periods} cannot fit {len(names)} regressors plus "
            "an intercept; the system is underdetermined"
        )

    df = X.with_columns(y.alias("__y")).drop_nulls()
    y_arr = df["__y"].to_numpy().astype(float)
    x_arr = df.select(names).to_numpy().astype(float)
    n_rows, k = x_arr.shape
    n_params = k + 1

    # design matrix with intercept; the ones column makes the (0,0) moment the
    # observation count and the first row/column the plain sums, so the
    # intercept falls out of the same system as the slopes
    design = np.column_stack([np.ones(n_rows), x_arr])
    gram = _roll_sum(
        (design[:, :, None] * design[:, None, :]).reshape(n_rows, -1),
        lookback, min_periods,
    ).reshape(n_rows, n_params, n_params)
    moment = _roll_sum(design * y_arr[:, None], lookback, min_periods)

    coef = np.full((n_rows, n_params), np.nan)
    cond = np.full(n_rows, np.nan)
    factor_cond = np.full(n_rows, np.nan)
    ready = np.isfinite(gram).all(axis=(1, 2)) & np.isfinite(moment).all(axis=1)
    if ready.any():
        cond[ready] = np.linalg.cond(gram[ready])
        # The raw normal equations are necessarily unit-sensitive: an index
        # point and a bps rate may differ by orders of magnitude.  Convert
        # each trailing factor covariance matrix to correlations before taking
        # its condition number, so this answers the research question (are
        # factors separately identified?) rather than a units question.
        n_ready = gram[ready, 0, 0]
        sums_x = gram[ready, 0, 1:]
        cross_x = gram[ready, 1:, 1:]
        cov_x = cross_x / n_ready[:, None, None] - (
            sums_x / n_ready[:, None]
        )[:, :, None] * (sums_x / n_ready[:, None])[:, None, :]
        std_x = np.sqrt(np.clip(np.diagonal(cov_x, axis1=1, axis2=2), 0.0, None))
        valid_scale = np.all(std_x > np.finfo(float).eps, axis=1)
        if valid_scale.any():
            corr_x = cov_x[valid_scale] / (
                std_x[valid_scale, :, None] * std_x[valid_scale, None, :]
            )
            ready_idx = np.flatnonzero(ready)
            factor_cond[ready_idx[valid_scale]] = np.linalg.cond(corr_x)
        solvable = ready & np.isfinite(cond) & (cond <= max_cond)
        if solvable.any():
            # (S, P, 1) rather than (S, P): a 2-D b reads as one matrix, not a
            # stack of right-hand sides
            rhs = moment[solvable][:, :, None]
            try:
                coef[solvable] = np.linalg.solve(gram[solvable], rhs).squeeze(-1)
            except np.linalg.LinAlgError:
                # only reachable once max_cond is raised past the point where
                # the system is numerically rank-deficient; the least-norm
                # solution is the defensible answer there, not a crash
                coef[solvable] = (
                    np.linalg.pinv(gram[solvable]) @ rhs
                ).squeeze(-1)

    yhat = (design * coef).sum(axis=1)
    resid = y_arr - yhat

    # r2 mirrors roll_lr: each bar's residual uses that bar's coefficients,
    # summed over the window against the window's total variation in y
    sums = _roll_sum(
        np.column_stack([resid ** 2, y_arr, y_arr ** 2, np.ones(n_rows)]),
        lookback, min_periods,
    )
    ss_res, sum_y, sum_yy, n_obs = sums[:, 0], sums[:, 1], sums[:, 2], sums[:, 3]
    with np.errstate(invalid="ignore", divide="ignore"):
        ss_tot = sum_yy - sum_y ** 2 / n_obs
        r2 = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, np.nan)

    out = pl.DataFrame({
        "y": y_arr,
        "alpha": coef[:, 0],
        **{f"beta_{name}": coef[:, i + 1] for i, name in enumerate(names)},
        "yhat": yhat,
        "resid": resid,
        "r2": r2,
        "cond": cond,
        "factor_cond": factor_cond,
    })
    # warmup and unidentified bars arrive as NaN from numpy; the rest of the
    # codebase tests missingness with null (drop_nulls, is_not_null), and
    # roll_lr already emits nulls, so normalize before handing the frame back
    return out.with_columns(pl.exclude("y").fill_nan(None))


def roll_mlr_diff(
    X: Union[pl.DataFrame, pd.DataFrame],
    y: Union[pl.Series, pd.Series],
    lookback: int = 100,
    min_periods: int = None,
    max_cond: float = 1e10,
) -> pl.DataFrame:
    """Changes-based rolling multiple regression: dy = alpha + B.dX.

    The multivariate counterpart to roll_lr_diff, and the one to use for
    hedge ratios: betas fitted on daily changes are the weights that actually
    neutralize the regressors, whereas levels betas absorb any shared trend.

    Returns the roll_mlr columns plus `resid_cum`, the residual accumulated
    back into level space for z-scoring. Output is one row shorter than the
    input, the first differencing row having no predecessor.
    """
    X = to_pl_df(X)
    y = to_pl_series(y)
    names = list(X.columns)

    levels = X.with_columns(y.alias("__y")).drop_nulls()
    changes = levels.select(
        [pl.col(c).diff().alias(c) for c in names] + [pl.col("__y").diff().alias("__y")]
    ).slice(1)

    out = roll_mlr(
        changes.select(names), changes["__y"],
        lookback=lookback, min_periods=min_periods, max_cond=max_cond,
    )
    return out.with_columns(out["resid"].cum_sum().alias("resid_cum"))


def roll_beta(x, y, lookback=100):
    return roll_lr(x, y, lookback)["beta"]


def roll_resid(x, y, lookback=100):
    return roll_lr(x, y, lookback)["resid"]
