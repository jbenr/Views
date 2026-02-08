"""Rolling OLS — no lookahead bias."""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Union


def roll_lr(
    x: Union[pd.Series, pd.DataFrame],
    y: pd.Series,
    lookback: int = 100,
    min_periods: int = None,
) -> pd.DataFrame:
    """y = alpha + beta*x, rolled over trailing `lookback` rows.
    Returns: x, y, alpha, beta, yhat, resid, r2.
    """
    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0]
    if min_periods is None:
        min_periods = lookback

    combined = pd.concat([x.rename('x'), y.rename('y')], axis=1).dropna()
    xs, ys = combined['x'].values.astype(float), combined['y'].values.astype(float)
    n = len(xs)

    alpha = np.full(n, np.nan)
    beta = np.full(n, np.nan)
    yhat = np.full(n, np.nan)
    resid = np.full(n, np.nan)
    r2 = np.full(n, np.nan)

    for i in range(min_periods - 1, n):
        s = max(0, i - lookback + 1)
        xw, yw = xs[s:i+1], ys[s:i+1]
        xm, ym = xw.mean(), yw.mean()
        dx, dy = xw - xm, yw - ym
        ss_xx = (dx * dx).sum()
        if ss_xx == 0:
            continue

        b = (dx * dy).sum() / ss_xx
        a = ym - b * xm
        yhat[i] = a + b * xs[i]
        resid[i] = ys[i] - yhat[i]
        alpha[i], beta[i] = a, b

        ss_tot = (dy * dy).sum()
        r2[i] = 1 - ((yw - (a + b * xw)) ** 2).sum() / ss_tot if ss_tot else np.nan

    return pd.DataFrame({
        'x': combined['x'].values, 'y': combined['y'].values,
        'alpha': alpha, 'beta': beta, 'yhat': yhat, 'resid': resid, 'r2': r2,
    }, index=combined.index)


def roll_beta(x, y, lookback=100):
    return roll_lr(x, y, lookback)['beta']


def roll_resid(x, y, lookback=100):
    return roll_lr(x, y, lookback)['resid']
