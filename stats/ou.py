"""Ornstein-Uhlenbeck process: dX = theta*(mu - X)*dt + sigma*dW"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Union


def half_life(series: pd.Series) -> float:
    """half-life of mean reversion via AR(1): delta_x = phi*x_lag + eps"""
    s = series.dropna().values.astype(float)
    if len(s) < 3:
        return np.nan

    y, x = np.diff(s), s[:-1]  # delta vs lagged level
    xm = x.mean()
    dx = x - xm
    ss_xx = (dx * dx).sum()
    if ss_xx == 0:
        return np.nan

    phi = (dx * (y - y.mean())).sum() / ss_xx
    return -np.log(2) / np.log(1 + phi) if phi < 0 else np.nan


def ou_params(series: pd.Series, dt: float = 1.0) -> dict:
    """estimate theta, mu, sigma from discrete series via AR(1) OLS"""
    s = series.dropna().values.astype(float)
    if len(s) < 3:
        return {'theta': np.nan, 'mu': np.nan, 'sigma': np.nan, 'half_life': np.nan}

    y, x = np.diff(s), s[:-1]
    xm, ym = x.mean(), y.mean()
    dx = x - xm
    ss_xx = (dx * dx).sum()
    if ss_xx == 0:
        return {'theta': np.nan, 'mu': np.nan, 'sigma': np.nan, 'half_life': np.nan}

    b = (dx * (y - ym)).sum() / ss_xx
    a = ym - b * xm
    theta = -b / dt
    mu = -a / b if b != 0 else np.nan
    sigma = (y - (a + b * x)).std() / np.sqrt(dt)
    hl = -np.log(2) / np.log(1 + b) if b < 0 else np.nan

    return {'theta': theta, 'mu': mu, 'sigma': sigma, 'half_life': hl}


def ou_zscore(series: pd.Series, lookback: int = None) -> pd.Series:
    """z-score vs OU equilibrium (rolling if lookback given, else full-sample)"""
    if lookback is not None:
        mu = series.rolling(lookback).mean()
        std = series.rolling(lookback).std()
    else:
        p = ou_params(series)
        mu, std = p['mu'], series.std()

    return ((series - mu) / std).rename('ou_zscore')


def roll_half_life(series: pd.Series, lookback: int = 100, min_periods: int = None) -> pd.Series:
    """rolling half-life over trailing window"""
    if min_periods is None:
        min_periods = lookback

    s = series.dropna()
    vals = s.values.astype(float)
    n = len(vals)
    hl = np.full(n, np.nan)

    for i in range(min_periods - 1, n):
        start = max(0, i - lookback + 1)
        w = vals[start:i+1]
        if len(w) < 3:
            continue
        y, x = np.diff(w), w[:-1]
        xm = x.mean()
        dx = x - xm
        ss_xx = (dx * dx).sum()
        if ss_xx == 0:
            continue
        phi = (dx * (y - y.mean())).sum() / ss_xx
        if phi < 0:
            hl[i] = -np.log(2) / np.log(1 + phi)

    return pd.Series(hl, index=s.index, name='half_life')


def ou_summary(series: pd.Series, lookback: int = None) -> pd.DataFrame:
    """single-row summary: theta, mu, sigma, half_life, current zscore & level"""
    window = series.dropna().iloc[-lookback:] if lookback else series.dropna()
    p = ou_params(window)
    z = ou_zscore(series, lookback=lookback).iloc[-1] if len(series.dropna()) > 0 else np.nan

    return pd.DataFrame([{
        'theta': p['theta'], 'mu': p['mu'], 'sigma': p['sigma'],
        'half_life': p['half_life'], 'current_zscore': z,
        'current_level': series.dropna().iloc[-1] if len(series.dropna()) > 0 else np.nan,
    }])
