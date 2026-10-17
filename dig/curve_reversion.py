"""When does a 10s30s residual revert, and what is it a residual OF?

10s30s regressed on one of its own legs is a statement about the OTHER leg:
  vs 10y:  d10s30s = a + b*d10y  <=>  d30y = a + (1+b)*d10y
           residual = what 30y did that 10y did not explain; package 30y - (1+b)*10y
  vs 30y:  d10s30s = a + g*d30y  <=>  d10y = -a + (1-g)*d30y
           residual = minus what 10y did that 30y did not explain; package (1-g)*30y - 10y
Both are fitted as Strategy does (changes OLS, residual re-summed over the beta
window), and each is tested on what it can actually trade: the forward P&L of
its own package with the hedge ratio frozen at entry. The signal's own forward
change is reported alongside, because a windowed sum reverts mechanically as
old days roll out of it, and that part is not P&L. Conditional tables ask when
fading an extreme pays: by speed of the move, vol regime, beta drift, rate
trend, side, and era. Bucket cut points are full-sample, so this is diagnosis,
not a gate.

A mixed variant takes the SIGNAL from a levels fit (is 10s30s rich/cheap vs
where 10y says it should be?) but keeps the changes-beta package as the trade
(the hedge that actually neutralises daily 10y moves).

    python -m dig.curve_reversion
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from stats import roll_lr, roll_lr_diff
from utils.market_data import load_wide

# ---- config -----------------------------------------------------------------

START = "2010-01-01"
TICKERS = {
    "10y": "USGG10YR Index",  # % -> scaled to bps at load
    "10s30s": "USYC1030 Index",  # already quoted in bps
}
BPS_COLS = ["10y"]
TARGET = "10s30s"
HEDGE_LEGS = ("10y", "30y")
LOOKBACK = 252
HORIZONS = (5, 20, 60)
EXTREME_Z = 1.5
COND_HORIZON = 20
ERAS = [("2012", "2015"), ("2016", "2019"), ("2020", "2022"), ("2023", "2026")]


# ---- helpers ----------------------------------------------------------------


def load_curve() -> pd.DataFrame:
    """10y and 10s30s in bps, plus 30y = 10y + 10s30s."""
    df = load_wide(TICKERS, START, bps_cols=BPS_COLS, to_pandas=True).dropna()
    df["30y"] = df["10y"] + df["10s30s"]
    return df


def fit_hedge(df: pd.DataFrame, leg: str) -> pd.DataFrame:
    """Changes OLS of d10s30s on d(leg); windowed residual, its z, package weights,
    and per-horizon forward P&L of the package with beta frozen at t."""
    reg = roll_lr_diff(df[leg], df[TARGET], lookback=LOOKBACK).to_pandas()
    reg.index = df.index[-len(reg):]  # roll_lr_diff drops leading rows
    out = pd.DataFrame(index=df.index)
    out["beta"] = reg["beta"]
    out["resid"] = reg["resid"].rolling(LOOKBACK).sum()
    out["z"] = out["resid"] / out["resid"].rolling(LOOKBACK).std()
    # package = 10s30s - beta*leg, written as weights on 30y and 10y
    out["w30"] = 1 - out["beta"] if leg == "30y" else 1.0
    out["w10"] = -1.0 if leg == "30y" else -(1 + out["beta"])
    for h in HORIZONS:
        d30 = df["30y"].shift(-h) - df["30y"]
        d10 = df["10y"].shift(-h) - df["10y"]
        out[f"leg30_{h}"] = out["w30"] * d30
        out[f"leg10_{h}"] = out["w10"] * d10
        out[f"pkg_{h}"] = out[f"leg30_{h}"] + out[f"leg10_{h}"]
        out[f"outright_{h}"] = df[TARGET].shift(-h) - df[TARGET]
        out[f"own_{h}"] = out["resid"].shift(-h) - out["resid"]
    return out


def with_levels_signal(df: pd.DataFrame, fit: pd.DataFrame, leg: str) -> pd.DataFrame:
    """Swap fit's signal for the levels-OLS residual of 10s30s on leg; the
    package (changes-beta hedge, frozen at t) and its forward P&L are kept."""
    reg = roll_lr(df[leg], df[TARGET], lookback=LOOKBACK).to_pandas()
    reg.index = df.index[-len(reg):]  # roll_lr drops leading rows
    out = fit.copy()
    out["resid"] = reg["resid"]
    out["z"] = out["resid"] / out["resid"].rolling(LOOKBACK).std()
    for h in HORIZONS:
        out[f"own_{h}"] = out["resid"].shift(-h) - out["resid"]
    return out


def hac_slope(y: pd.Series, x: pd.Series, lags: int) -> tuple[float, float]:
    """Slope and Newey-West t of y on x (overlapping forward windows)."""
    d = pd.concat([y, x], axis=1).dropna()
    fit = sm.OLS(d.iloc[:, 0], sm.add_constant(d.iloc[:, 1])).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags}
    )
    return float(fit.params.iloc[1]), float(fit.tvalues.iloc[1])


def describe_fit(df: pd.DataFrame, fit: pd.DataFrame, leg: str) -> dict:
    """What the package is: median weights, beta range, residual vs leg vol."""
    d = df.diff()
    return {
        "hedge_leg": leg,
        "beta_med": fit["beta"].median(),
        "beta_p10": fit["beta"].quantile(0.1),
        "beta_p90": fit["beta"].quantile(0.9),
        "w30_med": pd.Series(fit["w30"]).median(),
        "w10_med": pd.Series(fit["w10"]).median(),
        "corr_d10_d30": d["10y"].corr(d["30y"]),
        "vol_d10s30s": d[TARGET].std(),
        "vol_daily_pkg": (fit["w30"] * d["30y"] + fit["w10"] * d["10y"]).std(),
    }


def reversion_table(fit: pd.DataFrame) -> pd.DataFrame:
    """Per horizon: slope (bps per 1 sigma of z) of forward package, outright and
    the signal's own change on z. Negative = reverts."""
    rows = []
    for h in HORIZONS:
        row = {"h": h}
        for name in ("pkg", "outright", "own"):
            b, t = hac_slope(fit[f"{name}_{h}"], fit["z"], lags=h)
            row[f"{name}_slope"] = b
            row[f"{name}_t"] = t
        rows.append(row)
    return pd.DataFrame(rows)


def fade_stats(fit: pd.DataFrame, mask: pd.Series, h: int) -> dict:
    """Fade every |z|>EXTREME_Z day in mask: mean package P&L, hit rate, which
    leg paid, and the direction component the outright adds on top."""
    sel = fit[mask & (fit["z"].abs() > EXTREME_Z)].dropna(subset=[f"pkg_{h}"])
    side = -np.sign(sel["z"])
    pnl = side * sel[f"pkg_{h}"]
    return {
        "days": len(sel),
        "episodes": int((sel.index.to_series().diff().dt.days > h).sum() + (len(sel) > 0)),
        "pkg_pnl": pnl.mean(),
        "hit": (pnl > 0).mean(),
        "from_30y_leg": (side * sel[f"leg30_{h}"]).mean(),
        "from_10y_leg": (side * sel[f"leg10_{h}"]).mean(),
        "outright_pnl": (side * sel[f"outright_{h}"]).mean(),
    }


def conditional_table(df: pd.DataFrame, fit: pd.DataFrame, h: int) -> pd.DataFrame:
    """fade_stats split by candidate 'is it reversionary' conditions."""
    resid = fit["resid"]
    built_fast = ((resid - resid.shift(20)) / resid).clip(-2, 2) > 0.5
    vol = df["10y"].diff().rolling(60).std()
    drift = (fit["beta"] - fit["beta"].shift(60)).abs()
    trend = df["10y"] - df["10y"].shift(60)
    splits = {
        "all": {"all": pd.Series(True, index=fit.index)},
        "side": {"steep (z>0)": fit["z"] > 0, "flat (z<0)": fit["z"] < 0},
        "speed": {"built in last 20d": built_fast, "built slowly": ~built_fast},
        "10y vol 60d": dict(zip(["low", "mid", "high"], _terciles(vol))),
        "beta drift 60d": dict(zip(["stable", "mid", "drifting"], _terciles(drift))),
        "10y trend 60d": {"rates up": trend > 0, "rates down": trend <= 0},
        "era": {f"{a}-{b}": (fit.index >= a) & (fit.index <= f"{b}-12-31") for a, b in ERAS},
    }
    rows = []
    for split, buckets in splits.items():
        for bucket, mask in buckets.items():
            rows.append({"split": split, "bucket": bucket, **fade_stats(fit, pd.Series(mask, index=fit.index), h)})
    return pd.DataFrame(rows)


def _terciles(s: pd.Series) -> list[pd.Series]:
    lo, hi = s.quantile([1 / 3, 2 / 3])
    return [s <= lo, (s > lo) & (s <= hi), s > hi]


# ---- main -------------------------------------------------------------------


def main() -> dict:
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:,.2f}")

    # the curve and its two legs
    df = load_curve()

    # one changes fit per hedge leg: each defines a different package
    hedges = {leg: fit_hedge(df, leg) for leg in HEDGE_LEGS}

    # what each residual is: package weights and how much vol is left
    packages = pd.DataFrame([describe_fit(df, hedges[leg], leg) for leg in HEDGE_LEGS])
    print("\n== what each residual is ==\n", packages.to_string(index=False))

    # signal variants on the same packages: changes residual, or levels residual
    fits = {}
    for leg in HEDGE_LEGS:
        fits[f"changes signal vs {leg}"] = hedges[leg]
        fits[f"levels signal vs {leg}, changes hedge"] = with_levels_signal(df, hedges[leg], leg)

    # does it revert? tradable package vs outright vs the signal itself
    reversion = {name: reversion_table(fit) for name, fit in fits.items()}
    for name, tab in reversion.items():
        print(f"\n== {name}: slope of forward change on z (bps per 1 sigma; <0 reverts) ==")
        print(tab.to_string(index=False))

    # when does fading an extreme pay?
    conditional = {name: conditional_table(df, fit, COND_HORIZON) for name, fit in fits.items()}
    for name, tab in conditional.items():
        print(f"\n== {name}: fade |z|>{EXTREME_Z}, {COND_HORIZON}d forward, bps ==")
        print(tab.to_string(index=False))

    return {"df": df, "hedges": hedges, "fits": fits, "packages": packages,
            "reversion": reversion, "conditional": conditional}


if __name__ == "__main__":
    state = main()
