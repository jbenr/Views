"""Reusable rate construction helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Optional

import pandas as pd


def synthetic_real_rate(nominal_rate, inflation_rate):
    """Synthetic real rate: nominal rate minus inflation swap rate."""
    return nominal_rate - inflation_rate


def linear_forward(r1, t1, r2, t2):
    """Linear (money-market) forward rate between t1 and t2 years.

        f = (r2 * t2 - r1 * t1) / (t2 - t1)

    Works on floats, pandas Series, or polars expressions -- it is pure
    arithmetic. Units follow the inputs: bps in, bps out.

    Symmetric in its (rate, tenor) pairs: swapping them leaves the result
    unchanged, so two forwards differ only if their TENORS differ. Market
    shorthand <a>y<b>y means the b-year rate starting a years forward, whose
    legs are a and a+b -- 2y3y is linear_forward(r2, 2, r5, 5), not
    linear_forward(r2, 2, r3, 3), which is 2y1y.

    A forward built from two tenors is a linear combination of them, so it
    carries any spread between those tenors directly. Choosing a forward that
    spans a curve you are trying to model means the explanatory variable
    contains the target: 5y5y is 1.5x the 5s10s spread plus the 5y/10y
    midpoint, so regressing 5s10s on it is largely regressing the target on
    itself. Pick a forward whose legs sit outside the target's, or one drawn
    from a different instrument.
    """
    if t2 == t1:
        raise ValueError(f"t2 must differ from t1; got t1={t1}, t2={t2}")
    return (r2 * t2 - r1 * t1) / (t2 - t1)



def synthetic_5y5y_real(sofr_5y, sofr_10y, zcis_5y, zcis_10y):
    """Approximate synthetic 5y5y real rate from SOFR and ZCIS rates."""
    real_5y = synthetic_real_rate(sofr_5y, zcis_5y)
    real_10y = synthetic_real_rate(sofr_10y, zcis_10y)
    return linear_forward(real_5y, 5, real_10y, 10)


DEFAULT_REAL_RATE_COLUMNS = {
    "sofr_5y": "sofr_5y",
    "sofr_10y": "sofr_10y",
    "zcis_5y": "zcis_5y",
    "zcis_10y": "zcis_10y",
}


def with_synthetic_real_rates(
    df: pd.DataFrame,
    columns: Optional[Mapping[str, str]] = None,
    real_5y_col: str = "real_5y",
    real_10y_col: str = "real_10y",
    real_5y5y_col: str = "real_5y5y",
    sofr_5y5y_col: str = "5y5y_sfr",
    zcis_5y5y_col: str = "5y5y_zc",
    overwrite: bool = False,
) -> pd.DataFrame:
    """Add synthetic real-rate columns when required source columns exist.

    Expected units must be consistent across inputs. If rates are in bps, outputs
    are in bps. If rates are decimals, outputs are decimals.
    """
    cols = {**DEFAULT_REAL_RATE_COLUMNS, **(dict(columns) if columns else {})}
    required = [cols["sofr_5y"], cols["sofr_10y"], cols["zcis_5y"], cols["zcis_10y"]]
    out = df.copy()
    if not all(col in out.columns for col in required):
        return out

    if overwrite or real_5y_col not in out.columns:
        out[real_5y_col] = synthetic_real_rate(out[cols["sofr_5y"]], out[cols["zcis_5y"]])
    if overwrite or real_10y_col not in out.columns:
        out[real_10y_col] = synthetic_real_rate(out[cols["sofr_10y"]], out[cols["zcis_10y"]])
    if overwrite or real_5y5y_col not in out.columns:
        out[real_5y5y_col] = synthetic_5y5y_real(
            out[cols["sofr_5y"]],
            out[cols["sofr_10y"]],
            out[cols["zcis_5y"]],
            out[cols["zcis_10y"]],
        )
    if overwrite or sofr_5y5y_col not in out.columns:
        out[sofr_5y5y_col] = linear_forward(out[cols["sofr_5y"]], 5, out[cols["sofr_10y"]], 10)
    if overwrite or zcis_5y5y_col not in out.columns:
        out[zcis_5y5y_col] = linear_forward(out[cols["zcis_5y"]], 5, out[cols["zcis_10y"]], 10)
    return out
