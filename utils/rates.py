"""Reusable rate construction helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Optional

import pandas as pd


def synthetic_real_rate(nominal_rate, inflation_rate):
    """Synthetic real rate: nominal rate minus inflation swap rate."""
    return nominal_rate - inflation_rate


def linear_5y5y_forward(r5, r10):
    """Trader shortcut for 5y5y forward: approximately 2 * 10y - 5y."""
    return 2 * r10 - r5


def synthetic_5y5y_real(sofr_5y, sofr_10y, zcis_5y, zcis_10y):
    """Approximate synthetic 5y5y real rate from SOFR and ZCIS rates."""
    real_5y = synthetic_real_rate(sofr_5y, zcis_5y)
    real_10y = synthetic_real_rate(sofr_10y, zcis_10y)
    return linear_5y5y_forward(real_5y, real_10y)


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
        out[sofr_5y5y_col] = linear_5y5y_forward(out[cols["sofr_5y"]], out[cols["sofr_10y"]])
    if overwrite or zcis_5y5y_col not in out.columns:
        out[zcis_5y5y_col] = linear_5y5y_forward(out[cols["zcis_5y"]], out[cols["zcis_10y"]])
    return out
