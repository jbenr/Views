"""Market-data loading from md.index_eod — the single research data entry point.

Every research/strategy file used to hand-roll the same pattern: query
md.index_eod, pivot long → wide, rename tickers to short aliases, scale
yields ×100 so everything downstream is natively in bps. That lives here now.

    from utils.market_data import load_wide

    TICKERS = {"10y": "USGG10YR Index", "30y": "USGG30YR Index"}
    df = load_wide(TICKERS, start="2010-01-01", bps_cols=["10y", "30y"])

`long_to_wide` is the pure transformation (testable without a DB);
`load_wide` wraps it with the query.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Union

import pandas as pd
import polars as pl

from utils.helpers import query_db


def _require_columns(data: pl.DataFrame, columns: Iterable[str]) -> list[str]:
    missing = [c for c in columns if c not in data.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    return list(columns)


def align_columns(
    data: pl.DataFrame,
    columns: Iterable[str],
    date_col: str = "ts",
    optional: Iterable[str] = (),
) -> pl.DataFrame:
    """Return date + required columns on the common non-null sample.

    ``optional`` columns are carried through but kept out of the non-null
    subset: an exogenous series that starts years after the model columns
    must not truncate the sample it is only decorating. They still have to
    exist — a misspelled name is an error, not a silent drop.
    """
    col_list = list(columns)
    opt_list = [c for c in optional if c not in col_list]
    cols = _require_columns(data, [date_col, *col_list, *opt_list])
    return data.select(cols).drop_nulls(subset=col_list)


def coverage_report(
    data: pl.DataFrame,
    columns: Iterable[str],
    date_col: str = "ts",
) -> pl.DataFrame:
    """Coverage and overlap report for a set of model input columns."""
    col_list = list(columns)
    cols = _require_columns(data, [date_col, *col_list])
    data = data.select(cols)
    n_rows = len(data)
    rows = []

    for col in col_list:
        valid = data.filter(pl.col(col).is_not_null())
        rows.append(
            {
                "series": col,
                "observations": len(valid),
                "missing": n_rows - len(valid),
                "pct_populated": len(valid) / n_rows if n_rows else 0.0,
                "first_valid": valid[date_col].min() if len(valid) else None,
                "last_valid": valid[date_col].max() if len(valid) else None,
            }
        )

    overlap = align_columns(data, col_list, date_col=date_col)
    rows.append(
        {
            "series": "OVERLAP",
            "observations": len(overlap),
            "missing": n_rows - len(overlap),
            "pct_populated": len(overlap) / n_rows if n_rows else 0.0,
            "first_valid": overlap[date_col].min() if len(overlap) else None,
            "last_valid": overlap[date_col].max() if len(overlap) else None,
        }
    )
    return pl.DataFrame(rows)


def long_to_wide(
    long_df: Union[pl.DataFrame, pd.DataFrame],
    tickers: Union[Mapping[str, str], Iterable[str]],
    bps_cols: Union[Iterable[str], str, None] = None,
    to_pandas: bool = False,
) -> Union[pl.DataFrame, pd.DataFrame]:
    """Pivot a long (ts, ticker, px) frame wide, rename to aliases, scale to bps.

    Args:
        long_df: Long-format frame with columns ts, ticker, px.
        tickers: Either {alias: ticker} (columns renamed to aliases) or a
                 plain iterable of tickers (columns keep ticker names).
        bps_cols: Post-rename column names to scale ×100, or "all", or None.
        to_pandas: Return a pandas DataFrame indexed by ts instead of polars.
    """
    df = pl.from_pandas(long_df) if isinstance(long_df, pd.DataFrame) else long_df

    if df.is_empty():
        empty = pl.DataFrame(schema={"ts": pl.Date})
        return empty.to_pandas().set_index("ts") if to_pandas else empty

    wide = (
        df.with_columns(
            pl.col("ts").cast(pl.Date),
            # Bloomberg returns NaN (not NULL) for market-closed days, and the
            # DB stores it as a real NaN float. polars drop_nulls does not drop
            # NaN, so it would survive align_columns and then poison a whole
            # rolling window. Normalize to null at the single point of entry.
            pl.col("px").cast(pl.Float64).fill_nan(None),
        )
        .pivot(index="ts", on="ticker", values="px")
        .sort("ts")
    )

    if isinstance(tickers, Mapping):
        inv = {v: k for k, v in tickers.items()}
        wide = wide.rename({c: inv[c] for c in wide.columns if c in inv})

    if bps_cols == "all":
        bps_cols = [c for c in wide.columns if c != "ts"]
    if bps_cols:
        scale = [pl.col(c) * 100.0 for c in bps_cols if c in wide.columns]
        if scale:
            wide = wide.with_columns(scale)

    if to_pandas:
        return wide.to_pandas().set_index("ts")
    return wide


def load_wide(
    tickers: Union[Mapping[str, str], Iterable[str]],
    start: str,
    end: str | None = None,
    bps_cols: Union[Iterable[str], str, None] = None,
    to_pandas: bool = False,
) -> Union[pl.DataFrame, pd.DataFrame]:
    """Load tickers from md.index_eod as a wide frame (one column per series).

    Args:
        tickers: {alias: ticker} to rename columns, or an iterable of tickers.
        start: Inclusive start date, e.g. "2010-01-01".
        end: Optional inclusive end date.
        bps_cols: Post-rename columns to scale ×100 into bps, or "all", or None.
        to_pandas: Return pandas (indexed by ts) instead of polars.
    """
    ticker_list = list(tickers.values()) if isinstance(tickers, Mapping) else list(tickers)
    sql = """
        SELECT ts, ticker, px_last::float AS px
        FROM md.index_eod
        WHERE ticker = ANY(%s) AND ts >= %s
    """
    params: list = [ticker_list, start]
    if end is not None:
        sql += " AND ts <= %s"
        params.append(end)
    sql += " ORDER BY ts"

    long_df = query_db(sql, params=params)
    return long_to_wide(long_df, tickers, bps_cols=bps_cols, to_pandas=to_pandas)


def swaption_point(expiry: str, tenor: int, strike: int = 0) -> str:
    """Canonical column alias for one point on the swaption surface."""
    suffix = "" if strike == 0 else f"_{strike:+d}"
    return f"vol_{expiry}_{tenor}{suffix}"


def load_swaption_wide(
    points: Iterable[tuple[str, int]],
    start: str,
    end: str | None = None,
    strike: int = 0,
    to_pandas: bool = False,
) -> Union[pl.DataFrame, pd.DataFrame]:
    """Load ATM-relative swaption vols as a wide frame, one column per point.

    md.swaption_vol is stored long and keyed (ts, expiry, tenor, strike), so
    unlike md.index_eod there is no ticker to pivot on. Columns come back
    named by `swaption_point`, e.g. ("1Mo", 30) -> "vol_1Mo_30".

    Vols are normal vols in bps/yr and are already in bps -- do NOT pass these
    through a bps_cols scaling the way index_eod yields need.

    Args:
        points: (expiry, tenor) pairs, e.g. [("1Mo", 30), ("3Mo", 10)].
        start: Inclusive start date.
        end: Optional inclusive end date.
        strike: Strike offset in bps from ATM; 0 is the ATM surface.
    """
    pairs = [(str(e), int(t)) for e, t in points]
    empty = pl.DataFrame(schema={"ts": pl.Date})
    if not pairs:
        return empty.to_pandas().set_index("ts") if to_pandas else empty

    sql = """
        SELECT ts, expiry, tenor, vol::float AS vol
        FROM md.swaption_vol
        WHERE ts >= %s AND strike = %s
          AND expiry = ANY(%s) AND tenor = ANY(%s)
    """
    params: list = [
        start,
        strike,
        list(dict.fromkeys(e for e, _ in pairs)),
        list(dict.fromkeys(t for _, t in pairs)),
    ]
    if end is not None:
        sql += " AND ts <= %s"
        params.append(end)
    sql += " ORDER BY ts"

    long_df = query_db(sql, params=params)
    if long_df.empty:
        return empty.to_pandas().set_index("ts") if to_pandas else empty

    # expiry and tenor are filtered independently in SQL, so the result is the
    # full cross-product; keep only the pairs actually asked for.
    wanted = [swaption_point(e, t, strike) for e, t in pairs]
    suffix = "" if strike == 0 else f"_{strike:+d}"
    wide = (
        pl.from_pandas(long_df)
        .with_columns(
            pl.col("ts").cast(pl.Date),
            pl.col("vol").cast(pl.Float64).fill_nan(None),  # see long_to_wide
            (
                pl.lit("vol_") + pl.col("expiry") + pl.lit("_")
                + pl.col("tenor").cast(pl.Utf8) + pl.lit(suffix)
            ).alias("point"),
        )
        .filter(pl.col("point").is_in(wanted))
        .pivot(index="ts", on="point", values="vol")
        .sort("ts")
    )
    if to_pandas:
        return wide.to_pandas().set_index("ts")
    return wide


def swaption_last_updated(wake: bool = True) -> dict | None:
    """Freshness of md.swaption_vol -- newest bar date and newest write time.

    The sibling of `last_updated` for the surface, which has no ticker column
    to filter on. Unlike md.index_eod this table has no updated_at, so the
    write time is created_at and is a lower bound: a row re-upserted later
    leaves no trace of it.
    """
    out = query_db(
        """
        SELECT MAX(ts) AS last_ts, MAX(created_at) AS last_written
        FROM md.swaption_vol
        """,
        wake=wake,
    )
    if out.empty or pd.isna(out.loc[0, "last_ts"]):
        return None
    return {
        "last_ts": out.loc[0, "last_ts"],
        "last_written": out.loc[0, "last_written"],
    }


def last_updated(
    tickers: Union[Mapping[str, str], Iterable[str]],
    wake: bool = True,
) -> dict | None:
    """Freshness of md.index_eod itself for a set of tickers.

    Returns the newest bar date and the newest row write time, or None if the
    tickers have no rows. This is the database's own clock -- how current the
    source is, independent of any cached copy of it.

    Write time is updated_at, not created_at: the live BDP pulls upsert
    today's row all day, so created_at stays pinned to the first insert and
    would report a bar as hours old seconds after it was rewritten.

    Rows written before updated_at existed have it NULL and fall back to
    created_at, which is only a lower bound: such a row may have been upserted
    any number of times since, leaving no trace. Once a pull has run they
    carry a real updated_at and the answer is exact.

    wake=False fails fast rather than waking a sleeping WSL/postgres, for
    callers that treat freshness as optional decoration.
    """
    ticker_list = (
        list(tickers.values()) if isinstance(tickers, Mapping) else list(tickers)
    )
    out = query_db(
        """
        SELECT MAX(ts) AS last_ts,
               MAX(COALESCE(updated_at, created_at)) AS last_written
        FROM md.index_eod
        WHERE ticker = ANY(%s)
        """,
        params=[ticker_list],
        wake=wake,
    )
    if out.empty or pd.isna(out.loc[0, "last_ts"]):
        return None
    return {
        "last_ts": out.loc[0, "last_ts"],
        "last_written": out.loc[0, "last_written"],
    }


def pick_ticker(candidates: list[str], start: str) -> str | None:
    """Pick the candidate ticker with the most md.index_eod rows since `start`.

    Useful when the same series exists under multiple tickers (e.g. swap vs
    ZCIS versions) and you want whichever has the best coverage.
    """
    out = query_db(
        """
        SELECT ticker, COUNT(*) AS n
        FROM md.index_eod
        WHERE ticker = ANY(%s) AND ts >= %s
        GROUP BY ticker
        ORDER BY n DESC
        LIMIT 1
        """,
        params=(candidates, start),
    )
    if out.empty:
        return None
    return str(out.loc[0, "ticker"])
