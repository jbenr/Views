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
        df.with_columns(pl.col("ts").cast(pl.Date), pl.col("px").cast(pl.Float64))
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
