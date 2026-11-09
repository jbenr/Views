"""Headline Treasury market for the overview: yields, curves and flies, each with its
last level and its 1d, 1m and YTD moves.

Pulled from the md.index_eod generics on the overview's Ref and cached to
store/live_data/headline.parquet. A page load reads only that cache, as the
signals do, so the page never waits on the database.
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import polars as pl

from utils.helpers import query_db
from utils.market_data import long_to_wide

YIELDS = {
    "2y": "USGG2YR Index", "3y": "USGG3YR Index", "5y": "USGG5YR Index", "7y": "USGG7YR Index",
    "10y": "USGG10YR Index", "20y": "USGG20YR Index", "30y": "USGG30YR Index",
}
# Bloomberg's own curve tickers, as the curve signals use: they can differ from the
# difference of the two generics by a few tenths of a bp
CURVES = {
    "2s5s": "USYC2Y5Y Index", "2s10s": "USYC2Y10 Index", "5s7s": "USYC5Y7Y Index",
    "7s10s": "USYC7Y10 Index", "5s30s": "USYC5Y30 Index", "10s30s": "USYC1030 Index",
}
TICKERS = {**YIELDS, **CURVES}
FLIES = [("2y", "5y", "10y"), ("5y", "7y", "10y"), ("5y", "10y", "30y"), ("7y", "10y", "30y")]

# Example only: no live trades are wired to the dashboard yet. Name -> direction.
HELD = {"10s30s": "long"}
HELD_IS_EXAMPLE = True


def _name(*legs: str) -> str:
    """('7y', '10y', '30y') -> '7s10s30s', the panel catalog's names."""
    return "".join(f"{leg[:-1]}s" for leg in legs)


def _cache_path() -> Path:
    store = os.getenv("VIEWS_STORE_DIR", str(Path(__file__).resolve().parents[1] / "store"))
    return Path(store) / "live_data" / "headline.parquet"


def pull(today: dt.date | None = None) -> pl.DataFrame:
    """Fresh yields and curves (bps) since December of last year, enough for YTD and 1m, cached.
    wake=False: the overview must not hang on a sleeping database."""
    today = today or dt.date.today()
    long_df = query_db(
        "SELECT ts, ticker, px_last::float AS px FROM md.index_eod WHERE ticker = ANY(%s) AND ts >= %s ORDER BY ts",
        params=[list(TICKERS.values()), dt.date(today.year - 1, 12, 1)],
        wake=False,
    )
    frame = long_to_wide(long_df, TICKERS, bps_cols=list(YIELDS))  # curve tickers are quoted in bps already
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)
    return frame


def cached() -> tuple[pl.DataFrame | None, dt.datetime | None]:
    """The last pull and when it was written, or (None, None) before the first Ref."""
    path = _cache_path()
    if not path.exists():
        return None, None
    return pl.read_parquet(path), dt.datetime.fromtimestamp(path.stat().st_mtime)


def _moves(ts: pl.Series, level: pl.Series, name: str) -> dict:
    """Last level and its change over the last bar, ~1 month and the year to date."""
    frame = pl.DataFrame({"ts": ts, "level": level}).drop_nulls()
    if frame.is_empty():
        return dict(name=name, last=None, d1=None, m1=None, ytd=None)
    last_ts, last = frame["ts"][-1], frame["level"][-1]

    def since(cutoff):
        earlier = frame.filter(pl.col("ts") <= cutoff)["level"]
        return last - earlier[-1] if len(earlier) else None

    prior = frame["level"][-2] if len(frame) > 1 else None
    return dict(name=name, last=last, d1=None if prior is None else last - prior,
                m1=since(last_ts - dt.timedelta(days=30)), ytd=since(dt.date(last_ts.year - 1, 12, 31)))


def board(frame: pl.DataFrame) -> dict[str, list[dict]]:
    """Yields, curves and flies as rows of {name, last, d1, m1, ytd, held}. Yields' last
    is in percent; every other number is bps. Flies are 2*belly - wings (positive =
    belly cheap), the catalog's convention."""
    ts = frame["ts"]

    def rows(series: list[tuple[str, pl.Series]]) -> list[dict]:
        return [{**_moves(ts, s, name), "held": HELD.get(name)} for name, s in series]

    have = [k for k in TICKERS if k in frame.columns]
    yields = rows([(k, frame[k]) for k in YIELDS if k in have])
    for row in yields:
        row["last"] = None if row["last"] is None else row["last"] / 100
    curves = rows([(k, frame[k]) for k in CURVES if k in have])
    flies = rows([(_name(a, b, c), 2 * frame[b] - frame[a] - frame[c]) for a, b, c in FLIES
                  if a in have and b in have and c in have])
    return {"yields": yields, "curves": curves, "flies": flies}


def asof(frame: pl.DataFrame) -> dt.date | None:
    """The newest bar any headline series has."""
    return None if frame.is_empty() else frame["ts"][-1]
