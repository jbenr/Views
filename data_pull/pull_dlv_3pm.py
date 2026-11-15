#!/usr/bin/env python3
"""Pull 3pm New York closes for every deliverable bond into md.ust_eod_3pm.

Treasury futures settle at 3pm New York while md.ust_eod holds 5pm bond closes,
so a basis built from the two carries the 3pm-to-5pm market move as noise.
Bloomberg's BBT3 pricing source is a 3pm bond close with daily history from
2008-07-15. This pulls it for every bond in a contract's deliverable basket,
over the whole time that contract was the one the basis is measured on (found
from the 5pm closes). utils.basis reads the result when cash_close="3pm".

Incremental: a bond already stored is only pulled from its last stored date.
    python pull_dlv_3pm.py          # extend what is stored
    python pull_dlv_3pm.py --full   # pull every bond's whole window again
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import polars as pl
import psycopg
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_pull.berg import Bbg
from utils.basis import DELIVERABLE_ROOTS, load_baskets, net_basis
from utils.helpers import DB_DSN, query_db

SOURCE = "BBT3"
HISTORY_START = "2008-07-15"  # first day BBT3 has prices
BATCH_SIZE = 50
SLACK_DAYS = 7  # a stored range this close to the wanted one counts as complete


def ensure_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS md.ust_eod_3pm (
                ts          date    NOT NULL,
                cusip       text    NOT NULL,
                px_last     double precision,
                source      text    NOT NULL DEFAULT 'BBT3',
                created_at  timestamptz NOT NULL DEFAULT now(),
                updated_at  timestamptz NOT NULL DEFAULT now(),
                PRIMARY KEY (cusip, ts)
            )
            """
        )
    conn.commit()


def basket_windows(nb: pl.DataFrame, baskets: pl.DataFrame) -> pl.DataFrame:
    """Date range to hold for each deliverable: cusip (8-char), lo, hi.

    `nb` gives the days each contract was the measured one, `baskets` the bonds
    in each contract. A bond sits in consecutive contracts, so its range is
    one unbroken span from the first of them to the last.
    """
    spans = nb.group_by("contract").agg(pl.col("ts").min().alias("lo"), pl.col("ts").max().alias("hi"))
    return (
        baskets.select("contract", "cusip").unique()
        .join(spans, on="contract")
        .group_by("cusip").agg(pl.col("lo").min(), pl.col("hi").max())
        .sort("cusip")
    )


def stored_ranges() -> pl.DataFrame:
    """Last date md.ust_eod_3pm already holds per bond: cusip (8-char), have_hi."""
    raw = query_db("SELECT left(cusip, 8) AS cusip, max(ts) AS have_hi FROM md.ust_eod_3pm GROUP BY 1")
    return pl.from_pandas(raw).cast({"cusip": pl.String, "have_hi": pl.Date})


def nine_char(cusips: list[str]) -> dict[str, str]:
    """8-char basket CUSIP -> the 9-char CUSIP Bloomberg and md.ust_eod use."""
    raw = query_db("SELECT DISTINCT cusip FROM sec.auctioned_securities WHERE left(cusip, 8) = ANY(%s)",
                   params=[cusips])
    return {c[:8]: c for c in raw["cusip"]}


def pulls_needed(windows: pl.DataFrame, stored: pl.DataFrame, today: dt.date) -> pl.DataFrame:
    """The (cusip, start, end) requests that bring storage up to the windows.

    A window that reaches the latest date in the basis is still open, so it
    runs to today and its last stored day is pulled again, which replaces an
    intraday print with the close.

    Stored bonds are only ever extended forward: a new issue has no BBT3
    price for its first days in a window, so a gap at the front is not a gap
    to fill.
    """
    latest = windows["hi"].max()
    slack = pl.duration(days=SLACK_DAYS)
    return (
        windows.join(stored, on="cusip", how="left")
        .with_columns(
            pl.when(pl.col("hi") == latest).then(pl.lit(today)).otherwise(pl.col("hi")).alias("end"),
            pl.coalesce("have_hi", "lo").alias("start"),
        )
        .filter(pl.col("have_hi").is_null() | (pl.col("hi") == latest)
                | (pl.col("have_hi") < pl.col("hi") - slack))
        .select("cusip", "start", "end")
        .sort("start", "end", "cusip")
    )


def fetch_3pm(bbg: Bbg, cusips9: list[str], start: dt.date, end: dt.date) -> pd.DataFrame:
    """BBT3 closes for one batch: ts, cusip (9-char), px_last."""
    data = bbg.bdh([f"{c}@{SOURCE} Govt" for c in cusips9], ["PX_LAST"], start=start, end=end)
    frames = [
        df.reset_index().rename(columns={"date": "ts", "PX_LAST": "px_last"}).assign(cusip=ticker[:9])
        for ticker, df in data.items()
        if df is not None and not df.empty and "PX_LAST" in df.columns
    ]
    if not frames:
        return pd.DataFrame(columns=["ts", "cusip", "px_last"])
    return pd.concat(frames, ignore_index=True)[["ts", "cusip", "px_last"]].dropna()


def upsert(conn, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    rows = [(r.ts, r.cusip, float(r.px_last), SOURCE) for r in df.itertuples(index=False)]
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO md.ust_eod_3pm (ts, cusip, px_last, source) VALUES (%s, %s, %s, %s)
            ON CONFLICT (cusip, ts) DO UPDATE SET
              px_last = EXCLUDED.px_last, source = EXCLUDED.source, updated_at = now()
            """,
            rows,
        )
    conn.commit()
    return len(rows)


def main() -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--full", action="store_true", help="pull every bond's whole window, not just what is new")
    args = parser.parse_args()
    today = dt.date.today()
    with psycopg.connect(DB_DSN) as conn:
        ensure_table(conn)

        # when each contract was the measured one (from the 5pm closes), and the bonds in it
        nb = net_basis(DELIVERABLE_ROOTS, start=HISTORY_START, cash_close="5pm")
        windows = basket_windows(nb, load_baskets(DELIVERABLE_ROOTS))

        # what is missing from storage
        todo = pulls_needed(windows, stored_ranges().clear() if args.full else stored_ranges(), today)
        ids = nine_char(todo["cusip"].to_list())
        todo = todo.with_columns(pl.col("cusip").replace_strict(ids, default=None).alias("cusip9")).drop_nulls("cusip9")
        print(f"{windows.height} deliverables since {HISTORY_START}; {todo.height} need 3pm closes.")
        if todo.is_empty():
            return {"windows": windows, "todo": todo, "rows": 0}

        # pull and store, one request per shared date range
        bbg = Bbg()
        rows = 0
        groups = todo.group_by(["start", "end"], maintain_order=True).agg("cusip9")
        for start, end, cusips in tqdm(groups.iter_rows(), total=groups.height, desc="BBT3 3pm closes", unit="range"):
            for i in range(0, len(cusips), BATCH_SIZE):
                rows += upsert(conn, fetch_3pm(bbg, cusips[i:i + BATCH_SIZE], start, end))

    print(f"Upserted {rows} rows into md.ust_eod_3pm")
    return {"windows": windows, "todo": todo, "rows": rows}


if __name__ == "__main__":
    main()
