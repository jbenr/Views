#!/usr/bin/env python3
"""
Build the options greek exposure tables from md.opt_eod + md.fut_eod (engine: utils/options.py).

- md.opt_greeks            per option per day: future price, years to expiry, implied vol,
                           delta / gamma / vega / vanna / charm, open interest
- md.opt_strike_exposure   per day, expiry and strike: call and put open interest, its change
                           on the day, and greek x open interest by side
- md.opt_profile           per day: call / put / net gamma and delta if the futures moved
                           -4 to +4 points (vols held)
- md.opt_levels            per day: net and gross gamma, flip, call and put walls, gamma peak,
                           max pain, hedge flow for futures moves, charm and vanna

Raw positioning: calls and puts are kept apart; "net" is calls - puts (see utils/options.py).

Usage:
    python build_opt_exposure.py             # incremental: from the last built day, with overlap
    python build_opt_exposure.py --rebuild   # every day in md.opt_eod
    python build_opt_exposure.py --root TY
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys

import polars as pl
import psycopg

sys.path.append(os.path.expanduser("~/werk/Views"))

from utils.options import (FLOW_MOVES, _tag, daily_levels, load_future_px, load_option_eod,  # noqa: E402
                           price_options, strike_exposure)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DB_DSN = os.getenv("DB_DSN", "postgresql://benjils:snickers@raptor:5432/markets")
ROOTS = ["TY"]
OVERLAP_DAYS = 7        # rebuild the last week: md.opt_eod restates recent days
DIFF_LOOKBACK_DAYS = 10  # earlier days loaded so the first rebuilt day has an open-interest change

F = "double precision"
TABLES = {
    "opt_greeks": (["ts date", "ticker text", "root text", "kind text", "underlying text", "expiry date",
                    f"strike {F}", "put_call char(1)", f"settle {F}", f"oi {F}", f"volume {F}", f"fut_px {F}",
                    f"t_years {F}", f"iv {F}", f"delta {F}", f"gamma {F}", f"vega {F}", f"vanna {F}", f"charm {F}"],
                   "ts, ticker"),
    "opt_strike_exposure": (["ts date", "root text", "expiry date", f"strike {F}", "kind text", "underlying text",
                             f"fut_px {F}"]
                            + [f"{s}_{c} {F}" for s in ("call", "put")
                               for c in ("oi", "delta", "gamma", "vega", "vanna", "charm")]
                            + [f"d_call_oi {F}", f"d_put_oi {F}"], "ts, root, expiry, strike"),
    "opt_profile": (["ts date", "root text", f"move {F}"] + [f"{c} {F}" for c in (
        "call_gamma", "call_delta", "put_gamma", "put_delta", "net_gamma", "net_delta")], "ts, root, move"),
    "opt_levels": (["ts date", "root text", "underlying text", f"px {F}", f"options_oi {F}", f"call_gamma {F}",
                    f"put_gamma {F}", f"net_gamma {F}", f"gross_gamma {F}", f"weekly_gamma_share {F}",
                    f"flip_move {F}", f"flip {F}", f"call_wall {F}", f"put_wall {F}", f"gamma_peak {F}",
                    "front_expiry date", f"max_pain {F}", f"net_charm {F}", f"net_vanna {F}"]
                   + [f"flow_{_tag(m)} {F}" for m in FLOW_MOVES], "ts, root"),
}


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def get_conn():
    return psycopg.connect(DB_DSN)


def ensure_tables(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("CREATE SCHEMA IF NOT EXISTS md;")
        for name, (cols, pk) in TABLES.items():
            cur.execute(f"CREATE TABLE IF NOT EXISTS md.{name} ({', '.join(cols)}, PRIMARY KEY ({pk}));")
    conn.commit()


def columns(name: str) -> list[str]:
    return [c.split()[0] for c in TABLES[name][0]]


def replace_from(conn, name: str, root: str, start: dt.date, df: pl.DataFrame) -> int:
    """Delete the root's rows from ``start`` on and copy ``df`` in, in one transaction."""
    cols = columns(name)
    with conn.cursor() as cur:
        cur.execute(f"DELETE FROM md.{name} WHERE root = %s AND ts >= %s", (root, start))
        with cur.copy(f"COPY md.{name} ({', '.join(cols)}) FROM STDIN") as copy:
            for row in df.select(cols).iter_rows():
                copy.write_row(row)
    conn.commit()
    return len(df)


def first_day(conn, root: str, rebuild: bool) -> dt.date | None:
    with conn.cursor() as cur:
        cur.execute("SELECT max(ts) FROM md.opt_levels WHERE root = %s", (root,))
        last = cur.fetchone()[0]
        if last and not rebuild:
            return last - dt.timedelta(days=OVERLAP_DAYS)
        cur.execute("SELECT min(ts) FROM md.opt_eod WHERE root = %s", (root,))
        return cur.fetchone()[0]


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build(conn, root: str, rebuild: bool) -> None:
    start = first_day(conn, root, rebuild)
    if start is None:
        print(f"{root}: no options in md.opt_eod")
        return
    load_from = (start - dt.timedelta(days=DIFF_LOOKBACK_DAYS)).isoformat()
    priced = price_options(load_option_eod(root, load_from), load_future_px(root, load_from))
    if priced.is_empty():
        print(f"{root}: nothing priced from {start}")
        return
    print(f"{root}: {priced['ts'].n_unique()} days priced from {load_from}, writing from {start}")
    ladder = strike_exposure(priced).filter(pl.col("ts") >= start)
    priced = priced.filter(pl.col("ts") >= start)
    lv, prof = daily_levels(priced)
    tag = pl.lit(root).alias("root")
    n = replace_from(conn, "opt_greeks", root, start, priced.with_columns(
        tag, pl.col("px").alias("fut_px"), pl.col("T").alias("t_years")))
    print(f"  ✅ md.opt_greeks {n:,} rows")
    n = replace_from(conn, "opt_strike_exposure", root, start, ladder.with_columns(tag))
    print(f"  ✅ md.opt_strike_exposure {n:,} rows")
    n = replace_from(conn, "opt_profile", root, start, prof.with_columns(tag))
    print(f"  ✅ md.opt_profile {n:,} rows")
    n = replace_from(conn, "opt_levels", root, start, lv.with_columns(tag))
    print(f"  ✅ md.opt_levels {n:,} rows, last {lv['ts'].max()}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rebuild", action="store_true", help="rebuild every day in md.opt_eod")
    ap.add_argument("--root", action="append", help="option root (repeatable); default TY")
    args = ap.parse_args()
    conn = get_conn()
    ensure_tables(conn)
    for root in args.root or ROOTS:
        build(conn, root.upper(), args.rebuild)
    conn.close()


if __name__ == "__main__":
    main()
