#!/usr/bin/env python3
"""
Live BDP snapshot — same fields as BDH pulls, writes is_live=True.

L1 (default): fut + index
L2:           + ust + strips

Usage:
    python pull_live.py             # L1 dry run
    python pull_live.py --level 2   # L2 dry run
    python pull_live.py --write     # write to DB
"""
from __future__ import annotations

import datetime as dt
import os
import sys
from pathlib import Path
from typing import List
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

if os.name == "nt":
    os.system("")

_BOLD  = "\033[1m"
_DIM   = "\033[2m"
_CYAN  = "\033[96m"
_RESET = "\033[0m"

import pandas as pd
import psycopg

sys.path.insert(0, str(Path(__file__).parent.parent))
from berg import Bbg
from utils import tickers as tkr

DB_DSN = os.getenv("DB_DSN", "postgresql://benjils:snickers@raptor:5432/markets")
TODAY  = dt.date.today()

FUT_GENERIC_RANKS = {
    "TU": range(1, 3),
    "FV": range(1, 3),
    "TY": range(1, 3),
    "UXY": range(1, 3),
    "US": range(1, 3),
    "WN": range(1, 3),
    "FF": range(1, 9),
    "SER": range(1, 9),
    "SFR": range(1, 9),
}
FUT_TICKERS = [
    f"{root}{rank} Comdty"
    for root, ranks in FUT_GENERIC_RANKS.items()
    for rank in ranks
]

INDEX_TICKERS = (
    tkr.ust + tkr.crv + tkr.fly +
    tkr.sofr_ois + tkr.swp_spd +
    tkr.tips + tkr.be + tkr.zc +
    tkr.stonk + tkr.sector +
    tkr.prec_met + tkr.enrgy + tkr.ag + tkr.ind_met
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _bdp_batched(bbg: Bbg, tickers: List[str], fields: List[str], batch: int = 100) -> pd.DataFrame:
    frames = []
    for i in range(0, len(tickers), batch):
        frames.append(bbg.bdp(tickers[i:i+batch], fields))
    return pd.concat(frames) if frames else pd.DataFrame(columns=fields)


def _n(v):
    """pandas NaN -> None for psycopg."""
    try:
        return None if pd.isna(v) else v
    except (TypeError, ValueError):
        return v


def _fmt_et(d: dt.datetime) -> str:
    """Apr 30, 2026 9:43:22pm ET"""
    d = d.astimezone(ET)
    h = int(d.strftime('%I'))  # 12h, strip leading zero
    return f"{d.strftime('%b')} {d.day}, {d.year} {h}:{d.strftime('%M:%S')}{d.strftime('%p').lower()} ET"


# ─────────────────────────────────────────────────────────────────────────────
# Pull
# ─────────────────────────────────────────────────────────────────────────────

def pull_fut(bbg: Bbg) -> pd.DataFrame:
    print(f"  {_DIM}fut ({len(FUT_TICKERS)}){_RESET}", flush=True)
    raw = bbg.bdp(FUT_TICKERS, ['PX_LAST', 'VOLUME', 'FUT_CUR_GEN_TICKER'])
    df = (
        raw
        .rename(columns={
            'PX_LAST':            'px_last',
            'VOLUME':             'volume',
            'FUT_CUR_GEN_TICKER': 'contract',
        })
        .rename_axis('generic_ticker')
        .reset_index()
    )
    df['generic_ticker'] = df['generic_ticker'].str.replace(' Comdty', '', regex=False)
    df['ts']      = TODAY
    df['source']  = 'BGN'
    df['is_live'] = True
    return df


def pull_index(bbg: Bbg) -> pd.DataFrame:
    print(f"  {_DIM}index ({len(INDEX_TICKERS)}){_RESET}", flush=True)
    raw = _bdp_batched(bbg, INDEX_TICKERS, ['PX_LAST'])
    df = (
        raw
        .rename(columns={'PX_LAST': 'px_last'})
        .rename_axis('ticker')
        .reset_index()
    )
    df['ts']      = TODAY
    df['source']  = 'BGN'
    df['is_live'] = True
    return df


def pull_ust(bbg: Bbg, conn) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT cusip FROM sec.auctioned_securities "
            "WHERE cusip IS NOT NULL AND security_type IN ('Note', 'Bond') "
            "AND maturity_date >= %s ORDER BY cusip",
            (TODAY,)
        )
        cusips = [r[0] for r in cur.fetchall()]
    tickers = [f"{c} Govt" for c in cusips]
    print(f"  {_DIM}ust ({len(tickers)}){_RESET}", flush=True)
    raw = _bdp_batched(bbg, tickers, ['PX_LAST', 'YLD_YTM_MID'])
    df = (
        raw
        .rename(columns={'PX_LAST': 'px_last', 'YLD_YTM_MID': 'yld_ytm_mid'})
        .rename_axis('ticker')
        .reset_index()
    )
    df['cusip']   = df['ticker'].str.replace(' Govt', '', regex=False)
    df['ts']      = TODAY
    df['source']  = 'BGN'
    df['is_live'] = True
    return df


def pull_strips(bbg: Bbg, conn) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute("SELECT cusip FROM sec.strips WHERE maturity_date > %s", (TODAY,))
        cusips = [r[0] for r in cur.fetchall()]
    if not cusips:
        return pd.DataFrame()
    tickers = [f"{c} Govt" for c in cusips]
    print(f"  {_DIM}strips ({len(tickers)}){_RESET}", flush=True)
    raw = _bdp_batched(bbg, tickers, ['PX_LAST', 'YLD_YTM_MID'])
    df = (
        raw
        .rename(columns={'PX_LAST': 'px_last', 'YLD_YTM_MID': 'yld_ytm_mid'})
        .rename_axis('ticker')
        .reset_index()
    )
    df['cusip']   = df['ticker'].str.replace(' Govt', '', regex=False)
    df['ts']      = TODAY
    df['source']  = 'BGN'
    df['is_live'] = True
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Write
# ─────────────────────────────────────────────────────────────────────────────

def write_fut(conn, df: pd.DataFrame) -> int:
    rows = [
        (row.ts, row.generic_ticker, row.contract, _n(row.px_last), _n(row.volume), row.source, row.is_live)
        for row in df.itertuples()
    ]
    with conn.cursor() as cur:
        cur.executemany("""
            INSERT INTO md.fut_eod (ts, generic_ticker, contract, px_last, volume, source, is_live)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (contract, ts) DO UPDATE SET
                generic_ticker = EXCLUDED.generic_ticker,
                px_last        = EXCLUDED.px_last,
                volume         = EXCLUDED.volume,
                source         = EXCLUDED.source,
                is_live        = EXCLUDED.is_live
        """, rows)
    conn.commit()
    return len(rows)


def write_index(conn, df: pd.DataFrame) -> int:
    rows = [
        (row.ts, row.ticker, _n(row.px_last), row.source, row.is_live)
        for row in df.itertuples()
    ]
    with conn.cursor() as cur:
        cur.executemany("""
            INSERT INTO md.index_eod (ts, ticker, px_last, source, is_live)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (ticker, ts) DO UPDATE SET
                px_last = EXCLUDED.px_last,
                source  = EXCLUDED.source,
                is_live = EXCLUDED.is_live
        """, rows)
    conn.commit()
    return len(rows)


def write_ust(conn, df: pd.DataFrame) -> int:
    rows = [
        (row.ts, row.cusip, _n(row.px_last), _n(row.yld_ytm_mid), row.source, row.is_live)
        for row in df.itertuples()
    ]
    with conn.cursor() as cur:
        cur.executemany("""
            INSERT INTO md.ust_eod (ts, cusip, px_last, yld_ytm_mid, source, is_live)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (cusip, ts) DO UPDATE SET
                px_last     = EXCLUDED.px_last,
                yld_ytm_mid = EXCLUDED.yld_ytm_mid,
                source      = EXCLUDED.source,
                is_live     = EXCLUDED.is_live
        """, rows)
    conn.commit()
    return len(rows)


def write_strips(conn, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    rows = [
        (row.ts, row.cusip, _n(row.px_last), _n(row.yld_ytm_mid), row.source, row.is_live)
        for row in df.itertuples()
    ]
    with conn.cursor() as cur:
        cur.executemany("""
            INSERT INTO md.strips_eod (ts, cusip, px_last, yld_ytm_mid, source, is_live)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (cusip, ts) DO UPDATE SET
                px_last     = EXCLUDED.px_last,
                yld_ytm_mid = EXCLUDED.yld_ytm_mid,
                source      = EXCLUDED.source,
                is_live     = EXCLUDED.is_live
        """, rows)
    conn.commit()
    return len(rows)


# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--level', type=int, default=1, choices=[1, 2])
    parser.add_argument('--write', action='store_true', help='Write to DB (default: dry run)')
    args = parser.parse_args()

    bbg = Bbg()
    conn = psycopg.connect(DB_DSN) if (args.level >= 2 or args.write) else None

    try:
        fut_df   = pull_fut(bbg)
        index_df = pull_index(bbg)
        ust_df   = strips_df = None

        if args.level >= 2:
            ust_df    = pull_ust(bbg, conn)
            strips_df = pull_strips(bbg, conn)

        if args.write:
            total  = write_fut(conn, fut_df)
            total += write_index(conn, index_df)
            if args.level >= 2:
                total += write_ust(conn, ust_df)
                total += write_strips(conn, strips_df)
        else:
            total = None

        # ── summary ──────────────────────────────────────────────────────────
        now_et  = dt.datetime.now(ET)
        idx     = index_df.set_index('ticker')['px_last']
        ten_yr  = idx.get('USGG10YR Index')
        spx     = idx.get('SPX Index')
        rows_str = f"  {_DIM}{total} rows written{_RESET}" if total is not None else f"  {_DIM}dry run{_RESET}"
        print(f"\n  {_BOLD}{_fmt_et(now_et)}{_RESET}")
        print(f"  {_CYAN}10Y{_RESET}  {_BOLD}{ten_yr:.3f}{_RESET}    {_CYAN}SPX{_RESET}  {_BOLD}{spx:,.0f}{_RESET}")
        print(rows_str)

    finally:
        if conn:
            conn.close()
        bbg.session.stop()


if __name__ == '__main__':
    main()
