#!/usr/bin/env python3
"""
Pull daily settlements, open interest and volume for Treasury futures options from Bloomberg.

- Stores into md.opt_eod (Timescale hypertable), keyed by (ticker, ts)
- Quarterly options and the two serial months before them, all on the quarterly future
  (TYF7 and TYG7 options exercise into TYH7). Weekly options are not pulled.
- Bloomberg keeps history of price, open interest and volume per option but not of
  implied vol or greeks; those are derived from settlements in research/gamma.py.
- Daily: the live chains of the first two futures, strikes within LIVE_STRIKE_RANGE points
  of the futures price, from each underlying's last stored day (with overlap).
- Backfill: past quarterly futures. Option tickers are built on a strike grid around the
  range the future traded over the options' life; strikes that never listed come back
  invalid and are skipped. Bloomberg caps distinct securities per month, so the grid is
  kept tight.

Usage:
    python pull_options.py                         # daily update of live chains
    python pull_options.py backfill 2025-01-01     # past expiries from this date
    python pull_options.py backfill 2025-01-01 FV  # another root
"""

from __future__ import annotations

import datetime as dt
import math
import os
import re
import sys

import pandas as pd
import psycopg
from tqdm import tqdm

from berg import Bbg

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DB_DSN = os.getenv("DB_DSN", "postgresql://benjils:snickers@raptor:5432/markets")

ROOTS = ["TY"]                       # daily update; any root with quarterly futures in md.fut_eod works
MONTH_CODES = "FGHJKMNQUVXZ"
QUARTERLY = {"H": "FGH", "M": "JKM", "U": "NQU", "Z": "VXZ"}  # future -> option months exercising into it
FIELDS = ["PX_SETTLE", "OPEN_INT", "VOLUME"]
LIFE_DAYS = 200                      # history pulled before each expiry
LIVE_STRIKE_RANGE = 4.0              # points either side of the futures price, daily chains
GRID_FINE, GRID_COARSE = 0.25, 0.5   # strike steps inside / outside the traded range
GRID_MARGIN = 2.0                    # points beyond the traded range, coarse steps
BATCH = 50
OVERLAP_DAYS = 7

TICKER_RE = re.compile(r"^(?P<root>[A-Z]+?)(?P<code>[FGHJKMNQUVXZ])(?P<y>\d)(?P<pc>[CP])\s+(?P<strike>[\d.]+)\s+Comdty$")


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def get_conn():
    return psycopg.connect(DB_DSN)


def ensure_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("CREATE SCHEMA IF NOT EXISTS md;")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS md.opt_eod (
                ts              date NOT NULL,
                ticker          text NOT NULL,
                root            text NOT NULL,
                underlying      text NOT NULL,   -- quarterly future with a 2-digit year, e.g. TYU25
                expiry          date NOT NULL,
                strike          double precision NOT NULL,
                put_call        char(1) NOT NULL,
                settle          double precision,
                open_interest   double precision,
                volume          double precision,
                source          text NOT NULL DEFAULT 'BGN',
                created_at      timestamptz NOT NULL DEFAULT now(),
                updated_at      timestamptz NOT NULL DEFAULT now(),
                PRIMARY KEY (ticker, ts)
            );
            """
        )
        cur.execute("SELECT create_hypertable('md.opt_eod', 'ts', if_not_exists => TRUE);")
        cur.execute("CREATE INDEX IF NOT EXISTS ix_opt_eod_underlying ON md.opt_eod (underlying, ts);")
    conn.commit()


def upsert(conn, rows: list[tuple]) -> int:
    if not rows:
        return 0
    sql = """
    INSERT INTO md.opt_eod (ts, ticker, root, underlying, expiry, strike, put_call, settle, open_interest, volume)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (ticker, ts) DO UPDATE SET
      settle = EXCLUDED.settle, open_interest = EXCLUDED.open_interest, volume = EXCLUDED.volume,
      expiry = EXCLUDED.expiry, underlying = EXCLUDED.underlying, updated_at = now();
    """
    with conn.cursor() as cur:
        cur.executemany(sql, rows)
    conn.commit()
    return len(rows)


def future_prices(conn, root: str) -> pd.DataFrame:
    """Every stored day of the root's quarterly futures, keyed by a 2-digit-year contract (TYU25)."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT ts, contract, px_last::float FROM md.fut_eod WHERE contract ~ %s AND px_last IS NOT NULL",
            (f"^{root}[HMUZ][0-9]{{1,2}}$",),
        )
        df = pd.DataFrame(cur.fetchall(), columns=["ts", "contract", "px"])
    df["underlying"] = [future_key(c, ts) for c, ts in zip(df["contract"], df["ts"])]
    return df


def last_stored(conn, underlying: str) -> dt.date | None:
    with conn.cursor() as cur:
        cur.execute("SELECT max(ts) FROM md.opt_eod WHERE underlying = %s", (underlying,))
        return cur.fetchone()[0]


# ---------------------------------------------------------------------------
# Tickers
# ---------------------------------------------------------------------------

def future_key(contract: str, ts: dt.date) -> str:
    """md.fut_eod names recent contracts with one year digit (TYU5) and older ones with two (TYU24)."""
    m = re.match(r"^([A-Z]+?)([HMUZ])(\d{1,2})$", contract)
    root, code, yy = m.groups()
    if len(yy) == 1:  # the decade a 1-digit contract trades in: the nearest year at or after ts
        year = ts.year - ts.year % 10 + int(yy)
        if year < ts.year - 1:
            year += 10
        yy = f"{year % 100:02d}"
    return f"{root}{code}{yy}"


def option_ticker(root: str, code: str, year: int, pc: str, strike: float) -> str:
    return f"{root}{code}{year % 10}{pc} {strike:g} Comdty"


def parse_ticker(ticker: str, expiry: dt.date) -> dict | None:
    """Root, put/call, strike and the quarterly underlying (2-digit year) of an option ticker."""
    m = TICKER_RE.match(re.sub(r"\s+", " ", ticker))
    if not m:
        return None
    code = m["code"]
    fut = next(q for q, months in QUARTERLY.items() if code in months)
    # the option's month is at or after its expiry month (serials expire the month before)
    year = expiry.year + (1 if MONTH_CODES.index(code) + 1 < expiry.month else 0)
    if year % 10 != int(m["y"]):
        return None
    return dict(root=m["root"], pc=m["pc"], strike=float(m["strike"]), underlying=f"{m['root']}{fut}{year % 100:02d}")


def strike_grid(lo: float, hi: float) -> list[float]:
    """Quarter-point strikes across the traded range, half-points GRID_MARGIN beyond it."""
    fine = [k * GRID_FINE for k in range(math.floor(lo / GRID_FINE), math.ceil(hi / GRID_FINE) + 1)]
    coarse = [k * GRID_COARSE for k in range(math.floor((lo - GRID_MARGIN) / GRID_COARSE),
                                              math.ceil((hi + GRID_MARGIN) / GRID_COARSE) + 1)]
    return sorted(set(fine) | set(coarse))


# ---------------------------------------------------------------------------
# Bloomberg fetch
# ---------------------------------------------------------------------------

def expiries(bbg: Bbg, tickers: list[str]) -> dict[str, dt.date]:
    """Valid option tickers and their expiry dates (invalid strikes are dropped)."""
    out = {}
    for i in range(0, len(tickers), BATCH):
        data = bbg.bdp(tickers[i:i + BATCH], ["OPT_EXPIRE_DT"])
        if data is None or data.empty or "OPT_EXPIRE_DT" not in data:
            continue
        for ticker, value in data["OPT_EXPIRE_DT"].dropna().items():
            out[ticker] = pd.to_datetime(value).date()
    return out


def pull(bbg: Bbg, conn, meta: dict[str, dt.date], start: dt.date, end: dt.date) -> int:
    """BDH settle / open interest / volume for option tickers with known expiries, upserted."""
    rows_total = 0
    tickers = list(meta)
    for i in tqdm(range(0, len(tickers), BATCH), desc="option batches"):
        batch = tickers[i:i + BATCH]
        data = bbg.bdh(batch, FIELDS, start, end)
        rows = []
        for ticker, frame in data.items():
            info = parse_ticker(ticker, meta[ticker])
            if info is None or frame.empty or "error" in frame:
                continue
            frame = frame.reindex(columns=FIELDS)
            for day, rec in frame.iterrows():
                if pd.isna(rec["PX_SETTLE"]) and pd.isna(rec["OPEN_INT"]):
                    continue
                rows.append((day, ticker, info["root"], info["underlying"], meta[ticker], info["strike"],
                             info["pc"], _num(rec["PX_SETTLE"]), _num(rec["OPEN_INT"]), _num(rec["VOLUME"])))
        rows_total += upsert(conn, rows)
    return rows_total


def _num(v):
    return None if pd.isna(v) else float(v)


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def daily(bbg: Bbg, conn, root: str) -> None:
    """Live chains of the first two futures, near-the-money strikes, from each underlying's last stored day."""
    gens = [f"{root}{n} Comdty" for n in (1, 2)]
    ref = bbg.bdp(gens, ["FUT_CUR_GEN_TICKER", "PX_LAST"])
    today = dt.date.today()
    for gen in gens:
        contract, px = str(ref.loc[gen, "FUT_CUR_GEN_TICKER"]), float(ref.loc[gen, "PX_LAST"])
        chain = bbg.bds([gen], "OPT_CHAIN")[gen].iloc[:, 0].str.replace(r"\s+", " ", regex=True).tolist()
        near = [t for t in chain if (m := TICKER_RE.match(t)) and abs(float(m["strike"]) - px) <= LIVE_STRIKE_RANGE]
        meta = expiries(bbg, near)
        underlying = future_key(contract, today)
        last = last_stored(conn, underlying)
        start = (last - dt.timedelta(days=OVERLAP_DAYS)) if last else min(meta.values()) - dt.timedelta(days=LIFE_DAYS)
        print(f"{gen} = {contract} @ {px:.3f}: {len(meta)} options within {LIVE_STRIKE_RANGE} pts, from {start}")
        n = pull(bbg, conn, meta, start, today)
        print(f"  ✅ {n:,} rows")


def backfill(bbg: Bbg, conn, root: str, since: dt.date) -> None:
    """Every quarterly future whose options expired from ``since`` on, using a strike grid around its traded range."""
    futs = future_prices(conn, root)
    today = dt.date.today()
    for underlying, prices in sorted(futs.groupby("underlying"), key=lambda kv: (kv[0][-2:], "HMUZ".index(kv[0][-3]))):
        code, year = underlying[-3], 2000 + int(underlying[-2:])
        delivery = dt.date(year, MONTH_CODES.index(code) + 1, 1)
        if delivery < since or delivery > today + dt.timedelta(days=200):
            continue
        months = QUARTERLY[code]
        window = prices[prices["ts"] >= delivery - dt.timedelta(days=LIFE_DAYS + 100)]
        strikes = strike_grid(window["px"].min(), window["px"].max())
        tickers = [option_ticker(root, m, year, pc, k) for m in months for pc in "CP" for k in strikes]
        meta = {t: e for t, e in expiries(bbg, tickers).items() if parse_ticker(t, e)}
        if not meta:
            print(f"{underlying}: no valid options")
            continue
        start = min(meta.values()) - dt.timedelta(days=LIFE_DAYS)
        end = min(max(meta.values()), today)
        print(f"{underlying}: {len(meta)} of {len(tickers)} tickers valid, {start} to {end}")
        n = pull(bbg, conn, meta, start, end)
        print(f"  ✅ {n:,} rows")


def main() -> None:
    args = sys.argv[1:]
    mode = args[0].lower() if args else "daily"
    if mode not in {"daily", "backfill"}:
        sys.exit(f"Unknown mode {mode!r}: use daily or backfill START [ROOT].")
    conn, bbg = get_conn(), Bbg()
    ensure_table(conn)
    if mode == "backfill":
        if len(args) < 2:
            sys.exit("backfill needs a start date, e.g. backfill 2025-01-01")
        since = dt.date.fromisoformat(args[1])
        for root in (args[2:] or ROOTS):
            backfill(bbg, conn, root.upper(), since)
    else:
        for root in ROOTS:
            daily(bbg, conn, root)
    conn.close()


if __name__ == "__main__":
    main()
