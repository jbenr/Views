#!/usr/bin/env python3
"""
Pull daily settlements, open interest and volume for Treasury futures options from Bloomberg.

- Stores into md.opt_eod (Timescale hypertable), keyed by (ticker, ts)
- Quarterly options and the two serial months before them, all on the quarterly future
  (TYF7 and TYG7 options exercise into TYH7): kind 'standard'.
- Weekly options (kind 'weekly'): Monday VBY, Tuesday TJP, Wednesday TYW (chain TYY),
  Thursday TJW and Friday 1M-5M, with their own tickers (VBYX26C2 107, 1MX6C 107). A weekly
  exercises into the quarterly future whose quarterly options expire next on or after it
  (a weekly after the December options' expiry goes into March).
- Bloomberg keeps history of price, open interest and volume per option but not of
  implied vol or greeks; those are derived from settlements by utils/options.py
  (data_pull/build_opt_exposure.py stores them).
- Daily: the live chains of the first two futures, strikes within LIVE_STRIKE_RANGE points
  of the futures price, from each underlying's last stored day (with overlap); live weekly
  chains within WEEKLY_STRIKE_RANGE, over their whole (short) lives.
- Backfill: past quarterly futures. Option tickers are built on a strike grid around the
  range the future traded over the options' life; strikes that never listed come back
  invalid and are skipped. Bloomberg caps distinct securities per month, so the grid is
  kept tight.
- Backfill-weekly: expired weeklies from a date, tickers built from each weekday's position
  in its month (the n-th Friday is nM, the n-th Monday VBY..n), strikes within
  WEEKLY_STRIKE_RANGE of where the future settled over the weekly's last days. Each
  expiry costs about 2 x (2 x range / 0.25 + 1) distinct securities against the cap.

Usage:
    python pull_options.py                                # daily update of live chains
    python pull_options.py backfill 2025-01-01            # past expiries from this date
    python pull_options.py backfill 2025-01-01 FV         # another root
    python pull_options.py backfill-weekly 2026-05-01     # expired TY weeklies from this date
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

# Weekly options per futures root: the chain roots to list live weeklies from, and the ticker
# prefix by weekday for expired ones. The n-th Friday's weekly is nM; Monday to Thursday carry
# the week number after the put/call letter.
WEEKLY_CHAINS = {"TY": ["1M", "2M", "3M", "4M", "5M", "TYY", "VBY", "TJP", "TJW"]}
WEEKLY_DAYS = {"TY": {0: "VBY", 1: "TJP", 2: "TYW", 3: "TJW", 4: "M"}}
WEEKLY_STRIKE_RANGE = 2.0            # points either side of the futures price
WEEKLY_LIFE_DAYS = 40                # history pulled for a weekly (they list about 5 weeks out)
WEEKLY_PRICE_DAYS = 10               # backfill: strikes around the future's range over these days

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
                kind            text NOT NULL DEFAULT 'standard',   -- standard | weekly
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
        cur.execute("ALTER TABLE md.opt_eod ADD COLUMN IF NOT EXISTS kind text NOT NULL DEFAULT 'standard';")
        cur.execute("CREATE INDEX IF NOT EXISTS ix_opt_eod_underlying ON md.opt_eod (underlying, ts);")
    conn.commit()


def upsert(conn, rows: list[tuple]) -> int:
    if not rows:
        return 0
    sql = """
    INSERT INTO md.opt_eod (ts, ticker, root, underlying, expiry, strike, put_call, settle, open_interest, volume, kind)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (ticker, ts) DO UPDATE SET
      settle = EXCLUDED.settle, open_interest = EXCLUDED.open_interest, volume = EXCLUDED.volume,
      expiry = EXCLUDED.expiry, underlying = EXCLUDED.underlying, kind = EXCLUDED.kind, updated_at = now();
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


def quarterly_expiries(conn, root: str) -> list[tuple[dt.date, str]]:
    """(expiry of the quarterly options, underlying) per stored quarterly future, by expiry."""
    with conn.cursor() as cur:
        cur.execute("SELECT max(expiry), underlying FROM md.opt_eod WHERE root = %s AND kind = 'standard' "
                    "GROUP BY underlying ORDER BY 1", (root,))
        return cur.fetchall()


def weekly_underlying(expiry: dt.date, quarterlies: list[tuple[dt.date, str]]) -> str | None:
    """The quarterly future a weekly exercises into: the first whose quarterly options expire on or after it."""
    return next((und for q_expiry, und in quarterlies if q_expiry >= expiry), None)


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


def standard_meta(expiry_by_ticker: dict[str, dt.date]) -> dict[str, dict]:
    """Ticker -> info for quarterly / serial options, from their tickers (unparseable ones dropped)."""
    out = {}
    for ticker, expiry in expiry_by_ticker.items():
        info = parse_ticker(ticker, expiry)
        if info:
            out[ticker] = {**info, "expiry": expiry, "kind": "standard"}
    return out


def weekly_meta(bbg: Bbg, tickers: list[str], root: str, quarterlies) -> dict[str, dict]:
    """Ticker -> info for weekly options, from Bloomberg's reference fields (invalid tickers dropped)."""
    out = {}
    for i in range(0, len(tickers), BATCH):
        data = bbg.bdp(tickers[i:i + BATCH], ["OPT_EXPIRE_DT", "OPT_PUT_CALL", "OPT_STRIKE_PX"])
        if data is None or data.empty or "OPT_EXPIRE_DT" not in data:
            continue
        for ticker, rec in data.dropna(subset=["OPT_EXPIRE_DT"]).iterrows():
            expiry = pd.to_datetime(rec["OPT_EXPIRE_DT"]).date()
            underlying = weekly_underlying(expiry, quarterlies)
            if underlying is None or pd.isna(rec.get("OPT_STRIKE_PX")):
                continue
            out[ticker] = {"root": root, "pc": str(rec["OPT_PUT_CALL"])[0].upper(), "strike": float(rec["OPT_STRIKE_PX"]),
                           "underlying": underlying, "expiry": expiry, "kind": "weekly"}
    return out


def pull(bbg: Bbg, conn, meta: dict[str, dict], start: dt.date, end: dt.date) -> int:
    """BDH settle / open interest / volume for option tickers with known info, upserted."""
    rows_total = 0
    tickers = list(meta)
    for i in tqdm(range(0, len(tickers), BATCH), desc="option batches"):
        batch = tickers[i:i + BATCH]
        data = bbg.bdh(batch, FIELDS, start, end)
        rows = []
        for ticker, frame in data.items():
            info = meta.get(ticker)
            if info is None or frame.empty or "error" in frame:
                continue
            frame = frame.reindex(columns=FIELDS)
            for day, rec in frame.iterrows():
                if pd.isna(rec["PX_SETTLE"]) and pd.isna(rec["OPEN_INT"]):
                    continue
                rows.append((day, ticker, info["root"], info["underlying"], info["expiry"], info["strike"],
                             info["pc"], _num(rec["PX_SETTLE"]), _num(rec["OPEN_INT"]), _num(rec["VOLUME"]),
                             info["kind"]))
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
        meta = standard_meta(expiries(bbg, near))
        underlying = future_key(contract, today)
        last = last_stored(conn, underlying)
        start = ((last - dt.timedelta(days=OVERLAP_DAYS)) if last
                 else min(m["expiry"] for m in meta.values()) - dt.timedelta(days=LIFE_DAYS))
        print(f"{gen} = {contract} @ {px:.3f}: {len(meta)} options within {LIVE_STRIKE_RANGE} pts, from {start}")
        n = pull(bbg, conn, meta, start, today)
        print(f"  ✅ {n:,} rows")
    if root in WEEKLY_CHAINS:
        weekly_daily(bbg, conn, root, [str(ref.loc[g, "FUT_CUR_GEN_TICKER"]) for g in gens],
                     float(ref.loc[gens[0], "PX_LAST"]))


def _strike(ticker: str) -> float | None:
    try:
        return float(ticker.split(" ")[1])
    except (IndexError, ValueError):
        return None


def weekly_daily(bbg: Bbg, conn, root: str, contracts: list[str], px: float) -> None:
    """Live weekly chains on the first two futures, strikes within WEEKLY_STRIKE_RANGE, over their whole lives."""
    today = dt.date.today()
    near = []
    for chain_root in WEEKLY_CHAINS[root]:
        for contract in contracts:  # TYZ6 -> 1MZ6 Comdty: the weeklies listed on that future
            gen = f"{chain_root}{contract[len(root):]} Comdty"
            try:
                chain = bbg.bds([gen], "OPT_CHAIN")[gen].iloc[:, 0].str.replace(r"\s+", " ", regex=True).tolist()
            except (KeyError, IndexError):
                continue
            near += [t for t in chain if t.endswith(" Comdty") and _strike(t) is not None
                     and abs(_strike(t) - px) <= WEEKLY_STRIKE_RANGE]
    meta = weekly_meta(bbg, sorted(set(near)), root, quarterly_expiries(conn, root))
    print(f"{root} weeklies @ {px:.3f}: {len(meta)} options within {WEEKLY_STRIKE_RANGE} pts")
    n = pull(bbg, conn, meta, today - dt.timedelta(days=WEEKLY_LIFE_DAYS), today)
    print(f"  ✅ {n:,} rows")


def weekly_tickers(root: str, expiry: dt.date, strikes: list[float]) -> list[str]:
    """Tickers of the weekly expiring on ``expiry``, built from its weekday and week of the month."""
    prefix = WEEKLY_DAYS[root].get(expiry.weekday())
    if prefix is None:
        return []
    n = (expiry.day - 1) // 7 + 1
    code = MONTH_CODES[expiry.month - 1]
    if prefix == "M":  # Friday: nM + month + year digit, e.g. 1MX6C 107
        return [f"{n}M{code}{expiry.year % 10}{pc} {k:g} Comdty" for pc in "CP" for k in strikes]
    return [f"{prefix}{code}{expiry.year % 100:02d}{pc}{n} {k:g} Comdty" for pc in "CP" for k in strikes]


def backfill_weekly(bbg: Bbg, conn, root: str, since: dt.date) -> None:
    """Expired weeklies from ``since``: every weekday that is not a standard expiry, strikes near the future."""
    futs = future_prices(conn, root)
    quarterlies = quarterly_expiries(conn, root)
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT expiry FROM md.opt_eod WHERE root = %s AND kind = 'standard'", (root,))
        standard = {r[0] for r in cur.fetchall()}
    today = dt.date.today()
    day = since
    while day < today:
        und = weekly_underlying(day, quarterlies)
        if day.weekday() < 5 and day not in standard and und:
            window = futs[(futs["underlying"] == und) & (futs["ts"] <= day)
                          & (futs["ts"] >= day - dt.timedelta(days=WEEKLY_PRICE_DAYS))]
            if not window.empty:
                lo, hi = window["px"].min() - WEEKLY_STRIKE_RANGE, window["px"].max() + WEEKLY_STRIKE_RANGE
                strikes = [k * GRID_FINE for k in range(math.ceil(lo / GRID_FINE), math.floor(hi / GRID_FINE) + 1)]
                meta = {t: m for t, m in weekly_meta(bbg, weekly_tickers(root, day, strikes), root,
                                                     quarterlies).items() if m["expiry"] == day}
                if meta:
                    n = pull(bbg, conn, meta, day - dt.timedelta(days=WEEKLY_LIFE_DAYS), day)
                    print(f"{root} weekly {day}: {len(meta)} options, {n:,} rows")
        day += dt.timedelta(days=1)


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
        meta = standard_meta(expiries(bbg, tickers))
        if not meta:
            print(f"{underlying}: no valid options")
            continue
        start = min(m["expiry"] for m in meta.values()) - dt.timedelta(days=LIFE_DAYS)
        end = min(max(m["expiry"] for m in meta.values()), today)
        print(f"{underlying}: {len(meta)} of {len(tickers)} tickers valid, {start} to {end}")
        n = pull(bbg, conn, meta, start, end)
        print(f"  ✅ {n:,} rows")


def main() -> None:
    args = sys.argv[1:]
    mode = args[0].lower() if args else "daily"
    if mode not in {"daily", "backfill", "backfill-weekly"}:
        sys.exit(f"Unknown mode {mode!r}: use daily, backfill START [ROOT] or backfill-weekly START [ROOT].")
    conn, bbg = get_conn(), Bbg()
    ensure_table(conn)
    if mode in {"backfill", "backfill-weekly"}:
        if len(args) < 2:
            sys.exit(f"{mode} needs a start date, e.g. {mode} 2025-01-01")
        since = dt.date.fromisoformat(args[1])
        for root in (args[2:] or ROOTS):
            if mode == "backfill":
                backfill(bbg, conn, root.upper(), since)
            elif root.upper() in WEEKLY_DAYS:
                backfill_weekly(bbg, conn, root.upper(), since)
            else:
                sys.exit(f"No weekly tickers known for {root}.")
    else:
        for root in ROOTS:
            daily(bbg, conn, root)
    conn.close()


if __name__ == "__main__":
    main()
