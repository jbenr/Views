#!/usr/bin/env python3
"""
Terms and daily closes for the government bonds deliverable into non-US futures.

The bonds are the ISINs in sec.fut_contracts.dlv_basket (pull_dlv_basket.py).
Each is pulled over the window it matters for a basket: from nine months before
its first contract month to its last contract's final delivery day. A bond
already stored resumes from its last close, with a short overlap.

- sec.govt_bonds : one row per ISIN — coupon, maturity, dates, conventions
- md.govt_eod    : daily clean price and yield, keyed by (isin, ts)

Usage:
    python pull_govt_bonds.py
"""

from __future__ import annotations

import datetime as dt
import os
import sys

import pandas as pd
import psycopg
from tqdm import tqdm

sys.path.append(os.path.expanduser("~/werk/Views"))
from data_pull.berg import Bbg

DB_DSN = os.getenv("DB_DSN", "postgresql://benjils:snickers@raptor:5432/markets")
TERM_FIELDS = ["SECURITY_DES", "TICKER", "COUNTRY_ISO", "CRNCY", "CPN", "MATURITY", "ISSUE_DT", "INT_ACC_DT",
               "FIRST_CPN_DT", "CPN_FREQ", "DAY_CNT_DES", "DAYS_TO_SETTLE", "EX_DIV_DAYS"]
EOD_FIELDS = ["PX_LAST", "YLD_YTM_MID"]
BATCH_SIZE = 25
LEAD_DAYS = 270   # history kept before a bond's first contract month
OVERLAP_DAYS = 7


def ensure_tables(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sec.govt_bonds (
                isin              text PRIMARY KEY,
                name              text,
                ticker            text,
                country           text,
                crncy             text,
                coupon            double precision,
                maturity          date,
                issue_date        date,
                int_acc_date      date,
                first_coupon_date date,
                cpn_freq          integer,
                day_count         text,
                days_to_settle    integer,
                ex_div_days       integer,
                updated_at        timestamptz NOT NULL DEFAULT now()
            );
            CREATE TABLE IF NOT EXISTS md.govt_eod (
                ts          date NOT NULL,
                isin        text NOT NULL,
                px_last     double precision,
                yld_ytm_mid double precision,
                source      text NOT NULL DEFAULT 'BGN',
                created_at  timestamptz NOT NULL DEFAULT now(),
                PRIMARY KEY (isin, ts)
            );
            CREATE INDEX IF NOT EXISTS govt_eod_ts_idx ON md.govt_eod (ts DESC);
        """)
    conn.commit()


def bond_windows(conn) -> pd.DataFrame:
    """Each deliverable ISIN with the dates to pull: isin, start, end."""
    with conn.cursor() as cur:
        cur.execute("""
            WITH baskets AS (
                SELECT d->>'isin' AS isin, contract_month, last_delivery_date
                FROM sec.fut_contracts, jsonb_array_elements(dlv_basket->'deliverables') d
                WHERE d ? 'isin'
            )
            SELECT b.isin,
                   GREATEST(min(b.contract_month) - %s, COALESCE(max(e.last_ts) - %s, DATE '1900-01-01')) AS start,
                   LEAST(max(b.last_delivery_date), CURRENT_DATE) AS "end"
            FROM baskets b
            LEFT JOIN (SELECT isin, max(ts) AS last_ts FROM md.govt_eod GROUP BY isin) e USING (isin)
            GROUP BY b.isin
        """, (LEAD_DAYS, OVERLAP_DAYS))
        rows = cur.fetchall()
    windows = pd.DataFrame(rows, columns=["isin", "start", "end"])
    return windows[windows["start"] <= windows["end"]].sort_values("start").reset_index(drop=True)


def to_date(value):
    return None if value is None or pd.isna(value) else pd.to_datetime(value).date()


def to_int(value):
    return None if value is None or pd.isna(value) else int(value)


def fetch_terms(bbg: Bbg, isins: list[str]) -> list[tuple]:
    """One row per ISIN for sec.govt_bonds."""
    rows = []
    for i in tqdm(range(0, len(isins), 100), desc="Bond terms", unit="batch"):
        batch = isins[i:i + 100]
        data = bbg.bdp([f"{isin} Govt" for isin in batch], TERM_FIELDS)
        for isin in batch:
            if f"{isin} Govt" not in data.index:
                continue
            r = data.loc[f"{isin} Govt"]
            rows.append((isin, r.get("SECURITY_DES"), r.get("TICKER"), r.get("COUNTRY_ISO"), r.get("CRNCY"),
                         None if pd.isna(r.get("CPN")) else float(r.get("CPN")), to_date(r.get("MATURITY")),
                         to_date(r.get("ISSUE_DT")), to_date(r.get("INT_ACC_DT")), to_date(r.get("FIRST_CPN_DT")),
                         to_int(r.get("CPN_FREQ")), r.get("DAY_CNT_DES"), to_int(r.get("DAYS_TO_SETTLE")),
                         to_int(r.get("EX_DIV_DAYS"))))
    return rows


def upsert_terms(conn, rows: list[tuple]) -> int:
    with conn.cursor() as cur:
        cur.executemany("""
            INSERT INTO sec.govt_bonds (isin, name, ticker, country, crncy, coupon, maturity, issue_date,
                                        int_acc_date, first_coupon_date, cpn_freq, day_count, days_to_settle,
                                        ex_div_days, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
            ON CONFLICT (isin) DO UPDATE SET
                name = EXCLUDED.name, ticker = EXCLUDED.ticker, country = EXCLUDED.country, crncy = EXCLUDED.crncy,
                coupon = EXCLUDED.coupon, maturity = EXCLUDED.maturity, issue_date = EXCLUDED.issue_date,
                int_acc_date = EXCLUDED.int_acc_date, first_coupon_date = EXCLUDED.first_coupon_date,
                cpn_freq = EXCLUDED.cpn_freq, day_count = EXCLUDED.day_count,
                days_to_settle = EXCLUDED.days_to_settle, ex_div_days = EXCLUDED.ex_div_days, updated_at = now()
        """, rows)
    conn.commit()
    return len(rows)


def fetch_closes(bbg: Bbg, windows: pd.DataFrame) -> list[tuple]:
    """Daily closes for one batch of bonds, each cut to its own window."""
    data = bbg.bdh([f"{isin} Govt" for isin in windows["isin"]], EOD_FIELDS,
                   start=windows["start"].min(), end=windows["end"].max(), periodicity="DAILY")
    rows = []
    for w in windows.itertuples(index=False):
        df = data.get(f"{w.isin} Govt")
        if df is None or df.empty or "PX_LAST" not in df.columns:
            continue
        for ts, r in df.iterrows():
            day = pd.to_datetime(ts).date()
            if w.start <= day <= w.end and pd.notna(r["PX_LAST"]):
                yld = r.get("YLD_YTM_MID")
                rows.append((day, w.isin, float(r["PX_LAST"]), None if pd.isna(yld) else float(yld)))
    return rows


def upsert_closes(conn, rows: list[tuple]) -> int:
    with conn.cursor() as cur:
        cur.executemany("""
            INSERT INTO md.govt_eod (ts, isin, px_last, yld_ytm_mid) VALUES (%s, %s, %s, %s)
            ON CONFLICT (isin, ts) DO UPDATE SET px_last = EXCLUDED.px_last, yld_ytm_mid = EXCLUDED.yld_ytm_mid
        """, rows)
    conn.commit()
    return len(rows)


def main() -> None:
    with psycopg.connect(DB_DSN) as conn:
        ensure_tables(conn)

        # the bonds to pull and the dates each is needed for
        windows = bond_windows(conn)
        print(f"{len(windows)} deliverable bonds to update")
        if windows.empty:
            return
        bbg = Bbg()

        # terms
        terms = upsert_terms(conn, fetch_terms(bbg, windows["isin"].tolist()))
        print(f"Upserted {terms} rows into sec.govt_bonds")

        # closes, in batches of bonds with neighbouring windows
        closes = 0
        for i in tqdm(range(0, len(windows), BATCH_SIZE), desc="Bond closes", unit="batch"):
            closes += upsert_closes(conn, fetch_closes(bbg, windows.iloc[i:i + BATCH_SIZE]))
        print(f"Upserted {closes} rows into md.govt_eod")


if __name__ == "__main__":
    main()
