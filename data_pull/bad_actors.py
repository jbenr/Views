"""Find and repull suspicious rows from md.ust_eod."""
from __future__ import annotations

import psycopg
import pandas as pd
from berg import Bbg
from tqdm import tqdm

DB_DSN = "postgresql://benjils:snickers@raptor:5432/markets"

SQL = """
    SELECT ts, cusip, px_last, yld_ytm_mid
    FROM md.ust_eod
    WHERE yld_ytm_mid > 20 OR px_last > 200
    ORDER BY cusip, ts;
"""


def get_bad_rows() -> pd.DataFrame:
    with psycopg.connect(DB_DSN) as conn:
        return pd.read_sql(SQL, conn).dropna()



def repull(bad: pd.DataFrame) -> pd.DataFrame:
    bbg = Bbg()
    results = []
    for _, r in tqdm(bad.iterrows(), total=len(bad), desc="Repulling"):
        ticker = f"{r.cusip.strip()} Govt"
        d = pd.Timestamp(r.ts).strftime("%Y-%m-%d")
        try:
            data = bbg.bdh([ticker], ["PX_LAST", "YLD_YTM_MID"], start=d, end=d)
            df = data.get(ticker, pd.DataFrame())
            if df.empty or "error" in df.columns:
                px = yld = None
            else:
                px = df.iloc[0].get("PX_LAST")
                yld = df.iloc[0].get("YLD_YTM_MID")
        except Exception:
            px = yld = None
        results.append({
            "cusip": r.cusip, "ts": r.ts,
            "db_px": r.px_last, "db_yld": r.yld_ytm_mid,
            "bbg_px": px, "bbg_yld": yld,
        })
    return pd.DataFrame(results)


if __name__ == "__main__":
    bad = get_bad_rows()
    print(f"Found {len(bad)} suspicious rows")
    if not bad.empty:
        result = repull(bad)
        # result.to_csv("bad_actors_repulled.csv", index=False)
        print(result.to_string())
