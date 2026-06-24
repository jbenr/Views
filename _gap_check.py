import sys; sys.path.insert(0, 'C:/Users/benjils/werk/Views')
from utils.helpers import query_db

gaps = query_db("""
    WITH biz AS (
        SELECT d::date FROM generate_series(CURRENT_DATE - 90, CURRENT_DATE, '1 day'::interval) d
        WHERE EXTRACT(DOW FROM d) NOT IN (0,6)
    )
    SELECT b.d AS missing_date
    FROM biz b
    LEFT JOIN md.headline h ON h.ts = b.d AND h.tenor = '10-Year'
        AND h.asset_class = 'nominal' AND h.status = 'c'
    WHERE h.ts IS NULL ORDER BY b.d
""")
print(f"10Y OTR headline gaps: {len(gaps)}")
if not gaps.empty:
    print(gaps.to_string(index=False))

rows = query_db("SELECT ts FROM md.ust_eod WHERE cusip = '91282CQQ7' ORDER BY ts")
print(f"\n91282CQQ7 rows in ust_eod: {len(rows)}")
print(list(rows['ts']))
