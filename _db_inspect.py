import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.helpers import query_db

df = query_db("""
    SELECT contract, COUNT(*) AS rows, MIN(ts) AS lo, MAX(ts) AS hi
    FROM md.fut_eod
    --WHERE contract LIKE 'SFR%'
    --OR contract LIKE 'FF%'
    WHERE contract like '%Z6'
    GROUP BY contract
""")

if df.empty:
    print("No data found.")
else:
    print(df.to_string(index=False))
