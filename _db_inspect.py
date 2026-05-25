import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.helpers import query_db

df = query_db("""
    SELECT ticker, COUNT(*) AS rows, MIN(ts) AS lo, MAX(ts) AS hi
    FROM md.index_eod
    WHERE ticker = 'MTGEFNCL Index'
    GROUP BY ticker
""")

if df.empty:
    print("MTGEFNCL Index not found in md.index_eod — need to run the pull.")
else:
    print(df.to_string(index=False))
