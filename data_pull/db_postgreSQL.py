import psycopg2
import pandas as pd
from utils import pdf

dsn = "postgresql://benjils:snickers@localhost:5432/markets"

conn = psycopg2.connect(dsn)

df = pd.read_sql(
    """
    SELECT cusip, 
    auction_date::date, issue_date::date, maturity_date::date,
    int_rate, original_security_term, security_type, reopening,
    offering_amt, inflation_index_security
    FROM auctioned_securities a
    WHERE a.security_type != 'Bill'
    and a.floating_rate = 'No'
    --and a.inflation_index_security = 'Yes'
    ORDER BY a.auction_date DESC
    LIMIT 40;
    """, conn)

# for c in df.columns:
#     if "date" in c:
#         df[c] = pd.to_datetime(df[c], errors="coerce").dt.date

pdf(df)
# print(df)
