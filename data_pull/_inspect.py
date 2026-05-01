import psycopg

conn = psycopg.connect('postgresql://benjils:snickers@raptor:5432/markets')
cur = conn.cursor()

tables = ['md.fut_eod', 'md.ust_eod', 'md.index_eod', 'md.strips_eod', 'md.headline', 'md.breakeven', 'sec.auctioned_securities', 'sec.fut_contracts']
for t in tables:
    schema, tbl = t.split('.')
    cur.execute(
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_schema=%s AND table_name=%s ORDER BY ordinal_position",
        (schema, tbl)
    )
    cols = cur.fetchall()
    cur.execute(f"SELECT COUNT(*), MIN(ts), MAX(ts) FROM {t}") if 'ts' in [c[0] for c in cols] else None
    stats = cur.fetchone() if 'ts' in [c[0] for c in cols] else None
    print(f"\n{t}:")
    for c in cols:
        print(f"  {c[0]:35s} {c[1]}")
    if stats:
        print(f"  --> {stats[0]:,} rows  {stats[1]} to {stats[2]}")

conn.close()
