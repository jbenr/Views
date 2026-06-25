"""
Tickers and CUSIPs that Bloomberg consistently returns no data for.
Subtract from pull universes before queuing — prevents perpetual historical re-pulls.

Add here when a series permanently fails. Remove if Bloomberg resumes publishing.
"""

# CUSIPs in sec.strips that Bloomberg cannot serve via BDH.
# Verified: queried from 2000-01-01 on multiple runs, always 0 rows returned.
STRIP_CUSIPS: frozenset[str] = frozenset({
    "912834ZQ4",  # Coupon STRIP, matures 2026-08-15
    "912834ZR2",  # Coupon STRIP, matures 2026-08-15
    "912834K80",  # Coupon STRIP, matures 2026-12-31
    "912803BN2",  # TIPS STRIP, matures 2028-04-15 — Bloomberg does not carry TIPS STRIPS
    "912803CF8",  # TIPS STRIP, matures 2029-04-15 — Bloomberg does not carry TIPS STRIPS
})

# Generic index tickers in md.index_eod that Bloomberg has discontinued.
# Verified: no new data returned since the dates shown.
INDEX_TICKERS: frozenset[str] = frozenset({
    "BF022030 Index",  # 2Y/20Y/30Y butterfly — last data 2026-05-12
    "BF032030 Index",  # 3Y/20Y/30Y butterfly — last data 2026-05-12
})
