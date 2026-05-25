
headline = [2,3,5,7,10,20,30]
ust = [f"USGG{x}YR Index" for x in headline]
crv = [f"USYC{a}{'Y' if a < 10 else ''}{b}{'Y' if b < 10 else ''} Index"
       for a in headline
       for b in headline
       if a < b]
fly = [f"BF{a:02d}{b:02d}{c:02d} Index"
       for a in headline
       for b in headline
       for c in headline
       if a < b < c]
sofr_ois = [f"USOSFR{a} Curncy" for a in headline+[1,4]]
swp_spd = [f"USSFCT{a:02d} Curncy" for a in headline]
tips = [f"USGGT{a:02d}Y Index" for a in [5,10,30]]
be = [f"USGGBE{a:02d} Index" for a in [5,10,30]]
zc = [f"USSWIT{a} Curncy" for a in headline+[1,4]]

# stonks
stonk = ["SPX Index", "NDX Index", "INDU Index"]
sector = [f"{a} US Equity" for a in ["XLC","XLY","XLP","XLE","XLF","XLV","XLI","XLB","XLRE","XLK","XLU"]]

# comods
prec_met = [f"{a} Curncy" for a in ["XAU","XAG","XPT","XPD","DXY"]]
enrgy = [f"{a}{b} Comdty" for a in ["CL","CO","NG","XB","HO"] for b in [1,2]]
ag = [f"{a}{' ' if len(a) == 1 else ''}{b} Comdty" for a in ["C","W","S","KC","SB","CT"] for b in [1,2]]
ind_met = [f"{a}{b} Comdty" for a in ["HG","LA","LX","LN","TIO"] for b in [1,2]]

# mortgage
mtg = ["MTGEFNCL Index"]  # Fannie 30yr current coupon yield — use for MBS basis vs 10yr

# swaption vol surface — normal vol (bps/yr), expiry × tenor × strike
# Ticker format: {strike_prefix}{expiry_code}{tenor} Curncy
# Special case: when expiry >= 10yr AND tenor >= 10yr, expiry uses a letter to stay within 8 chars
_SWPN_STRIKE_PREFIX = {
    -200: "USRHA", -100: "USREA",  -50: "USRCA", -25: "USRBA",
       0: "USSNA",
      25: "USROA",   50: "USRZA", 100: "USRQA", 200: "USRTA",
}
_SWPN_MONTH_CODE   = {"1Mo": "A", "3Mo": "C", "6Mo": "F", "9Mo": "I"}
_SWPN_LONGEXP_CODE = {10: "J", 15: "O", 20: "T", 30: "Z"}

def _swpn(strike, expiry, tenor):
    prefix = _SWPN_STRIKE_PREFIX[strike]
    if isinstance(expiry, str):
        exp = _SWPN_MONTH_CODE[expiry]
    elif expiry >= 10 and tenor >= 10:
        exp = _SWPN_LONGEXP_CODE[expiry]
    else:
        exp = str(expiry)
    return f"{prefix}{exp}{tenor} Curncy"

_SWPN_EXPIRIES = ["1Mo", "3Mo", "6Mo", "9Mo", 1, 2, 3, 5, 7, 10, 15, 20, 30]
_SWPN_TENORS   = [1, 2, 5, 10, 15, 20, 30]
_SWPN_STRIKES  = [-200, -100, -50, -25, 0, 25, 50, 100, 200]

swpn_atm = [_swpn(0, exp, ten) for exp in _SWPN_EXPIRIES for ten in _SWPN_TENORS]
swpn_vol  = [_swpn(k, exp, ten) for exp in _SWPN_EXPIRIES for ten in _SWPN_TENORS for k in _SWPN_STRIKES]

_SWPN_EXPIRY_STR = {
    "1Mo": "1Mo", "3Mo": "3Mo", "6Mo": "6Mo", "9Mo": "9Mo",
    1: "1Y", 2: "2Y", 3: "3Y", 5: "5Y", 7: "7Y",
    10: "10Y", 15: "15Y", 20: "20Y", 30: "30Y",
}

# ticker -> {expiry, tenor, strike} for structured inserts into md.swaption_vol
swpn_meta = {
    _swpn(k, exp, ten): {
        "expiry": _SWPN_EXPIRY_STR[exp],
        "tenor":  ten,
        "strike": k,
    }
    for exp in _SWPN_EXPIRIES
    for ten in _SWPN_TENORS
    for k in _SWPN_STRIKES
}

# print(sector)