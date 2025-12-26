
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
sofr_ois = [f"USOSFR{a} Curncy" for a in headline+[4]]
swp_spd = [f"USSFCT{a:02d} Curncy" for a in headline]
tips = [f"USGGT{a:02d}Y Index" for a in [5,10,30]]
be = [f"USGGBE{a:02d}Y Index" for a in [5,10,30]]
zc = [f"USSWIT{a} Curncy" for a in headline+[1,4]]

# stonks
stonk = ["SPX Index", "NDX Index", "INDU Index"]
sector = [f"{a} US Equity" for a in ["XLC","XLY","XLP","XLE","XLF","XLV","XLI","XLB","XLRE","XLK","XLU"]]

# comods
prec_met = [f"{a} Curncy" for a in ["XAU","XAG","XPT","XPD"]]
enrgy = [f"{a}{b} Comdty" for a in ["CL","CO","NG","XB","HO"] for b in [1,2]]
ag = [f"{a}{b} Comdty" for a in ["C","W","S","KC","SB","CT"] for b in [1,2]]
ind_met = [f"{a}{b} Comdty" for a in ["HG","LA","LX","LN","TIO"] for b in [1,2]]


print(sector)