# Building a Basis Model from the Red Book — Implementation Roadmap

Every formula, algorithm, and worked number in *The Treasury Bond Basis* (3rd ed.) that you'd
actually write code against, ordered as a dependency graph. Context and market history stripped out.

Companion to [`RedBook_TOC.md`](RedBook_TOC.md), which maps the whole book including the narrative
chapters. **Page refs are `p.N / pdf N`** — printed page and PDF page. Offset is a constant +30.

**The best thing about this book for implementation is that nearly every formula ships with a worked
numeric example.** Those are unit-test fixtures. They're collected in [§ Test fixtures](#test-fixtures)
at the bottom, and each layer below points at its own. Build against them.

---

## Status in this repo

`utils/basis.py` already covers Layers 0–2 and part of Layer 3:

| Built | Where |
|---|---|
| Act/act accrued interest, semiannual coupon periods | `accrued_interest`, `coupon_period` |
| Forward price with intervening coupon | `net_basis` (inline) |
| Invoice price, gross basis, net basis (32nds and bps) | `net_basis` |
| Implied repo rate | `net_basis` |
| CTD selection by highest implied repo | `net_basis` |
| Conversion factors | sourced from `sec.fut_contracts.dlv_basket`, not computed |
| Delivery date | assumed last business day of contract month |

Not built — this is what the roadmap below is for: **carry decomposed explicitly** (Layer 1),
**crossover / duration ranking** (Layer 2), **the whole delivery-option model** (Layer 4), the
**end-of-month sub-model** (Layer 5), the **DV01 / hedge-ratio stack** (Layer 7), and the **derived
products** (Layer 8).

Two documented approximations in `utils/basis.py` become explicit modelling choices later:
financing is a SOFR/FF proxy rather than term repo on CTD collateral (Layer 4 needs real term repo,
and specialness handling — p.84/pdf 114), and delivery is assumed last-business-day, which means
the timing option is not priced (Layer 6 — the book says that's fine, see below).

---

## Layer 0 — Units and conventions

Get this wrong and every downstream number is quietly off. p.5–6/pdf 35–36, p.13/pdf 43.

```
# Quote parsing
"91-14"   = 91 + 14/32
"124-21+" = 124 + 21.5/32          # "+" is a 64th, i.e. half a 32nd
"91.14"   = 91 + 14/32             # some feeds use a period, NOT a decimal

# Order of operations (p.6/pdf 36, "Important Point")
#   convert prices to decimal FIRST, compute basis in decimal, THEN x32.
basis_32 = (P_decimal - F_decimal * CF) * 32

# Dollar values
tick_value = {"US": 31.25,   # 1/32 on $100,000
              "TY": 15.625,  # 1/2 of 1/32 on $100,000
              "FV": 15.625,  # 1/2 of 1/32 on $100,000
              "TU": 15.625}  # 1/4 of 1/32 on $200,000
carry_32 = carry_dollars / 312.50    # per $1mm par: one 32nd = $312.50
```

**Day counts are deliberately mixed** (p.12/pdf 42): coupon income accrues actual/actual
(`Days / DCOUP`, where `DCOUP` is 181–186), financing accrues actual/360. That asymmetry is real,
not a bug — don't "fix" it.

The RP rate must be a **term** repo rate matched to settlement→delivery, not an overnight rate
(p.12/pdf 42).

---

## Layer 1 — Static per-bond primitives

Pure functions of (bond terms, prices, rates, dates). No optionality yet.

### 1.1 Conversion factor — Appendix A, p.233/pdf 263

The exact CBOT/CME formula under the 6% coupon assumption. Rounded to 4 decimals, constant through
a delivery cycle, unique per (bond, contract month).

```
Factor = a * [ Coupon/2 + c + d ] - b        # round to 4dp

a = 1 / 1.03**(v/6)
b = (Coupon/2) * (6 - v)/6
c = 1 / 1.03**(2n)      if z < 7
    1 / 1.03**(2n + 1)  otherwise
d = (Coupon/0.06) * (1 - c)

# n = whole years to maturity from the first day of the delivery month
# z = whole months beyond n, rounded DOWN to the nearest quarter for bond
#     and 10-year futures (z in {0,3,6,9}); to the nearest month for 5- and
#     2-year futures (z in 0..11)
# v = z                if z < 7
#     3                if z > 7 and contract is bond or 10-year
#     (z - 6)          if z > 7 and contract is 5- or 2-year
```

`Coupon` is the annual coupon **in decimals** (0.04875, not 4.875).

Note the truncation: odd days are discarded. A bond with 16y 2m 14d to maturity is treated as
16y 0m for a bond contract (p.6/pdf 36).

> **Fixtures:** `4-7/8% of 2/15/12`, Jun-2004 → **0.9328**. `3-5/8% of 5/15/13`, same → **0.8401**.
> Both worked in full at p.234/pdf 264. See [errata](#errata) — the second example has a typo.

Characteristics worth asserting in tests (p.6–7/pdf 36–37): coupon > 6% ⇒ CF > 1; coupon < 6% ⇒
CF < 1; CFs for high-coupon bonds drift *down* across successive contract months, low-coupon drift
*up*.

### 1.2 Basis — p.4/pdf 34

```
B = P - (F * CF)
```

`P` = spot clean price per $100 face. This is the *gross* basis.

> **Fixture:** 7-1/2% of 11/15/16, CF 1.1484, cash 120-20, futures 103-30 →
> `120.6250 - 103.9375*1.1484 = 1.2632` → **40.4/32nds** (p.7/pdf 37).

### 1.3 Invoice price — p.7/pdf 37

```
InvoicePrice  = F * CF + AccruedInterest(delivery_date)
InvoiceAmount = 1000 * InvoicePrice          # per contract, $100k par
```

> **Fixture:** `103.9375 * 1.1484 + 0.91712 = 120.2789` → **$120,278.90** (p.10–11/pdf 40–41).

### 1.4 Carry — p.11/pdf 41, Appendix B p.235/pdf 265

```
Carry = CouponIncome - FinancingCost

# No intervening coupon
CouponIncome  = (C/2) * (D / DCOUP)
FinancingCost = (P + AI) * (RP/100) * (D/360)

# One intervening coupon
CouponIncome  = (C/2) * ( D1/DCOUP1 + D2/DCOUP2 )
FinancingCost = (P + AI) * (RP/100) * (D1/360)  +  P * (RP/100) * (D2/360)

# C      annual coupon per $100 face
# D      actual days settlement -> delivery      D1 = settle -> coupon date, D2 = D - D1
# DCOUP  actual days in the coupon period        DCOUP1 = current, DCOUP2 = next
# RP     term repo rate in full percentage points
```

Note the second-leg financing in the intervening-coupon case uses `P`, not `P + AI` — accrued
resets at the coupon date.

> **Fixture:** 7-1/4% of 8/15/22 on 4/5/01, full price 120.8764, RP 4.54%, D=84, DCOUP=181 →
> coupon income **1.68232**, financing **1.280484**, carry **0.401836** = **12.9/32nds**
> (p.12–13/pdf 42–43).

**Caveat to encode as a comment, not code** (p.13/pdf 43): even at a locked term rate, repo
agreements require collateral changes when prices move past a threshold, so this is an
approximation. In a basis trade the net effect is small — futures variation margin largely funds
the collateral calls.

### 1.5 Basis net of carry

```
BNOC = B - Carry
```

The single most important derived quantity in the book. It is the market's price for the short's
delivery options — see Layer 4.

### 1.6 Implied repo rate — p.15/pdf 45

```
# No intervening coupon
IRR = (InvoicePrice/PurchasePrice - 1) * (360/n)

# One intervening coupon (closed form after solving the reinvestment recursion)
IRR = ( (InvoicePrice + C/2 - PurchasePrice) * 360 )
      / ( PurchasePrice * n  -  (C/2) * n2 )

# PurchasePrice = clean + accrued at settlement (full price)
# InvoicePrice  = F*CF + accrued at delivery
# n  = days settlement -> delivery
# n2 = days coupon date -> delivery
```

Both invoice and purchase price include accrued. The intervening-coupon form assumes the coupon is
reinvested at the IRR itself.

> **Fixture:** 7-1/4% of 8/15/22, purchase 120.8764, invoice 122.014346, n=84 →
> **4.03%** (p.16/pdf 46).

---

## Layer 2 — CTD selection

### 2.1 Three ranking measures — p.31–33/pdf 61–63

Implement all three; they disagree and the disagreement is informative.

| Rank by | Rule | When it's right | When it breaks |
|---|---|---|---|
| **Implied repo** | highest IRR | industry standard, reliable in most environments | ignores that different bonds finance at different rates |
| **IRR − own term repo** | least negative | **correct when issues trade special** | needs per-CUSIP term repo, which most feeds don't carry |
| **BNOC** | lowest | quick and readable | biased when competing bonds trade at different prices — a bond with the same BNOC but a higher price is genuinely cheaper |

The IRR−term measure is negative for every bond; the value *is* the market's cost of the strategic
delivery options, and the least-negative bond is cheapest. If all bonds finance at the same rate,
this collapses to the IRR ranking exactly (p.32/pdf 62).

> **Fixture:** Exhibit 2.3, p.33/pdf 63 — all 33 deliverables on 4/5/01 ranked three ways side by
> side. CTD is the 7-5/8% of 11/15/22: IRR 4.06, IRR−term −0.48, BNOC 4.53. The exhibit shows the
> rankings diverging further down the list (the 8% of 11/21 is 3rd by IRR despite a larger BNOC
> than two bonds below it).

### 2.2 Optimal delivery date — p.34/pdf 64

```
if carry > 0:   deliver last business day     # IRR to last day > IRR to first day
if carry < 0:   early delivery may pay, but the short forfeits all remaining
                switch + end-of-month option value
```

The book's empirical rule: the RP rate has had to be *significantly* higher than long bond yields
to justify early delivery (p.35/pdf 65). See Layer 4.4 for the actual valuation.

> **Fixture:** Exhibit 2.4, p.35/pdf 65 — March 2001 deliveries. Curve was inverted 0–2y, positive
> beyond. 2-year notes were delivered throughout the month (negative carry); 5s, 10s and bonds all
> on the last possible day (positive carry).

### 2.3 Crossover / duration heuristics — p.36–38/pdf 66–68

Not a substitute for §2.1, but you need this for scenario analysis in Layer 4 and it's the fastest
sanity check on a CTD switch.

```
# Rule 1 (duration).  At the same yield:
#   yields BELOW 6% -> LOWEST duration bond is cheapest
#   yields ABOVE 6% -> HIGHEST duration bond is cheapest
# Rule 2 (yield).  At the same duration: HIGHEST yield is cheapest.
```

Why 6%: conversion factors are the approximate prices at which deliverables would yield 6%, so at
a uniform 6% yield every converted price equals 100 and the short is indifferent. Away from 6%,
duration decides (p.36–37/pdf 66–67).

**Crossover yield** = the yield at which two bonds' converted prices (`P/CF`) cross. Compute by
solving `P_i(y)/CF_i = P_j(y)/CF_j` across the basket; the futures price at expiration traces the
lower envelope of the converted-price curves.

> **Fixture:** Exhibit 2.6, p.39/pdf 69 — three bonds, crossovers at **4.98%** and **6.13%**
> (as of 4/5/01). Below 4.98% the 8-7/8% of 8/17 is CTD; 4.98–6.13% the 7-5/8% of 11/22;
> above 6.13% the 5-1/2% of 8/28.

---

## Layer 3 — Forward price and rate risk

The bridge from cash to futures, and the point where the book departs from every simplified
treatment. **p.96–103/pdf 126–133.**

### 3.1 Forward price written out

Rather than "spot less carry", write financing and coupon explicitly so the two rate sensitivities
separate. For a bond with one intervening coupon at `d1` days, delivery at `n` days
(the book's example: n=84, d1=39, d2=45):

```
F = (S + AI) * [1 + R*(d1/360)] * [1 + R*(d2/360)]
    - (C/2)  * [1 + R*(d2/360) + (d2/DCOUP2)]
```

### 3.2 Spot DV01 of the forward — p.97/pdf 127

```
dF/dy_s = [ 1 + R*(n/360) + R**2 * (d1/360)*(d2/360) ] * dS/dy_s

# no intervening coupon:
dF/dy_s = [ 1 + R*(n/360) ] * dS/dy_s
```

The forward moves **more** than the spot — financing amplifies. Roughly +1% at 84 days and a 4.5%
repo rate.

### 3.3 Repo DV01 of the forward — p.98/pdf 128

```
dF/dR = (S + AI) * [ (n/360) + R*(d1/360)*(d2/360) ]  -  (C/2)*(d2/360)

# no intervening coupon:
dF/dR = (S + AI) * (n/360)

# per basis point per $100,000 par: divide by 10,000
```

**Sign is opposite to the spot DV01**: a rise in the repo rate *raises* the forward price (it cuts
carry). The book's convention (p.102/pdf 132) is to report spot DV01s positive and repo DV01s
negative, precisely to flag the difference.

> **Fixture:** 7-5/8% of 11/22, R=0.0454, spot DV01 145.45, n=84, d1=39, d2=45, full price 127.616:
> spot DV01 of the forward = **−146.99** per $100k (p.101/pdf 131). Simplified repo DV01 =
> `127,616*(84/360)/10,000` = **$2.98** (p.102/pdf 132). Cross-check at p.103/pdf 133: a 1bp rise
> in the money rate moves the PV of the offsetting term investment by $2.95, and
> `2.98/[1+0.0455*(84/360)] = 2.95`. ✓ (The printed with-coupon intermediates on p.101 don't
> reconcile — see [errata](#errata); implement from the formula.)

### 3.4 Why this matters

**Spot yields and term repo rates are independent at hedging horizons.** Exhibits 5.5 and 5.6
(p.99–100/pdf 129–130), Jan 1988 – Oct 2003: in *levels*, 1-month repo vs the 5y/10y/30y yield has
R² of 0.78 / 0.64 / 0.47. In *weekly changes*, R² collapses to **0.025 / 0.011 / 0.002** — no usable
relationship at all.

So a complete hedge of a spot bond is **two instruments**: futures for spot-yield risk, plus a term
money-market position for repo risk. That second leg is "stub risk" (p.103/pdf 133). In the book's
example the repo DV01 is ~$3/bp against a spot DV01 of ~$145/bp — often small enough to ignore, but
size it before you decide to.

Synthetic bond construction falls straight out (p.102/pdf 132): forward position + term money
market instrument maturing on the delivery date ≡ the spot bond. With **futures** rather than
forwards you need only `spot_DV01 / forward_DV01` contracts — 0.9895 in the example, because futures
gains are realised today rather than at delivery.

---

## Layer 4 — The delivery option model ⭐

**This is the core. Chapter 4 in full, p.75–86/pdf 105–116.** Everything else is plumbing around it.

### 4.1 The identities — p.75–76/pdf 105–106, Exhibit 4.1

```
ActualBasis          = Carry + MarketDeliveryOptionValue
MarketDeliveryOptVal = BNOC                              # by definition
TheoreticalBasis     = Carry + TheoreticalDeliveryOptionValue

OptionAdjustedBasis  = ActualBasis - TheoreticalBasis
                     = BNOC - TheoreticalDeliveryOptionValue

MarketFuturesPrice      = (CashPrice - ActualBasis) / CF
TheoreticalFuturesPrice = (CashPrice - TheoreticalBasis) / CF

MarketPrice - TheoreticalPrice = -OptionAdjustedBasis / CF
```

**Decision rules:** `OAB > 0` ⇒ basis rich ⇒ **futures cheap**. `OAB < 0` ⇒ basis cheap ⇒ futures
rich. The whole question of futures fair value reduces to whether the delivery options are fairly
valued.

### 4.2 The algorithm — p.78–80/pdf 108–110

The theoretical delivery option value **is** the expected BNOC at expiration. Two steps.

**Step 1 — build the joint distribution over (yield level, curve slope).**

Rows are curve scenarios, columns are level scenarios, each with a probability. The book's
illustrative 3×3 (Exhibit 4.3) uses levels {−100, 0, +100} at probs {0.16, 0.68, 0.16} and curve
{steeper +20, beta, flatter −20} at the same probs. Use a finer grid in production; 3×3 just shows
the shape.

Level distribution comes from a reference bond of your choosing (e.g. the OTR). Slope distribution
is a separate axis — see §4.3.

**Step 2 — fill each cell.** For every scenario cell:

```
for cell in grid:
    1. reprice every deliverable at futures expiration under (level, slope)
    2. identify the CTD in this scenario
    3. value the end-of-month option for this scenario      # Layer 5
    4. Futures = (CTD_Price - CTD_Carry - EndOfMonthOption) / CTD_Factor
    5. for each deliverable: BNOC_i = f(cell) using that futures price
```

**Step 3 — for each issue, take the probability-weighted average of its BNOC across all cells.**
That expected BNOC is the theoretical delivery option value for that issue, and the expected gross
payoff to buying its basis.

Repeat for every issue so you get theoretical BNOC for the whole deliverable set. If the work is
right, relative richness across the basket is consistent with the futures contract's own
richness/cheapness.

> **Fixture — reproduce this exactly (p.80/pdf 110):**
> ```
> 0.16 * [0.16*20 + 0.68*2 + 0.16*20]     (steeper)  = 0.16 * 7.76
> 0.68 * [0.16*25 + 0.68*1 + 0.16*15]     (beta)     = 0.68 * 7.08
> 0.16 * [0.16*30 + 0.68*3 + 0.16*10]     (flatter)  = 0.16 * 8.44
>                                                     = 7.41 (32nds)
> ```
> The BNOC values are for a middling-duration issue — cheapest if yields are unchanged, expensive
> if they move either way, hence the low centre column. Note the curve effect flips sign with the
> direction of the level move (p.79–80/pdf 109–110): if yields fall, the new CTD is lower-duration,
> so *steepening* makes it less cheap and cuts our issue's BNOC; if yields rise, reversed.

### 4.3 Volatility inputs — p.82–84/pdf 112–114

**Yield level vol.** Start from implied price vol on listed options on futures, convert:

```
sigma_y = (Yield * ModifiedDuration) * sigma_p
# yield and mod duration are the CTD's; both sigmas are relative
```

Acknowledged imperfection: futures are driven by a basket, and the options expire roughly a month
before the futures. It's a starting point, not the answer.

**Systematic curve reshaping — yield betas.** A bond's yield beta is the expected change in its
yield per 1bp change in the reference issue's yield. This captures the curve flattening as yields
rise / steepening as they fall. **The book warns explicitly** (p.83/pdf 113) that estimated betas
vary greatly over time and that your beta assumption has a big influence on delivery option value.
Treat it as a model parameter to stress, not a constant.

**Unexpected spread vol.** After the beta captures the systematic part, the residual spread moves
are a separate distribution. Exhibit 4.5 (p.83/pdf 113) gives standard deviations of Treasury curve
slope changes, in bps, by horizon:

| Segment | 2wk | 1m | 2m | 3m | 6m |
|---|---|---|---|---|---|
| 30y − 15y | 3.7 | 4.9 | 6.0 | 7.3 | 9.3 |
| 10y − 7y | 3.5 | 4.6 | 5.9 | 7.0 | 10.3 |
| 5y − 4y | 2.5 | 3.3 | 4.5 | 5.1 | 4.2 |

> ⚠️ **Do not scale spread vol by √t.** Because yield changes across the curve are positively
> correlated, the usual square-root-of-time rule does not apply — the 15s30s slope sd does not
> double for a quadrupling of horizon (p.84/pdf 114). The table above shows it: 3.7 → 7.3 over
> 2wk → 3m is a 6× horizon for a 2× sd, and the 5s4s row is *non-monotonic*.

### 4.4 Four consistency checks — p.84/pdf 114

These are your model-validation gates. Encode them as assertions, not comments.

1. **Forward price consistency.** Each issue's probability-weighted expected price at futures
   expiration must equal its actual market forward price (cash net of carry to expiry). If your
   distribution doesn't reprice the forwards, nothing downstream is trustworthy.
2. **Option-market consistency.** From your scenario distribution, price a hypothetical at-the-money
   call on bond futures, back out its implied price vol, and compare to listed options on futures.
   If your model misprices the ATM call, adjust the assumed yield vol until it doesn't. The book
   notes this check works better for the bond and 10-year contracts than for 5s and 2s.
3. **Term repo specials.** When an issue on special is CTD (uncommon but real), adjust its forward
   price for the specialness.
4. **Anticipated new issues.** Treasury's auction cycle is predictable, so predicted new issues must
   be injected into the deliverable basket — **critical for 2-year and 5-year contracts, where
   anticipated new issues can be the bulk of the deliverable set.** The hard parts are placing an
   unissued bond on the curve and modelling behaviour when you don't yet know its coupon.

Related, from p.64/pdf 94: the anticipated-new-issue option is usually worth little, because a new
issue is on-the-run and trades at a *lower* yield, making it expensive to deliver. The exception —
5s and 10s when yields are high but falling, so the new issue has both the lowest coupon and longest
maturity and therefore the highest duration in the set.

### 4.5 Early delivery — p.80–82/pdf 110–112

Only relevant when carry is negative.

```
# If early delivery is the short's best choice, arbitrage forces:
FuturesPrice = CTD_Price / CTD_Factor          # plus a day's carry, no remaining options

# Decision: compare negative carry against remaining switch + EOM option value.
#   low vol  -> negative carry dominates -> deliver first delivery day
#   high vol -> option value dominates   -> wait

# Valuing the option (rather than just deciding) requires the joint distribution
# of negative carry, which needs distributions of BOTH term repo rates and
# deliverable prices:
E[Futures] = P(early) * E[CTD Price / CTD Factor on early delivery date]
           + P(late)  * E[Futures Price on last trading day]
```

### 4.6 Output and validation

> **Fixture — Exhibit 4.6, p.85/pdf 115.** All 33 deliverables into Jun-2001 bond futures on
> 4/4/01, with columns: Closing Price, CF, Basis, Carry, BNOC, Theoretical Option Value, OAB. This
> is a full regression target for the entire model. Selected rows (32nds):
>
> | Issue | Price | CF | Basis | Carry | BNOC | Theo Opt | OAB |
> |---|---|---|---|---|---|---|---|
> | 5-3/8 2/15/31 | 98-08 | 0.9140 | 104.0 | 7.0 | 97.0 | 101.5 | −4.5 |
> | **7-5/8 11/15/22 (CTD)** | **124-20** | **1.1936** | **18.1** | **13.6** | **4.5** | **8.0** | **−3.5** |
> | 6 2/15/26 | 104-18 | 1.0000 | 20.0 | 8.8 | 11.2 | 14.7 | −3.5 |
> | 7-1/2 11/15/16 | 120-20 | 1.1484 | 40.4 | 14.0 | 26.4 | 29.4 | −3.0 |
>
> **Two structural facts to assert as tests** (p.86/pdf 116):
> - OAB is negative for *every* issue in the set — the basis was uniformly cheap, futures rich.
> - **Only the CTD's BNOC is pure option value.** For every other issue, BNOC = pure delivery
>   option value + the amount by which it is expensive to deliver. The 5-3/8%'s BNOC of 97/32nds
>   isn't more optionality, it's a bond you'd lose ~3 price points delivering. In option language:
>   the CTD's delivery options are at-the-money, everyone else's are in-the-money.
>
> **Mispricing round-trip:** CTD OAB = −3.5 ticks, CF = 1.1936 →
> `-(-3.5)/1.1936 = 2.93 ticks rich`. Fair futures 103-27 vs market 103-30. ✓

---

## Layer 5 — End-of-month option

Sub-model called from Layer 4 step 3. **p.64–70/pdf 94–100.**

Between the last trading day and the last delivery day (seven business days for bonds), **the
futures settlement price is frozen while cash prices keep moving.** The short can still switch
bonds. This is what stops the CTD's BNOC from going to zero on the last trading day.

### 5.1 Payoff

Once the final settlement price `F*` is fixed, the net cost of delivering any eligible bond is:

```
NetCost_i = CashPrice_i - (CF_i * F*) - Carry_i
```

The short delivers whichever bond minimises `NetCost`. Since `F*` is frozen, the only thing that
varies is each bond's cash price — and they have **different BPVs**, so a parallel yield move
reshuffles the ranking.

```
# yield move needed for bond j to overtake the current cheapest bond i:
delta_y = (NetCost_j - NetCost_i) / (BPV_j - BPV_i)
```

> **Fixture — Exhibit 3.9, p.65/pdf 95.** Three bonds at the close of trading:
>
> | Bond | BNOC (32nds) | BPV (32nds) | Δy to become CTD |
> |---|---|---|---|
> | 6-1/4% of 8/23 | 1.0 | 4.25 | −4.0 bp |
> | **7-1/4% of 8/22** | **0.0** | **4.50** | **0.0** (currently CTD) |
> | 8-3/4% of 8/20 | 1.5 | 4.70 | +7.5 bp |
>
> `1.5/(4.70−4.50) = 7.5bp` up; `1.0/(4.50−4.25) = 4.0bp` down. Payoff at +10bp is 0.5/32nds; at
> −10bp it's 1.5/32nds (p.68/pdf 98).

**Shape: a long strangle** (Exhibit 3.11, p.69/pdf 99), struck at the two switch points. **Cost =
the CTD's BNOC at the close of the last trading day** — zero in this stylised example, positive in
practice, which is exactly why the CTD's BNOC doesn't converge to zero.

### 5.2 The counterintuitive part — p.64–65/pdf 94–95

Before expiry, a rise in yields favours **high-duration** issues. After expiry, a rise in yields
favours **high-BPV** issues — and high-BPV issues are high-*coupon*, hence typically **low**
duration. So a yield move that makes a bond cheap to deliver before the last trading day often
makes it expensive after. Model the two regimes separately; don't reuse the switch-option logic.

### 5.3 Implementation details that bite

- **Hedge ratio changes to 1:1 after the last trading day** (p.65/pdf 95). Each open short calls
  for $100k par regardless of price, so the conversion-factor weighting that governs a normal basis
  position no longer applies.
- **Repeatable.** The option can be exercised as often as yields move enough to flip the CTD
  (p.69/pdf 99).
- **Last possible switch is the second-to-last business day**, not the next-to-last. Delivery is due
  9am Chicago; a cash purchase doesn't reach the Fed wire until 2pm the following day (p.69/pdf 99).
- **You cannot repo the bonds out the night before delivery** — they must be in the box. Financing
  that last night is materially above the repo rate, and that is a real cost of a long basis
  position (p.70/pdf 100).
- A "plan to fail" in the cash market is the standard protection against a fail on the new bonds —
  far cheaper than failing into the futures contract (p.70/pdf 100).

---

## Layer 6 — Timing options (deprioritise)

**The book explicitly skips these** (p.77/pdf 107): "the incremental contribution of the timing
options tends to be small… we will skirt the complications." Match that unless you have a reason.

Three options, in play from first notice day to month end (p.70–73/pdf 100–103):

1. **Carry option** — deliver early when carry is negative. Already covered by Layer 4.5.
2. **Wild card** — a large cash move after futures close. Breakeven:
   ```
   (CF - 1) * (CF * F - P) = B
   # CF = conversion factor, F = closing futures price,
   # P  = aftermarket cash price, B = expected opening basis
   # LHS = profit from covering the tail; RHS = basis surrendered by delivering early
   ```
   The mechanism is "covering the tail": if you're long $100mm of a CF-1.3093 bond you're short
   1,309 contracts but can only deliver on 1,000, so you must buy $30.9mm more cash. If CF > 1 you
   need a price *drop*; if CF < 1, a rise.
   > **Fixture (p.72–73/pdf 102–103):** 8-3/4% of 8/20, CF 1.3093, F = 104-10, basis 6/32nds.
   > Giving up the basis costs $187,500. Required price = 0.6068 points below invoice price
   > (`$187,500/$30,900,000 × 100`), i.e. the bond must fall to 135-31, a total drop of 25.4/32nds
   > after the close — of which 19.4/32nds is below the invoice price (`6/32 / 0.3093`).
3. **Limit moves** — "adds next to nothing." Only two notice days are governed by limits, and the
   expiring bond contract has no limits in its delivery month.

---

## Layer 7 — DV01 and hedge ratios

**Chapter 5, p.87–128/pdf 117–158.** Four escalating layers; each fixes a defect in the previous.

### 7.1 Rules of thumb — p.88–91/pdf 118–121

```
# Rule 1
FuturesDV01 = CTD_DV01 / CTD_Factor
HedgeRatio  = PortfolioDV01 / FuturesDV01

# Rule 2
FuturesDuration = CTD_Duration

# with modified duration:
n_futures = (PortDuration * PortMarketValue)
            / (FuturesDuration * (FuturesPrice/100) * ContractPar)
```

Both rest on the assumption that at expiration `Futures = CTD_Price / CTD_Factor`, hence
`ΔF = ΔP_ctd / CF`. Rule 2 follows from Rule 1 by dividing through by the futures price.

**Rule 1 is what Bloomberg returns** when you ask it for a hedge ratio (p.94/pdf 124, Exhibit 5.3).

> **Fixture (p.91/pdf 121):** hedge $10mm of the 5% of 2/15/11 (OTR 10y), portfolio DV01 $7,762.40.
> CTD is the 5-1/2% of 2/15/08, DV01 $59.14/$100k, CF 0.9734 → futures DV01 = `59.14/0.9734` =
> **$60.76**; HR = `7,762.40/60.76` = **127.8** → short **128** contracts.

### 7.2 Add repo DV01 — Layer 3.3

The hedge is two instruments. Size the stub. Fed funds futures are the book's preferred futures
solution for stub risk (p.104/pdf 134); term repo is more effective but has wide bid/ask and is
costly to unwind.

### 7.3 Option-adjusted DV01 — p.104–107/pdf 134–137

**Why the rules of thumb fail** (Exhibit 5.7, p.105/pdf 135): below the crossover yield the
rule-of-thumb DV01 is too low (overhedged); above it, too high (underhedged); and **at** the
crossover it takes a discrete jump. A 1bp move can produce an enormous change in the hedge ratio.
The theoretical futures price curve, by contrast, bends smoothly because it prices the changing
option value.

Consequence worth naming: around a crossover the futures price exhibits **negative convexity** — the
DV01 *increases* as yields rise. Away from crossovers you see the positive convexity of an
entrenched CTD.

**Computation is numerical, and it reuses Layer 4:**

```
# 1. produce a schedule of THEORETICAL futures prices over a range of yield shifts
#    (this is the Layer 4 model, run at each shift)
# 2. central difference:
DV01_oa(y) = ( [F(y-10bp) - F(y)]/10  +  [F(y) - F(y+10bp)]/10 ) / 2
```

> **Fixture — Exhibit 5.8, p.107/pdf 137.** Jun-2001 bond futures: `F(−10bp)=105.531`,
> `F(0)=104.250`, `F(+10bp)=103.000`. Down-move DV01 = $128.10, up-move = $125.00, average =
> **$126.55**. The full schedule runs −60bp to +60bp and shows the DV01 *falling* over the first
> 30bp of a rally then rising again — the signature of a shift in emphasis from a high-duration to
> a low-duration CTD, after which the low-duration bond's own positive convexity takes over.

```
# Option-adjusted duration
Duration_oa = DV01_oa * 100bps / PortfolioEquivalentValue * 100   # in %
```
> `66.13 * 100 / 106,250 = 6.22%` for the Jun-01 10-year (p.110/pdf 140).

### 7.4 Yield betas — p.108/pdf 138, and the appendix p.118–128/pdf 148–158

```
YieldBetaHedgeRatio = (DV01_i * beta_i) / (DV01_futures * beta_ctd)

# and the futures DV01 wrt a different reference bond:
DV01_futures(wrt OTR) = DV01_futures(wrt CTD) * (beta_ctd / beta_otr)
```

> **Fixture — Exhibit 5.9, p.109/pdf 139.** DV01/beta: 7.5% of 11/16 → 117.78 / 1.1365;
> 7.625% of 11/22 (CTD) → 145.45 / 1.0857; 5.375% of 2/31 (OTR) → 144.03 / 1.0000. Futures DV01
> wrt CTD = 122.42, wrt OTR = 132.92, and `122.42 * 1.0857 = 132.91` ≅ 132.92 ✓. Hedge ratio for the 7.5%
> is **1.007** computed either way — the denominators are identical by construction.

**The honest caveats** (appendix, p.124–128/pdf 154–158) — these belong in the model docstring:

- **Betas are unstable.** Exhibit A5.6 shows yield betas by year; they move a lot.
- **Below ρ = 1.0 there is no single correct hedge ratio.** DV01-neutral, minimum-expected-change,
  and minimum-variance hedges all differ, and they answer different questions (p.125–128/pdf
  155–158). Exhibit A5.7 gives correlations and sds of weekly yield changes for OTR issues,
  1990–1997. Pick your objective explicitly rather than defaulting.

### 7.5 Validation target

> **Exhibit 5.10, p.110/pdf 140** — for 2y/5y/10y/bond × Jun-01/Sep-01: market price, theoretical
> price, rule-of-thumb DV01, option-adjusted DV01 wrt CTD yield and wrt OTR yield, option-adjusted
> duration wrt each, and repo DV01. Jun-01 bond: RoT 121.86, OA(CTD) 122.42, OA(OTR) 132.92,
> repo DV01 −2.46.
>
> The instructive row is the **10-year**: RoT DV01 60.76 vs option-adjusted 66.13, because the CTD
> was the lowest-duration note in the basket and higher-duration competitors were pulling on the
> futures price. **The rule of thumb gives hedge ratios ~10% too large** (p.110/pdf 140).

---

## Layer 8 — Derived products

### 8.1 Futures fair value — p.142/pdf 172

```
FuturesFairValue = (CTD_Price - CTD_BasisFairValue) / CTD_Factor
```
where `CTD_BasisFairValue = Carry + TheoreticalDeliveryOptionValue` from Layer 4.

> **Fixture (p.143/pdf 173):** 5-3/4% of 11/05, price 105-2.5/32, theoretical June basis 13.8/32 →
> `(105-2.5/32 - 13.8/32)/0.9904 = 105-21.2/32`. Market was 105-22 → 0.8/32 rich.

Sign relationship to check: futures mispricing ≈ −OAB/CF, same magnitude, opposite sign. A futures
contract being underpriced is another way of saying its basis is rich.

### 8.2 Calendar spread fair value — p.143/pdf 173

```
SpreadFairValue = FuturesFairValue(front) - FuturesFairValue(back)
```

> **Fixture — Exhibit 6.9, p.145/pdf 175.** 10-year Jun/Sep on 4/4/01: actual 106-08 − 105-25+ =
> **14.5/32**; theoretical 106-059 − 105-211 = **16.9/32** → spread **2.4/32 cheap**. Bond spread
> 1.7/32 cheap, 5-year 1.1/32 cheap.

Two caveats before trading the spread outright (p.145/pdf 175): calendar spreads carry substantial
yield-curve risk, and **nothing forces a calendar spread to converge to fair value at expiration** —
unlike a basis. The book's preferred use is to establish cheaper basis positions in the deferred
month rather than to trade the spread naked.

Seasonal pattern (Exhibit 6.10, p.146/pdf 176): the average bond calendar spread **falls going into
first notice day**, driven by longs rolling to avoid early delivery. Buying the spread just before
first notice day has historically favoured institutions rolling a short position forward.

### 8.3 Squeeze-adjusted CTD net basis — p.141/pdf 171

When the CTD's float can't cover expected deliveries, its net basis can go **negative** — which
looks like an arbitrage violation but isn't. Buying a negative net basis locks in a riskless profit
only if you can actually deliver the CTD; if you can't, you deliver the second CTD and eat its net
basis.

```
FairValue(CTD_BNOC) = P(delivery failure) * (CTD_BNOC - 2nd_CTD_BNOC)

# invert to read the market's own squeeze probability:
P_implied(failure) = CTD_BNOC / (CTD_BNOC - 2nd_CTD_BNOC)
```

> **Fixture (p.141/pdf 171):** CTD supply covers 90% of deliveries, BNOC gap = 1 point = 32/32nds →
> fair CTD net basis = `0.10 * 32` = **−3.2/32nds**.

Context: June 2005 CTD net bases in both 5-year and 10-year futures traded negative for most of the
delivery cycle; the June 10-year CTD net basis averaged **−2/32nds** over the first two weeks of May
2005.

Positioning for a shortage (p.142/pdf 172): sell the basis of an expensive-to-deliver bond, buy the
CTD or asset-swap it, or buy the calendar spread. ⚠️ **Exchange rules prohibit trades designed to
create shortages or squeezes**, with penalties at the exchange's discretion — the book flags this
explicitly.

---

## Build order

```
0.  units + quote parsing                    ← trivial, blocks everything
1.  conversion factor, AI, invoice, carry,   ← ✅ mostly in utils/basis.py
    basis, BNOC, IRR
2.  CTD selection (3 measures) + crossovers  ← ✅ partial (IRR only)
3.  forward price, spot DV01, repo DV01      ← ✅ forward only; DV01s missing
─────────────────────────────────────────────
4.  delivery option model  ⭐                ← the real work; needs 1-3
5.  end-of-month sub-model                   ← called by 4; can stub to 0 initially
7.  option-adjusted DV01                     ← numerical, needs 4
7.  yield betas                              ← feeds 4's slope distribution AND 7
─────────────────────────────────────────────
8.  futures FV, calendar spread, squeeze     ← all cheap once 4 exists
6.  timing options                           ← book says skip; do last or never
```

Layer 4 is the gate. Until it exists, `cash_futures.py` is trading BNOC's OU z-score — i.e. fading
the *market* price of the delivery options without a *theoretical* price to compare it to. Layer 4
turns that into an OAB signal, which is the book's actual thesis.

Note the circularity to plan for: Layer 4 needs yield betas (§4.3) and Layer 7.4 also needs them.
Build the beta estimation once, as a shared input.

---

## Test fixtures

Every worked number in the book, in one table. Trade date 4/5/01, settle 4/6/01, delivery 6/29/01
unless noted.

| # | What | Inputs | Expected | Page |
|---|---|---|---|---|
| 1 | Conversion factor | 4-7/8% of 2/15/12, Jun-04 | 0.9328 | 234 / 264 |
| 2 | Conversion factor | 3-5/8% of 5/15/13, Jun-04 | 0.8401 | 234 / 264 |
| 3 | Basis | 7-1/2% 11/16, 120-20, F 103-30, CF 1.1484 | 1.2632 = 40.4/32 | 7 / 37 |
| 4 | Invoice price | F 103.9375, CF 1.1484, AI 0.91712 | 120.2789 | 10 / 40 |
| 5 | Invoice amount | ×1000 | $120,278.90 | 11 / 41 |
| 6 | Coupon income | C 7.25, D 84, DCOUP 181 | 1.68232 | 12 / 42 |
| 7 | Financing cost | full 120.8764, RP 4.54, D 84 | 1.280484 | 12 / 42 |
| 8 | Carry | difference | 0.401836 = 12.9/32 | 13 / 43 |
| 9 | Implied repo | purchase 120.8764, invoice 122.014346, n 84 | 4.03% | 16 / 46 |
| 10 | CTD ranking, 33 bonds | Exhibit 2.3 | CTD = 7-5/8% 11/22 | 33 / 63 |
| 11 | Crossover yields | Exhibit 2.6, 3 bonds | 4.98%, 6.13% | 39 / 69 |
| 12 | EOM switch trigger | BNOC gap 1.5, BPV gap 0.20 | +7.5 bp | 66 / 96 |
| 13 | EOM switch trigger | BNOC gap 1.0, BPV gap 0.25 | −4.0 bp | 66 / 96 |
| 14 | Wild card breakeven | CF 1.3093, F 104-10, basis 6/32 | fall 25.4/32 | 73 / 103 |
| 15 | **Expected BNOC (Layer 4)** | 3×3 grid, Exhibit 4.3 | **7.41/32** | 80 / 110 |
| 16 | Full OAB report, 33 bonds | Exhibit 4.6 | CTD OAB −3.5/32 | 85 / 115 |
| 17 | Futures mispricing | OAB −3.5, CF 1.1936 | 2.93 ticks rich | 86 / 116 |
| 18 | Rule-of-thumb futures DV01 | CTD DV01 59.14, CF 0.9734 | $60.76 | 91 / 121 |
| 19 | Rule-of-thumb hedge ratio | port DV01 7,762.40 | 127.8 → 128 | 91 / 121 |
| 20 | Forward spot DV01 | spot DV01 145.45, R .0454, n 84 | −146.99 | 101 / 131 |
| 21 | Repo DV01 (simplified) | full 127,616, n 84 | $2.98 | 102 / 132 |
| 22 | Stub offset | PV of 1bp on term investment | $2.95 | 103 / 133 |
| 23 | Futures needed vs forward | 145.45 / 146.99 | 0.9895 | 103 / 133 |
| 24 | Option-adjusted DV01 | F(−10)=105.531, F(0)=104.250, F(+10)=103.000 | $126.55 | 107 / 137 |
| 25 | Futures DV01 wrt OTR | 122.42 × 1.0857 | 132.92 | 108 / 138 |
| 26 | Yield-beta hedge ratio | Exhibit 5.9 | 1.007 (both routes) | 109 / 139 |
| 27 | Option-adjusted duration | 66.13/bp, PEV 106,250 | 6.22% | 110 / 140 |
| 28 | Full risk-measure table | Exhibit 5.10 | 4 contracts × 2 months | 110 / 140 |
| 29 | Squeeze fair value | P(fail) 0.10, BNOC gap 32/32 | −3.2/32 | 141 / 171 |
| 30 | Futures fair value | price 105-2.5/32, basis FV 13.8/32, CF 0.9904 | 105-21.2/32 | 143 / 173 |
| 31 | Calendar spread FV | Exhibit 6.9, 10y Jun/Sep | 16.9/32 vs 14.5 actual | 145 / 175 |

Fixtures 15, 16 and 24 are the ones that actually test the model. The rest test the plumbing.

---

## Errata

Found while extracting formulas from this scan. Verify against another copy before trusting my
corrections, but the arithmetic below checks out.

1. **Appendix A, p.234/pdf 264, second example.** The final line reads
   `Factor = 0.985329 × [(0.03625/2) + 0.605016 + 0.238636] − 0.840072 = 0.840072`. The subtracted
   term should be **b = 0.009063**, not 0.840072 (the result has been duplicated into the operand).
   Working it through: `0.985329 × 0.861777 = 0.849134`, `− 0.009062 = 0.840072` → **0.8401** ✓.
   The stated answer is right; the printed line isn't.

2. **Exhibit 5.8, p.107/pdf 137.** The yield-change column lists `−60, −50, −50, −30, …` — two rows
   labelled −50. The second should be **−40** (the theoretical prices, 110.563 and 109.281, are
   distinct and evenly spaced).

3. **p.101/pdf 131, repo DV01 with intervening coupon.** The printed intermediates don't reconcile:
   `R*(39/360)*(45/360)` with R = 0.0454 evaluates to 0.000615, not the printed 0.000143, and the
   final `= 28.8398/10 = 2.88398` doesn't follow from the numerator shown. This may be OCR damage
   in this particular scan. **Implement from the formula**, and validate against the simplified
   no-coupon case on p.102 ($2.98) and the internally-consistent stub cross-check on p.103
   ($2.98/[1+0.0455×84/360] = $2.95).

---

## What has changed since 2005

The book predates several things that affect implementation. Re-source anything contract-specific;
the pricing theory is unaffected.

- **Exhibit 1.1's contract specs are 2005-vintage CBOT.** CBOT merged into CME Group in 2007. Pull
  current specs — grades, maturity windows, tick sizes, last trading days — from the CME rulebook,
  not from the exhibit. The 6% conversion-factor coupon assumption (Appendix A) has not changed.
- **The deliverable complex is wider.** This repo's `DELIVERABLE_ROOTS` already lists six roots
  (`TU, FV, TY, UXY, US, WN`) against the book's four — Ultra Bond and Ultra 10-Year postdate it.
  Chapter 8's "rise of notes, fall of bonds" narrative stops well before that.
- **Stub hedging.** The book recommends Fed funds futures, 1-month LIBOR futures, or 3-month
  Eurodollar futures (p.104/pdf 134). LIBOR and Eurodollar futures are gone; SOFR futures are the
  replacement. Fed funds futures still exist.
- **Financing.** `utils/basis.py` proxies term repo with front SOFR/FF futures. Layer 4's specials
  handling (§4.4 check 3) needs genuine per-CUSIP term repo, which the current data doesn't carry —
  that's a data-sourcing dependency, not a modelling one.
- **Chapter 9's appendices** (German, JGB, gilt market conventions) are 2001/2005 JPMorgan outlines.
  Treat as historical.
