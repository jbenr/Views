# The Treasury Bond Basis — Annotated Table of Contents

Burghardt, Belton, Lane, Papa — **3rd Edition (2005)**, McGraw-Hill. `RedBook.pdf`, 312 pages.

## How to use this file

- **`p.N` = printed book page. `pdf N` = page to type into your PDF reader.**
- **Offset is constant: `pdf = printed + 30`.** (Front matter runs pdf 1–30.)
- The PDF is a ClearScan OCR of a scan. Text extraction works, but the OCR mangles some
  characters — searching is easier if you know the quirks:
  - `%` renders as `°/o` (e.g. `6°/o`)
  - Running headers show `CHAPTER S` for Chapter 5 and `CHAPTER ?` for Chapter 7
  - Headings have injected spaces: `C O NV E RS I O N FACTO RS`. **Grep for lowercase body
    prose, not headings.**
- Extract text with `pdftotext -layout -f <pdf_first> -l <pdf_last> RedBook.pdf -`.

---

## Chapter map — start here

| Ch | Title | p. | pdf | The question it answers |
|----|-------|----|-----|-------------------------|
| 1 | Basic Concepts | 1 | **31** | What *is* the basis, mechanically? Definitions, CF, invoice price, carry, IRR. |
| 2 | What Drives the Basis? | 27 | **57** | Why is the basis not just carry? CTD search, crossovers, basis-as-option. |
| 3 | The Short's Strategic Delivery Options | 51 | **81** | Anatomy of each delivery option: switch, end-of-month, timing/wild card. |
| 4 | The Option-Adjusted Basis | 75 | **105** | How to *price* those options and decide rich/cheap. |
| 5 | Approaches to Hedging | 87 | **117** | Hedge ratios: rules of thumb → repo DV01 → option-adjusted DV01 → yield betas. |
| 6 | Trading the Basis | 129 | **159** | The actual trades, plus squeezes, specials, calendar spreads. |
| 7 | Volatility Arbitrage in the Treasury Bond Basis | 157 | **187** | Basis vol vs. listed-option implied vol. (Reprint of a 1993 JPM article.) |
| 8 | Nine Eras of the Bond Basis | 173 | **203** | Market history 1977–2005 and why the contract behaved differently in each regime. |
| 9 | Non-Dollar Government Bond Futures | 197 | **227** | Bund/Bobl/Schatz, JGB, Gilt — contract specs and European basis themes. |
| 10 | Applications for Portfolio Managers | 217 | **247** | Duration targeting with futures, synthetic assets, yield enhancement. |

**Reference sections:** Appendix A (conversion factors) p.233/pdf 263 · Appendix B (carry) p.235/pdf 265 ·
Appendix G (glossary) p.263/pdf 293 · Index p.271/pdf 301.

**Front matter:** List of Exhibits p.xiii/pdf 13 · Preface to 3rd Ed. p.xix/pdf 19 ·
Preface to 2nd Ed. p.xxi/pdf 21 · Preface to 1st Ed. p.xxv/pdf 25.

---

## Chapter 1 — Basic Concepts · p.1 / **pdf 31**

The vocabulary chapter. Everything downstream assumes these definitions cold.

| Section | p. | pdf |
|---|---|---|
| Treasury Bond and Note Futures Contract Specifications | 2 | 32 |
| Definition of the Bond Basis (`B = P − F × CF`) · Units | 4 | 34 |
| Conversion Factors · Characteristics of Conversion Factors | 6 | 36 |
| Futures Invoice Price | 7 | 37 |
| Carry: Profit or Loss of Holding Bonds | 11 | 41 |
| Theoretical Bond Basis | 13 | 43 |
| Implied Repo Rate | 15 | 45 |
| Buying and Selling the Basis (incl. EFP mechanics) | 17 | 47 |
| Sources of Profit in a Basis Trade | 20 | 50 |
| Alternative Summary P/L | 22 | 52 |
| RP versus Reverse RP Rates | 24 | 54 |

**Key formulas:** basis definition p.4/pdf 34 · carry p.11–12/pdf 41–42 · IRR, both the simple form
and the version with an intervening coupon, p.15/pdf 45.

**The one exhibit to internalize:** Exhibit 1.4, p.14/pdf 44 — with a single deliverable bond, basis
= total carry to delivery; height = total carry, slope = −daily carry, converges to zero at delivery.
Every later complication is a deviation from this line.

---

## Chapter 2 — What Drives the Basis? · p.27 / **pdf 57**

Why futures are chronically "too low" on a pure carry basis: the short's optionality. This is the
intuition chapter — Chapter 3 does the mechanics, Chapter 4 does the pricing.

| Section | p. | pdf |
|---|---|---|
| The Short's Alternatives | 27 | 57 |
| Search for the Cheapest Bond to Deliver (incl. **Repo Specials**, p.32) | 28 | 58 |
| The Best Time to Deliver a Bond | 34 | 64 |
| Rules of Thumb (duration/crossover heuristics) | 36 | 66 |
| The Bond Basis Is Like an Option | 38 | 68 |
| Shifts in the Cheapest to Deliver | 41 | 71 |
| History of the Most Delivered Bond | 46 | 76 |
| — Examples of Buying and Selling the CTD Basis | 48 | 78 |
| The Importance of Embedded Options | 50 | 80 |

**Core payoff pictures** (the "aha" sequence — read these four back to back):
- Exh 2.6, p.39/pdf 69 — cash/futures price relationships with crossover points
- Exh 2.7, p.40/pdf 70 — low-duration bond basis ≈ **call** on futures
- Exh 2.8, p.41/pdf 71 — high-duration bond basis ≈ **put** on futures
- Exh 2.9, p.42/pdf 72 — mid-duration bond basis ≈ **straddle**

**Also:** Exh 2.3, p.33/pdf 63 — the three competing cheapness measures side by side (BNOC, IRR,
IRR-minus-own-term-repo). Exh 2.10, p.44/pdf 74 — BNOC scenario analysis. Exh 2.11, p.47/pdf 77 —
most-delivered-bond history.

---

## Chapter 3 — The Short's Strategic Delivery Options · p.51 / **pdf 81**

Decomposes the short's optionality into named, separately-valuable pieces.

| Section | p. | pdf |
|---|---|---|
| Structure of the Delivery Process — Delivery Process (3-day) · Delivery Month | 52 | 82 |
| **The Switch Option** | 54 | 84 |
| — Parallel Changes in Yield Levels | 55 | 85 |
| — Changes in Yield Spreads *(non-parallel; where the curve enters)* | 57 | 87 |
| — The End-of-Month Option | 64 | 94 |
| **Timing Options** *(incl. the wild card)* | 70 | 100 |

**Why this chapter matters for curve work:** p.57–63/pdf 87–93 is the explicit treatment of
**non-parallel** yield moves. Yields don't shift in parallel; the CTD switch is driven as much by
spread changes as by level. Exh 3.4 (yield spread relationships), Exh 3.6 (nonsystematic spread
changes), Exh 3.7 (effect of spread changes on the CTD).

**End-of-month option:** the futures price is *fixed* after last trading day while cash keeps moving —
Exh 3.9, p.65/pdf 95 and Exh 3.10, p.67/pdf 97.

**Wild card:** worked example and the payoff expression `(CF − 1) × (CF × F − P)` at p.72/pdf 102.

---

## Chapter 4 — The Option-Adjusted Basis · p.75 / **pdf 105**

Short chapter, high density. The valuation framework.

| Section | p. | pdf |
|---|---|---|
| An Outline for Pricing the Short's Delivery Options | 76 | 106 |
| — Option Structures · Valuing Switch and End-of-Month Options | 77 | 107 |
| — Expected Basis Net of Carry · Value of Early Delivery | 79 | 109 |
| Practical Considerations | 82 | 112 |
| Volatility and Distribution of Yield Levels · Yield Betas · Spread vols · Consistency checks · Term repo specials · Anticipated new issues | 82 | 112 |
| The Option-Adjusted Basis in Practice | 85 | 115 |
| — If the Basis Is Cheap, Futures Are Rich | 86 | 116 |
| — The CTD's BNOC Is Pure Option Value | 86 | 116 |

**The two rules that fall out** (p.76/pdf 106): the basis is *rich* if OAB > 0, i.e. if BNOC exceeds the
theoretical value of the delivery options — and the mirror statement, cheap basis ⇔ rich futures.

**Caveat worth reading:** p.84/pdf 114 — because yield changes across the curve are positively
correlated, **the square-root-of-time rule does not apply to yield-*spread* volatility.** Directly
relevant to any spread-vol scaling.

---

## Chapter 5 — Approaches to Hedging · p.87 / **pdf 117**

The most quantitatively useful chapter. Builds hedge ratios in four escalating layers.

| Section | p. | pdf |
|---|---|---|
| DV01 Hedge Ratios and Competing Objectives | 88 | 118 |
| Standard Industry Rules of Thumb — #1 (p.89) · #2 (p.90) | 88 | 118 |
| The Rules of Thumb in Practice | 91 | 121 |
| **Shortcomings of the Rules of Thumb** | 95 | 125 |
| Spot and Repo DV01s | 96 | 126 |
| — Forward Prices as a Function of Spot Yields and Repo Rates | 96 | 126 |
| — Short-Term Independence of Spot Yields and Term Repo Rates | 98 | 128 |
| Creating Synthetic Bonds with Forwards and Futures | 102 | 132 |
| Handling Repo Stub Risk | 103 | 133 |
| **Option-Adjusted DV01s** | 104 | 134 |
| **Yield Betas** | 108 | 138 |
| Putting It All Together | 109 | 139 |
| Reckoning the P/L of a Hedge | 111 | 141 |
| Evaluating Hedge Performance | 112 | 142 |
| Working with Durations | 113 | 143 |
| Duration of a Futures Contract | 116 | 146 |

**Rule of thumb #1:** `HR = Portfolio DV01 / (CTD DV01 / CF)`. Note at p.94/pdf 124: **this is what
Bloomberg returns** when you ask it for a hedge ratio. Exh 5.3 shows the screen.

**Two independent risks, not one** (p.96/pdf 126): futures price risk splits into spot-yield risk and
term-repo risk. Exh 5.5 and 5.6 (p.99–100/pdf 129–130) show levels and weekly changes of Treasury
yields vs. one-month repo, 1988–2003, to argue they're near-independent at short horizons.

**Option-adjusted DV01** (p.104/pdf 134): a plain DV01 hedge leaves you overhedged at low yields and
underhedged at high yields, with an abrupt jump at the crossover (p.106/pdf 136). Exh 5.8 (p.107) is
the worked calculation.

**Exh 5.11, p.113/pdf 143:** a futures hedge is like a long straddle. Worth a minute on its own.

### Appendix to Chapter 5: Better Hedges with Yield Betas? · p.118 / **pdf 148**
*Burghardt & Lyden, originally a 1998 Carr Futures research note.*

| Section | p. | pdf |
|---|---|---|
| Using Yield Betas to Improve Hedges | 119 | 149 |
| — Estimating Yield Betas for Treasury Bonds and Notes | 120 | 150 |
| Using Yield Betas to Improve Hedges *(worked example)* | 122 | 152 |
| Hedging Something Other Than the Current Long Bond | 123 | 153 |
| **When Yield Betas Can Get You into Trouble** | 124 | 154 |
| — Unstable Yield Betas (p.124) · Competing Hedge Ratios When ρ < 1.0 (p.125) | 124 | 154 |
| Competing Hedge Ratios · Sample Calculations (p.127) | 125 | 155 |

The premise: the curve flattens as yields rise and steepens as they fall, so a hedge instrument at a
different maturity point has systematically different yield variability — scale it by a beta. The
honest part is the second half: betas are **unstable over time** (Exh A5.6, betas by year, p.124), and
when correlation is below 1.0 there is **no single right hedge ratio** — the DV01-neutral,
minimum-expected-change, and minimum-variance hedges all differ (p.125–128/pdf 155–158). Exh A5.7
gives correlations and standard deviations of weekly yield changes for OTR issues, 1990–1997.

---

## Chapter 6 — Trading the Basis · p.129 / **pdf 159**

| Section | p. | pdf |
|---|---|---|
| Selling the Basis When It Is Expensive | 129 | 159 |
| — Selling the CTD Basis (p.130) · Selling the Basis of Non-Cheap Bonds (p.135) | 130 | 160 |
| Buying the Basis When It Is Cheap | 136 | 166 |
| Trading the Basis of "Hot-Run" Bonds | 138 | 168 |
| **Basis Trading When the CTD Is in Short Supply** | 140 | 170 |
| Trading the Calendar Spread | 142 | 172 |
| — Fair Values for Note Calendar Spreads (p.143) · Profiting from Mispricings (p.143) · Patterns (p.146) | 143 | 173 |
| Practical Considerations in Trading the Basis | 147 | 177 |
| — RP Specials (p.147) · Term vs. Overnight Financing (p.150) | 147 | 177 |
| — **Short Squeezes** (p.150) · The Short Squeeze of 1986 (p.151) | 150 | 180 |
| — Taking a Basis Trade into the Delivery Month | 153 | 183 |
| — Setting Up for Delivery | 156 | 186 |

**Squeeze math, p.141/pdf 171:** implied probability of delivery failure =
`CTD BNOC / (CTD BNOC − 2nd CTD BNOC)`. A clean way to read squeeze risk straight off the basis screen.

**Exh 6.6, p.138/pdf 168 — "High-Duration Bond Bases Widen When the Curve Flattens."** The direct
curve-level→basis link, and the most transferable exhibit in the chapter for RV work.

**Honest-accounting note, p.134/pdf 164:** "Why Were the Trades Successful?" — selling OTM options wins
most of the time, so a high win ratio is *not* evidence of edge. Worth re-reading before scoring any
short-vol strategy.

**Other exhibits:** Exh 6.2, p.133 — CTD 10-year basis, Jun-1998 to Jun-2004. Exh 6.7, p.139 — yield
spread patterns around 5-year auctions. Exh 6.10, p.146 — average calendar spreads by business days
to first notice.

---

## Chapter 7 — Volatility Arbitrage in the Treasury Bond Basis · p.157 / **pdf 187**

*Reprint of a Journal of Portfolio Management article, Spring 1993. Data is 1989–1992 — the reasoning
is durable, the numbers are historical.*

| Section | p. | pdf |
|---|---|---|
| Overview | 158 | 188 |
| The Options Embedded in Bond Futures | 158 | 188 |
| Calls, Puts, and Straddles | 159 | 189 |
| **Two Arenas for Trading Volatility** | 161 | 191 |
| The Option-Adjusted Bond Basis | 162 | 192 |
| History of Mispricings | 164 | 194 |
| Volatility Arbitrage | 165 | 195 |
| Report Card | 167 | 197 |
| Examples of Yield Enhancement | 169 | 199 |
| Leverage | 170 | 200 |
| Words of Caution | 170 | 200 |
| Other Applications | 171 | 201 |

**The thesis:** take implied vol from listed options on bond futures as the input, value the delivery
options with it, and any residual mispricing in the basis is a vol trade *between the two arenas*.
Exh 7.6, p.164/pdf 194 — OAB history, May 1989 to May 1992.

**Read the caveats** (p.166/pdf 196 and p.170/pdf 200): this is not classical arbitrage — the two
vehicles differ, and the position sizing discussion notes the practical cap is roughly 25× capital.

---

## Chapter 8 — Nine Eras of the Bond Basis · p.173 / **pdf 203**

Regime history. Read it as a case study in how a contract's dominant driver migrates over time.

| Era | Period | p. | pdf |
|---|---|---|---|
| The Birth and Maturation of Bond Futures | — | 173 | 203 |
| Volatility of Yields Since 1977 | — | 174 | 204 |
| 1. Cash and Carry | 1977–78 | 176 | 206 |
| 2. Negative Yield Curve *(Volcker)* | 1979–81 | 177 | 207 |
| 3. Positive Carry | 1982–84 | 178 | 208 |
| 4. **The Golden Age of Yield Enhancement** | 1985–89 | 179 | 209 |
| 5. Volatility Arbitrage | 1990–91 | 181 | 211 |
| 6. **The Death of Gamma** | Jun 1991 – Jun 1993 | 182 | 212 |
| 7. The Callables' Last Hurrah | Jul 1993 – 1994 | 186 | 216 |
| 8. The Long Dry Spell of the 11-1/4% | 1995–99 | 188 | 218 |
| 9. **6% Factors and the Rebirth of Basis Trading** | 2000 – ? | 189 | 219 |
| Changing of the Guard — Rise of Notes, Fall of Bonds | — | 193 | 223 |
| Where Do We Go from Here? | — | 195 | 225 |

**The structural break to know:** the CBOT cut the conversion-factor coupon from 8% to 6%, first
effective for the March 2000 contract (listed April 1999). This lowered crossover points across the
deliverable set and revived switch optionality after the dead 1995–99 stretch. p.189–192/pdf 219–222.
Anything written before ~1999 assumes 8% factors.

**Exh 8.1, p.175/pdf 205:** bond yield, curve slope, and yield vol on one chart — the backdrop for all
nine eras. **Exh 8.14–8.15, p.189–190:** CTD maturity vs. delivery option value, the cleanest picture
of why the dry spell happened.

---

## Chapter 9 — Non-Dollar Government Bond Futures · p.197 / **pdf 227**

| Section | p. | pdf |
|---|---|---|
| Active Non-Dollar Government Bond and Note Futures | 198 | 228 |
| — Transition to Electronic Trading (p.200) · Portfolio Equivalent Value (p.201) | 200 | 230 |
| Contract Specifications | 201 | 231 |
| Maturities, Settlement Windows, and Last Trading Days | 203 | 233 |
| — Cash Settlement of SFE's CGB Contracts (p.203) | 203 | 233 |
| Cash/Futures Relationships | 205 | 235 |
| — Key Cash Market Features (p.205) · Auction Cycles and Deliverable Sets (p.207) | 205 | 235 |
| — **Basis Reference Sheets: Germany, Japan, U.K.** (p.208) | 208 | 238 |
| — Optionality and Futures Mispricings (p.209) | 209 | 239 |
| Trading Themes in European Bond Bases | 211 | 241 |
| — Squeezes of CTD Bonds (p.213) · Bonds Exiting the Basket (p.214) · New Issuance (p.215) | 213 | 243 |
| A Word of Caution | 216 | 246 |

The German basket is narrow and the repo market behaves differently, so squeezes are a *structural*
theme rather than an occasional event. **Exh 9.13, p.214/pdf 244** — effect of a CTD squeeze on the
Mar/Jun 2001 Eurobund calendar spread (reached −70 cents).

Deeper market-convention reference lives in the appendices (D/E/F below), not here.

---

## Chapter 10 — Applications for Portfolio Managers · p.217 / **pdf 247**

| Section | p. | pdf |
|---|---|---|
| Hedging and Asset Allocation | 217 | 247 |
| — Advantages of Using Futures (p.218) | 218 | 248 |
| — Managing a Portfolio's Duration with Futures (p.219) | 219 | 249 |
| — Calculating the Duration of a Portfolio Containing Futures (p.220) | 220 | 250 |
| — Example: Targeting Portfolio Duration (p.221) · Solving for Hedge Ratios via Target Durations (p.221) | 221 | 251 |
| Synthetic Assets | 224 | 254 |
| — Trade Construction (p.225) · How Well Has It Worked? (p.227) | 225 | 255 |
| — Historical Record on Yield Enhancement (p.229) · Variations on a Theme (p.229) | 229 | 259 |
| Caveats | 232 | 262 |

**Exh 10.1, p.218/pdf 248** — bid/ask spreads across government bond markets; the liquidity argument
for using futures at all.

**Caveats, p.232/pdf 262:** cash bond gains/losses are unrealized while held; **futures mark to market
daily**. The financing and operational asymmetry is the practical catch on every synthetic-asset trade
in the chapter.

---

## Appendices

| # | Title | p. | pdf | Contents |
|---|---|---|---|---|
| **A** | Calculating Conversion Factors | 233 | **263** | The formula + worked example (4-7/8% of 2/15/12, June 2004 contract). Rounded to 4 decimals. |
| **B** | Calculating Carry | 235 | **265** | Coupon income and financing cost, with and without an intervening coupon. Notes that repo-market rules make financing costs an *estimate* even at a term rate. |
| **C** | Conventions in Major Government Bond Markets | 237 | **267** | Cross-market comparison table. *(JPMorgan Government Bond Outlines, Oct 2001.)* |
| **D** | German Federal Bonds and Notes (Bubills, Schätze, Bobls, Bunds) | 243 | **273** | Characteristics, transactions, settlement, interest/yield calcs, special features, screens. |
| **E** | Japanese Government Bonds (JGBs) | 251 | **281** | Same structure. Note: JGBs trade on a **simple-yield** basis. |
| **F** | Gilts | 257 | **287** | Same structure. Decimal pricing, strips, the various gilt types. |
| **G** | **Glossary** | 263 | **293** | Government-bond terms worldwide. Includes the day-count-rule variants at p.266/pdf 296. |
| — | Index | 271 | **301** | Book-page numbers; add 30. |

Appendices C and E are from the October 2001 JPMorgan outlines; D, F, and G from April 2005.

---

## Topic index — jump straight to it

*Page pairs are `printed / pdf`.*

**Definitions & mechanics**
- Basis definition `B = P − F×CF` — 4 / 34
- Conversion factors — 6 / 36; formula in App. A 233 / 263
- Invoice price — 7 / 37
- Carry — 11 / 41; formula in App. B 235 / 265
- Implied repo rate (both forms) — 15 / 45
- Basis net of carry (BNOC) — introduced 20 / 50; as pure option value 86 / 116
- EFP mechanics — 18 / 48
- RP vs. reverse RP — 24 / 54
- Delivery process, three-day — 52 / 82
- Contract specs, U.S. — 4 / 34 · non-dollar — 201 / 231

**Cheapest to deliver**
- The search / competing cheapness measures — 28–33 / 58–63
- Best delivery date within the month — 34 / 64
- Duration & crossover rules of thumb — 36 / 66
- Crossover points chart — 39 / 69
- Shifts in the CTD — 41 / 71
- Most-delivered-bond history — 46 / 76
- CTD in short supply; squeeze probability formula — 140–141 / 170–171

**Optionality**
- Basis as call / put / straddle — 38–42 / 68–72; again at 159 / 189
- Switch option — 54 / 84
- Non-parallel yield-spread effects — 57 / 87
- End-of-month option — 64 / 94
- Timing options and the wild card — 70–72 / 100–102
- Pricing framework (OAB) — 76 / 106
- Rich/cheap decision rules — 76 / 106 and 85–86 / 115–116
- Why √t fails for spread vol — 84 / 114

**Hedging**
- Rules of thumb #1 and #2 — 89–90 / 119–120
- What Bloomberg actually returns — 94 / 124
- Shortcomings of the rules of thumb — 95 / 125
- Spot vs. repo DV01 — 96 / 126
- Synthetic bonds from forwards/futures — 102 / 132
- Repo stub risk — 103 / 133
- Option-adjusted DV01 — 104 / 134
- Yield betas — 108 / 138; full treatment 118–128 / 148–158
- Beta instability, competing hedge ratios when ρ<1 — 124–128 / 154–158
- Hedge P/L attribution — 111 / 141
- Duration of a futures contract — 116 / 146
- Portfolio duration targeting with futures — 219–222 / 249–252

**Trading & market structure**
- Selling an expensive basis — 129 / 159
- Buying a cheap basis — 136 / 166
- Hot-run / OTR basis — 138 / 168
- Curve flattening widens high-duration bases — 138 / 168
- Calendar spreads: fair value, mispricing, seasonal patterns — 142–146 / 172–176
- RP specials — 32 / 62 and 147 / 177
- Term vs. overnight financing — 150 / 180
- Short squeezes; the 1986 squeeze — 150–152 / 180–182
- Carrying a trade into the delivery month — 153 / 183
- Volatility arbitrage, basis vs. listed options — 161–167 / 191–197
- Auction-cycle yield spread patterns — 139 / 169
- European squeezes and calendar spread distortion — 213–214 / 243–244
- Synthetic assets / yield enhancement — 224–231 / 254–261

**History**
- Yield level, curve slope, vol since 1977 — 175 / 205
- The nine eras — 176–192 / 206–222
- 8% → 6% conversion factor change — 189 / 219
- Notes overtaking bonds (volume & OI) — 193–194 / 223–224
