# Notes

## Factors

Every rates position decomposes into:

- **Direction** — parallel level moves (PC1, ~85% of curve variance)
- **Curve** — steepener/flattener (PC2, ~10%)
- **Basis** — between curves: swap spreads, cross-currency, UST vs Bund, swaps vs cash
- **Vol** — convexity, lives in the options/swaptions surface

Be deliberate about which factor you're taking. Naive curve trades carry hidden direction because the long leg moves more (e.g. 30Y vs 5Y in 5s30s). Regress curve on short leg, trade the residual → closer to pure curve.

Current research thread: **direction → curve interaction**. 10s vs 10s30s as the clean expression. Question is whether the level of 10Y has predictive power for the slope of 10s30s.

### How each factor plays into 10s vs 10s30s

**Direction (10Y level)**
Naive intuition: low yields → easy → steep curve, high yields → tight → flat. Reality is regime-dependent. Late cycle = bear steepening (yields up, curve steepens via term premium). Early cycle = bull steepening. Mid-cycle hiking = bear flattening. Same level move → opposite curve behavior depending on which leg is pricing.

**Curve (10s30s itself)**
PC2, but long-end has its own drivers most of the curve doesn't:
- Treasury issuance / supply (more 30Y auctions = cheaper 30s = steeper)
- Pension / insurance LDI demand (chronic bid for 30Y = flatter)
- Term premium repricing (slow-moving)

10s30s moves for reasons that have nothing to do with where 10Y is.

**Basis**
30Y swap spread tightening (dealer balance sheet, repo) → 30Y cash cheapens vs swaps → 10s30s steepens for non-fundamental reasons. Japanese lifer hedged-yield flows lean on the 30Y bid. Funding events fake out fundamental signals.

**Vol / Convexity** (the sneakiest one)
Mortgage convexity hedging:
- Rates fall → MBS duration shortens → hedgers receive → 10s30s flattens
- Rates rise → MBS duration extends → hedgers pay → 10s30s steepens

In high-vol regimes, 10s30s moves with hedging flows, not your fundamental thesis. MOVE / 1m10y / 1m30y swaption vol tells you when convexity dominates.

### Four diagnostics
1. Scatter 10Y vs 10s30s, colored by regime (vol bucket, cycle phase, Fed stance) — same scatter looks like four different relationships when split.
2. Rolling 60d corr of Δ10Y vs Δ10s30s — when does it sign-flip? What was the regime?
3. Event studies: 2019 inversion, Mar 2020, 2022 hiking, Mar 2023 SVB. What did each factor do?
4. Regression: `Δ10s30s ~ Δ10Y + Δswap_spread + ΔMOVE + Δmortgage_OAS`. 10Y coefficient = pure signal; the rest are noise channels to clean out.

---
