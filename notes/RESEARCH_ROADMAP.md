# Research Roadmap

Purpose: produce falsifiable evidence that a signal process works, then make it easy to resume. Check a box only when its evidence is saved or linked.

## Resume here

Last updated: 2026-07-30

- **Active:** validate the research/promotion method on one existing curve strategy.
- **Next:** specify SFR–FF basis sign, target, and data panel.
- **Then:** choose one bond index and one inflation forecast target.
- **Rule:** no more than three active tasks at once.

## Shared signal process

Promotion path: **Idea → Spec → Data-ready → Candidate → Finalist → Shadow → Promoted / Rejected**.

### 1. Spec

- [ ] State the economic mechanism before testing.
- [ ] Define target, sign, horizon, trade, and P&L units.
- [ ] State what would falsify the idea.
- [ ] Freeze the initial feature families and search budget.

### 2. Data-ready

- [ ] Audit coverage, gaps, stale values, rolls, and units.
- [ ] Verify every feature is available at the decision timestamp.
- [ ] Use point-in-time vintages for revised macro/index data.
- [ ] Save the data as-of timestamp and source list.

### 3. Candidate

- [x] Coarse causal search exists: `--predict`.
- [x] Exit search exists: `--exit`.
- [x] Exact engine exists: `--sweep`.
- [x] One-command funnel exists: `--cook`.
- [ ] Log every attempted candidate, not only finalists.
- [ ] Compare against simple economic and no-skill baselines.
- [ ] Run sign-flip, shuffled-target, and placebo-feature tests.

### 4. Finalist

- [ ] Rerun the **entire** selection funnel in purged, embargoed walk-forward windows.
- [ ] Include costs, carry, roll, slippage, and realistic execution timing.
- [ ] Require parameter-neighborhood stability; reject isolated peaks.
- [ ] Check eras, events, best-trade removal, and concentration.
- [ ] Compute DSR/PBO using the full candidate ledger.
- [ ] Write explicit pass/fail thresholds before viewing final OOS results.

### 5. Shadow / promotion

- [ ] Freeze parameters, code hash, data version, and rationale.
- [ ] Run unchanged in shadow mode through enough independent events.
- [ ] Compare realized fills/outcomes with backtest assumptions.
- [ ] Promote only with linked evidence and a named owner.
- [ ] Define monitoring, degradation, pause, and retirement rules.
- [ ] Keep rejected and retired signals visible.

## Make the funnel easier

Verdict: keep the statistical stages; simplify the interface and bookkeeping.

- [ ] Make `--cook` the canonical research command.
- [ ] Add a run manifest: config, git SHA, data as-of, candidates, costs, artifacts.
- [ ] Add `--status`: last completed stage, pass/fail, artifact paths, next action.
- [ ] Add `--resume`: continue only when config/data hashes still match.
- [ ] Add `--promote`: enforce gates, freeze parameters, bump model version.
- [ ] Generate one compact HTML/Markdown report per run.
- [ ] Separate **research score** from **promotion decision**.

## Strategy: SOFR–Fed Funds basis

Working sign convention:

- Rate basis = `SOFR − EFFR`; wider means secured funding is expensive versus Fed Funds.
- Futures display `SER − FF = EFFR − SOFR`; it has the opposite sign.
- Abundant cash relative to collateral tends to lower SOFR; cash scarcity, collateral supply, or dealer constraints tend to raise it.
- Bill buying alone is not the mechanism; test repo cash, collateral, reserves, and balance-sheet channels separately.

### Define and build

- [x] Pull `FF1–8`, `SER1–8`, and `SFR1–8`; retain actual contracts.
- [ ] Verify histories, contract mapping, rolls, settlement, and missing ranks.
- [ ] Build monthly `SER − FF` and rate-space `SOFR − EFFR` curves.
- [ ] Build each SFR contract against a date/DV01-matched FF strip.
- [ ] Choose target: basis level, change, event move, or mean reversion.

### Cash / collateral panel

- [ ] NY Fed: SOFR, EFFR, TGCR/BGCR, volumes, and percentiles.
- [ ] Fed H.4.1: reserve balances, TGA, ON RRP, and changes.
- [ ] Treasury: bill issuance, settlements, maturities, and operating cash.
- [ ] SEC/ICI: government MMF assets, flows, WAM/WAL, repo, and bill holdings.
- [ ] NY Fed: dealer financing, positions, and Treasury fails.
- [ ] Calendar: tax dates, month/quarter/year-end, FOMC, debt-limit episodes.

### Test

- [ ] Plot unconditional basis distribution, persistence, and roll behavior.
- [ ] Event-study settlements, tax dates, quarter-ends, and reserve drains.
- [ ] Test feature signs individually before multivariate models.
- [ ] Compare monthly SER–FF with quarterly SFR–FF results.
- [ ] Walk forward with futures costs, carry, rolls, and execution lag.
- [ ] Decide: forecast widening/narrowing or fade dislocations.

Sources: [NY Fed reference rates](https://www.newyorkfed.org/markets/reference-rates), [Fed H.4.1](https://www.federalreserve.gov/releases/h41/), [Daily Treasury Statement](https://fiscal.treasury.gov/accounting/daily-treasury-statement), [SEC MMF statistics](https://www.sec.gov/data-research/statistics-data-visualizations/money-market-fund-statistics).

## Strategy: bond index additions / deletions

### Scope and data

- [ ] Pick one index first; start with Bloomberg US Aggregate or a Treasury sub-index.
- [ ] Obtain methodology, rebalance calendar, projected universe, and return universe.
- [ ] Store point-in-time constituent snapshots and change timestamps.
- [ ] Map CUSIP, ratings, amount outstanding, maturity, calls, sector, and liquidity.
- [ ] Gather ETF holdings/AUM; separate passive assets from benchmark-aware assets.

### Flow estimate

- [ ] Reproduce eligibility rules exactly.
- [ ] Predict additions/deletions before the projected-universe change.
- [ ] Estimate weight change: eligible market value / index market value.
- [ ] Estimate forced flow: weight change × passive AUM × implementation factor.
- [ ] Convert flow to days of volume, issue size, spread/DV01, and dealer capacity.

### Test and trade

- [ ] Define tradable timestamps: prediction, announcement, month-end, next open.
- [ ] Event-study price/spread/volume around inclusion and deletion.
- [ ] Use matched controls by duration, rating, sector, and liquidity.
- [ ] Separate new issue, maturity, downgrade, call, and size-threshold events.
- [ ] Test pre-positioning versus month-end execution and post-event reversal.
- [ ] Include bid/ask, borrow, financing, hedge, and crowding costs.
- [ ] Validate predicted flows against observed ETF holdings changes.

Source: [Bloomberg fixed-income index methodology](https://assets.bbhub.io/professional/sites/10/20230426_Fixed-Income-Index-Methodology.pdf).

## Strategy: inflation

Keep two projects separate: **forecast the release** and **trade inflation markets**.

### Inflation forecast

- [ ] Choose first target: headline/core CPI m/m SA and release date.
- [ ] Build as-released vintages, release calendar, revisions, and weights.
- [ ] Reproduce headline/core aggregation from component indexes.
- [ ] Benchmark against no-change, seasonal, AR, consensus, and Cleveland Fed.
- [ ] Build component groups: shelter, vehicles, energy, food, medical, travel, other core.
- [ ] Map timely inputs to components; document lag and publication time.
- [ ] Treat PPI/import prices as component inputs, not CPI substitutes.
- [ ] Run rolling pseudo-real-time forecasts; never use revised future vintages.
- [ ] Score MAE/RMSE, sign, surprise error, stability, and calibration.

### Inflation-market signal

- [ ] Choose instrument: CPI fixing, TIPS breakeven, inflation swap, or event future.
- [ ] Measure forecast minus market-implied/consensus inflation.
- [ ] Separate expected inflation, real-yield beta, liquidity, and risk premium.
- [ ] Build breakeven fair value versus oil, FX, growth, policy, and liquidity.
- [ ] Test release reaction separately from medium-horizon valuation reversion.
- [ ] Include carry, seasonality, indexation lag, convexity, and bid/ask.
- [ ] Attribute P&L to nominal, real-yield, inflation, and liquidity legs.

Sources: [BLS CPI weights](https://www.bls.gov/cpi/tables/relative-importance/), [BLS seasonal adjustment](https://www.bls.gov/cpi/seasonal-adjustment/using-seasonally-adjusted-data.htm), [BLS PPI methodology](https://www.bls.gov/ppi/methodology-reports/methodology.htm), [Cleveland Fed nowcast](https://www.clevelandfed.org/indicators-and-data/inflation-nowcasting).

## Immediate next actions

- [ ] Audit one existing curve strategy against the shared checklist; record every gap.
- [ ] Write the one-page SFR–FF hypothesis/spec with exact sign and target.
- [ ] Audit availability of the SFR–FF cash/collateral panel.
- [ ] Choose the first index and secure historical projected constituents.
- [ ] Choose the first inflation target and build its release/vintage table.
