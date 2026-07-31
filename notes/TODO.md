# Project TODO

North star: a live, timestamped, auditable macro trading platform with a real track record.

Focused research, promotion, and strategy checklists: [`RESEARCH_ROADMAP.md`](RESEARCH_ROADMAP.md).

**Direction update (2026-07-31):** the actual repo (see root [`README.md`](../README.md)) is now the authoritative source for architecture and schema, not this file or `PLAN.md`. Several items below were built, just differently than originally scoped here — those are struck through with a pointer to what exists instead. Checked items are confirmed done. Everything else is untouched because it wasn't verified either way, not because it's still accurate as written.

---

## Data Foundation

**Understand the instruments**
- [ ] Document what an on-the-run (OTR) bond is and why it matters - OTR is the most recently auctioned Treasury at each tenor, tightest bid/ask, used for live P&L tracking (`md.headline`). Generics (`md.index_eod`) are constant-maturity benchmark series used for research.
- [ ] Document breakevens - nominal yield minus TIPS yield at matching tenor. Represents the inflation rate the market is pricing over that horizon. `md.breakeven` is built by `build_breakeven.py`.
- [ ] Document swap spreads - Treasury yield minus same-maturity swap rate (SOFR or LIBOR-era). Negative = Treasuries trade rich to swaps. Reflects balance sheet cost, dealer positioning, and flight-to-quality flows.
- [ ] Document the vol surface - MOVE index is the rates equivalent of VIX (1m Treasury vol). Swaption vol is the implied vol of an option on a swap. `md.swaption_vol` has expiry x tenor x strike surface from Bloomberg.
- [ ] Document real rates - nominal minus inflation swap (ZCIS). `utils/rates.py` has `synthetic_real_rate` and `with_synthetic_real_rates`.
- [ ] Document the curve - 2s5s, 2s10s, 5s30s are just spreads between two points. Beta-weighted curve means you size the legs so the DV01 is equal, removing hidden duration.

**Audit existing data**
- [ ] Run `data_pull/ref.py` end to end and confirm all steps pass
- [ ] Audit `md.index_eod` - what tickers exist, what date range, any gaps in recent history
- [ ] Audit `md.headline` - OTR bond coverage by tenor, confirm current OTRs are updating
- [ ] Audit `md.breakeven` - confirm 5Y, 10Y, 20Y, 30Y are populated and current
- [ ] Audit `md.swaption_vol` - confirm surface is populated and recent
- [ ] Check CFTC positioning data (`pull_cftc.py`) - is it current, what fields exist

**Fill data gaps**
- [ ] Build swap spread series - Treasury yield minus swap rate at 2Y, 5Y, 10Y, 30Y - store in DB or as a derived series in research
- [ ] Confirm SOFR forward curve is available and correct (key anchor for duration model)
- [ ] Confirm oil (CL1), DXY, SPX, IG/HY credit spreads are in `md.index_eod`
- [ ] Add any missing macro series needed for research anchors

**Data access layer**
- ~~Write a single `load_research_data(start, end)` function that returns a clean wide DataFrame of all research-ready series with consistent date alignment~~ (built differently: `utils/market_data.py`'s `load_wide(TICKERS, start, ...)` takes a per-strategy ticker dict rather than one blanket all-series loader)
- [x] Confirm `query_db` in `utils/helpers.py` is the standard entry point - no raw psycopg calls in research files
- [ ] Build a lightweight data health check - confirm no stale series, no gaps in last 30 days

---

## Valuation Model Techniques

The model quality determines the signal quality. Work through these in order.

**Single-factor (baseline)**
- [ ] Audit which single-factor anchors produce the cleanest residuals for 10Y - IC, half-life, ADF stat
- [ ] Document the winner and why (this is the baseline everything else gets compared to)

**Pairs / spread trading**
- [x] `book/duration/spread_rv.py` - trade 10Y vs 5y5y fwd inflation as a two-leg spread (in progress)
- [ ] Extend to other pairs: 10Y vs BE10, 10Y vs 5y5y SOFR fwd, 5y vs 5y BE, 10Y vs 30Y
- [ ] Verify: P&L of the pairs trade = fading the spread residual (hedge ratio from rolling beta)
- [ ] Compare pairs: IC, hit rate, Sharpe at 20d - which pairing has the most consistent edge?

**Multi-factor regression**
- [ ] Build multi-factor OLS model for 10Y - 2 to 4 anchors chosen by forward IC not in-sample R-squared
- [ ] Test whether multi-factor residual has better OOS IC than the best single-factor
- [x] Check beta stability - unstable betas mean the model is overfit (`beta_cv` in `stats/`, per README's model-stability gates)

**PCA decomposition**
- [ ] Apply PCA to the full rates curve - confirm PC1/PC2/PC3 = level/slope/curvature
- [ ] Model 10Y as a function of its PC loadings, compute PCA residual
- [ ] Compare PCA residual IC to single-factor and multi-factor at same horizon
- [ ] Extend to inflation surface PCA (BEs, ZCIS, TIPS)

**Combine and evaluate**
- [ ] Summary table: single-factor vs multi-factor vs PCA vs best pair - IC, hit rate, Sharpe at 20d OOS
- [ ] Pick the model that wins OOS and use it as the primary signal

---

## Research - Duration Signal

**Model foundation**
- [ ] Finalize single-factor anchor models for 10Y (oil, DXY, equities, breakevens, real rates, SOFR fwds)
- [ ] Choose the best 2-3 multi-factor model combinations and document rationale
- [ ] Confirm residual construction has no lookahead bias

**Signal backtest**
- [ ] Backtest raw residual signal at 5d, 20d, 60d horizons - IC, hit rate, Sharpe
- [ ] Build summary table comparing all anchors side by side
- [ ] Identify which anchors have the strongest and most stable edge

**Regime conditioning**
- [x] Add `hurst_exponent` and `roll_hurst` to `stats/ou.py`
- ~~Add `resid_hurst_60d` and anchor trend features to `signal_context.py`~~ (built differently: `signal_context.py` has `beta_trend`, `r2_trend`, `zscore_mom_*`, `vol_ratio` instead of an explicit rolling-Hurst feature)
- [x] Add residual range breakout features to `signal_context.py` (`resid_pct_hi_Nd`, `resid_dist_hi_Nd`, `resid_dist_lo_Nd`)
- [x] Run OOS edge test across all features, all anchors (`oos_edge_test` / `oos_edge_summary` / `oos_edge_summary_fast` in `signal_context.py`)
- [x] Identify the 2-3 filters with the most consistent OOS Sharpe lift (`filtered_sharpe_summary` picks the best filter per anchor by `sharpe_filtered`)
- [x] Define regime gate: conditions under which the signal fires vs stands down (`backtest.lab.gate_scan` + `entry_filter_fn`, per README's parameter-lab section)

**Signal definition**
- [ ] Write the final signal function `signal_fn` returning standard schema (signal, size, confidence, vol, time_stop)
- [ ] Document entry/exit/stop logic

---

## Research - Curve Signal

- [ ] Build beta-weighted 5s30s residual model
- [ ] Strip hidden duration from naive curve spread
- [ ] Test anchors: Fed path, inflation, term premium, supply
- [ ] Backtest residual signal at standard horizons
- [ ] Apply regime conditioning (same framework as duration)
- [ ] Write `signal_fn` for curve

---

## Research - Inflation Signal

- [ ] Build 10Y breakeven model vs oil, DXY, CPI trend, real yields
- [ ] Construct breakeven residual
- [ ] Distinguish breakeven signal from real yield signal from nominal signal
- [ ] Backtest residual signal
- [ ] Apply regime conditioning
- [ ] Write `signal_fn` for inflation

---

## Research - Rates Vol Signal

- [ ] Build implied vs realized vol model for 1m10y swaption
- [ ] Construct vol richness residual
- [ ] Add event calendar conditioning (CPI, FOMC, NFP proximity)
- [ ] Backtest vol signal
- [ ] Write `signal_fn` for vol

---

## Research - Cross-Market RV

- [ ] Build USD-hedged gilt vs UST spread model
- [ ] Incorporate FX forwards and cross-currency basis correctly
- [ ] Backtest hedged spread residual
- [ ] Write `signal_fn` for cross-market RV

---

## Research - Event-Driven

- [ ] Build CPI surprise reaction model (first 15min price action vs continuation)
- [ ] Build auction tail reaction model
- [ ] Backtest post-event signals
- [ ] Write `signal_fn` for event-driven

---

## Signal Ledger Infrastructure

Superseded as a whole: the ledger was built as an append-only parquet
(`store/signal_ledger.parquet`), not JSON files + git commits. See
[`dashboard/README.md`](../dashboard/README.md) for the actual schema and
the atomic-write (`os.replace`) mechanism that gives it the "can't be
silently edited" property a different way.

- ~~Define signal JSON schema (all required fields - see section 7.2 in PLAN.md)~~
- ~~Write `write_signal.py` - saves signal JSON to `signals/YYYY/YYYY-MM-DD.json`~~
- ~~Write `validate_signal.py` - rejects malformed or incomplete signals~~
- ~~Write `hash_signal.py` - SHA256 of each signal file~~
- ~~Wire git auto-commit: daily signal file committed with timestamp~~
- ~~Test: generate a fake signal, log it, confirm it cannot be silently edited~~ (append-only parquet + atomic write is the actual mechanism)

---

## Model Versioning

- [ ] Add model version + git commit hash to every signal record
- [x] Add data as-of timestamp to every signal record (`data_asof_ts` in `signal_ledger.parquet`)
- [ ] Write version bump protocol: any model change increments version

---

## Performance Tracking

**Forecast book**
- [ ] Build `forecast_book.parquet` - one row per signal, outcome filled in daily
- [ ] Calculate direction accuracy at 1d, 1w, 1m, 3m for each signal
- [ ] Calculate IC and average move by strategy family and conviction bucket

**Strategy book**
- [ ] Define position construction: how each signal maps to a trade expression and size
- [ ] Build `strategy_book.parquet` - daily P&L per trade
- [ ] Build NAV series with transaction cost assumptions
- [ ] Build drawdown series
- [ ] Build basic tearsheet: return, vol, Sharpe, max DD, hit rate, profit factor

**Attribution**
- [ ] P&L by strategy family
- [ ] P&L by horizon
- [ ] P&L by conviction bucket
- [ ] P&L by regime (MR confirmed vs uncertain)

---

## Dashboard

- [x] Basic Dash app skeleton with routing (`dashboard/app.py`)
- ~~Signal ledger page - every signal ever logged, searchable and filterable~~ (built differently: **Live Overview** groups by traded target instead of a flat ledger browse; **Signal Deep Dive** is the per-signal detail view — see `dashboard/README.md`)
- [ ] NAV page - total return, Sharpe, max drawdown, monthly returns table
- [ ] Forecast accuracy page - IC and direction accuracy by strategy and horizon
- [ ] Strategy performance page - breakdown by family
- [ ] Research notes page - monthly writeups
- [ ] Make losing signals visible and easy to find (no cherry-picking in the UI)

---

## Live Signal Generation

- [ ] Write daily signal runner - pulls fresh data, runs all `signal_fn`s, logs output
- [ ] Schedule runner (cron or task scheduler)
- [ ] Alert on runner failure
- [ ] Confirm each daily run produces a new committed signal file before market open

---

## Broker Reconciliation

- [ ] Build broker statement importer for actual trades
- [ ] Match live fills to model signals
- [ ] Calculate actual vs model P&L
- [ ] Track slippage and execution quality
- [ ] Surface discrepancies in dashboard

---

## Hardening

- [ ] Backtest framework passes lookahead audit (`_test_lookahead.py` covers all models)
- [ ] All signal functions tested on out-of-sample holdout period
- [ ] No signal claims live performance that came from a backtest
- [ ] Public dashboard clearly labels backtest vs paper vs live
