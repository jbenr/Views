# Project TODO

North star: a live, timestamped, auditable macro trading platform with a real track record.

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
- [ ] Write a single `load_research_data(start, end)` function that returns a clean wide DataFrame of all research-ready series with consistent date alignment
- [ ] Confirm `query_db` in `utils/helpers.py` is the standard entry point - no raw psycopg calls in research files
- [ ] Build a lightweight data health check - confirm no stale series, no gaps in last 30 days

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
- [ ] Add `hurst_exponent` and `roll_hurst` to `stats/ou.py`
- [ ] Add `resid_hurst_60d` and anchor trend features to `signal_context.py`
- [ ] Add residual range breakout features to `signal_context.py`
- [ ] Run OOS edge test across all features, all anchors
- [ ] Identify the 2-3 filters with the most consistent OOS Sharpe lift
- [ ] Define regime gate: conditions under which the signal fires vs stands down

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

- [ ] Define signal JSON schema (all required fields - see section 7.2 in PLAN.md)
- [ ] Write `write_signal.py` - saves signal JSON to `signals/YYYY/YYYY-MM-DD.json`
- [ ] Write `validate_signal.py` - rejects malformed or incomplete signals
- [ ] Write `hash_signal.py` - SHA256 of each signal file
- [ ] Wire git auto-commit: daily signal file committed with timestamp
- [ ] Test: generate a fake signal, log it, confirm it cannot be silently edited

---

## Model Versioning

- [ ] Add model version + git commit hash to every signal record
- [ ] Add data as-of timestamp to every signal record
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

- [ ] Basic Dash app skeleton with routing
- [ ] Signal ledger page - every signal ever logged, searchable and filterable
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
