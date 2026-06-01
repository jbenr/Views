# Macro Signal Ledger

A live, timestamped, auditable macro strategy tracker built to prove ex-ante investment judgment, systematic signal quality, and disciplined risk-taking across rates, inflation, volatility, cross-market relative value, and event-driven macro trades.

This project is not just a backtester.

The goal is to build a real research-and-trading evidence stack: every signal is generated live, logged before the outcome is known, versioned, timestamped, tracked through time, and evaluated honestly.

---

## 1. The Goal

The goal of this project is to prove that I can build and manage a real macro trading process.

Not just:

- show a good backtest
- explain a market view after the fact
- cherry-pick good calls
- build a model that only looks good in research

But instead:

- generate live macro signals
- log them before the outcome is known
- track every signal honestly
- convert signals into standardized model portfolios
- evaluate performance by strategy, asset class, horizon, and risk unit
- reconcile live trades where applicable
- build a public-facing track record that is difficult to fake

The core idea:

> If a signal is real, it should survive being timestamped, tracked, and judged out of sample.

---

## 2. The Philosophy

This project is built around a simple idea: macro trading is not about having one magic model.

It is about building a repeatable process for identifying dislocations, sizing them intelligently, managing downside, and learning from live outcomes.

The process should answer:

- What did the model believe?
- When did it believe it?
- What market expression did it choose?
- How much risk did it want?
- What happened next?
- Was the thesis right?
- Was the trade construction right?
- Was the sizing right?
- Was the drawdown acceptable?
- Did the strategy still make sense after costs?

This system exists to keep the research honest.

---

## 2.1 Stat-Arb Discovery Engine

The deeper goal is to build a laboratory for discovering conditional relative-value edges.

Given a target market, like 10Y rates, the process should systematically test what explains it, what residuals are left over, what state variables make those residuals more or less tradable, and what trading policy works conditional on those states.

Residual volatility is one example of a useful conditioning variable. But the tool should not assume residual volatility is the answer upfront. It should help discover whether residual volatility, residual momentum, beta, R2, drawdown, z-score streaks, macro volatility, curve slope, cross-asset momentum, liquidity proxies, or something else explains when a residual becomes tradable.

The workflow:

1. Choose a target: 10Y yield, curve slope, breakeven, swap spread, volatility point, or cross-market spread.
2. Choose an explanatory model: regress the target on candidates such as 30Y, DXY, SOFR forwards, breakevens, oil, MOVE, mortgage convexity, equities, or other macro/rates variables.
3. Construct the residual: isolate the part of the target not explained by the model.
4. Generate candidate state variables: residual vol, residual momentum, rolling beta, R2, drawdown, streaks, macro vol, curve shape, cross-asset momentum, liquidity, positioning, or event regimes.
5. Ask when the residual is predictive, when it is noise, when mean reversion works, when entries should be wider, when exits should be faster, and when the strategy should not trade.
6. Convert those findings into conditional policies: if regime A, trade with params X; if regime B, trade with params Y; if regime C, stand down or trade differently.

This is stat arb because the process is not: "I have a discretionary view that 10Y should rally."

The process is:

> There is a statistical relationship between 10Y and some explanatory basket. When 10Y deviates from that relationship, the deviation may mean-revert. But the quality of that mean reversion depends on measurable state variables.

The arbitrage is not riskless. The "arb" is the systematic exploitation of relative mispricing versus a statistical fair-value model. The edge comes from identifying the right fair-value model, the right residual, the right conditioning variables, and the right regime-dependent trading rule.

The real objective is not merely to optimize a strategy. It is to discover when a statistical relationship becomes tradable.

---

## 3. What This Is

Macro Signal Ledger is a framework for live macro signal generation and performance tracking.

It will include:

- systematic signal generation
- timestamped signal logs
- model version tracking
- feature/data version tracking
- model portfolio construction
- live performance attribution
- trade-level analytics
- broker reconciliation where possible
- public dashboarding
- monthly research reviews

The public output should look like a serious macro research book.

The private code can contain the actual alpha logic.

---

## 4. What This Is Not

This is not:

- a discretionary trade journal only
- a collection of screenshots
- a spreadsheet of hand-picked trades
- a retrofitted backtest
- a signal generator with no accountability
- a public dump of proprietary model logic

The system should make it hard to rewrite history.

---

## 5. Strategy Universe

The platform will track multiple macro strategy families.

Each strategy family can have many individual signals, but all signals should follow the same core lifecycle:

1. Generate signal
2. Log signal
3. Assign trade expression
4. Size risk
5. Track live outcome
6. Attribute performance
7. Review and improve

---

# 6. Strategy Families

## 6.1 Directional Duration

Directional duration strategies express outright views on the level of interest rates.

These can be traded through:

- Treasury futures
- cash Treasuries
- swaps
- swap spreads
- inflation-linked products
- real rates
- breakevens

Example views:

- long duration
- short duration
- receive fixed in swaps
- pay fixed in swaps
- long real rates
- short real rates
- long breakevens
- short breakevens
- long/short swap spreads

Core research questions:

- Is the market pricing too many or too few cuts?
- Are yields rich or cheap versus macro fundamentals?
- Are real yields too high or too low versus growth, inflation, risk appetite, and policy?
- Are swap spreads dislocated versus balance sheet, issuance, funding, and liquidity variables?
- Is the market overreacting or underreacting to economic data?

Potential features:

- Fed pricing
- OIS forwards
- real yields
- breakevens
- inflation swaps
- oil
- dollar
- equities
- credit spreads
- financial conditions
- payrolls
- CPI
- ISM
- unemployment claims
- Treasury supply
- auction tails
- volatility
- positioning proxies

Example signal:

```json
{
  "strategy_family": "directional_duration",
  "signal_name": "10y_nominal_fair_value",
  "instrument": "10Y Treasury future / cash 10Y equivalent",
  "direction": "long_duration",
  "horizon": "1m",
  "conviction": 0.72,
  "model_score": 1.8,
  "entry_level": "10Y yield 4.62%",
  "rationale": "10Y yield is cheap versus macro fair value after controlling for inflation, Fed path, oil, dollar, and risk sentiment."
}
```

---

## 6.2 Curve

Curve strategies express relative value views between different points on the rates curve.

These are similar to duration trades, but the focus is on the shape of the curve rather than the outright level of rates.

Trade types:

- 2s10s steepener/flattener
- 5s30s steepener/flattener
- 2s5s10s fly
- 5s10s30s fly
- front-end vs belly
- belly vs long-end
- beta-weighted curve structures
- PCA-neutral curve trades
- carry/rolldown-aware curve trades

Core research questions:

- Is the curve too steep or too flat versus macro conditions?
- Is the belly rich or cheap versus wings?
- Is the curve shape consistent with Fed pricing, inflation risk, growth risk, and term premium?
- Is the trade truly curve risk, or just hidden duration risk?
- Does beta-weighting improve stability versus simple spread construction?

Potential features:

- Fed path
- terminal rate
- cuts priced over 6m/1y/2y
- breakevens
- real yield curve
- nominal curve
- term premium estimates
- Treasury issuance
- auction cycle
- oil
- DXY
- equity returns
- credit spreads
- realized volatility
- rates vol
- curve momentum
- curve carry and rolldown

Example signal:

```json
{
  "strategy_family": "curve",
  "signal_name": "beta_weighted_5s30s_residual",
  "instrument": "5s30s Treasury curve",
  "direction": "steepener",
  "horizon": "2w_3m",
  "conviction": 0.68,
  "model_score": 2.1,
  "risk_unit": "beta_weighted_dv01",
  "rationale": "5s30s is too flat versus macro fair value and curve regime features."
}
```

Important implementation point:

All curve trades should be tracked in multiple forms:

- simple spread
- regression beta-weighted spread
- DV01-neutral spread
- PCA-neutral spread where applicable
- carry-adjusted version

This helps prove whether the signal is real or just an artifact of naive construction.

---

## 6.3 Rates Volatility

Rates volatility strategies focus on the level, slope, and relative value of the rates vol surface.

The initial bias will likely be toward selling volatility, but the system should not blindly sell vol. It should identify when the compensation for selling vol is attractive versus realized volatility, event risk, skew, term structure, and macro regime.

Trade types:

- short gamma
- long gamma
- short straddles/strangles
- payer/receiver skew trades
- vol term structure steepeners/flatteners
- forward vol trades
- expiry switches
- swaption vol RV
- conditional curve/duration expressions through options

Core research questions:

- Is implied vol too high or too low versus realized vol?
- Is the term structure too steep or too flat?
- Is event premium overpriced or underpriced?
- Is the market overpaying for tails?
- Is short vol compensated after accounting for jump/event risk?
- Does vol richness line up with macro uncertainty or positioning stress?

Potential features:

- implied vol
- realized vol
- implied-realized spread
- event calendar
- CPI/FOMC/NFP proximity
- curve level and slope
- MOVE index
- swaption vol surface
- futures options vol
- skew
- vol carry
- vol rolldown
- liquidity proxies
- risk sentiment

Example signal:

```json
{
  "strategy_family": "rates_vol",
  "signal_name": "1m10y_implied_realized_richness",
  "instrument": "1m10y swaption vol",
  "direction": "sell_vol",
  "horizon": "2w_1m",
  "conviction": 0.64,
  "model_score": 1.6,
  "rationale": "1m10y implied vol is rich versus realized vol and event-adjusted fair value."
}
```

Risk rule:

Vol strategies must track tail risk separately from mark-to-market P&L.

At minimum, each vol trade should include:

- premium collected/paid
- breakeven move
- realized move
- implied-realized spread
- max drawdown
- event exposure
- stress scenario loss
- gamma/theta/vega attribution

---

## 6.4 Inflation

Inflation strategies model inflation-linked markets and trade deviations between TIPS, breakevens, inflation swaps, commodities, FX, and macro inflation indicators.

Trade types:

- long/short breakevens
- long/short real yields
- TIPS versus nominals
- breakeven curve trades
- inflation swap trades
- real yield curve trades
- beta-adjusted TIPS relative value

Core research questions:

- Are breakevens rich or cheap versus inflation fundamentals?
- Are TIPS cheap versus nominal Treasuries after accounting for liquidity and beta?
- Are real yields too high or too low versus growth and Fed reaction function?
- Are inflation markets underreacting to oil, commodities, wages, or FX?
- Is the inflation curve mispriced relative to spot inflation and forward inflation risk?

Potential features:

- CPI
- core CPI
- PCE
- oil
- gasoline
- commodities
- wages
- dollar
- import prices
- inflation swaps
- breakevens
- real yields
- nominal yields
- Fed pricing
- inflation expectations
- TIPS liquidity proxies
- auction cycles
- seasonal inflation effects

Example signal:

```json
{
  "strategy_family": "inflation",
  "signal_name": "10y_breakeven_macro_beta_residual",
  "instrument": "10Y breakeven inflation",
  "direction": "long_breakeven",
  "horizon": "1m_3m",
  "conviction": 0.70,
  "model_score": 2.0,
  "rationale": "10Y breakevens are cheap versus oil, dollar, real yield regime, and inflation trend features."
}
```

Implementation point:

Inflation models should explicitly distinguish between:

- nominal yield signal
- real yield signal
- breakeven signal
- inflation swap signal
- TIPS liquidity signal

A good breakeven trade can fail if the real yield leg dominates, so attribution matters.

---

## 6.5 Cross-Market Relative Value

Cross-market strategies compare global interest rate markets on a common benchmark.

The key idea is to convert foreign bonds or rates into USD-equivalent terms using FX forwards, cross-currency basis, and hedging costs, then compare them against U.S. Treasuries or USD swaps.

Markets:

- U.S. Treasuries
- USD swaps
- Gilts
- Bunds
- JGBs
- Canadian bonds
- Australian bonds
- cross-currency basis swaps
- FX forwards
- global inflation-linked bonds where applicable

Core research questions:

- Is a foreign government bond cheap or rich versus the cash-flow-equivalent U.S. Treasury?
- Are hedged yields dislocated versus USD yields?
- Are cross-currency basis markets creating relative value opportunities?
- Are global curves pricing inconsistent macro outcomes?
- Does FX hedging fully explain the yield pickup?
- Are there structural balance sheet, collateral, or funding reasons for the spread?

Potential features:

- local yields
- FX spot
- FX forwards
- cross-currency basis
- USD swap curve
- local swap curves
- inflation differentials
- central bank pricing
- sovereign spreads
- term premium proxies
- global risk sentiment
- hedging cost
- carry and rolldown

Example signal:

```json
{
  "strategy_family": "cross_market_rv",
  "signal_name": "gbp_gilt_usd_hedged_vs_ust",
  "instrument": "10Y gilt hedged to USD versus 10Y UST",
  "direction": "long_hedged_gilt_short_ust",
  "horizon": "1m_6m",
  "conviction": 0.66,
  "model_score": 1.7,
  "rationale": "USD-hedged gilt yield is cheap versus duration-matched Treasury after accounting for FX forwards, cross-currency basis, and curve beta."
}
```

Implementation point:

Cross-market trades need unusually careful construction.

Each trade should track:

- local yield
- USD-hedged yield
- FX forward points
- cross-currency basis
- duration hedge ratio
- curve hedge ratio
- carry
- rolldown
- funding assumption
- liquidity assumption
- transaction cost estimate

The credibility here comes from showing the actual conversion math, not just saying one bond is cheap versus another.

---

## 6.6 Event-Driven Macro

Event-driven strategies focus on market behavior around known macro catalysts and supply events.

Events:

- CPI
- PCE
- payrolls
- unemployment claims
- ISM
- retail sales
- FOMC
- Fed speakers
- Treasury refunding
- Treasury auctions
- elections
- geopolitical shocks
- major fiscal announcements

Trade types:

- pre-event positioning
- post-event continuation
- post-event mean reversion
- auction tail/follow-through
- intraday futures momentum
- intraday futures reversal
- vol event premium trades
- curve reaction trades

Core research questions:

- Does the market systematically overreact or underreact to certain data?
- Are there reliable intraday patterns around auctions?
- Do surprise components predict follow-through in rates, curve, vol, or breakevens?
- Does price action after the first few minutes contain information?
- Are event risk premia overpriced or underpriced?
- Which events matter most by regime?

Potential features:

- event type
- consensus
- actual
- surprise
- prior revision
- market-implied expectation
- pre-event positioning
- pre-event realized vol
- implied event vol
- first 1m/5m/15m reaction
- order flow
- futures volume
- bid/ask spread
- auction tail
- bid-to-cover
- dealer takedown
- indirect/direct bidding

Example signal:

```json
{
  "strategy_family": "event_driven_macro",
  "signal_name": "post_cpi_10y_futures_continuation",
  "instrument": "10Y Treasury futures",
  "direction": "short_duration",
  "horizon": "intraday_3d",
  "conviction": 0.61,
  "model_score": 1.4,
  "rationale": "Hot CPI surprise with strong first-15-minute bearish price action historically shows continuation in this regime."
}
```

Implementation point:

Event-driven models should separate:

- pre-event signal
- event reaction signal
- post-event follow-through signal
- event-vol signal

These are different trades and should not be mixed together.

---

# 7. Core System Design

## 7.1 Signal Lifecycle

Every signal follows the same lifecycle.

```text
research idea
  -> feature build
  -> model output
  -> trade expression
  -> risk sizing
  -> timestamped signal log
  -> model portfolio update
  -> live performance tracking
  -> review
```

Each signal must be logged before it can be evaluated.

No log, no claim.

---

## 7.2 Signal Record

Every signal should produce a structured JSON record.

Minimum fields:

```json
{
  "signal_id": "unique_id",
  "timestamp_utc": "2026-04-30T14:30:00Z",
  "as_of_date": "2026-04-30",
  "strategy_family": "curve",
  "strategy_name": "beta_weighted_5s30s_residual",
  "model_version": "v0.1.0",
  "model_hash": "abc123",
  "data_hash": "def456",
  "instrument": "5s30s Treasury curve",
  "trade_expression": "long 30Y duration / short 5Y duration beta weighted",
  "direction": "steepener",
  "horizon": "1m_3m",
  "entry_level": "observed_level_here",
  "signal_value": 2.1,
  "conviction": 0.68,
  "target_position": 1.0,
  "risk_unit": "standardized_strategy_risk",
  "stop": "defined_before_trade",
  "take_profit": "defined_before_trade",
  "rationale": "plain English explanation",
  "status": "open"
}
```

The signal record should be boring, structured, and consistent.

That is what makes it auditable.

---

## 7.3 Timestamping

The system should prove that each signal existed before the outcome.

Initial version:

- write daily signal file
- commit to GitHub
- use signed commits
- never rewrite old signal history

Stronger version:

- hash each signal file
- timestamp the hash using OpenTimestamps or similar
- store timestamp proof next to signal file
- expose proof links in dashboard

The rule:

> Once a signal is logged, it is permanent.

If the model changes its mind, log a new signal update. Do not edit the old one.

---

## 7.4 Model Versioning

Every signal should know which model created it.

Track:

- model name
- model version
- git commit hash
- feature set
- training window
- data as-of timestamp
- model parameters where appropriate
- human override flag

If the model changes, the version changes.

This prevents hidden strategy drift.

---

# 8. Performance Tracking

## 8.1 Two Separate Books

The system should track two separate books.

## Forecast Book

The forecast book tracks whether the macro call was right.

Example:

- signal said 10Y yields should fall
- entry 4.60%
- 1m outcome 4.42%
- call was correct

This proves market judgment.

## Strategy Book

The strategy book tracks actual tradable P&L.

Example:

- long TY futures
- sized at 1 risk unit
- included transaction costs
- exited after stop/take-profit/model flip
- realized +1.3R

This proves trading process.

Both matter.

---

## 8.2 Standard Performance Metrics

Track performance overall and by strategy family.

Metrics:

- total return
- annualized return
- annualized volatility
- Sharpe ratio
- Sortino ratio
- max drawdown
- Calmar ratio
- hit rate
- average win
- average loss
- win/loss ratio
- profit factor
- average holding period
- turnover
- transaction costs
- slippage
- exposure by strategy
- exposure by asset class
- exposure by risk factor

---

## 8.3 Macro-Specific Metrics

Also track metrics that actually matter for macro signals.

Forecast metrics:

- direction accuracy by horizon
- average move after signal
- information coefficient
- signal decay
- hit rate by regime
- hit rate by event type
- hit rate by asset class
- performance by conviction bucket
- performance by z-score bucket

Risk metrics:

- drawdown by strategy
- duration exposure
- curve exposure
- inflation exposure
- vol exposure
- cross-market exposure
- event exposure
- correlation between strategies
- crowding proxy where possible

---

# 9. Dashboard

The public dashboard should be simple and credible.

Core pages:

## Home

- project description
- live NAV
- total return
- Sharpe
- max drawdown
- current open signals
- latest monthly review

## Signal Ledger

A table of every signal ever generated:

- timestamp
- strategy family
- signal name
- instrument
- direction
- horizon
- conviction
- entry level
- current/exit level
- result
- status
- proof hash

## Strategy Performance

Performance by strategy family:

- directional duration
- curve
- rates vol
- inflation
- cross-market RV
- event-driven macro

## Forecast Accuracy

Evaluate calls independent of execution:

- 1d accuracy
- 1w accuracy
- 1m accuracy
- 3m accuracy
- by asset class
- by conviction bucket
- by regime

## Trade Blotter

For actual traded/model trades:

- entry date
- exit date
- instrument
- expression
- size
- entry
- exit
- P&L
- R multiple
- notes

## Research Notes

Monthly writeups:

- what worked
- what failed
- current themes
- model changes
- risk review
- lessons learned

---

# 10. Repo Structure

```text
macro-signal-ledger/
  README.md
  methodology.md
  signal_policy.md
  risk_policy.md
  data_policy.md

  data/
    raw/
    processed/
    external/

  signals/
    2026/
      2026-04-30.json
      2026-05-01.json

  proofs/
    2026/
      2026-04-30.ots
      2026-05-01.ots

  portfolios/
    forecast_book.parquet
    strategy_book.parquet
    positions.parquet
    nav.parquet

  reports/
    daily/
    monthly/

  src/
    config.py
    data/
      load_market_data.py
      build_features.py
    strategies/
      directional_duration.py
      curve.py
      rates_vol.py
      inflation.py
      cross_market_rv.py
      event_driven_macro.py
    portfolio/
      construct_positions.py
      calculate_pnl.py
      calculate_risk.py
      attribution.py
    ledger/
      write_signal.py
      hash_signal.py
      timestamp_signal.py
      validate_signal.py
    reporting/
      build_tearsheet.py
      build_monthly_review.py

  dashboard/
    app.py
    pages/
      1_signal_ledger.py
      2_strategy_performance.py
      3_forecast_accuracy.py
      4_trade_blotter.py
      5_research_notes.py
```

---

# 11. MVP Build Plan

## Phase 1: Build the Ledger

Goal:

Create a system that logs daily signals in a structured, timestamped format.

Deliverables:

- strategy config file
- signal schema
- daily signal generator
- JSON signal writer
- GitHub auto-commit script
- basic signal validation

Success criteria:

- every signal is saved
- every signal has a timestamp
- every signal has a model version
- old signals are not edited

---

## Phase 2: Build the First Strategy Book

Goal:

Turn signals into a standardized model portfolio.

Start with the simplest strategies:

1. directional duration
2. curve
3. inflation

Deliverables:

- position construction logic
- daily P&L calculation
- transaction cost assumptions
- NAV series
- drawdown series
- basic tearsheet

Success criteria:

- every signal maps to a trade expression
- every trade has standardized risk
- P&L updates daily
- performance can be reviewed honestly

---

## Phase 3: Build the Dashboard

Goal:

Create a public-facing dashboard that shows live performance and the full signal history.

Deliverables:

- Streamlit dashboard
- signal ledger page
- NAV page
- strategy performance page
- forecast accuracy page
- research notes page

Success criteria:

- someone can understand the project in 90 seconds
- every signal is visible
- losing signals are visible
- performance is easy to audit

---

## Phase 4: Add Broker Reconciliation

Goal:

Connect real trades to the model book.

Deliverables:

- broker statement importer
- fill parser
- position reconciliation
- actual-vs-model P&L comparison
- trade slippage analysis

Success criteria:

- live trades can be matched to model signals
- actual performance can be compared to theoretical performance
- execution quality can be measured

---

## Phase 5: Expand Strategy Families

Goal:

Add more complex strategy families after the core framework works.

Add in this order:

1. rates vol
2. cross-market RV
3. event-driven macro

Reason:

These are more data-intensive and implementation-sensitive. The ledger and tracking infrastructure should be solid before adding them.

---

# 12. First Three Strategies to Build

Do not start with everything.

Start with three strategies that are clean, explainable, and close to the core macro skillset.

## 1. 10Y Duration Fair Value

Question:

Is the 10Y yield rich or cheap versus macro fundamentals?

Target:

- 10Y Treasury yield
- TY futures equivalent
- cash Treasury equivalent

Features:

- Fed path
- real yields
- breakevens
- oil
- DXY
- equities
- credit spreads
- realized vol

Output:

- long duration
- short duration
- neutral

---

## 2. Beta-Weighted 5s30s Curve

Question:

Is the long-end curve too steep or too flat versus macro fundamentals?

Target:

- 5s30s curve
- beta-weighted 5s30s
- DV01-neutral 5s30s

Features:

- Fed path
- inflation
- term premium proxy
- oil
- DXY
- risk sentiment
- Treasury supply
- curve momentum

Output:

- steepener
- flattener
- neutral

---

## 3. 10Y Breakeven Inflation Fair Value

Question:

Are 10Y breakevens rich or cheap versus inflation fundamentals?

Target:

- 10Y breakeven
- TIPS/nominal pair
- inflation swap where available

Features:

- oil
- gasoline
- dollar
- commodities
- CPI trend
- real yields
- Fed path
- risk sentiment

Output:

- long breakeven
- short breakeven
- neutral

---

# 13. Rules of the Game

These rules matter.

## No hindsight

Signals must be logged before outcomes are known.

## No deleting losers

Bad signals stay in the ledger.

## No silent edits

If a model changes, bump the version.

## No fake precision

Confidence scores should be calibrated over time.

## No hidden leverage

Every trade should report risk exposure.

## No untracked overrides

Human overrides are allowed, but they must be flagged.

## No claiming live performance from backtests

Backtest, paper, model, and live results must be clearly separated.

---

# 14. Career Framing

This project should tell a clear story.

> I built a live macro signal ledger that converts systematic rates and macro research into timestamped, auditable trading signals. The platform tracks directional duration, curve, inflation, rates volatility, cross-market relative value, and event-driven macro strategies. Every signal is logged ex-ante, versioned, evaluated out of sample, and tied to a model portfolio or live trade record.

The point is not just to show good returns.

The point is to show:

- investment process
- market intuition
- systematic research ability
- risk discipline
- honesty about drawdowns
- ability to convert macro views into tradeable expressions

That is the bridge from execution trading to a research-driven risk-taking seat.

---

# 15. The North Star

The end state is a live, auditable macro trading research platform.

A hiring manager, PM, or allocator should be able to look at it and say:

> This person can generate macro ideas, structure trades, size risk, track results, learn from mistakes, and run a disciplined process.

That is the proof.

Not a backtest.

Not a spreadsheet.

Not a story.

A live, timestamped, honest track record.
