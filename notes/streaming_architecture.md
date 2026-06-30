# Real-Time Streaming Architecture

## Goal

Stream Bloomberg live data intraday → update signals and residuals in real time → trigger mock (and eventually live) trades when a signal fires.

## Core Principle: Signal Engine is In-Memory, Not DB-Centric

Postgres is **not** in the hot path for signal computation. It only gets written to when something meaningful happens: signal fires, mock trade logged, EOD snapshot taken.

## Data Flow

```
Bloomberg BLPAPI subscription (real-time push)
  ↓
Tick handler (Python process, in-memory state)
  ↓  recompute signal on each new tick
Signal engine (stats/ functions on rolling window)
  ↓  if threshold crossed
Trade event → Postgres (audit log) + LISTEN/NOTIFY → UI
```

## Key Components

**Bloomberg BLPAPI subscription**
- Official Python SDK supports subscription mode: give it a list of tickers, a callback fires on every price update.
- Different API mode from the request/response used in `data_pull/` scripts, but same license.

**In-memory state object**
- Holds the rolling window for each signal.
- On each tick: append new value, drop oldest, recompute signal.
- For rates macro (not HFT), recomputing from scratch on each tick is fine — Bloomberg intraday updates on 10Y/30Y yields are not microsecond.
- Current `stats/` library (roll_lr, ou_zscore, etc.) is batch-oriented; that's OK — just re-run on the updated window.

**Postgres (audit log only)**
- Write on signal fire: timestamp, signal value, ticker, size, confidence.
- Write on mock trade: entry price, signal that triggered, stop, target.
- LISTEN/NOTIFY: trigger on insert fires `pg_notify('signal_fired', payload)` → connected UI gets pushed the event immediately.

## What We Don't Need (Yet)

- **Redis / message queue**: unnecessary until multiple independent signal processes need to consume the same feed. The Bloomberg subscription callback *is* the stream.
- **Streaming DB (TimescaleDB, QuestDB, etc.)**: overkill until tick volume justifies it. Postgres handles intraday signal logging fine.
- **Tick storage**: don't write every Bloomberg tick to Postgres — store only the signal events and trade log.

## Build Order

1. Get BLPAPI subscription working for key tickers (10Y, 30Y, etc.)
2. Build in-memory state object: rolling window + signal recompute on each tick
3. Write to Postgres only on signal fire / mock trade log
4. Add LISTEN/NOTIFY trigger → push fired signals to UI
5. Build mock trade execution layer (log entry/exit, track PnL vs signal)
