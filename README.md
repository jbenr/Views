# Views

<p align="center">
  <img src="assets/cover.webp" width="700">
</p>

## Overview

**Views** is a Python research framework for **systematic macro rates / relative-value analysis**. It is the research engine behind the **Macro Signal Ledger** — a live, timestamped, auditable macro trading platform (the full vision, strategy specs, and signal schema live in [`notes/PLAN.md`](notes/PLAN.md); the working backlog in [`notes/TODO.md`](notes/TODO.md)).

The core research loop:

1. Pick a target (10Y yield, curve slope, breakeven, vol point).
2. Model it against explanatory anchors (rolling OLS, PCA) → the **residual** is the dislocation.
3. Diagnose the residual (half-life, Hurst, OU z-score, horizon IC/hit/Sharpe).
4. Condition on regime, wrap it in a `SignalPipeline`, backtest it through the shared engine.
5. (Eventually) log it ex-ante to the ledger and execute it.

## Repository map

| Directory | Status | What it is |
|---|---|---|
| `stats/` | production | Quant primitives: rolling OLS (`ols.py`), OU/half-life/Hurst (`ou.py`), PCA (`pca.py`), residual diagnostics (`diagnostics.py`) |
| `research/` | production | Reusable alpha-discovery methods: conditional dislocations, pair/PC relative value, and multi-factor fair value. These research studies generate evidence and candidates; they do not replace the exact backtest. |
| `utils/` | production | DB access (`helpers.py`), market-data loaders (`market_data.py`), rate construction (`rates.py`), ticker universes (`tickers.py`), formatting, `Viz` plotting (`viz.py`), Dash helpers (`research_app.py`), DB browser (`peep.py`) |
| `backtest/` | production | The single shared backtest/signal engine: `Engine`, `SignalPipeline`, `TradeDef`, sizing, metrics, parameter scans, selection diagnostics, cross-sectional `SpreadBook` |
| `book/` | production | Strategy families: `duration/`, `curve/`, `inflation/`, `rate_vol/`, `cross_market_rv/`, `event_driven/`. Each has a `strategy.py` (or `pipeline.py`) with the standard strategy layout; research scripts live alongside. (Named `strategy.py`, not `signal.py` — that shadows the stdlib `signal` module) |
| `execution/` | boilerplate | Interactive Brokers adapter — dry-run only, see [`execution/README.md`](execution/README.md) |
| `data_pull/` | ingestion | Bloomberg → PostgreSQL ingestion scripts. Run on the data box; nothing in research imports from here except `berg.py` for live Bloomberg work |
| `dig/` | exploratory | Research scratch space. Extract reusable pieces into `stats/`/`utils/` when they prove out; do not treat as production |
| `tests/` | production | Synthetic-data tests — no DB or Bloomberg required |
| `notes/` | docs | `PLAN.md` (the north star), `TODO.md`, `notes.md` (factor decomposition) |

Root-level `main.py`, `_db_inspect.py`, `_gap_check.py` are scratch scripts.

## Setup

```bash
# 1. create an env with the scientific stack (or use the existing one)
conda create -n views python=3.12 numpy pandas polars psycopg tabulate
conda activate views

# 2. install the repo as an editable package (this is what makes
#    `from stats import ...` work everywhere — no sys.path hacks)
pip install -e ".[dev]"

# 3. verify
pytest
```

Research extras (plotting, dashboards, stats): `matplotlib plotly dash statsmodels tqdm`.

## Data access

Market data lives in a local PostgreSQL database (`markets` on host `raptor`), populated by `data_pull/`. Override the connection string with the `DB_DSN` environment variable (see `utils/helpers.py`).

Key tables:

- `md.index_eod` — Bloomberg generics (constant-maturity yields, equities, commodities, FX, MOVE). **Research/signals run on these.**
- `md.headline` — on-the-run Treasuries. **Live P&L tracks these.**
- `md.breakeven`, `md.swaption_vol`, futures/CFTC tables — see `data_pull/`.

Load data through the shared loader — never hand-roll the pivot:

```python
from utils.market_data import load_wide

TICKERS = {"10y": "USGG10YR Index", "30y": "USGG30YR Index", "oil": "CO1 Comdty"}
df = load_wide(TICKERS, start="2010-01-01", bps_cols=["10y", "30y"])
# → polars frame: ts | 10y | 30y | oil, yields scaled ×100 so PnL is in bps
```

`bps_cols` scales yields ×100 at load time — the repo convention is that all yield math downstream is natively in bps. `pick_ticker()` resolves the best-covered ticker among candidates; `to_pandas=True` gives an indexed pandas frame. Ticker universes for ingestion live in `utils/tickers.py`.

## How research files are structured

Every research script follows the same layout (see `CLAUDE.md` §5): module docstring → imports → config constants (`START`, `TICKERS`, lookbacks) → small composable helpers that *return* data → a `main()` that reads top-to-bottom as the story and returns a dict of interesting state → `if __name__ == "__main__":`. Good examples: `book/curve/research.py`, `book/duration/spread_rv.py`.

The building blocks are already written — check before reimplementing:

```python
from stats import (
    roll_lr, roll_lr_diff,          # rolling OLS (levels / changes — changes for hedge ratios)
    ou_zscore, roll_ou_zscore,      # z-score vs OU equilibrium
    half_life, roll_half_life,      # mean-reversion speed → time stops
    hurst_exponent, roll_hurst,     # memory structure: <0.5 mean-reverting
    fit_pca, roll_pca, residual_from_pca,
    horizon_backtest,               # fade-the-residual IC / hit / Sharpe by horizon
    beta_cv, quality_weight,        # model stability gates
)
```

## Building a new strategy

**Start from the template: [`book/rate_vol/template.py`](book/rate_vol/template.py).** It is a complete, runnable walkthrough on synthetic data:

```bash
python -m book.rate_vol.template
```

## Research methods

Use the research modules to investigate an alpha hypothesis before turning it
into a strategy/backtest candidate. They accept column names from any wide
panel, so a method is not tied to curves, duration, inflation, or basis:

```python
from research import DislocationStudy, PairRVStudy, FairValueStudy

# 1. Conditional overreaction: unusual 10s30s move given 10Y and MOVE.
shock = DislocationStudy("10s30s", ("10y", "move"))
shock_evidence = shock.research(data)

# 2. Literal relative-value package: one leg versus another.
rv = PairRVStudy("30y", "10y", weighting="beta")
rv_evidence = rv.research(data)

# 3. Declared economic fair value; `.search()` explores a controlled
# factor-family library with at most one selected input from each family.
fair = FairValueStudy("10s30s", ("term_premium", "move", "supply"))
fair_evidence = fair.research(data)
```

Each study returns its signal frame plus the diagnostics appropriate to its
research type. A promising result is still only a candidate: freeze its rules,
then take that candidate to the exact backtest and walk-forward process.

Every curve strategy module is configuration on the shared template
(`backtest/strategy.py`), which provides the signal construction, all four
research modes, the CLI, and the artifact files:

```python
from backtest.strategy import Strategy, synthetic_pair

STRATEGY = Strategy(
    name="real10y_2s10s",
    module="book.curve.real10y_2s10s",   # sweep workers import this
    path=Path(__file__),                   # funnel parquets: data/<name>/
    tickers={"real10y": "USGGT10Y Index", "2s10s": "USYC2Y10 Index"},
    bps_cols=["real10y"],
    target="2s10s", feature="real10y",
    synthetic_fn=...,                      # no-DB substitute for tests/dev
    feature_fn=...,                        # optional derived features (e.g. PC1)
)
compute, make_pipeline, pipeline = STRATEGY.compute, STRATEGY.make_pipeline, STRATEGY.pipeline

if __name__ == "__main__":
    state = STRATEGY.cli()
```

Every Strategy grid (predict/exit/sweep search spaces, costs, params) is a
constructor field — override per strategy as needed. Strategies with a custom
`compute` (non-residual-fade families) can still build a raw `SignalPipeline`
against the engine directly.

**Every strategy module must also be scriptable**, with a `main(use_db: bool = True) -> dict` that prints the four standard blocks and returns its state for interactive chaining:

1. data line — rows, date span, columns, source (`db` / `synthetic`)
2. residual horizon backtest table (IC / hit / Sharpe at 5/20/60d)
3. engine `BACKTEST SUMMARY`
4. latest signal line — ts, z, resid, beta, r2, action

```bash
python -m book.curve.tens_10s30s              # live DB data
python -m book.curve.tens_10s30s --synthetic  # no DB needed
```

`book/curve/tens_10s30s.py` (the direction→curve research thread: 10Y vs 10s30s) is the live example of this pattern; `book/curve/real10y_2s10s.py` and `book/curve/pc1_10s30s.py` are graduates of the cross-pair explorer (`book/curve/xy_scan.py`, funnel step 0).

Two conventions inside a strategy module:

- **Identity config at the top** (family, name, `TICKERS`, target/feature) — what the strategy *is*.
- **Tuning goes through the Strategy fields** (`default_params`, the grid lists) — how it's *tuned*. `main(params={...})` overrides ad hoc; `make_pipeline(params)` is the contract the parameter lab uses to sweep it.

## Backtesting

The engine consumes pipelines and wide data:

```python
from backtest import Engine, BacktestConfig, print_summary
from book.curve.tens_10s30s import pipeline, TICKERS
from utils.market_data import load_wide

data = load_wide(TICKERS, start="2010-01-01", bps_cols=["10y"])
result = Engine(BacktestConfig(transaction_cost_bps=0.5)).add_signal(pipeline).run(data)
print_summary(result)          # PnL, hit rate, Sharpe, drawdown, holding period
result.summary()               # same as a dict
```

Signals are computed vectorially (polars); position management (entries, signal/stop/time/trailing exits, custom `exit_fn`/`entry_filter_fn`) is a row-by-row state machine. The parameter lab in `backtest.lab` supplies vectorized and exact grid searches; `backtest.validation` supplies selection-aware Sharpe and overfitting diagnostics.

Two conventions keep backtests honest: signals on generics, P&L on OTRs; and no lookahead — every rolling stat in `stats/` uses trailing windows only.

## The parameter lab (`backtest/lab.py`)

The scalable search-and-select layer. The funnel, cheapest to most exact:

```
signal_matrix + fast_scan   →   sweep_strategy   →   MetricStore   →   gate_scan
(vectorized coarse scan,        (exact engine,       (parquet run       (conditional
 GPU-ready via cupy)             all CPU cores)       history)           edge buckets)
```

1. **`fast_scan`** — approximate threshold backtest of an entire signal matrix in one shot, **with gates as a scan dimension**: pass `gates=` (the condition matrices from `signal_matrix(..., return_conditions=True)`: r2, beta_cv, abs_beta, resid_vol20, resid_mom10) and every combo is also evaluated once per (condition × quantile bucket) entry gate — the grid becomes `K × entries × (1 + gates × buckets)`. Gate buckets are causal expanding percentiles with a configurable history warmup, so future observations never alter earlier entry decisions. `add_gate_lift()` then scores each gate against its own ungated baseline. Pure array math (custom CUDA kernels for the scans), so on the NVIDIA tower it runs on GPU: `pip install -e ".[gpu]"`, then `device="gpu"`. Approximations: hysteresis exits, no stops, next-bar fills, and per-column relative percentile labels.
2. **`sweep_strategy("book.curve.tens_10s30s", data, grid)`** — the exact row-by-row Engine per combo, parallel across every CPU core (32-core box → 32 combos at a time). Real stops, time-stops, costs, trade logs.
3. **`MetricStore`** — every run appends to `store/backtests.parquet` (git-ignored). `leaderboard()` ranks across strategies and time; `matrix(x="beta_lb", y="z_lb", metric="sharpe", agg="max")` pivots any metric across any two parameter dimensions.
4. **`gate_scan`** — the highest-confidence-setup layer: bucket candidate state variables (r2, beta_cv, residual vol, regime flags) at hypothetical entry bars and measure per-bucket hit/PnL lift vs the unconditional baseline. Buckets that concentrate the edge graduate into `SignalConfig.entry_filter_fn` gates.

All four are wired into the strategy modules as scriptable modes:

```bash
python -m book.curve.tens_10s30s --sweep    # exact grid → leaderboard + matrix + store
python -m book.curve.tens_10s30s --fast     # gated coarse scan, ~200k evals (--gpu on the tower)
python -m book.curve.tens_10s30s --gates    # conditional edge table
```

Guardrails: a Sharpe that only exists in one grid cell is an overfit candidate, not a finding. The strategy funnel requires at least eight non-overlapping forecast windows before a setup advances. Its exact sweep saves `*_validation.parquet` with the Probabilistic/Deflated Sharpe Ratio and CSCV Probability of Backtest Overfitting for the exact finalists. Those statistics are deliberately labeled `exact_finalists_only`: they do not account for candidates discarded during `--predict` or `--exit`, so they are lower bounds on the full research process's selection penalty. Treat `fast_scan` output as a shortlist generator only, retain every attempted path for full-funnel DSR/PBO, and require a purged, embargoed walk-forward rerun of the entire selection funnel before promotion.

## Ledger and execution (the intended live path)

Per `notes/PLAN.md`: each strategy's latest `compute()` row becomes a structured, timestamped JSON signal record (schema in PLAN §7.2), committed **before the outcome is known** — no edits, no deletions, model version on every record. Signals then map to standardized model-portfolio positions (forecast book vs strategy book), and finally to broker orders.

The last hop is stubbed in `execution/`: a `TargetPosition` → `OrderPreview` translation with a dry-run-only `IBKRExecutor`. Live submission is deliberately unimplemented and double-gated — read [`execution/README.md`](execution/README.md) before touching it.

## Tests

```bash
pytest            # synthetic data only — no DB, no Bloomberg
```

Covers the stats primitives, the market-data transformation, the backtest engine, the strategy template end-to-end, and the execution safety gates.
