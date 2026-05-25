# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

## 5. Python File Structure

**Every research/analysis file follows this layout:**

1. **Module docstring** — one paragraph describing what pattern/analysis this file implements
2. **Imports + path setup**
3. **Config constants** — DSN, START date, ticker dicts, etc.
4. **Helper functions** — each does ONE focused step (load data, one diagnostic, one plot). Return data; don't print or display.
5. **`main()`** — the story. Each block is a terse one-line comment + a few lines that produce named variables. Reading `main()` top-to-bottom tells you exactly what work has been done and what's available to chain off of.
6. **`if __name__ == "__main__":` at the bottom.**

`main()` returns a dict of all interesting variables so an interactive caller can do `state = main()` and inspect `state["resid_10_2"]` etc.

**Avoid:**
- `# %%` cell markers — those are for notebooks
- Free-floating top-level code outside helpers/`main()`
- Helpers that print or display — side effects belong in `main()` so helpers stay composable

---

## 6. Use the Project Libraries

**Before writing any utility code, check `utils/` and `stats/` first. This codebase has substantial shared infrastructure — use it.**

### `utils/helpers.py`
General-purpose helpers:
- `query_db(sql, params?)` — run a SQL query, returns `pd.DataFrame`. Manages connection lifecycle. Use this instead of raw psycopg.
- `query_df(conn, sql)` — same but on an existing connection.
- `to_pl_series(s)` / `to_pl_df(df)` — convert pandas ↔ polars.
- `fix_outliers(expr, hi, lo)` — replace out-of-range values with interpolation.
- `DB_DSN` — the canonical DB connection string. Don't redefine it.

### `utils/rates.py`
Rate construction helpers:
- `synthetic_real_rate(nominal, inflation)` — nominal minus inflation swap.
- `linear_5y5y_forward(r5, r10)` — trader shortcut: `2*10y - 5y`.
- `with_synthetic_real_rates(df)` — adds real rate columns to a DataFrame in one call.

### `utils/viz.py`
Visualization — `Viz` class wraps Plotly/matplotlib for rates-style charts:
- `viz.line(df, ...)` — time series line chart.
- `viz.table(df, ...)` — formatted table display.
- `viz.rolling_corr(df, col1, col2, window)` — rolling correlation chart.

### `stats/` (imported as `from stats import ...`)
Quantitative building blocks — all Polars-native, accept pandas or polars:
- `roll_lr(x, y, lookback)` — vectorized rolling OLS. Returns DataFrame with `alpha, beta, yhat, resid, r2`.
- `roll_lr_diff(x, y, lookback)` — changes-based rolling OLS (correct for beta-weighting).
- `roll_beta(x, y, lookback)` / `roll_resid(x, y, lookback)` — convenience wrappers.
- `half_life(series)` — mean-reversion half-life via AR(1).
- `ou_params(series)` — OU theta, mu, sigma, half_life.
- `ou_zscore(series, lookback?)` — z-score vs OU equilibrium.
- `roll_half_life(series, lookback)` — rolling half-life (vectorized).
- `ou_summary(series, lookback?)` — single-row summary DataFrame.
- `fit_pca(df, n_components)` / `roll_pca(df, lookback)` — PCA fit and rolling PCA.
- `reconstruct(result)` / `residual_from_pca(result, n_components)` — PCA reconstruction and residuals.
- `explain(result)` — variance explained summary table.

**The principle:** if you need a rolling regression, OU z-score, PCA, DB query, or rate construction — it's already here. Don't reimplement it.

If new functionality needs to be built and it's genuinely reusable (not one-off analysis logic), add it to the appropriate existing file in `stats/` or `utils/` rather than defining it inline or in a script. The test: would this be useful in more than one place? If yes, centralize it. `stats/` is for quantitative/statistical primitives; `utils/` is for data access, formatting, rate construction, and visualization.

---

## 7. Environment

**Always use the `2s10s` conda env** — it has all package dependencies (psycopg, polars, pandas, statsmodels, plotly, etc.).

- Env location: `C:\Users\benjils\miniforge3\envs\2s10s\`
- To run via PowerShell: `mamba run -n 2s10s python script.py`

---

## Project Context

**Views** is a Python macro research framework for systematic rates and macro RV analysis. It pulls from a local PostgreSQL database (host: `raptor`, db: `markets`) and is the research engine behind a larger project called the **Macro Signal Ledger**.

### What we're building

A live, timestamped, auditable macro trading platform. The goal is to prove ex-ante investment judgment — every signal is generated live, logged before the outcome is known, versioned, and evaluated honestly. No hindsight, no deleting losers, no silent edits.

**Six strategy families:**
1. Directional duration — is 10Y rich or cheap vs macro fundamentals?
2. Curve — is 5s30s too steep/flat? (beta-weighted, residual construction)
3. Inflation — are breakevens rich or cheap vs oil, dollar, CPI trend?
4. Rates volatility — is implied vol compensated vs realized?
5. Cross-market RV — hedged foreign bonds vs USTs
6. Event-driven macro — CPI, FOMC, NFP, auction reactions

**MVP phases:** ledger → strategy book → public dashboard → broker reconciliation → expanded strategies.

### Current research thread

Studying the direction → curve interaction: does the level of 10Y have predictive power for 10s30s slope? Key insight: naive 5s30s carries hidden direction; regress curve on the short leg, trade the residual for clean curve exposure. Swap spreads, mortgage convexity (MOVE), and vol regime all contaminate the signal and need to be stripped out.

### Architecture notes
- Signals computed on Bloomberg generics (`md.index_eod`), PnL tracked on OTR bonds (`md.headline`)
- `signal_fn` returns `pl.DataFrame` with columns: `signal`, `time_stop`, `size`, `confidence`, `vol`
- Scale yields ×100 at load time so all PnL is natively in bps
- See `PLAN.md` for full strategy specs and signal schema, `notes.md` for factor decomposition
