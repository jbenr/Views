# dashboard

Live monitoring for promoted signals: which setups are "live," what they
last read, and when they were last checked. Works for any `book/*` module
that exposes a `STRATEGY = backtest.strategy.Strategy(...)` object -- not
curve-specific.

Static by design: the page shows whatever was last pulled/analyzed. Nothing
recomputes on its own. Two buttons per signal drive everything.

## Workflow

```
# 1. run a strategy's full funnel if you haven't already
python -m book.curve.tens_10s30s --cook

# 2. promote a winner -- snapshots one sweep_results row (default: rank 0,
#    best by sharpe, the file's own sort order) as the frozen live config
python -m dashboard.registry --promote book.curve.tens_10s30s
python -m dashboard.registry --promote book.curve.tens_10s30s --rank 2  # a different row
# Or promote a module's deliberately curated default_params:
python -m dashboard.registry --promote book.curve.twos_10s30s --defaults
# Named variants coexist with the base signal instead of replacing it:
python -m dashboard.registry --promote book.curve.twos_10s30s --variant ou430_e09
python -m dashboard.registry --promote book.curve.twos_10s30s --variant ou490_e07
python -m dashboard.registry --list
python -m dashboard.registry --remove book.curve.twos_10s30s::ou430_e09

# 3. open the dashboard
mamba run -n 2s10s python -m dashboard.app
# http://127.0.0.1:8052
```

## What the two buttons do

| Button | Action | Ledger write? |
|---|---|---|
| **Re-pull data** | `Strategy.load_data()` fresh from the DB, cached to `store/live_data/<name>.parquet` | No -- just refreshes the inputs and the charts |
| **Re-run analysis** | `Strategy.compute()` on the cached data with the promoted params, checked against `entry_threshold` (and the gate, if any) | Yes -- one timestamped row appended to `store/signal_ledger.parquet` |

Re-pulling and re-running are deliberately separate: you can refresh the
displayed data as often as you like without polluting the audit trail;
"re-run analysis" is the auditable event -- what the signal read, logged
before any outcome is known.

On first load (or after "re-pull data"), the card's charts always reflect
`Strategy.compute()` on whatever data is currently cached -- that computation
itself is never logged, only an explicit "re-run analysis" click is.

## Storage (`store/`)

- `live_signals.parquet` -- the registry: one row per promoted base signal or
  named variant, with one immutable `signal_id`, its frozen
  params (`entry_signal`, `beta_lb`, `ou_lb`, `entry_threshold`,
  `exit_style`, `exit_param`, `gate`, `gate_window`, `z_gate`,
  `stop_loss_bps`) plus the backtest metrics it was promoted on.
- `signal_ledger.parquet` -- append-only: one row per "re-run analysis"
  click (`run_ts`, `data_asof_ts`, `level`, `signal`, `resid`, `ou_z`,
  `beta`, `r2`, `half_life`, `gate_value`, `gate_percentile`,
  `gate_allow`, `fired`).
- `live_data/<name>.parquet` -- the last data pulled per signal, so the
  dashboard can render without hitting the DB on every page load. Named
  variants from the same strategy module share this market-data cache while
  retaining independent parameters, metrics, ledger history, and controls.

All three follow `backtest.lab.MetricStore`'s atomic-write pattern
(write to a temp file, `os.replace`).

## Dashboard layout

The **Live Overview** tab is the trading surface. It groups signals into one
table per traded target and shows the exact replayed position state, current
threshold reading, causal gate status, current signal versus its entry level,
net PnL, frozen backtest statistics, and data timestamp. Use **Refresh live
status** to re-pull each promoted module's data from the DB (deduped, so
named variants of the same module only pull once) and recompute those rows.

The **Signal Deep Dive** tab has a signal selector and renders the complete
research card for just that strategy:

Name / pair / module, a stat row (data as-of, last analysis run, current
reading, gate state, and an explanatory param summary), then two charts side
by side: the tradable level and the entry signal (residual or OU z-score)
with `±entry_threshold` bands and the current reading flagged. Gated signals
also show a compact causal-percentile chart identifying the gate rule and
whether it is currently open or closed. The fourth chart is the exact
engine's cumulative daily marked-to-market PnL, including open-position
movements rather than only realized trade exits, rebased to zero at the
left edge of the selected chart window. The chart-window controls sit
immediately above the chart grid. "Snap chart to view" includes five
business days before the earliest visible entry. Trade-table directions
are green for long and red for short. Charts are rendered through `utils.viz.Viz`
(`dashboard/charts.py`'s `_PngViz` renders straight to a PNG instead of
Viz's own live-server registry, since refresh here is button-driven, not a
background loop) -- same colors, endpoint flags, and residual fill as every
other chart in the codebase.

## Scope / limitations

`fired` is a **snapshot** check -- does the latest reading cross
`±entry_threshold` (and pass the gate, if any) -- not a continuous
position replay through the trade engine. It won't tell you "we've been
long since March," only "as of the last analysis run, this would fire
long." Full position tracking would mean replaying the exact engine
forward from history on every run; a reasonable v2 if the snapshot view
isn't enough.
