# Research app

Run `python -m research.app` in the `2s10s` environment (port 8052).

Setup remembers the last successfully loaded target and feature selections in the local, git-ignored `research/data/preferences.json`. Custom targets also retain their weights text. Each new page reads that file; missing, corrupt or removed selections fall back safely. Failed loads do not replace the saved preferences. This is shared across local browser sessions and does not change saved research runs or other default settings.

1. In **Setup**, choose the target, fixed or beta leg weights, features and data start. **Load** and **Fill tabs** both load the selected panel and carry its target/features into the research tabs. Custom weights appear only for a custom target; beta settings appear only for beta weighting. The dependent-leg override is under Advanced beta settings.
2. In **Dislocation**, choose a loaded feature and scan regression bases, model lookbacks, normalization/OU windows, entry thresholds, forecast horizons and gates. Normalized residual and OU z-score are distinct choices. By default discovery uses the first 70% of the aligned sample and reports the later 30% separately. Rows are ranked by median IC across tested model lookbacks, then individual IC, using the discovery period only. Event hit rate measures favorable forward moves after threshold crossings, not winning trades. Non-overlapping event counts expose dependence between forward labels.
3. Select **Backtest** on a discovery row. Its model, gate, panel snapshot and entry threshold carry into **Trade mechanics**. Test any combination of fixed observation counts, z-score bands, fractional signal reversion and entry-half-life multiples. All valid combinations run through the full engine with each selected hard stop and the specified round-trip cost. Exit bands at or above entry are excluded.
4. Inspect any configuration's equity curve, yearly P&L, exit reasons and closed trades. **Trade win rate** counts profitable closed trades. Average trade P&L uses closed trades; total equity also includes open marks. MAE/MFE and stops use available observations, not intraday prices. A half-life timer is rounded up and frozen at entry. Invalid half-lives cannot open a half-life-exit trade.

Progress beneath the loading logo reports database reads, regression fitting, completed discovery models and completed full backtests. The local server must run as one threaded process; progress is isolated per browser page and held in server memory.

Signals can be selected together. The scaled residual is `r / rolling_std(r)`; OU z-score is `(r - fitted_OU_equilibrium) / rolling_std(r)`. Both use the same residual series and rolling sample standard deviation, but different centres. Each signal has separate rankings, later-period joins and frozen backtest candidates. Regression fitting is shared between signals.

**Select all sweep options** fills every discovery and exit-grid multi-select/checklist without starting work or changing initial defaults. It leaves the target, feature, data split, minimum-events filter, transaction cost and execution convention unchanged. A live discovery model/cell count warns about very large grids. Exhaustive selections may require substantial RAM and runtime; selecting everything does not make that grid cheap.

Execution defaults to the next observation, with signals, gates and weights shifted together. Beta leg weights are fixed at entry and actual held legs are marked through exit. Discovery also scores beta baskets using weights frozen at each event, avoiding gains from changing hedge ratios. Same-observation execution is available as an explicitly optimistic diagnostic.

Every discovery/backtest run gets a new directory under `research/data/runs/`, with data, results, metadata and source fingerprints. Backtests also save trades, equity and yearly P&L for every configuration. They do not overwrite existing promoted strategy artifacts or alter live positions. Changing the loaded panel clears the previous results and requires fresh candidate selection.

The earlier/later split is a chronological research diagnostic, not a complete walk-forward validation. Model coefficients continue to update causally. Later P&L includes positions carried across the split. Repeated selection using later results consumes that sample. Relative Value and Fair Value tabs retain their existing unfinished status but receive the loaded context.

## Saved discovery runs

After loading a target and feature, **Saved discovery runs** automatically lists archived runs for that pair. Select a run to see its creation timestamp, exact data period, saved settings, missing requested settings, input-fingerprint comparison and calculation-code status. **Open saved run** ranks its existing scores without repeating discovery; it shows the whole archived grid, not just the currently requested subset. The minimum-events display filter remains adjustable when opening.

**Run discovery** always starts a fresh run. New runs record the complete grid, trade weights, input fingerprint and snapshot dates. Historical data revisions count as changed inputs even when the last date is unchanged. Changed data or chronological splits require a fresh run: results are not silently stitched across snapshots. Missing settings are identified, but partial-grid continuation and in-flight checkpoints are not implemented.

Exit tests selected from an archived discovery use its original data and weights with the current engine code. Older archives can be viewed, but incomplete legacy metadata prevents certified coverage matching; legacy runs without saved trade weights cannot be used for exit testing. Opening a different discovery clears the previously selected candidate and displayed exit results. The archive loader currently covers discovery runs, not reopening previously saved exit grids.
