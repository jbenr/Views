# Repo Consolidation Prompt

Copy everything below the line into a fresh Claude session opened at the repo
root. It is self-contained — it assumes no memory of any prior conversation.

---

## Your task

The `Views` repo at `/Users/jamesreichert/werk/Views` has grown to 149 tracked
files and ~30,000 lines of Python across 11 top-level directories. The README is
accurate and well-written — this is **not** a documentation problem. It's a
volume problem: the owner can no longer tell which code is load-bearing and which
is abandoned research.

Your job has two phases:

1. **Audit (read-only)** — produce `REPO_MAP.md`, an evidence-backed
   classification of every Python file. Then **stop and wait for approval.**
2. **Execute (only after approval)** — move dead research to `archive/`, delete
   scratch artifacts, update the README, verify tests still pass.

## Hard constraints

- **Environment:** use the `views` conda env on this Mac. `CLAUDE.md` §7 mentions
  `2s10s` — that is the Windows data box, not this machine.
- **Phase 1 writes exactly one file** (`REPO_MAP.md`) and nothing else. No `Edit`,
  no `mv`, no `rm`, no `git` mutations until the owner approves the map.
- **Never touch anything the import graph or `tests/` reaches.**
- **`data_pull/` is out of scope for deletion.** Its scripts run on a separate
  Windows box and are CLI entry points, not library imports — "zero inbound
  imports" is the *expected* state there, not evidence of death.
- **`execution/` is out of scope.** It's deliberately stubbed, safety-gated, and
  covered by tests.
- **`notes/PLAN.md` is out of scope.** It's the project north star.
- Obey `CLAUDE.md` §3 (Surgical Changes). A file being moved is moved *verbatim* —
  no reformatting, no import cleanup, no "while I'm in here" improvements.

---

## Phase 0 — Safety net

Before reading anything else:

1. `git status` — confirm a clean working tree. If it isn't, stop and report.
2. `git switch -c consolidate`
3. Run `pytest` and **record the exact result** (e.g. `147 passed, 2 failed`).
   Write this number into `REPO_MAP.md`. It is the acceptance criterion for
   Phase 2: a test failing before must be allowed to keep failing, and **zero
   new failures are acceptable.** Do not fix pre-existing failures — report them.

---

## Phase 1 — Build `REPO_MAP.md`

Classify **every** tracked `.py` and `.ipynb` file into exactly one bucket:

| Bucket | Meaning |
|---|---|
| `LIVE` | Imported by the spine, by `tests/`, or by `dashboard/registry.py` |
| `ENTRY` | Not imported, but a genuine CLI entry point — `python -m ...` modules, `data_pull/` scripts, `book/*/` strategy modules that expose `STRATEGY` |
| `ARCHIVE` | Self-contained research, no inbound reference, value is historical only |
| `DELETE` | Scratch, debug dumps, notebook duplicates of a `.py`, build artifacts |

### How to determine inbound references — read this carefully

A naive `grep` for the dotted module path **produces false negatives**, because
`stats/__init__.py`, `utils/__init__.py`, and `backtest/__init__.py` re-export
with `from .module import *`. A file with zero direct references to
`stats.pca` is still very much alive if callers do `from stats import fit_pca`.

For each candidate module you believe is dead:

1. Check whether its package `__init__.py` re-exports it.
2. If so, extract the module's public symbol names and grep for **those**, across
   `*.py` **and** `*.ipynb`.
3. Only after both checks come back empty may you call it unreferenced.

`stats/pca.py` is the worked example: zero hits on `stats.pca`, but
`stats/__init__.py` does `from .pca import *` and `fit_pca` / `roll_pca` /
`residual_from_pca` are used by `tests/test_stats.py`, `dig/pca_curve_signals.py`,
and two notebooks. It is `LIVE`.

Also gather, per file:

- Line count.
- `git log -1 --format=%ad --date=short -- <file>` for staleness.
- For each `.py`/`.ipynb` pair with the same stem: extract the notebook's source
  cells and diff against the `.py` to confirm one is genuinely derivative of the
  other **before** marking either `DELETE`. If they've diverged, both are
  `ARCHIVE`.

### `REPO_MAP.md` output shape

1. **Orientation (one paragraph + one command).** What this repo is, and the
   single command that runs the live thing. This is the "help me get my bearings"
   deliverable — write it for someone returning after three months away.
2. **The spine.** The ~15 files that carry the system: path, role in one line,
   inbound count. Anything not on this list is, by definition, peripheral.
3. **Full inventory table.** `path | LOC | last commit | bucket | evidence`.
   The evidence column must be a *fact* ("imported by tests/test_lab.py"), not a
   judgment ("looks important").
4. **Proposed actions.** Explicit `git mv` / `git rm` commands, grouped by
   rationale, **with a line-count total per group** so the owner can see the size
   of each decision before agreeing to it.
5. **Open questions.** Anything you could not classify confidently. Do not guess
   — an honest "I can't tell whether `dig/gate_window_oos.py` is still in play"
   is far more useful than a wrong bucket.

Then **stop.** Present the map and wait.

---

## Phase 2 — Execute (only after explicit approval)

1. Create `archive/README.md` stating: this is dead research kept for reference;
   it is not imported, not tested, not maintained; recover context with
   `git log --follow <path>`.
2. `git mv` the `ARCHIVE` set, preserving structure (`archive/duration/`,
   `archive/dig/`). Use `git mv`, not `mv` — history must follow.
3. `git rm` the `DELETE` set. For every removed file, append `path + last commit
   SHA` to a recovery table in `REPO_MAP.md` so it can be restored with
   `git show <sha>:<path>`.
4. Add missing ignore patterns to `.gitignore` (`.DS_Store`, `.idea/`,
   `.pytest_cache/`, `*.egg-info/`, `.ipynb_checkpoints/`) and
   `git rm --cached` any that are currently tracked.
5. Exclude `archive/` from tooling — pytest `norecursedirs`/`testpaths` in
   `pyproject.toml`, and any packaging `exclude`.
6. **Re-run `pytest`. It must match the Phase 0 baseline exactly.** If it doesn't,
   revert the offending move and report — do not patch around it.
7. Update `README.md`:
   - Add an `archive/` row to the repository-map table.
   - Line 33 lists `main.py`, `_db_inspect.py`, `_gap_check.py` as scratch —
     correct it once they're gone.
   - **Line 76 cites `book/curve/research.py`, which does not exist.** Repoint it
     and the `book/duration/spread_rv.py` reference at surviving examples
     (`book/curve/tens_10s30s.py`, `book/rate_vol/template.py`).
8. Commit one group at a time so any single decision is independently revertible:
   - `chore: archive dead duration research`
   - `chore: archive dig/ exploratory scripts`
   - `chore: remove scratch and debug artifacts`
   - `docs: update repo map and README`

---

## Anti-goals — do not do these

- Do **not** rewrite, reformat, or "modernize" a file that is merely being moved.
- Do **not** consolidate `book/duration/`'s four large research files into one.
  That's a research decision, not a cleanup decision, and it isn't yours.
- Do **not** touch `data_pull/`, `execution/`, or `notes/PLAN.md`.
- Do **not** delete tests, including ones that look redundant.
- Do **not** delete anything from `book/basis/` — it is active work (last touched
  2026-08-07, the most recent commit in the repo).
- Do **not** summarize or shorten `REPO_MAP.md` to make it readable. It's a
  reference document; completeness beats brevity.

---

## Appendix — measured evidence (verify, don't trust)

The following was measured on 2026-08-08. Treat it as a **starting hypothesis you
must confirm**, not as ground truth. Files change.

### Likely `DELETE` — scratch and debug artifacts

Tracked in git, all clearly disposable:

```
dashboard/_callback_out.txt      dashboard/_server_out.txt
dashboard/_inspect_out.txt       dashboard/_server_out2.txt
dashboard/_server_err.txt        dashboard/_slug_out.txt
dashboard/_server_err2.txt
dashboard/_callback_test.py      (79 lines, 2026-07-23)
dashboard/_inspect_callback.py   (6 lines,  2026-07-23)
_gap_check.py                    (21 lines, 2026-06-23)
book/duration/_tmp_research.py   (835 lines, 2026-06-30)
```

Untracked but present, should be gitignored: `.DS_Store`, `.idea/`,
`.pytest_cache/`, `views.egg-info/`, `.ipynb_checkpoints/`.

Judgment calls, not obvious — flag these rather than assuming:
`_db_inspect.py` (19 lines) and `data_pull/_inspect.py` (23 lines) are named like
scratch but `_db_inspect.py` was touched 2026-07-30. `main.py` (95 lines,
2026-01-13) is a Bloomberg-API scratchpad the README already labels as such.

### Duplicate `.py` / `.ipynb` pairs — diff before deciding

```
book/duration/drill_to_the_core.ipynb (429)   <->  .py (1825)
dig/beta_scan_10s_factors.ipynb      (1034)   <->  .py (294)
dig/beta_scan_breakevens.ipynb       (2496)   <->  .py (311)
```

Note the line counts diverge sharply in both directions — these are probably
*not* clean exports of one another. Diff before deleting either side.

### Likely `ARCHIVE` — `book/duration/`, ~5,700 lines, zero inbound references

```
setups.py              1867   2026-06-30
drill_to_the_core.py   1825   2026-07-20
_tmp_research.py        835   2026-06-30   (-> DELETE, not archive)
signal_context.py       604   2026-06-30
exits_ou.py             333   2026-06-30
spread_rv.py            157   2026-07-09
pipeline.py              47   2026-07-09
research.ipynb         1022   2026-06-30
drill_to_the_core.ipynb 429   2026-06-30
```

Verified: nothing in `tests/`, `dashboard/`, or `backtest/` imports any of it.
The only cross-references are internal (`drill_to_the_core.py` imports
`signal_context.py`) plus two prose mentions in `README.md` and `notes/TODO.md`.

### Likely `ARCHIVE` — `dig/`, ~5,300 `.py` lines + ~9,600 notebook lines

```
pca_dash.py             1161   2026-07-09
intratrade_research.py   629   2026-02-13
pca_curve_signals.py     616   2026-07-09
sizing_research.py       396   2026-02-13
direction_curve.py       380   2026-07-09
beta_scan_breakevens.py  311   2026-07-09
beta_scan_10s_factors.py 294   2026-07-09
gate_window_oos.py       265   2026-07-29   <- recent; ask before archiving
strat_curve_book.py      182   2026-02-13

ten_vs_fly.ipynb            2851   2026-03-07
beta_scan_breakevens.ipynb  2496   2026-03-08
db_explorer.ipynb           1794   2026-03-08
beta_scan_10s_factors.ipynb 1034   2026-03-07
beta_weighted_10s30s.ipynb   680   2026-04-30
basis.ipynb                  506   2026-03-08
```

Self-contained: only `pca_dash.py` ↔ `pca_curve_signals.py` reference each other.
`dig/db_explorer.ipynb` may still be a useful interactive tool — ask.
`dig/gate_window_oos.py` is recent enough to still be in play — ask.

### The spine — measured inbound import counts

```
utils/viz.py            <-17   2366 lines
utils/helpers.py        <-16    281
utils/market_data.py    <-12    229
backtest/strategy.py    <- 7   1608
utils/rates.py          <- 6     72
data_pull/berg.py       <- 5    849
backtest/lab.py         <- 4   1555
stats/ou.py             <- 4    289
utils/research_app.py   <- 4    236
utils/tickers.py        <- 4     77
backtest/engine.py      <- 3   1170
dashboard/registry.py   <- 3    349
stats/ols.py            <- 3    297
```

`dashboard/registry.py` is what makes a strategy "live" — it promotes any
`book/*` module exposing a `STRATEGY` object. The promoted curve strategies are
`book/curve/tens_10s30s.py`, `twos_10s30s.py`, `real10y_2s10s.py`,
`pc1_10s30s.py`. All are `LIVE`.

`tests/` imports: `backtest`, `backtest.engine`, `backtest.lab`,
`backtest.strategy`, `backtest.validation`, `book.rate_vol.template`,
`book.curve.*`, `dashboard`, `dashboard.charts`, `dashboard.params`,
`dashboard.registry`, `execution`, `execution.ibkr`, `stats`, `stats.ols`,
`utils.market_data`. Everything on that list is `LIVE` regardless of anything
else you find.

### Sanity checks

```bash
git ls-files '*.py' | wc -l          # expect 106
ls book/curve/research.py            # expect: no such file (README line 76 is stale)
grep -rn "book\.duration" --include='*.py' . | grep -v '^\./book/duration/'   # expect no hits
```
