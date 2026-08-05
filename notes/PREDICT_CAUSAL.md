# Making `--predict` a causal stage

**Status: proposal, not as-built.** Nothing here is implemented. `--predict`
today is what `backtest/lab.py:971` (`predict_scan`) and
`backtest/strategy.py:31-37` describe. This document argues for changing what
that stage *emits*, not where it sits in the funnel.

Companion reading: [`RESEARCH_ROADMAP.md`](RESEARCH_ROADMAP.md) §3 (Candidate),
whose unchecked boxes — "log every attempted candidate", "compare against
no-skill baselines", "run sign-flip, shuffled-target, and placebo-feature
tests" — this design is built to close.

---

## 1. What the stage does today

`predict_scan` sweeps (entry signal × beta_lb × ou_lb × threshold × horizon ×
gate × bucket), samples the **first bar** of each threshold excursion, and emits
four numbers per cell (`lab.py:1044-1049`):

```
n_obs   ic   hit_rate   fire_rate
```

Selection then ranks on `nbr_ic` — the median IC over adjacent grid cells
(`lab.py:1169`) — with floors on `predict_min_obs`, `predict_min_neighbors`,
and `predict_min_independent_events`.

Two things about this are already right and should survive any redesign:
**first-crossing sampling**, which stops one excursion from contributing forty
correlated observations, and **neighborhood-median selection**, which refuses
to promote a lone grid spike. Most research stacks have neither.

### What IC is actually measuring here

`ic` is a Pearson correlation of `z` against `-Δlevel`, pooled across long and
short triggers (`lab.py:1031-1042`). After threshold truncation `z` is bimodal
— a cluster near `+τ…+3τ` and one near `−τ…−3τ` — so that correlation is
driven mostly by the *gap between the two groups*, not by magnitude within
either tail. It is closer to a standardized two-group mean difference than to a
cross-sectional IC. That is a defensible screen. It is just not what the name
suggests, and it has four gaps:

| Gap | Consequence |
|---|---|
| **No mean move emitted** | IC is edge ÷ dispersion. A cell can top the board because forward vol was low, not because the edge was large. You cannot tell a clean 2bp reversion from a tradeable 15bp one — and only one survives `transaction_cost_bps`. |
| **No √n scaling** | IC 0.35 on 31 events ranks alongside IC 0.35 on 400. `predict_min_obs` is a floor, not a correction. |
| **Pearson is outlier-fragile** | Two March-2020 triggers can carry a cell. |
| **One horizon at a time** | Six scalar scans, then pick the best. Lossy, and picking the max is an unpriced selection tax. |

The last gap is the expensive one, and fixing it fixes the other three as a
side effect.

---

## 2. The reframe

Stop asking *"does the size of z correlate with the forward move?"* and start
asking *"what happens to the traded level after a trigger, relative to what
would have happened without one?"*

That is a counterfactual, and it names the pieces:

- **Treatment** — the first bar where `|z_t| ≥ τ` (and the gate allows).
- **Outcome** — the change in the traded level over the following `h` bars.
- **Control group** — every eligible non-trigger bar.
- **Confounders** — anything that causes both the trigger and the forward move.

The estimand is the **impulse response**: β at each horizon, in bps, with an
honest standard error. Not one number — a curve.

---

## 3. The estimator: local projections

For each horizon `h ∈ {−H⁻ … 0 … H⁺}`, one regression:

$$\Delta y_{t \to t+h} \;=\; \alpha_h \;+\; \beta_h D_t \;+\; \delta_h O_{t,h} \;+\; \gamma_h' X_t \;+\; \varepsilon_{t+h}$$

- $\Delta y_{t\to t+h}$ — **sign-adjusted** change in the traded level:
  $-\operatorname{sign}(z_t)\,(y_{t+h}-y_t)$, so a profitable fade is positive
  regardless of direction. Units: bps (yields are already ×100 at load).
- $D_t \in \{0,1\}$ — trigger indicator, first-crossing only.
- $O_{t,h}$ — count of additional first-crossings in $(t,\,t+h]$. Without it,
  $\beta_h$ at long horizons quietly absorbs the *next* signal. Direct port of
  the `other` term in the article that prompted this.
- $X_t$ — pre-trigger controls (§4).
- Errors: **HAC / Newey–West, `maxlags = h`**. Non-negotiable — shifting the
  outcome forward `h` bars makes residuals overlap across `t` by construction,
  and plain OLS standard errors will overstate significance badly.

Because the outcome is a *change from t*, $\beta_h$ is **cumulative by
construction**: it is the expected bps earned from holding `h` bars, gross of
costs. The marginal bar is $\beta_h - \beta_{h-1}$. That parameterization is
what makes the output directly usable by `--exit`.

### Estimate long and short triggers separately first

Pooling forces symmetry. Steepener and flattener triggers do not behave alike,
and a pooled IC cannot tell you so. Fit $\beta_h^{+}$ and $\beta_h^{-}$, test
$\beta_h^{+} = \beta_h^{-}$, and pool only if you fail to reject.

---

## 4. Control set, and the rule that governs it

**The timing rule: anything dated at or before `t` is a control; anything dated
after `t` is a mediator, and conditioning on it destroys the estimate.**

This is the failure the source article diagnoses. Their gradient booster gave
the treatment ~zero credit because `lag1` of the outcome absorbed it — `lag1`
sits *on the causal path*, so controlling for it blocks the path. The same
variable dated before treatment would have been a legitimate confounder
control. Same series, opposite role, decided purely by timing.

**Legitimate controls** (all observable at the entry decision):

- pre-trigger level momentum: $\Delta y_{t-k \to t}$ for a couple of `k`
- residual vol, `r2`, `beta_cv`, `beta_trend` at `t`
- the gate's causal percentile at `t`
- realized vol / MOVE regime at `t`

**Never controls:** anything at `t+k` for `k>0`; forward realized vol; the
residual at `t+h` (that *is* the outcome); the exit level.

### The generalization worth carrying into the rest of the book

`notes.md` diagnostic #4 proposes:

```
Δ10s30s ~ Δ10Y + Δswap_spread + ΔMOVE + Δmortgage_OAS
```

described as "10Y coefficient = pure signal; the rest are noise channels to
clean out." If a 10Y move *causes* swap spreads and MOVE to move, which then
move the curve, those regressors are mediators, not confounders. The 10Y
coefficient then estimates the **direct** effect only — the part that bypasses
the vol and basis channels — not the total effect. Both are legitimate targets.
It just has to be a choice, and right now the note reads as if the two were the
same quantity.

### Gates are subgroup analysis; controls are adjustment

Worth naming, because the current stage only has the first. `gate_scan` splits
the sample and measures the effect inside a bucket. Local projections keep the
whole sample and estimate the effect holding the covariate fixed. Both are
valid. But subgroup analysis across ~5 conditions × buckets × percentile
windows is a multiple-comparisons problem that grows multiplicatively, while
covariate adjustment is one regression. Use adjustment for known confounders;
reserve gating for genuine regime effects you expect to be non-linear.

---

## 5. The threat this is really built to catch

**Regression to the mean is not alpha.** If the residual is measured with noise
— a stale print, a holiday mark, bid/ask bounce on an EOD generic — then an
extreme reading is partly noise, the noise does not repeat, and the residual
"reverts." That looks identical to edge in an IC table and is completely
untradeable.

The response *curve* separates the two and a single-horizon IC cannot:

| Shape of $\beta_h$ | Reading |
|---|---|
| Jumps at `h=1`, flat thereafter | One-bar snapback. Microstructure or a bad print. Not tradeable. |
| Accumulates smoothly over `h=1…20`, then flattens | Genuine slow repricing. This is the one you want. |
| Rises, peaks, then decays back toward zero | Real but overshooting — the exit matters enormously. |
| Never separates from zero | No effect. |

A 20-day IC can be driven entirely by a one-day bounce, and today you would
never see it.

**The paired test:** estimate $\beta_h$ twice, entering at `t` and at `t+1`. If
the edge dies with one bar of delay, it is a print artifact, not a signal. This
is cheap and it should be a hard gate.

---

## 6. Falsification battery

Three tests, ranked by how much they actually buy.

### 6a. Randomization (sharp null) — the strongest

Resample trigger dates under the null, preserving *count and clustering*
(stationary block bootstrap over the trigger series, block length ≈ the
residual's half-life). Re-estimate $\beta_h$ each draw. The resulting
distribution gives a p-value that respects the serial structure of the data,
which HAC asymptotics do not when `n_obs` is 30. **With event counts this
small, treat the bootstrap p-value as primary and the HAC t-stat as a
cross-check, not the reverse.**

Closes: "shuffled-target test" in the roadmap.

### 6b. Timing placebo — cheapest, most diagnostic

Shift every trigger forward by `k = 3, 5, 10` bars and re-estimate. If
$\beta_h$ survives a misdated trigger nearly intact, the signal has **no timing
content** — it is identifying a regime, not a moment. That is still tradeable
information, but it is a different strategy with a different exit, and it
should not be promoted as a threshold-crossing fade.

### 6c. Placebo outcome

Apply the same trigger to an unrelated target with comparable persistence and
vol. A non-zero $\beta_h$ means the "effect" is a generic property of extreme
readings, not of this pair.

Closes: "placebo-feature test" in the roadmap.

### A note on the pre-trend test

The obvious test — estimate at `h < 0`, where there can be no effect — is
**mechanically non-zero for a residual fade** and must not be read as the
article reads it. A large positive residual arose *because* the target rose
relative to fitted, so $\Delta y_{t-k\to t}$ is positive by construction. Run
it, plot it, but interpret it only against a matched-control baseline. It is a
plumbing check (does it look wilder than construction explains?), not the
identification test. 6a and 6b are the real ones.

---

## 7. Output schema

Per `(setup, horizon, direction)`, replacing the current four columns:

| Column | Meaning |
|---|---|
| `beta_bps` | Cumulative expected move, bps, gross |
| `se_bps`, `t_hac` | HAC standard error and t-stat |
| `ci_lo`, `ci_hi` | 95% interval |
| `beta_marginal` | `beta_h − beta_{h−1}` — the value of one more bar |
| `beta_raw` | Same, **without** controls. `beta_raw − beta_bps` is your estimate of how much confounding the controls removed. Publish both. |
| `beta_lag1` | Entry delayed one bar (§5) |
| `p_boot` | Randomization p-value (6a) |
| `beta_placebo_k` | Timing-placebo coefficient (6b) |
| `n_events`, `n_independent` | Kept from today |
| `h_star` | `argmax_h beta_bps` — the natural holding period |
| `h_dead` | First `h` where the CI covers zero |
| `ic`, `hit_rate`, `fire_rate` | **Kept.** Cheap, familiar, and useful as a cross-check |

**Rank on:** `beta_bps` at `h_star`, net of `transaction_cost_bps`, penalized
by `se_bps` — and keep the existing neighborhood logic by taking the median of
`beta_bps` across adjacent grid cells rather than the cell's own. A setup whose
bps edge does not clear costs is not a candidate, no matter how clean its IC.

---

## 8. What this hands to the rest of the funnel

This is why the change is worth making even though the ordering stays
`predict → exit → sweep`.

- **`--exit` gets a time stop it does not have to search for.** `h_star` and
  `h_dead` read straight off the curve. A parameter you *estimated* is not a
  parameter you *searched*, so this strictly reduces the selection burden
  carried into DSR/PBO. `exit_style="half_life_frac"` already reaches for this
  via the OU fit; the IRF is the nonparametric version that does not assume
  AR(1) dynamics hold.
- **`--sweep` gets a prior on stop placement.** The dispersion around
  $\beta_h$ bounds what a stop can cost you before it starts cutting the edge
  itself.
- **Six horizon scans collapse into one estimation**, and the "pick the best
  horizon" step — an unpriced selection decision today — disappears.

---

## 9. Implementation sketch

Add `local_projection_scan()` to `backtest/lab.py` alongside `predict_scan`;
do not replace it. Same inputs, same `combos` frame, same gate-mask machinery
(`_gate_masks`), so gates remain a scan dimension.

Cost is not the obstacle. For fixed `h`, each signal column is one small OLS
with `p ≈ 8` regressors. Build $X'X$ (8×8) and $X'y$ for all `K` columns at
once with `einsum`, then one batched `np.linalg.solve` over a `(K, H, p, p)`
stack. Even at `K = 2500` and `H = 12` that is 30k 8×8 solves — trivial, and
the same array shapes the current `_metrics` already produces, so the `cupy`
path carries over unchanged.

The bootstrap (6a) is the only real expense: `B ≈ 500` draws × the batched
solve. Run it on the shortlist that survives ranking, not the full grid.

Suggested rollout:

1. Implement `local_projection_scan` and run it *beside* the current scan on
   `book.curve.tens_10s30s`. Compare leaderboards. Where they disagree, one of
   them is wrong and it is worth knowing which.
2. Add `beta_lag1` and the timing placebo. Expect casualties.
3. Add the bootstrap on the shortlist.
4. Only then consider changing what `--predict` writes to `setups_file`.

---

## 10. What this does not buy

Say this out loud, because the phrase "causal analysis" invites more than it
delivers.

The trigger is **endogenous by construction** — it fires precisely when the
residual is extreme. This is a conditional event study, not a randomized
experiment. Local projections give you an *adjusted association*. The causal
reading rests on the assumption that, conditional on `X_t`, the trigger is
as-good-as-random, and in markets that assumption is never exactly true.
Unobserved confounders — positioning, a dealer unwind, an auction nobody
modeled — survive every test in §6.

What you actually get is worth having anyway:

1. **Magnitude in bps** instead of a unitless ratio, so cost comparison becomes
   possible.
2. **Correct inference** under overlapping windows, which the current IC
   silently violates.
3. **The response shape**, which distinguishes a real slow reversion from a
   one-bar print artifact.
4. **A falsification battery** that can kill a candidate before it consumes a
   sweep.
5. **Explicit control** of the confounders you *can* name, and a published
   estimate (`beta_raw − beta_bps`) of how much they mattered.

That is a large upgrade over a sorted IC column. It is not identification, and
the docs should never claim it is.

---

## 11. Roadmap checklist

Slots into [`RESEARCH_ROADMAP.md`](RESEARCH_ROADMAP.md) §3:

- [ ] `local_projection_scan()` in `backtest/lab.py`, emitting §7's schema
- [ ] Long/short triggers estimated separately, symmetry tested before pooling
- [ ] `O_{t,h}` subsequent-trigger control wired in
- [ ] HAC errors at `maxlags = h`
- [ ] `beta_lag1` one-bar-delay gate
- [ ] Block-bootstrap randomization p-value on the shortlist
- [ ] Timing placebo at `k = 3, 5, 10`
- [ ] Placebo outcome on an unrelated pair
- [ ] Both `beta_raw` and `beta_bps` published, never just the adjusted one
- [ ] Ranking switched to cost-adjusted bps with neighborhood median
- [ ] `h_star` / `h_dead` consumed by `--exit` as the default time stop
- [ ] Every attempted candidate logged, so DSR/PBO stops being a lower bound
