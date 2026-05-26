"""
Lookahead invariant test for oos_edge_test_fast.

Uses a fully deterministic, small dataset where I can trace every value by hand.

Setup
-----
  N=60, horizon=5, train_window=10
  feat[k]  = +1 if k is odd, -1 if k is even  (alternating, deterministic)
  z[k]     = +1 always                         (always "short" direction)
  pnl[k]   = feat[k] * 10                      (perfect positive IC = +1.0)
  EXCEPT pnl[PERTURB_BAR] = -feat[PERTURB_BAR] * 9000  (anti-correlated outlier)

Without the outlier: IC = +1 everywhere → filter passes odd bars (feat=+1 > median=0)
After the outlier enters the training window: IC < 0 → filter passes even bars (feat=-1 < median)

The key question: at which bar does the filter *first* flip its answer?

  New code (shift=horizon=5): first affected bar = PERTURB_BAR + 5
  Old code (shift=1):         first affected bar = PERTURB_BAR + 1

Bars in (PERTURB_BAR+1, PERTURB_BAR+5) are the "lookahead zone" — new code must
not change there, old code should.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strats.duration.signal_context import oos_edge_test_fast

# ── parameters ────────────────────────────────────────────────────────────────
N            = 300
HORIZON      = 5
TRAIN_WINDOW = 30    # min_periods = max(30, 30//4) = 30, valid
PERTURB_BAR  = 100   # outlier planted here
SCALE        = 10.0
OUTLIER      = -9000.0  # will be multiplied by feat[PERTURB_BAR]; makes pnl anti-correlated

# ── deterministic data ────────────────────────────────────────────────────────
idx     = pd.date_range("2020-01-01", periods=N, freq="B")
feat    = np.array([(1 if k % 2 else -1) for k in range(N)], dtype=float)
z       = np.ones(N, dtype=float)   # always +1

# Clean pnl: perfect positive IC (pnl = feat * SCALE)
pnl_clean = feat * SCALE

# Perturbed pnl: outlier at PERTURB_BAR makes that one point strongly anti-correlated
pnl_with_outlier = pnl_clean.copy()
pnl_with_outlier[PERTURB_BAR] = feat[PERTURB_BAR] * OUTLIER  # opposite direction, huge

# ── build y from pnl ─────────────────────────────────────────────────────────
# pnl[k] = sign(z[k]) * (-fwd[k]) = -fwd[k]  →  fwd[k] = -pnl[k]
# fwd[k] = y[k+H] - y[k]  →  y[k+H] = y[k] + fwd[k] = y[k] - pnl[k]

def build_y(pnl_arr: np.ndarray) -> pd.Series:
    y = np.full(N, np.nan)
    y[:HORIZON] = 100.0
    for k in range(N - HORIZON):
        y[k + HORIZON] = y[k] - pnl_arr[k]
    return pd.Series(y, index=idx)

y_clean   = build_y(pnl_clean)
y_outlier = build_y(pnl_with_outlier)

# ── features frame ────────────────────────────────────────────────────────────
features = pd.DataFrame({"ou_zscore": z, "feat": feat}, index=idx)


# ── helper: run old code (shift=1) inline ─────────────────────────────────────
def _run_old_code(y_in: pd.Series) -> pd.Series:
    fwd = y_in.diff(HORIZON).shift(-HORIZON)
    combined = pd.concat([
        features["ou_zscore"].rename("z"),
        features["feat"].rename("feat"),
        fwd.rename("fwd"),
    ], axis=1).dropna()
    combined = combined[combined["z"] != 0].copy()
    combined["pnl"] = np.sign(combined["z"]) * (-combined["fwd"])
    feat_c = combined["feat"]
    pnl_c  = combined["pnl"]
    min_p  = max(30, TRAIN_WINDOW // 4)
    ri  = feat_c.shift(1).rolling(TRAIN_WINDOW, min_periods=min_p).corr(pnl_c.shift(1))
    rm  = feat_c.shift(1).rolling(TRAIN_WINDOW, min_periods=min_p).median()
    am  = feat_c > rm
    return ri, am, (ri.notna() & ((ri > 0) & am | (ri <= 0) & ~am))


# ── run new code ──────────────────────────────────────────────────────────────
result_clean   = oos_edge_test_fast(features, y_clean,   "feat", horizon=HORIZON, train_window=TRAIN_WINDOW)
result_outlier = oos_edge_test_fast(features, y_outlier, "feat", horizon=HORIZON, train_window=TRAIN_WINDOW)

mask_new_clean   = result_clean["filter_mask"]
mask_new_outlier = result_outlier["filter_mask"]

ri_old_clean, _, mask_old_clean   = _run_old_code(y_clean)
ri_old_outlier, _, mask_old_outlier = _run_old_code(y_outlier)

# ── compare ───────────────────────────────────────────────────────────────────
# Align to the shared index
shared = mask_new_clean.index
new_changed = (mask_new_clean != mask_new_outlier.reindex(shared, fill_value=False))
old_changed = (mask_old_clean != mask_old_outlier.reindex(shared, fill_value=False))

bar_of = {d: i for i, d in enumerate(shared)}
changed_new = sorted(bar_of[d] for d in shared[new_changed])
changed_old = sorted(bar_of[d] for d in shared[old_changed])

lookahead_zone = range(PERTURB_BAR + 1, PERTURB_BAR + HORIZON)

print("=" * 65)
print(f"N={N}, horizon={HORIZON}, train_window={TRAIN_WINDOW}")
print(f"PERTURB_BAR={PERTURB_BAR}")
print()
print("Expected first bar where filter changes:")
print(f"  new code: ≥ {PERTURB_BAR + HORIZON}   (outlier first visible at k-horizon={PERTURB_BAR})")
print(f"  old code: ≈ {PERTURB_BAR + 1}   (shift=1 peeks into bar {PERTURB_BAR})")
print()
print(f"Actual first bar changed:")
print(f"  new code: {changed_new[0] if changed_new else 'none'}")
print(f"  old code: {changed_old[0] if changed_old else 'none'}")
print()
print("Lookahead zone bars [PERTURB_BAR+1 .. PERTURB_BAR+HORIZON-1] =",
      list(lookahead_zone))
lz_new = [b for b in changed_new if b in lookahead_zone]
lz_old = [b for b in changed_old if b in lookahead_zone]
print(f"  new code changed in zone: {lz_new}  (must be empty for correct behavior)")
print(f"  old code changed in zone: {lz_old}  (non-empty shows lookahead)")
print()

# Print a mini table around PERTURB_BAR for the new code
print("─" * 65)
print("Detail: rolling IC and filter_mask around PERTURB_BAR (new code)")
print(f"  {'bar':>4}  {'feat':>5}  {'pnl_base':>10}  {'pnl_out':>10}  "
      f"{'mask_clean':>10}  {'mask_out':>10}  {'changed':>7}")

# Peek at internal IC: recompute manually for new code
fwd_o = y_outlier.diff(HORIZON).shift(-HORIZON)
comb_o = pd.concat([features["ou_zscore"].rename("z"), features["feat"].rename("feat"),
                     fwd_o.rename("fwd")], axis=1).dropna()
comb_o = comb_o[comb_o["z"] != 0].copy()
comb_o["pnl"] = np.sign(comb_o["z"]) * (-comb_o["fwd"])

for b in range(PERTURB_BAR - 2, min(PERTURB_BAR + HORIZON + 3, len(shared))):
    d = shared[b]
    f_val = feat[b]
    p_clean_val  = pnl_clean[b]  if b < len(pnl_clean) else np.nan
    p_out_val    = pnl_with_outlier[b] if b < len(pnl_with_outlier) else np.nan
    mc = mask_new_clean.iloc[b] if b < len(mask_new_clean) else "-"
    mo = mask_new_outlier.iloc[b] if b < len(mask_new_outlier) else "-"
    ch = "*" if b in changed_new else " "
    # annotate lookahead zone
    lz = " [LOOKAHEAD]" if b in lookahead_zone else ""
    print(f"  {b:>4}  {f_val:>5.0f}  {p_clean_val:>10.1f}  {p_out_val:>10.1f}  "
          f"{'T' if mc else 'F':>10}  {'T' if mo else 'F':>10}  {ch:>7}{lz}")

print()

# ── verdict ───────────────────────────────────────────────────────────────────
new_ok = len(lz_new) == 0 and (not changed_new or changed_new[0] >= PERTURB_BAR + HORIZON)
old_bad = len(lz_old) > 0 or (changed_old and changed_old[0] < PERTURB_BAR + HORIZON)

print("─" * 65)
if new_ok and old_bad:
    print("PASS  new code: no lookahead  |  old code (shift=1): had lookahead")
elif new_ok and not old_bad:
    print("INCONCLUSIVE  both appear clean — outlier may not have flipped IC")
elif not new_ok:
    print("FAIL  new code still has lookahead bias")
print("─" * 65)
