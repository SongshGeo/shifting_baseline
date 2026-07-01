# Data & Reproduction

We do **not** release the full raw corpus (site-level documentary records and the
individual tree-ring reconstructions). Instead, we share the **two derived series**
that the analysis actually consumes, so readers can reproduce the central results and
plug in their own data:

| Shared series | What it is | Type |
| --- | --- | --- |
| **H-WDI** | Overall (regional) Wet/Dry **level** inferred from the historical archives | ordinal, 5 classes |
| **N-WDI** | Wet/Dry **z-score** inferred from the tree-ring reconstruction (Bayesian-integrated, standardised) | continuous |

Everything upstream of these — loading raw archives, spatial aggregation, and the
Bayesian combination of reconstructions — is *data production* and is not needed to
reproduce the analysis. The reproduction entry point is these two series.

## Expected format

Both are annual series indexed by calendar year (CE). A minimal, plug-in-your-own
schema:

```text
# h_wdi.csv  — historical-archive WDI (regional aggregate)
year,level
1470,-1
1471,0
1472,2
...            # level ∈ {-2,-1,0,1,2} = {SD, MD, N, MW, SW}; ~1470–1900 CE

# n_wdi.csv  — natural-proxy WDI (tree-ring z-score)
year,z
1470,-0.42
1471,0.15
1472,1.83
...            # z = standardised anomaly (mean 0, sd 1); ~1470–2000 CE
```

- **H-WDI** is the *regional* level per year (already aggregated across sites), encoded
  as an ordinal integer. `-2 … 2` maps to Severe Dry → Severe Wet (see
  [`constants`](../api/constants.md) `MAP`).
- **N-WDI** is a continuous z-score; the pre-instrumental span (≈1470–1900 CE) overlaps
  H-WDI, and the validation span (1901–2000 CE) is included for the instrumental check.

!!! tip "Bring your own data"
    Any archive that can be graded into **five ordinal levels** and any proxy that can
    be expressed as a **continuous z-score series indexed by year** drops straight into
    the pipeline below — no other preprocessing is assumed.

## Reproduce the central result

```python
import pandas as pd
from shifting_baseline.filters import calc_std_deviation, classify
from shifting_baseline.compare import compare_corr
from shifting_baseline.calibration import MismatchReport

h = pd.read_csv("h_wdi.csv", index_col="year")["level"]   # ordinal H-WDI
n = pd.read_csv("n_wdi.csv", index_col="year")["z"]        # continuous N-WDI z-score

# align on the shared (pre-instrumental) overlap
idx = h.index.intersection(n.index)
h, n = h.loc[idx], n.loc[idx]

# 1) sliding-window re-standardisation + rank correlation → the ~30-year optimum
r, p, k = compare_corr(
    h, n,
    filter_func=calc_std_deviation, filter_side="right",
    corr_method="kendall", window=30, min_periods=10,
)
print(f"Kendall τ (30-yr window) = {r:.3f}  (p = {p:.3g}, n = {k})")

# 2) mismatch / shifted-comparison bias, vs a Monte-Carlo null
report = MismatchReport(pred=h, true=classify(n), value_series=n)
report.analyze_error_patterns(mc_runs=1000)
fig = report.generate_report_figure()
```

- `classify(n)` turns the N-WDI z-score into the same five ordinal levels as H-WDI
  (empirical thresholds ±1.17σ, ±0.33σ).
- `calc_std_deviation` performs the sliding-window re-standardisation that is the
  mechanism under test. Sweep the `window` (see the
  [Analysis Pipeline](pipeline.md)) to recover the 20–40-year optimum.

## The ABM needs no shared data

The agent-based model generates its own synthetic climate, so it is fully reproducible
on its own — see the **[Agent-Based Model](abm.md)** guide. That is where the
generational-amnesia mechanism is tested against the empirical window recovered above.
