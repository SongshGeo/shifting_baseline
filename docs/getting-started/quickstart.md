# Quick Start

Two reproducible entry points: the **empirical analysis from the shared data**, and the
**self-contained ABM**. Neither needs the raw corpus — see
[Data & Reproduction](../guide/data.md) for the shared series and their format.

## 1. Reproduce the empirical result (from shared data)

Load the two shared series — the historical-archive **H-WDI** (ordinal levels) and the
tree-ring **N-WDI** (z-score) — and run the sliding-window correlation:

```python
import pandas as pd
from shifting_baseline.filters import calc_std_deviation, classify
from shifting_baseline.compare import compare_corr

h = pd.read_csv("h_wdi.csv", index_col="year")["level"]   # H-WDI ordinal levels
n = pd.read_csv("n_wdi.csv", index_col="year")["z"]        # N-WDI z-score
idx = h.index.intersection(n.index); h, n = h.loc[idx], n.loc[idx]

# sliding-window re-standardisation + rank correlation (the ~30-year optimum)
r, p, k = compare_corr(
    h, n,
    filter_func=calc_std_deviation, filter_side="right",
    corr_method="kendall", window=30, min_periods=10,
)
print(f"Kendall τ (30-yr window) = {r:.3f}  (p = {p:.3g}, n = {k})")
```

The full walk-through — the mismatch/shifted-comparison test and the window sweep — is
in **[Data & Reproduction](../guide/data.md)** and the **[Examples](../examples/basic-usage.md)**.

## 2. Run the ABM (self-contained)

The agent-based model generates its own synthetic climate, so it runs with no data.
It is a Hydra app supporting `--multirun` sweeps:

```bash
# main analysis: AR(1) forcing under the personal baseline (H1)
uv run python shifting_baseline/abm.py +model.climate_process=ar1

# discriminate the mechanisms: sweep the four baselines
uv run python shifting_baseline/abm.py --multirun \
  model.memory_baseline=personal,collective,collective_lifetime,model
```

Each run writes its figures, logs and resolved `.hydra/config.yaml` to a timestamped
directory under `outputs/` (single) or `multirun/` (sweeps). See the
**[Agent-Based Model guide](../guide/abm.md)**.

## 3. Configuration

The ABM parameters and analysis knobs (e.g. `corr_method`, `low_pass.window_size`) are
Hydra-configured and overridable on the command line — see
**[Configuration](configuration.md)**.
