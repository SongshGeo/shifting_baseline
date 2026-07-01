# Basic Usage

Practical, copy-pasteable snippets against the current API, starting from the two
[shared series](../guide/data.md).

## Load the shared data

```python
import pandas as pd

h_wdi = pd.read_csv("h_wdi.csv", index_col="year")["level"]  # ordinal H-WDI (levels)
n_wdi = pd.read_csv("n_wdi.csv", index_col="year")["z"]      # continuous N-WDI z-score

idx = h_wdi.index.intersection(n_wdi.index)                  # shared years
h_wdi, n_wdi = h_wdi.loc[idx], n_wdi.loc[idx]
```

## Build comparable categorical series

```python
from shifting_baseline.filters import classify

h_cat = h_wdi                # H-WDI is already ordinal
n_cat = classify(n_wdi)      # N-WDI z-score → 5 ordinal levels
```

## Correlate with sliding-window re-standardisation

```python
from shifting_baseline.filters import calc_std_deviation
from shifting_baseline.compare import compare_corr

r, p, n = compare_corr(
    h_wdi, n_wdi,
    filter_func=calc_std_deviation,   # re-standardise the natural series
    filter_side="right",
    corr_method="kendall",
    window=30,                        # ~30-year optimum
    min_periods=10,
)
print(f"Kendall τ = {r:.3f}  (p = {p:.3g}, n = {n})")
```

## Mismatch report

```python
from shifting_baseline.calibration import MismatchReport

report = MismatchReport(
    pred=h_cat,          # historical categories
    true=n_cat,          # natural categories
    value_series=n_wdi,  # underlying continuous values
)
report.analyze_error_patterns(mc_runs=1000)
fig = report.generate_report_figure()
```

Next: **[Advanced Analysis](advanced-analysis.md)** (the window sweep + Monte-Carlo
test) and **[ABM Simulation](abm-simulation.md)**.
