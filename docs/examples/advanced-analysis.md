# Advanced Analysis

These examples reproduce the headline empirical results: the window sweep that yields
the ~30-year optimum, and the Monte Carlo significance test for the mismatch bias.

## The window sweep (~30-year optimum)

`compare.sweep_slices` generates rolling time windows over the record; for each slice,
`sweep_max_corr_year` finds the re-standardisation window that maximises correlation —
the analysis behind the headline plot.

```python
from shifting_baseline.compare import sweep_slices, sweep_max_corr_year

# rolling 200-year slices, every 20 years
slices = sweep_slices(start=1469, span=200, step=20)

# for each slice, the window size maximising Kendall τ
optima = sweep_max_corr_year(
    h_cat, n_wdi,
    slices=slices,
    corr_method=cfg.corr_method,
    filter_side=cfg.filter_side,
    min_periods=cfg.min_period,
)
print(optima)   # optimal window per slice — clusters in the 20–40-yr band
```

!!! note
    Function signatures evolve — check the [`compare` API](../api/compare.md) (generated
    from the code) for the exact current parameters.

## Correlation surface

```python
from shifting_baseline.compare import experiment_corr_2d

corr_df, r_benchmark, ax = experiment_corr_2d(
    data1=h_cat,
    data2=n_wdi,
    corr_method=cfg.corr_method,
)
# corr_df: τ over (window size × minimum samples); r_benchmark: no re-standardisation
```

## Monte Carlo significance ([`calibration`](../api/calibration.md))

The shifted-comparison bias is tested against a null in which archival grades are
independently redrawn at their fixed marginal frequencies. This is built into
[`MismatchReport`][shifting_baseline.calibration.MismatchReport] via `mc_runs`
(default **1000**):

```python
from shifting_baseline.calibration import MismatchReport

report = MismatchReport(pred=h_cat, true=classify(n_wdi), value_series=n_wdi)
report.analyze_error_patterns(mc_runs=1000)   # per-cell z / p-values vs the null
fig = report.generate_report_figure()
```

## Data-scenario robustness

Re-run under different proxy-integration scenarios (`ds=pure|best|…`) or a different
validation dataset (`using_val_data=china|gpcc|cru`) to confirm the pattern is not an
artefact of one dataset:

```bash
uv run python -m shifting_baseline how=correlation ds=best using_val_data=gpcc
```

Next: **[ABM Simulation](abm-simulation.md)**.
