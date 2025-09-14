# Comparison Module

The `shifting_baseline.compare` module provides comprehensive tools for statistical comparison and correlation analysis between different climate datasets.

## Overview

This module handles:
- Correlation analysis between time series
- Rolling window correlation calculations
- Statistical significance testing
- Visualization of correlation patterns
- Time window sweeping and optimization

## Core Functions

### `compare_corr(data1, data2, filter_func=None, filter_side="both", corr_method="pearson", window_error="raise", n_diff_w=2, penalty=False, **rolling_kwargs)`

Compare two time series and calculate correlation coefficients.

**Parameters:**
- `data1`: First time series (pd.Series)
- `data2`: Second time series (pd.Series)
- `filter_func`: Filtering function to apply (optional)
- `filter_side`: Which series to filter ("both", "left", "right")
- `corr_method`: Correlation method ("pearson", "kendall", "spearman")
- `window_error`: Error handling for small windows ("raise", "nan")
- `n_diff_w`: Window difference parameter
- `penalty`: Whether to apply penalty for small windows
- `**rolling_kwargs`: Additional rolling window parameters

**Returns:**
- `tuple[float, float, int]`: (correlation, p-value, sample_count)

**Example:**
```python
from shifting_baseline.compare import compare_corr
import pandas as pd

# Basic correlation
r, p, n = compare_corr(series1, series2, corr_method="kendall")

# With filtering
from shifting_baseline.filters import calc_std_deviation
r, p, n = compare_corr(
    series1,
    series2,
    filter_func=calc_std_deviation,
    filter_side="right",
    window=20
)
```

### `compare_corr_2d(data1, data2, windows, min_periods, filter_func=None, corr_method="pearson", n_diff_w=2, penalty=False, **rolling_kwargs)`

Batch correlation analysis with multiple window sizes.

**Parameters:**
- `data1`: First time series
- `data2`: Second time series
- `windows`: Array of window sizes
- `min_periods`: Array of minimum periods for each window
- `filter_func`: Filtering function
- `corr_method`: Correlation method
- `n_diff_w`: Window difference parameter
- `penalty`: Penalty flag
- `**rolling_kwargs`: Additional rolling parameters

**Returns:**
- `tuple[np.ndarray, np.ndarray, np.ndarray]`: (correlations, p_values, sample_counts)

**Example:**
```python
import numpy as np
from shifting_baseline.compare import compare_corr_2d

# Define window parameters
windows = np.arange(5, 50, 5)
min_periods = np.repeat(3, len(windows))

# Batch correlation analysis
rs, ps, ns = compare_corr_2d(
    data1=historical_series,
    data2=climate_series,
    windows=windows,
    min_periods=min_periods,
    corr_method="kendall"
)
```

### `experiment_corr_2d(data1, data2, corr_method="pearson", time_slice=slice(None), filter_side="right", filter_func=None, sample_threshold=1, std_offset=0.2, p_threshold=0.05, penalty=False, n_diff_w=2, ax=None)`

Complete experimental correlation analysis with visualization.

**Parameters:**
- `data1`: First time series
- `data2`: Second time series
- `corr_method`: Correlation method
- `time_slice`: Time slice to analyze
- `filter_side`: Filter side
- `filter_func`: Filtering function
- `sample_threshold`: Minimum samples per window
- `std_offset`: Standard deviation offset for visualization
- `p_threshold`: P-value threshold for significance
- `penalty`: Penalty flag
- `n_diff_w`: Window difference parameter
- `ax`: Matplotlib axes for plotting

**Returns:**
- `tuple[pd.DataFrame, float, plt.Axes]`: (filtered_correlations, benchmark_correlation, axes)

**Example:**
```python
from shifting_baseline.compare import experiment_corr_2d
from shifting_baseline.filters import calc_std_deviation

# Complete correlation experiment
corr_df, r_benchmark, ax = experiment_corr_2d(
    data1=historical_series,
    data2=climate_series,
    corr_method="kendall",
    time_slice=slice(1600, 1900),
    filter_func=calc_std_deviation,
    filter_side="right",
    sample_threshold=2.0,
    p_threshold=0.01
)

# Display results
print(f"Benchmark correlation: {r_benchmark:.3f}")
print(f"Significant correlations: {(corr_df > 0.3).sum().sum()}")
```

## Time Window Analysis

### `sweep_slices(start_year, window_size, step_size, end_year)`

Generate all possible time windows for analysis.

**Parameters:**
- `start_year`: Starting year
- `window_size`: Size of each window
- `step_size`: Step size between windows
- `end_year`: Ending year

**Returns:**
- `tuple[list[slice], list[int], list[str]]`: (slices, mid_years, labels)

**Example:**
```python
from shifting_baseline.compare import sweep_slices

# Generate 200-year windows with 20-year steps
slices, mid_years, labels = sweep_slices(
    start_year=1000,
    window_size=200,
    step_size=20,
    end_year=1900
)

print(f"Generated {len(slices)} time windows")
for i, (slice_obj, mid_year, label) in enumerate(zip(slices, mid_years, labels)):
    print(f"Window {i+1}: {label} (mid-year: {mid_year})")
```

### `sweep_max_corr_year(data1, data2, slices, windows, min_periods, ratio=0.1, **compare_kwargs)`

Find maximum correlation for each time window.

**Parameters:**
- `data1`: First time series
- `data2`: Second time series
- `slices`: List of time slices
- `windows`: Array of window sizes
- `min_periods`: Array of minimum periods
- `ratio`: Ratio for top correlation selection
- `**compare_kwargs`: Additional comparison parameters

**Returns:**
- `tuple[list, list]`: (max_corr_years, max_correlations)

**Example:**
```python
from shifting_baseline.compare import sweep_max_corr_year
import numpy as np

# Define parameters
windows = np.arange(2, 100)
min_periods = np.repeat(5, len(windows))

# Find maximum correlations
max_corr_years, max_correlations = sweep_max_corr_year(
    data1=historical_series,
    data2=climate_series,
    slices=slices,
    windows=windows,
    min_periods=min_periods,
    corr_method="kendall"
)
```

## Utility Functions

### `get_filtered_corr(rs, ps, ns, windows, sample_threshold=2, p_threshold=0.01)`

Filter correlation results based on significance and sample size.

**Parameters:**
- `rs`: Correlation coefficients array
- `ps`: P-values array
- `ns`: Sample counts array
- `windows`: Window sizes array
- `sample_threshold`: Minimum samples per window
- `p_threshold`: P-value threshold

**Returns:**
- `np.ndarray`: Filtered correlation coefficients

**Example:**
```python
from shifting_baseline.compare import get_filtered_corr

# Filter correlations
filtered_rs = get_filtered_corr(
    rs=correlation_results,
    ps=p_values,
    ns=sample_counts,
    windows=window_sizes,
    sample_threshold=3.0,
    p_threshold=0.01
)
```

## Advanced Analysis Patterns

### 1. Multi-Method Correlation Analysis

```python
import pandas as pd
from shifting_baseline.compare import compare_corr

# Compare different correlation methods
methods = ["pearson", "kendall", "spearman"]
results = {}

for method in methods:
    r, p, n = compare_corr(
        historical_series,
        climate_series,
        corr_method=method
    )
    results[method] = {"correlation": r, "p_value": p, "samples": n}

# Create results DataFrame
results_df = pd.DataFrame(results).T
print(results_df)
```

### 2. Rolling Window Analysis

```python
import numpy as np
import matplotlib.pyplot as plt
from shifting_baseline.compare import compare_corr_2d

# Define rolling window parameters
window_sizes = np.arange(10, 100, 10)
min_periods = np.repeat(5, len(window_sizes))

# Calculate rolling correlations
rs, ps, ns = compare_corr_2d(
    data1=historical_series,
    data2=climate_series,
    windows=window_sizes,
    min_periods=min_periods,
    corr_method="kendall"
)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(window_sizes, rs, 'o-', label='Correlation')
plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
plt.xlabel('Window Size')
plt.ylabel('Correlation Coefficient')
plt.title('Rolling Window Correlation Analysis')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 3. Time Period Comparison

```python
from shifting_baseline.compare import experiment_corr_2d

# Define time periods
periods = {
    "Medieval": slice(1000, 1469),
    "Little Ice Age": slice(1469, 1659),
    "Early Modern": slice(1659, 1900),
    "Modern": slice(1900, 2000)
}

# Analyze each period
period_results = {}

for period_name, time_slice in periods.items():
    corr_df, r_benchmark, ax = experiment_corr_2d(
        data1=historical_series,
        data2=climate_series,
        time_slice=time_slice,
        corr_method="kendall"
    )

    period_results[period_name] = {
        "correlation": r_benchmark,
        "dataframe": corr_df
    }

    # Save plot
    ax.figure.savefig(f"correlation_{period_name.lower().replace(' ', '_')}.png")
```

### 4. Sensitivity Analysis

```python
# Test different filter parameters
filter_params = {
    "window": [10, 20, 30, 40, 50],
    "min_periods": [3, 5, 7, 10]
}

sensitivity_results = []

for window in filter_params["window"]:
    for min_period in filter_params["min_periods"]:
        try:
            r, p, n = compare_corr(
                historical_series,
                climate_series,
                filter_func=calc_std_deviation,
                window=window,
                min_periods=min_period,
                corr_method="kendall"
            )

            sensitivity_results.append({
                "window": window,
                "min_periods": min_period,
                "correlation": r,
                "p_value": p,
                "samples": n
            })
        except ValueError:
            # Skip invalid parameter combinations
            continue

# Analyze sensitivity
sensitivity_df = pd.DataFrame(sensitivity_results)
print(sensitivity_df.sort_values("correlation", ascending=False))
```

## Performance Optimization

### Memory Management

For large datasets, use chunked processing:

```python
# Process data in chunks
def chunked_correlation_analysis(data1, data2, chunk_size=1000):
    results = []

    for i in range(0, len(data1), chunk_size):
        chunk1 = data1.iloc[i:i+chunk_size]
        chunk2 = data2.iloc[i:i+chunk_size]

        r, p, n = compare_corr(chunk1, chunk2)
        results.append({"chunk": i//chunk_size, "correlation": r, "p_value": p})

    return pd.DataFrame(results)
```

### Parallel Processing

```python
from multiprocessing import Pool
from functools import partial

def parallel_correlation_analysis(data1, data2, window_sizes, n_processes=4):
    # Create partial function
    corr_func = partial(compare_corr, data1, data2)

    # Use multiprocessing
    with Pool(n_processes) as pool:
        results = pool.map(corr_func, window_sizes)

    return results
```

## Error Handling

```python
try:
    r, p, n = compare_corr(
        series1,
        series2,
        window=5,  # Very small window
        window_error="raise"
    )
except ValueError as e:
    print(f"Window too small: {e}")
    # Fallback to larger window
    r, p, n = compare_corr(series1, series2, window=20)
```

## Integration with Other Modules

### With Calibration Module

```python
from shifting_baseline.calibration import MismatchReport
from shifting_baseline.compare import compare_corr

# Calculate correlation
r, p, n = compare_corr(historical_series, climate_series)

# Create mismatch report
report = MismatchReport(
    pred=classified_historical,
    true=classified_climate,
    value_series=climate_series
)

# Compare correlation with mismatch analysis
print(f"Correlation: {r:.3f}")
print(f"Mismatch statistics: {report.get_statistics_summary(as_str=True)}")
```

### With Filters Module

```python
from shifting_baseline.filters import calc_std_deviation, classify
from shifting_baseline.compare import experiment_corr_2d

# Use filter functions in correlation analysis
corr_df, r_benchmark, ax = experiment_corr_2d(
    data1=historical_series,
    data2=climate_series,
    filter_func=calc_std_deviation,
    filter_side="right"
)
```
