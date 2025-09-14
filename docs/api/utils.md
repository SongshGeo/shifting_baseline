# Utils Module

The `past1000.utils` module provides utility functions and helper classes for various operations throughout the Past1000 library.

## Overview

This module contains several submodules:
- `calc`: Statistical calculation utilities
- `config`: Configuration management
- `email`: Email notification system
- `log`: Logging utilities
- `plot`: Plotting and visualization helpers
- `types`: Type definitions and annotations

## Calculation Utilities (`utils.calc`)

### `calc_corr(data1, data2, how="pearson")`

Calculate correlation between two time series.

**Parameters:**
- `data1`: First time series
- `data2`: Second time series
- `how`: Correlation method ("pearson", "kendall", "spearman")

**Returns:**
- `tuple[float, float, int]`: (correlation, p-value, sample_count)

**Example:**
```python
from past1000.utils.calc import calc_corr

# Calculate correlation
r, p, n = calc_corr(series1, series2, how="kendall")
print(f"Correlation: {r:.3f}, P-value: {p:.3f}, Samples: {n}")
```

### `rand_generate_from_std_levels(data, mu=0.0, sigma=1.0)`

Generate random values from standard deviation levels.

**Parameters:**
- `data`: Input data array
- `mu`: Mean value
- `sigma`: Standard deviation

**Returns:**
- `tuple[np.ndarray, np.ndarray]`: (generated_values, standard_deviations)

**Example:**
```python
from past1000.utils.calc import rand_generate_from_std_levels

# Generate values from levels
values, stds = rand_generate_from_std_levels(level_data, mu=0.0, sigma=1.0)
```

### `find_top_max_indices(array, ratio=0.1)`

Find indices of top maximum values in an array.

**Parameters:**
- `array`: Input array
- `ratio`: Ratio of top values to find

**Returns:**
- `np.ndarray`: Indices of top maximum values

**Example:**
```python
from past1000.utils.calc import find_top_max_indices

# Find top 10% of values
top_indices = find_top_max_indices(correlation_array, ratio=0.1)
```

## Configuration Utilities (`utils.config`)

### `format_by_config(cfg)`

Format configuration object.

**Parameters:**
- `cfg`: Configuration object

**Returns:**
- `DictConfig`: Formatted configuration

**Example:**
```python
from past1000.utils.config import format_by_config

# Format configuration
formatted_cfg = format_by_config(raw_config)
```

### `get_output_dir()`

Get output directory path.

**Returns:**
- `Path`: Output directory path

**Example:**
```python
from past1000.utils.config import get_output_dir

# Get output directory
output_dir = get_output_dir()
print(f"Output directory: {output_dir}")
```

## Email Utilities (`utils.email`)

### `send_notification_email(success=True, error_msg=None, start_time=None)`

Send email notification about process completion.

**Parameters:**
- `success`: Whether the process was successful
- `error_msg`: Error message if failed
- `start_time`: Process start time

**Example:**
```python
from past1000.utils.email import send_notification_email
from datetime import datetime

# Send success notification
start_time = datetime.now()
send_notification_email(success=True, start_time=start_time)

# Send error notification
send_notification_email(success=False, error_msg="Process failed", start_time=start_time)
```

## Logging Utilities (`utils.log`)

### Logging Configuration

```python
from past1000.utils.log import setup_logging

# Setup logging
setup_logging(level="INFO", log_file="past1000.log")
```

## Plotting Utilities (`utils.plot`)

### `plot_correlation_windows(max_corr_year, max_corr, mid_year, slice_labels)`

Plot correlation windows visualization.

**Parameters:**
- `max_corr_year`: Maximum correlation years
- `max_corr`: Maximum correlations
- `mid_year`: Mid-year values
- `slice_labels`: Slice labels

**Returns:**
- `plt.Axes`: Plot axes

**Example:**
```python
from past1000.utils.plot import plot_correlation_windows

# Plot correlation windows
ax = plot_correlation_windows(
    max_corr_year,
    max_corr,
    mid_year,
    slice_labels
)
plt.show()
```

### `plot_confusion_matrix(cm_df, title=None, ax=None)`

Plot confusion matrix.

**Parameters:**
- `cm_df`: Confusion matrix DataFrame
- `title`: Plot title
- `ax`: Matplotlib axes

**Returns:**
- `plt.Axes`: Plot axes

**Example:**
```python
from past1000.utils.plot import plot_confusion_matrix

# Plot confusion matrix
ax = plot_confusion_matrix(confusion_matrix, title="Classification Results")
plt.show()
```

### `plot_mismatch_matrix(actual_diff_aligned, p_value_matrix, false_count_matrix, ax=None)`

Plot mismatch analysis matrix.

**Parameters:**
- `actual_diff_aligned`: Actual difference matrix
- `p_value_matrix`: P-value matrix
- `false_count_matrix`: False count matrix
- `ax`: Matplotlib axes

**Returns:**
- `plt.Axes`: Plot axes

**Example:**
```python
from past1000.utils.plot import plot_mismatch_matrix

# Plot mismatch matrix
ax = plot_mismatch_matrix(
    diff_matrix,
    p_value_matrix,
    false_count_matrix
)
plt.show()
```

### `heatmap_with_annot(matrix, p_value=None, ax=None)`

Create annotated heatmap.

**Parameters:**
- `matrix`: Data matrix
- `p_value`: P-value matrix for significance
- `ax`: Matplotlib axes

**Returns:**
- `plt.Axes`: Plot axes

**Example:**
```python
from past1000.utils.plot import heatmap_with_annot

# Create annotated heatmap
ax = heatmap_with_annot(data_matrix, p_value=significance_matrix)
plt.show()
```

## Type Definitions (`utils.types`)

### Type Aliases

```python
from past1000.utils.types import (
    PathLike,
    Region,
    Stages,
    HistoricalAggregateType,
    ToStdMethod,
    CorrFunc,
    FilterSide
)
```

**Common Types:**
- `PathLike`: Union of path types
- `Region`: Regional identifier type
- `Stages`: Stage specification type
- `HistoricalAggregateType`: Aggregation method type
- `ToStdMethod`: Standardization method type
- `CorrFunc`: Correlation function type
- `FilterSide`: Filter side type

## Advanced Usage Patterns

### 1. Custom Calculation Pipeline

```python
from past1000.utils.calc import calc_corr, find_top_max_indices
from past1000.utils.plot import plot_correlation_windows

def custom_correlation_analysis(data1, data2, window_sizes):
    """Custom correlation analysis pipeline."""

    results = []

    for window in window_sizes:
        # Calculate correlation for window
        r, p, n = calc_corr(
            data1.rolling(window).mean(),
            data2.rolling(window).mean(),
            how="kendall"
        )

        results.append({
            "window": window,
            "correlation": r,
            "p_value": p,
            "samples": n
        })

    # Find top correlations
    correlations = [r["correlation"] for r in results]
    top_indices = find_top_max_indices(np.array(correlations), ratio=0.1)

    # Plot results
    top_results = [results[i] for i in top_indices]
    ax = plot_correlation_windows(
        [r["window"] for r in top_results],
        [r["correlation"] for r in top_results],
        [r["window"] for r in top_results],
        [f"Window {r['window']}" for r in top_results]
    )

    return results, ax

# Use custom pipeline
results, ax = custom_correlation_analysis(series1, series2, range(10, 100, 10))
plt.show()
```

### 2. Configuration Management

```python
from past1000.utils.config import format_by_config, get_output_dir
from past1000.utils.log import setup_logging

def setup_analysis_environment(config):
    """Setup analysis environment with proper configuration."""

    # Format configuration
    formatted_config = format_by_config(config)

    # Setup logging
    output_dir = get_output_dir()
    log_file = output_dir / "analysis.log"
    setup_logging(level="INFO", log_file=str(log_file))

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    return formatted_config, output_dir

# Setup environment
config, output_dir = setup_analysis_environment(raw_config)
```

### 3. Email Notifications

```python
from past1000.utils.email import send_notification_email
from datetime import datetime
import traceback

def run_analysis_with_notification(analysis_func, *args, **kwargs):
    """Run analysis with email notification."""

    start_time = datetime.now()

    try:
        # Run analysis
        result = analysis_func(*args, **kwargs)

        # Send success notification
        send_notification_email(
            success=True,
            start_time=start_time
        )

        return result

    except Exception as e:
        # Send error notification
        error_msg = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        send_notification_email(
            success=False,
            error_msg=error_msg,
            start_time=start_time
        )
        raise

# Use with notification
result = run_analysis_with_notification(my_analysis_function, data)
```

### 4. Custom Plotting

```python
from past1000.utils.plot import plot_confusion_matrix, heatmap_with_annot
import matplotlib.pyplot as plt

def create_analysis_dashboard(data, results):
    """Create comprehensive analysis dashboard."""

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Confusion matrix
    plot_confusion_matrix(
        results["confusion_matrix"],
        title="Classification Results",
        ax=axes[0, 0]
    )

    # Correlation heatmap
    heatmap_with_annot(
        results["correlation_matrix"],
        p_value=results["significance_matrix"],
        ax=axes[0, 1]
    )

    # Time series plot
    axes[1, 0].plot(data.index, data.values, label="Data")
    axes[1, 0].set_title("Time Series")
    axes[1, 0].legend()

    # Distribution plot
    axes[1, 1].hist(data.values, bins=30, alpha=0.7)
    axes[1, 1].set_title("Data Distribution")

    plt.tight_layout()
    return fig

# Create dashboard
dashboard = create_analysis_dashboard(time_series, analysis_results)
plt.show()
```

### 5. Type-Safe Operations

```python
from past1000.utils.types import Stages, CorrFunc, FilterSide
from typing import Union, List

def type_safe_analysis(
    data1: pd.Series,
    data2: pd.Series,
    time_range: Stages,
    corr_method: CorrFunc = "kendall",
    filter_side: FilterSide = "right"
) -> dict:
    """Type-safe analysis function."""

    # Type checking is handled by the type annotations
    # This ensures proper usage of the function

    # Perform analysis
    r, p, n = calc_corr(data1, data2, how=corr_method)

    return {
        "correlation": r,
        "p_value": p,
        "samples": n,
        "time_range": time_range,
        "method": corr_method
    }

# Use with type safety
result = type_safe_analysis(
    series1,
    series2,
    time_range="1800-1900",
    corr_method="spearman",
    filter_side="both"
)
```

## Error Handling

### Robust Utility Functions

```python
def robust_calc_corr(data1, data2, how="pearson", max_retries=3):
    """Robust correlation calculation with error handling."""

    for attempt in range(max_retries):
        try:
            # Clean data
            clean_data1 = data1.dropna()
            clean_data2 = data2.dropna()

            # Align data
            aligned_data1, aligned_data2 = clean_data1.align(clean_data2, join='inner')

            # Calculate correlation
            r, p, n = calc_corr(aligned_data1, aligned_data2, how=how)

            return r, p, n

        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Correlation calculation failed: {e}")
                return np.nan, np.nan, 0
            else:
                print(f"Attempt {attempt + 1} failed: {e}")
                continue

# Use robust function
r, p, n = robust_calc_corr(problematic_series1, problematic_series2)
```

## Performance Optimization

### Caching Utilities

```python
from functools import lru_cache
from past1000.utils.calc import calc_corr

@lru_cache(maxsize=128)
def cached_calc_corr(data1_hash, data2_hash, how):
    """Cached correlation calculation."""
    # This would need to be implemented with actual data
    # For demonstration purposes
    pass

def efficient_correlation_analysis(datasets, methods):
    """Efficient correlation analysis with caching."""

    results = []

    for i, data1 in enumerate(datasets):
        for j, data2 in enumerate(datasets[i+1:], i+1):
            for method in methods:
                # Use cached calculation
                r, p, n = cached_calc_corr(
                    hash(tuple(data1.values)),
                    hash(tuple(data2.values)),
                    method
                )

                results.append({
                    "dataset1": i,
                    "dataset2": j,
                    "method": method,
                    "correlation": r,
                    "p_value": p
                })

    return pd.DataFrame(results)
```

## Integration Examples

### With Data Module

```python
from past1000.data import HistoricalRecords
from past1000.utils.calc import calc_corr
from past1000.utils.plot import plot_correlation_windows

# Load data
history = HistoricalRecords("data/regions.shp", "data/historical_data.xlsx")

# Calculate correlation
r, p, n = calc_corr(history.data.mean(axis=1), climate_series)

# Plot results
ax = plot_correlation_windows([r], [r], [2020], ["Historical vs Climate"])
plt.show()
```

### With Comparison Module

```python
from past1000.compare import experiment_corr_2d
from past1000.utils.plot import heatmap_with_annot

# Run experiment
corr_df, r_benchmark, ax = experiment_corr_2d(data1, data2)

# Enhance plot
heatmap_with_annot(corr_df.values, ax=ax)
plt.show()
```

### With Calibration Module

```python
from past1000.calibration import MismatchReport
from past1000.utils.plot import plot_confusion_matrix

# Create report
report = MismatchReport(pred, true)

# Plot confusion matrix
ax = plot_confusion_matrix(report.cm_df, title="Classification Results")
plt.show()
```
