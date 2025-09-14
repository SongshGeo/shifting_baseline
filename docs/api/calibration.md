# Calibration Module

The `past1000.calibration` module provides comprehensive tools for evaluating the accuracy and reliability of climate reconstruction data through mismatch analysis and statistical validation.

## Overview

This module handles:
- Mismatch analysis between predicted and observed data
- Statistical significance testing
- Error pattern analysis
- Confusion matrix generation
- Monte Carlo validation
- Visualization of calibration results

## Core Classes

### MismatchReport

The main class for analyzing mismatches between predicted and observed climate data.

```python
from past1000.calibration import MismatchReport
```

#### Constructor

```python
MismatchReport(
    pred: pd.Series,
    true: pd.Series,
    labels: list[str] | None = None,
    value_series: pd.Series | None = None
)
```

**Parameters:**
- `pred`: Predicted/classified data series (level values: -2, -1, 0, 1, 2)
- `true`: True/observed data series (level values: -2, -1, 0, 1, 2)
- `labels`: Level labels (default: ["SD", "MD", "N", "MW", "SW"])
- `value_series`: Original continuous values for error pattern analysis

**Example:**
```python
from past1000.calibration import MismatchReport
from past1000.filters import classify

# Create mismatch report
report = MismatchReport(
    pred=classified_predictions,
    true=classified_observations,
    value_series=raw_climate_data
)
```

#### Key Methods

##### `analyze_error_patterns(value_series=None, mc_runs=1000)`

Analyze patterns in classification errors.

**Parameters:**
- `value_series`: Original continuous values (if not provided in constructor)
- `mc_runs`: Number of Monte Carlo simulations

**Returns:**
- `pd.DataFrame`: Error analysis matrix

**Example:**
```python
# Analyze error patterns
error_df = report.analyze_error_patterns(mc_runs=2000)

# Display error analysis
print(error_df[['value', 'pred', 'true', 'exact', 'diff']])
```

##### `get_statistics_summary(as_str=False, weights="quadratic")`

Get comprehensive statistical summary.

**Parameters:**
- `as_str`: Whether to return string format
- `weights`: Weight type for Cohen's kappa ("quadratic", "linear", "none")

**Returns:**
- `str` or `dict`: Statistical summary

**Example:**
```python
# Get statistics as dictionary
stats = report.get_statistics_summary()
print(f"Kappa: {stats['kappa']:.3f}")
print(f"Accuracy: {stats['accuracy']:.3f}")

# Get statistics as string
stats_str = report.get_statistics_summary(as_str=True)
print(stats_str)  # "Kappa: 0.45, Kendall's Tau: 0.32**"
```

##### `plot_confusion_matrix(ax=None, title=None)`

Plot confusion matrix.

**Parameters:**
- `ax`: Matplotlib axes (optional)
- `title`: Plot title (optional)

**Returns:**
- `plt.Axes`: Plot axes

**Example:**
```python
import matplotlib.pyplot as plt

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))
report.plot_confusion_matrix(ax=ax, title="Classification Results")
plt.show()
```

##### `plot_mismatch_analysis(ax=None)`

Plot mismatch analysis visualization.

**Parameters:**
- `ax`: Matplotlib axes (optional)

**Returns:**
- `plt.Axes`: Plot axes

**Note:** Requires prior call to `analyze_error_patterns()`

**Example:**
```python
# First analyze errors
report.analyze_error_patterns()

# Then plot analysis
fig, ax = plt.subplots(figsize=(10, 6))
report.plot_mismatch_analysis(ax=ax)
plt.show()
```

##### `generate_report_figure(figsize=(5, 3), save_path=None, **kwargs)`

Generate complete report figure.

**Parameters:**
- `figsize`: Figure size tuple
- `save_path`: Path to save figure
- `**kwargs`: Additional plotting parameters

**Returns:**
- `plt.Figure`: Complete report figure

**Example:**
```python
# Generate and save complete report
fig = report.generate_report_figure(
    figsize=(12, 6),
    save_path="mismatch_report.png"
)
```

#### Properties

##### `n_samples`
Number of valid samples after cleaning.

##### `n_mismatches`
Number of mismatched classifications.

##### `false_count_matrix`
Matrix showing count of false classifications.

##### `error_analyzed`
Boolean indicating if error analysis has been performed.

**Example:**
```python
print(f"Total samples: {report.n_samples}")
print(f"Mismatches: {report.n_mismatches}")
print(f"Error analysis done: {report.error_analyzed}")
```

## Statistical Methods

### Cohen's Kappa

Measures inter-rater agreement for categorical data.

```python
# Get kappa with different weights
kappa_quadratic = report.get_statistics_summary(weights="quadratic")
kappa_linear = report.get_statistics_summary(weights="linear")
kappa_none = report.get_statistics_summary(weights="none")
```

### Kendall's Tau

Measures rank correlation between predicted and true values.

```python
stats = report.get_statistics_summary()
tau = stats['kendall_tau']
tau_p_value = stats['tau_p_value']
```

### Accuracy

Simple accuracy measure (exact matches / total samples).

```python
accuracy = report.get_statistics_summary()['accuracy']
```

## Monte Carlo Validation

The module includes Monte Carlo simulation for statistical validation:

```python
# Run Monte Carlo validation
error_df = report.analyze_error_patterns(mc_runs=1000)

# Access p-values from Monte Carlo
p_values = report.p_value_matrix
print("Significant error patterns:")
print(p_values[p_values < 0.05])
```

## Advanced Analysis Patterns

### 1. Comprehensive Calibration Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt
from past1000.calibration import MismatchReport
from past1000.filters import classify

def comprehensive_calibration_analysis(predicted, observed, raw_values):
    """Perform comprehensive calibration analysis."""

    # Create mismatch report
    report = MismatchReport(
        pred=predicted,
        true=observed,
        value_series=raw_values
    )

    # Analyze error patterns
    error_df = report.analyze_error_patterns(mc_runs=2000)

    # Get statistics
    stats = report.get_statistics_summary()

    # Generate visualizations
    fig = report.generate_report_figure(figsize=(15, 8))

    return {
        'report': report,
        'error_df': error_df,
        'statistics': stats,
        'figure': fig
    }

# Usage
results = comprehensive_calibration_analysis(
    predicted=classified_predictions,
    observed=classified_observations,
    raw_values=raw_climate_data
)
```

### 2. Time Series Calibration

```python
def time_series_calibration(historical_series, climate_series, window_size=50):
    """Analyze calibration over time."""

    results = []

    for i in range(0, len(historical_series) - window_size, window_size):
        # Extract window
        hist_window = historical_series.iloc[i:i+window_size]
        clim_window = climate_series.iloc[i:i+window_size]

        # Classify data
        hist_classified = classify(hist_window)
        clim_classified = classify(clim_window)

        # Create report
        report = MismatchReport(
            pred=hist_classified,
            true=clim_classified,
            value_series=clim_window
        )

        # Get statistics
        stats = report.get_statistics_summary()

        results.append({
            'start_year': hist_window.index[0],
            'end_year': hist_window.index[-1],
            'kappa': stats['kappa'],
            'accuracy': stats['accuracy'],
            'n_samples': stats['n_samples']
        })

    return pd.DataFrame(results)

# Analyze calibration over time
calibration_timeline = time_series_calibration(
    historical_series,
    climate_series,
    window_size=100
)
```

### 3. Multi-Model Comparison

```python
def compare_models(model_predictions, observed_data, model_names):
    """Compare multiple models using calibration metrics."""

    comparison_results = []

    for model_name, predictions in zip(model_names, model_predictions):
        # Create report
        report = MismatchReport(
            pred=predictions,
            true=observed_data
        )

        # Get statistics
        stats = report.get_statistics_summary()

        comparison_results.append({
            'model': model_name,
            'kappa': stats['kappa'],
            'accuracy': stats['accuracy'],
            'kendall_tau': stats['kendall_tau'],
            'n_samples': stats['n_samples']
        })

    return pd.DataFrame(comparison_results)

# Compare different models
models = ['Model A', 'Model B', 'Model C']
predictions = [pred_a, pred_b, pred_c]

comparison_df = compare_models(predictions, observed_data, models)
print(comparison_df.sort_values('kappa', ascending=False))
```

### 4. Sensitivity Analysis

```python
def calibration_sensitivity_analysis(data, thresholds_list):
    """Analyze calibration sensitivity to different thresholds."""

    results = []

    for i, thresholds in enumerate(thresholds_list):
        # Classify with different thresholds
        pred_classified = classify(data['predicted'], thresholds=thresholds)
        true_classified = classify(data['observed'], thresholds=thresholds)

        # Create report
        report = MismatchReport(
            pred=pred_classified,
            true=true_classified
        )

        # Get statistics
        stats = report.get_statistics_summary()

        results.append({
            'threshold_set': i,
            'thresholds': thresholds,
            'kappa': stats['kappa'],
            'accuracy': stats['accuracy']
        })

    return pd.DataFrame(results)

# Test different threshold sets
threshold_sets = [
    [-1.0, -0.5, 0.5, 1.0],
    [-1.5, -0.5, 0.5, 1.5],
    [-1.2, -0.3, 0.3, 1.2]
]

sensitivity_results = calibration_sensitivity_analysis(data, threshold_sets)
```

## Visualization Examples

### 1. Confusion Matrix Heatmap

```python
import seaborn as sns
import matplotlib.pyplot as plt

def plot_confusion_heatmap(report, title="Confusion Matrix"):
    """Create a detailed confusion matrix heatmap."""

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot heatmap
    sns.heatmap(
        report.cm_df.values,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=report.cm_df.columns,
        yticklabels=report.cm_df.index,
        ax=ax
    )

    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

    return fig

# Create confusion matrix
fig = plot_confusion_heatmap(report, "Climate Classification Results")
plt.show()
```

### 2. Error Pattern Analysis

```python
def plot_error_patterns(report):
    """Visualize error patterns from mismatch analysis."""

    if not report.error_analyzed:
        report.analyze_error_patterns()

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot difference matrix
    im1 = axes[0].imshow(report.diff_matrix.values, cmap='RdBu_r', aspect='auto')
    axes[0].set_title('Error Pattern Differences')
    axes[0].set_xlabel('True Level')
    axes[0].set_ylabel('Predicted Level')
    plt.colorbar(im1, ax=axes[0])

    # Plot p-value matrix
    im2 = axes[1].imshow(report.p_value_matrix.values, cmap='viridis', aspect='auto')
    axes[1].set_title('Statistical Significance (p-values)')
    axes[1].set_xlabel('True Level')
    axes[1].set_ylabel('Predicted Level')
    plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    return fig

# Plot error patterns
fig = plot_error_patterns(report)
plt.show()
```

## Integration with Other Modules

### With Comparison Module

```python
from past1000.compare import compare_corr
from past1000.calibration import MismatchReport

# Calculate correlation
r, p, n = compare_corr(historical_series, climate_series)

# Create calibration report
report = MismatchReport(
    pred=classified_historical,
    true=classified_climate
)

# Compare correlation with calibration
print(f"Correlation: {r:.3f}")
print(f"Calibration Kappa: {report.get_statistics_summary()['kappa']:.3f}")
```

### With Filters Module

```python
from past1000.filters import classify
from past1000.calibration import MismatchReport

# Classify data with different methods
pred_classified = classify(predictions, handle_na="skip")
true_classified = classify(observations, handle_na="skip")

# Create calibration report
report = MismatchReport(
    pred=pred_classified,
    true=true_classified,
    value_series=raw_predictions
)
```

## Performance Optimization

### Memory Management

For large datasets, use chunked processing:

```python
def chunked_calibration_analysis(data, chunk_size=1000):
    """Process large datasets in chunks."""

    results = []

    for i in range(0, len(data), chunk_size):
        chunk = data.iloc[i:i+chunk_size]

        # Process chunk
        report = MismatchReport(
            pred=chunk['predicted'],
            true=chunk['observed']
        )

        stats = report.get_statistics_summary()
        results.append(stats)

    return pd.DataFrame(results)
```

### Parallel Processing

```python
from multiprocessing import Pool
from functools import partial

def parallel_calibration_analysis(datasets, n_processes=4):
    """Run calibration analysis in parallel."""

    def analyze_dataset(data):
        report = MismatchReport(
            pred=data['predicted'],
            true=data['observed']
        )
        return report.get_statistics_summary()

    with Pool(n_processes) as pool:
        results = pool.map(analyze_dataset, datasets)

    return pd.DataFrame(results)
```
