# Filters Module

The `past1000.filters` module provides data filtering, classification, and preprocessing functions for climate reconstruction analysis.

## Overview

This module handles:
- Statistical filtering of time series data
- Classification of continuous values into discrete levels
- Standard deviation calculations
- Data preprocessing and cleaning
- Threshold-based categorization

## Core Functions

### `calc_std_deviation(series)`

Calculate the standard deviation of the last value relative to the past window years.

**Parameters:**
- `series`: A pandas Series with time index

**Returns:**
- `float`: The number of standard deviations the last value is from the mean

**Raises:**
- `ValueError`: If series is empty, has only one value, or window is larger than series length

**Example:**
```python
from past1000.filters import calc_std_deviation
import pandas as pd

# Calculate standard deviation
series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
std_dev = calc_std_deviation(series)
print(f"Last value is {std_dev:.2f} standard deviations from mean")
```

### `classify_single_value(value, thresholds=None, levels=None)`

Classify a single value based on standard deviation thresholds.

**Parameters:**
- `value`: The numeric value to classify
- `thresholds`: List of threshold values (default: [-1.17, -0.33, 0.33, 1.17])
- `levels`: List of classification levels (default: [-2, -1, 0, 1, 2])

**Returns:**
- `int`: The classification level for the input value

**Raises:**
- `ValueError`: If levels length is not thresholds length + 1, if thresholds are not in ascending order, or if input value is NaN or infinite
- `TypeError`: If value is not a numeric type

**Example:**
```python
from past1000.filters import classify_single_value

# Classify single values
print(classify_single_value(-1.5))  # -2 (severe drought)
print(classify_single_value(0.0))   # 0 (normal)
print(classify_single_value(1.5))   # 2 (severe wet)

# Custom thresholds
custom_thresholds = [-1.0, -0.5, 0.5, 1.0]
custom_levels = [-2, -1, 0, 1, 2]
level = classify_single_value(0.3, thresholds=custom_thresholds, levels=custom_levels)
```

### `classify_series(series, thresholds=None, levels=None, handle_na="raise")`

Classify values in a series based on standard deviation thresholds.

**Parameters:**
- `series`: Input data to classify (Series or array)
- `thresholds`: List of threshold values (default: [-1.17, -0.33, 0.33, 1.17])
- `levels`: List of classification levels (default: [-2, -1, 0, 1, 2])
- `handle_na`: Strategy for handling NaN values ("raise", "skip", "fill")

**Returns:**
- `pd.Series`: Series with classification labels

**Raises:**
- `ValueError`: If levels length is not thresholds length + 1, if thresholds are not in ascending order, if series is empty, or if NaN values are encountered with handle_na="raise"
- `TypeError`: If series contains non-numeric data types

**Example:**
```python
from past1000.filters import classify_series
import pandas as pd
import numpy as np

# Basic classification
data = pd.Series([-1.5, -0.5, 0.0, 0.5, 1.5])
classified = classify_series(data)
print(classified)  # [-2, -1, 0, 1, 2]

# Handle NaN values
data_with_na = pd.Series([-1.5, np.nan, 0.5])
classified_na = classify_series(data_with_na, handle_na="skip")
print(classified_na)  # [-2, <NA>, 1]

# Custom thresholds
custom_thresholds = [-1.0, -0.5, 0.5, 1.0]
classified_custom = classify_series(data, thresholds=custom_thresholds)
```

### `classify(series, thresholds=None, levels=None, handle_na="raise")`

Backward compatibility alias for `classify_series`.

**Note:** This function is maintained for backward compatibility. For new code, consider using `classify_series()` for better parameter control and error handling options.

**Example:**
```python
from past1000.filters import classify

# Same as classify_series
classified = classify(data, handle_na="skip")
```

## Classification Levels

The module uses a standardized 5-level classification system:

| Level | Label | Description | Threshold Range |
|-------|-------|-------------|-----------------|
| -2 | SD | Severe Drought | x ≤ -1.17 |
| -1 | MD | Moderate Drought | -1.17 < x ≤ -0.33 |
| 0 | N | Normal | -0.33 < x ≤ 0.33 |
| 1 | MW | Moderate Wet | 0.33 < x ≤ 1.17 |
| 2 | SW | Severe Wet | x > 1.17 |

## Advanced Usage Patterns

### 1. Custom Classification Schemes

```python
def create_custom_classifier(thresholds, levels, labels):
    """Create a custom classification function."""

    def classify_custom(series, handle_na="raise"):
        return classify_series(
            series,
            thresholds=thresholds,
            levels=levels,
            handle_na=handle_na
        )

    return classify_custom

# Create a 3-level classifier
three_level_thresholds = [-0.5, 0.5]
three_levels = [-1, 0, 1]
three_level_labels = ["Dry", "Normal", "Wet"]

classify_three_level = create_custom_classifier(
    three_level_thresholds,
    three_levels,
    three_level_labels
)

# Use the custom classifier
data = pd.Series([-1.0, 0.0, 1.0])
classified = classify_three_level(data)
print(classified)  # [-1, 0, 1]
```

### 2. Rolling Window Classification

```python
def rolling_classification(series, window=30, step=1):
    """Apply classification to rolling windows."""

    results = []

    for i in range(0, len(series) - window + 1, step):
        window_data = series.iloc[i:i+window]

        # Calculate statistics for the window
        window_mean = window_data.mean()
        window_std = window_data.std()

        # Standardize the last value
        if window_std > 0:
            z_score = (window_data.iloc[-1] - window_mean) / window_std
            classified_level = classify_single_value(z_score)
        else:
            classified_level = 0  # Normal if no variation

        results.append({
            'index': series.index[i + window - 1],
            'value': window_data.iloc[-1],
            'z_score': z_score if window_std > 0 else 0,
            'level': classified_level
        })

    return pd.DataFrame(results)

# Apply rolling classification
rolling_results = rolling_classification(climate_series, window=50, step=10)
```

### 3. Multi-Threshold Analysis

```python
def multi_threshold_analysis(data, threshold_sets):
    """Analyze data with multiple threshold sets."""

    results = {}

    for name, (thresholds, levels) in threshold_sets.items():
        classified = classify_series(data, thresholds=thresholds, levels=levels)

        # Calculate statistics
        level_counts = classified.value_counts().sort_index()
        level_proportions = level_counts / len(classified)

        results[name] = {
            'classified': classified,
            'counts': level_counts,
            'proportions': level_proportions
        }

    return results

# Define multiple threshold sets
threshold_sets = {
    'standard': ([-1.17, -0.33, 0.33, 1.17], [-2, -1, 0, 1, 2]),
    'conservative': ([-1.5, -0.5, 0.5, 1.5], [-2, -1, 0, 1, 2]),
    'sensitive': ([-0.8, -0.2, 0.2, 0.8], [-2, -1, 0, 1, 2])
}

# Analyze with multiple thresholds
multi_results = multi_threshold_analysis(climate_data, threshold_sets)

# Compare results
for name, result in multi_results.items():
    print(f"{name}: {result['proportions']}")
```

### 4. Data Quality Assessment

```python
def assess_classification_quality(series, thresholds=None, levels=None):
    """Assess the quality of classification results."""

    if thresholds is None:
        thresholds = [-1.17, -0.33, 0.33, 1.17]
    if levels is None:
        levels = [-2, -1, 0, 1, 2]

    # Classify the data
    classified = classify_series(series, thresholds=thresholds, levels=levels)

    # Calculate quality metrics
    level_counts = classified.value_counts().sort_index()
    level_proportions = level_counts / len(classified)

    # Check for extreme values
    extreme_dry = level_proportions.get(-2, 0)
    extreme_wet = level_proportions.get(2, 0)
    normal = level_proportions.get(0, 0)

    # Calculate entropy (measure of distribution uniformity)
    entropy = -sum(p * np.log2(p) for p in level_proportions if p > 0)

    quality_metrics = {
        'total_samples': len(classified),
        'level_distribution': level_proportions,
        'extreme_dry_proportion': extreme_dry,
        'extreme_wet_proportion': extreme_wet,
        'normal_proportion': normal,
        'entropy': entropy,
        'is_balanced': 0.1 < normal < 0.6,  # Heuristic for balanced distribution
        'has_extremes': extreme_dry > 0.05 or extreme_wet > 0.05
    }

    return classified, quality_metrics

# Assess classification quality
classified_data, quality = assess_classification_quality(climate_series)
print(f"Classification quality: {quality}")
```

## Integration with Other Modules

### With Data Module

```python
from past1000.data import HistoricalRecords
from past1000.filters import classify

# Load historical data
history = HistoricalRecords(
    shp_path="data/regions.shp",
    data_path="data/historical_data.xlsx"
)

# Classify historical data
historical_classified = classify(history.data.mean(axis=1))

# Classify climate data
climate_classified = classify(climate_series)
```

### With Comparison Module

```python
from past1000.compare import compare_corr
from past1000.filters import calc_std_deviation, classify

# Use filter functions in correlation analysis
r, p, n = compare_corr(
    series1,
    series2,
    filter_func=calc_std_deviation,
    filter_side="right"
)

# Classify data for further analysis
classified1 = classify(series1)
classified2 = classify(series2)
```

### With Calibration Module

```python
from past1000.calibration import MismatchReport
from past1000.filters import classify

# Classify data for calibration analysis
pred_classified = classify(predictions)
true_classified = classify(observations)

# Create mismatch report
report = MismatchReport(
    pred=pred_classified,
    true=true_classified,
    value_series=raw_predictions
)
```

## Error Handling and Validation

### Input Validation

```python
def validate_classification_inputs(series, thresholds, levels):
    """Validate inputs for classification functions."""

    # Check series
    if len(series) == 0:
        raise ValueError("Cannot classify empty series")

    if not pd.api.types.is_numeric_dtype(series):
        raise TypeError("Series must contain numeric data")

    # Check thresholds and levels
    if len(levels) != len(thresholds) + 1:
        raise ValueError(
            f"Levels must be one element longer than thresholds. "
            f"Got {len(levels)} levels and {len(thresholds)} thresholds"
        )

    # Check threshold ordering
    if not all(thresholds[i] < thresholds[i + 1] for i in range(len(thresholds) - 1)):
        raise ValueError("Thresholds must be in strictly ascending order")

    return True

# Use validation
try:
    validate_classification_inputs(data, thresholds, levels)
    classified = classify_series(data, thresholds=thresholds, levels=levels)
except (ValueError, TypeError) as e:
    print(f"Validation error: {e}")
```

### Robust Classification

```python
def robust_classify(series, thresholds=None, levels=None, handle_na="skip", max_retries=3):
    """Robust classification with error handling and retries."""

    if thresholds is None:
        thresholds = [-1.17, -0.33, 0.33, 1.17]
    if levels is None:
        levels = [-2, -1, 0, 1, 2]

    for attempt in range(max_retries):
        try:
            # Clean data
            clean_series = series.copy()

            # Handle infinite values
            clean_series = clean_series.replace([np.inf, -np.inf], np.nan)

            # Classify
            result = classify_series(
                clean_series,
                thresholds=thresholds,
                levels=levels,
                handle_na=handle_na
            )

            return result

        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Classification failed after {max_retries} attempts: {e}")
                # Return a default classification
                return pd.Series([0] * len(series), index=series.index)
            else:
                print(f"Attempt {attempt + 1} failed: {e}")
                continue

# Use robust classification
robust_result = robust_classify(problematic_data)
```

## Performance Optimization

### Vectorized Operations

```python
def vectorized_classification(series, thresholds, levels):
    """Optimized vectorized classification."""

    # Convert to numpy for faster operations
    values = series.values
    n = len(values)

    # Initialize result array
    result = np.full(n, levels[0], dtype=int)

    # Vectorized threshold application
    for i, threshold in enumerate(thresholds, 1):
        result[values > threshold] = levels[i]

    return pd.Series(result, index=series.index)

# Use vectorized classification for large datasets
large_series = pd.Series(np.random.randn(100000))
classified_large = vectorized_classification(
    large_series,
    thresholds=[-1.17, -0.33, 0.33, 1.17],
    levels=[-2, -1, 0, 1, 2]
)
```

### Memory-Efficient Processing

```python
def chunked_classification(series, chunk_size=10000, **kwargs):
    """Process large series in chunks to manage memory."""

    results = []

    for i in range(0, len(series), chunk_size):
        chunk = series.iloc[i:i+chunk_size]
        chunk_classified = classify_series(chunk, **kwargs)
        results.append(chunk_classified)

    return pd.concat(results, ignore_index=False)

# Process large dataset in chunks
large_classified = chunked_classification(
    very_large_series,
    chunk_size=50000,
    handle_na="skip"
)
```
