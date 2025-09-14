# Constants Module

The `shifting_baseline.constants` module provides essential constants and configuration values used throughout the Shifting Baseline library.

## Overview

This module defines:
- Classification thresholds and levels
- Color schemes for visualization
- Time period definitions
- Probability distributions
- Default parameter values

## Classification Constants

### Grade Values

```python
from shifting_baseline.constants import GRADE_VALUES

# Original grade values (1-5 scale)
print(GRADE_VALUES)  # [5, 4, 3, 2, 1]
```

### Standard Deviation Thresholds

```python
from shifting_baseline.constants import STD_THRESHOLDS

# Standard deviation thresholds for classification
print(STD_THRESHOLDS)  # [-1.17, -0.33, 0, 0.33, 1.17]
```

### Classification Levels

```python
from shifting_baseline.constants import LEVELS

# Classification levels (-2 to 2)
print(LEVELS)  # [-2, -1, 0, 1, 2]
```

### Level Probabilities

```python
from shifting_baseline.constants import LEVELS_PROB

# Probability distribution for each level
print(LEVELS_PROB)  # [0.1, 0.25, 0.30, 0.25, 0.1]
```

### Classification Thresholds

```python
from shifting_baseline.constants import THRESHOLDS

# Thresholds for 5-level classification
print(THRESHOLDS)  # [-1.17, -0.33, 0.33, 1.17]
```

## Visualization Constants

### Color Scheme

```python
from shifting_baseline.constants import COLORS

# Color palette for visualization
print(COLORS)  # ["#EF7722", "#FAA533", "#BBDCE5", "#0BA6DF"]
```

### Tick Labels

```python
from shifting_baseline.constants import TICK_LABELS

# Labels for classification levels
print(TICK_LABELS)  # ["SD", "MD", "N", "MW", "SW"]
```

### Verbose Labels

```python
from shifting_baseline.constants import VERBOSE_LABELS

# Descriptive labels for classification levels
print(VERBOSE_LABELS)  # ["Very dry", "Moderate dry", "Normal", "Moderate wet", "Very wet"]
```

## Time Period Constants

### Historical Periods

```python
from shifting_baseline.constants import START, STAGE1, STAGE2, END, FINAL

# Historical time periods
print(f"Start: {START}")      # 1000
print(f"Stage 1: {STAGE1}")   # 1469
print(f"Stage 2: {STAGE2}")   # 1659
print(f"End: {END}")          # 1900
print(f"Final: {FINAL}")      # 2010
```

### Stage Bins

```python
from shifting_baseline.constants import STAGES_BINS

# Time period boundaries
print(STAGES_BINS)  # [1000, 1469, 1659, 1900, 2010]
```

### Stage Labels

```python
from shifting_baseline.constants import LABELS

# Labels for each stage
print(LABELS)  # ["1000-1469", "1469-1659", "1659-1900", "1900-2021"]
```

## Mapping Constants

### Level to Standard Deviation Mapping

```python
from shifting_baseline.constants import MAP

# Mapping from levels to standard deviations
print(MAP)  # {-2: -1.5, -1: -0.5, 0: 0, 1: 0.5, 2: 1.5}
```

## Agent-Based Model Constants

### Maximum Age

```python
from shifting_baseline.constants import MAX_AGE

# Maximum age for climate observers
print(MAX_AGE)  # 40
```

## Usage Examples

### 1. Classification Setup

```python
from shifting_baseline.constants import LEVELS, THRESHOLDS, TICK_LABELS
from shifting_baseline.filters import classify_series

# Use constants for classification
data = pd.Series([-1.5, -0.5, 0.0, 0.5, 1.5])
classified = classify_series(
    data,
    thresholds=THRESHOLDS,
    levels=LEVELS
)

# Map to labels
label_mapping = dict(zip(LEVELS, TICK_LABELS))
labeled = classified.map(label_mapping)
print(labeled)
```

### 2. Time Period Analysis

```python
from shifting_baseline.constants import STAGES_BINS, LABELS

# Define time periods
def get_stage_info(stage_number):
    start = STAGES_BINS[stage_number - 1]
    end = STAGES_BINS[stage_number]
    label = LABELS[stage_number - 1]
    return start, end, label

# Get stage 2 information
start, end, label = get_stage_info(2)
print(f"Stage 2: {label} ({start}-{end})")
```

### 3. Visualization Setup

```python
import matplotlib.pyplot as plt
from shifting_baseline.constants import COLORS, TICK_LABELS

# Create a color-coded plot
def plot_classification(data, classified):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Use constants for colors and labels
    for i, (level, color) in enumerate(zip(LEVELS, COLORS)):
        mask = classified == level
        ax.scatter(data[mask].index, data[mask].values,
                  c=color, label=TICK_LABELS[i])

    ax.legend()
    ax.set_ylabel('Value')
    ax.set_xlabel('Time')
    return fig

# Use the plotting function
fig = plot_classification(time_series, classified_data)
plt.show()
```

### 4. Probability Weighted Analysis

```python
from shifting_baseline.constants import LEVELS_PROB, MAP

# Calculate probability-weighted values
def probability_weighted_value(level):
    """Calculate probability-weighted value for a level."""
    level_index = LEVELS.index(level)
    probability = LEVELS_PROB[level_index]
    std_value = MAP[level]
    return std_value * probability

# Apply to classified data
weighted_values = classified_data.map(probability_weighted_value)
print(weighted_values)
```

### 5. Agent-Based Model Configuration

```python
from shifting_baseline.constants import MAX_AGE
from shifting_baseline.abm import ClimateObservingModel

# Use constants in model configuration
model_config = {
    "years": 100,
    "max_age": MAX_AGE,
    "new_agents": 10,
    "min_age": 10
}

# Create model with constants
model = ClimateObservingModel(**model_config)
```

## Custom Constants

### Creating Custom Classification Schemes

```python
# Define custom classification constants
CUSTOM_THRESHOLDS = [-1.0, -0.5, 0.5, 1.0]
CUSTOM_LEVELS = [-2, -1, 0, 1, 2]
CUSTOM_LABELS = ["Very Dry", "Dry", "Normal", "Wet", "Very Wet"]
CUSTOM_COLORS = ["#8B0000", "#FF4500", "#FFFF00", "#00FF00", "#0000FF"]

# Use custom constants
def custom_classify(data):
    return classify_series(
        data,
        thresholds=CUSTOM_THRESHOLDS,
        levels=CUSTOM_LEVELS
    )
```

### Regional Constants

```python
# Define regional-specific constants
REGIONAL_CONSTANTS = {
    "华北地区": {
        "thresholds": [-1.17, -0.33, 0.33, 1.17],
        "levels": [-2, -1, 0, 1, 2],
        "labels": ["严重干旱", "中度干旱", "正常", "中度湿润", "严重湿润"]
    },
    "华南地区": {
        "thresholds": [-1.0, -0.3, 0.3, 1.0],
        "levels": [-2, -1, 0, 1, 2],
        "labels": ["严重干旱", "中度干旱", "正常", "中度湿润", "严重湿润"]
    }
}

# Use regional constants
def regional_classify(data, region):
    constants = REGIONAL_CONSTANTS[region]
    return classify_series(
        data,
        thresholds=constants["thresholds"],
        levels=constants["levels"]
    )
```

## Validation Functions

### Validate Constants

```python
def validate_constants():
    """Validate that constants are consistent."""

    # Check thresholds and levels
    assert len(LEVELS) == len(THRESHOLDS) + 1, "Levels and thresholds mismatch"
    assert len(LEVELS) == len(TICK_LABELS), "Levels and labels mismatch"
    assert len(LEVELS) == len(VERBOSE_LABELS), "Levels and verbose labels mismatch"

    # Check probabilities
    assert abs(sum(LEVELS_PROB) - 1.0) < 1e-6, "Probabilities don't sum to 1"
    assert len(LEVELS_PROB) == len(LEVELS), "Probabilities and levels mismatch"

    # Check mapping
    assert all(level in MAP for level in LEVELS), "Missing level in mapping"

    # Check time periods
    assert START < STAGE1 < STAGE2 < END < FINAL, "Time periods not ordered"
    assert len(STAGES_BINS) == len(LABELS) + 1, "Stage bins and labels mismatch"

    print("All constants are valid")
    return True

# Validate constants
validate_constants()
```

### Check Consistency

```python
def check_consistency():
    """Check consistency between related constants."""

    # Check threshold ordering
    for i in range(len(THRESHOLDS) - 1):
        assert THRESHOLDS[i] < THRESHOLDS[i + 1], f"Thresholds not ordered: {THRESHOLDS}"

    # Check level ordering
    for i in range(len(LEVELS) - 1):
        assert LEVELS[i] < LEVELS[i + 1], f"Levels not ordered: {LEVELS}"

    # Check mapping consistency
    for level in LEVELS:
        assert level in MAP, f"Level {level} missing from mapping"

    print("Constants are consistent")
    return True

# Check consistency
check_consistency()
```

## Export Constants

### Export to Configuration

```python
def export_constants_to_config():
    """Export constants to a configuration dictionary."""

    config = {
        "classification": {
            "levels": LEVELS,
            "thresholds": THRESHOLDS,
            "probabilities": LEVELS_PROB,
            "labels": TICK_LABELS,
            "verbose_labels": VERBOSE_LABELS
        },
        "visualization": {
            "colors": COLORS,
            "tick_labels": TICK_LABELS
        },
        "time_periods": {
            "start": START,
            "stage1": STAGE1,
            "stage2": STAGE2,
            "end": END,
            "final": FINAL,
            "bins": STAGES_BINS,
            "labels": LABELS
        },
        "mapping": MAP,
        "abm": {
            "max_age": MAX_AGE
        }
    }

    return config

# Export constants
constants_config = export_constants_to_config()
print(constants_config)
```

## Integration Examples

### With Data Module

```python
from shifting_baseline.data import HistoricalRecords
from shifting_baseline.constants import STAGES_BINS, LABELS

# Use constants in data loading
history = HistoricalRecords(
    shp_path="data/regions.shp",
    data_path="data/historical_data.xlsx"
)

# Get stage information
for i, (start, end) in enumerate(zip(STAGES_BINS[:-1], STAGES_BINS[1:])):
    stage_data = history.period(slice(start, end))
    print(f"{LABELS[i]}: {len(stage_data)} years")
```

### With Comparison Module

```python
from shifting_baseline.compare import experiment_corr_2d
from shifting_baseline.constants import COLORS

# Use constants in visualization
corr_df, r_benchmark, ax = experiment_corr_2d(
    data1=historical_series,
    data2=climate_series
)

# Apply color scheme
ax.set_prop_cycle(color=COLORS)
```

### With Calibration Module

```python
from shifting_baseline.calibration import MismatchReport
from shifting_baseline.constants import TICK_LABELS

# Use constants in calibration
report = MismatchReport(
    pred=predicted_levels,
    true=observed_levels,
    labels=TICK_LABELS
)
```
