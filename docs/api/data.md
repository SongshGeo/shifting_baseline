# Data Module

The `past1000.data` module provides comprehensive data loading, processing, and management capabilities for historical climate reconstruction analysis.

## Overview

This module handles:
- Loading and standardizing climate reconstruction data
- Processing historical records from various sources
- Data validation and quality control
- Integration of multiple data sources

## Core Classes

### HistoricalRecords

The main class for handling historical drought/flood records from Chinese historical documents.

```python
from past1000.data import HistoricalRecords
```

#### Constructor

```python
HistoricalRecords(
    shp_path: PathLike,
    data_path: PathLike,
    region: Region | None = "华北地区",
    symmetrical_level: bool = True,
    to_std: Optional[ToStdMethod] = None
)
```

**Parameters:**
- `shp_path`: Path to shapefile containing regional boundaries
- `data_path`: Path to Excel file containing historical records
- `region`: Target region for analysis (default: "华北地区")
- `symmetrical_level`: Whether to convert to symmetrical level scale
- `to_std`: Standardization method ("mapping", "sampling", or None)

**Supported Regions:**
- 华北地区 (North China)
- 东北地区 (Northeast China)
- 华东地区 (East China)
- 华中地区 (Central China)
- 华南地区 (South China)
- 西南地区 (Southwest China)
- 西北地区 (Northwest China)

#### Key Methods

##### `aggregate(how, inplace=False, name=None, to_int=True, weights=None, **kwargs)`

Aggregate multi-station data into a single time series.

**Parameters:**
- `how`: Aggregation method
  - `"mean"`: Simple arithmetic mean
  - `"median"`: Median value
  - `"mode"`: Most frequent value
  - `"weighted_mean"`: Weighted average
  - `"probability_weighted"`: Probability-weighted aggregation
  - `"bayesian"`: Bayesian aggregation
- `inplace`: Whether to modify data in place
- `name`: Name for the resulting series
- `to_int`: Whether to convert to integer levels
- `weights`: Weights for weighted aggregation

**Returns:**
- `pd.Series` or `HistoricalRecords`: Aggregated data

**Example:**
```python
# Simple mean aggregation
mean_series = history.aggregate("mean")

# Bayesian aggregation with spatial correlation
bayesian_series = history.aggregate(
    "bayesian",
    spatial_correlation=0.5,
    uncertainty_factor=1.0
)
```

##### `get_time_slice(stage)`

Convert stage input to a time slice for data selection.

**Parameters:**
- `stage`: Stage specification (int, slice, or str)
  - Integer: 1, 2, 3, 4 (stage indices)
  - Slice: `slice(1, 3)` for stages 1-2
  - String: "stage1", "1:3", "1000-1469", "all"

**Returns:**
- `slice`: Time slice object

**Examples:**
```python
# Different ways to specify time periods
stage1 = history.get_time_slice(1)           # 1000-1469
stage2_3 = history.get_time_slice("2:3")     # 1469-1889
custom = history.get_time_slice("1600-1800") # 1600-1800
all_data = history.get_time_slice("all")     # All data
```

##### `period(stage)`

Select data by historical stage(s) or explicit year range.

**Parameters:**
- `stage`: Stage specification (same as `get_time_slice`)

**Returns:**
- `pd.DataFrame` or `pd.Series`: Selected data

**Example:**
```python
# Get data for specific periods
stage1_data = history.period(1)
recent_data = history.period("1800-1900")
```

##### `merge_with(other, time_range="all", split=False)`

Merge historical data with other datasets.

**Parameters:**
- `other`: Other dataset (Series or DataFrame)
- `time_range`: Time range for merging
- `split`: Whether to return separate series

**Returns:**
- `pd.DataFrame` or `tuple[pd.Series, pd.Series]`: Merged data

**Example:**
```python
# Merge with climate data
merged_df = history.merge_with(climate_data, time_range="all")

# Split into separate series
hist_series, climate_series = history.merge_with(
    climate_data,
    time_range="1800-1900",
    split=True
)
```

##### `corr_with(arr2, col=None, how="pearson")`

Calculate correlation with another time series.

**Parameters:**
- `arr2`: Other time series
- `col`: Column name to use (if DataFrame)
- `how`: Correlation method ("pearson", "kendall", "spearman")

**Returns:**
- `tuple[float, float, int]`: (correlation, p-value, sample_count)

#### Advanced Aggregation Methods

##### `bayesian_aggregation(spatial_correlation=0.5, uncertainty_factor=1.0, distance_matrix=None, correlation_decay=0.1)`

Perform Bayesian aggregation considering spatial correlation and uncertainty.

**Parameters:**
- `spatial_correlation`: Spatial correlation coefficient or matrix
- `uncertainty_factor`: Uncertainty scaling factor
- `distance_matrix`: Distance matrix between stations
- `correlation_decay`: Distance decay parameter

**Returns:**
- `pd.Series`: Bayesian aggregated time series

##### `_probability_weighted_aggregation(weights=None)`

Probability-weighted aggregation based on level probabilities.

**Parameters:**
- `weights`: Station weights dictionary

**Returns:**
- `pd.Series`: Probability-weighted time series

## Utility Functions

### `load_data(cfg)`

Load natural and historical data with uncertainties.

**Parameters:**
- `cfg`: Configuration object (DictConfig)

**Returns:**
- `tuple[pd.DataFrame, pd.DataFrame, HistoricalRecords]`: (datasets, uncertainties, history)

**Example:**
```python
from past1000.data import load_data
from omegaconf import DictConfig

# Load all data
datasets, uncertainties, history = load_data(config)
```

### `load_nat_data(folder, includes, index_name="year", start_year=1000, standardize=True, end_year=2010)`

Load natural climate reconstruction data.

**Parameters:**
- `folder`: Directory containing data files
- `includes`: List of strings to match in filenames
- `index_name`: Name for the time index
- `start_year`: Starting year for data
- `standardize`: Whether to standardize data
- `end_year`: Ending year for data

**Returns:**
- `tuple[pd.DataFrame, pd.DataFrame]`: (datasets, uncertainties)

### `check_distribution(data, only_best=True)`

Check the statistical distribution of data.

**Parameters:**
- `data`: Input data (Series or DataFrame)
- `only_best`: Whether to return only the best-fitting distribution

**Returns:**
- `pd.DataFrame` or `pd.Series`: Distribution analysis results

### `load_validation_data(data_path, resolution=0.25)`

Load validation data from NetCDF files.

**Parameters:**
- `data_path`: Path to NetCDF file
- `resolution`: Spatial resolution in degrees

**Returns:**
- `xr.DataArray`: Validation data

## Data Processing Pipeline

### 1. Data Loading
```python
# Load historical records
history = HistoricalRecords(
    shp_path="data/regions.shp",
    data_path="data/historical_data.xlsx",
    region="华北地区"
)

# Load climate reconstruction data
datasets, uncertainties = load_nat_data(
    folder="data/tree_ring/",
    includes=["shi2018", "yellow2019"]
)
```

### 2. Data Standardization
```python
# Standardize historical data
history_std = HistoricalRecords(
    shp_path="data/regions.shp",
    data_path="data/historical_data.xlsx",
    to_std="mapping"  # or "sampling"
)

# Standardize climate data
from past1000.mc import standardize_both
std_data, std_uncertainty = standardize_both(raw_data)
```

### 3. Data Aggregation
```python
# Aggregate multi-station data
aggregated = history.aggregate("bayesian", spatial_correlation=0.5)

# Select specific time periods
stage_data = history.period("1469-1659")
```

### 4. Data Integration
```python
# Merge different datasets
merged_data = history.merge_with(climate_data, time_range="all")

# Calculate correlations
corr, p_val, n = history.corr_with(climate_data, how="kendall")
```

## Error Handling

The module includes comprehensive error handling for common data issues:

```python
try:
    history = HistoricalRecords(
        shp_path="data/regions.shp",
        data_path="data/historical_data.xlsx"
    )
except FileNotFoundError as e:
    print(f"Data file not found: {e}")
except ValueError as e:
    print(f"Invalid data format: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Performance Optimization

For large datasets, consider these optimization strategies:

1. **Lazy Loading**: Use xarray with dask for large NetCDF files
2. **Chunked Processing**: Process data in chunks to manage memory
3. **Caching**: Use built-in caching for frequently accessed data
4. **Parallel Processing**: Use multiprocessing for independent operations

```python
# Example: Chunked processing for large datasets
from past1000.process import load_and_combine_datasets_chunked

# Load data in chunks
combined_data = load_and_combine_datasets_chunked(
    extract_dir="data/",
    chunk_size=5
)
```

## Examples

### Complete Data Analysis Workflow

```python
import pandas as pd
import matplotlib.pyplot as plt
from past1000.data import HistoricalRecords, load_data
from past1000.compare import experiment_corr_2d

# 1. Load data
history = HistoricalRecords(
    shp_path="data/north_china_precip_regions.shp",
    data_path="data/paleo_recon_data.xlsx",
    region="华北地区"
)

# 2. Process data
historical_series = history.aggregate("mean", inplace=True)

# 3. Load climate data
datasets, uncertainties, _ = load_data(config)
climate_series = datasets.mean(axis=1)

# 4. Perform analysis
corr_df, r_benchmark, ax = experiment_corr_2d(
    data1=historical_series,
    data2=climate_series,
    corr_method="kendall"
)

plt.show()
```

### Multi-Region Analysis

```python
# Analyze multiple regions
regions = ["华北地区", "东北地区", "华东地区"]
results = {}

for region in regions:
    history = HistoricalRecords(
        shp_path="data/regions.shp",
        data_path="data/historical_data.xlsx",
        region=region
    )

    # Aggregate and analyze
    series = history.aggregate("bayesian")
    results[region] = series

# Combine results
combined_df = pd.DataFrame(results)
```
