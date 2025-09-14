# Quick Start Guide

This guide will get you up and running with Past1000 in just a few minutes.

## Basic Workflow

The typical workflow with Past1000 involves three main steps:

1. **Load Data**: Import historical records and climate reconstruction data
2. **Process Data**: Standardize and prepare data for analysis
3. **Analyze**: Perform statistical comparisons and generate reports

## Example 1: Basic Data Loading and Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt
from past1000 import HistoricalRecords, load_data
from past1000.compare import experiment_corr_2d
from past1000.calibration import MismatchReport

# Step 1: Load historical records
history = HistoricalRecords(
    shp_path="data/north_china_precip_regions.shp",
    data_path="data/paleo_recon_data.xlsx",
    region="华北地区",
    symmetrical_level=True
)

# Step 2: Load climate reconstruction data
# (Assuming you have a configuration object)
datasets, uncertainties, _ = load_data(config)

# Step 3: Aggregate historical data
historical_series = history.aggregate("mean", inplace=True)

# Step 4: Perform correlation analysis
corr_df, r_benchmark, ax = experiment_corr_2d(
    data1=historical_series,
    data2=climate_data,
    corr_method="kendall",
    filter_side="right"
)

plt.show()
```

## Example 2: Classification and Mismatch Analysis

```python
from past1000.filters import classify
from past1000.calibration import MismatchReport

# Classify continuous data into discrete levels
predicted_levels = classify(historical_series)
observed_levels = classify(climate_data)

# Create mismatch report
report = MismatchReport(
    pred=predicted_levels,
    true=observed_levels,
    value_series=climate_data
)

# Analyze error patterns
error_df = report.analyze_error_patterns()

# Generate visualization
fig = report.generate_report_figure(save_path="mismatch_analysis.png")
```

## Example 3: Agent-Based Modeling

```python
from past1000.abm import ClimateObservingModel, repeat_run
from abses import Experiment

# Configure the model
config = {
    "model": {
        "years": 100,
        "max_age": 40,
        "new_agents": 10,
        "memory_baseline": "personal",
        "loss_rate": 0.2
    }
}

# Run the simulation
exp = Experiment.new(ClimateObservingModel, cfg=config)
exp.batch_run(repeats=10, parallels=4)

# Analyze results
model = exp.models[0]
corr_curve = model.get_corr_curve(window_length=50)
mismatch_report = model.mismatch_report
```

## Configuration

Past1000 uses Hydra for configuration management. Create a configuration file:

```yaml
# config/config.yaml
ds:
  noaa: "data/tree_ring/"
  includes: ["shi2018", "yellow2019"]
  atlas:
    shp: "data/north_china_precip_regions.shp"
    file: "data/paleo_recon_data.xlsx"
  out:
    tree_ring: "outputs/tree_ring_data.csv"
    tree_ring_uncertainty: "outputs/tree_ring_uncertainty.csv"

corr_method: "kendall"
filter_side: "right"
agg_method: "mean"
to_std: "mapping"
```

## Data Formats

### Historical Records
Historical data should be in Excel format with:
- Years as row index (1000-2020)
- Regions as columns
- Values: 1-5 (drought levels) or standardized values

### Climate Reconstruction Data
Supported formats:
- NetCDF files (.nc)
- CSV files (.csv)
- Text files with tab/space separation

### Spatial Data
Shapefiles (.shp) with region information for spatial analysis.

## Common Patterns

### Time Series Analysis
```python
# Select specific time periods
stage1_data = history.period("1000-1469")  # Stage 1
stage2_data = history.period("1469-1659")  # Stage 2

# Or use slice notation
recent_data = history.period(slice(1800, 1900))
```

### Data Standardization
```python
# Different standardization methods
history_mapping = HistoricalRecords(..., to_std="mapping")
history_sampling = HistoricalRecords(..., to_std="sampling")

# Manual standardization
from past1000.mc import standardize_both
standardized_data, uncertainty = standardize_both(raw_data)
```

### Statistical Analysis
```python
# Multiple correlation methods
from past1000.utils.calc import calc_corr

pearson_r, p_val, n = calc_corr(data1, data2, how="pearson")
kendall_tau, p_val, n = calc_corr(data1, data2, how="kendall")
spearman_rho, p_val, n = calc_corr(data1, data2, how="spearman")
```

## Next Steps

Now that you have the basics, explore:

- **[API Reference](api/)**: Detailed documentation for all modules
- **[Examples](examples/)**: More complex usage scenarios
- **[Configuration Guide](getting-started/configuration.md)**: Advanced configuration options

## Getting Help

- 📖 Check the [API documentation](api/) for detailed function descriptions
- 🐛 Report issues on [GitHub](https://github.com/SongshGeo/past1000/issues)
- 💬 Ask questions via [email](mailto:songshgeo@gmail.com)
