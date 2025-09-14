# Basic Usage Examples

This guide provides practical examples of using Past1000 for common climate reconstruction analysis tasks.

## Example 1: Loading and Analyzing Historical Data

### Load Historical Records

```python
import pandas as pd
import matplotlib.pyplot as plt
from past1000.data import HistoricalRecords

# Load historical drought/flood records
history = HistoricalRecords(
    shp_path="data/north_china_precip_regions.shp",
    data_path="data/paleo_recon_data.xlsx",
    region="华北地区",
    symmetrical_level=True
)

# Display basic information
print(f"Data shape: {history.data.shape}")
print(f"Time range: {history.data.index.min()}-{history.data.index.max()}")
print(f"Stations: {list(history.data.columns)}")
```

### Aggregate Multi-Station Data

```python
# Simple mean aggregation
mean_series = history.aggregate("mean")
print(f"Mean series length: {len(mean_series)}")

# Bayesian aggregation with spatial correlation
bayesian_series = history.aggregate(
    "bayesian",
    spatial_correlation=0.5,
    uncertainty_factor=1.0
)

# Compare different aggregation methods
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(mean_series.index, mean_series.values, label="Mean", alpha=0.7)
ax.plot(bayesian_series.index, bayesian_series.values, label="Bayesian", alpha=0.7)
ax.set_title("Historical Data Aggregation Comparison")
ax.set_xlabel("Year")
ax.set_ylabel("Drought Level")
ax.legend()
plt.show()
```

## Example 2: Climate Data Processing

### Load and Standardize Climate Data

```python
from past1000.data import load_nat_data
from past1000.mc import standardize_both

# Load tree ring reconstruction data
datasets, uncertainties = load_nat_data(
    folder="data/tree_ring/",
    includes=["shi2018", "yellow2019"],
    start_year=1000,
    standardize=True
)

print(f"Climate data shape: {datasets.shape}")
print(f"Uncertainty shape: {uncertainties.shape}")

# Standardize data
std_data, std_uncertainty = standardize_both(datasets.mean(axis=1))
print(f"Standardized data range: {std_data.min():.3f} to {std_data.max():.3f}")
```

### Classify Climate Data

```python
from past1000.filters import classify

# Classify continuous data into discrete levels
classified_data = classify(std_data)
print(f"Classification levels: {classified_data.value_counts().sort_index()}")

# Visualize classification
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Original data
ax1.plot(std_data.index, std_data.values, alpha=0.7, label="Original")
ax1.set_title("Original Climate Data")
ax1.set_ylabel("Z-score")
ax1.legend()

# Classified data
ax2.plot(classified_data.index, classified_data.values, alpha=0.7, label="Classified")
ax2.set_title("Classified Climate Data")
ax2.set_xlabel("Year")
ax2.set_ylabel("Level")
ax2.legend()

plt.tight_layout()
plt.show()
```

## Example 3: Correlation Analysis

### Basic Correlation Analysis

```python
from past1000.compare import compare_corr, experiment_corr_2d
from past1000.filters import calc_std_deviation

# Prepare data
historical_series = history.aggregate("mean")
climate_series = datasets.mean(axis=1)

# Basic correlation
r, p, n = compare_corr(
    historical_series,
    climate_series,
    corr_method="kendall"
)

print(f"Correlation: {r:.3f}")
print(f"P-value: {p:.3f}")
print(f"Sample size: {n}")
```

### Advanced Correlation Analysis

```python
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

print(f"Benchmark correlation: {r_benchmark:.3f}")
print(f"Significant correlations: {(corr_df > 0.3).sum().sum()}")

plt.show()
```

## Example 4: Mismatch Analysis

### Create Mismatch Report

```python
from past1000.calibration import MismatchReport

# Classify both datasets
historical_classified = classify(historical_series)
climate_classified = classify(climate_series)

# Create mismatch report
report = MismatchReport(
    pred=historical_classified,
    true=climate_classified,
    value_series=climate_series
)

# Analyze error patterns
error_df = report.analyze_error_patterns(mc_runs=1000)

# Display statistics
stats = report.get_statistics_summary()
print(f"Kappa: {stats['kappa']:.3f}")
print(f"Accuracy: {stats['accuracy']:.3f}")
print(f"Kendall's Tau: {stats['kendall_tau']:.3f}")
```

### Visualize Mismatch Analysis

```python
# Generate comprehensive report
fig = report.generate_report_figure(
    figsize=(15, 8),
    save_path="mismatch_analysis.png"
)

plt.show()

# Display error analysis
print("Error Analysis:")
print(error_df[['value', 'pred', 'true', 'exact', 'diff']].head())
```

## Example 5: Time Period Analysis

### Analyze Different Historical Periods

```python
# Define historical periods
periods = {
    "Medieval": slice(1000, 1469),
    "Little Ice Age": slice(1469, 1659),
    "Early Modern": slice(1659, 1900)
}

# Analyze each period
period_results = {}

for period_name, time_slice in periods.items():
    # Get data for period
    hist_period = historical_series.loc[time_slice]
    clim_period = climate_series.loc[time_slice]

    # Calculate correlation
    r, p, n = compare_corr(hist_period, clim_period, corr_method="kendall")

    # Create mismatch report
    hist_classified = classify(hist_period)
    clim_classified = classify(clim_period)

    report = MismatchReport(hist_classified, clim_classified)
    stats = report.get_statistics_summary()

    period_results[period_name] = {
        "correlation": r,
        "p_value": p,
        "samples": n,
        "kappa": stats["kappa"],
        "accuracy": stats["accuracy"]
    }

# Display results
results_df = pd.DataFrame(period_results).T
print(results_df)
```

### Visualize Period Comparison

```python
# Plot correlation by period
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Correlations
periods = list(period_results.keys())
correlations = [period_results[p]["correlation"] for p in periods]
p_values = [period_results[p]["p_value"] for p in periods]

ax1.bar(periods, correlations, alpha=0.7)
ax1.set_title("Correlation by Historical Period")
ax1.set_ylabel("Kendall's Tau")
ax1.tick_params(axis='x', rotation=45)

# Kappa values
kappas = [period_results[p]["kappa"] for p in periods]
ax2.bar(periods, kappas, alpha=0.7, color='orange')
ax2.set_title("Kappa by Historical Period")
ax2.set_ylabel("Cohen's Kappa")
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
```

## Example 6: Rolling Window Analysis

### Rolling Correlation Analysis

```python
from past1000.compare import sweep_slices, sweep_max_corr_year
import numpy as np

# Generate time windows
slices, mid_years, slice_labels = sweep_slices(
    start_year=1000,
    window_size=200,
    step_size=20,
    end_year=1900
)

# Define window parameters
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

# Plot results
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(mid_years, max_correlations, 'o-', alpha=0.7)
ax.set_title("Maximum Correlations by Time Window")
ax.set_xlabel("Mid-Year of Window")
ax.set_ylabel("Maximum Correlation")
ax.grid(True, alpha=0.3)
plt.show()
```

## Example 7: Data Quality Assessment

### Check Data Distribution

```python
from past1000.data import check_distribution

# Check distribution of climate data
dist_results = check_distribution(datasets, only_best=True)
print("Best fitting distributions:")
print(dist_results)

# Check distribution of historical data
hist_dist = check_distribution(historical_series, only_best=True)
print(f"\nHistorical data best fit: {hist_dist.name}")
```

### Validate Data Quality

```python
def assess_data_quality(series, name):
    """Assess data quality metrics."""

    # Basic statistics
    stats = {
        "name": name,
        "length": len(series),
        "missing": series.isna().sum(),
        "missing_pct": series.isna().sum() / len(series) * 100,
        "mean": series.mean(),
        "std": series.std(),
        "min": series.min(),
        "max": series.max(),
        "range": series.max() - series.min()
    }

    # Check for outliers (beyond 3 standard deviations)
    z_scores = np.abs((series - series.mean()) / series.std())
    outliers = (z_scores > 3).sum()
    stats["outliers"] = outliers
    stats["outlier_pct"] = outliers / len(series) * 100

    return stats

# Assess both datasets
hist_quality = assess_data_quality(historical_series, "Historical")
clim_quality = assess_data_quality(climate_series, "Climate")

# Display results
quality_df = pd.DataFrame([hist_quality, clim_quality])
print(quality_df)
```

## Example 8: Export Results

### Save Analysis Results

```python
import json
from pathlib import Path

# Create output directory
output_dir = Path("outputs/basic_analysis")
output_dir.mkdir(parents=True, exist_ok=True)

# Save data
historical_series.to_csv(output_dir / "historical_series.csv")
climate_series.to_csv(output_dir / "climate_series.csv")
classified_data.to_csv(output_dir / "classified_data.csv")

# Save correlation results
corr_df.to_csv(output_dir / "correlation_analysis.csv")

# Save mismatch report
mismatch_stats = report.get_statistics_summary()
with open(output_dir / "mismatch_stats.json", 'w') as f:
    json.dump(mismatch_stats, f, indent=2)

# Save period results
results_df.to_csv(output_dir / "period_analysis.csv")

print(f"Results saved to {output_dir}")
```

### Generate Summary Report

```python
def generate_summary_report(historical_series, climate_series, report):
    """Generate a summary report of the analysis."""

    # Basic statistics
    hist_stats = assess_data_quality(historical_series, "Historical")
    clim_stats = assess_data_quality(climate_series, "Climate")

    # Correlation analysis
    r, p, n = compare_corr(historical_series, climate_series, corr_method="kendall")

    # Mismatch analysis
    mismatch_stats = report.get_statistics_summary()

    # Create summary
    summary = {
        "analysis_date": pd.Timestamp.now().isoformat(),
        "data_quality": {
            "historical": hist_stats,
            "climate": clim_stats
        },
        "correlation_analysis": {
            "kendall_tau": r,
            "p_value": p,
            "sample_size": n
        },
        "mismatch_analysis": mismatch_stats,
        "recommendations": []
    }

    # Add recommendations
    if r < 0.3:
        summary["recommendations"].append("Low correlation detected. Consider data quality issues.")

    if mismatch_stats["kappa"] < 0.4:
        summary["recommendations"].append("Poor classification agreement. Review classification thresholds.")

    if hist_stats["missing_pct"] > 10:
        summary["recommendations"].append("High missing data in historical records. Consider data imputation.")

    return summary

# Generate summary
summary = generate_summary_report(historical_series, climate_series, report)

# Save summary
with open(output_dir / "summary_report.json", 'w') as f:
    json.dump(summary, f, indent=2)

print("Summary report generated")
```

## Next Steps

After completing these basic examples, you can explore:

- **[Advanced Analysis](advanced-analysis.md)**: More complex analysis patterns
- **[ABM Simulation](abm-simulation.md)**: Agent-based modeling examples
- **[API Reference](../api/)**: Detailed function documentation

## Troubleshooting

### Common Issues

1. **Data Loading Errors**
   - Check file paths and formats
   - Ensure data files exist and are accessible
   - Verify data structure matches expected format

2. **Memory Issues**
   - Use chunked processing for large datasets
   - Clear variables when not needed
   - Use lazy loading for NetCDF files

3. **Correlation Analysis Issues**
   - Ensure data alignment (same time periods)
   - Check for sufficient sample size
   - Handle missing values appropriately

4. **Visualization Problems**
   - Check data types and ranges
   - Ensure proper matplotlib backend
   - Save figures with appropriate DPI
