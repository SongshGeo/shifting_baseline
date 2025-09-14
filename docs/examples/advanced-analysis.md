# Advanced Analysis Examples

This guide demonstrates advanced analysis patterns and techniques using Past1000 for complex climate reconstruction research.

## Example 1: Multi-Scale Temporal Analysis

### Hierarchical Time Series Analysis

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from past1000.data import HistoricalRecords
from past1000.compare import sweep_slices, sweep_max_corr_year
from past1000.filters import classify

# Load data
history = HistoricalRecords(
    shp_path="data/north_china_precip_regions.shp",
    data_path="data/paleo_recon_data.xlsx",
    region="华北地区"
)

# Aggregate data
historical_series = history.aggregate("bayesian", spatial_correlation=0.6)

# Define multiple time scales
time_scales = {
    "decadal": 10,
    "centennial": 100,
    "multi_centennial": 300
}

# Analyze each time scale
scale_results = {}

for scale_name, window_size in time_scales.items():
    # Generate slices for this scale
    slices, mid_years, slice_labels = sweep_slices(
        start_year=1000,
        window_size=window_size,
        step_size=window_size // 2,
        end_year=1900
    )

    # Calculate correlations
    windows = np.arange(5, min(50, window_size // 2))
    min_periods = np.repeat(3, len(windows))

    max_corr_years, max_correlations = sweep_max_corr_year(
        data1=historical_series,
        data2=climate_series,
        slices=slices,
        windows=windows,
        min_periods=min_periods,
        corr_method="kendall"
    )

    scale_results[scale_name] = {
        "slices": slices,
        "mid_years": mid_years,
        "max_correlations": max_correlations,
        "window_size": window_size
    }

# Visualize multi-scale analysis
fig, axes = plt.subplots(3, 1, figsize=(15, 12))

for i, (scale_name, results) in enumerate(scale_results.items()):
    ax = axes[i]
    ax.plot(results["mid_years"], results["max_correlations"], 'o-', alpha=0.7)
    ax.set_title(f"{scale_name.title()} Scale Analysis (Window: {results['window_size']} years)")
    ax.set_ylabel("Maximum Correlation")
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Mid-Year of Window")
plt.tight_layout()
plt.show()
```

## Example 2: Spatial Correlation Analysis

### Multi-Region Comparison

```python
from past1000.data import HistoricalRecords
from past1000.compare import compare_corr
from past1000.calibration import MismatchReport

# Define regions to analyze
regions = ["华北地区", "东北地区", "华东地区", "华中地区"]

# Load data for each region
regional_data = {}
for region in regions:
    try:
        history = HistoricalRecords(
            shp_path="data/north_china_precip_regions.shp",
            data_path="data/paleo_recon_data.xlsx",
            region=region
        )
        regional_data[region] = history.aggregate("mean")
    except Exception as e:
        print(f"Failed to load data for {region}: {e}")

# Calculate cross-regional correlations
correlation_matrix = pd.DataFrame(index=regions, columns=regions)

for region1 in regions:
    for region2 in regions:
        if region1 in regional_data and region2 in regional_data:
            r, p, n = compare_corr(
                regional_data[region1],
                regional_data[region2],
                corr_method="kendall"
            )
            correlation_matrix.loc[region1, region2] = r
        else:
            correlation_matrix.loc[region1, region2] = np.nan

# Visualize correlation matrix
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(correlation_matrix.astype(float), cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(regions)))
ax.set_yticks(range(len(regions)))
ax.set_xticklabels(regions, rotation=45)
ax.set_yticklabels(regions)

# Add correlation values
for i in range(len(regions)):
    for j in range(len(regions)):
        text = ax.text(j, i, f"{correlation_matrix.iloc[i, j]:.3f}",
                      ha="center", va="center", color="black")

plt.colorbar(im, ax=ax)
ax.set_title("Cross-Regional Correlation Matrix")
plt.tight_layout()
plt.show()
```

## Example 3: Uncertainty Quantification

### Monte Carlo Uncertainty Analysis

```python
from past1000.mc import combine_reconstructions
from past1000.calibration import MismatchReport
import numpy as np

def monte_carlo_uncertainty_analysis(datasets, uncertainties, n_simulations=1000):
    """Perform Monte Carlo uncertainty analysis."""

    results = []

    for i in range(n_simulations):
        # Add noise to data
        noisy_datasets = datasets + np.random.normal(0, uncertainties.values)

        # Combine reconstructions
        combined, _ = combine_reconstructions(
            reconstructions=noisy_datasets,
            uncertainties=uncertainties,
            standardize=True
        )

        # Calculate correlation
        r, p, n = compare_corr(
            historical_series,
            combined["mean"],
            corr_method="kendall"
        )

        results.append({
            "simulation": i,
            "correlation": r,
            "p_value": p,
            "samples": n
        })

    return pd.DataFrame(results)

# Run uncertainty analysis
uncertainty_results = monte_carlo_uncertainty_analysis(
    datasets,
    uncertainties,
    n_simulations=500
)

# Analyze uncertainty
print("Uncertainty Analysis Results:")
print(f"Mean correlation: {uncertainty_results['correlation'].mean():.3f}")
print(f"Std correlation: {uncertainty_results['correlation'].std():.3f}")
print(f"95% confidence interval: {uncertainty_results['correlation'].quantile([0.025, 0.975]).values}")

# Visualize uncertainty
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(uncertainty_results['correlation'], bins=50, alpha=0.7, density=True)
ax.axvline(uncertainty_results['correlation'].mean(), color='red', linestyle='--', label='Mean')
ax.axvline(uncertainty_results['correlation'].quantile(0.025), color='orange', linestyle='--', label='2.5%')
ax.axvline(uncertainty_results['correlation'].quantile(0.975), color='orange', linestyle='--', label='97.5%')
ax.set_xlabel('Correlation Coefficient')
ax.set_ylabel('Density')
ax.set_title('Monte Carlo Uncertainty Analysis')
ax.legend()
plt.show()
```

## Example 4: Advanced Classification Analysis

### Multi-Threshold Sensitivity Analysis

```python
from past1000.filters import classify_series
from past1000.calibration import MismatchReport

def sensitivity_analysis(data, threshold_sets):
    """Analyze sensitivity to different classification thresholds."""

    results = []

    for threshold_name, (thresholds, levels) in threshold_sets.items():
        # Classify data
        classified = classify_series(data, thresholds=thresholds, levels=levels)

        # Calculate statistics
        level_counts = classified.value_counts().sort_index()
        level_proportions = level_counts / len(classified)

        # Calculate entropy (measure of distribution uniformity)
        entropy = -sum(p * np.log2(p) for p in level_proportions if p > 0)

        results.append({
            "threshold_set": threshold_name,
            "thresholds": thresholds,
            "levels": levels,
            "level_counts": level_counts.to_dict(),
            "level_proportions": level_proportions.to_dict(),
            "entropy": entropy,
            "is_balanced": 0.1 < level_proportions.get(0, 0) < 0.6
        })

    return pd.DataFrame(results)

# Define multiple threshold sets
threshold_sets = {
    "standard": ([-1.17, -0.33, 0.33, 1.17], [-2, -1, 0, 1, 2]),
    "conservative": ([-1.5, -0.5, 0.5, 1.5], [-2, -1, 0, 1, 2]),
    "sensitive": ([-0.8, -0.2, 0.2, 0.8], [-2, -1, 0, 1, 2]),
    "three_level": ([-0.5, 0.5], [-1, 0, 1]),
    "five_level_extended": ([-2.0, -1.0, 0.0, 1.0, 2.0], [-3, -2, -1, 0, 1, 2])
}

# Run sensitivity analysis
sensitivity_results = sensitivity_analysis(climate_series, threshold_sets)

# Display results
print("Sensitivity Analysis Results:")
for _, row in sensitivity_results.iterrows():
    print(f"\n{row['threshold_set']}:")
    print(f"  Entropy: {row['entropy']:.3f}")
    print(f"  Balanced: {row['is_balanced']}")
    print(f"  Level proportions: {row['level_proportions']}")

# Visualize sensitivity
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

for i, (_, row) in enumerate(sensitivity_results.iterrows()):
    ax = axes[i]
    levels = list(row['level_proportions'].keys())
    proportions = list(row['level_proportions'].values())

    ax.bar(levels, proportions, alpha=0.7)
    ax.set_title(f"{row['threshold_set']} (Entropy: {row['entropy']:.3f})")
    ax.set_xlabel("Level")
    ax.set_ylabel("Proportion")
    ax.set_ylim(0, 1)

plt.tight_layout()
plt.show()
```

## Example 5: Advanced Mismatch Analysis

### Detailed Error Pattern Analysis

```python
def detailed_mismatch_analysis(pred, true, value_series):
    """Perform detailed mismatch analysis with multiple metrics."""

    # Create mismatch report
    report = MismatchReport(pred, true, value_series=value_series)

    # Analyze error patterns
    error_df = report.analyze_error_patterns(mc_runs=2000)

    # Calculate additional metrics
    confusion_matrix = report.cm_df

    # Calculate per-class metrics
    per_class_metrics = {}
    for level in report.cm_df.index:
        tp = confusion_matrix.loc[level, level]  # True positives
        fp = confusion_matrix.loc[level, :].sum() - tp  # False positives
        fn = confusion_matrix.loc[:, level].sum() - tp  # False negatives
        tn = confusion_matrix.sum().sum() - tp - fp - fn  # True negatives

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        per_class_metrics[level] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn
        }

    # Calculate overall metrics
    overall_stats = report.get_statistics_summary()

    # Analyze error patterns by value range
    error_by_value = error_df.groupby(pd.cut(error_df['value'], bins=10)).agg({
        'exact': 'mean',
        'diff': ['mean', 'std', 'count']
    }).round(3)

    return {
        "report": report,
        "error_df": error_df,
        "per_class_metrics": per_class_metrics,
        "overall_stats": overall_stats,
        "error_by_value": error_by_value
    }

# Run detailed analysis
detailed_results = detailed_mismatch_analysis(
    historical_classified,
    climate_classified,
    climate_series
)

# Display per-class metrics
per_class_df = pd.DataFrame(detailed_results["per_class_metrics"]).T
print("Per-Class Metrics:")
print(per_class_df[['precision', 'recall', 'f1_score']])

# Visualize error patterns
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Confusion matrix
report.plot_confusion_matrix(ax=axes[0, 0])

# Error by value range
error_by_value = detailed_results["error_by_value"]
axes[0, 1].bar(range(len(error_by_value)), error_by_value[('exact', 'mean')])
axes[0, 1].set_title("Accuracy by Value Range")
axes[0, 1].set_xlabel("Value Range")
axes[0, 1].set_ylabel("Accuracy")

# Error distribution
axes[1, 0].hist(detailed_results["error_df"]['diff'].dropna(), bins=30, alpha=0.7)
axes[1, 0].set_title("Error Distribution")
axes[1, 0].set_xlabel("Error Value")
axes[1, 0].set_ylabel("Frequency")

# Per-class F1 scores
levels = list(detailed_results["per_class_metrics"].keys())
f1_scores = [detailed_results["per_class_metrics"][level]["f1_score"] for level in levels]
axes[1, 1].bar(levels, f1_scores, alpha=0.7)
axes[1, 1].set_title("Per-Class F1 Scores")
axes[1, 1].set_xlabel("Level")
axes[1, 1].set_ylabel("F1 Score")

plt.tight_layout()
plt.show()
```

## Example 6: Time Series Decomposition

### Trend and Seasonality Analysis

```python
from scipy import signal
from sklearn.decomposition import PCA

def time_series_decomposition(series, window_size=30):
    """Decompose time series into trend, seasonal, and residual components."""

    # Detrend using rolling mean
    trend = series.rolling(window=window_size, center=True).mean()
    detrended = series - trend

    # Remove trend
    detrended_clean = detrended.dropna()

    # Apply seasonal decomposition (if sufficient data)
    if len(detrended_clean) > 2 * window_size:
        from statsmodels.tsa.seasonal import seasonal_decompose

        # Create a regular time series for decomposition
        regular_series = pd.Series(
            detrended_clean.values,
            index=pd.date_range(start='1000-01-01', periods=len(detrended_clean), freq='Y')
        )

        try:
            decomposition = seasonal_decompose(regular_series, model='additive', period=10)
            seasonal = decomposition.seasonal
            residual = decomposition.resid
        except:
            seasonal = pd.Series(0, index=detrended_clean.index)
            residual = detrended_clean
    else:
        seasonal = pd.Series(0, index=detrended_clean.index)
        residual = detrended_clean

    return {
        "original": series,
        "trend": trend,
        "seasonal": seasonal,
        "residual": residual,
        "detrended": detrended
    }

# Decompose both series
hist_decomposition = time_series_decomposition(historical_series)
clim_decomposition = time_series_decomposition(climate_series)

# Visualize decomposition
fig, axes = plt.subplots(3, 2, figsize=(15, 12))

# Historical series decomposition
axes[0, 0].plot(hist_decomposition["original"].index, hist_decomposition["original"].values, label="Original")
axes[0, 0].plot(hist_decomposition["trend"].index, hist_decomposition["trend"].values, label="Trend")
axes[0, 0].set_title("Historical Series Decomposition")
axes[0, 0].legend()

axes[1, 0].plot(hist_decomposition["seasonal"].index, hist_decomposition["seasonal"].values, label="Seasonal")
axes[1, 0].set_title("Historical Seasonal Component")
axes[1, 0].legend()

axes[2, 0].plot(hist_decomposition["residual"].index, hist_decomposition["residual"].values, label="Residual")
axes[2, 0].set_title("Historical Residual Component")
axes[2, 0].legend()

# Climate series decomposition
axes[0, 1].plot(clim_decomposition["original"].index, clim_decomposition["original"].values, label="Original")
axes[0, 1].plot(clim_decomposition["trend"].index, clim_decomposition["trend"].values, label="Trend")
axes[0, 1].set_title("Climate Series Decomposition")
axes[0, 1].legend()

axes[1, 1].plot(clim_decomposition["seasonal"].index, clim_decomposition["seasonal"].values, label="Seasonal")
axes[1, 1].set_title("Climate Seasonal Component")
axes[1, 1].legend()

axes[2, 1].plot(clim_decomposition["residual"].index, clim_decomposition["residual"].values, label="Residual")
axes[2, 1].set_title("Climate Residual Component")
axes[2, 1].legend()

plt.tight_layout()
plt.show()

# Analyze correlations between components
components = ["trend", "seasonal", "residual"]
component_correlations = {}

for component in components:
    hist_comp = hist_decomposition[component].dropna()
    clim_comp = clim_decomposition[component].dropna()

    # Align data
    aligned_hist, aligned_clim = hist_comp.align(clim_comp, join='inner')

    if len(aligned_hist) > 10:  # Sufficient data
        r, p, n = compare_corr(aligned_hist, aligned_clim, corr_method="kendall")
        component_correlations[component] = {"correlation": r, "p_value": p, "samples": n}

print("Component Correlations:")
for component, stats in component_correlations.items():
    print(f"{component}: r={stats['correlation']:.3f}, p={stats['p_value']:.3f}")
```

## Example 7: Machine Learning Integration

### Feature Engineering and Prediction

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

def create_features(series, window_sizes=[5, 10, 20, 50]):
    """Create features from time series data."""

    features = pd.DataFrame(index=series.index)

    # Original values
    features['value'] = series

    # Rolling statistics
    for window in window_sizes:
        features[f'mean_{window}'] = series.rolling(window=window).mean()
        features[f'std_{window}'] = series.rolling(window=window).std()
        features[f'min_{window}'] = series.rolling(window=window).min()
        features[f'max_{window}'] = series.rolling(window=window).max()
        features[f'range_{window}'] = features[f'max_{window}'] - features[f'min_{window}']

    # Lag features
    for lag in [1, 2, 3, 5, 10]:
        features[f'lag_{lag}'] = series.shift(lag)

    # Difference features
    for diff in [1, 2, 5]:
        features[f'diff_{diff}'] = series.diff(diff)

    # Trend features
    features['trend'] = series.rolling(window=20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])

    return features

# Create features for both series
hist_features = create_features(historical_series)
clim_features = create_features(climate_series)

# Align features
aligned_hist, aligned_clim = hist_features.align(clim_features, join='inner')

# Remove rows with NaN values
clean_data = pd.concat([aligned_hist, aligned_clim], axis=1).dropna()

# Split features and target
X = clean_data[aligned_hist.columns]
y = clean_data[aligned_clim.columns]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model Performance:")
print(f"MSE: {mse:.3f}")
print(f"R²: {r2:.3f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Visualize predictions
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Actual vs Predicted
axes[0, 0].scatter(y_test.values.flatten(), y_pred.flatten(), alpha=0.6)
axes[0, 0].plot([y_test.min().min(), y_test.max().max()], [y_test.min().min(), y_test.max().max()], 'r--')
axes[0, 0].set_xlabel('Actual')
axes[0, 0].set_ylabel('Predicted')
axes[0, 0].set_title('Actual vs Predicted')

# Residuals
residuals = y_test.values.flatten() - y_pred.flatten()
axes[0, 1].scatter(y_pred.flatten(), residuals, alpha=0.6)
axes[0, 1].axhline(y=0, color='r', linestyle='--')
axes[0, 1].set_xlabel('Predicted')
axes[0, 1].set_ylabel('Residuals')
axes[0, 1].set_title('Residuals vs Predicted')

# Feature importance
top_features = feature_importance.head(10)
axes[1, 0].barh(range(len(top_features)), top_features['importance'])
axes[1, 0].set_yticks(range(len(top_features)))
axes[1, 0].set_yticklabels(top_features['feature'])
axes[1, 0].set_xlabel('Importance')
axes[1, 0].set_title('Feature Importance')

# Time series comparison
time_index = y_test.index
axes[1, 1].plot(time_index, y_test.values.flatten(), label='Actual', alpha=0.7)
axes[1, 1].plot(time_index, y_pred.flatten(), label='Predicted', alpha=0.7)
axes[1, 1].set_xlabel('Time')
axes[1, 1].set_ylabel('Value')
axes[1, 1].set_title('Time Series Comparison')
axes[1, 1].legend()

plt.tight_layout()
plt.show()
```

## Example 8: Comprehensive Research Workflow

### Complete Analysis Pipeline

```python
def comprehensive_analysis_pipeline(historical_data_path, climate_data_path, output_dir):
    """Complete analysis pipeline for climate reconstruction research."""

    from pathlib import Path
    import json
    from datetime import datetime

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize results
    results = {
        "analysis_date": datetime.now().isoformat(),
        "data_quality": {},
        "correlation_analysis": {},
        "mismatch_analysis": {},
        "uncertainty_analysis": {},
        "recommendations": []
    }

    # 1. Data Loading and Quality Assessment
    print("Step 1: Loading and assessing data quality...")

    # Load historical data
    history = HistoricalRecords(
        shp_path="data/north_china_precip_regions.shp",
        data_path=historical_data_path,
        region="华北地区"
    )

    # Load climate data
    datasets, uncertainties = load_nat_data(
        folder=climate_data_path,
        includes=["shi2018", "yellow2019"],
        start_year=1000,
        standardize=True
    )

    # Assess data quality
    hist_quality = assess_data_quality(history.data.mean(axis=1), "Historical")
    clim_quality = assess_data_quality(datasets.mean(axis=1), "Climate")
    results["data_quality"] = {"historical": hist_quality, "climate": clim_quality}

    # 2. Data Processing and Aggregation
    print("Step 2: Processing and aggregating data...")

    # Aggregate historical data
    historical_series = history.aggregate("bayesian", spatial_correlation=0.5)
    climate_series = datasets.mean(axis=1)

    # 3. Correlation Analysis
    print("Step 3: Performing correlation analysis...")

    # Basic correlation
    r, p, n = compare_corr(historical_series, climate_series, corr_method="kendall")
    results["correlation_analysis"]["basic"] = {"correlation": r, "p_value": p, "samples": n}

    # Advanced correlation analysis
    corr_df, r_benchmark, ax = experiment_corr_2d(
        data1=historical_series,
        data2=climate_series,
        corr_method="kendall",
        filter_func=calc_std_deviation,
        filter_side="right"
    )
    results["correlation_analysis"]["advanced"] = {
        "benchmark_correlation": r_benchmark,
        "significant_correlations": (corr_df > 0.3).sum().sum()
    }

    # 4. Classification and Mismatch Analysis
    print("Step 4: Performing classification and mismatch analysis...")

    # Classify data
    hist_classified = classify(historical_series)
    clim_classified = classify(climate_series)

    # Create mismatch report
    report = MismatchReport(
        pred=hist_classified,
        true=clim_classified,
        value_series=climate_series
    )

    # Analyze error patterns
    error_df = report.analyze_error_patterns(mc_runs=1000)
    mismatch_stats = report.get_statistics_summary()
    results["mismatch_analysis"] = mismatch_stats

    # 5. Uncertainty Analysis
    print("Step 5: Performing uncertainty analysis...")

    # Monte Carlo uncertainty analysis
    uncertainty_results = monte_carlo_uncertainty_analysis(
        datasets, uncertainties, n_simulations=200
    )

    results["uncertainty_analysis"] = {
        "mean_correlation": uncertainty_results['correlation'].mean(),
        "std_correlation": uncertainty_results['correlation'].std(),
        "confidence_interval": uncertainty_results['correlation'].quantile([0.025, 0.975]).tolist()
    }

    # 6. Generate Recommendations
    print("Step 6: Generating recommendations...")

    if r < 0.3:
        results["recommendations"].append("Low correlation detected. Consider data quality issues.")

    if mismatch_stats["kappa"] < 0.4:
        results["recommendations"].append("Poor classification agreement. Review classification thresholds.")

    if hist_quality["missing_pct"] > 10:
        results["recommendations"].append("High missing data in historical records. Consider data imputation.")

    if uncertainty_results['correlation'].std() > 0.2:
        results["recommendations"].append("High uncertainty in correlation estimates. Consider additional data sources.")

    # 7. Save Results
    print("Step 7: Saving results...")

    # Save data
    historical_series.to_csv(output_dir / "historical_series.csv")
    climate_series.to_csv(output_dir / "climate_series.csv")
    hist_classified.to_csv(output_dir / "historical_classified.csv")
    clim_classified.to_csv(output_dir / "climate_classified.csv")

    # Save analysis results
    corr_df.to_csv(output_dir / "correlation_analysis.csv")
    error_df.to_csv(output_dir / "error_analysis.csv")
    uncertainty_results.to_csv(output_dir / "uncertainty_analysis.csv")

    # Save summary
    with open(output_dir / "analysis_summary.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Generate visualizations
    fig = report.generate_report_figure(figsize=(15, 10))
    fig.savefig(output_dir / "mismatch_analysis.png", dpi=300, bbox_inches='tight')

    # Correlation heatmap
    corr_df.to_csv(output_dir / "correlation_heatmap.csv")

    print(f"Analysis complete! Results saved to {output_dir}")
    return results

# Run comprehensive analysis
results = comprehensive_analysis_pipeline(
    historical_data_path="data/paleo_recon_data.xlsx",
    climate_data_path="data/tree_ring/",
    output_dir="outputs/comprehensive_analysis"
)

# Display summary
print("\nAnalysis Summary:")
print(f"Correlation: {results['correlation_analysis']['basic']['correlation']:.3f}")
print(f"Kappa: {results['mismatch_analysis']['kappa']:.3f}")
print(f"Accuracy: {results['mismatch_analysis']['accuracy']:.3f}")
print(f"Recommendations: {len(results['recommendations'])}")
```

## Next Steps

After completing these advanced examples, explore:

- **[ABM Simulation](abm-simulation.md)**: Agent-based modeling examples
- **[API Reference](../api/)**: Detailed function documentation
- **[Development Guide](../development/)**: Contributing and extending the library
