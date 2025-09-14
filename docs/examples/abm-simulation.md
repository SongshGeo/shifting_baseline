# ABM Simulation Examples

This guide demonstrates how to use the Agent-Based Model (ABM) module in Shifting Baseline to simulate climate event recording and collective memory formation.

## Example 1: Basic ABM Simulation

### Simple Climate Observer Simulation

```python
from shifting_baseline.abm import ClimateObservingModel, ClimateObserver
import matplotlib.pyplot as plt
import pandas as pd

# Basic model configuration
config = {
    "model": {
        "years": 100,
        "max_age": 40,
        "new_agents": 10,
        "min_age": 10,
        "memory_baseline": "personal",
        "loss_rate": 0.2
    }
}

# Create and run model
model = ClimateObservingModel(**config["model"])

# Run simulation
print("Running simulation...")
while model.running:
    model.step()

print(f"Simulation completed. Final tick: {model.time.tick}")
print(f"Total agents created: {len(model.agents)}")

# Analyze results
climate_df = model.climate_df
print(f"Climate data shape: {climate_df.shape}")
print(f"Objective climate range: {climate_df['climate'].min():.3f} to {climate_df['climate'].max():.3f}")
print(f"Collective memory range: {climate_df['collective_memory_climate'].min():.3f} to {climate_df['collective_memory_climate'].max():.3f}")
```

### Visualize Basic Results

```python
# Plot climate time series
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Objective vs Collective Memory
axes[0, 0].plot(climate_df.index, climate_df['climate'], label='Objective', alpha=0.7)
axes[0, 0].plot(climate_df.index, climate_df['collective_memory_climate'], label='Collective Memory', alpha=0.7)
axes[0, 0].set_title('Objective vs Collective Memory Climate')
axes[0, 0].set_xlabel('Year')
axes[0, 0].set_ylabel('Z-score')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Correlation over time
correlations = []
window_size = 20
for i in range(window_size, len(climate_df)):
    window_data = climate_df.iloc[i-window_size:i]
    r = window_data['climate'].corr(window_data['collective_memory_climate'])
    correlations.append(r)

axes[0, 1].plot(climate_df.index[window_size:], correlations, alpha=0.7)
axes[0, 1].set_title('Rolling Correlation (20-year window)')
axes[0, 1].set_xlabel('Year')
axes[0, 1].set_ylabel('Correlation')
axes[0, 1].grid(True, alpha=0.3)

# Agent age distribution
agent_ages = [agent.age() for agent in model.agents if hasattr(agent, 'age')]
axes[1, 0].hist(agent_ages, bins=20, alpha=0.7)
axes[1, 0].set_title('Agent Age Distribution')
axes[1, 0].set_xlabel('Age')
axes[1, 0].set_ylabel('Count')

# Memory length distribution
memory_lengths = [len(agent.memory) for agent in model.agents if hasattr(agent, 'memory')]
axes[1, 1].hist(memory_lengths, bins=20, alpha=0.7)
axes[1, 1].set_title('Agent Memory Length Distribution')
axes[1, 1].set_xlabel('Memory Length')
axes[1, 1].set_ylabel('Count')

plt.tight_layout()
plt.show()
```

## Example 2: Parameter Sensitivity Analysis

### Memory Baseline Comparison

```python
def compare_memory_baselines(years=100, n_runs=10):
    """Compare different memory baseline strategies."""

    baselines = ["personal", "model", "collective"]
    results = {}

    for baseline in baselines:
        print(f"Testing {baseline} baseline...")

        baseline_results = []

        for run in range(n_runs):
            # Create model with specific baseline
            model = ClimateObservingModel(
                years=years,
                max_age=40,
                new_agents=10,
                min_age=10,
                memory_baseline=baseline,
                loss_rate=0.2
            )

            # Run simulation
            while model.running:
                model.step()

            # Calculate final correlation
            climate_df = model.climate_df
            final_corr = climate_df['climate'].corr(climate_df['collective_memory_climate'])

            # Get mismatch statistics
            mismatch_report = model.mismatch_report
            stats = mismatch_report.get_statistics_summary()

            baseline_results.append({
                "run": run,
                "correlation": final_corr,
                "kappa": stats["kappa"],
                "accuracy": stats["accuracy"]
            })

        results[baseline] = pd.DataFrame(baseline_results)

    return results

# Run comparison
baseline_results = compare_memory_baselines(years=50, n_runs=20)

# Analyze results
print("Memory Baseline Comparison:")
for baseline, df in baseline_results.items():
    print(f"\n{baseline.upper()}:")
    print(f"  Mean correlation: {df['correlation'].mean():.3f} ± {df['correlation'].std():.3f}")
    print(f"  Mean kappa: {df['kappa'].mean():.3f} ± {df['kappa'].std():.3f}")
    print(f"  Mean accuracy: {df['accuracy'].mean():.3f} ± {df['accuracy'].std():.3f}")

# Visualize comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

baselines = list(baseline_results.keys())
correlations = [baseline_results[b]['correlation'].values for b in baselines]
kappas = [baseline_results[b]['kappa'].values for b in baselines]
accuracies = [baseline_results[b]['accuracy'].values for b in baselines]

axes[0].boxplot(correlations, labels=baselines)
axes[0].set_title('Correlation by Memory Baseline')
axes[0].set_ylabel('Correlation')

axes[1].boxplot(kappas, labels=baselines)
axes[1].set_title('Kappa by Memory Baseline')
axes[1].set_ylabel('Kappa')

axes[2].boxplot(accuracies, labels=baselines)
axes[2].set_title('Accuracy by Memory Baseline')
axes[2].set_ylabel('Accuracy')

plt.tight_layout()
plt.show()
```

## Example 3: Loss Rate Sensitivity

### Event Loss Rate Analysis

```python
def analyze_loss_rate_sensitivity(loss_rates, years=100, n_runs=10):
    """Analyze sensitivity to event loss rate."""

    results = []

    for loss_rate in loss_rates:
        print(f"Testing loss rate: {loss_rate}")

        for run in range(n_runs):
            # Create model with specific loss rate
            model = ClimateObservingModel(
                years=years,
                max_age=40,
                new_agents=10,
                min_age=10,
                memory_baseline="personal",
                loss_rate=loss_rate
            )

            # Run simulation
            while model.running:
                model.step()

            # Calculate metrics
            climate_df = model.climate_df
            final_corr = climate_df['climate'].corr(climate_df['collective_memory_climate'])

            mismatch_report = model.mismatch_report
            stats = mismatch_report.get_statistics_summary()

            results.append({
                "loss_rate": loss_rate,
                "run": run,
                "correlation": final_corr,
                "kappa": stats["kappa"],
                "accuracy": stats["accuracy"]
            })

    return pd.DataFrame(results)

# Test different loss rates
loss_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
loss_rate_results = analyze_loss_rate_sensitivity(loss_rates, years=50, n_runs=15)

# Analyze results
loss_rate_summary = loss_rate_results.groupby('loss_rate').agg({
    'correlation': ['mean', 'std'],
    'kappa': ['mean', 'std'],
    'accuracy': ['mean', 'std']
}).round(3)

print("Loss Rate Sensitivity Analysis:")
print(loss_rate_summary)

# Visualize results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Correlation vs Loss Rate
corr_means = loss_rate_results.groupby('loss_rate')['correlation'].mean()
corr_stds = loss_rate_results.groupby('loss_rate')['correlation'].std()
axes[0].errorbar(loss_rates, corr_means, yerr=corr_stds, marker='o', capsize=5)
axes[0].set_title('Correlation vs Loss Rate')
axes[0].set_xlabel('Loss Rate')
axes[0].set_ylabel('Correlation')
axes[0].grid(True, alpha=0.3)

# Kappa vs Loss Rate
kappa_means = loss_rate_results.groupby('loss_rate')['kappa'].mean()
kappa_stds = loss_rate_results.groupby('loss_rate')['kappa'].std()
axes[1].errorbar(loss_rates, kappa_means, yerr=kappa_stds, marker='o', capsize=5)
axes[1].set_title('Kappa vs Loss Rate')
axes[1].set_xlabel('Loss Rate')
axes[1].set_ylabel('Kappa')
axes[1].grid(True, alpha=0.3)

# Accuracy vs Loss Rate
acc_means = loss_rate_results.groupby('loss_rate')['accuracy'].mean()
acc_stds = loss_rate_results.groupby('loss_rate')['accuracy'].std()
axes[2].errorbar(loss_rates, acc_means, yerr=acc_stds, marker='o', capsize=5)
axes[2].set_title('Accuracy vs Loss Rate')
axes[2].set_xlabel('Loss Rate')
axes[2].set_ylabel('Accuracy')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Example 4: Agent Behavior Analysis

### Individual Agent Analysis

```python
def analyze_agent_behavior(model):
    """Analyze individual agent behavior patterns."""

    agent_data = []

    for i, agent in enumerate(model.agents):
        if hasattr(agent, 'memory') and len(agent.memory) > 0:
            agent_data.append({
                "agent_id": i,
                "age": agent.age(),
                "memory_length": len(agent.memory),
                "memory_mean": agent.memory.mean(),
                "memory_std": agent.memory.std(),
                "is_recording": agent.age() >= model._min_age,
                "memory_baseline": model.p.memory_baseline
            })

    return pd.DataFrame(agent_data)

# Analyze agent behavior
agent_df = analyze_agent_behavior(model)

print("Agent Behavior Analysis:")
print(f"Total agents: {len(agent_df)}")
print(f"Recording agents: {agent_df['is_recording'].sum()}")
print(f"Average age: {agent_df['age'].mean():.1f}")
print(f"Average memory length: {agent_df['memory_length'].mean():.1f}")

# Visualize agent behavior
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Age vs Memory Length
axes[0, 0].scatter(agent_df['age'], agent_df['memory_length'], alpha=0.6)
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Memory Length')
axes[0, 0].set_title('Age vs Memory Length')
axes[0, 0].grid(True, alpha=0.3)

# Memory Statistics
axes[0, 1].scatter(agent_df['memory_mean'], agent_df['memory_std'], alpha=0.6)
axes[0, 1].set_xlabel('Memory Mean')
axes[0, 1].set_ylabel('Memory Std')
axes[0, 1].set_title('Memory Statistics')
axes[0, 1].grid(True, alpha=0.3)

# Age Distribution
axes[1, 0].hist(agent_df['age'], bins=20, alpha=0.7)
axes[1, 0].set_xlabel('Age')
axes[1, 0].set_ylabel('Count')
axes[1, 0].set_title('Age Distribution')
axes[1, 0].grid(True, alpha=0.3)

# Memory Length Distribution
axes[1, 1].hist(agent_df['memory_length'], bins=20, alpha=0.7)
axes[1, 1].set_xlabel('Memory Length')
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_title('Memory Length Distribution')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Example 5: Advanced Model Configuration

### Complex Model Setup

```python
def create_advanced_model(config):
    """Create model with advanced configuration."""

    # Create model
    model = ClimateObservingModel(**config["model"])

    # Add custom properties
    model.custom_properties = {
        "analysis_start": model.time.tick,
        "config": config
    }

    return model

# Advanced configuration
advanced_config = {
    "model": {
        "years": 200,
        "max_age": 50,
        "new_agents": 15,
        "min_age": 12,
        "memory_baseline": "collective",
        "loss_rate": 0.15,
        "perception_scale": 1.2,
        "recording_threshold": 0.1
    }
}

# Create and run advanced model
advanced_model = create_advanced_model(advanced_config)

print("Running advanced simulation...")
while advanced_model.running:
    advanced_model.step()

print(f"Advanced simulation completed. Final tick: {advanced_model.time.tick}")

# Get correlation curve
corr_curve = advanced_model.get_corr_curve(window_length=50, min_window=5)
print(f"Correlation curve shape: {corr_curve.shape}")

# Visualize correlation curve
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(corr_curve.index, corr_curve['kendall'], 'o-', alpha=0.7, label='Kendall')
ax.plot(corr_curve.index, corr_curve['p_value'], 's-', alpha=0.7, label='P-value')
ax.set_xlabel('Window Size')
ax.set_ylabel('Value')
ax.set_title('Correlation Curve Analysis')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()
```

## Example 6: Batch Processing and Experimentation

### Multiple Model Runs

```python
def batch_run_models(configs, n_runs=10):
    """Run multiple models in batch."""

    all_results = []

    for config_name, config in configs.items():
        print(f"Running {config_name}...")

        config_results = []

        for run in range(n_runs):
            # Create model
            model = ClimateObservingModel(**config["model"])

            # Run simulation
            while model.running:
                model.step()

            # Collect results
            climate_df = model.climate_df
            final_corr = climate_df['climate'].corr(climate_df['collective_memory_climate'])

            mismatch_report = model.mismatch_report
            stats = mismatch_report.get_statistics_summary()

            config_results.append({
                "config": config_name,
                "run": run,
                "correlation": final_corr,
                "kappa": stats["kappa"],
                "accuracy": stats["accuracy"],
                "n_agents": len(model.agents)
            })

        all_results.extend(config_results)

    return pd.DataFrame(all_results)

# Define multiple configurations
configs = {
    "baseline": {
        "model": {
            "years": 100,
            "max_age": 40,
            "new_agents": 10,
            "min_age": 10,
            "memory_baseline": "personal",
            "loss_rate": 0.2
        }
    },
    "high_agents": {
        "model": {
            "years": 100,
            "max_age": 40,
            "new_agents": 20,
            "min_age": 10,
            "memory_baseline": "personal",
            "loss_rate": 0.2
        }
    },
    "low_loss": {
        "model": {
            "years": 100,
            "max_age": 40,
            "new_agents": 10,
            "min_age": 10,
            "memory_baseline": "personal",
            "loss_rate": 0.1
        }
    },
    "collective_memory": {
        "model": {
            "years": 100,
            "max_age": 40,
            "new_agents": 10,
            "min_age": 10,
            "memory_baseline": "collective",
            "loss_rate": 0.2
        }
    }
}

# Run batch experiments
batch_results = batch_run_models(configs, n_runs=15)

# Analyze results
print("Batch Experiment Results:")
summary = batch_results.groupby('config').agg({
    'correlation': ['mean', 'std'],
    'kappa': ['mean', 'std'],
    'accuracy': ['mean', 'std'],
    'n_agents': ['mean', 'std']
}).round(3)

print(summary)

# Visualize results
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Correlation by configuration
configs_list = batch_results['config'].unique()
corr_data = [batch_results[batch_results['config'] == c]['correlation'].values for c in configs_list]
axes[0, 0].boxplot(corr_data, labels=configs_list)
axes[0, 0].set_title('Correlation by Configuration')
axes[0, 0].set_ylabel('Correlation')
axes[0, 0].tick_params(axis='x', rotation=45)

# Kappa by configuration
kappa_data = [batch_results[batch_results['config'] == c]['kappa'].values for c in configs_list]
axes[0, 1].boxplot(kappa_data, labels=configs_list)
axes[0, 1].set_title('Kappa by Configuration')
axes[0, 1].set_ylabel('Kappa')
axes[0, 1].tick_params(axis='x', rotation=45)

# Accuracy by configuration
acc_data = [batch_results[batch_results['config'] == c]['accuracy'].values for c in configs_list]
axes[1, 0].boxplot(acc_data, labels=configs_list)
axes[1, 0].set_title('Accuracy by Configuration')
axes[1, 0].set_ylabel('Accuracy')
axes[1, 0].tick_params(axis='x', rotation=45)

# Agent count by configuration
agent_data = [batch_results[batch_results['config'] == c]['n_agents'].values for c in configs_list]
axes[1, 1].boxplot(agent_data, labels=configs_list)
axes[1, 1].set_title('Agent Count by Configuration')
axes[1, 1].set_ylabel('Number of Agents')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
```

## Example 7: Real-World Data Integration

### Compare with Historical Data

```python
from shifting_baseline.data import HistoricalRecords
from shifting_baseline.compare import compare_corr

def compare_with_historical_data(model, historical_data_path):
    """Compare model results with real historical data."""

    # Load historical data
    history = HistoricalRecords(
        shp_path="data/north_china_precip_regions.shp",
        data_path=historical_data_path,
        region="华北地区"
    )

    # Aggregate historical data
    historical_series = history.aggregate("mean")

    # Get model results
    climate_df = model.climate_df

    # Align data
    aligned_hist, aligned_model = historical_series.align(
        climate_df['collective_memory_climate'],
        join='inner'
    )

    # Calculate correlation
    r, p, n = compare_corr(aligned_hist, aligned_model, corr_method="kendall")

    # Create comparison plot
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    # Time series comparison
    axes[0].plot(aligned_hist.index, aligned_hist.values, label='Historical Data', alpha=0.7)
    axes[0].plot(aligned_model.index, aligned_model.values, label='Model Results', alpha=0.7)
    axes[0].set_title(f'Historical vs Model Data (r={r:.3f}, p={p:.3f})')
    axes[0].set_xlabel('Year')
    axes[0].set_ylabel('Value')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Scatter plot
    axes[1].scatter(aligned_hist.values, aligned_model.values, alpha=0.6)
    axes[1].plot([aligned_hist.min(), aligned_hist.max()],
                 [aligned_hist.min(), aligned_hist.max()], 'r--', alpha=0.8)
    axes[1].set_xlabel('Historical Data')
    axes[1].set_ylabel('Model Results')
    axes[1].set_title('Historical vs Model Scatter Plot')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return {
        "correlation": r,
        "p_value": p,
        "samples": n,
        "historical_data": aligned_hist,
        "model_data": aligned_model
    }

# Run model
model = ClimateObservingModel(
    years=100,
    max_age=40,
    new_agents=10,
    min_age=10,
    memory_baseline="personal",
    loss_rate=0.2
)

while model.running:
    model.step()

# Compare with historical data
comparison_results = compare_with_historical_data(
    model,
    "data/paleo_recon_data.xlsx"
)

print(f"Model-Historical Correlation: {comparison_results['correlation']:.3f}")
print(f"P-value: {comparison_results['p_value']:.3f}")
print(f"Sample size: {comparison_results['samples']}")
```

## Example 8: Comprehensive ABM Analysis

### Complete ABM Research Workflow

```python
def comprehensive_abm_analysis(config, output_dir):
    """Complete ABM analysis workflow."""

    from pathlib import Path
    import json
    from datetime import datetime

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize results
    results = {
        "analysis_date": datetime.now().isoformat(),
        "config": config,
        "simulation_results": {},
        "agent_analysis": {},
        "correlation_analysis": {},
        "recommendations": []
    }

    # 1. Run simulation
    print("Step 1: Running ABM simulation...")

    model = ClimateObservingModel(**config["model"])

    # Run simulation
    while model.running:
        model.step()

    results["simulation_results"] = {
        "final_tick": model.time.tick,
        "total_agents": len(model.agents),
        "simulation_years": config["model"]["years"]
    }

    # 2. Analyze agent behavior
    print("Step 2: Analyzing agent behavior...")

    agent_df = analyze_agent_behavior(model)
    results["agent_analysis"] = {
        "total_agents": len(agent_df),
        "recording_agents": agent_df['is_recording'].sum(),
        "average_age": agent_df['age'].mean(),
        "average_memory_length": agent_df['memory_length'].mean(),
        "memory_statistics": {
            "mean": agent_df['memory_mean'].mean(),
            "std": agent_df['memory_std'].mean()
        }
    }

    # 3. Analyze correlations
    print("Step 3: Analyzing correlations...")

    climate_df = model.climate_df
    final_corr = climate_df['climate'].corr(climate_df['collective_memory_climate'])

    # Get correlation curve
    corr_curve = model.get_corr_curve(window_length=50, min_window=5)

    results["correlation_analysis"] = {
        "final_correlation": final_corr,
        "correlation_curve": corr_curve.to_dict(),
        "max_correlation": corr_curve['kendall'].max(),
        "min_correlation": corr_curve['kendall'].min()
    }

    # 4. Mismatch analysis
    print("Step 4: Performing mismatch analysis...")

    mismatch_report = model.mismatch_report
    mismatch_stats = mismatch_report.get_statistics_summary()
    results["mismatch_analysis"] = mismatch_stats

    # 5. Generate recommendations
    print("Step 5: Generating recommendations...")

    if final_corr < 0.3:
        results["recommendations"].append("Low correlation between objective and collective memory. Consider adjusting model parameters.")

    if mismatch_stats["kappa"] < 0.4:
        results["recommendations"].append("Poor classification agreement. Review classification thresholds or model parameters.")

    if agent_df['is_recording'].sum() < len(agent_df) * 0.5:
        results["recommendations"].append("Low proportion of recording agents. Consider adjusting min_age parameter.")

    # 6. Save results
    print("Step 6: Saving results...")

    # Save data
    climate_df.to_csv(output_dir / "climate_data.csv")
    agent_df.to_csv(output_dir / "agent_data.csv")
    corr_curve.to_csv(output_dir / "correlation_curve.csv")

    # Save summary
    with open(output_dir / "abm_analysis_summary.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Generate visualizations
    fig = model.mismatch_report.generate_report_figure(figsize=(15, 10))
    fig.savefig(output_dir / "mismatch_analysis.png", dpi=300, bbox_inches='tight')

    print(f"ABM analysis complete! Results saved to {output_dir}")
    return results

# Run comprehensive analysis
abm_results = comprehensive_abm_analysis(
    config=advanced_config,
    output_dir="outputs/abm_analysis"
)

# Display summary
print("\nABM Analysis Summary:")
print(f"Final correlation: {abm_results['correlation_analysis']['final_correlation']:.3f}")
print(f"Kappa: {abm_results['mismatch_analysis']['kappa']:.3f}")
print(f"Total agents: {abm_results['agent_analysis']['total_agents']}")
print(f"Recording agents: {abm_results['agent_analysis']['recording_agents']}")
print(f"Recommendations: {len(abm_results['recommendations'])}")
```

## Next Steps

After completing these ABM examples, explore:

- **[Advanced Analysis](advanced-analysis.md)**: More complex analysis patterns
- **[API Reference](../api/)**: Detailed function documentation
- **[Development Guide](../development/)**: Contributing and extending the library

## Troubleshooting

### Common ABM Issues

1. **Model Not Running**
   - Check configuration parameters
   - Ensure valid parameter ranges
   - Verify model initialization

2. **Low Correlations**
   - Adjust model parameters
   - Check data quality
   - Consider different memory baselines

3. **Memory Issues**
   - Reduce simulation years
   - Decrease number of agents
   - Use chunked processing

4. **Agent Behavior Issues**
   - Check agent age parameters
   - Verify memory baseline settings
   - Adjust loss rate parameters
