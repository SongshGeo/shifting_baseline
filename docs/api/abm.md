# Agent-Based Model Module

The `past1000.abm` module provides a comprehensive agent-based modeling framework for simulating climate event recording and collective memory formation.

## Overview

This module implements:
- Climate observer agents with realistic behavior patterns
- Collective memory simulation
- Bias analysis in historical recording
- Monte Carlo simulation capabilities
- Statistical validation of model results

## Core Classes

### ClimateObservingModel

The main model class that simulates a world with climate observers who record extreme climate events.

```python
from past1000.abm import ClimateObservingModel
```

#### Constructor

```python
ClimateObservingModel(*args, **kwargs)
```

**Key Parameters:**
- `years`: Total simulation years (default: 100)
- `max_age`: Maximum age of an observer (default: 40)
- `new_agents`: Number of new agents per step (default: 10)
- `min_age`: Minimum age for recording events (default: 10)
- `memory_baseline`: Baseline for perception ("personal", "model", "collective")
- `loss_rate`: Event loss rate (0-1, default: 0.2)

#### Key Properties

##### `climate_now`
Current climate value at the current tick (Z-score).

```python
current_climate = model.climate_now
print(f"Current climate Z-score: {current_climate:.3f}")
```

##### `climate_series`
Full climate time series.

```python
climate_data = model.climate_series
print(f"Climate series length: {len(climate_data)}")
```

##### `collective_memory_climate`
Mean of recorded events per year (cached per tick).

```python
collective_memory = model.collective_memory_climate
print(f"Collective memory years: {len(collective_memory)}")
```

##### `climate_df`
Climate DataFrame with both objective and collective memory climate.

```python
climate_df = model.climate_df
print(climate_df.head())
```

#### Key Methods

##### `step()`
Advance the model by one tick.

```python
# Run simulation
for tick in range(1000):
    model.step()
    if not model.running:
        break
```

##### `get_corr_curve(window_length=100, min_window=2, corr_method="kendall", **rolling_kwargs)`
Get the correlation curve of the model.

**Parameters:**
- `window_length`: Maximum window length
- `min_window`: Minimum window length
- `corr_method`: Correlation method
- `**rolling_kwargs`: Additional rolling parameters

**Returns:**
- `pd.DataFrame`: Correlation results

**Example:**
```python
# Get correlation curve
corr_curve = model.get_corr_curve(
    window_length=50,
    min_window=5,
    corr_method="kendall"
)

print(corr_curve.head())
```

##### `archive_it(extreme)`
Record an extreme climate event.

**Parameters:**
- `extreme`: The classified extreme event level

**Example:**
```python
# Record extreme event
model.archive_it(2)  # Record severe wet event
```

### ClimateObserver

Individual climate observer agent.

```python
from past1000.abm import ClimateObserver
```

#### Constructor

```python
ClimateObserver(*args, max_age=40, **kwargs)
```

**Parameters:**
- `max_age`: Maximum age for the observer
- `min_age`: Minimum age to start recording events

#### Key Methods

##### `perceive(climate)`
Perceive the z-score of the current climate.

**Parameters:**
- `climate`: Current climate Z-score value

**Returns:**
- `float`: Z-score of the current climate

**Example:**
```python
# Perceive climate
z_score = observer.perceive(current_climate)
print(f"Perceived Z-score: {z_score:.3f}")
```

##### `write_down(z_score, scale=1, f0=0.1)`
Decide whether to record an extreme event.

**Parameters:**
- `z_score`: Standardized z-score of the event
- `scale`: Scale for the z-score
- `f0`: Base probability to record (0 < f0 < 0.5)

**Returns:**
- `bool`: Whether the event is recorded

**Example:**
```python
# Decide whether to record
should_record = observer.write_down(z_score, scale=1.0, f0=0.15)
if should_record:
    print("Event recorded!")
```

##### `step()`
Update observer state at each step.

```python
# Update observer
observer.step()
```

## Model Configuration

### Basic Configuration

```python
# Basic model setup
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

# Create model
model = ClimateObservingModel(**config["model"])
```

### Advanced Configuration

```python
# Advanced configuration with multiple parameters
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
```

## Simulation Patterns

### 1. Basic Simulation

```python
def basic_simulation(years=100, max_age=40, new_agents=10):
    """Run a basic simulation."""

    # Create model
    model = ClimateObservingModel(
        years=years,
        max_age=max_age,
        new_agents=new_agents
    )

    # Run simulation
    while model.running:
        model.step()

    return model

# Run simulation
model = basic_simulation(years=200)
print(f"Simulation completed. Final tick: {model.time.tick}")
```

### 2. Parameter Sweep

```python
def parameter_sweep(param_name, param_values, base_config):
    """Run simulations with different parameter values."""

    results = []

    for value in param_values:
        # Update configuration
        config = base_config.copy()
        config["model"][param_name] = value

        # Run simulation
        model = ClimateObservingModel(**config["model"])
        while model.running:
            model.step()

        # Collect results
        corr_curve = model.get_corr_curve()
        final_corr = corr_curve["kendall"].iloc[-1]

        results.append({
            param_name: value,
            "final_correlation": final_corr,
            "n_agents": len(model.agents)
        })

    return pd.DataFrame(results)

# Sweep loss rate parameter
loss_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
sweep_results = parameter_sweep("loss_rate", loss_rates, base_config)
print(sweep_results)
```

### 3. Monte Carlo Analysis

```python
def monte_carlo_analysis(n_runs=100, config=None):
    """Run multiple simulations for statistical analysis."""

    if config is None:
        config = {"model": {"years": 100, "max_age": 40, "new_agents": 10}}

    results = []

    for run in range(n_runs):
        # Create model
        model = ClimateObservingModel(**config["model"])

        # Run simulation
        while model.running:
            model.step()

        # Collect results
        corr_curve = model.get_corr_curve()
        mismatch_report = model.mismatch_report

        results.append({
            "run": run,
            "final_correlation": corr_curve["kendall"].iloc[-1],
            "kappa": mismatch_report.get_statistics_summary()["kappa"],
            "accuracy": mismatch_report.get_statistics_summary()["accuracy"]
        })

    return pd.DataFrame(results)

# Run Monte Carlo analysis
mc_results = monte_carlo_analysis(n_runs=50)
print(f"Mean correlation: {mc_results['final_correlation'].mean():.3f}")
print(f"Std correlation: {mc_results['final_correlation'].std():.3f}")
```

## Analysis and Visualization

### 1. Correlation Analysis

```python
def analyze_correlations(model):
    """Analyze correlation patterns in the model."""

    # Get correlation curve
    corr_curve = model.get_corr_curve(window_length=50)

    # Plot results
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Correlation curve
    axes[0, 0].plot(corr_curve.index, corr_curve["kendall"])
    axes[0, 0].set_title("Correlation Curve")
    axes[0, 0].set_xlabel("Window Size")
    axes[0, 0].set_ylabel("Correlation")

    # P-values
    axes[0, 1].plot(corr_curve.index, corr_curve["p_value"])
    axes[0, 1].set_title("P-values")
    axes[0, 1].set_xlabel("Window Size")
    axes[0, 1].set_ylabel("P-value")

    # Sample sizes
    axes[1, 0].plot(corr_curve.index, corr_curve["n_samples"])
    axes[1, 0].set_title("Sample Sizes")
    axes[1, 0].set_xlabel("Window Size")
    axes[1, 0].set_ylabel("N Samples")

    # Climate time series
    climate_df = model.climate_df
    axes[1, 1].plot(climate_df.index, climate_df["climate"], label="Objective")
    axes[1, 1].plot(climate_df.index, climate_df["collective_memory_climate"], label="Collective")
    axes[1, 1].set_title("Climate Time Series")
    axes[1, 1].set_xlabel("Year")
    axes[1, 1].set_ylabel("Z-score")
    axes[1, 1].legend()

    plt.tight_layout()
    return fig

# Analyze model
fig = analyze_correlations(model)
plt.show()
```

### 2. Agent Behavior Analysis

```python
def analyze_agent_behavior(model):
    """Analyze individual agent behavior patterns."""

    agent_data = []

    for agent in model.agents:
        if hasattr(agent, 'memory') and len(agent.memory) > 0:
            agent_data.append({
                "age": agent.age(),
                "memory_length": len(agent.memory),
                "memory_mean": agent.memory.mean(),
                "memory_std": agent.memory.std(),
                "is_recording": agent.age() >= model._min_age
            })

    return pd.DataFrame(agent_data)

# Analyze agent behavior
agent_df = analyze_agent_behavior(model)
print(agent_df.describe())
```

### 3. Mismatch Analysis

```python
def analyze_mismatches(model):
    """Analyze mismatches between objective and collective memory."""

    # Get mismatch report
    mismatch_report = model.mismatch_report

    # Analyze error patterns
    error_df = mismatch_report.analyze_error_patterns()

    # Generate visualization
    fig = mismatch_report.generate_report_figure(figsize=(12, 6))

    return mismatch_report, error_df, fig

# Analyze mismatches
mismatch_report, error_df, fig = analyze_mismatches(model)
plt.show()
```

## Batch Processing

### 1. Multiple Runs

```python
def batch_run(config, n_runs=10, parallel=False):
    """Run multiple simulations in batch."""

    if parallel:
        from multiprocessing import Pool
        from functools import partial

        def run_single_simulation(run_id):
            model = ClimateObservingModel(**config["model"])
            while model.running:
                model.step()
            return model

        with Pool() as pool:
            models = pool.map(run_single_simulation, range(n_runs))
    else:
        models = []
        for run in range(n_runs):
            model = ClimateObservingModel(**config["model"])
            while model.running:
                model.step()
            models.append(model)

    return models

# Run batch simulations
models = batch_run(config, n_runs=20, parallel=True)
print(f"Completed {len(models)} simulations")
```

### 2. Experiment Management

```python
def run_experiment(experiment_name, configs, n_runs=10):
    """Run a complete experiment with multiple configurations."""

    results = []

    for config_name, config in configs.items():
        print(f"Running {config_name}...")

        # Run multiple simulations
        models = batch_run(config, n_runs=n_runs)

        # Collect results
        for i, model in enumerate(models):
            corr_curve = model.get_corr_curve()
            mismatch_report = model.mismatch_report

            results.append({
                "experiment": experiment_name,
                "config": config_name,
                "run": i,
                "final_correlation": corr_curve["kendall"].iloc[-1],
                "kappa": mismatch_report.get_statistics_summary()["kappa"]
            })

    return pd.DataFrame(results)

# Define experiment configurations
configs = {
    "personal_memory": {"model": {"memory_baseline": "personal"}},
    "model_memory": {"model": {"memory_baseline": "model"}},
    "collective_memory": {"model": {"memory_baseline": "collective"}}
}

# Run experiment
experiment_results = run_experiment("memory_baseline", configs, n_runs=20)
print(experiment_results.groupby("config")["final_correlation"].describe())
```

## Integration with Other Modules

### With Calibration Module

```python
from past1000.calibration import MismatchReport

# Get mismatch report from model
mismatch_report = model.mismatch_report

# Analyze error patterns
error_df = mismatch_report.analyze_error_patterns()

# Generate visualization
fig = mismatch_report.generate_report_figure()
```

### With Comparison Module

```python
from past1000.compare import compare_corr

# Compare objective and collective memory
climate_df = model.climate_df
r, p, n = compare_corr(
    climate_df["climate"],
    climate_df["collective_memory_climate"],
    corr_method="kendall"
)

print(f"Correlation: {r:.3f}, P-value: {p:.3f}")
```

### With Data Module

```python
from past1000.data import HistoricalRecords

# Compare with historical data
history = HistoricalRecords(
    shp_path="data/regions.shp",
    data_path="data/historical_data.xlsx"
)

# Compare model results with historical data
historical_series = history.aggregate("mean")
model_series = model.climate_df["collective_memory_climate"]

r, p, n = compare_corr(historical_series, model_series)
```

## Performance Optimization

### Memory Management

```python
def memory_efficient_simulation(years=100, max_age=40, new_agents=10):
    """Memory-efficient simulation for large models."""

    # Create model with memory management
    model = ClimateObservingModel(
        years=years,
        max_age=max_age,
        new_agents=new_agents
    )

    # Run with periodic cleanup
    for tick in range(years):
        model.step()

        # Cleanup every 100 ticks
        if tick % 100 == 0:
            import gc
            gc.collect()

    return model
```

### Parallel Processing

```python
from multiprocessing import Pool
from functools import partial

def parallel_simulation(config, n_runs=10, n_processes=4):
    """Run simulations in parallel."""

    def run_single(config):
        model = ClimateObservingModel(**config["model"])
        while model.running:
            model.step()
        return model

    with Pool(n_processes) as pool:
        models = pool.map(run_single, [config] * n_runs)

    return models
```

## Error Handling and Validation

### Model Validation

```python
def validate_model(model):
    """Validate model state and results."""

    # Check basic properties
    assert hasattr(model, 'climate_series'), "Model missing climate_series"
    assert hasattr(model, 'collective_memory_climate'), "Model missing collective_memory_climate"

    # Check data consistency
    climate_df = model.climate_df
    assert len(climate_df) > 0, "Empty climate DataFrame"
    assert not climate_df.isna().all().all(), "All NaN values in climate DataFrame"

    # Check correlation curve
    corr_curve = model.get_corr_curve()
    assert len(corr_curve) > 0, "Empty correlation curve"

    return True

# Validate model
try:
    validate_model(model)
    print("Model validation passed")
except AssertionError as e:
    print(f"Model validation failed: {e}")
```

### Robust Simulation

```python
def robust_simulation(config, max_retries=3):
    """Run simulation with error handling and retries."""

    for attempt in range(max_retries):
        try:
            model = ClimateObservingModel(**config["model"])

            while model.running:
                model.step()

            # Validate results
            validate_model(model)

            return model

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise
            continue
```
