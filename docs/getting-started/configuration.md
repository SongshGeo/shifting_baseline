# Configuration Guide

Shifting Baseline uses Hydra for flexible configuration management. This guide covers all configuration options and how to customize them.

## Configuration Structure

The main configuration file is located at `config/config.yaml` and follows this structure:

```yaml
# Main configuration file
ds:                    # Data sources
  noaa: "data/tree_ring/"
  includes: ["shi2018", "yellow2019"]
  atlas:
    shp: "data/north_china_precip_regions.shp"
    file: "data/paleo_recon_data.xlsx"
  out:
    tree_ring: "outputs/tree_ring_data.csv"
    tree_ring_uncertainty: "outputs/tree_ring_uncertainty.csv"

# Analysis parameters
corr_method: "kendall"     # Correlation method: pearson, kendall, spearman
filter_side: "right"       # Filter side: left, right, both
agg_method: "mean"         # Aggregation method
to_std: "mapping"          # Standardization method

# Model parameters
model:
  years: 100
  max_age: 40
  new_agents: 10
  memory_baseline: "personal"  # personal, model, collective
  loss_rate: 0.2
  repeats: 10
  num_process: 4

# Output settings
output_dir: "outputs"
save_figures: true
figure_format: "png"
dpi: 300
```

## Data Source Configuration

### Tree Ring Data (`ds.noaa`)

```yaml
ds:
  noaa: "data/tree_ring/"           # Directory containing tree ring data
  includes:                         # Files to include (partial matching)
    - "shi2018"
    - "yellow2019"
    - "other_reconstruction"
  out:
    tree_ring: "outputs/tree_ring_data.csv"
    tree_ring_uncertainty: "outputs/tree_ring_uncertainty.csv"
```

**File Format Requirements:**
- Text files with tab or space separation
- First column: years (index)
- Second column: reconstruction values
- Files matching `includes` patterns will be processed

### Historical Records (`ds.atlas`)

```yaml
ds:
  atlas:
    shp: "data/north_china_precip_regions.shp"    # Shapefile for regions
    file: "data/paleo_recon_data.xlsx"            # Excel file with historical data
```

**Excel File Structure:**
- Sheet names: Region names (e.g., "华北地区", "东北地区")
- Row index: Years (1000-2020)
- Columns: Station names or grid points
- Values: 1-5 (drought levels) or standardized values

### Instrumental Data (`ds.instrumental`)

```yaml
ds:
  instrumental:
    input: "data/instrumental/"      # Directory with NetCDF files
    output: "outputs/summer_precip.nc"
    agg_months: [7, 8, 9]           # Summer months (JAS)
    agg_method: "sum"                # sum or mean
```

## Analysis Parameters

### Correlation Analysis

```yaml
corr_method: "kendall"              # pearson, kendall, spearman
filter_side: "right"                # left, right, both
sample_threshold: 1.0               # Minimum samples per window
p_threshold: 0.05                   # Significance threshold
std_offset: 0.2                     # Standard deviation offset
penalty: false                      # Apply penalty for small windows
```

### Data Aggregation

```yaml
agg_method: "mean"                  # mean, median, mode, weighted_mean,
                                   # probability_weighted, bayesian
to_std: "mapping"                   # mapping, sampling, null
```

**Aggregation Methods:**
- `mean`: Simple arithmetic mean
- `median`: Median value
- `mode`: Most frequent value
- `weighted_mean`: Weighted average
- `probability_weighted`: Probability-weighted aggregation
- `bayesian`: Bayesian aggregation with spatial correlation

**Standardization Methods:**
- `mapping`: Map levels to standard deviations using predefined mapping
- `sampling`: Generate values from probability distributions
- `null`: No standardization

## Agent-Based Model Configuration

### Model Parameters

```yaml
model:
  years: 100                        # Simulation years
  max_age: 40                       # Maximum agent age
  new_agents: 10                    # New agents per step
  min_age: 10                       # Minimum age for recording
  memory_baseline: "personal"       # personal, model, collective
  loss_rate: 0.2                    # Event loss rate (0-1)
  repeats: 10                       # Number of simulation runs
  num_process: 4                    # Parallel processes
```

### Agent Behavior

```yaml
model:
  memory_baseline: "personal"       # Baseline for perception
  loss_rate: 0.2                    # Probability of event loss
  perception_scale: 1.0             # Scale factor for perception
  recording_threshold: 0.1          # Base recording probability
```

**Memory Baseline Options:**
- `personal`: Use agent's personal memory
- `model`: Use model's climate series
- `collective`: Use collective memory

## Output Configuration

### File Outputs

```yaml
output_dir: "outputs"               # Base output directory
save_figures: true                  # Save generated figures
figure_format: "png"                # png, pdf, svg, jpg
dpi: 300                           # Figure resolution
```

### Logging

```yaml
logging:
  level: "INFO"                     # DEBUG, INFO, WARNING, ERROR
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/shifting_baseline.log"         # Log file path
```

## Environment-Specific Configurations

### Development Environment

```yaml
# config/dev.yaml
model:
  years: 10                         # Shorter runs for testing
  repeats: 2
  num_process: 1

logging:
  level: "DEBUG"
```

### Production Environment

```yaml
# config/prod.yaml
model:
  years: 500
  repeats: 100
  num_process: 8

output_dir: "/data/shifting_baseline/outputs"
logging:
  level: "INFO"
```

## Command Line Overrides

You can override any configuration parameter from the command line:

```bash
# Override specific parameters
python -m shifting_baseline model.years=200 model.repeats=20

# Override multiple parameters
python -m shifting_baseline model.years=200 model.repeats=20 corr_method=pearson

# Use different config file
python -m shifting_baseline --config-name=production

# Override with environment variables
HYDRA_OVERRIDES="model.years=100" python -m shifting_baseline
```

## Configuration Validation

Shifting Baseline includes built-in validation for configuration parameters:

```python
from shifting_baseline.utils.config import validate_config

# Validate configuration
config = load_config("config/config.yaml")
validate_config(config)  # Raises error for invalid parameters
```

## Custom Configuration Files

Create custom configuration files for specific use cases:

```yaml
# config/experiments/experiment1.yaml
defaults:
  - ../config
  - _self_

model:
  years: 200
  memory_baseline: "collective"
  loss_rate: 0.1

corr_method: "spearman"
filter_side: "both"
```

## Best Practices

1. **Use meaningful names** for configuration files
2. **Document custom parameters** in configuration files
3. **Use environment-specific configs** for different deployment scenarios
4. **Validate configurations** before running long simulations
5. **Keep sensitive data** in environment variables, not config files
6. **Version control** your configuration files
7. **Test configurations** with small datasets first

## Troubleshooting

### Common Configuration Issues

**1. File Path Issues**
```yaml
# Use absolute paths or relative to project root
ds:
  noaa: "/absolute/path/to/data/"  # Absolute path
  # or
  noaa: "data/tree_ring/"          # Relative to project root
```

**2. Memory Issues**
```yaml
# Reduce data size for testing
model:
  years: 10
  repeats: 1
  num_process: 1
```

**3. Invalid Parameters**
```python
# Check available parameters
from shifting_baseline.utils.config import get_available_parameters
print(get_available_parameters())
```
