# Installation

This guide will help you install Past1000 and its dependencies.

## Prerequisites

- Python 3.11+ (required)
- Poetry (recommended) or pip
- Git (for development)

## Installation Methods

### Method 1: Using Poetry (Recommended)

Poetry is the recommended package manager for this project as it ensures consistent dependency management.

```bash
# Clone the repository
git clone https://github.com/SongshGeo/past1000.git
cd past1000

# Install dependencies using Poetry
poetry install

# Activate the virtual environment
poetry shell
```

### Method 2: Using pip

If you prefer using pip, you can install the package in development mode:

```bash
# Clone the repository
git clone https://github.com/SongshGeo/past1000.git
cd past1000

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
```

## Dependencies

Past1000 relies on several scientific Python packages:

### Core Dependencies
- **xarray** (>=2023): Multi-dimensional labeled arrays
- **pandas** (>=1.5): Data manipulation and analysis
- **numpy** (>=1.20): Numerical computing
- **matplotlib** (>=3.9.2): Plotting library
- **scipy** (>=1.9): Scientific computing

### Climate Data Processing
- **netcdf4** (>=1.7): NetCDF file format support
- **rioxarray** (>=0.17.0): Geospatial raster data
- **xclim** (^0.53.2): Climate data analysis
- **cf-xarray** (*): CF conventions for xarray

### Geospatial Analysis
- **geopandas** (*): Geospatial data processing
- **cartopy** (>=0.24.1): Cartographic projections

### Statistical Analysis
- **scikit-learn** (>=1.0): Machine learning tools
- **arviz** (^0.20.0): Bayesian analysis
- **pymc** (^5.20.1): Probabilistic programming

### Agent-Based Modeling
- **abses** (>=0.7.5): Agent-based simulation framework

### Additional Utilities
- **tqdm** (*): Progress bars
- **hydra-core** (~1.3): Configuration management
- **openpyxl** (*): Excel file support
- **fitter** (^1.7.1): Distribution fitting

## Verification

After installation, verify that Past1000 is working correctly:

```python
# Test basic import
import past1000
print(f"Past1000 version: {past1000.__version__}")

# Test core modules
from past1000.data import HistoricalRecords
from past1000.compare import experiment_corr_2d
from past1000.calibration import MismatchReport

print("✅ All imports successful!")
```

## Troubleshooting

### Common Issues

**1. Poetry Installation Issues**
```bash
# Update Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Clear Poetry cache
poetry cache clear --all pypi
```

**2. NetCDF4 Installation Problems**
```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get install libhdf5-dev libnetcdf-dev

# Or using conda
conda install -c conda-forge netcdf4
```

**3. Cartopy Installation Issues**
```bash
# Install system dependencies
sudo apt-get install libproj-dev proj-data proj-bin
sudo apt-get install libgeos-dev

# Or using conda
conda install -c conda-forge cartopy
```

**4. Memory Issues with Large Datasets**
```bash
# Increase virtual memory
export MALLOC_ARENA_MAX=2
```

### Getting Help

If you encounter issues during installation:

1. Check the [GitHub Issues](https://github.com/SongshGeo/past1000/issues)
2. Create a new issue with:
   - Operating system and version
   - Python version
   - Installation method used
   - Complete error message
3. Contact: [songshgeo@gmail.com](mailto:songshgeo@gmail.com)

## Development Installation

For development work, install with additional development dependencies:

```bash
# Using Poetry
poetry install --with dev

# Using pip
pip install -e ".[dev]"
```

This includes:
- **pytest**: Testing framework
- **pytest-cov**: Coverage reporting
- **black**: Code formatting
- **flake8**: Linting
- **mypy**: Type checking
