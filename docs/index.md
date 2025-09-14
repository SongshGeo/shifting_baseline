# Past1000

A comprehensive Python library for analyzing historical climate reconstruction data and comparing it with collective memory records from historical documents.

## Overview

Past1000 is designed to bridge the gap between objective climate reconstructions and subjective historical records. The library provides tools for:

- **Data Processing**: Loading and standardizing climate reconstruction data from various sources
- **Historical Analysis**: Processing and analyzing historical drought/flood records
- **Statistical Comparison**: Advanced correlation analysis between different data sources
- **Agent-Based Modeling**: Simulating how historical observers might have recorded climate events
- **Calibration**: Evaluating the accuracy and reliability of different data sources

## Key Features

### 🔬 Scientific Analysis
- Advanced statistical methods for climate data analysis
- Monte Carlo simulations for uncertainty quantification
- Multiple correlation analysis techniques (Pearson, Kendall, Spearman)
- Time series filtering and windowing methods

### 📊 Data Integration
- Support for multiple climate reconstruction datasets
- Historical document processing and classification
- Spatial and temporal data aggregation
- Standardized data formats and interfaces

### 🤖 Agent-Based Modeling
- Climate observer simulation framework
- Collective memory modeling
- Bias analysis in historical recording
- Behavioral pattern simulation

### 📈 Visualization
- Comprehensive plotting utilities
- Statistical visualization tools
- Interactive correlation analysis plots
- Publication-ready figure generation

## Quick Start

```python
from past1000 import HistoricalRecords, load_data
from past1000.compare import experiment_corr_2d
from past1000.calibration import MismatchReport

# Load historical records
history = HistoricalRecords(
    shp_path="data/regions.shp",
    data_path="data/historical_data.xlsx",
    region="华北地区"
)

# Load climate reconstruction data
datasets, uncertainties, _ = load_data(config)

# Perform correlation analysis
corr_df, r_benchmark, ax = experiment_corr_2d(
    data1=history.aggregate("mean"),
    data2=climate_data,
    corr_method="kendall"
)

# Generate mismatch report
report = MismatchReport(
    pred=classified_predictions,
    true=classified_observations,
    value_series=raw_data
)
report.analyze_error_patterns()
```

## Installation

```bash
# Using Poetry (recommended)
poetry install

# Or using pip
pip install -e .
```

## Documentation Structure

- **[Getting Started](getting-started/installation.md)**: Installation and basic setup
- **[API Reference](api/)**: Complete API documentation for all modules
- **[Examples](examples/)**: Practical usage examples and tutorials
- **[Development](development/)**: Contributing guidelines and development setup

## Citation

If you use Past1000 in your research, please cite:

```bibtex
@software{past1000,
  title={Past1000: A Python Library for Historical Climate Reconstruction Analysis},
  author={Song, Shuang},
  year={2025},
  url={https://github.com/SongshGeo/past1000}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/SongshGeo/past1000/blob/main/LICENSE) file for details.

## Support

- 📧 Email: [songshgeo@gmail.com](mailto:songshgeo@gmail.com)
- 🐛 Issues: [GitHub Issues](https://github.com/SongshGeo/past1000/issues)
- 📖 Documentation: [This site](https://songshgeo.github.io/past1000)

---

*Built with ❤️ for climate science and historical research*
