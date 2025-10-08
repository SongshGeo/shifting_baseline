# Shifting Baseline Syndrome in Historical Climate Records

[![Status](https://img.shields.io/badge/status-under%20peer%20review-yellow)](https://github.com/SongshGeo/shifting_baseline)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This repository contains the code and analysis pipeline for the manuscript:

**Archival and palaeoenvironmental documentation of historical extreme events reveals perceptual bias in collective memory**

> **Status**: Currently under peer review

## Abstract

It is well-documented that collective memory shapes how societies respond to extreme climate events today. However, studies on the emergence of collective memory and its perceptual bias over time are lacking. Here, we utilise four centuries (1470–1900 CE) of climatic archives from northern China, along with tree–ring–based hydroclimate reconstructions, which are validated against instrumental observations of Wet/Dry Index (1901–2000 CE). We find that historical records reveal systematic, non-random mismatches, where severity assessments are often biased toward recent experience rather than absolute climatic anomalies. Critically, correlations between historical and natural records increase when the natural series is re-standardised with 20–40‑year sliding windows (optimal = ~30 years). A mechanism-driven agent‑based model, grounded in Shifting Baseline Syndrome (SBS), reproduces this periodicity and highlights the role of perceptual bias in forming collective memory. We argue that our findings provide direct insights into historical collective memory, with perceptual bias operating on generational timescales imprinting. We argue that this has implications for bias‑aware corrections for risk assessment, adaptation planning, and paleoclimate reconstruction.

## Repository Structure

```
shifting_baseline/
├── config/              # Configuration files (Hydra)
│   ├── ds/             # Data source configurations
│   ├── how/            # Analysis method configurations
│   └── model/          # Model configurations
├── data/               # Data files (historical records, reconstructions, etc.)
├── docs/               # Documentation
├── reports/            # Jupyter notebooks for analysis and visualization
├── shifting_baseline/  # Main Python package
│   ├── abm.py         # Agent-based model implementation
│   ├── calibration.py # Model calibration
│   ├── compare.py     # Comparison between historical and natural records
│   ├── data.py        # Data loading and preprocessing
│   ├── filters.py     # Signal processing filters
│   ├── mc.py          # Monte Carlo simulations
│   └── utils/         # Utility functions
└── tests/             # Unit tests
```

## Key Features

- **Historical Data Analysis**: Processing and analysis of four centuries of historical climate records from northern China
- **Paleoclimate Reconstruction**: Integration with tree-ring-based hydroclimate reconstructions
- **Correlation Analysis**: Systematic comparison between historical records and natural archives
- **Sliding Window Analysis**: Investigation of perceptual bias using 20–40-year sliding windows
- **Agent-Based Modeling**: Mechanistic simulation of collective memory formation based on Shifting Baseline Syndrome
- **Statistical Validation**: Comprehensive validation against instrumental observations (1901–2000 CE)

## Installation

### Prerequisites

- Python 3.11
- Poetry (recommended for dependency management)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/SongshGeo/shifting_baseline.git
cd shifting_baseline
```

2. Install dependencies using Poetry:
```bash
poetry install
```

Or using pip:
```bash
pip install -r requirements-docs.txt
```

3. Activate the environment:
```bash
poetry shell
```

## Usage

### Running Analyses

The project uses [Hydra](https://hydra.cc/) for configuration management. Main analyses can be executed through the command line:

```bash
# Run correlation analysis
python -m shifting_baseline how=correlation

# Run comparison analysis
python -m shifting_baseline how=compare

# Run data processing
python -m shifting_baseline how=process
```

### Jupyter Notebooks

Interactive analyses and visualizations are available in the `reports/` directory:

- `history.ipynb`: Historical records analysis
- `natural.ipynb`: Natural archives analysis
- `mismatch.ipynb`: Mismatch analysis between historical and natural records
- `abm.ipynb`: Agent-based model simulations

Launch Jupyter:
```bash
jupyter notebook reports/
```

### Configuration

Configuration files are located in the `config/` directory and follow the Hydra structure. You can modify:

- Data sources: `config/ds/`
- Analysis methods: `config/how/`
- Model parameters: `config/model/`

## Key Results

1. **Systematic Mismatch**: Historical records show non-random mismatches with paleoclimate reconstructions, biased toward recent experience
2. **Optimal Window**: Correlation peaks at ~30-year sliding windows, suggesting generational timescale of perceptual bias
3. **Mechanistic Model**: Agent-based model successfully reproduces observed patterns, validating Shifting Baseline Syndrome hypothesis
4. **Implications**: Findings inform bias-aware corrections for risk assessment and climate adaptation planning

## Data Sources

The analysis integrates multiple data sources:

- Historical climate archives from northern China (1470–1900 CE)
- Tree-ring-based hydroclimate reconstructions
- Instrumental Wet/Dry Index observations (1901–2000 CE)
- PMIP past1000 climate model outputs (ACCESS-ESM1-5, MIROC-ES2L, MRI-ESM2-0)

## Testing

Run tests using pytest:

```bash
poetry run pytest
```

With coverage:
```bash
poetry run pytest --cov=shifting_baseline
```

## Documentation

Full documentation is available in the `docs/` directory and can be built using MkDocs:

```bash
mkdocs serve
```

Then visit `http://127.0.0.1:8000/` in your browser.

## Citation

If you use this code or data, please cite:

```bibtex
@article{shifting_baseline_2025,
  title={Archival and palaeoenvironmental documentation of historical extreme events reveals perceptual bias in collective memory},
  author={[Authors]},
  journal={[Journal]},
  year={2025},
  note={Under peer review}
}
```

## Contributing

This repository contains research code under peer review. For questions or collaboration inquiries, please open an issue or contact the corresponding author.

## License

[Specify your license here - e.g., MIT, GPL-3.0, etc.]

## Acknowledgments

We acknowledge the use of historical climate archives from northern China and paleoclimate reconstruction data. Detailed acknowledgments will be provided upon publication.

## Contact

- **Author**: Shuang (Twist) Song
- **Email**: songshgeo@gmail.com
- **GitHub**: [@SongshGeo](https://github.com/SongshGeo)
- **Website**: [https://cv.songshgeo.com/](https://cv.songshgeo.com/)

---

**Note**: This repository is actively maintained during the peer review process. Code and documentation may be updated based on reviewer feedback.
