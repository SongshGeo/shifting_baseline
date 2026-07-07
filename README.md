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
│   ├── abm.py         # Agent-based model (personal vs collective baselines)
│   ├── calibration.py # Mismatch report + Monte Carlo randomisation null
│   ├── compare.py     # Sliding-window re-standardisation & correlation sweeps
│   ├── data.py        # Data loading and preprocessing
│   ├── filters.py     # Re-standardisation + 5-level classification
│   ├── mc.py          # Bayesian (PyMC) integration of reconstructions → N-WDI
│   ├── results.py     # Reproducible builders for the manuscript numbers
│   └── utils/         # Utility functions (calc, plot, log, types)
└── tests/             # Unit tests (+ results regression guardrail)
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
- uv

### Setup

1. Clone the repository:
```bash
git clone https://github.com/SongshGeo/shifting_baseline.git
cd shifting_baseline
```

2. Install dependencies using uv:
```bash
uv sync
```

Install documentation tooling when needed:
```bash
uv sync --group docs
```

3. Activate the environment:
```bash
source .venv/bin/activate
```

4. Configure local paths (machine-specific paths are never committed):
```bash
cp .env.example .env   # then edit SBS_ROOT / SBS_DATA_ROOT / SBS_OUT_DIR
```
`.env` is loaded automatically on `import shifting_baseline` and backs the
`${oc.env:...}` interpolations in `config/ds/*.yaml`. See [Data availability](#data-availability)
for how to obtain the raw datasets that `SBS_DATA_ROOT` should point at.

## Usage

### Running Analyses

The project uses [Hydra](https://hydra.cc/) for configuration management.

```bash
# Main pipeline (Steps 1–6: load → validate → mismatch → correlation sweep → figures)
python -m shifting_baseline                 # default config (ds=pure)
python -m shifting_baseline ds=best         # swap the data-source profile

# Agent-based model (supports --multirun sweeps)
python shifting_baseline/abm.py --multirun model=exp model.max_age=30,40,50 \
    model.memory_baseline=personal,collective
```

Each run writes figures, logs, and the resolved config to a timestamped folder under
`outputs/` (or `multirun/` for sweeps).

### Jupyter Notebooks

The figure-producing notebooks live in `reports/`:

- `natural.ipynb` — reconstruction/validation (Fig 2)
- `mismatch.ipynb` — H-WDI vs N-WDI mismatch (Fig 3)
- `history.ipynb` — sliding-window re-standardisation (Fig 4)
- `abm.ipynb` — agent-based model (Fig 5)
- `subplots.ipynb` — figure compositor
- Supplementary: `abm_mechanism.ipynb`, `archives.ipynb`, `climate_forcing.ipynb`, `life_expectancy.ipynb`

Launch Jupyter:
```bash
jupyter notebook reports/
```

### Reproducing the manuscript numbers

The numbers rendered into the manuscript are assembled by
`shifting_baseline.results.build_results` and stored in `results.json`. A regression
guardrail locks them (skips automatically when the datasets are not present locally):

```bash
uv run pytest tests/test_results_regression.py -m slow -v
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

- Historical climate archives from northern China (1470–1900 CE) — from the *Atlas of Extreme Droughts and Floods over the Past Millennium* (Yang et al., 2024)
- Tree-ring-based hydroclimate reconstructions — public NOAA paleoclimate repositories (see Supplementary Information S2, Table S8)
- Instrumental precipitation for validation (1901–2000 CE) — CRU TS, GPCC, and a China gridded product
- PMIP past1000 climate model outputs (ACCESS-ESM1-5, MIROC-ES2L, MRI-ESM2-0)

### Data availability

The raw datasets (multi-GB gridded NetCDF, base-map shapefiles, and the licensed
drought/flood atlas) are **not distributed with this repository** (`data/` is
gitignored). Point `SBS_DATA_ROOT` (in `.env`) at a local folder holding them; the
pipeline then writes standardized intermediates to `${SBS_ROOT}/data/*.csv`, which
are reused by default (`recalculate_data=false`). The reconstruction sources and
their DOIs are listed in the manuscript's Supplementary Information S2; instrumental
products are available from their original providers (CRU, GPCC). Please contact the
corresponding author for access details to the compiled archive datasets.

## Testing

Run tests using pytest:

```bash
uv run python -m pytest
```

With coverage:
```bash
uv run python -m pytest --cov=shifting_baseline
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

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

We acknowledge the use of historical climate archives from northern China and paleoclimate reconstruction data. Detailed acknowledgments will be provided upon publication.

## Contact

- **Author**: Shuang (Twist) Song
- **Email**: songshgeo@gmail.com
- **GitHub**: [@SongshGeo](https://github.com/SongshGeo)
- **Website**: [https://cv.songshgeo.com/](https://cv.songshgeo.com/)

---

**Note**: This repository is actively maintained during the peer review process. Code and documentation may be updated based on reviewer feedback.
