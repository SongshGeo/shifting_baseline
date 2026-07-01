# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Research codebase for the manuscript "Archival and palaeoenvironmental documentation of historical extreme events reveals perceptual bias in collective memory" (under peer review). It compares historical Chinese climate archives (1470–1900 CE) against tree-ring hydroclimate reconstructions and instrumental data (1901–2000 CE) to study Shifting Baseline Syndrome (SBS), and uses an agent-based model (ABM) to reproduce the observed perceptual-bias pattern.

Python 3.11 only. Dependencies are managed with **uv** (the `makefile` still references `poetry`, but the actual environment is `uv.lock` / `uv sync`). Do not use `poetry` commands — prefer `uv run ...` or activate `.venv`.

## Common commands

```bash
# Install / sync environment
uv sync                      # runtime + dev
uv sync --group docs         # add docs tooling

# Run the main analysis pipeline (Hydra entrypoint)
uv run python -m shifting_baseline                     # uses config/config.yaml defaults
uv run python -m shifting_baseline how=correlation     # override a config group
uv run python -m shifting_baseline test_mode=true      # smoke-test logging only

# Run the ABM directly (supports Hydra --multirun sweeps)
uv run python shifting_baseline/abm.py
uv run python shifting_baseline/abm.py --multirun model=exp model.max_age=30,40,50 model.memory_baseline=personal,collective

# Tests
uv run pytest                                    # full suite
uv run pytest tests/test_filters.py              # single file
uv run pytest tests/test_filters.py::test_name   # single test
uv run pytest -m "not slow"                      # skip slow-marked tests
uv run pytest --cov=shifting_baseline            # with coverage

# Docs
mkdocs serve
```

Hydra writes each run's outputs (figures, logs, resolved config) to a timestamped directory under `outputs/` (single run) or `multirun/` (sweeps). The `.hydra/config.yaml` inside that directory is the source of truth for what the run actually executed.

## Architecture

### Configuration (Hydra)
`config/config.yaml` composes three swappable groups:
- `ds/` — data sources (paths to historical records, reconstructions, validation datasets like `china`/`gpcc`/`cru`, PMIP model outputs). Selected via `ds=<name>`.
- `how/` — which analysis the `__main__` entrypoint runs (`process`, `correlation`, `compare`).
- `model/` — ABM parameters (`exp` for experiments, `test` for quick runs).

Top-level knobs the pipeline reads repeatedly: `corr_method` (default `kendall`), `filter_side`, `agg_method`, `resolution`, `to_std`, `min_period`, `low_pass.window_size` (the ~30-year sliding window is the central finding), `using_val_data`, `violin_windows`. Model configs interpolate these via `${corr_method}` etc., so changing a top-level value propagates into the ABM.

### Package layout (`shifting_baseline/`)
The pipeline is a chain from raw archives → classified categories → correlations → ABM validation:

- `data.py` — `load_data` / `load_validation_data`. Returns `(combined, uncertainties, history)`; `history` is a custom object exposing `.get_time_slice(...)`, `.aggregate(...)`, `.merge_with(...)`. `STAGE1`/`END` boundaries and other year constants live in `constants.py`.
- `process.py` — `batch_process_recon_data` ingests tree-ring reconstructions into the standardized form the rest of the pipeline expects.
- `filters.py` — `calc_std_deviation` (sliding-window re-standardization, the core SBS operation), `classify` / `classify_single_value` (continuous z-scores → ordinal wet/dry categories to match historical records).
- `compare.py` — correlation machinery: `experiment_corr_2d`, `compare_corr_2d`, `sweep_slices` (generates rolling time windows), `sweep_max_corr_year` (for each slice, finds the window size that maximizes correlation — produces the headline "~30-year optimum" plot).
- `calibration.py` — `MismatchReport`: confusion-matrix–style analysis between predicted (historical) and true (natural) categorical series, `analyze_error_patterns()` + `generate_report_figure()`.
- `abm.py` — `ClimateObservingModel` (an `abses.MainModel`) with observer agents that age, die, and record extremes relative to either **personal** or **collective** memory baselines. Meant to reproduce empirical correlation patterns under the SBS hypothesis. Runs as Hydra multirun experiments via `abses.Experiment`.
- `mc.py` — Monte Carlo null-distribution sampling.
- `utils/` — `config.py` (Hydra helpers incl. `get_output_dir`, `format_by_config`), `log.py` (`get_logger`, `setup_logger_from_hydra`), `calc.py`, `plot.py`, `anova.py`, `email.py` (run-completion notifier), `types.py`.

### Notebooks (`reports/`)
`history.ipynb`, `natural.ipynb`, `mismatch.ipynb`, `abm.ipynb`, `subplots.ipynb` — these are the figure-producing notebooks for the manuscript. `reports/results/` holds their saved outputs. `climate_scenario_experiments.py` is a script variant.

### Key architectural points
- **Categorical vs continuous**: historical records are ordinal (wet/dry grades), natural reconstructions are continuous. Everything is funneled through `classify()` before comparison — touch this carefully.
- **Sliding-window re-standardization** (`calc_std_deviation` with `filter_side`/`window_size`) is the mechanism under test, not just preprocessing. Default `low_pass.window_size: 30` is load-bearing.
- **Hydra interpolation**: ABM config pulls top-level keys via `${...}`, so the same correlation method is used across empirical analysis and simulated data — don't desync them.
- `recalculate_data: false` by default reuses cached intermediates under `${ds.processed}/*.csv`. Set `recalculate_data=true` when raw data or processing logic changes.

## Repository conventions

- Language in code/comments/log messages is mixed Chinese + English; match the surrounding file's style when editing.
- Tests live in `tests/` with `conftest.py` fixtures and a `slow` pytest marker (see `pyproject.toml`).
- `SHIFTING_BASELINE_COMPAT_REPORT.md` documents the `poetry → uv` migration; treat the `makefile`'s poetry targets as legacy.
