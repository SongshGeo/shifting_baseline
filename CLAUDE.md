# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Research codebase for the manuscript "Archival and palaeoenvironmental documentation of historical extreme events reveals perceptual bias in collective memory" (under peer review). It compares historical Chinese climate archives (1470–1900 CE) against tree-ring hydroclimate reconstructions and instrumental data (1901–2000 CE) to study Shifting Baseline Syndrome (SBS), and uses an agent-based model (ABM) to reproduce the observed perceptual-bias pattern.

Python 3.11 only. Dependencies are managed with **uv** (`uv.lock` / `uv sync`; the `makefile` targets already call `uv`). Do not use `poetry` commands — prefer `uv run ...` or activate `.venv`.

Machine-specific paths are supplied via a gitignored `.env` (loaded automatically on `import shifting_baseline`) that backs the `${oc.env:SBS_ROOT/SBS_DATA_ROOT/SBS_OUT_DIR}` interpolations in `config/ds/*.yaml`. Copy `.env.example` → `.env` and fill in your paths; no personal absolute path is committed.

## Common commands

```bash
# Install / sync environment
uv sync                      # runtime + dev
uv sync --group docs         # add docs tooling

# Run the main analysis pipeline (Hydra entrypoint)
uv run python -m shifting_baseline                     # uses config/config.yaml defaults
uv run python -m shifting_baseline ds=best             # override a config group
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
- `ds/` — data sources (paths to historical records, reconstructions, validation datasets like `china`/`gpcc`/`cru`, PMIP model outputs). Selected via `ds=<name>`; `pure` is the default (all-natural proxies, matches the main text).
- `how/` — nominal analysis selector. **Note:** `__main__._main` hard-codes its Step 1–6 pipeline and never reads `cfg.how`, so `how=...` overrides are accepted by Hydra but do not change what the main entrypoint runs (only `process.py`'s separate CLI reads `cfg.how.recon`). Not a bug, but don't rely on it.
- `model/` — ABM parameters (`exp` for experiments, `test` for quick runs).

Top-level knobs the pipeline reads repeatedly: `corr_method` (default `kendall`), `filter_side`, `agg_method`, `resolution`, `to_std`, `min_period`, `low_pass.window_size` (the ~30-year sliding window is the central finding), `using_val_data`, `violin_windows`. Model configs interpolate these via `${corr_method}` etc., so changing a top-level value propagates into the ABM.

### Package layout (`shifting_baseline/`)
The pipeline is a chain from raw archives → classified categories → correlations → ABM validation:

- `data.py` — `load_data` / `load_validation_data`. Returns `(combined, uncertainties, history)`; `history` is a custom object exposing `.get_time_slice(...)`, `.aggregate(...)`, `.merge_with(...)`. `STAGE1`/`END` boundaries and other year constants live in `constants.py`.
- `process.py` — instrumental-precipitation preprocessing (extract summer JAS precip → nc). Also contains `batch_process_recon_data` / `ProcessRecon`, an orphaned recon-ingest sub-pipeline wired to `config/how/process.yaml` but with no live caller in the main flow; both `process.py` and `mc.py` are kept as producers of the committed `${ds.processed}/*.csv` caches (reproducibility assets).
- `mc.py` — **Bayesian latent-variable integration** of the tree-ring reconstructions into the standardized N-WDI (`combine_reconstructions`, PyMC StudentT model) plus z-score/uncertainty helpers. (Despite the filename, this is *not* the Monte Carlo null — that lives in `calibration.py`.)
- `filters.py` — `calc_std_deviation` (sliding-window re-standardization, the core SBS operation), `classify` / `classify_single_value` (continuous z-scores → ordinal wet/dry categories to match historical records).
- `compare.py` — correlation machinery: `experiment_corr_2d`, `compare_corr_2d`, `sweep_slices` (generates rolling time windows), `sweep_max_corr_year` (for each slice, finds the window size that maximizes correlation — produces the headline "~30-year optimum" plot).
- `calibration.py` — `MismatchReport`: confusion-matrix analysis between predicted (historical) and true (natural) categorical series; `analyze_error_patterns()` builds the shifted-comparison bias matrix and runs the **Monte Carlo randomisation null** (`_run_significance_test`, seeded via `np.random.default_rng`).
- `abm.py` — `ClimateObservingModel` (an `abses.MainModel`) with observer agents that age, die, and record extremes relative to either **personal** or **collective** memory baselines. Meant to reproduce empirical correlation patterns under the SBS hypothesis. Runs as Hydra multirun experiments via `abses.Experiment`.
- `sensitivity.py` — global Sobol/Morris (SALib) sensitivity analysis of the ABM; driven by `reports/run_sensitivity.py`.
- `results.py` — `build_results(cfg)` / `compute_results1..3`: the single reproducible source for the manuscript's headline numbers (`results.json`), shared by the figure notebooks and `tests/test_results_regression.py`. Deterministic via `cfg.random_seed`.
- `utils/` — `log.py` (`get_logger`, `setup_logger_from_hydra`), `calc.py`, `plot.py`, `types.py`. (No `config.py` / `email.py` / `anova.py` — those were removed in the 2026-06 cleanup.)

### Notebooks (`reports/`)
Main-text figures: `natural.ipynb` (Fig 2 + `results1`), `mismatch.ipynb` (Fig 3 + `results2`), `history.ipynb` (Fig 4 + `results3`), `abm.ipynb` (Fig 5), `subplots.ipynb` (compositor). SI figures: `abm_mechanism.ipynb`, `archives.ipynb`, `climate_forcing.ipynb`, `life_expectancy.ipynb`. The three `results{1,2,3}`-writing cells all go through `shifting_baseline.results.build_results` so the notebooks and the regression guardrail cannot drift. `reports/results/` (gitignored) holds saved outputs. Real analysis scripts: `run_sensitivity.py`, `plot_sobol.py`, `climate_scenario_experiments.py`, `robustness_sampling_vs_mapping.py`. The `reports/_bug*_compare.py` files are one-shot reviewer-response reproducers, not part of the pipeline.

### Key architectural points
- **Categorical vs continuous**: historical records are ordinal (wet/dry grades), natural reconstructions are continuous. Everything is funneled through `classify()` before comparison — touch this carefully.
- **Sliding-window re-standardization** (`calc_std_deviation` with `filter_side`/`window_size`) is the mechanism under test, not just preprocessing. Default `low_pass.window_size: 30` is load-bearing.
- **Hydra interpolation**: ABM config pulls top-level keys via `${...}`, so the same correlation method is used across empirical analysis and simulated data — don't desync them.
- `recalculate_data: false` by default reuses cached intermediates under `${ds.processed}/*.csv`. Set `recalculate_data=true` when raw data or processing logic changes.

### Results guardrail
`results.json` holds the numbers rendered into the manuscript (`results1/2/3`). `tests/test_results_regression.py` (marked `slow`, auto-skips when the local datasets are absent) regenerates them via `build_results` and asserts they match `tests/data/golden_results.json` exactly. **Run it before and after any refactor** that could touch a number:

```bash
uv run pytest tests/test_results_regression.py -m slow -v
```

## Repository conventions

- Language in code/comments/log messages is mixed Chinese + English; match the surrounding file's style when editing.
- Tests live in `tests/` with `conftest.py` fixtures and a `slow` pytest marker (see `pyproject.toml`).
