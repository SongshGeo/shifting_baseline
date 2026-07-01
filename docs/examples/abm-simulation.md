# ABM Simulation

See the **[ABM guide](../guide/abm.md)** for the concepts (2×2 baselines, climate
forcing, Sobol sensitivity). This page shows how to *run* it.

## Command line (recommended)

The ABM is a Hydra app that supports `--multirun` sweeps via `abses.Experiment`.

```bash
# main analysis: AR(1) forcing under the personal baseline (H1)
uv run python shifting_baseline/abm.py +model.climate_process=ar1

# discriminate the mechanisms: sweep the four baselines
uv run python shifting_baseline/abm.py --multirun \
  model.memory_baseline=personal,collective,collective_lifetime,model

# lifespan sweep (the Sobol-dominant parameter)
uv run python shifting_baseline/abm.py --multirun model.max_age=20,30,40,50,60
```

Outputs land in a timestamped `multirun/` directory; each run writes its
window-by-correlation vector.

## Python

```python
from hydra import compose, initialize
from shifting_baseline.abm import ClimateObservingModel

with initialize(version_base=None, config_path="../config"):
    cfg = compose(config_name="config.yaml",
                  overrides=["model=test", "+model.climate_process=ar1"])

model = ClimateObservingModel(parameters=cfg)
model.run_model()

# window–τ curve between the collective archive and the true climate
curve = model.get_corr_curve(corr_method=cfg.corr_method)
print(curve.head())

# underlying series
df = model.climate_df          # objective climate + collective-memory climate
```

!!! note
    Exact constructor/method signatures evolve — check the generated
    [`abm` API](../api/abm.md). Under the `personal` baseline the window–τ curve peaks
    in the 20–40-year band; under `collective` it rises monotonically.

## Sensitivity analysis

The variance-based Sobol analysis lives in
[`sensitivity`](../api/sensitivity.md) and is typically run as a batch job (Saltelli
sampling over five parameters, many repeats). See that module's API and the project's
`slurm` scripts for the large sweeps.
