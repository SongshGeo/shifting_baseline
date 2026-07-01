# Configuration (Hydra)

All runtime behaviour is controlled by [Hydra](https://hydra.cc/). `config/config.yaml`
composes three swappable groups and exposes top-level knobs the pipeline reads
repeatedly.

## Composition

```yaml
defaults:
  - ds: pure        # data sources
  - how: process    # which analysis __main__ runs
  - model: exp      # ABM parameters
```

| Group | Options (`config/<group>/*.yaml`) | Meaning |
| --- | --- | --- |
| `ds` | `pure` (default), `best`, `mac`, `win_serve` | Paths to archives, reconstructions, validation datasets (`china`/`gpcc`/`cru`), PMIP outputs |
| `how` | `process` (default), `correlation`, `compare` | Which analysis the entrypoint runs |
| `model` | `exp` (default), `test` | ABM parameters (`exp` for experiments, `test` for quick runs) |

Select a group on the command line, e.g. `ds=best how=correlation model=test`.

## Top-level knobs

These are read across the pipeline (and interpolated into the ABM config via
`${...}`, so changing one propagates everywhere):

| Key | Default | Role |
| --- | --- | --- |
| `corr_method` | `kendall` | Correlation method (`pearson`/`kendall`/`spearman`) |
| `filter_side` | `right` | Which side the sliding-window filter uses |
| `agg_method` | `mean` | Cross-site aggregation of archive grades |
| `to_std` | `sampling` | Discrete→continuous mapping (`sampling` = truncated-normal; else midpoint) |
| `resolution` | `0.5` | Grid resolution for validation data |
| `min_period` | `10` | Minimum samples per rolling window |
| `ratio` | `0.10` | Top-% extreme interval |
| `low_pass.window_size` | `30` | The ~30-year sliding window — **load-bearing** (the central finding) |
| `using_val_data` | `china` | Validation dataset (`china`/`gpcc`/`cru`) |
| `violin_windows` | `[20, 40, 60]` | Window sizes for violin plots |
| `random_seed` | `42` | Fixed seed for reproducible sampling / Bayesian integration (set `null` for random) |
| `test_mode` | `false` | Log-only smoke test |
| `recalculate_data` | `false` | Recompute intermediates instead of reusing `${ds.processed}/*.csv` |

## Overriding

```bash
# change a knob
uv run python -m shifting_baseline low_pass.window_size=25 corr_method=spearman

# append a key that isn't in the selected group (Hydra struct mode)
uv run python shifting_baseline/abm.py +model.climate_process=ar1

# multirun sweep
uv run python shifting_baseline/abm.py --multirun model.max_age=30,40,50
```

!!! warning "Struct mode"
    Overriding a key that does not exist in the chosen config raises an error. Use
    the `+key=value` syntax to *append* a new key (e.g. `+model.climate_process=ar1`
    when the selected `model` group doesn't define it).

## What a run writes

Each run creates a timestamped directory under `outputs/` (single) or `multirun/`
(sweeps) containing figures, logs, and `.hydra/config.yaml` — the resolved config
that is the source of truth for the run.
