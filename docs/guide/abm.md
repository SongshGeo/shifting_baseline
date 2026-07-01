# Agent-Based Model

The ABM ([`abm.ClimateObservingModel`][shifting_baseline.abm]) is a deliberately
minimal, heuristic model. Its job is **mechanism discrimination**: does the empirical
~30-year optimum come from *generational amnesia* or *collective illusion*?

## How it works

A continuously turning-over population of finite-lifespan observer agents watches an
annual climate signal. Each year an agent perceives the current anomaly **relative to
a baseline** and, the more it departs from that baseline, the more likely it is to
record it. Recorded events form a shared archive, aggregated and classified into the
same five-level WDI used for the empirical data, so simulated and real records are
compared identically.

- **Perception:** `z = (cₜ − μ) / σ`, where `μ, σ` come from the chosen baseline.
- **Recording propensity:** `p = f₀ + 0.5 − Φ(|z|)` — a *negativity bias* (extremes
  are recorded more often).
- **Archive loss:** each record survives with probability `1 − loss_rate`.
- **Demography:** `new_agents` enter per year; agents past `max_age` leave.

## Climate forcing — [`climate_forcing`](../api/climate_forcing.md)

`climate_process` selects the driver:

| Process | Meaning |
| --- | --- |
| `iid` | Independent Gaussian noise — the "null" climate (config default) |
| `ar1` | AR(1) persistence `cₜ = φ·cₜ₋₁ + εₜ` (φ = `climate_phi`) — **used as the main analysis** in the paper |
| `trend_plus_noise` | Linear trend + noise (low-frequency mean drift) |
| `ar1_trend` | AR(1) persistence + linear trend |

Variability is `climate_sigma`. The forcing matrix (i.i.d. / AR(1) / trend, at annual
and sub-annual resolution) is a robustness check — the generational optimum survives
all of them.

## The 2×2 baseline design

`memory_baseline` crosses two factors — the *source* of the reference (own experience
vs shared archive) and its *temporal horizon* (own lifetime vs full record):

| | Generational horizon (one lifetime) | Cumulative horizon (full record) |
| --- | --- | --- |
| **Individual** (own memory) | `personal` — **H1: generational amnesia** | *(idealised limit → `model`)* |
| **Collective** (shared archive) | `collective_lifetime` | `collective` — **H2: collective illusion** |

- `personal` (**H1**) — re-standardise against the agent's own lived memory. **Only
  this reproduces the ~20–40-year generational optimum.**
- `collective` (**H2**) — use the whole accumulated archive; correlation rises
  monotonically with window length, no generational peak.
- `collective_lifetime` — shared archive but bounded to a lifetime; behaves like H2,
  showing the effect needs *individual experience*, not just a bounded horizon.
- `model` — reference the objective climate itself; a non-perceptual benchmark.

## Global sensitivity — [`sensitivity`](../api/sensitivity.md)

A variance-based **Sobol** analysis (Saltelli sampling) over five parameters:

| Parameter | Range |
| --- | --- |
| `max_age` (lifespan) | 15–80 yr |
| `new_agents` | 1–15 |
| `loss_rate` | 0–0.8 |
| `climate_sigma` | 0.5–2.0 |
| `climate_phi` | 0.0–0.9 |

Under H1 the optimal-window location is governed almost entirely by **agent lifespan**
(total-effect `S_T ≈ 1` for `max_age`) and scales in direct proportion to it — a
genuinely generational cause. Under H2 no parameter reproduces a generation-scaled
optimum.

## Running it

```bash
# main analysis: AR(1) forcing under the personal baseline (H1)
uv run python shifting_baseline/abm.py +model.climate_process=ar1

# baseline sweep
uv run python shifting_baseline/abm.py --multirun \
  model.memory_baseline=personal,collective,collective_lifetime,model
```

Defaults (`config/model/exp.yaml`): `max_age=40`, `min_age=10`, `new_agents=5`,
`loss_rate=0.4`, `memory_baseline=personal`, `repeats=100`. See the
[`abm` API](../api/abm.md) for classes and methods.
