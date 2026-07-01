# Overview & Science

## The question

Historical documentary archives are *perception-mediated*: people record extremes
relative to what they consider "normal". If that reference point drifts across
generations — **Shifting Baseline Syndrome (SBS)** — then archives should systematically
disagree with an independent, non-perceptual record of the same climate. This project
quantifies that disagreement and asks which cognitive mechanism produces it.

## Two indices

Everything is expressed as a **Wet/Dry Index (WDI)** with five ordinal levels:
Severe Dry (SD), Moderate Dry (MD), Normal (N), Moderate Wet (MW), Severe Wet (SW).

| Series | Source | Nature |
| --- | --- | --- |
| **H-WDI** | Historical archives (1470–1900 CE) | Ordinal grades, mediated by human perception |
| **N-WDI** | Tree-ring reconstructions, validated against instruments (1901–2000 CE) | Continuous, comparatively independent |

Because the archives are ordinal and the reconstructions are continuous, everything is
funnelled through [`classify`][shifting_baseline.filters.classify_series] before comparison.

## Three findings

1. **Mismatch has structure.** Where H-WDI and N-WDI disagree, the archive tends to
   over- or under-state severity *relative to a similar recent event* — a
   *shifted-comparison* bias. Its significance is tested against a Monte Carlo
   randomisation null built into
   [`MismatchReport`][shifting_baseline.calibration.MismatchReport] (`mc_runs`).
2. **A ~30-year window aligns them.** Re-standardising the natural series inside a
   sliding window ([`calc_std_deviation`][shifting_baseline.filters.calc_std_deviation])
   raises the correlation, and the improvement peaks for windows of **20–40 years
   (optimum ≈ 30)** — a generational timescale. This is the central result.
3. **Generational amnesia, not collective illusion.** An agent-based model
   ([`abm`](../api/abm.md)) reproduces the generational optimum **only** when agents
   judge against their own lived experience (H1), not against the shared archive (H2).

## Why an ABM?

The empirical comparison establishes *that* a ~30-year window exists, but not *why*.
Two SBS mechanisms are indistinguishable in the archives alone:

- **H1 — generational amnesia:** each generation's baseline is bounded by its own
  lived experience.
- **H2 — collective illusion:** shared narratives reshape how the past is remembered.

The ABM is a mechanism-discrimination test: run the same model under each baseline and
see which reproduces the empirical pattern. Only H1 does. A global (Sobol) sensitivity
analysis then shows the optimal-window location is set almost entirely by agent
**lifespan** — a genuinely generational cause. See the
**[ABM guide](abm.md)**.

## Key design points

- **Categorical vs continuous** — archives are ordinal, reconstructions continuous;
  they meet only after `classify()`.
- **Sliding-window re-standardisation is the mechanism under test**, not mere
  preprocessing. The default `low_pass.window_size: 30` is load-bearing.
- **Hydra interpolation** keeps the empirical and simulated analyses in sync (the same
  `corr_method`, window, etc. flow into the ABM config).

Continue to the **[Analysis Pipeline](pipeline.md)** for the concrete data flow.
