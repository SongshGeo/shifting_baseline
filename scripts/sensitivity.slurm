#!/bin/bash
# =============================================================================
# Global sensitivity analysis for the climate-observing ABM.
#
# Usage (from repo root, on a SLURM cluster):
#   sbatch scripts/sensitivity.slurm morris   personal
#   sbatch scripts/sensitivity.slurm morris   collective
#   sbatch scripts/sensitivity.slurm sobol    personal   1024
#   sbatch scripts/sensitivity.slurm sobol    collective 1024
#   sbatch scripts/sensitivity.slurm benchmark personal
#
# Args:
#   $1  stage         : smoke | benchmark | morris | sobol  (default: morris)
#   $2  baseline      : personal | collective | model       (default: personal)
#   $3  N (sobol only): Saltelli base size                  (default: 1024)
#
# Tweak --cpus-per-task / --time / --partition for your cluster. The Python
# orchestrator uses ThreadPoolExecutor over $SLURM_CPUS_PER_TASK to fan out
# concurrent ABM subprocesses; each subprocess uses model.num_process=1.
# =============================================================================
#SBATCH --job-name=sbs-sa
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00
# Memory note: 32 concurrent ABM subprocesses each need ~2-4 GB at peak
# (Python interp + abses + agent archive at corner cases). Empirically
# `--mem=32G` OOM'd at ~40% of one Morris sample under memory pressure
# (job 9503, 2026-04-25). 128 GB total = ~4 GB/worker, comfortable.
# Smaller-mem nodes? Drop both --cpus-per-task and --mem proportionally:
#   --cpus-per-task=16 --mem=64G   (still 4 GB/worker)
#   --cpus-per-task=8  --mem=32G   (still 4 GB/worker)
# The ThreadPoolExecutor in run_sensitivity.py uses $SLURM_CPUS_PER_TASK
# automatically.
# #SBATCH --partition=cpu             # uncomment & set for your site
# #SBATCH --account=your_account      # uncomment & set for your site
# #SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=songshgeo@gmail.com

set -euo pipefail

STAGE="${1:-morris}"
BASELINE="${2:-personal}"
N_SOBOL="${3:-1024}"

mkdir -p logs/slurm

# --- Environment ------------------------------------------------------------
# Adjust to your site. Two common patterns:
#   (a) module load + uv:           module load python/3.11 uv
#   (b) pre-built .venv via `uv sync`:  source .venv/bin/activate
# We try (b) first; fall back to invoking uv directly.

if [[ -f ".venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
    PY="python"
elif command -v uv >/dev/null 2>&1; then
    PY="uv run python"
else
    echo "Neither .venv/bin/activate nor 'uv' is available. Aborting." >&2
    exit 1
fi

# Avoid CPU oversubscription: each ABM subprocess uses BLAS internally; with
# 32 concurrent subprocesses, force single-threaded BLAS per process.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# --- Run --------------------------------------------------------------------
N_WORKERS="${SLURM_CPUS_PER_TASK:-8}"

case "$STAGE" in
    smoke)
        $PY reports/run_sensitivity.py smoke \
            --memory-baseline "$BASELINE" \
            --repeats 2 --years 20
        ;;
    benchmark)
        $PY reports/run_sensitivity.py benchmark \
            --memory-baseline "$BASELINE" \
            --repeats 30 --years 100 --n-samples 5
        ;;
    morris)
        # --timeout 3600 protects the queue from rare corner-case samples.
        # Empirical (cluster, 2026-04-25): mean 926s, max 2461s at
        # max_age=55, new_agents=15, loss_rate=0.02 (huge archive, O(archive^2)
        # in collective-baseline memory updates). 3600s = 1h covers the
        # observed max with headroom; timed-out samples are recorded as
        # "error: timeout ..." and SALib falls back to nanmedian fill.
        $PY reports/run_sensitivity.py morris \
            --memory-baseline "$BASELINE" \
            --r-trajectories 20 --num-levels 4 \
            --repeats 30 --years 100 \
            --n-workers "$N_WORKERS" \
            --timeout 3600 \
            --climate-process ar1
        ;;
    sobol)
        $PY reports/run_sensitivity.py sobol \
            --memory-baseline "$BASELINE" \
            --N "$N_SOBOL" \
            --repeats 30 --years 100 \
            --n-workers "$N_WORKERS" \
            --timeout 3600 \
            --climate-process ar1
        ;;
    *)
        echo "Unknown stage: $STAGE" >&2
        echo "Use one of: smoke | benchmark | morris | sobol" >&2
        exit 2
        ;;
esac

echo "[$STAGE/$BASELINE] finished at $(date -Iseconds)"
