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
#SBATCH --mem=32G
#SBATCH --time=48:00:00
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
        # --timeout 1800 protects the queue from rare corner-case samples
        # (high max_age × new_agents → huge archive → ABM goes O(archive^2)).
        # Timed-out samples are recorded as "error: ..." in raw_outputs.csv
        # and SALib analysis falls back to nanmedian-fill for missing Y.
        $PY reports/run_sensitivity.py morris \
            --memory-baseline "$BASELINE" \
            --r-trajectories 20 --num-levels 4 \
            --repeats 30 --years 100 \
            --n-workers "$N_WORKERS" \
            --timeout 1800 \
            --climate-process ar1
        ;;
    sobol)
        $PY reports/run_sensitivity.py sobol \
            --memory-baseline "$BASELINE" \
            --N "$N_SOBOL" \
            --repeats 30 --years 100 \
            --n-workers "$N_WORKERS" \
            --timeout 1800 \
            --climate-process ar1
        ;;
    *)
        echo "Unknown stage: $STAGE" >&2
        echo "Use one of: smoke | benchmark | morris | sobol" >&2
        exit 2
        ;;
esac

echo "[$STAGE/$BASELINE] finished at $(date -Iseconds)"
