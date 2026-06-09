#!/bin/bash
# =============================================================================
# Global sensitivity analysis for the climate-observing ABM.
#
# Usage (from repo root, on a SLURM cluster):
#   sbatch scripts/sensitivity.slurm morris   personal
#   sbatch scripts/sensitivity.slurm morris   collective
#   sbatch scripts/sensitivity.slurm morris   collective_lifetime
#   sbatch scripts/sensitivity.slurm sobol    personal            1024
#   sbatch scripts/sensitivity.slurm sobol    collective          1024
#   sbatch scripts/sensitivity.slurm sobol    collective_lifetime 1024
#   sbatch scripts/sensitivity.slurm benchmark personal
#
# To submit every baseline at once, use the batch wrapper:
#   bash scripts/submit_sensitivity.sh sobol 1024
#
# Resume after a time-limit kill (reuses raw_outputs.csv, only re-runs the
# missing sample_idx values):
#   sbatch --export=ALL,RESUME_DIR=reports/results/sensitivity/<existing-dir> \
#          scripts/sensitivity.slurm sobol collective 1024
#
# Resume + re-run failed samples with a longer timeout (e.g. timeout-corner
# samples that hit the default 7200s wall):
#   sbatch --export=ALL,\
# RESUME_DIR=reports/results/sensitivity/<existing-dir>,\
# SBS_RETRY_FAILED=1,SBS_SOBOL_TIMEOUT=14400 \
#          scripts/sensitivity.slurm sobol collective 512
#
# Args:
#   $1  stage         : smoke | benchmark | morris | sobol  (default: morris)
#   $2  baseline      : personal | collective | collective_lifetime | model
#                       (default: personal)
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
#SBATCH --mem=256G
#SBATCH --time=150:00:00
# Resource note: 32 concurrent ABM subprocesses; each peaks ~2-4 GB at
# corner cases. Empirically `--mem=32G` OOM'd at ~40% of one Morris
# sample (job 9503, 2026-04-25). 256 GB ≈ 8 GB/worker = generous margin.
#
# Time: After the 2026-05-04 collective_memory_climate fix (frozen
# per-year random samples), local A/B benchmark shows ~2.3× speedup;
# expected HPC mean drops from ~926s to ~400s per sample. Sobol N=1024
# / k=5 (12288 samples) at 32 workers ≈ 43h per baseline. 150h (~6.25
# days) leaves >70% headroom under the cluster's 7-day cap and absorbs
# variability from corner-case samples.
# Morris runs in <2 h regardless — it just inherits the same allocation.
#
# Smaller-mem nodes? Drop --cpus-per-task and --mem proportionally
# (keep ~8 GB/worker):
#   --cpus-per-task=16 --mem=128G
#   --cpus-per-task=8  --mem=64G
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

# Resume mode: if RESUME_DIR is set (e.g., via `sbatch --export=ALL,RESUME_DIR=…`),
# the orchestrator reuses that directory's raw_outputs.csv and only runs the
# missing sample_idx values. Use this after a SLURM time-limit kill.
RESUME_FLAG=()
if [[ -n "${RESUME_DIR:-}" ]]; then
    RESUME_FLAG=(--output-dir "$RESUME_DIR")
    echo "[resume] continuing into $RESUME_DIR"
fi

# SBS_RETRY_FAILED=1 (with RESUME_DIR) re-runs status!='ok' rows. Pair with a
# longer SBS_SOBOL_TIMEOUT / SBS_MORRIS_TIMEOUT to push corner-case samples
# past their previous wall-clock limit.
RETRY_FLAG=()
if [[ "${SBS_RETRY_FAILED:-0}" == "1" ]]; then
    RETRY_FLAG=(--retry-failed)
    echo "[retry-failed] only status==ok rows are treated as done"
fi

# Per-stage timeout overrides (defaults match the original tuning).
SBS_MORRIS_TIMEOUT="${SBS_MORRIS_TIMEOUT:-7200}"
SBS_SOBOL_TIMEOUT="${SBS_SOBOL_TIMEOUT:-7200}"

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
            --repeats 2 --years 20 \
            ${RESUME_FLAG[@]+"${RESUME_FLAG[@]}"}
        ;;
    benchmark)
        $PY reports/run_sensitivity.py benchmark \
            --memory-baseline "$BASELINE" \
            --repeats 30 --years 100 --n-samples 5 \
            ${RESUME_FLAG[@]+"${RESUME_FLAG[@]}"}
        ;;
    morris)
        # Default --timeout 7200 (2h). Override via SBS_MORRIS_TIMEOUT for a
        # resume that re-runs corner samples that hit the previous wall.
        # Empirical: 1h timeout caused 22.5% failure rate on collective Morris
        # (job 9683, 2026-05-04) even after the frozen-year-mean fix; 2h drops
        # it to <5%.
        $PY reports/run_sensitivity.py morris \
            --memory-baseline "$BASELINE" \
            --r-trajectories 20 --num-levels 4 \
            --repeats 30 --years 100 \
            --n-workers "$N_WORKERS" \
            --timeout "$SBS_MORRIS_TIMEOUT" \
            --climate-process ar1 \
            ${RESUME_FLAG[@]+"${RESUME_FLAG[@]}"} \
            ${RETRY_FLAG[@]+"${RETRY_FLAG[@]}"}
        ;;
    sobol)
        # Default --timeout 7200. Override via SBS_SOBOL_TIMEOUT (e.g. 14400
        # for a resume targeting collective corner samples).
        # Empirical: collective Sobol N=512 at 7200s timeout had 5.33%
        # failure rate (job 9687, 2026-05-04 .. 2026-05-07), all timeouts
        # in the high-max_age × high-new_agents corner. Doubling to 14400s
        # for the retry pass should clear them.
        $PY reports/run_sensitivity.py sobol \
            --memory-baseline "$BASELINE" \
            --N "$N_SOBOL" \
            --repeats 30 --years 100 \
            --n-workers "$N_WORKERS" \
            --timeout "$SBS_SOBOL_TIMEOUT" \
            --climate-process ar1 \
            ${RESUME_FLAG[@]+"${RESUME_FLAG[@]}"} \
            ${RETRY_FLAG[@]+"${RETRY_FLAG[@]}"}
        ;;
    *)
        echo "Unknown stage: $STAGE" >&2
        echo "Use one of: smoke | benchmark | morris | sobol" >&2
        exit 2
        ;;
esac

echo "[$STAGE/$BASELINE] finished at $(date -Iseconds)"
