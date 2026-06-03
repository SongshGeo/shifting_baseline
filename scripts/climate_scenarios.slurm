#!/bin/bash
# =============================================================================
# Climate scenario robustness check for the ABM.
#
# Runs four climate forcing scenarios (iid / ar1 / trend / ar1+trend) ×
# two baselines (personal / collective). Much lighter than full Sobol SA:
# 8 ABM batches total, ~1-2 hours on a single node.
#
# Usage:
#   sbatch scripts/climate_scenarios.slurm
#   sbatch scripts/climate_scenarios.slurm 100       # custom repeats
#
# Resume:
#   sbatch --export=ALL,RESUME_DIR=reports/results/climate_scenarios/<dir> \
#          scripts/climate_scenarios.slurm
# =============================================================================
#SBATCH --job-name=sbs-climate-scenarios
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail

REPEATS="${1:-100}"

RESUME_FLAG=()
if [[ -n "${RESUME_DIR:-}" ]]; then
    RESUME_FLAG=(--output-dir "$RESUME_DIR")
    echo "[resume] continuing into $RESUME_DIR"
fi

mkdir -p logs/slurm

if [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
    PY="python"
elif command -v uv >/dev/null 2>&1; then
    PY="uv run python"
else
    echo "Neither .venv nor uv available." >&2
    exit 1
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

$PY scripts/climate_scenarios.py \
    --repeats "$REPEATS" \
    --years 100 \
    --num-process 1 \
    --baselines personal collective \
    --timeout 3600 \
    ${RESUME_FLAG[@]+"${RESUME_FLAG[@]}"}

echo "[climate-scenarios] finished at $(date -Iseconds)"
