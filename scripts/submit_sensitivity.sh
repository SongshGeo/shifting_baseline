#!/bin/bash
# =============================================================================
# Batch-submit the global sensitivity analysis across memory baselines.
#
# Fans out one `sbatch scripts/sensitivity.slurm <stage> <baseline> <N>` job
# per baseline so all baselines run concurrently as independent SLURM jobs.
# Each job lands in its own timestamped dir under
# reports/results/sensitivity/ (named *-<stage>-<baseline>), which is exactly
# what reports/abm.ipynb's `latest_sobol_dir(baseline)` picks up.
#
# Usage (from repo root, on a SLURM cluster):
#   bash scripts/submit_sensitivity.sh                 # sobol N=1024, all 3 baselines
#   bash scripts/submit_sensitivity.sh morris          # morris, all 3 baselines
#   bash scripts/submit_sensitivity.sh sobol 512       # sobol N=512, all 3 baselines
#
# Override which baselines to submit (space-separated) via BASELINES, e.g.
# to add only the new collective_lifetime scenario to an existing run:
#   BASELINES="collective_lifetime" bash scripts/submit_sensitivity.sh sobol 1024
#
# Args:
#   $1  stage          : smoke | benchmark | morris | sobol  (default: sobol)
#   $2  N (sobol only) : Saltelli base size                  (default: 1024)
# Env:
#   BASELINES          : space-separated baselines
#                        (default: "personal collective collective_lifetime")
# =============================================================================
set -euo pipefail

STAGE="${1:-sobol}"
N_SOBOL="${2:-1024}"
BASELINES="${BASELINES:-personal collective collective_lifetime}"

SLURM_SCRIPT="scripts/sensitivity.slurm"
if [[ ! -f "$SLURM_SCRIPT" ]]; then
    echo "Run from repo root: $SLURM_SCRIPT not found." >&2
    exit 1
fi

echo "[submit] stage=$STAGE N=$N_SOBOL baselines: $BASELINES"
for baseline in $BASELINES; do
    jid=$(sbatch --parsable "$SLURM_SCRIPT" "$STAGE" "$baseline" "$N_SOBOL")
    echo "  submitted $STAGE/$baseline -> job $jid"
done
echo "[submit] all jobs queued. Track with: squeue -u \"$USER\""
