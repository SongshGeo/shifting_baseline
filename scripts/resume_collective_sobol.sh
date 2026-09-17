#!/bin/bash
# =============================================================================
# Complete the H2 (collective) Sobol run from N = 512 to N = 1024 on SLURM.
#
# Background: job 9687 (2026-05-04 .. 05-07) ran collective Sobol at N = 512
# with a 7200 s per-sample timeout; 191 of 3584 samples (5.3 %) timed out, all
# in the high-max_age × high-new_agents corner (spin-up of ~1000 years with
# ~1000 live agents × 30 serial replicates). SALib's Sobol sequence for
# N = 1024 starts with exactly the N = 512 rows (same seed), so the finished
# samples are reused and only the second half plus the failures are run.
#
# Two chained jobs:
#   1. resume  : run the missing sample_idx 3584..7167 (32 workers × 1 process,
#                timeout 4 h — twice the old wall).
#   2. retry   : after job 1 ends (any state), re-run every status != "ok" row
#                with 4 workers × 8 replicate processes (timeout 6 h), so each
#                corner sample runs its 30 replicates in parallel.
#
# Usage (from the repo root on the cluster, after syncing the code AND the
# existing directory reports/results/sensitivity/20260504-201712-sobol-collective):
#   bash scripts/resume_collective_sobol.sh            # submit both jobs
#   bash scripts/resume_collective_sobol.sh --dry-run  # print the sbatch commands
#
# Afterwards check:
#   python - <<'PY'
#   import pandas as pd
#   r = pd.read_csv("reports/results/sensitivity/20260504-201712-sobol-collective/raw_outputs.csv")
#   r = r.drop_duplicates("sample_idx", keep="last")
#   print(len(r), r.status.value_counts())   # expect 7168 rows, all "ok"
#   PY
# If any rows still fail, submit the retry job again (step 2) with a longer
# SBS_SOBOL_TIMEOUT. Do not recompute the indices until all 7168 rows are "ok":
# the analysis fills failed samples with the median, which is what we are
# replacing.
# =============================================================================
set -euo pipefail

RUN_DIR="reports/results/sensitivity/20260504-201712-sobol-collective"
N_SOBOL=1024
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

if [[ ! -f "$RUN_DIR/raw_outputs.csv" ]]; then
    echo "Missing $RUN_DIR/raw_outputs.csv — sync the existing N=512 results first." >&2
    exit 1
fi

submit() {
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "sbatch $*" >&2
        echo "0"
    else
        sbatch --parsable "$@"
    fi
}

# 1. resume: fill sample_idx 3584..7167
JOB1=$(submit \
    --job-name=sbs-sobol-h2-resume \
    --export=ALL,RESUME_DIR="$RUN_DIR",SBS_SOBOL_TIMEOUT=14400,SBS_N_WORKERS=32,SBS_NUM_PROCESS=1 \
    scripts/sensitivity.slurm sobol collective "$N_SOBOL" | tail -1)
echo "[resume] job $JOB1"

# 2. retry: re-run every non-ok row with replicate-level parallelism
JOB2=$(submit \
    --job-name=sbs-sobol-h2-retry \
    --dependency=afterany:"$JOB1" \
    --export=ALL,RESUME_DIR="$RUN_DIR",SBS_RETRY_FAILED=1,SBS_SOBOL_TIMEOUT=21600,SBS_N_WORKERS=4,SBS_NUM_PROCESS=8 \
    scripts/sensitivity.slurm sobol collective "$N_SOBOL" | tail -1)
echo "[retry]  job $JOB2 (runs after $JOB1 finishes)"
