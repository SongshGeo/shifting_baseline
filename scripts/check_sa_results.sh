#!/bin/bash
# Inspect the two most-recent Sobol SA runs.
#
# Usage (from repo root):
#   bash scripts/check_sa_results.sh
#
# Prints:
#   1. SLURM job status (last 10 sbs-sa jobs)
#   2. File inventory of the two newest sobol dirs
#   3. mtime of raw_outputs.csv (sanity-check that the two dirs are independent)
#   4. Per-sample failure rate (status column + NaN counts)
#   5. Sobol indices (S1, ST + 95% CI) for peak_window and peak_strength

set -uo pipefail

echo "================================================================"
echo "[1/5] SLURM job history (last 10 sbs-sa jobs)"
echo "================================================================"
sacct -u "$USER" --name=sbs-sa --starttime=2026-04-25 \
      --format=JobID,JobName,State,Elapsed,ExitCode,Submit -X 2>/dev/null \
    | tail -n 11

echo
echo "================================================================"
echo "[2/5] Two newest sobol output directories"
echo "================================================================"
DIRS=($(ls -dt reports/results/sensitivity/2026*-sobol-* 2>/dev/null | head -2))
if [[ ${#DIRS[@]} -lt 2 ]]; then
    echo "WARNING: found only ${#DIRS[@]} sobol dir(s); expected 2." >&2
fi
for d in "${DIRS[@]}"; do
    echo
    echo "--- $d ---"
    ls -la "$d/"
done

echo
echo "================================================================"
echo "[3/5] raw_outputs.csv mtime (confirm dirs are independent runs)"
echo "================================================================"
for d in "${DIRS[@]}"; do
    f="$d/raw_outputs.csv"
    [[ -f "$f" ]] && stat -c '%y  %n' "$f" 2>/dev/null \
                  || stat -f '%Sm  %N' "$f" 2>/dev/null
done

echo
echo "================================================================"
echo "[4/5] Per-sample failure rate (status + NaN check)"
echo "================================================================"
for d in "${DIRS[@]}"; do
    echo
    echo "--- $d ---"
    python3 - <<EOF
import pandas as pd, sys
try:
    df = pd.read_csv("$d/raw_outputs.csv")
except Exception as e:
    print(f"  ERROR reading raw_outputs.csv: {e}")
    sys.exit(0)
total = len(df)
ok = (df["status"] == "ok").sum() if "status" in df else float("nan")
nan_w = df["peak_window_mean"].isna().sum() if "peak_window_mean" in df else float("nan")
nan_s = df["peak_strength_mean"].isna().sum() if "peak_strength_mean" in df else float("nan")
print(f"  total rows         : {total}")
print(f"  status == ok       : {ok} ({100*ok/total:.2f}%)")
print(f"  NaN peak_window    : {nan_w}")
print(f"  NaN peak_strength  : {nan_s}")
if "status" in df:
    bad = df[df["status"] != "ok"]
    if len(bad):
        print(f"  non-ok status counts:")
        print(bad["status"].value_counts().to_string())
        print(f"  median elapsed (failed)  : {bad['elapsed_seconds'].median():.0f}s")
        print(f"  max    elapsed (failed)  : {bad['elapsed_seconds'].max():.0f}s")
print(f"  median elapsed (all): {df['elapsed_seconds'].median():.0f}s")
print(f"  max    elapsed (all): {df['elapsed_seconds'].max():.0f}s")
EOF
done

echo
echo "================================================================"
echo "[5/5] Sobol indices (S1 / ST with 95% CI)"
echo "================================================================"
for d in "${DIRS[@]}"; do
    for metric in peak_window peak_strength; do
        f="$d/sobol_${metric}.csv"
        echo
        echo "--- $f ---"
        if [[ -f "$f" ]]; then
            cat "$f"
        else
            echo "  (missing)"
        fi
    done
done

echo
echo "================================================================"
echo "Done. If failure rate is < 5% and all four sobol_*.csv are present,"
echo "results are usable for the SI."
echo "================================================================"
