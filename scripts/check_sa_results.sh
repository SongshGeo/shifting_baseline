#!/bin/bash
# Inspect the two most-recent Sobol SA runs.
#
# Usage (from repo root):
#   bash scripts/check_sa_results.sh
#
# Prints:
#   1. SLURM job status
#   2. File inventory + raw_outputs.csv mtime
#   3. Per-sample failure rate (status + NaN)
#   4. Failure-mode analysis (which params drive timeouts)
#   5. Sobol indices (S1, ST + 95% CI)
#   6. NaN-fill bias check (was nanmedian-fill triggered?)

set -uo pipefail

# Pick a python that actually has pandas. The bare `python3` on the cluster
# does not — must go through uv or the project venv.
if command -v uv >/dev/null 2>&1; then
    PY=(uv run python)
elif [[ -x ".venv/bin/python" ]]; then
    PY=(.venv/bin/python)
else
    echo "Error: need uv or .venv/bin/python (system python3 has no pandas)" >&2
    exit 1
fi

DIRS=($(ls -dt reports/results/sensitivity/2026*-sobol-* 2>/dev/null | head -2))

echo "================================================================"
echo "[1/6] SLURM job history (last 12 sbs-sa jobs)"
echo "================================================================"
sacct -u "$USER" --name=sbs-sa --starttime=2026-04-25 \
      --format=JobID,JobName,State,Elapsed,ExitCode,Submit -X 2>/dev/null \
    | tail -n 13

echo
echo "================================================================"
echo "[2/6] File inventory + raw_outputs.csv mtime"
echo "================================================================"
for d in "${DIRS[@]}"; do
    echo
    echo "--- $d ---"
    ls "$d"/*.csv "$d"/*.json "$d"/*.txt 2>/dev/null
    n_sample_dirs=$(ls -d "$d"/sample_* 2>/dev/null | wc -l)
    echo "  sample_NNNNNN sub-dirs: $n_sample_dirs (left-over from failed runs)"
    f="$d/raw_outputs.csv"
    [[ -f "$f" ]] && stat -c '  raw_outputs.csv mtime: %y' "$f" 2>/dev/null \
                  || stat -f '  raw_outputs.csv mtime: %Sm' "$f" 2>/dev/null
done

echo
echo "================================================================"
echo "[3/6] Per-sample failure rate"
echo "================================================================"
for d in "${DIRS[@]}"; do
    echo
    echo "--- $d ---"
    "${PY[@]}" - <<EOF
import pandas as pd
df = pd.read_csv("$d/raw_outputs.csv")
total = len(df)
ok = (df["status"] == "ok").sum()
print(f"  total rows         : {total}")
print(f"  status == ok       : {ok} ({100*ok/total:.2f}%)")
print(f"  failed             : {total - ok} ({100*(total-ok)/total:.2f}%)")
bad = df[df["status"] != "ok"]
if len(bad):
    print(f"  failure modes:")
    print(bad["status"].value_counts().to_string().replace("\n", "\n    "))
    print(f"  median elapsed (all completed): {df[df['status']=='ok']['elapsed_seconds'].median():.0f}s")
    print(f"  max    elapsed (all completed): {df[df['status']=='ok']['elapsed_seconds'].max():.0f}s")
else:
    print(f"  median elapsed: {df['elapsed_seconds'].median():.0f}s")
    print(f"  max    elapsed: {df['elapsed_seconds'].max():.0f}s")
EOF
done

echo
echo "================================================================"
echo "[4/6] Failure-mode analysis: which params drive timeouts?"
echo "================================================================"
for d in "${DIRS[@]}"; do
    echo
    echo "--- $d ---"
    "${PY[@]}" - <<EOF
import pandas as pd
df = pd.read_csv("$d/raw_outputs.csv")
bad = df[df["status"] != "ok"]
ok  = df[df["status"] == "ok"]
if len(bad) == 0:
    print("  (no failures)")
else:
    print(f"  Comparing parameter distributions: failed (n={len(bad)}) vs ok (n={len(ok)})")
    print(f"  {'param':<15} {'ok mean':>12} {'fail mean':>12} {'ratio':>8}")
    for p in ["max_age", "new_agents", "loss_rate", "climate_sigma", "climate_phi"]:
        if p in df.columns:
            ok_m, bad_m = ok[p].mean(), bad[p].mean()
            ratio = bad_m / ok_m if ok_m else float("nan")
            print(f"  {p:<15} {ok_m:>12.3f} {bad_m:>12.3f} {ratio:>8.2f}x")
    print()
    print(f"  Failed sample param ranges:")
    print(f"    max_age      : {bad['max_age'].min():.0f} .. {bad['max_age'].max():.0f}")
    print(f"    new_agents   : {bad['new_agents'].min():.0f} .. {bad['new_agents'].max():.0f}")
    print(f"    loss_rate    : {bad['loss_rate'].min():.3f} .. {bad['loss_rate'].max():.3f}")
    print()
    print(f"  Indices of failed samples (first 30):")
    print(f"    {bad['sample_idx'].head(30).tolist()}")
EOF
done

echo
echo "================================================================"
echo "[5/6] Sobol indices (S1, ST + 95% CI)"
echo "================================================================"
for d in "${DIRS[@]}"; do
    for metric in peak_window_mean peak_strength_mean; do
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
echo "[6/6] NaN-fill bias check"
echo "================================================================"
echo "If failure rate > 0, run_sobol() replaces NaN with np.nanmedian(Y) before"
echo "analysis. This is a soft fallback — indices are still usable but variance"
echo "contributions from the failed corner are biased toward the median."
echo
for d in "${DIRS[@]}"; do
    if [[ -f "$d/ERRORS.txt" ]]; then
        echo "--- $d/ERRORS.txt ---"
        cat "$d/ERRORS.txt"
    fi
done

echo
echo "================================================================"
echo "[bonus] Generating PNG visualizations"
echo "================================================================"
if command -v uv >/dev/null 2>&1; then
    uv run python reports/plot_sobol.py 2>&1 | sed 's/^/  /'
elif [[ -f ".venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
    python reports/plot_sobol.py 2>&1 | sed 's/^/  /'
else
    echo "  (skipped — neither uv nor .venv available)"
fi

echo
echo "================================================================"
echo "Interpretation guide:"
echo "  - failure rate < 1% : indices fully reliable"
echo "  - 1-5%              : usable, disclose nanmedian-fill in SI"
echo "  - 5-10%             : borderline — consider resume with longer timeout"
echo "                        on the failed sample_idx values only"
echo "  - > 10%             : do not use; resume required"
echo "================================================================"
