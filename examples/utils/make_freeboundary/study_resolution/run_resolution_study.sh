#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
# SPDX-License-Identifier: MIT
#
# Free-boundary resolution & initial/final-state study driver.
#
# Runs the 12-config matrix ONE AT A TIME (laptop-friendly):
#   paths {refresh, normal} x resolutions {m7n6, m11n10}
#                           x optimizers {dogbox, trf, twostage(trf 1.0->jac@10)}
# Low-resolution runs go first. Each run gets its own subfolder; stdout is
# tee'd into out.txt; xs_fixb/, *_opt.h5, *_opt.toml are retained; and after
# each run the init/final Beltrami grids + convergence figure are produced.
#
# Usage:  bash run_resolution_study.sh
# Re-running skips configs whose out.txt already shows completion.

set -u
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIVER=/home/smiet/code/spectre/scripts/run/calc_spectre_eq_freeb.py
PLOTOUT=/home/smiet/code/spectre/scripts/minimization_monitoring/plotout.py
PLOTGRID="$SDIR/plot_init_final.py"
TOTAL_ITERS=20
PER_RUN_TIMEOUT=5400   # 90 min hard cap so a stuck high-res postprocess can't stall the matrix
SUMMARY="$SDIR/study_summary.txt"

# optimizer name -> driver flags
optflags() {
  case "$1" in
    dogbox)   echo "--method dogbox" ;;
    trf)      echo "--method trf" ;;
    twostage) echo "--method trf --x-scale2 jac --switch-iter 10" ;;
  esac
}

echo "# resolution study started $(date)" >> "$SUMMARY"

# Low-res first, then high-res; both paths; three optimizers.
for res in m7n6 m11n10; do
  for path in refresh normal; do
    toml="$SDIR/tomls/ncsx_${path}_${res}.toml"
    if [[ ! -f "$toml" ]]; then echo "MISSING toml: $toml — skipping"; continue; fi
    for opt in dogbox trf twostage; do
      run="$SDIR/run_${path}_${res}_${opt}"
      out="$run/out.txt"
      if [[ -f "$out" ]] && grep -q "equilibrium done" "$out"; then
        echo "== SKIP (already done): ${path}_${res}_${opt}"; continue
      fi
      echo "== RUN: ${path}_${res}_${opt}  $(date)"
      rm -rf "$run"; mkdir -p "$run"
      cp "$toml" "$run/"
      tomlname="$(basename "$toml")"
      flags="$(optflags "$opt")"

      t0=$(date +%s)
      ( cd "$run" && OMP_NUM_THREADS=3 timeout "$PER_RUN_TIMEOUT" \
          mpirun -n 3 "$DRIVER" --bns-trigger outer --total-iters "$TOTAL_ITERS" \
          $flags "$tomlname" > out.txt 2>&1 )
      rc=$?
      t1=$(date +%s); dt=$((t1 - t0))
      echo "walltime_s=$dt rc=$rc" > "$run/walltime.txt"

      bestf=$(grep -E '^iter ' "$out" 2>/dev/null | sed -E 's/.*<f> ([0-9.eE+-]+).*/\1/' | sort -g | head -1)
      niter=$(grep -cE '^iter ' "$out" 2>/dev/null)
      done_=$(grep -c 'equilibrium done' "$out" 2>/dev/null)
      echo "${path}_${res}_${opt}: best_f=${bestf:-NA} iters=${niter} done=${done_} walltime_s=${dt} rc=${rc}" | tee -a "$SUMMARY"

      # Per-run figures (best-effort; don't abort the matrix on a plot failure).
      python "$PLOTGRID" --toml "$run/$tomlname" --xsdir "$run/xs_fixb" \
        --out "$run/grids_init_final.png" --errout "$run/field_err.txt" \
        --title "${path}_${res}_${opt}" > "$run/plot_grid.log" 2>&1 \
        || echo "  (grid plot failed — see $run/plot_grid.log)"
      python "$PLOTOUT" "$out" --save "$run/convergence.png" > "$run/plotout.log" 2>&1 \
        || echo "  (plotout failed — see $run/plotout.log)"
    done
  done
done

echo "# resolution study finished $(date)" >> "$SUMMARY"
echo "ALL DONE. Summary: $SUMMARY"
