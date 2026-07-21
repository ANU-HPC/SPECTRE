# Integrating the direct free-boundary solve (lvcvacuum) with beltrami-operator-seam

Notes for merging `vc-direct-freeboundary` (Laplace-virtual-casing direct solve, Henneberg et
al. JPP 87, 905870226 (2021)) with `beltrami-operator-seam` (matrix-free GMRES Beltrami backend,
same-(m,n) preconditioner, `Ldirgen` JVP force Jacobian, banded trust-region). Written
2026-07-21 against seam head `4611dde`.

## Where lvcvacuum touches the solve

- `matrices_mod::matrix` calls `vc_matrix_mod::vc_subtract_dvc(lvol, mn, lrad, dMA, NN)` for the
  vacuum volume: the virtual-casing operator is stored as **compact rows** (only the `Lme`/`Lmf`
  normal-field constraint rows are nonzero; layout in `vc_matrix_mod`, `dVCrow`/`dVCidx`) and
  subtracted from those rows of `dMA` in place. Cached on the boundary geometry.
- `matrices_mod::matrixBG` drops `iBns`/`iBnc` from `dMG` when `lvcvacuum == 1`.
- `forces_mod::get_perturbed_rhs` adds `+(d dVC/dx).a0` for `(vvol == Mvol, innout == 0)` via
  `vc_deriv_rhs_add` — per-harmonic, keyed on `dBdX%ii/%irz/%issym`, normally served from an
  all-DOF batch built by `vc_prepare_deriv(Lbatch=.true.)`.

## Three wiring points for the merge

### 1. `op_apply` must include the -dVC term for the vacuum volume (silent wrong answer otherwise)

`beltrami_operator_mod::op_apply` computes `(dMA - mu*dMD).x` matrix-free from the metric
integrals; with `lvcvacuum = 1` the vacuum-volume operator is `(dMA - dVC)`, so the GMRES
backend would silently solve the wrong system. Fix: after the standard action, subtract the
row contribution

```
ax(dVCidx(r)) -= dot(dVCrow(r, 0:NN), x(0:NN)),  r = 1..nvcrow
```

(guard on `lvcvacuum == 1 .and. lvol == Mvol`; ensure `vc_assemble_dvc` ran at the current
geometry). Cost is O(nvcrow * NN) ~ 1e7 flops per matvec at mpol/ntor = 20 — negligible next to
the intghs-projection matvec. The rows are (mn-1) x NN (~150 MB at mpol/ntor = 20); if that
matters, the operator factors further as (radial TT contraction) -> (mn x 2mn mode-coupling
core, ~11 MB) -> (row scatter), which `vc_assemble_dvc` computes as an intermediate (the
projected per-mode fields before the radial fill).

Until wired: `fatal` on `lvcvacuum == 1` with the iterative backend selected for the vacuum
volume, or force that volume onto the dense path.

### 2. `Ldirgen` JVPs hit the per-harmonic hook with a direction-state dBdX (silent wrong answer)

`solve_field_deriv_dir` sets `Ldirgen = .true.` and calls `get_perturbed_rhs`, where the
`vc_deriv_rhs_add` hook fires keyed on `dBdX%ii/%irz/%issym` — meaningless for a general
direction, so `lvcvacuum = 1` JVPs would be wrong. Two options:

- Cheap and exact: the dVC derivative is linear in the DOF, so `J@d`'s dVC term is
  `sum_p d_p * batch(:, p)` over the precomputed batch (`batch_e/batch_o`, layout
  `batch_ii/irz/issym`) — one dense (mn x batch_n) matvec. Requires the batch to be built
  (`vc_prepare_deriv(Lbatch=.true.)`) and psifactor folded the same way the direction
  coefficients fold it.
- Structurally nicer: a direction-mode weight synthesis. `vc_build_deriv_batch` contracts 8
  DOF-independent kernel fields with per-source-point weights (dR0, dRs, dRt, dRz, dZ0, dZs,
  dZt, dZz); for a general direction those weights are one Fourier synthesis of the direction
  coefficients on the source grid (the exact analogue of `direction_metric_response` in
  geometry_mod), then a single 8-kernel contraction (~0.5 s at mpol/ntor = 20, no DGEMM batch).

Until wired: `fatal` on `lvcvacuum == 1 .and. Ldirgen`.

### 3. Preconditioner coverage (performance, not correctness)

The same-(m,n) block preconditioner ignores dVC, which couples all (m,n) within the constraint
rows. Expectation: tolerable — the VC term is a compact (smoothing integral) perturbation, the
kind Krylov handles well — but it should be measured with the perf_study methodology on a
free-boundary case. Escape hatch already present: keep the vacuum volume on the dense path.
If needed, the same-(m,n) diagonal entries of dVC are directly available from `dVCrow` for
folding into the blocks.

## Compatibility positives

- The force Jacobian stays **block-tridiagonal** in the interface index under `lvcvacuum`: dVC
  lives in volume Mvol, whose only movable interface is the plasma boundary, so its
  contribution lands in that diagonal block. The banded trust-region solver's structural
  assumption holds if/when it is extended to free-boundary objectives.
- Memory at mpol/ntor ~ 20 after the compact-row + factorized-trig refactors (this branch):
  stored rows ~150 MB, trig tables < 10 MB, batch harmonics ~2 MB, transient assembly buffers
  ~200 MB. The remaining large consumer on the dense path is the pre-existing (NN+1)^2 Beltrami
  machinery (dMA/dMD/LU/oBI), which the seam's iterative backend eliminates. The dense
  projection tables `prjcos/prjsin` (Ntz x mn, ~340 MB at mpol/ntor = 20) are the next candidate
  for the same tensor factorization if they ever matter.
- Textual conflicts (`matrices_mod`, `forces_mod`, `field_mod`, `wrapper_funcs_mod`,
  `input_list_mod`, `memory_mod`) are small localized insertions on both sides.
