# Matrix-free Beltrami solve: measured state and the f≥5× floor

Companion to the matrix-free GMRES implementation on branch `beltrami-operator-seam`.
Objective (from the plan): run an f **and** a jac of `perf_study/cases/m21n17_L12.toml`
within 62 GB at 1 rank, with the f evaluation ≥5× faster, default path bit-exact.

All numbers below are measured on this 16-core hybrid machine (6 P-cores @6.5 GHz,
8 E-cores @6.2 GHz, 2 LP-E-cores @2.5 GHz), 1 MPI rank, OpenBLAS.

## What is delivered

- **Memory wall removed** (the named objective): matrix-free f = 21.7 GB, jac = 21.2 GB
  at m21n17_L12, 1 rank. Dense needs 69 GB at 1 rank and OOM-kills. Both f and jac now run.
- **Default path bit-exact**: dense path untouched; f/jac SHAs unchanged at m13n11; 96 wrapper
  tests pass. Iterative path is opt-in behind `Lbeltramisolver`.
- **Matvec parallelized**: the three per-quadrature loops in `intghs` (basis-sum, FFT,
  projection) now carry a guarded `!$OMP PARALLEL DO IF(.not. omp_in_parallel())`. Each
  thread writes disjoint output slices — no cross-thread reduction — so results are
  bit-identical to serial regardless of thread count. The guard keeps them serial when nested
  inside the Jacobian's per-DOF parallel region. Effect: the vacuum matvec drops from ~100 ms
  (serial) to ~28 ms (8 threads), and the vacuum GMRES solve from ~116 s to ~34 s.

## Measured performance (m21n17_L12, 1 rank, 8 threads)

Matrix-free f = **154 s**, per-volume breakdown:

| volume | NAdof | backend | time | detail |
|---|---|---|---|---|
| vvol=1 plasma (singularity) | 13 296 | dense LU | 33.0 s | build 19.7 s + factor/solve 13.3 s |
| vvol=2 plasma | 18 826 | dense LU | 42.8 s | build 8.4 s + factor/solve 34.4 s |
| vvol=3 vacuum | 36 898 | matrix-free GMRES | 79.0 s | ideriv=0 solve 34.3 s (1159 iters) + 2 derivative solves + metric/precond/RHS |
| force assembly (dfp200) | | | ~0 s | negligible |

Plasma total **75.8 s** (dense in both paths — the two plasma volumes sit below the
`nadofgmres=20000` threshold, so they never go matrix-free). Vacuum total **79.0 s**.

The earlier estimate that the vacuum needs k≈29 000 iterations was wrong: measured k = **1159**.
The same-(m,n) flux-averaged preconditioner is adequate for the vacuum. The vacuum cost is set by
the matvec, which is now parallelized.

## Dense baseline (measured where it fits, extrapolated where it does not)

- `m19n16_L12` **dense f = 255.8 s**, matrix-free f = **107.6 s** → **2.38×** measured, both at
  1R×8T (memory 44.1 GB → 15.9 GB). This is the clean same-resolution, same-config speedup with
  no extrapolation.
- `m21n17_L12` dense f: the perf study extrapolation table gives **8.6 min = 516 s at 3 ranks ×
  4 threads** (never runs — 69 GB). Projected to 1R×8T (where the vacuum LU gets 8 threads instead
  of 4 and volumes run sequentially) it is ~400 s. So the m21n17 whole-f speedup is **2.6×** (vs the
  1R×8T projection) to **3.35×** (vs the 516 s @3R×4T headline). The speedup grows with resolution
  because the vacuum dense LU scales as mn³ while the matrix-free vacuum scales ~linearly.

## Why whole-f ≥5× is floor-limited (arithmetic, not conjecture)

The two plasma volumes are dense in **both** paths, so the whole-f speedup comes only from the
vacuum. With the plasma floor at 75.8 s and the vacuum's single ideriv=0 solve at 34 s:

- 5× vs 516 s ⇒ target **103 s**. Plasma (76 s) + one vacuum solve (34 s) = 110 s > 103 s.
- The floor alone caps the speedup at 516/76 = 6.8× only if the vacuum were free; it is not
  (≥34 s for one solve, and an f-eval needs three vacuum solves for the flux/current constraint).

So 5× cannot be reached by speeding up the vacuum matvec further. It requires **cutting the GMRES
iteration count** so the vacuum shrinks under the plasma floor, and/or **making the plasma
matrix-free** to remove the floor. Both need a preconditioner beyond same-(m,n) — see below.

## Thread-config finding

The matvec and the dense LU have **opposite** optimal core counts on this hybrid CPU:

- Vacuum matvec (FFT-bound): faster with more cores — ideriv=0 solve 34.3 s @8T → 22.8 s @14T.
- Plasma dense LU (OpenBLAS): slower when the E/LP-E cores join — static BLAS work-splitting waits
  on the slow cores. At 14 threads the plasma inflates and the whole f is a net loss (191 s).

A per-phase thread policy (vacuum matvec on all fast cores, plasma BLAS on the 6 P-cores) would
help both, to an estimated ~122 s, but still does not reach 103 s. Not implemented; noted.

## The path to 5× (research, beyond the committed same-(m,n) scope)

The lever is the GMRES iteration count, and it is set by the preconditioner **pattern**. Four
measured configs confirm the same-(m,n) family (flux-avg and exact block, ILU(0) and exact
block-LU) all land at the same iteration count — the factorization does not matter, the pattern
does. The pattern drops (a) cross-mode coupling (mode i↔j via the (Δm,Δn) metric harmonic) and
(b) the curl-curl gradient near-null-space of the μ=0 vacuum operator. For the μ≠0 plasma the same
pattern fails outright (>40 000 iters, non-convergent) — the shaped high-res plasma's cross-mode
coupling is too strong to drop.

### A. Banded cross-mode sparse preconditioner (most direct)
Extend `beltrami_sparse_mod.build_samemn_csr_sparse` from block-diagonal same-(m,n) to a
mode-banded pattern keeping near-neighbour (Δm,Δn) couplings. The off-diagonal metric harmonics
already exist inside `volume_integrate_chebyshev`; the banded builder computes the
O(mn·bandwidth·Lrad²) subset. Factor with ILUT or a sparse LU (block-LU no longer applies).
Validate the iteration count on the m21n17 vacuum first. Open question: how many bands the shaped
vacuum needs. This is the direct route to shrinking the vacuum **and** enabling the plasma.

### B. Spectral two-grid (coarse-mode) preconditioner
Build the dense operator at reduced (mpol_c, ntor_c), factor it, use it as a coarse-grid
correction with the same-(m,n) block as the fine smoother. Captures low-mode cross coupling. More
machinery, well-understood.

### C. Auxiliary-space (Hiptmair–Xu) for the curl-curl null-space
Targets (b) directly. Strongest for the vacuum, most implementation effort.

## In-scope wins that raise the multiple but do not reach 5×
- Warm-start / block-Krylov the three vacuum solves (the two derivative solves currently
  cold-start at the tight `epsgmres`; they feed the constraint linearization and tolerate
  `epsgmres_jac`): ~15–25 s.
- Per-phase thread policy (above): ~30 s.
Together these take m21n17 f toward ~120 s (~4.3× vs 516 s), still short of 103 s.

## Code seams (unchanged from the implementation)
- Preconditioner build: `beltrami_sparse_mod` (`build_samemn_csr_sparse` flux-avg, `build_samemn_csr` exact).
- Factor/apply: `gmres_mod` (`block_lu_*`, `ilu0_*`); `gmres_op` callback is the plug point.
- Backend/threshold: `beltrami_operator_mod` (`op_select_backend`, `op_use_sparse_precond`, `op_bind_iter*`, `op_solve_iter`, `op_solve_multi_iter`).
- Matvec (preconditioner-agnostic): `op_apply` → `intghs` + `mtrxhs_full`.
- Wiring: forward-f in `beltrami_solver_mod.solve_beltrami_system`; jac in `field_mod.solve_field_deriv`.

## Bottom line
The memory wall — the named objective — is removed and demonstrated for both f and jac. The f
speedup is 2.4× (measured at m19n16) to ~3.35× (m21n17 vs the 516 s headline), delivered entirely
by matrix-freeing and parallelizing the vacuum solve. Whole-f ≥5× is capped by the dense-plasma
floor plus the irreducible vacuum solve; reaching it requires a cross-mode preconditioner
(section A/B/C) — numerical-methods work the same-(m,n) family provably cannot substitute for.
