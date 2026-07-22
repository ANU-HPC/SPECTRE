# Matrix-free Beltrami solve: the exact same-(m,n) preconditioner

Companion to the matrix-free GMRES work on branch `beltrami-operator-seam`.
All numbers measured on this 16-core hybrid machine (6 P-cores, 8 E-cores, 2 LP-E),
1 MPI rank, 8 OMP threads, OpenBLAS.

## Result

At `perf_study/cases/m21n17_L12.toml`, 1 rank, shipped defaults:

| | dense | matrix-free (before) | matrix-free (now) |
|---|---|---|---|
| force eval | 516 s (extrapolated; OOMs) | 141 s | **39.1 s** |
| speedup | 1x | 3.65x | **13.2x** |
| peak RSS | 69 GB (OOM-kills) | 21.7 GB | **15.0 GB** |
| vacuum GMRES iterations | n/a | 1159 | **46** |
| force_norm | | 2.463768e+01 | 2.463768e+01 |

The Jacobian at m21n17 completes in **79 min at 15.0 GB** (was 5.3 h at 21.2 GB with the
flux-averaged preconditioner; dense needs 69 GB and cannot run at 1 rank).

The dense path is untouched and bit-exact (m13n11: first-f `1.245e+02`,
fjac `8.484e+12`). 98 wrapper tests pass at 8 threads.

## What was actually wrong (two separate bugs)

### 1. The preconditioner was a tokamak approximation

`build_samemn_csr_sparse` ("flux-averaged") kept only the `g^(0,0)` metric harmonic.
The true diagonal mode-pair pulls two harmonics — `kijs(ii,ii)` -> `(0,0)` divisor 1 and
`kija(ii,ii)` -> `(2m_i, 2n_i)` divisor 2:

```
exact diagonal tensors = g^(0,0)  ±  1/2 · g^(2m_i, 2n_i)
```

and it dropped the `g_{s theta}` / `g_{s zeta}` (`TDstsc`/`TDszsc`) terms entirely. For a
shaped stellarator that is a poor approximation. The in-tree test encoded it: the sparse
builder needed a **0.1** tolerance against the dense extraction on the QA stellarator.

The fix (`build_samemn_csr_diag`) reads the tensors from
`volume_integrate_chebyshev(..., diag_only=.true.)` — the same integration the dense
assembler uses, restricted to `jj == ii`, at O(mn·lquad·Lrad²) instead of
O(mn²·lquad·Lrad²) — and evaluates the same `Wtete/Wzete/Wteze/Wzeze`, `Hzete/Hteze`
expressions. It is **bitwise identical** (0.000e+00) to the dense extraction on both the
axisymmetric and the shaped case, so the test tolerance is now `1e-12` for both.

Two things fall out rather than needing special cases: the `ii == 1` degeneracy is already
produced by the `kijs`/`kija` divisors (that is what `factorcc`/`factorss` stood in for),
and `pi2pi2nfphalf` must **not** be re-applied — the integration already scales by it.

### 2. The diagonal regularization was destroying the block

Even with exact values the solve barely improved. The CSR builders boost a near-zero
diagonal to `max(rowmax, 1)`. That dates from the **unpivoted** ILU(0), which needs nonzero
pivots. The Lagrange rows of this saddle-point operator genuinely have zero diagonals, and
`block_lu` factors each mode block with **partial pivoting** (DGETRF) — handling that
exactly, and shifting a block itself if it really is singular. The boost was therefore
unnecessary and perturbed the block badly enough to dominate everything else:

| m13n11 vacuum | k |
|---|---|
| flux-averaged block | 475 |
| exact block, diagonal boosted | **953** (worse than flux-averaged) |
| exact block, no boost | **44** |

This is why the exactness win was invisible until the boost came out. Offline scipy
experiments (`precond_lab.py`) predicted k≈36 for this case using pivoted `splu` on the raw
block — the discrepancy against Fortran was entirely the regularization.

## Per-volume behaviour

| volume | NAdof | backend | k |
|---|---|---|---|
| vvol=1 plasma (coordinate singularity) | 13 296 | **dense** (gated off) | does not converge |
| vvol=2 plasma (interior) | 18 826 | matrix-free | 88 |
| vvol=3 vacuum | 36 898 | matrix-free | 46 |

The innermost Zernike volume does not converge under the same-(m,n) family, and follow-up
structure measurements (scratchpad `vol1_lab.py` / `vol1_lab2.py`, m13n11 vol 1) show why —
and why no truncation-pattern preconditioner can fix it:

- **The mode-space coupling does not decay.** RMS block magnitude vs mode separation, relative
  to the diagonal: vol 2 (regular annulus) falls to 5-8% at distance one — perturbative, which
  is why block-diagonal works there. Vol 1 sits flat at 10-22% out to at least (dm=4, dn=3),
  in every signed direction (no helical ridge). Near the axis the mapping is polar-like and
  fully shaped, so the metric's Fourier content is O(1)-broad; the mode-space operator is
  effectively dense.
- **Diagonal dominance collapses with m.** ||A_mm||_F / ||A_m,rest||_F is 0.49 at m=0 and falls
  monotonically to 0.13 at m=12: the Zernike basis functions scale like s^m (concentrated at the
  outer edge, vanishing at the axis), so a high-m block's self-coupling shrinks while its
  coupling to everything else stays O(1). The blocks are *never* dominant, and any block-Jacobi
  preconditioner amplifies the neglected coupling by the inverse of those weak blocks.
- Measured accordingly: same-(m,n), same-n (all m), same-m (all n), same-n with |dm|<=4, one
  cross-mode band, and a coarse-mode two-grid all stall (>=1200 iterations, GMRES(60)); a wide
  band (|dm|<=6, |dn|<=3) drives the LU toward dense fill without converging. At m21n17 the
  stalled solve returned `force_norm` 2.5947 instead of 2.4638 — it **degrades the answer
  rather than failing** — so the volume is excluded by default via `nadofgmres_sing` and a
  stalled solve emits an explicit WARNING naming the volume.

A **frozen dense preconditioner** (keep the previous step's dense LU resident, use it as the
GMRES preconditioner across optimizer steps) was the natural next idea — and it is **measured
infeasible** for this volume (scratchpad `sens_lab.py` / `drift_lab.py`, m13n11):

| |dx|/|x| | vol 1 |dA|/|A| | vol 2 |dA|/|A| | vol 1 frozen-P k (1e-9) |
|---|---|---|---|
| 1e-6 | 6.4e-2 | 1.0e-5 | 9 |
| 1e-5 | 5.0e-1 | 1.0e-4 | 28 |
| 1e-4 | 1.3 | 1.0e-3 | fails (>400) |

Extraction is bitwise deterministic and repeated force evaluations at fixed geometry
reproduce the operator exactly, so this is genuine geometry sensitivity, not state drift.
The regular volume responds linearly with amplification ~10; the axis volume's amplification
is ~6e4 and **sublinear** in the step — the signature of non-smooth dependence, consistent
with the fractional-power (s^(m/2)) axis regularization making dA/dx effectively unbounded.
A frozen factor survives only steps below ~1e-5 relative — the extreme endgame — so a
k-triggered refresh would fire on essentially every real optimizer step, reducing frozen
mode to today's dense rebuild plus a wasted GMRES attempt.

**Conclusion: vol 1 stays dense, definitively.** Both escape routes are now measured shut:
sparse-pattern preconditioners fail because the operator is dense in mode space with
collapsing diagonal dominance, and cross-step factor reuse fails because the operator is
hypersensitive to the geometry through the axis. Its ~33 s (19.7 s tensor build + 13.3 s LU
at m21n17) is the irreducible per-evaluation floor short of changes to the axis-volume
formulation itself.

What *does* survive for cross-step reuse is warm-starting the **solution** of the volumes
that already converge (vol 2, vacuum): their operators vary smoothly (amplification ~10),
the persistent `Ate/Aze` store already holds the previous converged solution per volume,
and seeding costs one `packab('P')` call per solve.

**Cross-mode banding is not the lever.** Two-grid moved the interior plasma only 29 -> 27
at 17x the nonzeros, and did not rescue vvol=1 at production scale. The earlier conclusion
in this file — that the same-(m,n) family was exhausted and cross-mode research was
required — was wrong twice over: the family was never built exactly, and then its exact
form was being corrupted by the regularization.

## Flags

- `Lbeltramisolver` 0 off / 1 threshold / 2 force-iterative. Default 0.
- `nadofgmres` 5000 (was 20000; the old value suited the flux-averaged block, which only
  paid off on the very largest volume).
- `nadofgmres_sing` default `huge` — the coordinate-singularity volume opt-in.
- `nadofgmres_sparse` deprecated, no longer consulted.

## Code seams
- Exact builder: `beltrami_sparse_mod.build_samemn_csr_diag`; the dense-extraction
  `build_samemn_csr` is retained as the independent test oracle — do not delete it.
- Diagonal integration: `chebyshev_mod.volume_integrate_chebyshev(diag_only=)`, with
  `tensor_jslot()` the shared writer/reader convention for the collapsed tensors.
- Factorization: `gmres_mod.block_lu_*` (exact per-mode-block pivoted LU).
- Backend selection: `beltrami_operator_mod.op_select_backend`.

## Solution reuse across steps (warm starts)

Two distinct reuse levers, often conflated:

- **Warm-starting the solution** (x0 = previous solution) shortens the *distance* GMRES must
  travel: iterations scale with log(||r0||/tol), so a good seed saves a constant factor, most
  of it late in an optimization when geometry steps are small. It cannot rescue a solve whose
  *rate* stalls (vol 1). **Implemented** for the forward solves: each iterative solve seeds
  from the persistent `Ate/Aze` store (slots 0:2), written by `packab('U')` at the end of
  every solve — so seeding acts across force evaluations *and* across mu-iterations within
  one. Seed-only semantics are structural (GMRES judges the true residual; a zero store
  degenerates to a cold start). Measured: at m21n17, evaluations 2-4 of an optimization-like
  sequence (per-harmonic steps 1e-4..1e-3) take 11-23% fewer GMRES iterations than a cold
  reference (304-345 vs 387), growing to ~50% at 1e-8 steps (qa_7vol: 180 -> 86); never
  worse, and forces match the cold reference exactly. The per-column Jacobian warm start
  (`op_solve_multi_iter`) was already in place. The `niters_gmres_total` counter in
  allglobal exposes the iteration count to Python for tests and monitoring.
- **Freezing the preconditioner** (previous *factorization*) improves the *rate*: a stale
  exact inverse of a nearby operator clusters the spectrum at 1 + O(step). This is the lever
  that would make the axis volume cheap in an optimization loop, and it also applies to the
  other volumes (skip the per-mu preconditioner rebuild while k stays low).

## 2026-07-22: the map, not the volume — a conformal interior mesh fixes vol 1

The "vol 1 stays dense, definitively" conclusion above was scoped "short of changes to
the axis-volume formulation itself". That excluded route is measured decisive
(`perf_study/analysis/vol1_map_lab.py` / `.out`; m13n11_L12_prof, vol-1 NAdof 4230,
mu(1)=0 so A = dMA): replace the default sbar^m regularized blend with a **conformal
map2disc mesh** of interface 1 onto the vol-1 Gauss nodes, through the existing
`use_ext_mesh` seam (nothing else changed — same Zernike basis, same integrator, which
picks the mesh up automatically via compute_coordinates).

| vol 1, same-(m,n) family | sbar^m blend | conformal mesh |
|---|---|---|
| RMS coupling at mode distance 1 / 4 | 0.19-0.22 / 0.12-0.16 (flat) | 0.16 / **<= 0.001** (decaying) |
| block dominance \|\|A_mm\|\|/\|\|A_m,rest\|\|, m=0 -> 12 | 0.36 -> 0.045 (collapse) | **2.2 -> 4.4** (dominant, no collapse) |
| exact-block GMRES(60) k to 1e-9 | stall (1200 cap) | **47, converged** |

Confirmed at production scale (`vol1_map_lab21.py` / `.out`; m21n17_L12, vol-1
NAdof 13 296, mn=753): baseline stalls identically (coupling flat 0.07-0.17,
dominance 0.354 -> 0.040 at m=20); conformal mesh decays to <= 0.001 by distance 3,
dominance 1.5-10.6 at every m, **k=50 converged** (2.4 s in scipy). k = 47 -> 50
while NAdof tripled: the iteration count is resolution-independent, as the
scattering picture predicts. The mesh stays fold-free and matches the boundary to
1.3e-5 at the outermost node.

The conformal mesh is fold-free by construction (sign-definite sqrt(g) measured on the
quadrature grid) and lands on the same boundary (outermost-node R harmonics match the
blend to 3e-5). k=47 puts vol 1 in the same league as the regular volumes (44-88), so
the ~33 s "irreducible" dense floor at m21n17 (19.7 s tensor + 13.3 s LU) is not
irreducible — it is a property of the blend map.

Physics: a per-plane conformal map makes the in-plane metric isotropic,
g = |f'(w)|^2 delta — the s-theta cross terms vanish and the mode scattering is the
Fourier spectrum of the analytic |f'|^2, which decays exponentially; the sbar^m blend's
near-axis distortion is what made the coupling O(1)-broad and the high-m blocks weak.

Caveats / remaining steps: (i) both cases have mu(1)=0 (A = dMA) — confirm with
mu /= 0; (ii) forward solve + exact preconditioner under use_ext_mesh need no new
physics code, so an in-solver validation (Lbeltramisolver with nadofgmres_sing lowered,
ext mesh armed) is cheap; (iii) using this inside the optimizer needs the ext-mesh
derivative path (general-direction mode currently fatals; per-DOF factors skipped) —
the freeze-per-iterate mesh strategy. Also fixed: gen_mesh_external sampled the padded
gaussianabscissae column (max Iquad) instead of Iquad(1) rows — broadcast error
whenever Iquad varies across volumes.
