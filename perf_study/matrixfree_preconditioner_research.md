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

The viable route to a fast axis volume in an *optimization* is not a sparse pattern but a
**frozen dense preconditioner**: factor the volume's dense operator every K-th geometry step
(1.4 GB resident at m21n17) and use the stale factor as the GMRES preconditioner with the
current matrix-free matvec in between. Successive geometries differ by O(step), so the stale
exact inverse should give k of a few; that removes the per-step O(mn^2 Lrad^2) build (19.7 s)
and LU (13.3 s) from most evaluations. The `pc` closure in `beltrami_operator_mod` is the
plug point. This fixes the convergence *rate*; warm-starting the *solution* (see below) only
shortens the distance and cannot rescue a stalled solve.

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
  *rate* stalls (vol 1). Already implemented per-column in the Jacobian
  (`op_solve_multi_iter` seeds each column from the previous — spectrally neighbouring
  columns are similar). Not yet wired across optimizer iterations for the forward solve; a
  module-persistent per-volume store (SPEC's `GMRESlastsolution` pattern) is the designed
  follow-up, with seed-only semantics: never return the stale store as an answer, always
  judge convergence on the true residual.
- **Freezing the preconditioner** (previous *factorization*) improves the *rate*: a stale
  exact inverse of a nearby operator clusters the spectrum at 1 + O(step). This is the lever
  that would make the axis volume cheap in an optimization loop, and it also applies to the
  other volumes (skip the per-mu preconditioner rebuild while k stays low).
