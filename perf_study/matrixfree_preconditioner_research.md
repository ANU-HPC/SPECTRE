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

The innermost Zernike volume does not converge under the same-(m,n) family — measured at
m13n11 under flux-averaged, exact, one cross-mode band, and a coarse-mode two-grid pattern
alike, and at m21n17 it ran 4000 iterations and returned `force_norm` 2.5947 instead of
2.4638. It **degrades the answer rather than failing**, so it is excluded by default via
`nadofgmres_sing` (default: never), and a stalled solve now emits an explicit WARNING
naming the volume. Making it iterative is the open problem; a coarse-mode correction inside
the `pc` closure (`beltrami_operator_mod`) is the designed plug point.

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
