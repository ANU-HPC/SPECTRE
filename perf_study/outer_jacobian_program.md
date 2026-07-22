<!-- SPDX-FileCopyrightText: 2026 Proxima Fusion -->
<!-- SPDX-License-Identifier: MIT -->

# Outer force-Jacobian program: banded trust-region step + matrix-free JVP

Implementation of the program proposed in `force_jacobian_investigation-4.pdf`
(evidence base measured 2026-07-13): exploit the measured block-tridiagonal
structure of the outer force Jacobian and its cheap directional action. Three
pieces landed -- the banded step (1), the matrix-free JVP (2), and the
lagged-banded iteration consuming both (3); colored (probing) assembly was
evaluated and dropped.

## 1. Banded normal-equation trust-region step (`tr_solver='banded'`)

`spectre/least_squares/banded.py`, wired into the vendored trf (`trf_no_bounds`).
J is exactly block-tridiagonal in the interface index (out-of-band content is only
the ~1e-8-relative Lconstraint=3 dmupfdx coupling), so N = J^T J is
block-pentadiagonal: the More alpha-iteration runs on banded Cholesky
factorizations instead of one dense SVD of J per outer iteration.

- test_qa_nvol7 dimensions (ngdof=1734, 6 interfaces, m=6144): step solve
  1.80 s -> 0.39 s (4.6x), steps identical to 6e-15 relative. Cost scales
  O(interfaces) instead of O(interfaces^3).
- Guards (warn + fall back to the exact SVD, never silently): measured
  out-of-band mass > 1e-6, non-divisible block layout, inaccurate
  normal-equation solve at extreme conditioning (interior Gauss-Newton steps
  additionally require an iterative-refinement contraction, because at
  cond(N) ~ 1/eps a small residual does not certify the step).
- Opt-in: `min_params.tr_solver = 'banded'` (method='trf', fixed-boundary
  freal objective). Default behaviour unchanged.

## 2. Matrix-free Jacobian-vector products (`force_real_jvp`)

Fortran `solve_field_deriv_dir` (field_mod) evaluates J @ d with ONE
general-direction linearised solve per volume instead of one solve per
(interface, harmonic) column. The generality is confined to two sites that were
the only analytic consumers of the harmonic index:

- `compute_coordinates` (geometry_mod): a general-direction branch
  (`direction_metric_response`) synthesizes TOTAL displacement fields
  dR(:,0:3)/dZ(:,0:3) from direction coefficients (both boundaries, R and Z,
  cosine and sine, per-mode fj factors at the axis, moving-axis response
  fields), then evaluates universal first-order variation formulas of which
  the per-harmonic irz/issym branches are special cases (verified by algebraic
  expansion for all six existing formula sets).
- `volume` (volume_mod): general dvolume accumulated per boundary.

Everything downstream (intghs RHS projections, lbpol, compute_plasma_current,
compute_rotational_transform, force_real_evaldbb) is linear in those fields and
runs unmodified. The per-DOF local and global row assemblies contract to one
unified form; psifactor is folded into the direction coefficients. The direction
mode is off by default (`Ldirgen`), and the per-harmonic code paths are
untouched, so the dense build is bit-identical.

Measured (6 threads):

| case | dense jac build | single JVP call | marginal JVP |
|---|---|---|---|
| test_qa_nvol7 (ngdof=1734, 6 interfaces) | 9.7 s | 2.7 s (fixed factor build) | **0.11 s** (~88x) |

Golden agreement with the dense Jacobian: rel <= 2e-14 on qa_7vol, cyl_manyvol
(lcons 0/1/3 each) and test_qa_nvol7, for unit directions and random directions
spanning every interface (`tests/wrapper/test_force_jac_jvp.py`).

Contracts and limits: single MPI rank; `solve_field` must have run at the same
geometry (same contract as `solve_field_deriv`); Lconstraint=2 unsupported
(fatal); the adiabatic-pressure dvolume term uses each volume's own general
dvolume, while the per-DOF build reads a single module `dvolume` last set for
the perturbed interface's volume — identical whenever adiabatic = 0 (all
current cases; the per-DOF behaviour looks like an inherited quirk, see
volume_mod's innout wipe of dvolume).

## Colored (probing) assembly: evaluated, dropped

The PDF's third component (3-color probe directions building the sparse J in
3*lgdof probes) reduces the *probe count*, but with the JVP costing one
RHS-build + solve per affected volume the total serial work equals the dense
build's; the claimed build-cost independence of volume count holds only under
per-volume parallel wall-clock accounting. The serial wins are carried entirely
by pieces 1 and 2. Revisit if per-volume MPI parallel JVPs land.

## 3. Lagged-banded matrix-free outer iteration (`min_params.jvp_lag`)

The consumer of pieces 1 and 2 (`spectre/least_squares/lagged.py`, wired into
`trf_no_bounds`; opt-in `tr_solver='banded'` + `jvp_lag=k`). Between dense
Jacobian rebuilds the iteration keeps the banded factorization of the stale J as
a subspace source -- lagged banded trust-region direction, lagged gradient,
previous accepted step -- and refreshes the *model* every iterate with a batch of
fresh JVPs W = J Q. The step solves min ||f + W y|| over ||y|| <= Delta exactly
(small SVD); the predicted reduction and gain ratio come from the fresh W, so the
radius control stays honest under a stale subspace.

Safety invariant: convergence is never declared from lagged quantities. Any
termination test that fires while J is stale is converted into a dense rebuild
and a fresh recheck; a rejected step under a lagged model likewise forces a
rebuild instead of shrinking the radius against a possibly wrong subspace.

Measured, test_qa_nvol7 (ngdof=1734, 6 interfaces), 6 trf iterations, identical
settings/thread environment, sequential runs on one box:

| mode | wall | final cost | dense J builds | JVP batches |
|---|---|---|---|---|
| exact SVD | 1206.9 s | 2.4164e-05 | 7 | 0 |
| lagged-banded (jvp_lag=3) | **610.7 s (2.0x)** | 2.4262e-05 (+0.4%) | **3** | 4 (k=3) |

Steady-state lagged iteration ~41 s vs ~184 s exact (4.4x): the exact iteration
pays ~28 s dense build + ~100 s SVD of the (6144, 1734) Jacobian, the lagged one
a ~16 s JVP batch plus the force evaluation (~24 s), which now dominates. The
guards fired in the wild during this run: the first iterate tripped the banded
solver's conditioning fallback (piece 1's per-iteration SVD), and one late
low-quality lagged step was rejected and converted into a fresh rebuild.

Composes with the jurasic-lagged-jacobian MR (jac_lag), which gates stale-J
reuse but has no fresh-model correction.

## 4. Factor sharing between solve_field and the JVP batch

The JVP batch's cost was dominated by a fixed per-call overhead: rebuilding, for
every volume, the metric-integral tensors and the Beltrami factorization (dense
LU, or the same-(m,n) preconditioner) that the immediately preceding force
evaluation had just built at the same geometry. Measured at n7m09n07_L12
(ngdof=1995, 7 interfaces): fixed ~9.0 s per call vs marginal ~0.63 s per
direction, so a k=3 batch cost ~10.6 s against an 18.7 s dense build.

Now shared through an opt-in per-volume cache (`beltrami_operator_mod`
op_cache_*, master switch allglobal `Lbeltrami_factor_cache`, enabled by the
optimizer when `min_params.jvp_lag` is active): solve_beltrami_system deposits
each volume's final factorization (the last call of the mu-iteration is the
accepted one), dfp100 drops a volume's slot before re-solving it, and
solve_field_deriv_dir takes the handle back (move, not copy; matched on mu and
NAdof, silent rebuild on a miss) and returns it afterwards, so repeated batches
at one geometry all skip the build.

Measured (n7m09n07_L12, lcons=3): JVP k=3 batch 10.6 s -> **0.67 s** (~15x);
products bitwise identical with and without the cache and 1.3e-14 relative to
the dense Jacobian; no measurable cost on the force evaluation (the deposit is
a move). A lagged steady-state iteration is now essentially the force
evaluation plus ~1 s. Off by default because the dense factors stay resident
(O(NAdof^2) per dense volume). Remaining target: the force evaluation itself.
