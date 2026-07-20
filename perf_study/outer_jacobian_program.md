# Outer force-Jacobian program: banded trust-region step + matrix-free JVP

Implementation of the program proposed in `force_jacobian_investigation-4.pdf`
(evidence base measured 2026-07-13): exploit the measured block-tridiagonal
structure of the outer force Jacobian and its cheap directional action. Two
pieces landed; colored (probing) assembly was evaluated and dropped.

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

## What this enables next

A lagged-banded matrix-free outer iteration: keep the banded factorization of a
stale J as preconditioner/subspace source, refresh the *model* each iterate with
a small batch of fresh JVPs (k directions ~= k * 0.11 s vs 9.7 s per rebuild at
nvol7), rebuild the dense J only when the gate demands. Composes with the
jurasic-lagged-jacobian MR (jac_lag), which gates stale-J reuse but has no
fresh-model correction.
