# Free-boundary equilibrium optimization — debug session notes

## Introduction

SPECTRE finds MRxMHD equilibria by minimizing the force-balance residual
`[[B²/2 + p]]` across each interface in a stepped-pressure decomposition.
For free-boundary problems the residual is computed on a real-space
`(nt × nz)` grid per interface and the optimizer drives the Fourier
coefficients of the interface geometries until the forces are zero (or
small enough to be at the discretization noise floor).

The Python wrapper assembles this as a nonlinear least-squares problem:

- **Residuals `f(x)`**: `(mvol-1) × n_ntz_filtered` real-space force values,
  computed by `force_real` (calls Fortran `solve_field` + `calc_b2mag_jump`).
- **DOFs `x`**: Fourier coefficients of all internal interfaces (the plasma
  boundary is a DOF in free-boundary mode; the computational/coil boundary
  is fixed).
- **Jacobian `J = ∂f/∂x`**: analytic, computed by Fortran
  `solve_field_deriv` + `calc_b2mag_jump_deriv`.
- **Solver**: scipy's `least_squares` family (`trf` or `dogbox`), accessed
  through a thin wrapper in `spectre/least_squares/`.

This session debugged **persistent slow / stalled convergence** of the
free-boundary optimizer, which manifested as:

- The optimizer making real progress on inner interfaces but never moving
  the plasma boundary.
- Repeated intersection rejections at iter 0–1.
- Cost stalling several orders of magnitude above zero.
- Inner interfaces being pushed outward until they collided with the
  boundary.

The eventual root cause was physics, not numerics — but the journey
through the numerics exposed several real issues, fixed one bug, produced
three sibling driver scripts, and clarified what each optimizer knob
actually does. Those findings live below.

---

## Background — how the optimizer is wired

```
Force_Minimizer  ──►  scipy least_squares (custom-vendored)
        │                       │
        │                       ├── method='trf'        TRF (Branch 1999)
        │                       ├── method='dogbox'     Powell dogleg
        │                       └── method='lm'         MINPACK Levenberg-Marquardt
        │
        ├── obj_func(x)         calls update_bnorm, then force_real
        │                       evaluates `[[B²/2 + p]]` on the grid
        │
        └── jac_func(x)         calls solve_field_deriv + calc_b2mag_jump_deriv
                                analytic derivatives, propagated through
                                the Beltrami solve via `dBdX`
```

Free-boundary specifics:

- `update_bnorm` is called by `obj_func` before each `f` evaluation. It
  computes the plasma-induced normal field on the computational boundary
  via virtual casing and updates `iBns` via a relaxation blend
  `iBns ← gBnbld · iBns + (1−gBnbld) · ofmn`.
- With `gBnbld = 1` (the typical setting in this project), `iBns` is
  frozen at its input value — the plasma response only enters the cost
  through `bnserr`, not through self-consistent BC updates.
- In dogbox/TRF, `f` is always evaluated immediately before `J` at the
  same `x`, so the global Fortran state is consistent between them.

Initial optimizer settings in `MinimizationParameters` (Pydantic schema):

| param | default | notes |
|---|---|---|
| `method` | `'dogbox'` | TRF is more robust for rank-deficient J |
| `rcond_lstsq` | `None` | dogbox-only; SVD truncation in `np.linalg.lstsq` |
| `xtol` | `1e-22` | below machine ε; effectively disables xtol termination |
| `gtol` | `1e-13` | |
| `ftol` | `1e-13` | |

---

## Issues observed

### 1. Linear loss + huge outlier residual → wild Newton step

Initial state typically had `f_iface_5 ~ 10⁹` while `f_iface_0..4 ~ 0.1`,
a ratio of ~10¹⁰. With linear loss the Gauss–Newton step is dominated by
the outlier row — the proposed step is huge, the optimizer hits the
intersection cliff repeatedly at iter 0 (10+ rejected trial steps), and
the TR collapses.

### 2. `x_scale='jac'` over-restricts the outlier direction

`jac` scaling sets `x_scale_i = 1 / col_norm(J)_i`. With one residual row
10¹⁰× larger than the rest, the columns affecting it inherit that scale:
`col_norm(iface_5) ≈ 3.8 × 10¹⁷` vs `col_norm(iface_0) ≈ 7 × 10⁵`. The TR
constraint then allows `dx_iface_5 ≲ Δ / 3.8e+17 ~ 10⁻¹⁸` per step — the
plasma boundary cannot meaningfully move regardless of how big the
gradient is in that direction (`|J^T f|` at iface 5 was `1.7 × 10²⁸`).

This is `'jac'` doing exactly what its math says, but its assumption
("column norm reflects physical sensitivity per unit displacement") is
violated. In cold start the column norm reflects `(sensitivity) × (residual
magnitude)`, and the residual magnitude is what we're trying to reduce.

### 3. Intersection cliff collapses the trust region

`force_real` returns `1e10` per sample point when interfaces intersect
(`spectre/force_targets.py:115`). With ~14k residual points this gives a
discontinuous cost spike to `~1.4 × 10²⁴`. The TR ratio test treats this
identically to "model is bad" and reduces `Δ ← 0.25 · step_norm`. After 10
such rejections `Δ` is down by a factor `4¹⁰ ≈ 10⁶`, and the regularize
term `∝ 1/Δ²` blows up by another factor of 10¹², crushing the step.

### 4. The optimizer's "stupid direction"

When the iface-5 column was effectively unusable (whether from `jac`
clamping or from soft_l1 row-zeroing), the optimizer's best alternative
was to reduce the iface-5 residual by perturbing iface-4 or iface-3,
which affect vol-5's Beltrami field and thus iface-5's residual indirectly.
This pushed inner interfaces outward until they collided with iface 5 —
hence the "interfaces moving in one by one until they touch" pattern.

### 5. Vacuum Beltrami non-convergence

For initial conditions far from equilibrium, the vol-7 (vacuum) Beltrami
solve produced `ihybrj = 2` ("consider restart") with F-residuals of
O(1)–O(10), and `calc_beltrami_error` reported FIELD ERROR averages of
`10⁴ – 10⁵` in that volume. Despite this, `ImagneticOK(7)` stayed `.true.`
because that flag only checks LAPACK success (`idgetrf`), not the hybrj
Newton-iteration convergence. The Jacobian assembly proceeded with what
was effectively garbage derivatives in the vacuum-side columns.

### 6. Repeated convergence to the same local "minimum"

Multiple runs (varying `Lconstraint`, `Lrad[vacuum]`, `x_scale`, `loss`)
all stalled at cost values that were unreachable by the optimizer — not
because the algorithm was failing but because **the equilibrium being
sought did not exist** at the requested interface placement. See finding
in [Root cause](#root-cause-current-sheets-at-interfaces).

---

## Experiments tried

Three sibling driver scripts were created in `scripts/run/`, each passing
a custom optimizer through `Force_Minimizer(..., optimizer=...)` without
editing the library:

### Experiment A — `calc_spectre_eq_experimentA.py`

The "standard" Moré preconditioner stack:

```python
method='trf'
tr_solver='lsmr'
tr_options={'regularize': True, 'damp': 1e-3}  # Byrd damping + Tikhonov floor
x_scale='jac'
loss='linear'
```

**Result**: Cost dropped 1.002× over 17 iterations on the original
unhealthy initial condition. The plasma boundary moved by 26 % (iface-5
rms from `5.9e+9` to `4.4e+9`). Interior interfaces ballooned by 10⁴–10⁷×
(`iface 4 rms 1.4 → 4142`) because the optimizer was trying to reduce
iface-5 residual by perturbing neighbors. Intersection rejections at iter
1 and iter 15.

### Experiment R1 — `calc_spectre_eq_softl1.py`

Experiment A plus a robust loss:

```python
loss='soft_l1'
f_scale=10.0          # CLI-tunable
```

**Result**: Cost dropped 11× (raw equivalent ~4×10¹⁹) in 19 iterations on
the same setup. Iface-5 rms dropped 5–6 orders of magnitude. Interior
interfaces stayed quiet. Intersection rejections dropped to 5 at iter 0,
none thereafter. Vol-6 (vacuum) FIELD ERROR avg dropped 280× alongside.

**Mechanism**: soft_l1's gradient cap (`rho'(z) → 0` for `|f| ≫ f_scale`)
prevents the outlier row from monopolizing the GN step direction. The
optimizer takes small but well-spread steps. The cliff is mostly *avoided*
rather than *softened*; the height of the cliff in cost units is only
~10¹¹ instead of 10²⁰, but the TR still collapses if it's hit — soft_l1's
main contribution is avoiding hits.

### Experiment two-stage — `calc_spectre_eq_twostage.py`

Stage 1 with uniform `x_scale` and robust loss, Stage 2 with `'jac'` and
linear loss:

```python
# Stage 1 (cold start)
x_scale=1.0; loss='soft_l1'; f_scale=10
tr_options={'regularize': True, 'damp': 1e-3}; max_niter=30

# Stage 2 (polish)
x_scale='jac'; loss='linear'
tr_options={'regularize': True, 'damp': 1e-3}
```

**Result**: When applied to a healthy initial condition, Stage 1 made
rapid initial progress (cost dropped 1500–8000×). Stage 2 was never
cleanly tested because all subsequent runs ended up tripping the actual
underlying issue (next section).

### Diagnostic snippet — interactive J-column-norms probe

A pasteable `python -i` snippet was developed to dump `J(x_init)` and
report per-interface column norms, gradient projections, and SVD condition
numbers of each iface's column block. Confirmed in one case:

| iface | block_norm | top SV | bottom SV | cond |
|---|---:|---:|---:|---:|
| 0 | 6.9e+5 | 1.0e+6 | 70 | 1.5e+4 |
| 5 | **3.8e+17** | 3.8e+17 | 2.4e+7 | 1.6e+10 |

The plasma-boundary block is well-conditioned (cond `~10¹⁰`, not at noise
floor `~10¹⁵⁺`) — its column entries are 10¹¹× larger than interior's
because the residual it derives is 10¹⁰× larger. **Not** rank-deficient,
**not** corrupted vacuum derivatives. Just enormous.

### Bug fix — `force_real` shape inconsistency

While testing `method='lm'`, MINPACK raised `ValueError: The array
returned by a function changed size between calls`. Root cause: the
overlap-penalty return path in `force_real` returned `(mvol-1, ntz)`
shape, while the normal path applied `filter_stellsym_pts` and returned
`(mvol-1, nt*(nz//2+1))` — different sizes. `dogbox`/`trf` tolerated
this; `_lmder` checked and rejected.

Fix in `spectre/force_targets.py` (added 4 lines): apply the same
stellsym filter to the overlap-penalty array. Source edit was appropriate
here (correctness, not experiment).

---

## Root cause — current sheets at interfaces

After all of the above, the user identified the actual cause: **the
problematic interface had been placed at a location where a current sheet
forms**, making the force-balance equation `[[B²/2 + p]] = 0` impossible
to satisfy across that interface with smooth fields.

**Twice** in this session:

1. First case: iface-5 placement at an unidentified rational surface.
2. Second case: 3/4 resonance at the outer interface — iota at the plasma
   boundary was 0.7242, well within the perturbation halo of 3/4 = 0.75.

All the symptoms were downstream consequences of this:

- The huge iface-5 residual was the algorithm *correctly* reporting the
  current-sheet discontinuity.
- The vacuum Beltrami's poor convergence was driven by an unrealizable
  plasma BC from that discontinuity.
- The optimizer's "stupid direction" was its rational best alternative
  given an unsolvable row.
- `bnserr` stalling at ~10² was measuring the irreducible mismatch.

**No optimizer setting can fix this.** The cure is to revise the input
deck so interfaces avoid resonant locations.

### Physics filter for resonance checks

For a stellarator-symmetric `nfp`-periodic configuration, **only
resonances `p/q` (in lowest terms) where `p` is a multiple of `nfp`** can
exist in the constrained physics. For this project's nfp=3 runs:

- **Live hazards**: 3/q (q coprime to 3), 6/q (q coprime to 6), 9/q
  (q coprime to 9), …
- **Ruled out**: 5/7, 2/3, 4/5, 7/10, etc. — `p` not a multiple of 3.

In the typical iota band 0.69 – 0.75:
- 3/4 = 0.75 (dominant)
- 9/13 ≈ 0.692
- 12/17 ≈ 0.706

---

## Key technical insights

### `x_scale='jac'` is a category error in cold start

`'jac'` interprets `col_norm(J)_i` as "physical sensitivity per unit
displacement in `x_i`". In cold start with an outlier residual,
`col_norm` instead measures `(sensitivity) × (residual magnitude)` — the
second factor being what we want to drive to zero. Clamping the trust
region by `1/col_norm` then clamps the very directions we need to move.

Once residuals are balanced, the second factor becomes uniform and `'jac'`
correctly reflects per-mode sensitivity again. The signal that the regime
has switched: `||J||_F²` (printed as `fjac val`) plateaus instead of
continuing to drop.

### `soft_l1` is a *clamp*, not a smooth downweight

scipy implements robust loss via row scaling
`J_eff[i,:] = J[i,:] · sqrt(J_scale_i)` where
`J_scale_i = rho'(z_i) + 2·rho''(z_i)·f_i²`. For `|f|/C ≳ 1`, this
expression is **negative** and gets clamped to `EPS ≈ 10⁻¹⁶`. The effective
row is then `J[i,:] · 1.5e-8` — essentially zeroed out of the Gauss–Newton
system.

So `soft_l1` doesn't reduce outlier influence smoothly — it makes outlier
rows invisible to the GN direction. The size of the cliff in cost units
still exists; the optimizer just doesn't *use* outlier rows when picking
where to step. This explains why smaller `f_scale` makes things worse:
it shrinks the "trustworthy" region and zeros out more rows.

### `ImagneticOK` only catches LAPACK failures

The Jacobian assembly's guard at `field_mod.F90:1026` requires
`ImagneticOK(vvol)` and `ImagneticOK(vvol+1)` to be `.true.`. But that
flag is only set `.false.` when LAPACK's `DGETRF`/`DGETRS`/`DGERFS`/
`DGECON` return non-zero (`beltrami_solver_mod.F90:240`). hybrj
non-convergence (`ihybrj = 2`, "consider restart") leaves it `.true.`.

So derivatives computed from an unconverged Beltrami solution pass the
guard and feed into the optimizer's Jacobian. The case where this matters
most is the vacuum volume during the initial badly-out-of-equilibrium
state. A *second* flag `ImagneticConverged` based on the F-residual would
make this guard meaningful — but in light of the current-sheet diagnosis,
this is lower-priority than detecting the upstream cause.

### TRF + LSMR + regularize is the right substrate

The textbook nonlinear-LSQ stack — TRF for global convergence + LSMR for
the LM-style damped Newton + Byrd's adaptive `1/Δ²` regularizer — works
well *when the problem is healthy*. The damp parameter is an additional
Tikhonov floor; `damp=1e-3` is conservative and mostly inactive for this
problem. Switching to dogbox or LM loses the composability with `loss` /
`x_scale` and we've seen no benefit.

The two-stage `1.0 → 'jac'`, `soft_l1 → linear` recipe is the right
generalization for cold starts with outliers. It's implemented in
`calc_spectre_eq_twostage.py`.

---

## Bug fixes applied

- **`spectre/force_targets.py`**: `force_real`'s overlap-penalty return
  path now applies the same `filter_stellsym_pts` as the normal path, so
  the array size is consistent between intersecting and non-intersecting
  states. Without this, `method='lm'` raises `ValueError` and `dogbox`/`trf`
  silently mishandle pre-allocated Jacobian shapes.

---

## Scripts added

In `scripts/run/`:

- **`calc_spectre_eq_experimentA.py`** — TRF + LSMR + regularize + jac +
  linear loss. The "standard" textbook configuration. Reference baseline.
- **`calc_spectre_eq_softl1.py`** — experiment-A + `loss='soft_l1'`. CLI
  flag `--f-scale`. Robust-loss variant.
- **`calc_spectre_eq_twostage.py`** — Stage 1 (`x_scale=1.0`,
  `soft_l1`) → Stage 2 (`x_scale='jac'`, `linear`). CLI flags
  `--stage1-iters`, `--stage1-f-scale`, `--stage1-damp`, `--stage2-damp`,
  `--skip-stage2`. The recommended driver for cold-start problems.

All three pass a custom optimizer through `Force_Minimizer(...,
optimizer=...)` and require no library edits.

---

## Next steps and open questions

### High priority — diagnostic to detect current sheets early

A short routine in `Force_Minimizer.__init__`, after the initial
`print_all_errors`, that:

1. Computes `max(f_rms_per_iface) / median(f_rms_per_iface)`. Flag a
   likely current sheet if this exceeds ~10⁴.
2. Scans the input `iota[1:Nvol+1]` array for proximity to low-order
   rationals `p/q` with `p ∈ multiples of nfp`. Flag any closer than
   ~0.02 to such a value.
3. Reports per-volume FIELD ERROR averages and flags any volume an
   order of magnitude above its neighbors.

Output is a loud warning (not a fatal). The user can override and run
anyway, but the diagnosis is in the log from the first second of the
run instead of after a multi-hour debug session. Estimated effort: ~30
lines of Python in `force_minimizer.py`.

### Medium priority — `ImagneticConverged` second flag

Add a `logical, public :: ImagneticConverged(:)` to `allglobal_mod`,
set it in the `case (2)`/`case (4:5)` arms of
`beltrami_field_mod.F90:195-205` based on `norm(Fdof) < tolerance`, and
augment the assembly guard at `field_mod.F90:1026` to require both
flags. When the new flag is false, zero out that volume's row/column
contributions in `force_real_jac` rather than fatal-aborting. Estimated
effort: ~20 lines of Fortran, plus a regression test.

Lower priority than the diagnostic above because the current-sheet
detection catches the upstream cause; this would only matter for
mid-iteration vacuum-convergence failures during otherwise-healthy runs.

### Medium priority — R2 intersection backtracking

When `obj_func` is about to evaluate `f` at a trial step that would
cause intersection, geometrically bisect the step until non-intersecting,
then evaluate the cost at the bisected step. This decouples
"step too long geometrically" from "linear model is wrong" in the TR
ratio test, so cliff hits don't collapse `Δ`.

Implementable as a Python wrapper around `obj.obj_func` in a custom
optimizer — no library edits. The geometry check already exists in
`spectre/isect_tester.py`; we just need to bisect instead of return a
penalty.

Estimated effort: ~50 lines in a new sibling script
(`calc_spectre_eq_r2.py`). Most useful when applied on top of the
two-stage driver.

### Low priority — smooth barrier on inter-surface distance

Add a soft penalty `1/(d_min - ε)` or `exp(-d_min/τ)` for the minimum
inter-surface distance, with analytic gradient. Replaces the
hard intersection cliff with a smooth gradient that pushes surfaces
apart preventively. Larger engineering investment (need analytic
derivative of `d_min`), but principled.

### Open questions

- **The vacuum Beltrami's FIELD ERROR.** Even at moderate-distance
  initial conditions the vol-7 FIELD ERROR `avg` was ~10² – 10⁵, while
  hybrj's F-residual was ~10⁻⁵. The two quantities measure different
  things — the F-residual is the constraint equation, the FIELD ERROR
  is grid-based curl/div residual. **Question**: is the FIELD ERROR
  level reachable for the vacuum gap with `Lrad ~ 13`, or is the
  spectral basis structurally inadequate? If structurally inadequate,
  what's the minimum Lrad for a clean vacuum representation in this
  geometry?
- **Per-mode scaling for `x_scale`.** A physically-motivated `x_scale`
  array (e.g. `1 / (m + 1)²` to reflect typical Fourier amplitude
  falloff) might outperform both `1.0` and `'jac'` in cold start. Not
  tested. **Question**: what's the best lightweight per-mode prior
  that doesn't depend on the current residual magnitude?
- **Continuation in `pscale`/`phiedge`.** Standard MHD-equilibrium
  technique: solve a scaled-down problem (e.g. 1 % pressure), warm-start
  the next stage with `(α + Δα) · pressure`, repeat. Not implemented.
  **Question**: would this dodge the cold-start outlier problem
  altogether by keeping the residuals balanced throughout?
- **Quantitative resonance-proximity threshold.** "Within 0.02 of a
  low-order rational" was the rough threshold used here, but the
  perturbation halo's width depends on the local shear, pressure jump,
  and `nfp`. **Question**: what's the right formula for a per-interface
  resonance-risk score?
