# Free-boundary iBns refresh timing — the per-evaluation bug and the fix

## TL;DR

The free-boundary optimizer refreshed the plasma normal field `iBns` (via
`update_bnorm`, virtual casing) **before every objective evaluation**,
including the *rejected* trust-region trial steps. Because the Jacobian is
computed once per accepted iteration at a frozen `iBns`, while the residual
surface kept moving underneath it as trial evaluations mutated `iBns`, the
solve **ran away near convergence**. On `ncsx_TEST.toml` the legacy driver
dies at iteration 13 in a 35-rejected-step cascade.

The fix is the classic free-boundary outer/inner loop: **hold `iBns` frozen
during an inner force solve; refresh it once (damped) between solves.** The
new driver `scripts/run/calc_spectre_eq_freeb.py` does this and reaches a
**~2× lower `<f>`** than the legacy driver while running the full iteration
budget without a single trust-region cascade.

## The bug

`spectre/force_minimizer.py` — in free-boundary mode the rank-0 objective
wrapper (`obj_func_mod`) called `update_bnorm(...)` before **every** `f(x)`:

```python
def obj_func_mod(x, obj):
    _send_int_array(x, 2, obj.comm)
    update_bnorm(obj.field_obj, first_free_bound=False, print_info=True)  # ← every eval
    _send_int_array(x, 0, obj.comm)
    return obj_func2(x, obj.field_obj)
```

Two facts make this wrong:

1. The least-squares **callback fires only on accepted iterations**
   (`dogbox.py:264`, `trf.py:368/596`). Trust-region *trial* steps that get
   rejected still call `f` — and therefore still mutated the global `iBns`.
2. With `gBnbld < 1` (e.g. `ncsx_TEST.toml` uses `gBnbld = 0.666`),
   `update_bnorm` actually moves `iBns := gBnbld·iBns + (1-gBnbld)·ofmn`. On a
   bad/rejected trial geometry `ofmn` is garbage, so `iBns` drifts toward
   noise on steps the optimizer *rejected*.

The Jacobian `J = ∂f/∂x` is assembled at a fixed `iBns`; once `iBns` moves
between the Jacobian evaluation and the trial evaluations, the linear model no
longer predicts the cost change. Near convergence — where step sizes shrink
and the `iBns` drift becomes comparable to the real residual reduction — the
trust-region ratio test starts rejecting everything and `Δ` collapses.

### Evidence (`ncsx_TEST.toml`, 30-iteration budget, mpirun -n 3)

Legacy `calc_spectre_eq.py` (`bns_update_mode='per_eval'`), tail of the run:

```
iter    12   nfev     16    cost 3.505e-01   <f> 1.53973e-02   stpnrm 3.10e-02   costred 7.20e-02
iter    13   nfev     51    cost 3.505e-01   <f> 1.83986e-02   stpnrm 0.00e+00   costred 0.00e+00
```

`nfev` jumps 16 → 51 in a single "iteration" — **35 rejected trial steps** —
`step_norm` collapses to 0, `<f>` gets *worse* (1.54e-2 → 1.84e-2), and the
solve terminates at iteration 13. It never reaches the 30-iteration budget.

## The fix — outer/inner loop

`spectre/force_minimizer.py` + `spectre/file_io/input_parameters.py` gained a
**temporary** opt-out, `MinimizationParameters.bns_update_mode`:

- `"per_eval"` (default, legacy) — refresh before every eval. **Wrong**, kept
  only so existing runs are unchanged. *To be deleted* once the new cadence is
  the validated default (see `# TODO(remove)` at both sites).
- `"outer_loop"` / `"frozen"` — do **not** refresh inside `obj_func`. The
  driver controls `iBns` via the new `Force_Minimizer.refresh_bns(damping=...)`
  method (sends signal 2 to the MPI peers, runs `update_bnorm` collectively,
  returns `bnserr`).

`scripts/run/calc_spectre_eq_freeb.py` runs:

```
for outer cycle in budget:
    inner lstsq at FROZEN iBns       # Jacobian always valid
    f(accepted x)                    # set plasma solution at accepted geometry
    refresh_bns(damping=gBnbld)      # the ONLY place iBns moves; damped
    stop if bnserr < tol and inner force solve settled
```

### Timing logic — when to refresh (the bnserr ↔ step_norm ↔ force relation)

The inner solve is run at frozen `iBns`, so its Jacobian and trust region are
always self-consistent. We refresh `iBns` only when continuing to move the
geometry at the *current* `iBns` yields diminishing returns — i.e. when the
remaining residual is set by the `iBns` inconsistency (`bnserr`) rather than by
geometry error. Concretely the inner solve ends (→ refresh) when it
**converges** (`gtol`/`ftol`/`xtol`), **stalls** (`step_norm < step_tol` *and*
`cost_red < cost_red_frac·cost`), or hits the per-cycle `inner_iter_cap`. The
overall budget (`--total-iters`, default 30) spans all cycles. Damping uses the
TOML `gBnbld` (0.5–0.7 is the robust band from the picard study); a guard warns
if `bnserr` stops contracting across cycles.

The data show why mid-solve refresh is wrong: the prototype in-loop trigger
(refresh as soon as `step_norm < step_tol`, *inside* one solve) reached
`<f>=9.6e-3` during its frozen phase but **exploded on the first refresh**
(`<f>` 9.6e-3 → 9.3e-2, 33 rejected trials, trust region collapsed, died at
iter 17) — because refreshing `iBns` invalidated the cached `f`/`J` the
trust-region step was built from. The outer/inner loop avoids this by only
refreshing *between* solves, where the next solve rebuilds `f`/`J` from scratch.

## Benchmark — best `<f>` after a 30-iteration budget

`ncsx_TEST.toml` (nvol=2, Mvol=3, mpol=7, ntor=6), `OMP_NUM_THREADS=3
mpirun -n 3`, all capped at ~30 iterations. The cases do **not** fully
converge, so the figure of merit is the lowest `<f>` (FSA force metric)
reached within budget, and whether the run survives to the budget.

| driver | bns cadence | inner config | best `<f>` | reached | outcome |
| --- | --- | --- | ---: | ---: | --- |
| `calc_spectre_eq.py` (baseline) | per_eval | dogbox / linear / x1.0 | 1.54e-2 | iter 12 | **died iter 13** (35-rej cascade) |
| `calc_spectre_eq_softl1.py` | per_eval | trf / soft_l1 / lsmr | 9.25e-3 | iter 10 | **died iter 11** (34-rej cascade) |
| `calc_spectre_eq_twostage.py` | per_eval | trf soft_l1→linear | 8.05e-3 | iter 19 | both stages ran; each hit a cascade |
| `calc_spectre_eq_modes_then_freal.py` | per_eval | fmodes→freal | 3.77e-2 | iter 11 | script ran end-to-end; fmodes warm-up wasted budget, stage 2 hit a cascade |
| **`calc_spectre_eq_freeb.py` (outer)** | **outer_loop** | dogbox / linear / x1.0 | **7.82e-3** | iter 29 | **full 30 iters, no cascade** ✅ |
| `calc_spectre_eq_freeb.py` (inloop) | outer_loop | dogbox / linear / x1.0 | 9.64e-3 | iter 16 | died iter 17 (first mid-solve refresh) |

Observations:

- **Every legacy `per_eval` driver dies in a trust-region cascade** at
  `step_norm = 0`, regardless of loss (`linear`/`soft_l1`) or scaling
  (`1.0`/`jac`). The bns timing — not the optimizer knobs — is the killer.
- The **outer/inner loop is the only run that survives the full budget**, and
  it reaches the lowest `<f>` (7.82e-3), with `bnserr` contracting
  monotonically across cycles (5.3e-3 → 8.6e-4 at `gBnbld=0.666`).
- `twostage` limps close (8.05e-3) only because its two separate solves act as
  accidental restarts that dodge each cascade — fragile, not by design.
- The in-loop trigger confirms that the *frozen* phase is healthy; the damage
  is entirely from refreshing mid-solve.
- **Both multi-stage scripts run end-to-end** (answering the open question in
  `optimizer_overhaul_plan.md` / `free_boundary_optimization_notes.md` that
  "Stage 2 was never cleanly tested"): `twostage`'s nested two-`lstsq` and
  `modes_then_freal`'s two-`Force_Minimizer`-instance handoff both complete
  without MPI deadlock. But `modes_then_freal` *underperforms* on this small,
  already-well-conditioned case — the `fmodes` warm-up consumes half the
  budget without setting up a better `freal` start, and stage 2 still hits the
  per-eval cascade. The fmodes warm-up is for cold starts with outlier
  residuals, not cases like this.

**Default chosen: `--bns-trigger outer`.**

## Tuning sweep (`ncsx_TEST`, 30-iteration budget)

Holding `--bns-trigger outer` fixed, each knob was swept one at a time (one
`mpirun -n 3` job at a time). **All variants ran the full 30 iterations with no
trust-region cascade** — the outer-loop fix is robust across the whole knob
space; these differences are second-order optimization-rate effects, not
stability.

| variant (vs default) | best `<f>` @30 |
| --- | ---: |
| **`--method trf`** (else default) | **6.16e-3** ✅ |
| `--method trf --inner-iter-cap 10` | 7.01e-3 |
| `--inner-iter-cap 10` (dogbox) | 7.04e-3 |
| **default** (dogbox, cap 5, linear, x_scale 1.0, gBnbld 0.666) | 7.82e-3 |
| `--gbnbld 0.5` | 7.97e-3 |
| `--x-scale jac` | 8.13e-3 |
| `--gbnbld 0.85` | 8.62e-3 |
| `--inner-iter-cap 3` | 1.00e-2 |
| `--loss soft_l1 --f-scale 5` | 1.97e-2 |

Conclusions:

- **`method=trf` is the clear winner** (6.16e-3 vs dogbox 7.82e-3) — trust-region
  reflective converges the inner force solve faster than dogbox here. Made the
  new default.
- **`linear` loss ≫ `soft_l1`** (7.82e-3 vs 1.97e-2). Once the per-eval-`iBns`
  bug is gone the residuals are *not* outlier-dominated, so `soft_l1`'s
  row-clamping just discards good gradient information. (This also retro-justifies
  the doubt cast on the old reports, which leaned on `soft_l1` to tame an
  "outlier" that was largely the moving-`iBns` artifact.)
- **`x_scale=1.0` > `jac`** (7.82e-3 vs 8.13e-3) — consistent with the cold-start
  `jac`-clamping analysis in `free_boundary_optimization_notes.md`.
- **`gBnbld ≈ 0.666` is the damping sweet spot** (0.5→7.97e-3, 0.666→7.82e-3,
  0.85→8.62e-3).
- **Refresh frequency interacts with the method.** Under dogbox, *fewer*
  refreshes help (cap10 7.04e-3 < cap5 7.82e-3 < cap3 1.00e-2). Under trf, the
  default cap5 is best (trf+cap10 7.01e-3 > trf+cap5 6.16e-3). So the levers are
  not separable — `trf` with the default `cap 5` is the best combination found.

**New recommended defaults: `--method trf`, `--loss linear`, `--x-scale 1.0`,
`--gbnbld` from TOML (0.666), `--inner-iter-cap 5`.** The script's `--method`
default is updated to `trf` accordingly (dogbox remains selectable).

### Vacuum-`Lrad` field-error side-experiment — more resolution does NOT help

Does doubling the vacuum radial resolution (`Lrad = [11,11,44]` vs the
`[11,11,22]` baseline) tame the vacuum FIELD ERROR? Tested two ways:

1. A full free-boundary run at `Lrad=44` reaches the **same `<f>`** (6.14e-3 vs
   the `Lrad=22` trf 6.16e-3) — as expected, vacuum radial resolution does not
   affect the interface force balance.
2. A **controlled comparison** (the *same* `Lrad=22`-optimized geometry, solved
   at `Lrad=22` vs `Lrad=44`; `scripts/diagnostic/freeb_postprocess_fieldstate.py`)
   shows the vacuum FIELD ERROR does **not** improve — it slightly **worsens**:

   | vacuum (vol 2) | `Lrad=22` | `Lrad=44` |
   | --- | --- | --- |
   | avg (s/θ/φ) | [242, 109, 79] | [305, 144, 24] |
   | max | 3.26e10 | 9.53e10 |

**Conclusion: the vacuum FIELD ERROR is not an under-resolution problem.**
Adding radial points doesn't reduce it (the `max` even grows ~3×), which means
the error is driven by a **coordinate/metric pathology in the warped vacuum
region** — a near-degenerate metric, not too few basis functions. Raising
`Lrad` is therefore *not* a mitigation; the fix lies upstream in the vacuum
coordinate construction (the warped `fix2free` instantiation, `ncsx_BAD.toml`),
or in a different radial coordinate for the vacuum gap. This reinforces the
warped-instantiation diagnosis over the "spectral basis too small" hypothesis.

## Other improvements to consider

- **Delete the `per_eval` path.** Once the outer/inner loop is the validated
  default (it already wins decisively here), remove `bns_update_mode` and the
  per-eval branch entirely — they exist only for backward compatibility. See
  the `# TODO(remove)` markers.
- **The final FIELD ERROR printout (`~10¹⁰`) is faithful, not a state bug, and
  is mostly a coordinate-singularity artifact.** This is a *FIELD ERROR* (the
  pointwise Beltrami residual `∇×B − μB` on the quadrature grid), reported
  per **volume**, *not* a force on an interface. Investigated with two
  diagnostics (`scripts/diagnostic/freeb_postprocess_fieldstate.py`,
  `..._fielderr_trajectory.py`):
  - **Not a `run()` / stale-f90wrap-pointer / postprocess artifact.** Loading
    the optimized geometry fresh and measuring the field error four ways —
    before `run()`, after `run()`, after a re-solve, after clearing the
    f90wrap pointer caches — gives **byte-identical** results (vacuum max
    3.263e10 every time). The number faithfully reflects the optimized
    geometry.
  - **It's the `max`, dominated by isolated coordinate-singularity points; the
    `avg` is fine.** Across the optimizer trajectory the plasma volumes' *avg*
    field error stays flat and small (vol0 ≈ 0.25, vol1 ≈ 10) while their *max*
    grows monotonically (vol0 2e2→4.5e6, vol1 52→6e8) as the geometry deforms —
    the `s→0` magnetic-axis singularity (`√g→0`) sharpening. **`force_rms`
    falling while `max` field error rises is consistent**: the force is an
    interface *surface integral* of the B²-jump and is blind to the pointwise
    curl residual at isolated singular points.
  - **The vacuum volume is the genuinely degraded one** — `avg ~100–1000s`,
    `max ~1e8–1e13`, and noisy, with spikes lining up with the iBns-refresh
    boundaries. This is the real open issue (Issue #5 in
    `free_boundary_optimization_notes.md`: vacuum spectral basis inadequate /
    `Lrad` too low). Worth chasing: bump vacuum `Lrad`, and report the field
    error `avg` (not `max`) in `print_all_errors` since the `max` is dominated
    by cosmetic axis singularities. Ties to the warped `fix2free` instantiation
    (`ncsx_BAD.toml`) theme below.
- **Guard against warped instantiation up front.** The `fix2free` starting
  geometry for these cases (`ncsx_BAD.toml`) is not a clean equilibrium guess.
  A warped vacuum region is the most likely driver of both the vacuum FIELD
  ERROR above and the "bonkers initial force" the earlier reports
  (`free_boundary_optimization_notes.md`, `ibns_picard_investigation.md`)
  attributed to current sheets / H2–H6. A cheap up-front sanity check on the
  initial interface/vacuum geometry would catch it before a multi-hour debug
  session. **This reframes those reports' "physics, not numerics" conclusion as
  unproven** — see their revised banners.
- **Smooth the post-refresh sawtooth.** After each refresh the cost jumps up
  (the geometry is no longer optimal for the new `iBns`). A larger `gBnbld`
  (smaller `iBns` step) or an adaptive damping / Anderson acceleration on the
  `iBns` map (picard study, open Q4) would reduce the sawtooth and may let `<f>`
  fall further within budget.
- **Stronger inner-convergence detection.** The outer loop currently relies on
  `inner_iter_cap` because the inner solve rarely hits `gtol`/`ftol` on these
  cases. A per-interface force floor (stop the inner solve when the
  geometry-reducible residual is below the `bnserr`-induced floor) would spend
  iterations more efficiently.
- **Continuation in `pscale`/`phiedge`** for cold starts (picard study open
  question) — keep residuals balanced throughout instead of starting from a
  bonkers state.
