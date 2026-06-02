# Optimizer overhaul — consolidation target

Working notes for the `optimizer_overhaul` branch. The branch carries a finished library
core (committed) plus exploratory drivers and bns diagnostics (WIP commits). This file
records the target the exploration is converging toward, so the eventual PR is a fill-in
rather than a redesign.

## Goal

Reduce `scripts/run/` to **two** canonical equilibrium drivers, replacing the four
experimental variants:

| Replace | With |
| --- | --- |
| `calc_spectre_eq.py` (vanilla, TOML-only) | **`calc_spectre_eq.py` — one-stage** |
| `calc_spectre_eq_experimentA.py` | (folds into one-stage flags) |
| `calc_spectre_eq_softl1.py` | (folds into one-stage flags) |
| `calc_spectre_eq_twostage.py` | **`calc_spectre_eq_staged.py` — multi-stage** |
| `calc_spectre_eq_modes_then_freal.py` | (folds into multi-stage as a `fmodes` stage) |

### `calc_spectre_eq.py` — one-stage
Single `lstsq` call. Defaults reproduce current TOML-driven behaviour; proven knobs
exposed as CLI flags so a one-stage solve is fully specifiable without editing the TOML:

- `--method {trf,dogbox}` (default from `min_params.method`)
- `--x-scale {1.0,jac}` (default `1.0`)
- `--loss {linear,soft_l1}` (default `linear`) and `--f-scale` (soft_l1 transition scale)
- `--tr-solver {exact,lsmr}` (default `exact`)
- `--damp` (LSMR Tikhonov damp, with `tr_options={'regularize': True, 'damp': ...}`)

### `calc_spectre_eq_staged.py` — multi-stage
A list-of-stages optimizer: each stage is one `lstsq` call with its own
`{max_niter, method, x_scale, loss, f_scale, tr_solver, damp, objective}`. Behaviour to
preserve from the experimental drivers:

- carry `iteration_start` forward between stages (avoid `xs_fixb/` snapshot-name
  collisions);
- abort the whole chain on SIGUSR1 (`result.status == 5`) and return early;
- support an `objective` switch per stage (e.g. a `fmodes` warm-up stage with
  `min_params_overrides={'objective': 'fmodes', 'exp_fact': ...}` followed by a `freal`
  polish stage), via either nested `lstsq` calls in one `Force_Minimizer` (twostage style)
  or sequential `Force_Minimizer` instances (modes_then_freal style — note the MPI-safe
  forward solve with `field_obj.force_real()`, not `obj_func`, before the second
  `minimize()`).

Open design choice: one `Force_Minimizer` with nested `lstsq` (simpler, no dir wiping) vs
two instances (allows different objectives/meshes per stage). The nested approach is
preferred unless per-stage objective/mesh changes force separate instances.

## Shared pattern (documented, not yet extracted)

Every variant is the same shape — an optimizer callable injected via
`Force_Minimizer(inputfile, optimizer=...)`:

```python
def optimizer(obj, x_init, f, jac):
    p = obj.min_params
    return lstsq(
        f, x_init, jac=jac,
        method=..., verbose=2, args=[obj],
        gtol=p.gtol, xtol=p.xtol, ftol=p.ftol,
        max_nfev=p.max_nfev, max_niter=...,
        iteration_start=obj.iteration_start,
        rcond_lstsq=p.rcond_lstsq,
        x_scale=..., tr_solver=..., tr_options={...},
        loss=..., f_scale=...,
        cb=obj.callback,
    )
```

A shared `_staged_optimizer` factory may be extracted later; for now the two scripts can
carry their own small optimizer functions. Do not extract prematurely while the knob
defaults are still being tuned.

### Knob matrix from the experimental drivers

| driver | stages | x_scale | loss / f_scale | tr_solver / damp | objective |
| --- | --- | --- | --- | --- | --- |
| experimentA | 1 | jac | linear | lsmr / 1e-4 | freal |
| softl1 | 1 | 1.0 | soft_l1 / `--f-scale` 10 | lsmr / 1e-3 | freal |
| twostage | 2 | 1.0 → jac | soft_l1 → linear | lsmr / 1e-3 | freal |
| modes_then_freal | 2 (1a/1b + 2) | 1.0/jac → jac | soft_l1 / 5 → linear | lsmr / 1e-3 | fmodes → freal |

The right multi-stage defaults are still being determined — that is the point of the
exploratory phase.

## bns diagnosis thread

The free-boundary normal-field (bns) update is the other open item. See
`ibns_picard_investigation.md` and `free_boundary_optimization_notes.md`. Key questions:

- Is the single-shot `SPECTRE.update_bns()` (one `update_bnorm` refresh + dtflux Newton +
  tflux sync) self-consistent at geometries where virtual casing struggles
  (`relvcerr ~ 1` in the `bnorml :` log lines)? The alternating Picard map is not a
  contraction there — see `scripts/diagnostic/picard_*` and `vcasing_convergence_*`.
- Does the per-evaluation `update_bnorm()` inside the optimizer loop interact badly with
  the Jacobian? Current understanding: `f` is evaluated immediately before `jac` at the
  same `x`, and `gbnbld=1` freezes iBns — so do **not** "mirror update_bnorm into jac" as a
  fix. A force imbalance where one interface has a ~1e6× larger residual is more likely a
  current sheet (physics) than a numerics bug — check that first.
- Reproduce from `examples/utils/make_freeboundary/freeb_ncsx.toml` /
  `freeb_ncsx_vac.toml` (the pressure-issue inputs).
