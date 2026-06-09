<!--
SPDX-FileCopyrightText: 2026 Proxima Fusion

SPDX-License-Identifier: MIT
-->

# Free-boundary iBns timing, the trf driver, and `s.inspect`

## The bug: refreshing iBns every evaluation

In free-boundary mode the optimizer refreshed the plasma normal field `iBns`
(via `update_bnorm`, the virtual-casing principle) **before every objective
evaluation** — including the *rejected* trust-region trial steps. That is the
wrong cadence for two reasons:

1. The least-squares callback fires only on **accepted** iterations, but
   rejected trial steps still call `f(x)` and therefore still mutated the global
   `iBns`.
2. With `gBnbld < 1`, `update_bnorm` moves `iBns := gBnbld·iBns + (1-gBnbld)·ofmn`.
   On a rejected, bad-geometry trial step `ofmn` is garbage, so `iBns` drifted
   toward noise on steps the optimizer had *rejected*.

The Jacobian `J = ∂f/∂x` is assembled once per accepted iteration at a fixed
`iBns`. Once `iBns` moves between the Jacobian evaluation and the trial
evaluations, the linear model no longer predicts the cost change. Near
convergence — where steps shrink and the `iBns` drift becomes comparable to the
real residual reduction — the trust-region ratio test starts rejecting every
step and the trust radius collapses into a cascade.

## The fix: frozen iBns + outer/inner loop

The classic free-boundary outer/inner loop: **hold `iBns` frozen during an inner
force solve; refresh it once (damped) between solves.**

- `Force_Minimizer` keeps `iBns` frozen during the inner solve. Each objective
  evaluation only signals a force eval — it never refreshes `iBns`.
- `Force_Minimizer.refresh_bns(damping=...)` performs the refresh collectively:
  rank 0 sends MPI signal 2, all ranks run `update_bnorm` in lockstep, and it
  returns `bnserr` (the `iBns`/`ofmn` mismatch).
- `scripts/run/calc_spectre_eq.py` is the driver:

  ```
  for outer cycle in budget:
      inner lstsq at FROZEN iBns       # Jacobian always valid
      f(accepted x)                    # plasma solution at the accepted geometry
      refresh_bns(damping=gBnbld)      # the ONLY place iBns moves; damped
      stop if bnserr < tol and the inner solve settled
  ```

The inner solve runs at frozen `iBns`, so its Jacobian and trust region are
self-consistent throughout. It ends — triggering a refresh — when it
**converges** (`gtol`/`ftol`/`xtol`), **stalls** (`step_norm < step_tol` and
`cost_red < cost_red_frac·cost`), or hits the per-cycle `--inner-iter-cap`. The
overall budget spans all cycles (`--total-iters`, default 30). Damping of
`0.5–0.7` (`--gbnbld`, default from the TOML `gBnbld`) is robust.

A prototype in-loop trigger (`--bns-trigger inloop`) refreshes mid-solve when
`step_norm` drops below a threshold; it is kept for comparison only. Refreshing
mid-solve invalidates the cached `f`/`J` the trust-region step was built from,
so the outer trigger — which refreshes only *between* solves, where the next
solve rebuilds `f`/`J` from scratch — is the default.

## Why `trf` is the default method

`--method` defaults to `trf` (trust-region reflective). On these free-boundary
cases `trf` is resolution- and path-robust: it converges the inner force solve
faster and does not degrade with radial resolution, whereas `dogbox` degrades
noticeably. `dogbox` remains selectable via `--method dogbox`.

Once the frozen-`iBns` cadence is in place the residuals are *not*
outlier-dominated, so `--loss linear` (the default) is preferred over `soft_l1`,
and `--x-scale 1.0` over `jac`.

## The `s.inspect` plotting namespace

Grid/mesh inspection helpers live on the `inspect` namespace of a `SPECTRE`
instance and delegate to `spectre.plotting`:

```python
s = SPECTRE("case.toml")
s.inspect.plot_infaces()        # xin defaults to s.xin
s.inspect.plot_grid(xin)        # or pass an explicit geometry
s.inspect.plot_jacobian()
rij, zij = s.inspect.plot_true_mesh(vvol=0)   # reads the in-memory Fortran mesh
```

`inspect` is a cached property returning a lightweight `_Inspect(self)`. The
`xin`-based methods default `xin` to the instance's current `xin`, so they can
be called with no arguments; `plot_true_mesh` takes no `xin` and reads the
Fortran-computed mesh for the geometry currently in memory.
