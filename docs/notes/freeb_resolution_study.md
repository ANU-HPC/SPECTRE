# Free-boundary resolution & initial/final-state study (NCSX)

## Context

Follow-up to the `update_bns` timing fix and the field-error investigation
(`docs/notes/freeb_update_bns_timing.md`), which concluded the vacuum FIELD
ERROR is a **coordinate/warping pathology, not under-resolution**, and that the
"bonkers" forces trace to a **warped `fix2free` instantiation** rather than
physics. This study tests that diagnosis directly and characterizes how the
free-boundary solve scales with poloidal/toroidal resolution and with the
interface-generation path.

Two interface-generation paths (the two the `make_freeboundary_ncsx.ipynb`
notebook switches between), both built here by
`study_resolution/generate_ncsx_freeb.py`:

- **`normal`** = `fix2free`: keeps the VMEC-inherited plasma boundary and
  internal interfaces; only adds the computational boundary + vacuum B·n.
- **`refresh`** = `fix2free_refresh_infaces(flux_weighted_rho=True)`:
  regenerates the plasma boundary and all internal interfaces by conformally
  mapping the comp boundary inward at flux-corrected radii — a clean,
  well-nested grid.

## Matrix (12 runs)

`paths {refresh, normal}` × `resolutions {m7n6, m11n10}` ×
`optimizers {dogbox, trf, twostage = trf x_scale 1.0→jac @iter10}`. All driven
by `scripts/run/calc_spectre_eq_freeb.py --bns-trigger outer --total-iters 20`
(the corrected outer/inner-loop driver), one MPI job at a time
(`OMP_NUM_THREADS=3 mpirun -n 3`). Each run retains `out.txt`, `xs_fixb/`
(per-iteration geometry), `*_opt.h5`, `*_opt.toml`, and per-run figures.
Generated/driven by `study_resolution/run_resolution_study.sh`.

## Results — best `<f>` after 20 iterations

| resolution | path | dogbox | trf | twostage |
| --- | --- | ---: | ---: | ---: |
| m7n6 | refresh | 4.35e-3 | 4.43e-3 | 4.46e-3 |
| m7n6 | normal | **1.49e-1** | 1.10e-2 | 5.87e-2 |
| m11n10 | refresh | 1.90e-2 | **4.47e-3** | 5.48e-3 |
| m11n10 | normal | 7.87e-2 | **4.00e-3** † | 1.45e-2 |

† one transient `<f> = nan` at iter 2 (a degenerate FSA evaluation on the
warped path), which trf recovered from; value excludes that point.

**Wall time** (`mpirun -n 3`): m7n6 ≈ 234–283 s; m11n10 ≈ 3678–4371 s — a
**~16× increase** for m7n6→m11n10, consistent with the mode count `mn`
growing 98 → 242.

## Field error — initial vs final (per-volume Beltrami residual)

The axis volume (`plasma0`) and the vacuum volume tell the whole story.
Values are the `max` over the grid (the `max` is what blows up; the `avg`
stays moderate, as established in the timing report).

| config | axis (plasma0) max: init → final | vacuum max: init → final |
| --- | --- | --- |
| refresh m7n6 (trf) | **6.0** → 1.3e2 | 1.8e2 → 4.3e2 |
| normal  m7n6 (trf) | **6.3e8** → 6.1e9 | 2.6e2 → 5.2e1 |
| refresh m11n10 (trf) | **0.97** → 7.7e6 | 2.1e2 → 1.5e2 |
| normal  m11n10 (trf) | **3.9e6** → 2.7e9 | 2.1e2 → 6.7e1 |

The decisive number is the **axis field error at initialization**: the
`normal` path is born with `plasma0` max ≈ **6.3e8** (m7n6) / **3.9e6**
(m11n10), while the `refresh` path starts at **6.0** / **0.97** — eight orders
of magnitude cleaner. The normal-path initial Jacobian (figure below) shows a
**near-zero/negative region at the axis** (colorbar dips below 0 — a coordinate
fold), which is exactly the warped-instantiation signature. The vacuum field
error is *similar* between paths (~2e2 init both), so for these cases the
warping is concentrated at the **magnetic axis**, created by the inherited
interfaces failing to nest cleanly inside the new computational boundary.

## Figures

Clean (`refresh`) vs warped (`normal`) coordinate grid at m7n6 (note the axis
region of the `normal` initial Jacobian):

![refresh m7n6 grid](../../examples/utils/make_freeboundary/study_resolution/run_refresh_m7n6_trf/grids_init_final.png)
![normal m7n6 grid](../../examples/utils/make_freeboundary/study_resolution/run_normal_m7n6_trf/grids_init_final.png)

High resolution — the best run (`refresh` + `trf`) and the warped path
(`normal` + `dogbox`):

![refresh m11n10 trf grid](../../examples/utils/make_freeboundary/study_resolution/run_refresh_m11n10_trf/grids_init_final.png)
![normal m11n10 dogbox grid](../../examples/utils/make_freeboundary/study_resolution/run_normal_m11n10_dogbox/grids_init_final.png)

Convergence at m11n10 — dogbox (degrades) vs trf (resolution-robust):

![dogbox m11n10 convergence](../../examples/utils/make_freeboundary/study_resolution/run_refresh_m11n10_dogbox/convergence.png)
![trf m11n10 convergence](../../examples/utils/make_freeboundary/study_resolution/run_refresh_m11n10_trf/convergence.png)

(Every run's `grids_init_final.png`, `convergence.png`, and `field_err.txt`
live in its `run_<path>_<res>_<opt>/` folder.)

## Conclusions

1. **`refresh` ≫ `normal` at low resolution, and it's a clean-axis-grid
   effect — warped instantiation confirmed.** At m7n6 the refresh path reaches
   `<f>` ≈ 4.4e-3 with *every* optimizer, while the normal path stalls at
   1.1e-2 (trf) to 1.5e-1 (dogbox). The cause is unambiguous in the field
   error: the normal path's inherited interfaces produce a **near-degenerate
   (folded) coordinate at the magnetic axis from the very first evaluation**
   (`plasma0` init field-error max ≈ 6.3e8 vs the refresh path's ≈ 6). This is
   the controlled confirmation of the warped-instantiation diagnosis that the
   earlier reports could only hypothesize — and it retro-justifies down-grading
   the "current sheet / physics" conclusion in
   `free_boundary_optimization_notes.md`.

2. **`trf` is robust across both resolution and path; `dogbox` is not.**
   - Resolution: `refresh` trf gives 4.43e-3 (m7n6) and 4.47e-3 (m11n10) — *no
     resolution penalty* — while dogbox degrades from 4.35e-3 to 1.90e-2 (~4×
     worse) at higher resolution within the same 20-iter budget.
   - Path: at m11n10, trf even **rescues the warped `normal` path to 4.00e-3**
     (matching the refresh path), whereas dogbox is stuck at 7.87e-2 — a ~20×
     gap. Higher resolution + trf overcomes the warped axis for *force balance*
     (though the axis field error remains large — see below).
   This reinforces the timing-study decision to make `trf` the
   `calc_spectre_eq_freeb.py` default.

3. **The two-stage `x_scale 1.0→jac` switch gives no gain over plain trf.**
   refresh m11n10: trf 4.47e-3 vs twostage 5.48e-3; normal m11n10: trf 4.00e-3
   vs twostage 1.45e-2. Consistent with the timing sweep's `x_scale=1.0 > jac`
   finding — switching to `jac` mid-run slightly hurts.

4. **Resolution scaling.** Cost is ~16× per solve from m7n6→m11n10 (mn 98→242).
   Higher resolution gives a *cleaner initial* geometry (init field-error maxes
   drop: normal axis 6.3e8→3.9e6; refresh axis 6.0→0.97) because more modes
   represent the surfaces better — but the **axis singularity still sharpens
   during optimization** at high res (e.g. refresh m11n10 trf `plasma0` final
   max 7.7e6). So field quality at the axis is **geometry/coordinate-driven,
   not a resolution knob** — the same lesson as the vacuum-`Lrad` experiment in
   the timing report.

5. **The warped path can emit a transient `nan` `<f>`** (normal m11n10 trf,
   iter 2) from a degenerate FSA evaluation; trf recovered and went on to the
   best normal-path result. A guard against `nan`/`inf` force evaluations (skip
   the step rather than propagate) would make such runs more robust.

## Recommendation

Use the **`refresh` generation path** (clean nested axis grid) with **`trf`**.
This combination gives the best, resolution-independent force convergence
(~4.4e-3 @ 20 iters at both resolutions) and avoids the warped-axis
instantiation that cripples the `normal` path. For difficult/warped starts,
`trf` at higher resolution is the most robust recovery.

## Reproduce

```bash
cd examples/utils/make_freeboundary/study_resolution
# generate the 4 init TOMLs (2 paths x 2 resolutions)
for path in refresh normal; do
  python generate_ncsx_freeb.py --mpol 7  --ntor 6  --path $path --out tomls/ncsx_${path}_m7n6.toml
  python generate_ncsx_freeb.py --mpol 11 --ntor 10 --path $path --out tomls/ncsx_${path}_m11n10.toml
done
# run the 12-config matrix one at a time (low-res first), with per-run figures
bash run_resolution_study.sh          # see study_summary.txt for results
```

Cross-references: `docs/notes/freeb_update_bns_timing.md` (the bns-timing fix +
tuning sweep that make `trf` the default and identified the vacuum field error
as coordinate-driven); `free_boundary_optimization_notes.md` /
`docs/notes/ibns_picard_investigation.md` (the earlier reports whose
"warped-instantiation vs physics" question this study settles in favor of
warped instantiation).
