#!/usr/bin/env -S python -u

# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

"""Robustness experiment R1: soft_l1 loss on top of the experiment-A optimizer.

Identical wiring to `calc_spectre_eq_experimentA.py` but adds two arguments to
`lstsq`:

  * ``loss="soft_l1"`` — robust loss `rho(z) = 2*(sqrt(1+z) - 1)` with `z=(f/C)^2`.
    Quadratic for `|f| << C` (acts like the default linear loss), sqrt-like for
    `|f| >> C`. The Gauss-Newton-style gradient contribution scales like
    ``f / sqrt(C**2 + f**2)`` — bounded by ``1/C`` regardless of how big `f`
    gets.
  * ``f_scale=<--f-scale>`` — the transition scale `C`.

**Motivation.** At cold start one interface's residual can be ~1e8x larger
than the interior's (see free-boundary logs where iface-5 rms ~5e+08 vs
iface-0..4 rms ~1). With linear loss that single outlier monopolises the
Gauss-Newton step. soft_l1 caps its gradient contribution at ~1/C, leaving
real weight for the interior directions and giving more balanced first steps.

**Picking f_scale.**

  * Too small (~ typical inlier): even inliers get downweighted; slow.
  * Too large (~ outlier): equivalent to linear loss; no benefit.
  * Default 10.0 keeps inliers (rms ~1) firmly in the quadratic regime while
    capping the outlier:inlier gradient ratio at ~10. Raise to 100 or 1000 if
    the outlier's information is being thrown away; drop to 1 if outliers are
    still steering the step.

Usage:
  ./calc_spectre_eq_softl1.py path/to/input.toml
  ./calc_spectre_eq_softl1.py --f-scale 100 path/to/input.toml
  nohup ./calc_spectre_eq_softl1.py $FNAME > out_softl1.txt 2>&1 &
"""

from argparse import ArgumentParser

parser = ArgumentParser(
    prog="calc_spectre_eq_softl1",
    description=(
        "SPECTRE force minimization with the experiment-A optimizer plus the "
        "soft_l1 robust loss (R1)."
    ),
)
parser.add_argument("inputfile", help="Path to the input .toml file")
parser.add_argument(
    "--f-scale",
    type=float,
    default=10.0,
    help="soft_l1 transition scale C; default 10.0",
)
args = parser.parse_args()

from spectre import Force_Minimizer
from spectre.least_squares.least_squares import least_squares as lstsq


def optimizer_softl1(obj, x_init, f, jac):
    p = obj.min_params

    if obj.rank == 0:
        print(
            "[R1] method='trf'  tr_solver='lsmr'  "
            "tr_options={'regularize': True, 'damp':1e-3}  x_scale='jac'  "
            f"loss='soft_l1'  f_scale={args.f_scale}  "
            f"(overriding min_params.method={p.method!r})"
        )

    return lstsq(
        f,
        x_init,
        jac=jac,
        method="trf",
        verbose=2,
        args=[obj],
        gtol=p.gtol,
        xtol=p.xtol,
        ftol=p.ftol,
        max_nfev=p.max_nfev,
        max_niter=p.max_niter,
        iteration_start=obj.iteration_start,
        rcond_lstsq=p.rcond_lstsq,
        x_scale=1.0,
        tr_solver="lsmr",
        tr_options={"regularize": True, "damp":1e-3},
        loss="soft_l1",
        f_scale=args.f_scale,
        cb=obj.callback,
    )


obj = Force_Minimizer(args.inputfile, optimizer=optimizer_softl1)

obj.minimize()

obj.postprocess(save_json=False, save_pkl=False)

if obj.rank == 0:
    print("SPECTRE equilibrium calculation done (R1: soft_l1)!")
