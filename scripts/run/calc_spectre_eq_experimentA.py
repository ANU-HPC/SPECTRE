#!/usr/bin/env -S python -u

# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

"""Experiment A: TRF + LSMR + Byrd regularization + Jacobian-column scaling.

Same workflow as `calc_spectre_eq.py`, but injects a custom optimizer through
the `Force_Minimizer` constructor instead of editing the library. The four
overrides relative to `optimizer_default`:

  * method     : 'trf'     (was: min_params.method, typically 'dogbox')
  * tr_solver  : 'lsmr'    (was: None -> 'exact' for dense J)
  * tr_options : {'regularize': True}  (Byrd 1988 damping for rank-deficient J)
  * x_scale    : 'jac'     (was: 1.0; auto column-norm scaling per Moré 1977)

Tolerances, max_nfev, max_niter, rcond_lstsq are still read from min_params so
the input TOML stays the single source of truth for those knobs. `rcond_lstsq`
is dogbox-only and silently ignored by trf+lsmr.

Usage:
  ./calc_spectre_eq_experimentA.py path/to/input.toml
  nohup ./calc_spectre_eq_experimentA.py $FNAME > out.txt 2>&1 &
"""

from argparse import ArgumentParser

parser = ArgumentParser(
    prog="calc_spectre_eq_experimentA",
    description=(
        "Find equilibrium using SPECTRE force minimization with the experiment-A "
        "optimizer (TRF + LSMR + regularize + jac scaling)."
    ),
)
parser.add_argument("inputfile", help="Path to the input .toml file")
args = parser.parse_args()

from spectre import Force_Minimizer
from spectre.least_squares.least_squares import least_squares as lstsq


def optimizer_experimentA(obj, x_init, f, jac):
    p = obj.min_params

    if obj.rank == 0:
        print(
            "[experiment-A] method='trf'  tr_solver='lsmr'  "
            "tr_options={'regularize': True}  x_scale=1.0  "
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
        x_scale="jac",
        tr_solver="lsmr",
        tr_options={"regularize": True, "damp":1e-4},
        cb=obj.callback,
    )


obj = Force_Minimizer(args.inputfile, optimizer=optimizer_experimentA)

obj.minimize()

obj.postprocess(save_json=False, save_pkl=False)

if obj.rank == 0:
    print("SPECTRE equilibrium calculation done (experiment A)!")
