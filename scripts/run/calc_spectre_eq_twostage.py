#!/usr/bin/env -S python -u

# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

"""Two-stage free-boundary optimization.

Stage 1 — cold start, far from equilibrium, residual imbalance possible:

  * ``method='trf'``
  * ``tr_solver='lsmr'`` with ``regularize=True`` and ``damp=--stage1-damp``
  * ``x_scale=1.0`` — uniform trust region, so the worst-residual interface
    isn't clamped by the ``'jac'`` column-norm rescaling
  * ``loss='soft_l1'`` with ``f_scale=--stage1-f-scale`` — caps the gradient
    contribution of outlier residuals, so the Newton step direction isn't
    monopolized by them
  * Capped at ``--stage1-iters`` iterations (default 30); the lstsq call may
    terminate earlier on its usual gtol/ftol/xtol criteria, in which case
    Stage 2 still runs

Stage 2 — polish, residuals roughly balanced:

  * ``method='trf'``
  * ``tr_solver='lsmr'`` with ``regularize=True`` and ``damp=--stage2-damp``
  * ``x_scale='jac'`` — now appropriate; column norms reflect physical
    mode-by-mode sensitivity rather than transient residual magnitudes
  * ``loss='linear'`` — full Gauss-Newton quadratic convergence

Switching is unconditional after Stage 1 terminates (whether by ``max_niter``
or by tolerance), unless the run was aborted via SIGUSR1.

Usage:
  ./calc_spectre_eq_twostage.py path/to/input.toml
  ./calc_spectre_eq_twostage.py --stage1-iters 50 --stage1-f-scale 100 path/to/input.toml
  ./calc_spectre_eq_twostage.py --stage1-iters 0 path/to/input.toml   # skip Stage 1
  ./calc_spectre_eq_twostage.py --skip-stage2 path/to/input.toml      # only Stage 1
  nohup ./calc_spectre_eq_twostage.py $FNAME > out_twostage.txt 2>&1 &
"""

from argparse import ArgumentParser

parser = ArgumentParser(
    prog="calc_spectre_eq_twostage",
    description=(
        "Two-stage SPECTRE free-boundary force minimization: uniform-x_scale + "
        "soft_l1 to escape the outlier regime, then jac-x_scale + linear loss "
        "to polish."
    ),
)
parser.add_argument("inputfile", help="Path to the input .toml file")
parser.add_argument(
    "--stage1-iters",
    type=int,
    default=30,
    help="Max iterations for Stage 1 (default 30; 0 skips Stage 1).",
)
parser.add_argument(
    "--stage1-f-scale",
    type=float,
    default=10.0,
    help="soft_l1 transition scale `f_scale` for Stage 1 (default 10.0).",
)
parser.add_argument(
    "--stage1-damp",
    type=float,
    default=1e-3,
    help="LSMR Tikhonov `damp` for Stage 1 (default 1e-3).",
)
parser.add_argument(
    "--stage2-damp",
    type=float,
    default=1e-3,
    help="LSMR Tikhonov `damp` for Stage 2 (default 1e-3).",
)
parser.add_argument(
    "--skip-stage2",
    action="store_true",
    help="Only run Stage 1 (useful for tuning Stage 1 in isolation).",
)
args = parser.parse_args()

from spectre import Force_Minimizer
from spectre.force_minimizer import callback_with_interfaces
from spectre.least_squares.least_squares import least_squares as lstsq


def optimizer_twostage(obj, x_init, f, jac):
    p = obj.min_params
    rank0 = obj.rank == 0

    # ------------------------------------------------------------------
    # Stage 1: uniform x_scale + soft_l1
    # ------------------------------------------------------------------
    if args.stage1_iters > 0:
        if rank0:
            print(
                "\n=== STAGE 1 ===\n"
                f"  method='trf'  tr_solver='lsmr'\n"
                f"  tr_options={{'regularize': True, 'damp': {args.stage1_damp}}}\n"
                f"  x_scale=1.0  \n"
                f"  max_niter={args.stage1_iters}\n"
                f"  (overriding min_params.method={p.method!r})\n"
            )

        r1 = lstsq(
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
            max_niter=args.stage1_iters,
            iteration_start=obj.iteration_start,
            rcond_lstsq=p.rcond_lstsq,
            x_scale=1.0,
            tr_solver="lsmr",
            tr_options={"regularize": True, "damp": args.stage1_damp},
            loss="soft_l1",
            f_scale=args.stage1_f_scale,
            cb=obj.callback,
        )

        if rank0:
            print(
                "\n=== STAGE 1 DONE ===\n"
                f"  status={r1.status}  ({r1.message})\n"
                f"  cost={r1.cost:.4e}  nfev={r1.nfev}  optimality={r1.optimality:.4e}\n"
            )

        # Don't proceed to Stage 2 if SIGUSR1 abort
        if r1.status == 5:
            if rank0:
                print("=== Aborting before Stage 2 (SIGUSR1) ===")
            return r1

        x_for_stage2 = r1.x
        # Continue iteration numbering so the saved xs_fixb snapshots don't
        # collide between stages.
        stage2_iter_start = obj.iteration_start + args.stage1_iters
    else:
        if rank0:
            print("\n=== STAGE 1 SKIPPED (--stage1-iters=0) ===\n")
        r1 = None
        x_for_stage2 = x_init
        stage2_iter_start = obj.iteration_start

    if args.skip_stage2:
        if rank0:
            print("=== STAGE 2 SKIPPED (--skip-stage2) ===\n")
        return r1

    # ------------------------------------------------------------------
    # Stage 2: jac-scaled, linear loss
    # ------------------------------------------------------------------
    if rank0:
        print(
            "\n=== STAGE 2 ===\n"
            f"  method='trf'  tr_solver='lsmr'\n"
            f"  tr_options={{'regularize': True, 'damp': {args.stage2_damp}}}\n"
            f"  x_scale='jac'  loss='linear'\n"
            f"  iteration_start={stage2_iter_start}  max_niter={p.max_niter}\n"
        )

    r2 = lstsq(
        f,
        x_for_stage2,
        jac=jac,
        method="trf",
        verbose=2,
        args=[obj],
        gtol=p.gtol,
        xtol=p.xtol,
        ftol=p.ftol,
        max_nfev=p.max_nfev,
        max_niter=p.max_niter,
        iteration_start=stage2_iter_start,
        rcond_lstsq=p.rcond_lstsq,
        x_scale="jac",
        tr_solver="lsmr",
        tr_options={"regularize": True, "damp": args.stage2_damp},
        loss="linear",
        cb=obj.callback,
    )

    if rank0:
        print(
            "\n=== STAGE 2 DONE ===\n"
            f"  status={r2.status}  ({r2.message})\n"
            f"  cost={r2.cost:.4e}  nfev={r2.nfev}  optimality={r2.optimality:.4e}\n"
        )

    return r2


obj = Force_Minimizer(
    args.inputfile,
    optimizer=optimizer_twostage,
    callback=callback_with_interfaces,
)

obj.minimize()

obj.postprocess(save_json=False, save_pkl=False)

if obj.rank == 0:
    print("SPECTRE equilibrium calculation done (two-stage)!")
