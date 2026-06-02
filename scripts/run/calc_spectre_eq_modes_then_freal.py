#!/usr/bin/env -S python -u

# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

"""Two-stage free-boundary optimization, two `Force_Minimizer` runs back to back.

Stage 1: ``objective='fmodes'`` with ``exp_fact`` weighting. Compiled-Fortran
`dforce` path — fast. All ranks initialize with ``objective='fmodes'`` so
peer-rank listeners run `dforce` collectively (same MPI sequence as rank 0).

Stage 1 is split into two sub-stages within one `Force_Minimizer`:
  * Stage 1a: ``x_scale=1.0`` — uniform trust region — lets the
    worst-residual interface come down without `'jac'`'s column-norm
    clamping. Short (default 2 iters).
  * Stage 1b: ``x_scale='jac'`` — Moré-style preconditioning. Now
    appropriate because Stage 1a tamed the outlier; this is where the
    optimizer makes broad progress across the residual spectrum.

Stage 2: ``objective`` from the user's TOML (typically ``'freal'``). The
standard free-boundary force-balance optimization. Initial ``x`` for this
stage is the converged ``x`` from Stage 1, broadcast to all ranks.

This is implemented as two separate `Force_Minimizer` instances rather than a
mid-run objective swap, because the two objectives require different MPI
collective sequences (`dforce` vs `force_real`); the two-instance pattern
avoids any cross-stage MPI synchronization gymnastics. The cost is one
extra `SPECTRE` init + one extra initial forward solve at the Stage 2
boundary; in exchange the implementation is short and obviously correct.

Stage 1's objective and exp_fact are passed via
``Force_Minimizer(..., min_params_overrides={...})`` — programmatic,
no temp files. Every rank receives the same overrides dict from the
caller (this script), so post-override `min_params` is consistent.

`xs_fixb/` from Stage 1 is renamed to `xs_fixb_stage1/` before Stage 2 starts
(Stage 2's `Force_Minimizer` init would otherwise wipe it), so both
trajectories are preserved on disk.

Usage:
  ./calc_spectre_eq_modes_then_freal.py path/to/input.toml
  ./calc_spectre_eq_modes_then_freal.py --exp-fact -0.1 path/to/input.toml
  ./calc_spectre_eq_modes_then_freal.py --stage1-iters 15 path/to/input.toml
  ./calc_spectre_eq_modes_then_freal.py --skip-stage2 path/to/input.toml
"""

from argparse import ArgumentParser

parser = ArgumentParser(
    prog="calc_spectre_eq_modes_then_freal",
    description=(
        "Two-stage SPECTRE free-boundary optimization. Stage 1: fmodes "
        "(compiled-Fortran modal force with exp_fact weighting). Stage 2: "
        "standard objective (typically freal), initialized from Stage 1's "
        "converged x."
    ),
)
parser.add_argument("inputfile", help="Path to the input .toml file")
parser.add_argument(
    "--exp-fact",
    type=float,
    default=-0.05,
    help=(
        "Mode-weight exponent for Stage 1: w(m,n)=exp(exp_fact*(m^2+n^2)). "
        "exp_fact=0 is Parseval-equivalent to freal; exp_fact<0 damps high "
        "modes. Default -0.05."
    ),
)
parser.add_argument(
    "--stage1a-iters",
    type=int,
    default=4,
    help=(
        "Max iterations for Stage 1a (default 2). Stage 1a runs the fmodes "
        "objective with x_scale=1.0 — uniform trust region — which lets "
        "the outlier-residual interface move freely instead of being "
        "clamped by 'jac'-style column-norm scaling."
    ),
)
parser.add_argument(
    "--stage1b-iters",
    type=int,
    default=10,
    help=(
        "Max iterations for Stage 1b (default 10). Stage 1b continues with "
        "the same fmodes objective but switches to x_scale='jac', "
        "appropriate once the worst residual has come down to a comparable "
        "magnitude as the others."
    ),
)
parser.add_argument(
    "--stage1-damp",
    type=float,
    default=1e-3,
    help="LSMR Tikhonov damp for Stage 1 (default 1e-3).",
)
parser.add_argument(
    "--stage2-damp",
    type=float,
    default=1e-3,
    help="LSMR Tikhonov damp for Stage 2 (default 1e-3).",
)
parser.add_argument(
    "--skip-stage2",
    action="store_true",
    help="Only run Stage 1 (for tuning Stage 1 in isolation).",
)

parser.add_argument(
    "--f-scale",
    type=float,
    default=5.0,
    help="soft_l1 transition scale C; default 10.0",
)

args = parser.parse_args()

import os
import shutil

from spectre import Force_Minimizer
from spectre.error_checking import print_all_errors
from spectre.force_minimizer import callback_with_interfaces
from spectre.least_squares.least_squares import least_squares as lstsq


def stage1_optimizer(obj, x_init, f, jac):
    p = obj.min_params

    # ------------------------------------------------------------------
    # Stage 1a: x_scale=1.0 — uniform trust region.
    # With residual outliers, 'jac' clamps the very directions that most
    # need to move (the outlier-interface DOFs). Uniform x_scale lets the
    # outlier come down without the optimizer "destroying" the other
    # interfaces in compensation. Short (default 2 iters) — the outlier
    # typically tames in 1–2 steps.
    # ------------------------------------------------------------------
    if obj.rank == 0:
        print(
            "\n=== STAGE 1a (fmodes, x_scale=1.0) ===\n"
            f"  method='trf'  tr_solver='lsmr'\n"
            f"  tr_options={{'regularize': True, 'damp': {args.stage1_damp}}}\n"
            f"  x_scale=1.0  loss='soft_l1'  f_scale={args.f_scale}\n"
            f"  objective='fmodes'  exp_fact={p.exp_fact}\n"
            f"  max_niter={args.stage1a_iters}\n"
        )

    r1a = lstsq(
        f, x_init, jac=jac,
        method="trf", verbose=2, args=[obj],
        gtol=p.gtol, xtol=p.xtol, ftol=p.ftol,
        max_nfev=p.max_nfev, max_niter=args.stage1a_iters,
        iteration_start=obj.iteration_start,
        rcond_lstsq=p.rcond_lstsq,
        x_scale=1.0, tr_solver="lsmr",
        tr_options={"regularize": True, "damp": args.stage1_damp},
        loss="soft_l1",
        f_scale=args.f_scale,
        cb=obj.callback,
    )
    if obj.rank == 0:
        print(
            "\n=== STAGE 1a DONE ===\n"
            f"  status={r1a.status}  ({r1a.message})\n"
            f"  cost={r1a.cost:.4e}  nfev={r1a.nfev}  optimality={r1a.optimality:.4e}\n"
        )

    if r1a.status == 5:   # SIGUSR1
        return r1a

    # ------------------------------------------------------------------
    # Stage 1b: x_scale='jac' — Moré-style preconditioning is now
    # appropriate because the outlier residual was tamed in 1a. Same
    # objective (fmodes) and loss (soft_l1); same exp_fact. This is where
    # the optimizer actually makes broad progress across the residual
    # spectrum, with the column-norm rescaling helping (rather than
    # hurting) the convergence rate per iteration.
    # ------------------------------------------------------------------
    stage1b_iter_start = obj.iteration_start + args.stage1a_iters
    if obj.rank == 0:
        print(
            "\n=== STAGE 1b (fmodes, x_scale='jac') ===\n"
            f"  method='trf'  tr_solver='lsmr'\n"
            f"  tr_options={{'regularize': True, 'damp': {args.stage1_damp}}}\n"
            f"  x_scale='jac'  loss='soft_l1'  f_scale={args.f_scale}\n"
            f"  objective='fmodes'  exp_fact={p.exp_fact}\n"
            f"  iteration_start={stage1b_iter_start}  max_niter={args.stage1b_iters}\n"
        )

    r = lstsq(
        f, r1a.x, jac=jac,
        method="trf", verbose=2, args=[obj],
        gtol=p.gtol, xtol=p.xtol, ftol=p.ftol,
        max_nfev=p.max_nfev, max_niter=args.stage1b_iters,
        iteration_start=stage1b_iter_start,
        rcond_lstsq=p.rcond_lstsq,
        x_scale="jac", tr_solver="lsmr",
        tr_options={"regularize": True, "damp": args.stage1_damp},
        loss="soft_l1",
        f_scale=args.f_scale,
        cb=obj.callback,
    )
    if obj.rank == 0:
        print(
            "\n=== STAGE 1b DONE ===\n"
            f"  status={r.status}  ({r.message})\n"
            f"  cost={r.cost:.4e}  nfev={r.nfev}  optimality={r.optimality:.4e}\n"
        )
    return r


def stage2_optimizer(obj, x_init, f, jac):
    p = obj.min_params
    if obj.rank == 0:
        print(
            f"\n=== STAGE 2 ({p.objective}) ===\n"
            f"  method='trf'  tr_solver='lsmr'\n"
            f"  tr_options={{'regularize': True, 'damp': {args.stage2_damp}}}\n"
            f"  x_scale='jac'  loss='linear'\n"
            f"  max_niter={p.max_niter}\n"
        )

    r = lstsq(
        f, x_init, jac=jac,
        method="trf", verbose=2, args=[obj],
        gtol=p.gtol, xtol=p.xtol, ftol=p.ftol,
        max_nfev=p.max_nfev, max_niter=p.max_niter,
        iteration_start=obj.iteration_start,
        rcond_lstsq=p.rcond_lstsq,
        x_scale="jac", tr_solver="lsmr",
        tr_options={"regularize": True, "damp": args.stage2_damp},
        loss="linear",
        cb=obj.callback,
    )
    if obj.rank == 0:
        print(
            "\n=== STAGE 2 DONE ===\n"
            f"  status={r.status}  ({r.message})\n"
            f"  cost={r.cost:.4e}  nfev={r.nfev}  optimality={r.optimality:.4e}\n"
        )
    return r


# ----------------------------------------------------------------------
# Stage 1 — pass objective/exp_fact directly via constructor overrides; no
# temp TOML, no file I/O. Every rank constructs Force_Minimizer with the
# same overrides dict, so post-override min_params is consistent without an
# explicit MPI broadcast.
# ----------------------------------------------------------------------
obj1 = Force_Minimizer(
    args.inputfile,
    optimizer=stage1_optimizer,
    callback=callback_with_interfaces,
    min_params_overrides={"objective": "fmodes", "exp_fact": args.exp_fact},
)
obj1.minimize()

# Pull Stage 1's converged x off rank 0 and broadcast to all ranks so Stage 2
# can be initialized consistently.
if obj1.rank == 0:
    x_after_stage1 = obj1.result.x.copy()
else:
    x_after_stage1 = None
x_after_stage1 = obj1.comm.bcast(x_after_stage1, root=0)

if args.skip_stage2:
    if obj1.rank == 0:
        print("\n=== STAGE 2 SKIPPED (--skip-stage2) ===\n")
    obj1.postprocess(save_json=False, save_pkl=False)
    if obj1.rank == 0:
        print("SPECTRE equilibrium calculation done (Stage 1 only)!")
else:
    # Preserve Stage 1 snapshots — Stage 2's Force_Minimizer.__init__ will
    # wipe `xs_fixb/` if `iteration_start == 0`.
    if obj1.rank == 0 and os.path.isdir("xs_fixb"):
        if os.path.isdir("xs_fixb_stage1"):
            shutil.rmtree("xs_fixb_stage1")
        shutil.move("xs_fixb", "xs_fixb_stage1")

    # --------------------------------------------------------------
    # Stage 2 — use the user's original TOML (typically `objective='freal'`).
    # --------------------------------------------------------------
    obj2 = Force_Minimizer(
        args.inputfile,
        optimizer=stage2_optimizer,
        callback=callback_with_interfaces,
    )

    # Replace the just-initialized x_init with Stage 1's converged x on all
    # ranks. (Force_Minimizer reads x_init from `field_obj.xin`, which picks
    # up the current Fortran state — but `SPECTRE.__init__` may have reset
    # that during Stage 2's init, so we set it explicitly here.)
    obj2.x_init = x_after_stage1.copy()
    obj2.x_curr = x_after_stage1.copy()

    # Push x_after_stage1 into Fortran on all ranks and re-print the
    # diagnostics so the FIELD/FORCE table reflects Stage 1's converged
    # state. **Do not call `obj2.obj_func` here**: on rank 0 it's the
    # MPI-aware wrapper (sends signals + collective update_bnorm), on
    # peer ranks it's the plain solve_field path. Calling it here, before
    # `obj2.minimize()` has put peers in their signal-listener loop,
    # crosses two different MPI collective sequences on different ranks
    # and deadlocks. Use the bound SPECTRE.force_real instead — same
    # collectives on every rank, no signaling.
    obj2.field_obj.xin = x_after_stage1
    obj2.field_obj.force_real()
    if obj2.rank == 0:
        print("\n=== Stage 2 initial state (at Stage 1's converged x) ===")
    print_all_errors(obj2.field_obj, beltrami=True, force=True)

    obj2.minimize()
    obj2.postprocess(save_json=False, save_pkl=False)

    if obj2.rank == 0:
        print("SPECTRE equilibrium calculation done (modes-then-freal)!")
