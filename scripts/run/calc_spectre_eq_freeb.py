#!/usr/bin/env -S python -u

# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

"""Free-boundary equilibrium driver that refreshes iBns *at the right time*.

The legacy free-boundary path (``bns_update_mode='per_eval'`` in
`Force_Minimizer`) calls ``update_bnorm`` before **every** objective
evaluation — including rejected trust-region trial steps. Because the
Jacobian is computed once per accepted iteration at a *frozen* iBns, while
the residual surface moves under it as trial evaluations mutate iBns, the
solve runs away near convergence (see ``out_ncsx_TEST.txt`` and
``docs/notes/freeb_update_bns_timing.md``).

This driver fixes the cadence. It runs with ``bns_update_mode='outer_loop'``
so the per-evaluation refresh is disabled, and instead refreshes iBns only
between *converged/stalled* inner force solves (``--bns-trigger outer``,
default) — the Jacobian is always valid because iBns is frozen for the whole
inner solve. A prototype in-loop trigger (``--bns-trigger inloop``) is also
available for comparison: a single solve that refreshes iBns mid-run whenever
the step norm drops below a threshold.

Because these free-boundary cases do not fully converge in practice, the
figure of merit is the lowest ``<f>`` (FSA force metric) reached within a
fixed iteration budget (``--total-iters``, default 30).

Usage:
  OMP_NUM_THREADS=3 mpirun -n 3 ./calc_spectre_eq_freeb.py input.toml
  ./calc_spectre_eq_freeb.py --bns-trigger inloop input.toml
  ./calc_spectre_eq_freeb.py --total-iters 30 --inner-iter-cap 5 input.toml
"""

from argparse import ArgumentParser

parser = ArgumentParser(
    prog="calc_spectre_eq_freeb",
    description=(
        "Free-boundary SPECTRE optimization with correctly-timed iBns "
        "refresh (outer/inner loop, or prototype in-loop trigger)."
    ),
)
parser.add_argument("inputfile", help="Path to the input .toml file")
parser.add_argument(
    "--bns-trigger",
    choices=["outer", "inloop"],
    default="outer",
    help=(
        "When to refresh iBns. 'outer' (default): freeze iBns, run an inner "
        "force solve to convergence/stall, then refresh once, repeat. "
        "'inloop': single solve, refresh mid-run when step_norm < --step-tol."
    ),
)
parser.add_argument(
    "--total-iters",
    type=int,
    default=30,
    help="Total inner-iteration budget across all outer cycles (default 30).",
)
parser.add_argument(
    "--inner-iter-cap",
    type=int,
    default=5,
    help=(
        "Max inner iterations per outer cycle before forcing an iBns refresh "
        "(outer trigger only; default 5)."
    ),
)
parser.add_argument(
    "--step-tol",
    type=float,
    default=1e-3,
    help=(
        "Inner-solve is considered stalled (→ refresh iBns) when the accepted "
        "step norm drops below this. Also the in-loop refresh threshold. "
        "Default 1e-3."
    ),
)
parser.add_argument(
    "--cost-red-frac",
    type=float,
    default=1e-3,
    help=(
        "Inner-solve is considered stalled when cost reduction < this fraction "
        "of the current cost (combined with --step-tol). Default 1e-3."
    ),
)
parser.add_argument(
    "--gbnbld",
    type=float,
    default=None,
    help=(
        "iBns relaxation blend for the refresh: iBns := g*iBns + (1-g)*ofmn. "
        "Default None → use the TOML gbnbld. The picard study found 0.5-0.7 "
        "robust; 0 (full replacement) can diverge far from the fixed point."
    ),
)
parser.add_argument(
    "--method",
    choices=["trf", "dogbox"],
    default="trf",
    help=(
        "Inner least-squares method. Default 'trf' — the tuning sweep on "
        "ncsx_TEST found trf clearly beats dogbox (<f> 6.16e-3 vs 7.82e-3); "
        "see docs/notes/freeb_update_bns_timing.md. dogbox remains selectable."
    ),
)
parser.add_argument(
    "--x-scale",
    choices=["1.0", "jac"],
    default="1.0",
    help="x_scale for the inner solve (default 1.0; stage 1 if --x-scale2 is set).",
)
parser.add_argument(
    "--x-scale2",
    choices=["1.0", "jac"],
    default=None,
    help=(
        "Optional two-stage x_scale: outer cycles starting at/after --switch-iter "
        "use this x_scale instead of --x-scale. E.g. '--x-scale 1.0 --x-scale2 jac "
        "--switch-iter 10' runs the first 10 iters with x_scale=1.0, then 'jac'. "
        "Only affects the outer trigger."
    ),
)
parser.add_argument(
    "--switch-iter",
    type=int,
    default=10,
    help="Global iteration at which to switch to --x-scale2 (default 10).",
)
parser.add_argument(
    "--loss",
    choices=["linear", "soft_l1"],
    default="linear",
    help="Loss for the inner solve (default linear).",
)
parser.add_argument(
    "--f-scale",
    type=float,
    default=1.0,
    help="soft_l1 transition scale (default 1.0).",
)

args = parser.parse_args()


from spectre import Force_Minimizer
from spectre.force_minimizer import callback_with_interfaces
from spectre.least_squares.least_squares import least_squares as lstsq


def _x_scale_val(choice):
    return 1.0 if choice == "1.0" else "jac"


def _x_scale_for_cycle(cycle_start):
    """x_scale for an outer cycle: switch to --x-scale2 once cycle_start reaches
    --switch-iter (two-stage). Plain --x-scale if --x-scale2 is unset."""
    if args.x_scale2 is not None and cycle_start >= args.switch_iter:
        return _x_scale_val(args.x_scale2)
    return _x_scale_val(args.x_scale)


def _inner_lstsq_kwargs(obj, x_scale=None):
    """Shared kwargs for an inner force solve (frozen iBns)."""
    p = obj.min_params
    kw = dict(
        method=args.method,
        verbose=2,
        args=[obj],
        gtol=p.gtol,
        xtol=p.xtol,
        ftol=p.ftol,
        max_nfev=p.max_nfev,
        rcond_lstsq=p.rcond_lstsq,
        x_scale=_x_scale_val(args.x_scale) if x_scale is None else x_scale,
        loss=args.loss,
        f_scale=args.f_scale,
    )
    return kw


def optimizer_outer(obj, x_init, f, jac):
    """Outer/inner loop: frozen-iBns inner solves, damped refresh between."""
    p = obj.min_params

    # Plain single solve for fixed-boundary inputs — no bns machinery.
    if not obj.freeb_mode:
        return lstsq(
            f,
            x_init,
            jac=jac,
            max_niter=args.total_iters,
            iteration_start=obj.iteration_start,
            cb=obj.callback,
            **_inner_lstsq_kwargs(obj),
        )

    # state['last_it'] tracks the global iteration index so xs_fixb snapshots
    # don't collide and the 30-iter budget spans all outer cycles. The wrapped
    # callback also captures the latest step_norm / cost / cost_red so the
    # stall test can read them after the inner solve returns.
    state = {"last_it": obj.iteration_start, "step_norm": None, "cost": None, "cost_red": None}

    def cb_wrap(it, nfev, cost, cost_red, stp_nrm, opt, x, o):
        obj.callback(it, nfev, cost, cost_red, stp_nrm, opt, x, o)
        state["last_it"] = it
        state["step_norm"] = stp_nrm
        state["cost"] = cost
        state["cost_red"] = cost_red

    bnserr_hist = []
    x = x_init
    r = None
    outer = 0
    print(
        f"\n=== FREE-BOUNDARY OUTER/INNER LOOP ===\n"
        f"  total_iters={args.total_iters}  inner_iter_cap={args.inner_iter_cap}\n"
        f"  method={args.method}  x_scale={args.x_scale}"
        + (f" -> {args.x_scale2} @iter{args.switch_iter}\n" if args.x_scale2 is not None else "\n")
        + f"  loss={args.loss}\n"
        f"  step_tol={args.step_tol}  cost_red_frac={args.cost_red_frac}\n"
        f"  gbnbld={'TOML' if args.gbnbld is None else args.gbnbld}\n"
        f"  freeb_bnorm_tol={p.freeb_bnorm_tol}\n"
    )

    while state["last_it"] < obj.iteration_start + args.total_iters:
        remaining = obj.iteration_start + args.total_iters - state["last_it"]
        cap = min(args.inner_iter_cap, remaining)
        cycle_start = state["last_it"]
        max_niter = cycle_start + cap
        cycle_x_scale = _x_scale_for_cycle(cycle_start)

        print(
            f"\n--- outer cycle {outer} (frozen iBns) iters {cycle_start}->{max_niter} "
            f"x_scale={cycle_x_scale} ---"
        )

        # INNER: minimize force balance at FROZEN iBns. The Jacobian is valid
        # throughout because iBns does not move inside this call
        # (bns_update_mode='outer_loop').
        r = lstsq(
            f,
            x,
            jac=jac,
            max_niter=max_niter,
            iteration_start=cycle_start,
            cb=cb_wrap,
            **_inner_lstsq_kwargs(obj, x_scale=cycle_x_scale),
        )
        x = r.x

        if r.status == 5:  # SIGUSR1 abort
            print("=== Aborting (SIGUSR1) ===")
            break

        # Put the Fortran plasma solution at the accepted x so the iBns
        # extrapolation (virtual casing) is taken at the right geometry. In
        # outer_loop mode this f-eval does NOT itself refresh iBns.
        f(x, obj)

        # Refresh iBns once (damped). This is the only place iBns moves.
        bnserr = obj.refresh_bns(damping=args.gbnbld)
        bnserr_hist.append(bnserr)

        inner_converged = r.status in (1, 2, 3)  # gtol / ftol / xtol
        inner_stalled = (
            state["step_norm"] is not None
            and state["step_norm"] < args.step_tol
            and (
                state["cost_red"] is None
                or abs(state["cost_red"]) < args.cost_red_frac * max(state["cost"], 1e-300)
            )
        )
        print(
            f"--- cycle {outer} done: status={r.status} cost={r.cost:.4e} "
            f"bnserr={bnserr:.3e}  inner_converged={inner_converged} "
            f"inner_stalled={inner_stalled} ---"
        )

        # Whole-problem convergence: force balance solved AND iBns consistent.
        if bnserr < p.freeb_bnorm_tol and (inner_converged or inner_stalled):
            print("=== Converged: bnserr below tol and inner force solve settled ===")
            break

        # Safety: if the damped Picard map isn't contracting, warn (the
        # picard study suggests nudging gbnbld up toward 1, i.e. less
        # aggressive replacement).
        if len(bnserr_hist) >= 2 and bnserr_hist[-1] > bnserr_hist[-2]:
            print(
                f"  WARNING: bnserr grew ({bnserr_hist[-2]:.3e} -> "
                f"{bnserr_hist[-1]:.3e}); iBns map may be non-contractive. "
                f"Consider a larger --gbnbld."
            )

        outer += 1

    return r


def optimizer_inloop(obj, x_init, f, jac):
    """Prototype: single solve, refresh iBns mid-run when step_norm < step_tol."""
    if not obj.freeb_mode:
        return lstsq(
            f,
            x_init,
            jac=jac,
            max_niter=args.total_iters,
            iteration_start=obj.iteration_start,
            cb=obj.callback,
            **_inner_lstsq_kwargs(obj),
        )

    print(
        f"\n=== FREE-BOUNDARY IN-LOOP TRIGGER (prototype) ===\n"
        f"  total_iters={args.total_iters}  step_tol={args.step_tol}\n"
        f"  method={args.method}  x_scale={args.x_scale}  loss={args.loss}\n"
        f"  gbnbld={'TOML' if args.gbnbld is None else args.gbnbld}\n"
        "  NOTE: refreshing iBns mid-solve moves the residual surface under a\n"
        "  Jacobian/f computed at the old iBns — expected to be less robust\n"
        "  than the outer/inner loop. Here for comparison only.\n"
    )

    def cb_wrap(it, nfev, cost, cost_red, stp_nrm, opt, x, o):
        obj.callback(it, nfev, cost, cost_red, stp_nrm, opt, x, o)
        if stp_nrm is not None and stp_nrm < args.step_tol:
            bnserr = obj.refresh_bns(damping=args.gbnbld)
            print(
                f"  [in-loop] step_norm {stp_nrm:.2e} < {args.step_tol:.2e} "
                f"→ refreshed iBns, bnserr={bnserr:.3e}"
            )

    return lstsq(
        f,
        x_init,
        jac=jac,
        max_niter=obj.iteration_start + args.total_iters,
        iteration_start=obj.iteration_start,
        cb=cb_wrap,
        **_inner_lstsq_kwargs(obj),
    )


optimizer = optimizer_outer if args.bns_trigger == "outer" else optimizer_inloop

obj = Force_Minimizer(
    args.inputfile,
    optimizer=optimizer,
    callback=callback_with_interfaces,
    min_params_overrides={"bns_update_mode": "outer_loop"},
)

obj.minimize()
obj.postprocess(save_json=False, save_pkl=False)

if obj.rank == 0:
    print(f"SPECTRE free-boundary equilibrium done (bns-trigger={args.bns_trigger})!")
