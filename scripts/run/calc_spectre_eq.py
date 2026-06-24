#!/usr/bin/env -S python -u

# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT

"""Free-boundary equilibrium driver that refreshes iBns *at the right time*.

Refreshing the free-boundary normal field iBns before **every** objective
evaluation — including rejected trust-region trial steps — runs the solve away
near convergence: the Jacobian is computed once per accepted iteration at a
*frozen* iBns, while the residual surface moves under it as trial evaluations
mutate iBns (see ``docs/notes/freeb_bns_trf.md``).

`Force_Minimizer` keeps iBns frozen during an inner force solve. This driver
refreshes iBns only between *converged/stalled* inner solves
(``bns_trigger = "outer"``, the default) — the Jacobian is always valid because
iBns is frozen for the whole inner solve. A prototype in-loop trigger
(``bns_trigger = "inloop"``) is also available for comparison: a single solve
that refreshes iBns mid-run whenever the step norm drops below ``step_tol``.

All knobs live in the ``[minimization]`` section of the input TOML
(``bns_trigger``, ``inner_iter_cap``, ``step_tol``, ``cost_red_frac``,
``method``, ``loss``, ``f_scale``, ``x_scale``, ``x_scale2``,
``switch_iter``). The total inner-iteration budget across all outer cycles is
``max_niter``.

Usage:
  OMP_NUM_THREADS=3 mpirun -n 3 ./calc_spectre_eq.py input.toml
"""

from argparse import ArgumentParser

parser = ArgumentParser(
    prog="calc_spectre_eq",
    description=(
        "Free-boundary SPECTRE optimization with correctly-timed iBns "
        "refresh (outer/inner loop, or prototype in-loop trigger). All knobs "
        "are read from the [minimization] section of the input TOML."
    ),
)
parser.add_argument("inputfile", help="Path to the input .toml file")
args = parser.parse_args()


# Imported after argparse so `--help` does not pull in the heavy spectre/MPI stack.
from spectre import Force_Minimizer  # noqa: E402
from spectre.force_minimizer import callback_metrics  # noqa: E402
from spectre.least_squares.least_squares import least_squares as lstsq  # noqa: E402


def _x_scale_for_cycle(p, cycle_start):
    """Return the x_scale for an outer cycle.

    Switches to ``x_scale2`` once ``cycle_start`` reaches ``switch_iter``
    (two-stage); uses plain ``x_scale`` when ``x_scale2`` is unset.
    """
    if p.x_scale2 is not None and cycle_start >= p.switch_iter:
        return p.x_scale2
    return p.x_scale


def _inner_lstsq_kwargs(obj, x_scale=None):
    """Shared kwargs for an inner force solve (frozen iBns)."""
    p = obj.min_params
    return dict(
        method=p.method,
        verbose=2,
        args=[obj],
        gtol=p.gtol,
        xtol=p.xtol,
        ftol=p.ftol,
        max_nfev=p.max_nfev,
        rcond_lstsq=p.rcond_lstsq,
        x_scale=p.x_scale if x_scale is None else x_scale,
        loss=p.loss,
        f_scale=p.f_scale,
    )


def optimizer_outer(obj, x_init, f, jac):
    """Outer/inner loop: frozen-iBns inner solves, damped refresh between."""
    p = obj.min_params
    total_iters = p.max_niter  # total inner-iteration budget across all cycles

    # Plain single solve for fixed-boundary inputs — no bns machinery.
    if not obj.freeb_mode:
        return lstsq(
            f,
            x_init,
            jac=jac,
            max_niter=total_iters,
            iteration_start=obj.iteration_start,
            cb=obj.callback,
            **_inner_lstsq_kwargs(obj),
        )

    # state['last_it'] tracks the global iteration index so xs_fixb snapshots
    # don't collide and the budget spans all outer cycles. The wrapped callback
    # also captures the latest step_norm / cost / cost_red so the stall test can
    # read them after the inner solve returns.
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
        f"  max_niter={total_iters}  inner_iter_cap={p.inner_iter_cap}\n"
        f"  method={p.method}  x_scale={p.x_scale}"
        + (f" -> {p.x_scale2} @iter{p.switch_iter}\n" if p.x_scale2 is not None else "\n")
        + f"  loss={p.loss}\n"
        f"  step_tol={p.step_tol}  cost_red_frac={p.cost_red_frac}\n"
        f"  freeb_bnorm_tol={p.freeb_bnorm_tol}\n"
    )

    while state["last_it"] < obj.iteration_start + total_iters:
        remaining = obj.iteration_start + total_iters - state["last_it"]
        cap = min(p.inner_iter_cap, remaining)
        cycle_start = state["last_it"]
        max_niter = cycle_start + cap
        cycle_x_scale = _x_scale_for_cycle(p, cycle_start)

        print(
            f"\n--- outer cycle {outer} (frozen iBns) iters {cycle_start}->{max_niter} "
            f"x_scale={cycle_x_scale} ---"
        )

        # INNER: minimize force balance at FROZEN iBns. The Jacobian is valid
        # throughout because iBns does not move inside this call.
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
        # extrapolation (virtual casing) is taken at the right geometry. This
        # f-eval does NOT itself refresh iBns.
        f(x, obj)

        # Refresh iBns once (damped by the TOML gBnbld). The only place iBns moves.
        bnserr = obj.refresh_bns()
        bnserr_hist.append(bnserr)

        inner_converged = r.status in (1, 2, 3)  # gtol / ftol / xtol
        inner_stalled = (
            state["step_norm"] is not None
            and state["step_norm"] < p.step_tol
            and (
                state["cost_red"] is None
                or abs(state["cost_red"]) < p.cost_red_frac * max(state["cost"], 1e-300)
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

        # Safety: if the damped Picard map isn't contracting, warn (nudging
        # gbnbld up toward 1, i.e. less aggressive replacement, usually helps).
        if len(bnserr_hist) >= 2 and bnserr_hist[-1] > bnserr_hist[-2]:
            print(
                f"  WARNING: bnserr grew ({bnserr_hist[-2]:.3e} -> "
                f"{bnserr_hist[-1]:.3e}); iBns map may be non-contractive. "
                f"Consider a larger gbnbld."
            )

        outer += 1

    return r


def optimizer_inloop(obj, x_init, f, jac):
    """Prototype: single solve, refresh iBns mid-run when step_norm < step_tol."""
    p = obj.min_params
    total_iters = p.max_niter

    if not obj.freeb_mode:
        return lstsq(
            f,
            x_init,
            jac=jac,
            max_niter=total_iters,
            iteration_start=obj.iteration_start,
            cb=obj.callback,
            **_inner_lstsq_kwargs(obj),
        )

    print(
        f"\n=== FREE-BOUNDARY IN-LOOP TRIGGER (prototype) ===\n"
        f"  max_niter={total_iters}  step_tol={p.step_tol}\n"
        f"  method={p.method}  x_scale={p.x_scale}  loss={p.loss}\n"
        "  NOTE: refreshing iBns mid-solve moves the residual surface under a\n"
        "  Jacobian/f computed at the old iBns — expected to be less robust\n"
        "  than the outer/inner loop. Here for comparison only.\n"
    )

    def cb_wrap(it, nfev, cost, cost_red, stp_nrm, opt, x, o):
        obj.callback(it, nfev, cost, cost_red, stp_nrm, opt, x, o)
        if stp_nrm is not None and stp_nrm < p.step_tol:
            bnserr = obj.refresh_bns()
            print(
                f"  [in-loop] step_norm {stp_nrm:.2e} < {p.step_tol:.2e} "
                f"→ refreshed iBns, bnserr={bnserr:.3e}"
            )

    return lstsq(
        f,
        x_init,
        jac=jac,
        max_niter=obj.iteration_start + total_iters,
        iteration_start=obj.iteration_start,
        cb=cb_wrap,
        **_inner_lstsq_kwargs(obj),
    )


def optimizer(obj, x_init, f, jac):
    """Dispatch to the outer/inner-loop or in-loop optimizer per min_params.bns_trigger."""
    chosen = optimizer_outer if obj.min_params.bns_trigger == "outer" else optimizer_inloop
    return chosen(obj, x_init, f, jac)


obj = Force_Minimizer(
    args.inputfile,
    optimizer=optimizer,
    callback=callback_metrics,
)

obj.minimize()
obj.postprocess(save_json=False, save_pkl=False)

if obj.rank == 0:
    print(f"SPECTRE free-boundary equilibrium done (bns-trigger={obj.min_params.bns_trigger})!")
