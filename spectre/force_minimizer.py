# SPDX-FileCopyrightText: 2026 Proxima Fusion
# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

import numpy as np
from os import system
from pathlib import Path
from shutil import rmtree
from time import time
from glob import glob

from spectre import SPECTRE
from spectre.utils import (
    Timer,
    get_xinit_specwrap,
    get_freal_fsa_metric,
    save_errors_to_h5,
)
from spectre.input_dict import input_dict
from spectre.least_squares.least_squares import least_squares as lstsq
from spectre.error_checking import print_all_errors
from spectre.force_targets import force_real

from signal import signal, SIGUSR1


def _lstsq_setup(obj):
    """Build the solver configuration shared by every force solve of a run.

    Selecting the trust-region solver and wiring up the lagged Jacobian-vector-product
    mode are independent of how many solves the run is split into, so both the plain
    optimizer and the free-boundary outer/inner driver configure themselves here.

    Args:
        obj: The ``Force_Minimizer`` driving the solve.

    Returns:
        Tuple ``(params, kwargs)``: the resolved minimization parameters, and the
        least-squares keyword arguments common to every solve. The caller supplies the
        per-solve ``max_niter``, ``iteration_start``, ``cb`` and ``delta_prev``.
    """
    params = (
        input_dict(
            dict(
                gtol=1e-13, xtol=1e-22, ftol=1e-13, max_nfev=200, method="dogbox", rcond_lstsq=None
            )
        )
        if obj.min_params is None
        else obj.min_params
    )

    tr_solver = params.tr_solver
    tr_options = {}
    if tr_solver == "banded":
        # The banded solver relies on the block-tridiagonal interface structure of the
        # force Jacobian: pure force rows, one equal column block per interface.
        if params.objective != "freal" or params.inface_repr == "henn":
            print(
                "WARNING: tr_solver='banded' requires objective='freal' and a per-interface "
                "DOF representation; using exact SVD."
            )
            tr_solver = "exact"
        else:
            tr_options = {"n_blocks": int(obj.field_obj.allglobal_mod.mvol) - 1}

    jvp_lag = params.jvp_lag
    if jvp_lag and tr_solver != "banded":
        print("WARNING: min_params.jvp_lag requires tr_solver='banded'; ignored.")
    elif jvp_lag:
        # Lagged-banded matrix-free iteration: reuse the dense Jacobian (and its
        # banded factorization) for up to jvp_lag iterations, refreshing the model
        # each iterate with a small batch of exact Jacobian-vector products.
        # Lagging is valid in Picard free-boundary mode too: iBns is fixed for a whole
        # inner solve, so the residual there is a fixed function of x, and the lagged
        # subspace is rebuilt on entry to each solve -- exactly where iBns moves.
        mpol = int(obj.field_obj.input_list_mod.mpol)
        ntor = int(obj.field_obj.input_list_mod.ntor)
        nfp = int(obj.field_obj.input_list_mod.nfp)
        if params.inface_repr == "fourier" and (
            (params.mpol_rz is not None and params.mpol_rz < mpol)
            or (params.ntor_rz is not None and params.ntor_rz < ntor * nfp)
        ):
            print(
                "WARNING: jvp_lag needs the full perturbation spectrum "
                "(the JVP has no mpol_rz/ntor_rz truncation); ignored."
            )
        else:
            from spectre.force_targets import force_real_jvp

            # Share the Beltrami factorizations between each force evaluation and the JVP
            # batch that follows it (they are refactored from scratch otherwise -- the
            # dominant fixed cost of a batch). Costs resident memory for the dense factors,
            # hence opt-in with the lagged mode rather than always on.
            obj.field_obj.allglobal_mod.lbeltrami_factor_cache = True

            def jvp_func(xdof, dirs):
                t = time()
                D = obj.Dxmn_Dxdof_at_xdof(xdof, obj.field_obj)  # None = identity
                dirs_mn = np.asfortranarray(dirs if D is None else D @ dirs)
                if obj.comm.Get_size() > 1:
                    # force_real_jvp is collective: hand the workers the position and the
                    # (already mn-space) direction batch, then evaluate together.
                    _send_int_array(xdof, 3, obj.comm)
                    for r in range(1, obj.comm.Get_size()):
                        obj.comm.Send([np.array(dirs_mn.shape[1], dtype=np.int32), MPI_INT], dest=r)
                        obj.comm.Send([dirs_mn, MPI_DOUBLE], dest=r)
                jvp = force_real_jvp(
                    obj.xdof_to_xmn(xdof, obj.field_obj),
                    dirs_mn,
                    obj.field_obj,
                    nt=params.nt_dbb,
                    nz=params.nz_dbb,
                    filter_stellsym=params.filter_stellsym,
                ).reshape(-1, dirs.shape[1])
                print(f"fjvp: time =  {time() - t:.2e}     k    {dirs.shape[1]}")
                return jvp

            tr_options["jvp"] = jvp_func
            tr_options["lag"] = int(jvp_lag)

    return params, dict(
        method=params.method,
        verbose=2,
        args=[obj],
        gtol=params.gtol,
        xtol=params.xtol,
        ftol=params.ftol,
        max_nfev=params.max_nfev,
        rcond_lstsq=params.rcond_lstsq,
        tr_solver=tr_solver,
        tr_options=tr_options,
    )


def optimizer_default(obj, x_init, f, jac):
    """Minimize the force residual with a single least-squares solve."""
    params, kwargs = _lstsq_setup(obj)

    return lstsq(
        f,
        x_init,
        jac=jac,
        max_niter=params.max_niter,
        iteration_start=obj.iteration_start,
        cb=obj.callback,
        **kwargs,
    )


def optimizer_picard(obj, x_init, f, jac):
    """Minimize the force residual with iBns frozen inside each inner solve.

    The free-boundary problem couples two conditions: force balance at a given boundary
    normal field, and consistency of that normal field with the plasma it bounds. This
    driver solves them in alternation. Each inner least-squares solve runs at a *fixed*
    ``iBns``, so the Jacobian it builds describes the residual surface it is actually
    descending; ``iBns`` then moves once, between solves. Refreshing it more often --
    per objective evaluation, say -- blends it toward the extrapolation from rejected
    trial points and pulls the residual surface out from under the Jacobian.

    These cycles are the *only* Picard stepping in the run: ``_disable_fortran_picard``
    switches off the loop inside Fortran ``spectre()`` so that ``run()`` cannot slip an
    extra, unscored step in on either side of the minimization.

    Fixed-boundary runs and the direct Laplace-virtual-casing solve (``lvcvacuum=1``,
    which never reads ``iBns``) need no alternation and take a single solve.

    Args:
        obj: The ``Force_Minimizer`` driving the solve.
        x_init: Starting DOF vector.
        f: Objective function.
        jac: Jacobian function.

    Returns:
        The last inner solve's optimizer result.
    """
    params, kwargs = _lstsq_setup(obj)
    total_iters = params.max_niter  # budget across all cycles, not per cycle

    if not obj.picard_freeb:
        return lstsq(
            f,
            x_init,
            jac=jac,
            max_niter=total_iters,
            iteration_start=obj.iteration_start,
            cb=obj.callback,
            **kwargs,
        )

    # last_it is the global iteration index, so snapshots do not collide across cycles
    # and the budget spans the whole run. The wrapper also captures the quantities the
    # stall test needs once the inner solve has returned.
    state = {
        "last_it": obj.iteration_start,
        "step_norm": None,
        "cost": None,
        "cost_red": None,
    }

    def cb_wrap(it, nfev, cost, cost_red, stp_nrm, opt, x, o):
        obj.callback(it, nfev, cost, cost_red, stp_nrm, opt, x, o)
        state["last_it"] = it
        state["step_norm"] = stp_nrm
        state["cost"] = cost
        state["cost_red"] = cost_red

    print(
        f"\n=== FREE-BOUNDARY OUTER/INNER LOOP ===\n"
        f"  max_niter={total_iters}  inner_iter_cap={params.inner_iter_cap}\n"
        f"  method={params.method}  tr_solver={kwargs['tr_solver']}\n"
        f"  step_tol={params.step_tol}  cost_red_frac={params.cost_red_frac}\n"
        f"  freeb_bnorm_tol={params.freeb_bnorm_tol}"
        f"  gbnbld={float(obj.field_obj.input_list_mod.gbnbld):.4f}\n"
    )

    if bool(obj.field_obj.allglobal_mod.update_bnorm_inbeltrami):
        print(
            "  WARNING: update_bnorm_inbeltrami is set, so iBns moves inside every solve. "
            "That voids the frozen-iBns premise this loop relies on."
        )

    bnserr_hist = []
    x = x_init
    r = None
    outer = 0
    prev_tr = None  # final trust radius of the previous inner solve
    no_progress = 0  # consecutive cycles whose inner solve ran zero iterations

    while state["last_it"] < obj.iteration_start + total_iters:
        cycle_start = state["last_it"]
        remaining = obj.iteration_start + total_iters - cycle_start
        cap = min(params.inner_iter_cap, remaining)

        print(
            f"\n--- outer cycle {outer} (frozen iBns) iters {cycle_start}->{cycle_start + cap} ---"
        )

        # The inner solve resumes at the trust radius the previous cycle ended on, so it
        # does not spend its first iterations re-discovering a step size that already
        # worked.
        r = lstsq(
            f,
            x,
            jac=jac,
            max_niter=cycle_start + cap,
            iteration_start=cycle_start,
            cb=cb_wrap,
            delta_prev=prev_tr,
            **kwargs,
        )
        x = r.x
        prev_tr = r.tr_radius

        if r.status == 5:  # SIGUSR1 abort
            print("=== Aborting (SIGUSR1) ===")
            break

        # Take the virtual-casing extrapolation at the accepted geometry. The solve
        # normally ends with the field evaluated there, but a run that stopped right
        # after a rejected trial leaves it elsewhere.
        if obj._field_at is not None and np.array_equal(obj._field_at, x):
            print("field already at accepted x -- skipping re-evaluation")
        else:
            f(x, obj)

        # bnserr is measured before this refresh applies its blend, so it grades the iBns
        # the cycle that just finished was scored against, not the one the next cycle gets.
        bnserr = obj.refresh_bns()
        bnserr_hist.append(bnserr)

        inner_converged = r.status in (1, 2, 3)  # gtol / ftol / xtol
        inner_stalled = (
            state["step_norm"] is not None
            and state["step_norm"] < params.step_tol
            and (
                state["cost_red"] is None
                or abs(state["cost_red"]) < params.cost_red_frac * max(state["cost"], 1e-300)
            )
        )
        print(
            f"--- cycle {outer} done: status={r.status} cost={r.cost:.4e} "
            f"bnserr={bnserr:.3e}  inner_converged={inner_converged} "
            f"inner_stalled={inner_stalled} ---"
        )

        # Whole-problem convergence: force balance solved AND the iBns it was solved at
        # already self-consistent, so the refresh above barely moved it.
        if bnserr < params.freeb_bnorm_tol and (inner_converged or inner_stalled):
            print("=== Converged: bnserr below tol and inner force solve settled ===")
            break

        # A cycle whose inner solve ran no iterations does not consume the budget, so
        # without this the loop could spin on f0/jac0 evaluations forever.
        if state["last_it"] == cycle_start:
            no_progress += 1
            if no_progress >= 2:
                print(
                    "=== Stopping: two consecutive cycles ran zero inner iterations while "
                    "bnserr stayed above freeb_bnorm_tol. The inner solve keeps terminating "
                    "immediately (check gtol) before iBns is self-consistent "
                    "(check gbnbld). ==="
                )
                break
        else:
            no_progress = 0

        if len(bnserr_hist) >= 2 and bnserr_hist[-1] > bnserr_hist[-2]:
            print(
                f"  WARNING: bnserr grew ({bnserr_hist[-2]:.3e} -> {bnserr_hist[-1]:.3e}); "
                f"the iBns map may not be contracting. Consider a larger gbnbld."
            )

        outer += 1

    return r


def callback_default(it, nfev, cost, cost_red, stp_nrm, opt, x, obj):

    freal = np.sqrt(np.mean(obj.field_obj.f_real_shpd**2))
    fsa = get_freal_fsa_metric(obj.field_obj)

    obj.iter_iter[it] = it
    obj.iter_nfev[it] = nfev
    obj.iter_cost[it] = cost
    obj.iter_cost_red[it] = cost_red if cost_red is not None else -1.0
    obj.iter_step_norm[it] = stp_nrm if stp_nrm is not None else -1.0
    obj.iter_opt[it] = opt
    obj.iter_fsa[it] = fsa
    obj.iter_freal[it] = freal

    cost_red = " " * 8 if cost_red is None else f"{cost_red:.2e}"
    step_norm = " " * 8 if stp_nrm is None else f"{stp_nrm:.2e}"
    opt = " " * 5 if opt is None else f"{opt:.2e}"

    print(
        f"\niter {it:5}   nfev {nfev:6}    cost {cost:.3e}   <f> {fsa:.5e}   fr {freal:.4e}  stpnrm {step_norm}   costred {cost_red}   optim {opt}"
    )


def _send_int_array(arr, i, comm):
    # sends an integer and an array from rank 0 to other ranks
    # called from rank 0
    for r in range(1, comm.Get_size()):
        comm.Send([np.array(i, dtype=np.int32), MPI_INT], dest=r)
        comm.Send([arr, MPI_DOUBLE], dest=r)


def _recv_int_array(self):
    # recive and return an integer and an array
    # called from ranks > 0
    arr = self.x_curr
    comm_Recv = self.comm.Recv
    i = np.empty(1, dtype=np.int32)
    comm_Recv(i, source=0)
    comm_Recv(arr, source=0)
    return arr, i[0]


class Force_Minimizer:
    def __init__(
        self, filename, optimizer=optimizer_default, callback=callback_default, save_x_iter=True
    ):
        if filename[-5:] != ".toml":
            raise RuntimeError("Input file must be a .toml file")

        self.field_obj = SPECTRE(filename)

        self.rank = self.field_obj.comm.rank
        self.comm = self.field_obj.comm
        self.name = self.field_obj.name

        if self.rank == 0:
            self.field_obj.print_inputs()

        mpol = self.field_obj.input_list_mod.mpol
        ntor = self.field_obj.input_list_mod.ntor
        nfp = self.field_obj.input_list_mod.nfp

        self.min_params = self.field_obj.min_params

        mpol_rz = self.min_params.mpol_rz if self.min_params.mpol_rz is not None else mpol
        ntor_rz = (self.min_params.ntor_rz if self.min_params.ntor_rz is not None else ntor) * nfp

        if self.min_params.inface_repr == "fourier":
            from spectre.interface_dof import identity_dof

            xmn_to_xdof, xdof_to_xmn, Dxmn_Dxdof_at_xdof = identity_dof()
        elif self.min_params.inface_repr == "fourier red":
            from spectre.interface_dof import reduced_rzmn_dof

            xmn_to_xdof, xdof_to_xmn, Dxmn_Dxdof_at_xdof = reduced_rzmn_dof(mpol_rz, ntor_rz)
            self.min_params.mpol_rz = mpol_rz
            self.min_params.ntor_rz = ntor_rz
        elif self.min_params.inface_repr == "henn":
            from spectre.interface_dof import henneberg_dof

            xmn_to_xdof, xdof_to_xmn, Dxmn_Dxdof_at_xdof = henneberg_dof(
                alpha_fac=1,
                nmax=self.min_params.nmax,
                mmax=self.min_params.mmax,
                ntheta=64,
                nphi=64,
            )

        self.xmn_to_xdof = xmn_to_xdof
        self.xdof_to_xmn = xdof_to_xmn
        self.Dxmn_Dxdof_at_xdof = Dxmn_Dxdof_at_xdof

        self.freeb_mode = self.field_obj.input_list_mod.lfreebound
        # Whether the run needs the outer/inner Picard driver: iBns has to be brought to
        # self-consistency by repeated virtual-casing updates. With the direct
        # Laplace-virtual-casing solve (lvcvacuum=1) every field solve is already
        # self-consistent and iBns is never read, so a single inner solve is the whole job.
        self.picard_freeb = bool(self.freeb_mode) and self.field_obj.input_list_mod.lvcvacuum == 0

        self._disable_fortran_picard()

        self._set_init_iter()

        self.field_obj.input_list_mod.nppts = 0  # disable poincare at this stage since it's slow
        self.field_obj.run(save_output=True, infaces=self.infaces_prev)

        self.optimizer = optimizer

        if save_x_iter:

            def callback_temp(it, nfev, cost, cost_red, stp_nrm, opt, x, obj):
                callback(it, nfev, cost, cost_red, stp_nrm, opt, x, obj)
                np.save(f"xs_fixb/x_{obj.iter_freeb:03}_{it:04}.npy", xdof_to_xmn(x, obj.field_obj))
                # print_all_errors(obj.field_obj, beltrami=True, force=True)
        else:
            callback_temp = callback
        self.callback = callback_temp

        # Define objective function (and its Jacobian)
        if self.min_params.objective == "freal":
            from spectre.force_targets import freal_obj

            obj_func, jac_func = freal_obj(self.min_params)
        elif self.min_params.objective == "freal,len":
            from spectre.force_targets import freal_len_obj

            obj_func, jac_func = freal_len_obj(self.min_params, nfp)

        def obj_func_wrap(xdof, test):
            t = time()
            f = obj_func(xdof_to_xmn(xdof, test), test)
            if test.comm.Get_rank() == 0:
                print(f"f: time =  {time() - t:.2e}     val  {np.sum(f**2):.3e}")
            return f

        self.obj_func = obj_func_wrap

        def jac_func_wrap(xdof, test):
            t = time()
            fjac = jac_func(xdof_to_xmn(xdof, test), test)
            D = Dxmn_Dxdof_at_xdof(xdof, test)  # None = identity
            if D is not None:
                fjac = fjac @ D
            if test.comm.Get_rank() == 0:
                print(f"fjac: time =  {time() - t:.2e}     val  {np.sum(fjac**2):.3e}")
            return fjac

        self.jac_func = jac_func_wrap

        self.iter_freeb = 0

        from mpi4py import MPI

        global MPI_INT, MPI_DOUBLE
        MPI_INT = MPI.INT
        MPI_DOUBLE = MPI.DOUBLE

        # Geometry the Fortran field was last solved at, in DOF layout, or None when the
        # solution no longer corresponds to any position (an iBns refresh invalidates it).
        # Lets the driver skip a redundant force evaluation after an inner solve.
        self._field_at = None

        # jvp_lag's JVP batches are collective (force_real_jvp runs on every rank), so the
        # Beltrami factor cache they share with the force evaluations must be enabled on every
        # rank -- optimizer_default only runs on rank 0.
        if self.min_params.jvp_lag and self.min_params.tr_solver == "banded":
            self.field_obj.allglobal_mod.lbeltrami_factor_cache = True

        # Bootstrap iBns from the plasma the inputs actually describe, for cases whose Bns
        # namelist does not correspond to the starting geometry (e.g. a fresh fixed- to
        # free-boundary conversion). Collective: every rank runs it, the worker loop has
        # not started yet.
        if self._bootstrap_bns:
            self.field_obj.update_bns(mode="initialize", verbose=True)

        x_init = get_xinit_specwrap(self.field_obj)
        self.x_init = self.xmn_to_xdof(x_init, self.field_obj)

        if self.rank == 0:
            print(f"Interface DOFs:\nxmn {x_init.shape}\nxdof {self.x_init.shape}")

        self.f_init = self.obj_func(self.x_init, self.field_obj)

        self.x_curr = self.x_init.copy()  # buffer for mpi comms

        print_all_errors(self.field_obj, beltrami=True, force=True)

        # wrap obj_func, jac functions
        if self.rank == 0:
            obj_func2 = self.obj_func
            jac_func2 = self.jac_func

            # iBns is frozen for the whole inner force solve: an objective evaluation only
            # signals a force eval. Refreshing it here instead would blend iBns toward the
            # extrapolation from rejected trust-region trial points, moving the residual
            # surface under a Jacobian built once per accepted iteration. The driver
            # refreshes it between inner solves via refresh_bns().
            def obj_func_mod(x, obj):
                _send_int_array(x, 0, obj.comm)
                obj._field_at = np.asarray(x).copy()
                return obj_func2(x, obj.field_obj)

            def jac_func_mod(x, obj):
                _send_int_array(x, 1, obj.comm)
                obj._field_at = np.asarray(x).copy()
                return jac_func2(x, obj.field_obj)

            self.obj_func = obj_func_mod
            self.jac_func = jac_func_mod

            if self.iteration_start == 0:
                for d in ("xs_fixb", "xs_freeb"):
                    rmtree(d, ignore_errors=True)

                if self.min_params.save_xin_fixb:
                    Path("xs_fixb").mkdir(parents=True, exist_ok=True)

                if self.min_params.save_xin_freeb:
                    Path("xs_freeb").mkdir(parents=True, exist_ok=True)

            # The outer/inner driver keeps counting from iteration_start on a resumed run,
            # so the history has to span the whole global range, not just max_niter rows.
            n_hist = self.iteration_start + self.min_params.max_niter + 1
            self.iter_iter = [-1.0] * n_hist
            self.iter_nfev = [-1.0] * n_hist
            self.iter_cost = [-1.0] * n_hist
            self.iter_cost_red = [-1.0] * n_hist
            self.iter_step_norm = [-1.0] * n_hist
            self.iter_opt = [-1.0] * n_hist
            self.iter_fsa = [-1.0] * n_hist
            self.iter_freal = [-1.0] * n_hist

            print("Initialization done!")

    def minimize(self):
        if self.rank == 0:
            with Timer() as t:
                if self.freeb_mode:
                    print("SPECTRE Free-boundary force minimization")

                    self.result = self.optimizer(self, self.x_init, self.obj_func, self.jac_func)

                else:
                    print("SPECTRE Fixed-boundary force minimization...")

                    self.result = self.optimizer(self, self.x_init, self.obj_func, self.jac_func)

            _send_int_array(self.x_init, 9, self.comm)

            self.t = t

            print(f"Minimization done ({t.readout})!")

        else:
            # Set up the signal catcher for other ranks to avoid termination. Master takes care of the rest.
            def handler_stop_signals(*args):
                pass

            signal(SIGUSR1, handler_stop_signals)

            while True:
                x_temp, flag = _recv_int_array(self)

                if flag == 0:
                    # calculate the function
                    self.obj_func(x_temp, self.field_obj)
                elif flag == 1:
                    # calculate the jacobian
                    self.jac_func(x_temp, self.field_obj)
                elif flag == 2:
                    # join the collective B_norm refresh
                    self.field_obj.update_bns(mode="refresh")
                elif flag == 3:
                    # join a collective Jacobian-vector-product batch
                    self._jvp_worker(x_temp)
                elif flag == 9:
                    # finish running
                    break

    def _disable_fortran_picard(self):
        """Hand the Picard iteration over ``iBns`` to the outer/inner driver, exclusively.

        Fortran ``spectre()`` carries its own free-boundary loop, which steps ``iBns`` on
        *every* ``run()`` -- the one this constructor makes, and the one ``postprocess``
        makes. Those steps are taken behind the optimizer's back: the geometry was never
        scored against the ``iBns`` they produce, so the run ends on a state no force
        evaluation ever saw, and the ``_end.toml``/HDF5 it writes describe that state
        rather than the optimized one. With ``lautoinitbn=1`` the step is undamped
        (``gBnbld = 0``) on top of that, which throws away the damping the driver relies on.

        Switching both off leaves ``SPECTRE.update_bns`` -- the Python interface -- as the
        only thing that moves ``iBns``: once here as the bootstrap ``lautoinitbn`` asked
        for, and once per outer cycle from ``refresh_bns``. It also makes the written
        ``_end.toml`` a faithful restart file, since its ``bns`` is then the run's own and
        it carries the flags that stop a reload from overwriting it.
        """
        # initialize_bns_from_current is the Python spelling of what lautoinitbn asks the
        # Fortran loop to do, so honour either.
        self._bootstrap_bns = bool(self.min_params.initialize_bns_from_current)

        if not self.picard_freeb:
            return

        lautoinitbn = int(self.field_obj.input_list_mod.lautoinitbn)
        mfreeits = int(self.field_obj.input_list_mod.mfreeits)
        self._bootstrap_bns = self._bootstrap_bns or lautoinitbn == 1

        self.field_obj.input_list_mod.lautoinitbn = 0
        self.field_obj.input_list_mod.mfreeits = 0

        if self.rank == 0 and (lautoinitbn == 1 or mfreeits != 0):
            print(
                f"iBns: taking over the Picard iteration from spectre() "
                f"(lautoinitbn {lautoinitbn}->0, mfreeits {mfreeits}->0). "
                f"It now moves only through update_bns"
                + (", starting with one 'initialize' bootstrap." if self._bootstrap_bns else ".")
            )

    def refresh_bns(self):
        """Refresh the free-boundary normal field on every rank, once.

        This is the only place ``iBns`` moves during a free-boundary minimization: it
        belongs *between* inner force solves, never inside one, so that the Jacobian a
        solve is built on stays valid for that solve's whole residual surface.
        ``_disable_fortran_picard`` makes that exclusivity real.

        Returns:
            The ``bnserr`` of the ``iBns`` the last force solve used, on rank 0; ``nan`` on
            the peers. Measured before the update, so it is the residual the completed
            inner solve was actually scored against, independent of ``gbnbld``.
        """
        if self.rank != 0:
            return float("nan")  # peers refresh through the flag-2 listener

        _send_int_array(self.x_curr, 2, self.comm)
        bnserr = self.field_obj.update_bns(mode="refresh", verbose=True)
        # iBns moved, so the solved field no longer corresponds to the geometry it was
        # solved at: force the driver to re-evaluate before using it.
        self._field_at = None
        return bnserr

    def _jvp_worker(self, xdof):
        """Worker side of a jvp_lag batch: receive the direction batch, join the collective JVP.

        Args:
            xdof: Position (DOF layout) the batch is evaluated at, as received from rank 0.
        """
        from spectre.force_targets import force_real_jvp

        k = np.empty(1, dtype=np.int32)
        self.comm.Recv(k, source=0)
        ngdof = int(self.field_obj.allglobal_mod.ngdof)
        dirs = np.empty((ngdof, int(k[0])), dtype=np.float64, order="F")
        self.comm.Recv(dirs, source=0)

        force_real_jvp(
            self.xdof_to_xmn(xdof, self.field_obj),
            dirs,
            self.field_obj,
            nt=self.min_params.nt_dbb,
            nz=self.min_params.nz_dbb,
            filter_stellsym=self.min_params.filter_stellsym,
        )

    def postprocess(self, save_pkl=True, save_json=True, run_spec=True):
        field_obj = self.field_obj

        if self.rank == 0:
            result = self.result

            result.pop("jac")
            result.time_end = self.t.readout

            if save_pkl:
                result.save_pickle(f"{self.name}_res.pkl")
            if save_json:
                result.save_json(f"{self.name}_res.json")

            xdof_final = result.x

            self.x_final = self.xdof_to_xmn(xdof_final, field_obj)

            _send_int_array(xdof_final, 0, self.comm)
        else:
            xdof_final, _ = _recv_int_array(self)

        x_final = self.xdof_to_xmn(xdof_final, field_obj)

        if run_spec:
            field_obj.input_list_mod.linitialize = 0
            field_obj.input_list_mod.nppts = 0

            self.field_obj.allglobal_mod.filename_h5 = f"{field_obj.name}_opt.h5"
            field_obj.run(save_output=True)

            if self.rank == 0:
                system(f"mv {field_obj.name}_end.toml {field_obj.name}_opt.toml")

        force_real(
            x_final,
            self.field_obj,
            nt=self.min_params.nt_dbb,
            nz=self.min_params.nz_dbb,
            filter_stellsym=self.min_params.filter_stellsym,
        )

        fsa_err = get_freal_fsa_metric(field_obj)

        if self.rank == 0:
            print(f"\n  <f> = {fsa_err:.8e}")

        beltrami_errors_avg, beltrami_errors_max, f_errs_rms, f_errs_max = print_all_errors(
            field_obj, beltrami=True, force=True
        )

        if self.rank == 0:
            save_errors_to_h5(
                field_obj.allglobal_mod.filename_h5,
                fsa_err,
                beltrami_errors_avg,
                beltrami_errors_max,
                f_errs_rms,
                f_errs_max,
            )

            self._save_iter_info_to_h5()

    def _set_init_iter(self):

        inface_files = sorted(glob("xs_fixb/*"))

        # If previous iterations exist, start from the last one
        if inface_files and self.min_params.resume:
            infaces_prev = np.load(inface_files[-1])
            ngdof = int(self.field_obj.allglobal_mod.ngdof)
            if len(infaces_prev) == ngdof + 1:  # legacy checkpoint with the dead slot 0
                infaces_prev = infaces_prev[1:]
            if self.rank == 0:
                print(f"Starting from previous interfaces {inface_files[-1]}")
        else:
            infaces_prev = None

        iteration_start = 0

        if (
            self.min_params.max_niter < 10000 and self.min_params.resume
        ):  # max_niter is set not to default
            num_infaces = len(inface_files)
            if num_infaces > 0:
                iteration_start = num_infaces - 1
                if num_infaces >= self.min_params.max_niter:
                    if self.rank == 0:
                        print("Number of iterations already reached, quitting...")
                    exit()

        self.infaces_prev = infaces_prev
        self.iteration_start = iteration_start

    def _save_iter_info_to_h5(self):
        from h5py import File

        with File(self.field_obj.allglobal_mod.filename_h5, "a") as f:
            grp_errs = f.create_group("iterations")

            grp_errs.create_dataset("iter", data=self.iter_iter)
            grp_errs.create_dataset("nfev", data=self.iter_nfev)
            grp_errs.create_dataset("cost", data=self.iter_cost)
            grp_errs.create_dataset("cost_red", data=self.iter_cost_red)
            grp_errs.create_dataset("step_norm", data=self.iter_step_norm)
            grp_errs.create_dataset("opt", data=self.iter_opt)
            grp_errs.create_dataset("fsa", data=self.iter_fsa)
            grp_errs.create_dataset("freal", data=self.iter_freal)

    def run_repeat_func(self, num=20, which="f"):

        if which == "f":
            func = self.obj_func
        elif which == "jac":
            func = self.jac_func
        else:
            raise RuntimeError("which must be either 'f' or 'jac'")

        if self.rank == 0:
            with Timer() as t:
                for i in range(num):
                    func(self.x_curr, self)

            print(f"{which} loop done, time per iteration: {t.time / num:.4f} s")

            _send_int_array(self.x_init, 9, self.comm)

        else:
            while True:
                x_temp, flag = _recv_int_array(self)

                if flag == 0:
                    # calculate the function
                    self.obj_func(x_temp, self.field_obj)
                elif flag == 1:
                    # calculate the jacobian
                    self.jac_func(x_temp, self.field_obj)
                elif flag == 2:
                    # join the collective B_norm refresh
                    self.field_obj.update_bns(mode="refresh")
                elif flag == 9:
                    # finish running
                    break
