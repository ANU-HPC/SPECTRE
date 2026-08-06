# SPDX-FileCopyrightText: 2026 Proxima Fusion
# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

from enum import IntEnum
from glob import glob
from os import system
from signal import SIGUSR1, signal
from time import time

import numpy as np

from spectre import SPECTRE
from spectre.error_checking import print_all_errors
from spectre.input_dict import input_dict
from spectre.least_squares.lagged import JvpResult
from spectre.least_squares.least_squares import least_squares as lstsq
from spectre.utils import (
    Timer,
    get_freal_fsa_metric,
    get_xinit_specwrap,
    save_errors_to_h5,
)


class _WorkerCommand(IntEnum):
    """Messages sent by rank 0 to the minimizer worker loop."""

    FORCE = 0
    JACOBIAN = 1
    REFRESH_BNS = 2
    JVP = 3
    CACHE_ENTER = 4
    CACHE_CLEAR = 5
    CACHE_EXIT = 6
    STOP = 9


class _BeltramiFactorCacheScope:
    """Own a temporary all-rank factor-cache policy and rank-local storage."""

    def __init__(self, field_obj, worker_command=None):
        self._field_obj = field_obj
        self._worker_command = worker_command
        self._cache_was_on = None
        self._active = False

    def __enter__(self):
        if self._active:
            raise RuntimeError("Beltrami factor-cache scope is already active")
        self._cache_was_on = bool(self._field_obj.allglobal_mod.lbeltrami_factor_cache)
        if self._worker_command is not None:
            self._worker_command(_WorkerCommand.CACHE_ENTER)
        self._field_obj.allglobal_mod.lbeltrami_factor_cache = True
        self._active = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if not self._active:
            return False
        try:
            if self._worker_command is not None:
                self._worker_command(_WorkerCommand.CACHE_EXIT)
        finally:
            try:
                self._clear_local()
            finally:
                self._field_obj.allglobal_mod.lbeltrami_factor_cache = self._cache_was_on
                self._active = False
        return False

    def clear(self):
        """Clear every rank's manager while keeping the cache scope enabled."""
        if not self._active:
            raise RuntimeError("Beltrami factor-cache scope is not active")
        try:
            if self._worker_command is not None:
                self._worker_command(_WorkerCommand.CACHE_CLEAR)
        finally:
            self._clear_local()

    def _clear_local(self):
        """Release all factors owned by this rank's process-local manager."""
        self._field_obj.lib.beltrami_factor_cache_mod.factor_cache_clear()


def _make_beltrami_factor_cache_scope(field_obj, worker_command=None):
    """Construct the replaceable cache scope used by the lagged optimizer."""
    return _BeltramiFactorCacheScope(field_obj, worker_command)


def _lstsq_setup(obj, x_init, f, jac):
    """Resolve the solver options shared by the single-solve and outer/inner drivers.

    Returns:
        tuple: ``(params, kwargs, cache_scope)`` -- the minimization parameters, the
        keyword arguments for ``lstsq`` (including the possibly wrapped Jacobian), and
        the Beltrami factor-cache scope to hold open around the solve(s), or ``None``
        when lagging is off.
    """
    params = (
        input_dict(
            dict(gtol=1e-13, xtol=1e-22, ftol=1e-13, max_nfev=200, method="trf", rcond_lstsq=None)
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
    cache_scope = None
    jac_used = jac
    if jvp_lag and tr_solver != "banded":
        print("WARNING: min_params.jvp_lag requires tr_solver='banded'; ignored.")
    elif jvp_lag and params.method != "trf":
        # dogbox uses the banded kernel only for its Gauss-Newton step; the lagged
        # reduced-model iteration exists in trf alone.
        print("WARNING: min_params.jvp_lag requires method='trf'; ignored.")
    elif jvp_lag:
        # Lagged-banded matrix-free iteration: reuse the dense Jacobian (and its
        # banded factorization) for up to jvp_lag iterations, refreshing the model
        # each iterate with a small batch of exact Jacobian-vector products.
        # Lagging is valid in the Picard free-boundary mode too: iBns is frozen for a
        # whole inner solve, so the residual there is a fixed function of x, and the
        # lagged subspace is rebuilt on entry to each solve -- exactly where iBns moves.
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
            # hence opt-in with the lagged mode rather than always on. The scope below restores
            # every rank's previous flag and releases every manager on all exit paths.
            if obj.comm.Get_size() > 1:

                def cache_worker_command(command):
                    _send_int_array(x_init, command, obj.comm)

                cache_scope = _make_beltrami_factor_cache_scope(obj.field_obj, cache_worker_command)
            else:
                cache_scope = _make_beltrami_factor_cache_scope(obj.field_obj)

            def fan_out_jvp_batch(xdof, dirs_mn):
                _send_int_array(xdof, _WorkerCommand.JVP, obj.comm)
                for rank in range(1, obj.comm.Get_size()):
                    obj.comm.Send([np.array(dirs_mn.shape[1], dtype=np.int32), MPI_INT], dest=rank)
                    obj.comm.Send([dirs_mn, MPI_DOUBLE], dest=rank)

            def jvp_func(xdof, dirs):
                # A JVP is only usable when every volume reports status 0: a nonzero status
                # marks a product built from an unconverged or singular linearised solve, and
                # feeding one to the reduced model would poison the trust-region step. The
                # discarded batch costs one retry with freshly built factorizations; if that
                # fails too, the typed failure makes trf drop the lagged subspace and rebuild
                # the dense Jacobian (which selects the same per-volume backends as any other
                # build).
                t = time()
                dof_map = obj.Dxmn_Dxdof_at_xdof(xdof, obj.field_obj)
                dirs_mn = np.asfortranarray(dirs if dof_map is None else dof_map @ dirs)
                xmn = obj.xdof_to_xmn(xdof, obj.field_obj)
                for attempt in range(2):
                    if obj.comm.Get_size() > 1:
                        fan_out_jvp_batch(xdof, dirs_mn)
                    jvp, status = force_real_jvp(
                        xmn,
                        dirs_mn,
                        obj.field_obj,
                        nt=params.nt_dbb,
                        nz=params.nz_dbb,
                        filter_stellsym=params.filter_stellsym,
                        return_status=True,
                    )
                    jvp = jvp.reshape(-1, dirs.shape[1])
                    if not np.any(status):
                        print(f"fjvp: time =  {time() - t:.2e}     k    {dirs.shape[1]}")
                        return jvp
                    print(
                        f"WARNING: force_real_jvp reported per-volume status {list(status)}; "
                        "discarding this batch."
                    )
                    if np.any(status < 0):
                        # unsupported configuration, not a failed solve: a retry cannot help
                        break
                    if attempt == 0:
                        # Drop every rank's reused factors while leaving caching enabled, so
                        # locally rebuilt factors can be deposited and put back by the retry.
                        cache_scope.clear()
                return JvpResult.from_backend_failure(jvp.shape, status)

            def jac_checked(xdof, obj_):
                # The rebuild that follows a discarded JVP has no further fallback; warn when its
                # own linearised solves degraded, so a wandering minimization is attributable.
                fjac = jac(xdof, obj_)
                status = np.asarray(obj_.field_obj.allglobal_mod.force_jac_status)
                if np.any(status):
                    print(
                        f"WARNING: the force Jacobian was built from degraded linear solves "
                        f"(per-volume status {list(status)}); continuing with it. Tighten "
                        "epsgmres_jac, raise nitergmres, or select the dense Beltrami backend "
                        "(lbeltramisolver = 0)."
                    )
                return fjac

            jac_used = jac_checked
            tr_options["jvp"] = jvp_func
            tr_options["lag"] = int(jvp_lag)

    kwargs = dict(
        jac=jac_used,
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
    return params, kwargs, cache_scope


def _run_single_lstsq(obj, x_init, f, params, kwargs, cache_scope):
    """One least-squares solve over the full iteration budget."""

    def run_least_squares():
        return lstsq(
            f,
            x_init,
            max_niter=params.max_niter,
            iteration_start=obj.iteration_start,
            cb=obj.callback,
            **kwargs,
        )

    if cache_scope is None:
        return run_least_squares()
    with cache_scope:
        return run_least_squares()


def optimizer_default(obj, x_init, f, jac):
    """Minimize the force residual with a single least-squares solve."""
    params, kwargs, cache_scope = _lstsq_setup(obj, x_init, f, jac)
    return _run_single_lstsq(obj, x_init, f, params, kwargs, cache_scope)


def optimizer_picard(obj, x_init, f, jac):
    """Minimize the force residual with iBns frozen inside each inner solve.

    The free-boundary problem couples two conditions: force balance at a given boundary
    normal field, and consistency of that normal field with the plasma it bounds. This
    driver solves them in alternation. Each inner least-squares solve runs at a *fixed*
    ``iBns``, so the Jacobian it builds describes the residual surface it is actually
    descending; ``iBns`` then moves once, between solves, through ``refresh_bns``.
    Refreshing it more often -- per objective evaluation, say -- blends it toward the
    extrapolation from rejected trial points and pulls the residual surface out from
    under the Jacobian (the documented trust-region collapse).

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
    params, kwargs, cache_scope = _lstsq_setup(obj, x_init, f, jac)

    if not obj.picard_freeb:
        return _run_single_lstsq(obj, x_init, f, params, kwargs, cache_scope)

    total_iters = params.max_niter  # budget across all cycles, not per cycle
    budget_end = obj.iteration_start + total_iters

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

    def run_cycles():
        bnserr_hist = []
        x = x_init
        r = None
        outer = 0
        no_progress = 0  # consecutive cycles whose inner solve ran zero iterations

        while state["last_it"] < budget_end:
            cycle_start = state["last_it"]
            cap = min(params.inner_iter_cap, budget_end - cycle_start)

            print(
                f"\n--- outer cycle {outer} (frozen iBns) "
                f"iters {cycle_start}->{cycle_start + cap} ---"
            )

            # Iteration indices are global (the solvers resume at iteration_start and
            # bound against max_niter as an absolute index), so snapshots and history
            # rows never collide across cycles.
            r = lstsq(
                f,
                x,
                max_niter=cycle_start + cap,
                iteration_start=cycle_start,
                cb=cb_wrap,
                **kwargs,
            )
            x = r.x

            if r.status == 5:  # SIGUSR1 abort
                print("=== Aborting (SIGUSR1) ===")
                break

            # Put the plasma solution at the accepted geometry before extrapolating:
            # a solve that stopped right after a rejected trial leaves it elsewhere.
            f(x, obj)

            # bnserr is measured before this refresh applies its blend, so it grades the
            # iBns the cycle that just finished was scored against, not the one the next
            # cycle gets.
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

            # Whole-problem convergence: force balance solved AND the iBns it was solved
            # at already self-consistent, so the refresh above barely moved it.
            if bnserr < params.freeb_bnorm_tol and (inner_converged or inner_stalled):
                print("=== Converged: bnserr below tol and inner force solve settled ===")
                break

            # A cycle whose inner solve ran no iterations does not consume the budget, so
            # without this the loop could spin on f0/jac0 evaluations forever.
            if state["last_it"] == cycle_start:
                no_progress += 1
                if no_progress >= 2:
                    print(
                        "=== Stopping: two consecutive cycles ran zero inner iterations "
                        "while bnserr stayed above freeb_bnorm_tol. The inner solve keeps "
                        "terminating immediately (check gtol) before iBns is "
                        "self-consistent (check gbnbld). ==="
                    )
                    break
            else:
                no_progress = 0

            if len(bnserr_hist) >= 2 and bnserr_hist[-1] > bnserr_hist[-2]:
                print(
                    f"  WARNING: bnserr grew ({bnserr_hist[-2]:.3e} -> "
                    f"{bnserr_hist[-1]:.3e}); the iBns map may not be contracting. "
                    f"Consider a larger gbnbld."
                )

            outer += 1

        return r

    if cache_scope is None:
        return run_cycles()
    with cache_scope:
        return run_cycles()


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
        comm.Send([np.array(int(i), dtype=np.int32), MPI_INT], dest=r)
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

        # update_bns is the only thing that moves iBns: once at construction when the
        # bootstrap is requested, and once per outer cycle from refresh_bns(). That
        # exclusivity is what keeps every inner solve's Jacobian valid for the whole
        # residual surface it descends, and the written _end.toml a faithful restart file.
        self._bootstrap_bns = self.picard_freeb and bool(
            self.min_params.initialize_bns_from_current
        )

        self._set_init_iter()

        self.field_obj.input_list_mod.nppts = 0  # disable poincare at this stage since it's slow
        if self.infaces_prev is not None:
            self.field_obj.xin = self.infaces_prev
        self.field_obj.compute_fields()
        self.field_obj.save_output()

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
            dof_map = Dxmn_Dxdof_at_xdof(xdof, test)
            if dof_map is not None:
                fjac = fjac @ dof_map
            if test.comm.Get_rank() == 0:
                print(f"fjac: time =  {time() - t:.2e}     val  {np.sum(fjac**2):.3e}")
            return fjac

        self.jac_func = jac_func_wrap

        self.iter_freeb = 0

        from mpi4py import MPI

        global MPI_INT, MPI_DOUBLE
        MPI_INT = MPI.INT
        MPI_DOUBLE = MPI.DOUBLE

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
            # surface under a Jacobian built once per accepted iteration (the documented
            # trust-region collapse). The driver refreshes it between inner solves via
            # refresh_bns().
            def obj_func_mod(x, obj):
                _send_int_array(x, _WorkerCommand.FORCE, obj.comm)
                return obj_func2(x, obj.field_obj)

            def jac_func_mod(x, obj):
                _send_int_array(x, _WorkerCommand.JACOBIAN, obj.comm)
                return jac_func2(x, obj.field_obj)

            self.obj_func = obj_func_mod
            self.jac_func = jac_func_mod

            if self.iteration_start == 0:
                system("rm -r xs_fixb; rm -r xs_freeb")

                if self.min_params.save_xin_fixb:
                    system("mkdir xs_fixb")

                if self.min_params.save_xin_freeb:
                    system("mkdir xs_freeb")

            # Size to the highest *global* iteration index the callback can write. On a
            # resumed run the driver runs the max_niter budget on top of iteration_start,
            # so `it` reaches iteration_start + max_niter (iteration_start is 0 for fresh
            # runs, leaving this unchanged for them).
            n_iter_slots = self.iteration_start + self.min_params.max_niter + 1
            self.iter_iter = [-1.0] * n_iter_slots
            self.iter_nfev = [-1.0] * n_iter_slots
            self.iter_cost = [-1.0] * n_iter_slots
            self.iter_cost_red = [-1.0] * n_iter_slots
            self.iter_step_norm = [-1.0] * n_iter_slots
            self.iter_opt = [-1.0] * n_iter_slots
            self.iter_fsa = [-1.0] * n_iter_slots
            self.iter_freal = [-1.0] * n_iter_slots

            print("Initialization done!")

    def minimize(self):
        if self.rank == 0:

            def run_optimizer():
                try:
                    return self.optimizer(self, self.x_init, self.obj_func, self.jac_func)
                except BaseException:
                    # The optimizer's cache scope exits before this handler. Release the worker
                    # loops instead of leaving them blocked in their next receive.
                    _send_int_array(self.x_init, _WorkerCommand.STOP, self.comm)
                    raise

            with Timer() as t:
                if self.freeb_mode:
                    print("SPECTRE Free-boundary force minimization")

                    self.result = run_optimizer()

                else:
                    print("SPECTRE Fixed-boundary force minimization...")

                    self.result = run_optimizer()

            _send_int_array(self.x_init, _WorkerCommand.STOP, self.comm)

            self.t = t

            print(f"Minimization done ({self.t.readout})!")

        else:
            # Set up the signal catcher for other ranks to avoid termination. Master takes care of the rest.
            def handler_stop_signals(*args):
                pass

            signal(SIGUSR1, handler_stop_signals)

            cache_scope = _make_beltrami_factor_cache_scope(self.field_obj)
            try:
                while True:
                    x_temp, flag = _recv_int_array(self)
                    command = _WorkerCommand(flag)

                    if command is _WorkerCommand.FORCE:
                        self.obj_func(x_temp, self.field_obj)
                    elif command is _WorkerCommand.JACOBIAN:
                        self.jac_func(x_temp, self.field_obj)
                    elif command is _WorkerCommand.REFRESH_BNS:
                        # join the collective iBns refresh driven from rank 0
                        self.field_obj.update_bns(mode="refresh")
                    elif command is _WorkerCommand.JVP:
                        self._jvp_worker(x_temp)
                    elif command is _WorkerCommand.CACHE_ENTER:
                        cache_scope.__enter__()
                    elif command is _WorkerCommand.CACHE_CLEAR:
                        cache_scope.clear()
                    elif command is _WorkerCommand.CACHE_EXIT:
                        cache_scope.__exit__(None, None, None)
                    elif command is _WorkerCommand.STOP:
                        break
            finally:
                # Normally CACHE_EXIT already closed the scope. This guard handles an abnormal
                # worker-loop exit without touching the manager when lagging was never enabled.
                cache_scope.__exit__(None, None, None)

    def refresh_bns(self):
        """Refresh the free-boundary normal field on every rank, once.

        This is the only place ``iBns`` moves during a free-boundary minimization: it
        belongs *between* inner force solves, never inside one, so that the Jacobian a
        solve is built on stays valid for that solve's whole residual surface.
        ``_disable_fortran_picard`` makes that exclusivity real.

        Returns:
            The ``bnserr`` of the ``iBns`` the last force solve used, on rank 0; ``nan``
            on the peers (they refresh through the worker loop). Measured before the
            update, so it is the residual the completed inner solve was actually scored
            against, independent of ``gbnbld``.
        """
        if self.rank != 0:
            return float("nan")  # peers refresh through the REFRESH_BNS listener

        _send_int_array(self.x_curr, _WorkerCommand.REFRESH_BNS, self.comm)
        return self.field_obj.update_bns(mode="refresh", verbose=True)

    def _jvp_worker(self, xdof):
        """Receive an mn-space direction batch and join the collective JVP call."""
        from spectre.force_targets import force_real_jvp

        ndirs = np.empty(1, dtype=np.int32)
        self.comm.Recv([ndirs, MPI_INT], source=0)
        ngdof = int(self.field_obj.allglobal_mod.ngdof)
        dirs_mn = np.empty((ngdof, int(ndirs[0])), dtype=np.float64, order="F")
        self.comm.Recv([dirs_mn, MPI_DOUBLE], source=0)

        force_real_jvp(
            self.xdof_to_xmn(xdof, self.field_obj),
            dirs_mn,
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

            _send_int_array(xdof_final, _WorkerCommand.FORCE, self.comm)
        else:
            xdof_final, _ = _recv_int_array(self)

        x_final = self.xdof_to_xmn(xdof_final, field_obj)

        if run_spec:
            field_obj.input_list_mod.linitialize = 0
            field_obj.input_list_mod.nppts = 0

            self.field_obj.allglobal_mod.filename_h5 = f"{field_obj.name}_opt.h5"
            # Write the output at the accepted geometry: the last evaluation of the solve
            # may have been a rejected trial, so re-solve at x_final before writing.
            field_obj.xin = x_final
            field_obj.compute_fields()
            field_obj.save_output()

            if self.rank == 0:
                system(f"mv {field_obj.name}_end.toml {field_obj.name}_opt.toml")

        self.field_obj.force_real(
            x_final,
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
            if len(infaces_prev) == ngdof + 1:
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

            _send_int_array(self.x_init, _WorkerCommand.STOP, self.comm)

        else:
            while True:
                x_temp, flag = _recv_int_array(self)

                if flag == _WorkerCommand.FORCE:
                    # calculate the function
                    self.obj_func(x_temp, self.field_obj)
                elif flag == _WorkerCommand.JACOBIAN:
                    # calculate the jacobian
                    self.jac_func(x_temp, self.field_obj)
                elif flag == _WorkerCommand.REFRESH_BNS:
                    # join the collective iBns refresh
                    self.field_obj.update_bns(mode="refresh")
                elif flag == _WorkerCommand.STOP:
                    # finish running
                    break
