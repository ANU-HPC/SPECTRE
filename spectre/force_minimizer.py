# SPDX-FileCopyrightText: 2026 Proxima Fusion
# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

import numpy as np
from os import system
from time import time
from glob import glob

from spectre import SPECTRE
from spectre.utils import (
    Timer,
    get_xinit_specwrap,
    get_freal_fsa_metric,
    save_errors_to_h5,
    update_bnorm,
)
from spectre.input_dict import input_dict
from spectre.least_squares.least_squares import least_squares as lstsq
from spectre.error_checking import print_all_errors
from spectre.force_targets import force_real

from signal import signal, SIGUSR1


def optimizer_default(obj, x_init, f, jac):
    params = (
        input_dict(
            dict(
                gtol=1e-13, xtol=1e-22, ftol=1e-13, max_nfev=200, method="dogbox", rcond_lstsq=None
            )
        )
        if obj.min_params is None
        else obj.min_params
    )

    tr_solver = getattr(params, "tr_solver", None)
    tr_options = {}
    if tr_solver == "banded":
        # The banded solver relies on the block-tridiagonal interface structure of the
        # force Jacobian: pure force rows, one equal column block per interface.
        if getattr(params, "objective", "freal") != "freal" or params.inface_repr == "henn":
            print(
                "WARNING: tr_solver='banded' requires objective='freal' and a per-interface "
                "DOF representation; using exact SVD."
            )
            tr_solver = "exact"
        else:
            tr_options = {"n_blocks": int(obj.field_obj.allglobal_mod.mvol) - 1}

    jvp_lag = getattr(params, "jvp_lag", None)
    if jvp_lag and tr_solver != "banded":
        print("WARNING: min_params.jvp_lag requires tr_solver='banded'; ignored.")
    elif jvp_lag:
        # Lagged-banded matrix-free iteration: reuse the dense Jacobian (and its
        # banded factorization) for up to jvp_lag iterations, refreshing the model
        # each iterate with a small batch of exact Jacobian-vector products.
        mpol = int(obj.field_obj.input_list_mod.mpol)
        ntor = int(obj.field_obj.input_list_mod.ntor)
        nfp = int(obj.field_obj.input_list_mod.nfp)
        if obj.comm.Get_size() > 1:
            print("WARNING: jvp_lag needs a single MPI rank (force_real_jvp); ignored.")
        elif obj.freeb_mode:
            print(
                "WARNING: jvp_lag does not support free-boundary mode "
                "(bnorm updates change the residual between iterations); ignored."
            )
        elif params.inface_repr == "fourier" and (
            (params.mpol_rz is not None and params.mpol_rz < mpol)
            or (params.ntor_rz is not None and params.ntor_rz < ntor * nfp)
        ):
            print(
                "WARNING: jvp_lag needs the full perturbation spectrum "
                "(the JVP has no mpol_rz/ntor_rz truncation); ignored."
            )
        else:
            from spectre.force_targets import force_real_jvp

            def jvp_func(xdof, dirs):
                t = time()
                dirs_mn = obj.Dxmn_Dxdof_at_xdof(xdof, obj.field_obj) @ dirs
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

    return lstsq(
        f,
        x_init,
        jac=jac,
        method=params.method,
        verbose=2,
        args=[obj],
        gtol=params.gtol,
        xtol=params.xtol,
        ftol=params.ftol,
        max_nfev=params.max_nfev,
        max_niter=params.max_niter,
        iteration_start=obj.iteration_start,
        rcond_lstsq=params.rcond_lstsq,
        tr_solver=tr_solver,
        tr_options=tr_options,
        cb=obj.callback,
    )


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
            fjac = jac_func(xdof_to_xmn(xdof, test), test) @ Dxmn_Dxdof_at_xdof(xdof, test)
            if test.comm.Get_rank() == 0:
                print(f"fjac: time =  {time() - t:.2e}     val  {np.sum(fjac**2):.3e}")
            return fjac

        self.jac_func = jac_func_wrap

        self.iter_freeb = 0

        from mpi4py import MPI

        global MPI_INT, MPI_DOUBLE
        MPI_INT = MPI.INT
        MPI_DOUBLE = MPI.DOUBLE

        self.freeb_mode = self.field_obj.input_list_mod.lfreebound

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

            if self.freeb_mode:  # for strategy 2

                def obj_func_mod(x, obj):
                    _send_int_array(x, 2, obj.comm)
                    update_bnorm(obj.field_obj, first_free_bound=False, print_info=True)

                    _send_int_array(x, 0, obj.comm)
                    return obj_func2(x, obj.field_obj)
            else:

                def obj_func_mod(x, obj):
                    _send_int_array(x, 0, obj.comm)
                    return obj_func2(x, obj.field_obj)

            def jac_func_mod(x, obj):
                _send_int_array(x, 1, obj.comm)
                return jac_func2(x, obj.field_obj)

            # For strategy 1
            # def update_bnorm_loc(first_free_bound):
            #     _send_int_array(self.x_init, 2, self.field_obj.comm)
            #     update_bnorm(self.field_obj, first_free_bound, print_info=True)
            #     print(
            #         f"Calculated b_norm ({self.iter_freeb}) Bns_err: {self.field_obj.allglobal_mod.bnserr:.5e}"
            #     )

            #     bns_tol_reached = self.field_obj.allglobal_mod.bnserr < self.min_params.freeb_bnorm_tol
            #     return bns_tol_reached

            self.obj_func = obj_func_mod
            self.jac_func = jac_func_mod
            # self.update_bnorm = update_bnorm_loc # For strategy 1

            if self.iteration_start == 0:
                system("rm -r xs_fixb; rm -r xs_freeb")

                if self.min_params.save_xin_fixb:
                    system("mkdir xs_fixb")

                if self.min_params.save_xin_freeb:
                    system("mkdir xs_freeb")

            self.iter_iter = [-1.0] * (self.min_params.max_niter + 1)
            self.iter_nfev = [-1.0] * (self.min_params.max_niter + 1)
            self.iter_cost = [-1.0] * (self.min_params.max_niter + 1)
            self.iter_cost_red = [-1.0] * (self.min_params.max_niter + 1)
            self.iter_step_norm = [-1.0] * (self.min_params.max_niter + 1)
            self.iter_opt = [-1.0] * (self.min_params.max_niter + 1)
            self.iter_fsa = [-1.0] * (self.min_params.max_niter + 1)
            self.iter_freal = [-1.0] * (self.min_params.max_niter + 1)

            print("Initialization done!")

    def minimize(self):
        if self.rank == 0:
            with Timer() as t:
                if self.freeb_mode:
                    print("SPECTRE Free-boundary force minimization")

                    ## There are 3 strategies to do free-boundary minimization:
                    ## 1. Run a minimization for X iterations / convergence, then update bnorm, then re-run minimization for X iterations...
                    ## 2. Do an iteration, update bnorm, do next iteration...
                    ## 3. For each evaluation of the field/force, run an internal loop of field evaluation + bnorm update, for N number of cycles/convergence on bnorm
                    ## Note:
                    ## An iteration includes: 1) force eval, 2) jacobian eval, 3) finding good step and updating infaces
                    ## Bnorm update is usually quite cheap compared to force/jacobian eval

                    ## Strategy 1

                    # self.update_bnorm(first_free_bound=False)

                    # x_curr = self.x_init.copy()

                    # while self.iter_freeb < self.min_params.freeb_max_iter:
                    #     print(f"Force minimization ({self.iter_freeb})...  ")
                    #     self.result = self.optimizer(self, x_curr, self.obj_func, self.jac_func)
                    #     x_curr = self.result.x
                    #     if self.min_params.save_xin_freeb:
                    #         np.save(f"xs_freeb/x_{self.iter_freeb:03}.npy", x_curr)
                    #     self.iter_freeb += 1
                    #     self.update_bnorm(first_free_bound=False)
                    #     if self.result.status == 5:
                    #         print("Early ending freeb simulation...")
                    #         break

                    ## Strategy 2

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
                    # calculate the B_norm
                    self.field_obj.wrapper_funcs_mod.update_bnorm()
                elif flag == 9:
                    # finish running
                    break

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
                    # calculate the B_norm
                    self.field_obj.wrapper_funcs_mod.update_bnorm()
                elif flag == 9:
                    # finish running
                    break
