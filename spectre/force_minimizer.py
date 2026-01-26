import numpy as np
from os import system
from time import process_time
from glob import glob

from spectre.misc import Timer, get_xinit_specwrap, get_freal_fsa_metric, update_bnorm
from spectre.input_dict import input_dict
from spectre.least_squares.least_squares import least_squares as lstsq
from spectre import SPECTRE

from signal import signal, SIGUSR1

def optimizer_default(obj, x_init, f, jac):
    params = (
        input_dict(dict(gtol=1e-13, xtol=1e-22, ftol=1e-13, max_nfev=200, method="dogbox", rcond_lstsq=None))
        if obj.min_params is None
        else obj.min_params
    )

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
        cb=obj.callback,
    )


def callback_default(it, nfev, cost, cost_red, stp_nrm, opt, x, obj):
    cost_red = " " * 8 if cost_red is None else f"{cost_red:.2e}"
    step_norm = " " * 8 if stp_nrm is None else f"{stp_nrm:.2e}"
    opt = " " * 5 if opt is None else f"{opt:.2e}"

    freal = np.sqrt(np.mean(obj.field_obj.f_real_shpd**2))
    fsa = get_freal_fsa_metric(obj.field_obj)

    print(
        f"\niter {it:5}   nfev {nfev:6}    cost {cost:.3e}   <f> {fsa:.5e}   fr {freal:.4e}  stpnrm {step_norm}   costred {cost_red}   optim {opt}"
    )

    if obj.save_xin_fixb:
        np.save(f"xs_fixb/x_{obj.iter_freeb:03}_{it:04}.npy", x)


def send_int_array(arr, i, comm):
    # sends an integer and an array from rank 0 to other ranks
    # called from rank 0
    for r in range(1, comm.Get_size()):
        comm.Send([np.array(i, dtype=np.int32), MPI_INT], dest=r)
        comm.Send([arr, MPI_DOUBLE], dest=r)


def recv_int_array(self):
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
        self,
        filename,
        optimizer=optimizer_default,
        callback=callback_default,
        xmn_to_xdof=None,
        xdof_to_xmn=None,
        Dxmn_Dxdof_at_xdof=None,
        first_free_bound=True,
    ):
        if filename[-5:] != ".toml":
            raise RuntimeError("Input file must be a .toml file")
        
        self.field_obj = SPECTRE.from_input_file(filename)

        self.rank = self.field_obj.comm.rank
        self.comm = self.field_obj.comm
        self.name = self.field_obj.name
        
        if xmn_to_xdof is None:
            xmn_to_xdof = lambda x, t: x
            xdof_to_xmn = lambda x, t: x
            Dxmn_Dxdof_at_xdof = lambda x, t: np.identity(len(x))

        if self.rank == 0:
            print(f"Starting SPECTRE force minimization with {filename}")

        inface_files = sorted(glob('xs_fixb/*'))
        
        # If previous iterations exist, start from the last one
        if inface_files:
            infaces_prev = np.load(inface_files[-1])
            if self.rank == 0:
                print(f"Starting from previous interfaces {inface_files[-1]}")
        else:
            infaces_prev = None

        self.field_obj.run(save_output=True, infaces=infaces_prev)        

        self.optimizer = optimizer
        self.callback = callback
        self.xmn_to_xdof = xmn_to_xdof
        self.xdof_to_xmn = xdof_to_xmn

        # Broadcast minimization params
        if self.rank == 0:
            min_params_temp = self.field_obj.parameters.minimization.model_dump()
        else:
            min_params_temp = None
        self.min_params = input_dict(self.comm.bcast(min_params_temp, root=0))   

        # If max iterations reached, quit
        self.iteration_start = 0
        if self.min_params.max_niter < 10000: # max_niter is set not to default
            num_infaces = len(inface_files)
            if num_infaces > 0:
                self.iteration_start = num_infaces - 1
                if num_infaces >= self.min_params.max_niter:
                    if self.rank == 0:
                        print("Number of iterations already reached, quitting...")
                    exit()   
    
        self.save_xin_fixb = self.min_params.save_xin_fixb
        self.save_xin_freeb = self.min_params.save_xin_freeb
        # If the first iteration uses previous Bns or needs to compute it from scratch
        self.first_free_bound = first_free_bound 

        # Define objective function (and its Jacobian) for force eval on mesh in real space
        if self.min_params.objective == "freal":
            from spectre import force_real, force_real_jac
            
            obj_func = lambda x, t: force_real(x, t, nt=self.min_params.nt_dbb, nz=self.min_params.nz_dbb, filter_stellsym=self.min_params.filter_stellsym).flatten()
            jac_func = lambda x, t: force_real_jac(x, t, nt=self.min_params.nt_dbb, nz=self.min_params.nz_dbb, filter_stellsym=self.min_params.filter_stellsym).reshape(-1, t.allglobal_mod.ngdof+1)
        else:
            raise RuntimeError("Only ['freal'] objectives are available")

        def obj_func_wrap(xdof, test):
            t = process_time()
            f = obj_func(xdof_to_xmn(xdof, test), test)
            if test.comm.Get_rank() == 0:
                print(f"f: time =  {process_time() - t:.2e}     val  {np.sum(f**2):.3e}")
            return f

        self.obj_func = obj_func_wrap

        def jac_func_wrap(xdof, test):
            t = process_time()
            fjac = jac_func(xdof_to_xmn(xdof, test), test) @ Dxmn_Dxdof_at_xdof(xdof, test)
            if test.comm.Get_rank() == 0:
                print(f"fjac: time =  {process_time() - t:.2e}     val  {np.sum(fjac**2):.3e}")
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

        self.f_init = self.obj_func(self.x_init, self.field_obj)

        self.x_curr = self.x_init.copy()  # buffer for mpi comms

        # wrap obj_func, jac functions
        if self.rank == 0:
            obj_func2 = self.obj_func
            jac_func2 = self.jac_func

            def obj_func_mod(x, obj):
                send_int_array(x, 0, obj.comm)
                return obj_func2(x, obj.field_obj)

            def jac_func_mod(x, obj):
                send_int_array(x, 1, obj.comm)
                return jac_func2(x, obj.field_obj)

            def update_bnorm_loc(first_free_bound):
                send_int_array(self.x_init, 2, self.field_obj.comm)
                update_bnorm(self.field_obj, first_free_bound, print_info=True)
                print(
                    f"Calculated b_norm ({self.iter_freeb}) Bns_err: {self.field_obj.allglobal_mod.bnserr:.5e}"
                )

                bns_tol_reached = self.field_obj.allglobal_mod.bnserr < self.min_params.freeb_bnorm_tol
                return bns_tol_reached

            self.obj_func = obj_func_mod
            self.jac_func = jac_func_mod
            self.update_bnorm = update_bnorm_loc

            if self.rank == 0 and self.iteration_start == 0:
                system("rm -r xs_fixb; rm -r xs_freeb")

                if self.save_xin_fixb:
                    system("mkdir xs_fixb")

                if self.save_xin_freeb:
                    system("mkdir xs_freeb")

            if self.rank == 0:
                print(f"Initialization done!")

    def minimize(self):
        if self.rank == 0:

            print('Starting minimization...')
            
            with Timer() as t:
                if self.freeb_mode:
                    print("Free-boundary force minimization...")

                    self.update_bnorm(first_free_bound=self.first_free_bound)

                    x_curr = self.x_init.copy()

                    while self.iter_freeb < self.min_params.freeb_max_iter:
                        print(f"Force minimization ({self.iter_freeb})...  ")

                        self.result = self.optimizer(self, x_curr, self.obj_func, self.jac_func)

                        x_curr = self.result.x

                        if self.save_xin_freeb:
                            np.save(f"xs_freeb/x_{self.iter_freeb:03}.npy", x_curr)

                        self.iter_freeb += 1

                        self.update_bnorm(first_free_bound=False)

                        if self.result.status == 5:
                            print("Early ending freeb simulation...")
                            break

                else:
                    print("Fixed-boundary force minimization...")

                    self.result = self.optimizer(self, self.x_init, self.obj_func, self.jac_func)

            send_int_array(self.x_init, 9, self.comm)

            self.t = t

            print(f"Minimization done ({t.readout})!")

        else:

            # Set up the signal catcher for other ranks to avoid termination. Master takes care of the rest.
            def handler_stop_signals(*args):
                pass
            signal(SIGUSR1, handler_stop_signals)

            while True:
                x_temp, flag = recv_int_array(self)

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
            name = self.name

            result.pop("jac")
            result.time_end = self.t.readout

            if save_pkl:
                result.save_pickle(f"{name}_res.pkl")
            if save_json:
                result.save_json(f"{name}_res.json")

            xdof_final = result.x

            self.x_final = self.xdof_to_xmn(xdof_final, field_obj)

            send_int_array(xdof_final, 0, self.comm)
        else:
            xdof_final, _ = recv_int_array(self)

        x_final = self.xdof_to_xmn(xdof_final, field_obj)

        ngdof = field_obj.allglobal_mod.ngdof
        mvol = field_obj.allglobal_mod.mvol
        mn = field_obj.allglobal_mod.mn
        nt = field_obj.allglobal_mod.nt
        nz = field_obj.allglobal_mod.nz
        irbc = field_obj.allglobal_mod.irbc
        izbs = field_obj.allglobal_mod.izbs
        irbs = field_obj.allglobal_mod.irbs
        izbc = field_obj.allglobal_mod.izbc

        if run_spec:
            field_obj.packing_mod.packxi(
                ngdof, x_final, mvol, mn, nt, nz, irbc, izbs, irbs, izbc, "U", False, False
            )
            field_obj.input_list_mod.lfindzero = 0
            field_obj.input_list_mod.linitialize = 0
            field_obj.input_list_mod.nppts = 0

            self.field_obj.allglobal_mod.filename_h5 = f"{field_obj.name}_opt.h5"
            field_obj.run(save_output=True)

            if self.rank == 0:
                system(f"mv {field_obj.name}_end.toml {field_obj.name}_opt.toml")

        # # postprocess intermediate x files
        # if self.rank == 0:
        #     from numpy import save, load, argsort

        #     if self.save_xin_fixb:
        #         fname_pattern = "xs_fixb/x_*.npy"
        #         file_list = glob(fname_pattern)

        #         if len(file_list) == 0:
        #             return

        #         xs = []
        #         inds = []

        #         for fname in sorted(file_list):
        #             xs.append(load(fname))
        #             inds.append(fname.split("x_")[-1].split(".np")[0])

        #         sort_inds = np.argsort(inds)

        #         save("x_fixb_evolution.npy", np.array(xs)[sort_inds])

        #     if self.save_xin_freeb:
        #         fname_pattern = "xs_freeb/x_*.npy"
        #         file_list = glob(fname_pattern)

        #         if len(file_list) == 0:
        #             return

        #         xs = []
        #         inds = []

        #         for fname in sorted(file_list):
        #             xs.append(load(fname))
        #             inds.append(fname.split("x_")[-1].split(".np")[0])

        #         sort_inds = argsort(inds)

        #         save("x_freeb_evolution.npy", np.array(xs)[sort_inds])
