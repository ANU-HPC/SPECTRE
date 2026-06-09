# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

import time

import numpy as np
from h5py import File as h5py_File
from mpi4py import MPI

from spectre.field import bfield_flow_nonsing, bfield_flow_sing
from spectre.geometry import at_volume_edge, sth_to_RZ

# Available field-line ODE backends (see _integrate_trajectory):
#   "cyrk"  -- CyRK, CC-BY-NC-SA (NonCommercial). Fastest, but installed only via
#              the optional extra: `pip install spectre[non-commercial]`.
#   "scipy" -- scipy.integrate.solve_ivp, BSD-3 (permissive; core dependency).
# The default (rk_backend=None) prefers CyRK when it is installed, otherwise the
# permissive scipy backend, so a fully-permissive install needs no NC software.
RK_BACKENDS = ("cyrk", "scipy")


def default_rk_backend():
    """Return the default ODE backend: CyRK if installed (fast, NonCommercial), else scipy."""
    try:
        import CyRK  # noqa: F401

        return "cyrk"
    except ImportError:
        return "scipy"


class Field_Tracer:
    def __init__(
        self,
        mvol,
        igeometry,
        nfp,
        mpol=None,
        im=None,
        in_=None,
        ate=None,
        aze=None,
        ato=None,
        azo=None,
        rbs=None,
        rbc=None,
        zbs=None,
        zbc=None,
        lrad_arr=None,
        num_threads=1,
        batch_size=1,
        method="RK45",
        rtol=1.0e-6,
        atol=1.0e-6,
        comm=None,
        rk_backend=None,
    ):
        """Initialize the Field_Tracer object. Does not trace the field.

        Args:
            rk_backend (str): ODE integrator backend, one of RK_BACKENDS
                ("cyrk", "scipy"). Defaults to None, which resolves via
                :func:`default_rk_backend` to "cyrk" when CyRK (NonCommercial) is
                installed, otherwise the permissive "scipy" backend.
        """
        if rk_backend is None:
            rk_backend = default_rk_backend()
        if rk_backend not in RK_BACKENDS:
            raise ValueError(f"rk_backend must be one of {RK_BACKENDS}, got {rk_backend!r}")

        self.mvol = mvol
        self.igeometry = igeometry
        self.mpol = mpol
        self.im = im
        self.in_ = in_
        self.nfp = nfp
        self.ate = ate
        self.aze = aze
        self.ato = ato
        self.azo = azo
        self.rbs = rbs
        self.rbc = rbc
        self.zbs = zbs
        self.zbc = zbc
        self.lrad_arr = lrad_arr
        self.num_threads = num_threads
        self.batch_size = batch_size
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.rk_backend = rk_backend
        self.comm = MPI.COMM_WORLD if comm is None else comm

    def trace_field(
        self,
        init_pts,
        num_phi_planes=32,
        num_transits=1,
        method="RK45",
        rtol=None,
        atol=None,
        rk_backend=None,
    ):
        """Traces the magnetic field from given initial points. Volumes can have different number of trajectories, but every trajectory will have 'num_phi_planes * num_transits' turns. Sets the output as self attributes.

        Args:
            init_pts (List[ndarray[float]]): list of ndarray[num_trajectories, 2] with initial (s, theta) points for tracing
            num_phi_planes (int, optional): Number of phi planes to record the poincare. Defaults to 32.
                        num_transits (int, optional): Number of toroidal transits to perform. Defaults to 1.
            method (str, optional): Method for integrating the ode. Defaults to 'RK45'. Options: 'RK23', 'RK45', 'DOP853'
                        rtol (float, optional): Relative tolerance of the integrator. Defaults to 1.0e-6.
                        atol (float, optional): Absolute tolerance of the integrator. Defaults to 1.0e-6.

        Raises:
            RuntimeError: initial s is out of range [-1, 1] for a point
        """
        mvol = self.mvol
        rtol = self.rtol if rtol is None else rtol
        atol = self.atol if atol is None else atol
        method = self.method if method is None else method
        rk_backend = self.rk_backend if rk_backend is None else rk_backend

        comm = self.comm
        rank, size, local_indices = _mpi_divide_work(mvol, comm)

        num_pts_traj = num_phi_planes * num_transits
        phi_span = (0.0, (2.0 * np.pi / self.nfp) * num_transits)
        phi_eval = np.linspace(phi_span[0], phi_span[1], num_pts_traj, endpoint=False)

        sth_local = [None] * mvol
        RZ_local = [None] * mvol

        time_start = time.time()

        bad_traj_count_local = [
            [] for _ in range(mvol)
        ]  # Element 'k' contains tuples (traj_index, num_good_pts) of incomplete trajectories

        # loop over volumes, each rank gets its volumes
        for ivol in local_indices:
            sth_init = init_pts[ivol]

            if sth_init is None:
                continue

            if np.any(sth_init[:, 0] > 1.0) or np.any(sth_init[:, 0] < -1.0):
                raise RuntimeError(
                    f"Initial s must be in range [-1, 1], got s={sth_init[:, 0]} in vol {ivol}"
                )

            singular_vol = (ivol == 0) and (self.igeometry > 1)

            rbc_in, rbc_out = self.rbc[ivol : ivol + 2]
            zbs_in, zbs_out = self.zbs[ivol : ivol + 2]
            rbs_in, rbs_out = self.rbs[ivol : ivol + 2]
            zbc_in, zbc_out = self.zbc[ivol : ivol + 2]

            lrad = self.lrad_arr[ivol]

            ## Allow for tracing with joblib (can be multithreaded)
            # from joblib import Parallel, delayed
            # outputs = Parallel(n_jobs=num_threads,  batch_size=batch_size)(
            #     delayed(_solve_field_ode)(*a) for a in args) # use joblib to use multiple threads availabe

            sth_temp = []
            RZ_temp = []
            for itrj in range(len(sth_init)):
                sth, RZ = _solve_field_ode(
                    sth_init[itrj],
                    singular_vol,
                    self.aze[ivol],
                    self.ate[ivol],
                    self.azo[ivol],
                    self.ato[ivol],
                    rbc_in,
                    rbc_out,
                    zbs_in,
                    zbs_out,
                    rbs_in,
                    rbs_out,
                    zbc_in,
                    zbc_out,
                    self.mpol,
                    lrad,
                    self.im,
                    self.in_,
                    phi_eval,
                    phi_span,
                    method,
                    rtol,
                    atol,
                    rk_backend,
                )

                num_successful_pts = sth.shape[1]
                if num_successful_pts != num_pts_traj:
                    pad = np.full((2, num_pts_traj - num_successful_pts), -2.0)
                    sth = np.concatenate([sth, pad], axis=1)
                    RZ = np.concatenate([RZ, pad], axis=1)
                    bad_traj_count_local[ivol].append((itrj, num_successful_pts))

                sth_temp.append(sth)
                RZ_temp.append(RZ)

            sth_local[ivol] = np.array(sth_temp)

            RZ_local[ivol] = np.array(RZ_temp)

            print(f"Rank {rank}: done volume {ivol}")

        time_end = time.time()

        sth_total, RZ_total, bad_traj_count = _mpi_gather_results(
            comm, rank, local_indices, mvol, num_pts_traj, sth_local, RZ_local, bad_traj_count_local
        )

        if rank == 0:
            print(f"Poincare trace completed ({time_end - time_start:.2f}s)")

        # Compute iota, which comes from a linear fit to the theta vs phi curve
        iotas = None
        if rank == 0:
            iotas = _calc_all_iotas(mvol, phi_eval, sth_total, bad_traj_count)

        self.sth_total = sth_total
        self.RZ_total = RZ_total
        self.phi_eval = phi_eval
        self.bad_traj_count = bad_traj_count
        self.iotas = iotas
        self.num_phi_planes = num_phi_planes
        self.num_transits = num_transits
        self.num_pts_traj = num_pts_traj

    def trace_field_single(
        self,
        sth_init,
        ivol,
        num_phi_planes=32,
        num_transits=1,
        method="RK45",
        rtol=None,
        atol=None,
        rk_backend=None,
    ):
        """Traces the magnetic field for one initial point, the trajectory will have 'num_phi_planes * num_transits' turns. Doesn't set anything and returns the result.

        Args:
            init_pts (List[ndarray[float]]): list of ndarray[num_trajectories, 2] with initial (s, theta) points for tracing.
            ivol (int): Index of the volume where tracing is done.
            num_phi_planes (int, optional): Number of phi planes to record the poincare. Defaults to 32.
                        num_transits (int, optional): Number of toroidal transits to perform. Defaults to 1.
            method (str, optional): Method for integrating the ode. Defaults to 'RK45'. Options: 'RK23', 'RK45', 'DOP853'
                        rtol (float, optional): Relative tolerance of the integrator. Defaults to 1.0e-6.
                        atol (float, optional): Absolute tolerance of the integrator. Defaults to 1.0e-6.

        Raises:
            RuntimeError: initial s is out of range [-1, 1] for a point.

        Returns:
            List[float]: Final (s, theta) coordinates.
            List[float]: Final (R, Z) coordinates.
            float: Final iota.
            bool: True if integration did not reach the end, False otherwise.
        """
        rtol = self.rtol if rtol is None else rtol
        atol = self.atol if atol is None else atol
        method = self.method if method is None else method
        rk_backend = self.rk_backend if rk_backend is None else rk_backend

        num_pts_traj = num_phi_planes * num_transits
        phi_span = (0.0, (2.0 * np.pi / self.nfp) * num_transits)
        phi_eval = np.linspace(phi_span[0], phi_span[1], num_pts_traj, endpoint=False)

        sth_init = np.array(sth_init)
        if sth_init.shape != (2,):
            raise RuntimeError("sth_init has to be array-like with 2 elements")

        if np.any(sth_init[0] > 1.0) or np.any(sth_init[0] < -1.0):
            raise RuntimeError(
                f"Initial s must be in range [-1, 1], got s={sth_init[:, 0]} in vol {ivol}"
            )

        singular_vol = (ivol == 0) and (self.igeometry > 1)

        rbc_in, rbc_out = self.rbc[ivol : ivol + 2]
        zbs_in, zbs_out = self.zbs[ivol : ivol + 2]
        rbs_in, rbs_out = self.rbs[ivol : ivol + 2]
        zbc_in, zbc_out = self.zbc[ivol : ivol + 2]

        lrad = self.lrad_arr[ivol]

        sth, RZ = _solve_field_ode(
            sth_init,
            singular_vol,
            self.aze[ivol],
            self.ate[ivol],
            self.azo[ivol],
            self.ato[ivol],
            rbc_in,
            rbc_out,
            zbs_in,
            zbs_out,
            rbs_in,
            rbs_out,
            zbc_in,
            zbc_out,
            self.mpol,
            lrad,
            self.im,
            self.in_,
            phi_eval,
            phi_span,
            method,
            rtol,
            atol,
            rk_backend,
        )

        num_successful_pts = sth.shape[1]
        flag = num_successful_pts != num_pts_traj
        if flag:
            print(
                f"Integration incomplete, reached {num_successful_pts}/{num_pts_traj} points, theta_end {sth[1, -1]:.3f}"
            )

        iota = _calc_iota(phi_eval, sth[1, :])

        return sth, RZ, iota, flag

    def plot(
        self, ind_zetan=0, zetan=None, prange="full", ax=None, figsize=(6, 5), show=False, **kwargs
    ):
        """Plot the Poincare section at given toroidal location. Allow input of both zeta index and zetan float value. Zetan (if provided) takes precedence.

        Args:
            ind_zetan (int, optional): Index of toroidal lcoation. Defaults to 0.
            zetan (_type_, optional): Normalized zeta of toroidal location. Defaults to None.
            prange (str, optional): Designated part to plot. Defaults to "full". Options: 'full', 'upper', 'lower'.
            ax (_type_, optional): Existing axis object. Defaults to None.
            figsize (tuple, optional): Figure size. Defaults to (6, 5).
            show (bool, optional): Whether to call `plt.show()`. Defaults to False.
            kwargs: Additional keyword arguments for `ax.plot()`.

        Raises:
            ValueError

        Returns:
            ax: The axis object of the plot.
        """
        if zetan is not None:
            ind_zetan = np.argmin(
                np.abs(self.phi_eval[: self.num_phi_planes] - zetan * (2 * np.pi / self.nfp))
            )

        igeometry = self.igeometry

        import matplotlib.pyplot as plt

        fontsize_labels = None

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        else:
            plt.sca(ax)
            _ = plt.gcf()

        if kwargs.get("c") is None:
            kwargs.update({"c": "black"})
        if kwargs.get("ms") is None:
            kwargs.update({"ms": 0.8})

        kwargs.update({"lw": None})
        kwargs.update({"markeredgewidth": 0})

        for ivol in range(self.mvol):
            rz = self.RZ_total[ivol]

            if rz is None:
                continue

            if len(self.bad_traj_count[ivol]) > 0:
                for m in self.bad_traj_count[ivol]:
                    rz[m[0], :, m[1] :] = np.nan

            rr = rz[:, 0, ind_zetan :: self.num_phi_planes]
            zz = rz[:, 1, ind_zetan :: self.num_phi_planes]

            if prange == "full":
                ax.plot(rr, zz, "o", **kwargs)
            elif prange == "upper":
                ax.plot(rr[zz >= 0], zz[zz >= 0], ".", **kwargs)
            elif prange == "lower":
                ax.plot(rr[zz <= 0], zz[zz <= 0], ".", **kwargs)
            else:
                raise ValueError("prange should be one of ['full'(default), 'upper', 'lower'].")

        if igeometry == 3:
            ax.set_xlabel("R [m]", fontsize=fontsize_labels)
            ax.set_ylabel("Z [m]", fontsize=fontsize_labels)
        if igeometry == 2:
            ax.set_xlabel("X [m]", fontsize=fontsize_labels)
            plt.ylabel("Y [m]", fontsize=fontsize_labels)
        if igeometry == 1:
            ax.set_xlabel("Y [m]", fontsize=fontsize_labels)
            ax.set_ylabel("X [m]", fontsize=fontsize_labels)

        ax.set_title(
            f"Poincare at phi = {self.phi_eval[ind_zetan]:.2f}",
            fontsize=fontsize_labels if fontsize_labels is None else fontsize_labels + 12,
        )
        ax.axis("equal")
        plt.tight_layout()

        if show:
            plt.show()

        return ax

    def plot_iota(self, xaxis="R", ax=None, figsize=(6, 5), show=False, **kwargs):
        """Plots the iota profile obtained from the field line tracing. Iota is found as iota = (phi_final - phi_initial) / (theta_final - theta_initial).

        Args:
            xaxis (str, optional): What to plot on x-axis. Defaults to 'R'. Options: 'R', 's'.
            ax (_type_, optional): Existing axis object. Defaults to None.
            figsize (tuple, optional): Figure size. Defaults to (6, 5).
            show (bool, optional): Whether to call `plt.show()`. Defaults to False.
            kwargs: Additional keyword arguments for `ax.plot()`.

        Raises:
            RuntimeError
        """
        import matplotlib.pyplot as plt

        if kwargs.get("marker") is None:
            kwargs.update({"marker": "x"})
        if kwargs.get("ls") is None:
            kwargs.update({"ls": "--"})

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        else:
            plt.sca(ax)

        if xaxis == "R":
            x_data = self.RZ_total
        elif xaxis == "s":
            x_data = self.sth_total
        else:
            raise RuntimeError("xaxis must be one of ['R', 's'].")

        for ivol in range(self.mvol):
            if self.iotas[ivol] is None:
                continue
            ax.plot(x_data[ivol][:, 0, 0], self.iotas[ivol], **kwargs)

        plt.tight_layout()

        if show:
            plt.show()

    def save_h5(self, in_file="arrays.h5"):
        """Save the object to a .h5 file in the 'poincare_custom' group, including data and parameters. Does not allow for the loaded object to be re-run.

        Args:
            in_file (str/h5py.File): Where to store the data. If a string, it's used as the filename. If h5py.File, the File is directly appended to. Defaults to 'arrays.h5'.

        Raises:
            RuntimeError
        """
        flat_sth = [a.ravel() for a in self.sth_total]
        flat_RZ = [a.ravel() for a in self.RZ_total]
        flat_bad_traj_count = [
            np.array(a) if len(a) > 0 else np.array([[-1, -1]]) for a in self.bad_traj_count
        ]
        shapes = np.array([a.shape for a in self.sth_total])
        shapes_bad_traj_count = np.array([len(a) for a in flat_bad_traj_count])

        concat_sth = np.concatenate(flat_sth)
        concat_RZ = np.concatenate(flat_RZ)
        concat_iotas = np.concatenate(self.iotas)

        concat_bad_traj_count = np.concatenate(flat_bad_traj_count)

        if isinstance(in_file, str):
            f = h5py_File(in_file, "a")
        elif isinstance(in_file, h5py_File):
            f = in_file
        else:
            raise RuntimeError("in_file must be a string or h5py File object")

        grp = f.create_group("poincare_custom")

        grp.create_dataset("sth", data=concat_sth)
        grp.create_dataset("RZ", data=concat_RZ)
        grp.create_dataset("shapes", data=shapes)
        grp.create_dataset("phi_eval", data=self.phi_eval)
        grp.create_dataset("iotas", data=concat_iotas)
        grp.create_dataset("bad_traj_count", data=concat_bad_traj_count)
        grp.create_dataset("shapes_bad_traj_count", data=shapes_bad_traj_count)
        grp.attrs["num_phi_planes"] = self.num_phi_planes
        grp.attrs["num_transits"] = self.num_transits
        grp.attrs["num_pts_traj"] = self.num_pts_traj
        grp.attrs["nfp"] = self.nfp
        grp.attrs["igeometry"] = self.igeometry
        grp.attrs["mvol"] = self.mvol

        f.close()

    @classmethod
    def init_from_h5(cls, in_file="arrays.h5"):
        """Initialize a `Field_Tracer` object from a data saved in a .h5 file. The loaded object can NOT be used to redo the tracing, and is only fit for analysis,

        Args:
            in_file (str/h5py.File): From where to load the data. If a string, it's used as the filename. If h5py.File, the File is directly appended to. Defaults to 'arrays.h5'.

        Raises:
            RuntimeError

        Returns:
            Field_Tracer: The generated object.
        """
        if isinstance(in_file, str):
            f = h5py_File(in_file, "r")
        elif isinstance(in_file, h5py_File):
            f = in_file
        else:
            raise RuntimeError("in_file must be a string or h5py File object")

        grp = f["poincare_custom"]

        nfp = grp.attrs["nfp"]
        igeometry = grp.attrs["igeometry"]
        mvol = grp.attrs["mvol"]

        shapes = grp["shapes"][:]
        concat_sth = grp["sth"][:]
        concat_RZ = grp["RZ"][:]
        concat_iotas = grp["iotas"][:]
        concat_bad_traj_count = grp["bad_traj_count"][:]
        shapes_bad_traj_count = grp["shapes_bad_traj_count"][:]

        sth_total = []
        RZ_total = []
        iotas = []
        bad_traj_count = []
        acc = 0
        acc_iotas = 0
        acc_bad = 0
        for i in range(mvol):
            n1, n2, n3 = shapes[i]
            sth_total.append(concat_sth[acc : acc + n1 * n2 * n3].reshape((n1, n2, n3)))
            RZ_total.append(concat_RZ[acc : acc + n1 * n2 * n3].reshape((n1, n2, n3)))
            acc += n1 * n2 * n3

            iotas.append(concat_iotas[acc_iotas : acc_iotas + n1])
            acc_iotas += n1

            nb = shapes_bad_traj_count[i]
            bad_traj_count_temp = [
                tuple(concat_bad_traj_count[j]) for j in range(acc_bad, acc_bad + nb)
            ]
            bad_traj_count.append(bad_traj_count_temp if bad_traj_count_temp[0] != (-1, -1) else [])
            acc_bad += nb

        obj = cls(mvol, igeometry, nfp)
        obj.sth_total = sth_total
        obj.RZ_total = RZ_total
        obj.iotas = iotas
        obj.bad_traj_count = bad_traj_count

        obj.phi_eval = grp["phi_eval"][:]
        obj.num_phi_planes = grp.attrs["num_phi_planes"]
        obj.num_transits = grp.attrs["num_transits"]
        obj.num_pts_traj = grp.attrs["num_pts_traj"]

        f.close()

        return obj


def _solve_field_ode(
    sth_init,
    singular_vol,
    aze,
    ate,
    azo,
    ato,
    rac,
    rbc,
    zas,
    zbs,
    ras,
    rbs,
    zac,
    zbc,
    mpol,
    lrad,
    im,
    in_,
    phi_eval,
    phi_span,
    method,
    rtol,
    atol,
    rk_backend=None,
):
    """Trace the field-line ode, recording positions at `phi_eval` toroidal locations.

    The ODE integrator is selected by `rk_backend` (see RK_BACKENDS). All
    backends share the numba `bfield_flow_*` right-hand side and terminate when
    the field line leaves the volume (|s| > 1).

    Returns:
        ndarray[float]: s, theta points of the solution points
        ndarray[float]: R, Z points of the solution points
    """
    field_func = bfield_flow_sing if singular_vol else bfield_flow_nonsing
    args = (aze, ate, azo, ato, im, in_, lrad, mpol)

    sol_y = _integrate_trajectory(
        rk_backend, field_func, sth_init, phi_eval, phi_span, args, method, rtol, atol
    )

    RZarr = sth_to_RZ(
        rac,
        rbc,
        zas,
        zbs,
        ras,
        rbs,
        zac,
        zbc,
        im,
        in_,
        sol_y[0],
        sol_y[1],
        phi_eval,
        singular_vol,
    )

    return sol_y, RZarr


def _integrate_trajectory(
    rk_backend, field_func, sth_init, phi_eval, phi_span, args, method, rtol, atol
):
    """Dispatch a single field-line integration to the selected ODE backend.

    Returns the (2, n) array of (s, theta) states at the `phi_eval` planes that
    were reached strictly inside the volume.
    """
    if rk_backend is None:
        rk_backend = default_rk_backend()
    sth_init = np.asarray(sth_init, dtype=np.float64)

    if rk_backend == "scipy":
        return _integrate_scipy(field_func, sth_init, phi_eval, phi_span, args, method, rtol, atol)
    if rk_backend == "cyrk":
        return _integrate_cyrk(field_func, sth_init, phi_eval, phi_span, args, method, rtol, atol)
    raise ValueError(f"Unknown rk_backend {rk_backend!r}; expected one of {RK_BACKENDS}")


def _integrate_scipy(field_func, sth_init, phi_eval, phi_span, args, method, rtol, atol):
    """scipy.integrate.solve_ivp backend (BSD-3). Native event + t_eval support."""
    from scipy.integrate import solve_ivp

    def fun(t, y):
        dy = np.empty(2)
        field_func(dy, t, y, *args)
        return dy

    def edge(t, y):
        # Zero when |s| == 1, positive inside the volume, negative outside.
        return 1.0 - abs(y[0])

    edge.terminal = True
    edge.direction = -1.0

    solution = solve_ivp(
        fun,
        phi_span,
        sth_init,
        method=method,
        t_eval=phi_eval,
        rtol=rtol,
        atol=atol,
        dense_output=False,
        events=edge,
    )
    return solution.y


def _integrate_cyrk(field_func, sth_init, phi_eval, phi_span, args, method, rtol, atol):
    """CyRK backend (CC-BY-NC-SA, NonCommercial; optional dependency)."""
    from CyRK import pysolve_ivp

    at_volume_edge.terminal = True
    at_volume_edge.direction = 1.0
    solution = pysolve_ivp(
        field_func,
        phi_span,
        sth_init,
        method=method,
        args=args,
        rtol=rtol,
        atol=atol,
        t_eval=phi_eval,
        dense_output=False,
        pass_dy_as_arg=True,
        events=(at_volume_edge),
    )
    return solution.y


def _mpi_divide_work(mvol, comm):
    """Divides the work to different MPI processes. Rank 0 is supposed to get least amount of work, since it does more other work.

    Args:
        mvol (_type_): _description_
        comm (_type_): _description_

    Returns:
        int: MPI rank
        int: MPI size
        List[int]: Indices for which volumes current rank is responsible for.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    counts = np.full(size, mvol // size, dtype=int)
    counts[: mvol % size] += 1
    displs = np.insert(np.cumsum(counts), 0, 0)[0:-1]

    # Switch ranks 0 and (size-1), since rank 0 does more extra work
    if rank == 0:
        rank_mod = size - 1
    elif rank == (size - 1):
        rank_mod = 0
    else:
        rank_mod = rank

    start = displs[rank_mod]
    end = start + counts[rank_mod]
    local_indices = [i for i in range(start, end)]

    return rank, size, local_indices


def _mpi_gather_results(
    comm, rank, local_indices, mvol, num_pts, sth_local, RZ_local, bad_traj_count_local
):
    """Broadcast all results to rank 0.

    Returns:
        List[ndarray[float]]: s, theta of all trajectories
        List[ndarray[float]]: R, Z of all trajectories
        List[List[int]]: List of bad trajectories (where phi_eval[-1] hasn't been reached)
    """
    if comm.Get_size() == 1:
        return sth_local, RZ_local, bad_traj_count_local

    sth_total = None
    RZ_total = None
    if rank == 0:
        sth_total = [None] * mvol
        RZ_total = [None] * mvol

    for ivol in range(mvol):
        if ivol in local_indices and sth_local[ivol] is not None:
            sendbuf = sth_local[ivol].flatten()
            sendbuf2 = RZ_local[ivol].flatten()
            sendcount = len(sendbuf)
        else:
            sendbuf = np.empty(0, dtype=np.float64)
            sendbuf2 = np.empty(0, dtype=np.float64)
            sendcount = 0

        recvcounts = comm.gather(sendcount, root=0)

        if rank == 0:
            total = sum(recvcounts)
            recvbuf = np.empty(total, dtype=np.float64)
            recvbuf2 = np.empty(total, dtype=np.float64)
            displs2 = np.insert(np.cumsum(recvcounts), 0, 0)[0:-1]
        else:
            recvbuf = None
            recvbuf2 = None
            displs2 = None

        comm.Gatherv(sendbuf=sendbuf, recvbuf=(recvbuf, (recvcounts, displs2)), root=0)
        comm.Gatherv(sendbuf=sendbuf2, recvbuf=(recvbuf2, (recvcounts, displs2)), root=0)

        if rank == 0 and total > 0:
            sth_total[ivol] = recvbuf.copy().reshape(-1, 2, num_pts)
            RZ_total[ivol] = recvbuf2.copy().reshape(-1, 2, num_pts)

    bad_traj_count_temp = comm.gather(bad_traj_count_local, root=0)
    bad_traj_count_final = None
    if rank == 0:
        bad_traj_count_final = [[] for _ in range(mvol)]

        for proc_total in bad_traj_count_temp:
            for i, row in enumerate(proc_total):
                if row:
                    bad_traj_count_final[i] = row

    return sth_total, RZ_total, bad_traj_count_final


def _calc_all_iotas(mvol, phi_eval, sth_total, bad_traj_count):
    """Calculate iota of each trajectories.

    Returns:
        List[List[float]]: List of iotas calculated
    """
    iotas = [None] * mvol
    for ivol in range(mvol):
        if sth_total[ivol] is None:
            continue
        th_temp = sth_total[ivol][:, 1]
        end_pt_inds = -1 * np.ones(th_temp.shape[0], dtype=int)
        if len(bad_traj_count[ivol]) > 0:
            for j in bad_traj_count[ivol]:
                end_pt_inds[j[0]] = j[1] - 1

        th_temp = np.stack(
            [th_temp[:, 0], th_temp[(np.arange(th_temp.shape[0]), end_pt_inds)]], axis=0
        )
        iotas[ivol] = _calc_iota(phi_eval, th_temp)

    return iotas


def _calc_iota(phis, thetas):
    """Calculates field-line average iota, obtained from simple linear regression of theta vs phi curve for all toroidal transits. Vectorized in thetas, can be used for a 1d or Nd thetas array.

    Returns:
        float: The iota
    """
    d_phi = phis[-1] - phis[0]
    d_theta = thetas[-1] - thetas[0]
    iota = d_theta / d_phi

    return iota
