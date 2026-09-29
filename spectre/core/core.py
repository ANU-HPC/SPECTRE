# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

from logging import getLogger
from os import PathLike
from pathlib import Path

import numpy as np
import spectre4py_wrap
from mpi4py import MPI

from spectre.file_io import (
    InputParameters,
    read_input_parameters_from_toml,
    write_input_parameters_to_toml,
)
from spectre.input_dict import input_dict
from spectre.utils import get_xinit_specwrap, pad_dead_slot, unpack_interfaces, update_bnorm

from .helpers import allrzrz_from_interface_arrays

logger = getLogger(f"[{MPI.COMM_WORLD.Get_rank()}]" + __name__)


def _initialize(obj, input_params, name, verbose, comm):
    obj.name = name

    obj.comm = comm
    assert isinstance(verbose, bool), "verbose is either True or False."
    obj.verbose = verbose
    # Fortran libaries accessed via obj.lib
    obj.lib = spectre4py_wrap

    # wrap around modules
    modules = [
        "constants_mod",
        "numerical_mod",
        "io_mod",
        "cputiming_mod",
        "typedefns_mod",
        "input_list_mod",
        "allglobal_mod",
        "wrapper_funcs_mod",
        "packing_mod",
        "forces_mod",
        "geometry_mod",
        "field_mod",
        "hessian_mod",
    ]

    for key in modules:
        setattr(obj, key, getattr(obj.lib, key))

    # py2f converts the Python object to the Fortran integer identifying an MPI communicator.
    obj.allglobal_mod.set_mpi_comm(obj.comm.py2f())

    # Make the Fortran ouput according to the verbose option
    obj.io_mod.mute_output(not obj.verbose)
    if obj.comm.rank == 0:
        obj.last_set_params = input_params

    for group_name in obj.PYTHON_INPUTS:
        setattr(obj, f"{group_name}_params", input_params.__getattribute__(group_name))
    obj.parameters = input_params  # use the setter, initializes fortran

    # Broadcast minimization params into a 'min_params' attribute
    min_params_temp = obj.parameters.minimization.model_dump() if obj.comm.rank == 0 else None
    obj.min_params = input_dict(obj.comm.bcast(min_params_temp, root=0))

    # Set .h5 output filename
    obj.allglobal_mod.filename_h5 = obj.name + ".h5"


class SPECTRE:
    FORTRAN_INPUTS = ("physics", "numeric", "global_", "diagnostics")
    PYTHON_INPUTS = ("minimization",)

    def __init__(self, input_file: str | PathLike, name=None, verbose=True, comm=MPI.COMM_WORLD):
        """Initialize a SPECTRE object from a TOML file

        Args:
            input_file (str | PathLike): Path to the SPECTRE TOML input file.
            name (str, optional): Name for the SPECTRE run.
                Defaults to None, in which case the name is derived from input_file.
            verbose (bool): If wants screen outputs. (default: True).
            comm (MPI communicator): the comunicator assigned to SPECTRE. Default: MPI.COMM_WORLD
        Returns:
            SPECTRE object
        """

        if name is None:
            name = Path(input_file).stem

        input_params = read_input_parameters_from_toml(input_file)

        _initialize(self, input_params, name, verbose, comm)

    @classmethod
    def from_input_params(
        cls, input_params: InputParameters, name="spectre", verbose=True, comm=MPI.COMM_WORLD
    ):
        """
        Initialize a SPECTRE instance from an InputParameters instance.

        Args:
            input_params(InputParameters): An instance of InputParameters with the SPECTRE inputs.
            name (str, optional): Name for the SPECTRE run. Defaults to "spectre".
            verbose(bool): Whether or not to enable verbose output.
            comm(MPI communicator): MPI communicator to use.
        Returns:
            SPECTRE object
        """

        obj = cls.__new__(cls)

        _initialize(obj, input_params, name, verbose, comm)

        return obj

    @classmethod
    def from_deprecated_input_file(
        cls, input_file: str | PathLike, name=None, verbose=True, comm=MPI.COMM_WORLD
    ):
        """
        Initialize a SPECTRE instance from a SPECTRE input file.

        Args:
            input_file (str | PathLike): Path to the SPECTRE input file in TOML format.
            name (str, optional): Name for the SPECTRE run.
                Defaults to None, in which case the name is derived from input_file.
            verbose (bool): Whether or not to enable verbose output. Defaults to True.
            comm (MPI communicator): MPI communicator to use. Defaults to MPI.COMM_WORLD.

        Returns:
            SPECTRE instance initialized with the provided input file.
        """

        # print a deprecation error:
        import warnings

        from spectre.file_io.specfile_to_toml import _parameters_from_spec

        warnings.warn(
            "The 'from_deprecated_input_file' method is deprecated. "
            "Please convert your SPECTRE input files to TOML format and initialize with this.",
            DeprecationWarning,
            stacklevel=2,
        )

        input_params = _parameters_from_spec(input_file)

        if name is None:
            name = Path(input_file).stem  # first part of the string

        obj = cls.__new__(cls)

        _initialize(obj, input_params, name, verbose, comm)

        return obj

    def run(self, save_output: bool = False, infaces=None):
        """
        Args:
            save_output: Whether or not to save the hdf5 and restart files.
        """

        if save_output:
            self.allglobal_mod.ukip_write = False
            self.lib.sphdf5_mod.init_outfile()
            self.lib.sphdf5_mod.mirror_input_to_outfile()
        else:
            self.allglobal_mod.skip_write = True

        if infaces is not None:
            _ = unpack_interfaces(infaces, self)  # pack the interfaces

        self.lib.simulation_mod.spectre()

        self.lib.simulation_mod.final_diagnostics()

        if save_output:
            self.lib.sphdf5_mod.write_grid()
            self.lib.sphdf5_mod.hdfint()
            self.lib.sphdf5_mod.finish_outfile()

        if self.comm.rank == 0:
            # generate parameters from fortran memory and write to toml:
            self.wrapper_funcs_mod.prepare_endfile_writing()
            write_input_parameters_to_toml(
                self.parameters, f"{self.name}_end.toml"
            )  # write the final parameters to a toml file
        return

    @property
    def xin(self):
        """The current interface geometry packed as a DOF vector of length ``ngdof``."""
        return get_xinit_specwrap(self)

    def update_bns(self, mode="refresh", *, max_inner=20, tol_dtflux=1e-10, verbose=False):
        """Update the free-boundary normal field ``iBns`` from the plasma solution.

        The vacuum field is driven by ``iBns``, the normal field the plasma extrapolates
        onto the computational boundary by virtual casing. Two situations call for
        different treatments, selected by ``mode``:

        ``"refresh"``
            One damped update, ``iBns := gBnbld*iBns + (1-gBnbld)*ofmn``, taken from the
            plasma solution already sitting in Fortran memory. This is the step of the
            Picard free-boundary iteration and is what belongs *between* force solves.

        ``"initialize"``
            The one-shot bootstrap that replaces xspech's ``LautoinitBn`` block, for
            starting from a configuration whose ``Bns`` namelist does not yet correspond
            to the plasma. It forces ``gBnbld = 0`` so the update is a full replacement
            rather than a blend, settles the Beltrami solution with one ``dforce`` first
            so virtual casing has surface currents to integrate, then Newton-converges
            ``dtflux(Mvol)`` against the ``curtor`` constraint and propagates the result
            back into ``tflux``/``pflux``.

        ``"flux"``
            The ``dtflux`` Newton and flux write-back alone, leaving ``iBns`` at whatever
            the ``Bns`` namelist supplied. Use this where virtual casing is noisy
            (``relvcerr`` at or above 1 in the ``bnorml :`` log lines), since that noise
            can blow up the vacuum Beltrami solve; the flux convergence is unaffected by
            casing quality.

        There is deliberately no Picard loop inside ``"initialize"``: at fixed geometry the
        self-consistent ``iBns`` is uniquely determined, and where virtual casing struggles
        the alternating map is not a contraction, so iterating amplifies noise instead of
        damping it. Driving the geometry toward self-consistency is the outer optimizer's
        job.

        The plasma geometry is unchanged on exit; only the flux quantities at the
        plasma-vacuum boundary move.

        This is the only supported way to move ``iBns``. The Fortran free-boundary loop in
        ``spectre()`` steps it too, on every ``run()``, which is why the minimizer switches
        that loop off (``mfreeits``/``lautoinitbn``) and drives the iteration from here.

        Args:
            mode: ``"refresh"`` for the damped per-cycle update, ``"initialize"`` for the
                one-shot bootstrap, ``"flux"`` for the flux convergence alone.
            max_inner: Cap on the ``dtflux`` Newton step count (not used by ``"refresh"``).
            tol_dtflux: Convergence tolerance on ``|delta dtflux(Mvol)|`` (not used by
                ``"refresh"``).
            verbose: Print diagnostics on rank 0.

        Returns:
            ``bnserr``, the fixed-point residual ``mean|iBns - ofmn|`` of the ``iBns`` the
            field was solved at, i.e. measured *before* this call's update. It is therefore
            independent of the blend factor, and is the quantity to test for convergence of
            the alternating iteration. ``nan`` where no update applies: ``"flux"``, which
            leaves ``iBns`` alone, and any non-Picard case.
        """
        if mode not in ("refresh", "initialize", "flux"):
            raise ValueError(f"mode must be 'refresh', 'initialize' or 'flux', got {mode!r}")

        if int(self.input_list_mod.lfreebound) != 1:
            if verbose and self.comm.rank == 0:
                print("update_bns: not a free-boundary case; nothing to update.")
            return float("nan")

        # The direct Laplace virtual-casing solve carries the plasma field in the vacuum
        # operator itself, so it never reads iBns -- updating it would be a silent no-op.
        if int(self.input_list_mod.lvcvacuum) == 1:
            if verbose and self.comm.rank == 0:
                print("update_bns: lvcvacuum=1 ignores iBns; skipping the update.")
            return float("nan")

        if mode == "refresh":
            update_bnorm(self, first_free_bound=False, print_info=verbose)
            return float(self.allglobal_mod.bnserr)

        gbnbld_saved = float(self.input_list_mod.gbnbld)
        tflux_before = np.asarray(self.input_list_mod.tflux).copy()
        bnserr = float("nan")  # "flux" leaves iBns alone, so it has no residual to report
        try:
            ngdof = self.allglobal_mod.ngdof
            # dforce is a raw Fortran call, so both buffers keep SPEC's position(0:NGdof)
            # layout rather than the ngdof-wide Python one.
            x = pad_dead_slot(self.xin)
            force = np.zeros(ngdof + 1, dtype=np.float64, order="F")

            if mode == "initialize":
                # Full replacement: iBns := ofmn, not a blend with a value we do not trust.
                self.input_list_mod.gbnbld = 0.0
                ibns_before = np.asarray(self.allglobal_mod.ibns).copy()

                # Settle the Beltrami solution first so virtual casing has surface
                # currents to integrate.
                self.forces_mod.dforce(ngdof, x, force, False, True)
                self.wrapper_funcs_mod.update_bnorm()
                bnserr = float(self.allglobal_mod.bnserr)

                if verbose and self.comm.rank == 0:
                    ibns_now = np.asarray(self.allglobal_mod.ibns)
                    change = float(np.max(np.abs(ibns_now - ibns_before)))
                    print(
                        f"update_bns: iBns refreshed once (max|delta|={change:.3e}, "
                        f"residual of the incoming iBns bnserr={bnserr:.3e})"
                    )

                # Restore before any further dforce so the Newton cannot re-blend iBns.
                self.input_list_mod.gbnbld = gbnbld_saved

            for _ in range(max_inner):
                dt_before = float(self.allglobal_mod.dtflux[-1])
                self.forces_mod.dforce(ngdof, x, force, False, True)
                if abs(float(self.allglobal_mod.dtflux[-1]) - dt_before) < tol_dtflux:
                    break

            # dforce updates the per-volume dtflux/dpflux in place but leaves the
            # cumulative-normalized tflux/pflux in input_list_mod at the TOML guess: that
            # write-back lives in xspech's end-of-iteration sync, which calling dforce
            # directly bypasses. Without this a later preset call would rebuild the stale
            # state instead of the converged one.
            self._sync_tflux_pflux_from_dt_dp()

            if verbose and self.comm.rank == 0:
                # tflux is allocated (1:MNvol+1) with only the first Mvol entries real, so
                # index the vacuum volume explicitly rather than taking a trailing zero.
                ivac = int(self.allglobal_mod.mvol) - 1
                tf_vac = float(np.asarray(self.input_list_mod.tflux)[ivac])
                print(
                    f"update_bns: vacuum-volume tflux={tf_vac:.6e} "
                    f"(initial={float(tflux_before[ivac]):.6e}, "
                    f"delta={tf_vac - float(tflux_before[ivac]):+.6e})"
                )
        finally:
            self.input_list_mod.gbnbld = gbnbld_saved

        return bnserr

    def _sync_tflux_pflux_from_dt_dp(self):
        """Rebuild cumulative-normalized ``tflux``/``pflux`` from per-volume increments.

        Inverse of the diff-and-scale in ``preset_mod.initialize_internal_arrays``
        (``preset_mod.F90:410-418``): ``tflux[v] = sum_{k<=v} dtflux[k] * 2*pi/phiedge``,
        and likewise for ``pflux``, with the Igeometry 2/3 convention that ``pflux[0]``
        stays zero. Call this whenever ``dtflux``/``dpflux`` have been moved by code that
        does not round-trip through xspech's end-of-iteration sync.
        """
        phiedge = float(self.input_list_mod.phiedge)
        if phiedge == 0.0:
            return  # no scaling possible; nothing sensible to write

        norm = 2.0 * np.pi / phiedge
        mvol = int(self.allglobal_mod.mvol)
        igeometry = int(self.input_list_mod.igeometry)

        dtflux = self.allglobal_mod.dtflux
        dpflux = self.allglobal_mod.dpflux
        tflux = self.input_list_mod.tflux
        pflux = self.input_list_mod.pflux

        tflux[0] = float(dtflux[0]) * norm
        pflux[0] = 0.0 if igeometry in (2, 3) else float(dpflux[0]) * norm
        for v in range(1, mvol):
            tflux[v] = tflux[v - 1] + float(dtflux[v]) * norm
            pflux[v] = pflux[v - 1] + float(dpflux[v]) * norm

    @property
    def parameters(self) -> InputParameters:
        """
        retrieve input quantities from Fortran side and return a validated
        InputParameters instance
        """
        new_params = None
        if self.comm.rank == 0:
            newdict = dict()
            for (
                group_name
            ) in self.FORTRAN_INPUTS:  # iterate over groups, extract from fortran memory
                # print(f"Retrieving input group: {group_name}")
                this_dict = dict()
                newdict[group_name] = this_dict
                for field_name, _ in self.last_set_params.__getattribute__(group_name):
                    try:
                        field_value = self.input_list_mod.__getattribute__(field_name)
                        this_dict[field_name] = field_value
                    except AttributeError:
                        print(f"Unknown input key {field_name} in section {group_name}")
            # now fix the allRZRZ
            allrzrz = allrzrz_from_interface_arrays(
                self.allglobal_mod.irbc,
                self.allglobal_mod.izbs,
                self.allglobal_mod.irbs,
                self.allglobal_mod.izbc,
                self.allglobal_mod.mn,
                self.allglobal_mod.im,
                self.allglobal_mod.in_,
                self.allglobal_mod.mvol,
                self.input_list_mod.nfp,
            )
            newdict["physics"]["allrzrz"] = allrzrz
            # keep the python parameters unchanged:
            for group_name in self.PYTHON_INPUTS:
                newdict[group_name] = getattr(self, f"{group_name}_params").model_dump()
            new_params = InputParameters.model_validate(newdict)
        return new_params

    @parameters.setter
    def parameters(self, new_params: InputParameters):
        # SPECTRE reallocates many internal arrays and f90wraps pointer cache
        # can end up with swapped pointers. Clearing cache fixes this.
        # Must occur on all ranks
        self._clear_fortran_array_caches()

        if self.comm.rank == 0:
            self.last_set_params = new_params

            # first allocate all variables on the Fortran side:
            self.input_list_mod.initialize_inputs()
            self.wrapper_funcs_mod.alloc_allrzrz(
                new_params.physics.nvol, 128, 128
            )  # nasty hack cause allRZRZ is not fixed shape
            # then set all the variables by the appropriate keys in the dictionary:
            for group_name, parameter_group in new_params:
                if group_name in self.FORTRAN_INPUTS:
                    # print(f"Setting input group: {group_name}")
                    for field_name, field_value in parameter_group:
                        try:
                            self.input_list_mod.__setattr__(field_name, field_value)
                        except AttributeError:
                            print(f"Unknown input key {field_name} in section {group_name}")
            self.allglobal_mod.check_inputs()

        self.allglobal_mod.broadcast_inputs()
        self.lib.preset_mod.initialize_internal_arrays()

        return

    def _clear_fortran_array_caches(self):
        """
        On the python side, f90wrap keeps a cache of pointers to the Fortran arrays.
        Allocatable arrays that are reallocated en masse (when calling
        input_list_mod.initialize_inputs() for example), can shift the memory location,
        and python can end up with dangling pointers. This function clears those caches.
        see https://github.com/hiddenSymmetries/simsopt/pull/431 for this issue in simsopt
        """
        self.input_list_mod._arrays = {}
        self.allglobal_mod._arrays = {}

    def print_inputs(self):
        min_params = self.min_params
        physics = self.parameters.physics.model_dump()
        for k in [
            "rac",
            "zas",
            "ras",
            "zac",
            "rbc",
            "zbs",
            "rbs",
            "zbc",
            "zws",
            "rws",
            "zwc",
            "vns",
            "bns",
            "vnc",
            "bnc",
            "rwc",
            "pl",
            "ql",
            "pr",
            "qr",
            "lp",
            "lq",
            "rp",
            "rq",
            "allrzrz",
        ]:
            physics.pop(k)

        print("Minimization params:")
        for k in min_params:
            print(f"{k:23} {min_params[k]}")
        print("\nPhysics params:")
        for k in physics:
            print(f"{k:23} {physics[k]}")
        print()

    def create_spectreout(self):
        """Returns a ``SPECTREout`` object from the current ``SPECTRE`` instance.

        Creates a SPECTREout object containing the geometry and field
        information of the current instance. The field must be computed
        before calling this method.

        Returns:
            SPECTREout: A new object containing the geometry and field
                information of the current SPECTRE instance.
        """

        from spectre.postprocess import SPECTREout

        return SPECTREout.from_spectre(self)
