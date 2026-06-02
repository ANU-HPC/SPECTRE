# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

import functools
from logging import getLogger
from os import PathLike
from pathlib import Path

import numpy as np
import spectre4py_wrap
from mpi4py import MPI

from spectre import angle_targets, force_targets
from spectre.file_io import (
    InputParameters,
    read_input_parameters_from_toml,
    write_input_parameters_to_toml,
)
from spectre.input_dict import input_dict
from spectre.utils import _unpack_interfaces_impl, get_xinit_specwrap

from .helpers import allrzrz_from_interface_arrays


def _bind_xin_default(func):
    """Wrap a ``func(xin, test, *args, **kwargs)`` as a SPECTRE method.

    The resulting method takes ``self`` (= the SPECTRE instance, passed as
    ``test`` to ``func``) and an optional ``xin`` that defaults to
    ``self.xin``. All other positional and keyword arguments pass through.
    """

    @functools.wraps(func)
    def method(self, xin=None, *args, **kwargs):
        if xin is None:
            xin = self.xin
        return func(xin, self, *args, **kwargs)

    return method

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
            self.xin = infaces

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

    @property
    def xin(self):
        """Current interface-DOF vector packed from iRbc/iZbs/iRbs/iZbc.

        Returns a length-``(ngdof + 1)`` Fortran-contiguous numpy array.
        Reading is a fresh pull from Fortran state; setting unpacks back
        into the same arrays.
        """
        return get_xinit_specwrap(self)

    @xin.setter
    def xin(self, value):
        _unpack_interfaces_impl(value, self)

    @property
    def unpacked_interfaces(self):
        """Refs to the Fortran iRbc, iZbs, iRbs, iZbc interface-coefficient arrays.

        Read-only — returns the same arrays that ``xin = ...`` writes into. To
        change them, set ``xin`` (or write into the returned arrays in place,
        which mutates Fortran state directly).
        """
        return (
            self.allglobal_mod.irbc,
            self.allglobal_mod.izbs,
            self.allglobal_mod.irbs,
            self.allglobal_mod.izbc,
        )

    def update_bns(self, max_inner=20, tol_dtflux=1e-10,
                   refresh_ibns=True, verbose=False):
        """Drive ``dtflux(Mvol)`` Newton to convergence, optionally refresh ``iBns``.

        Replaces xspech's ``LautoinitBn`` block, called explicitly. Designed
        as a single deterministic pass — no Picard outer loop:

        1. (If ``refresh_ibns=True``) one ``dforce`` at current ``iBns`` to
           settle the Beltrami solution, then one ``update_bnorm`` to set
           ``iBns := ofmn`` from the plasma's virtual-casing extrapolation.
        2. Newton-converge ``dtflux(Mvol)`` against the curpol constraint at
           the (possibly refreshed) ``iBns`` by repeated ``dforce`` calls.
        3. Propagate the Newton-converged ``dtflux/dpflux`` back into the
           cumulative-normalized ``tflux/pflux`` in ``input_list_mod`` so a
           future ``preset_mod.initialize_internal_arrays`` reconstructs the
           same state rather than reverting to the TOML guess.

        Why no outer Picard loop: at fixed ``xin`` the self-consistent
        ``iBns`` is uniquely determined — it's the boundary term that makes
        the vacuum solution match the plasma's surface currents. If
        ``compute_bnorm_plasmab`` was numerically clean, a single refresh
        would already be self-consistent. In practice, at geometries where
        virtual casing struggles (``relvcerr`` ~ 1 or worse in the
        ``bnorml :`` log lines), the alternating map
        ``iBns_{n+1} = ofmn(plasma at iBns_n)`` is not a contraction —
        iterating amplifies noise rather than damping it. The single-shot
        design accepts that one refresh is the best we can do at the
        starting geometry, and lets the outer optimizer drive ``xin``
        toward configurations where ``iBns`` actually is self-consistent.

        When ``relvcerr`` is bad, the refreshed ``iBns`` carries
        virtual-casing noise that can blow up the vacuum-volume Beltrami
        solve. Pass ``refresh_ibns=False`` to skip the refresh entirely and
        keep whatever ``preset_mod`` loaded from the TOML's ``Bns`` namelist
        — you still get the dtflux(Mvol) Newton convergence, which is
        unaffected by casing quality.

        The plasma geometry (``self.xin``) is unchanged on exit; only the
        flux scalars-and-arrays at the plasma-vacuum boundary are.

        Args:
            max_inner: Cap on the dtflux Newton step count.
            tol_dtflux: Convergence tolerance for ``|delta dtflux(Mvol)|``.
            refresh_ibns: If True (default), do the one-shot ``iBns``
                refresh in step 1 above. If False, skip it.
            verbose: Print diagnostics on rank 0.
        """
        gbnbld_saved = float(self.input_list_mod.gbnbld)

        try:
            x = self.xin
            ngdof = self.allglobal_mod.ngdof
            force = np.zeros(ngdof + 1, dtype=np.float64, order="F")

            def _dtflux_newton_to_convergence():
                """Repeated dforce at fixed x until dtflux(Mvol) step < tol."""
                for _ in range(max_inner):
                    dt_before = float(self.allglobal_mod.dtflux[-1])
                    self.forces_mod.dforce(ngdof, x, force, False, True)
                    dt_after = float(self.allglobal_mod.dtflux[-1])
                    if abs(dt_after - dt_before) < tol_dtflux:
                        return

            if refresh_ibns:
                # One-shot iBns refresh, mirroring xspech's LautoinitBn block.
                # Force gBnbld=0 so update_bnorm does a full replacement,
                # iBns := ofmn (not a blend).
                self.input_list_mod.gbnbld = 0.0
                ibns_before = np.asarray(self.allglobal_mod.ibns).copy()

                # Settle the Beltrami solution at the TOML's iBns first so
                # compute_bnorm_plasmab has surface currents to integrate.
                self.forces_mod.dforce(ngdof, x, force, False, True)
                self.wrapper_funcs_mod.update_bnorm()

                if verbose and self.comm.rank == 0:
                    ibns_after = np.asarray(self.allglobal_mod.ibns)
                    ibns_change = float(np.max(np.abs(ibns_after - ibns_before)))
                    print(
                        f"update_bns: iBns refreshed once "
                        f"(max|delta|={ibns_change:.3e})"
                    )

                # Restore user's gBnbld before further dforce so the
                # dtflux-only convergence below doesn't accidentally
                # re-blend iBns inside dforce's dfp100.
                self.input_list_mod.gbnbld = gbnbld_saved

            # Newton-converge dtflux(Mvol) at the (possibly refreshed) iBns.
            _dtflux_newton_to_convergence()

            if verbose and self.comm.rank == 0:
                print(
                    "update_bns: dtflux(Mvol)="
                    f"{float(self.allglobal_mod.dtflux[-1]):.6e}"
                )

            # Sync cumulative-normalized tflux/pflux from per-volume
            # dtflux/dpflux so any future preset call (e.g. xspech via
            # SPECTRE.run(), or a fresh SPECTRE from the same TOML)
            # reconstructs the Newton-converged state rather than reverting
            # to the stale TOML guess. Inverse of preset_mod.F90:386-407.
            self._sync_tflux_pflux_from_dt_dp()
        finally:
            self.input_list_mod.gbnbld = gbnbld_saved

    def _sync_tflux_pflux_from_dt_dp(self):
        """Reconstruct cumulative-normalized tflux/pflux from per-volume dtflux/dpflux.

        Inverse of the diff-and-scale done inside ``preset_mod.initialize_internal_arrays``
        (``preset_mod.F90:386-407``). After this returns:

        * ``input_list_mod.tflux[v] = sum_{k<=v}(dtflux[k]) * 2*pi / phiedge``
        * ``input_list_mod.pflux`` mirrors the same for dpflux, with the
          Igeometry-2/3 convention that ``pflux[0]`` is left at zero.

        Run this any time ``dtflux`` or ``dpflux`` has been updated by code
        that doesn't itself round-trip through xspech's end-of-iteration
        sync (``simulation_mod.F90:370-381``).
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

        # Cumulative sum of the per-volume increments (in physical units),
        # then renormalize to the TOML's cumulative-normalized convention.
        tflux[0] = float(dtflux[0]) * norm
        pflux[0] = 0.0 if igeometry in (2, 3) else float(dpflux[0]) * norm
        for v in range(1, mvol):
            tflux[v] = tflux[v - 1] + float(dtflux[v]) * norm
            pflux[v] = pflux[v - 1] + float(dpflux[v]) * norm

    # ------------------------------------------------------------------
    # Bound force / angle targets — every entry below is a wrapper around the
    # corresponding free function in `spectre.force_targets` /
    # `spectre.angle_targets`. The instance method takes ``xin`` as a keyword
    # defaulting to ``self.xin``, so ``test.force_real()`` evaluates at the
    # current Fortran-side geometry without the caller having to pull xin out
    # explicitly. Passing ``test.force_real(some_xin)`` still works for
    # off-geometry evaluations (e.g. inside an optimizer).
    # ------------------------------------------------------------------
    force_real = _bind_xin_default(force_targets.force_real)
    force_real_jac = _bind_xin_default(force_targets.force_real_jac)
    force_modes = _bind_xin_default(force_targets.force_modes)
    force_modes_targetmn = _bind_xin_default(force_targets.force_modes_targetmn)
    force_modes_jac = _bind_xin_default(force_targets.force_modes_jac)
    force_modes_speclike = _bind_xin_default(force_targets.force_modes_speclike)
    force_modes_jac_speclike = _bind_xin_default(force_targets.force_modes_jac_speclike)
    force_modes_dforce = _bind_xin_default(force_targets.force_modes_dforce)
    force_modes_dforce_with_grad = _bind_xin_default(force_targets.force_modes_dforce_with_grad)
    angular_force_real = _bind_xin_default(angle_targets.angular_force_real)
    angular_force_likespec = _bind_xin_default(angle_targets.angular_force_likespec)
    angular_force_real_jac = _bind_xin_default(angle_targets.angular_force_real_jac)
    angular_force_jac_likespec = _bind_xin_default(angle_targets.angular_force_jac_likespec)
    len_cons_real = _bind_xin_default(angle_targets.len_cons_real)
    len_cons_real_jac = _bind_xin_default(angle_targets.len_cons_real_jac)
    get_theta_chord_length = _bind_xin_default(angle_targets.get_theta_chord_length)
    get_theta_chord_length_jac = _bind_xin_default(angle_targets.get_theta_chord_length_jac)

    def to_toml(self, filename):
        """Serialize the current SPECTRE state to a TOML file.

        The written TOML captures the *current* contents of Fortran-side
        memory (geometry, fluxes, Bns, etc.), not the original
        construction-time inputs. Call this after ``update_bns()`` to
        capture the refreshed ``iBns``; or after any other state mutation
        you want to inspect or hand off.

        Args:
            filename: Output path. The caller's responsibility to make sure
                the parent directory exists.
        """
        if self.comm.rank == 0:
            # Mirror what self.run() does just before its TOML write: the
            # Fortran side has bookkeeping (e.g. lbnszero handling) that
            # prepare_endfile_writing settles before we serialize.
            self.wrapper_funcs_mod.prepare_endfile_writing()
            write_input_parameters_to_toml(self.parameters, filename)
        self.comm.Barrier()

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
