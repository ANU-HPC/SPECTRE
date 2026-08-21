# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

import functools
import warnings
from logging import getLogger
from os import PathLike
from pathlib import Path

import numpy as np
import spectre4py_wrap
from mpi4py import MPI

import spectre.angle_targets as angle_targets
import spectre.force_targets as force_targets
from spectre.file_io import (
    InputParameters,
    read_input_parameters_from_toml,
    write_input_parameters_to_toml,
)
from spectre.input_dict import input_dict
from spectre.utils import _set_interfaces_from_xin, get_xinit_specwrap, pad_dead_slot

from .helpers import allrzrz_from_interface_arrays

logger = getLogger(f"[{MPI.COMM_WORLD.Get_rank()}]" + __name__)

# dforce passes allowed to settle dtflux(Mvol) in a free-boundary sweep (settles in 2-3).
_MAX_FLUX_PASSES = 10


def _bind_xin_default(func):
    """Bind a ``func(xin, spectre, ...)`` helper as a ``SPECTRE`` method."""

    @functools.wraps(func)
    def method(self, xin=None, *args, **kwargs):
        return func(self.xin if xin is None else xin, self, *args, **kwargs)

    return method


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

    # Snapshot of the geometry the Beltrami field was last solved for (packed xin),
    # used by `field_is_current` / `create_spectreout` to detect a stale field.
    # None means "no solve tracked on the python side yet". _field_solved_at additionally
    # keys the compute_fields skip on the exact solve arguments.
    obj._field_fingerprint = None
    obj._field_solved_at = None


class SPECTRE:
    FORTRAN_INPUTS = ("physics", "numeric", "global_", "diagnostics")
    PYTHON_INPUTS = ("minimization",)

    @functools.cached_property
    def inspect(self):
        """Return the inspection and visualization facade bound to this instance."""
        from .inspect import _Inspect

        return _Inspect(self)

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
        # Volumes and the adiabatic constants of the pressure profile, once per input load
        # as the removed Fortran driver did; every dforce-based consumer reads them.
        self.wrapper_funcs_mod.initialize_adiabatic()

        # initialize_internal_arrays() zeros the field, so any tracked solve is now stale
        self._field_fingerprint = None
        self._field_solved_at = None

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
        """Return the current packed interface degrees of freedom."""
        return get_xinit_specwrap(self)

    @xin.setter
    def xin(self, value):
        _set_interfaces_from_xin(value, self)

    @property
    def unpacked_interfaces(self):
        """Return references to the four Fortran interface-coefficient arrays."""
        return (
            self.allglobal_mod.irbc,
            self.allglobal_mod.izbs,
            self.allglobal_mod.irbs,
            self.allglobal_mod.izbc,
        )

    def unpack_interfaces(self, xin=None):
        """Unpack interface degrees of freedom into independent arrays.

        Args:
            xin (numpy.ndarray, optional): Packed interface degrees of freedom.
                Defaults to the live :attr:`xin` state.

        Returns:
            tuple[numpy.ndarray, ...]: Copies of ``irbc``, ``izbs``, ``irbs``, and
            ``izbc`` corresponding to ``xin``. The live Fortran state is unchanged.
        """
        xin = self.xin if xin is None else xin
        irbc = self.allglobal_mod.irbc.copy(order="F")
        izbs = self.allglobal_mod.izbs.copy(order="F")
        irbs = self.allglobal_mod.irbs.copy(order="F")
        izbc = self.allglobal_mod.izbc.copy(order="F")
        self.packing_mod.packxi(
            self.allglobal_mod.ngdof,
            pad_dead_slot(xin),
            self.allglobal_mod.mvol,
            self.allglobal_mod.mn,
            self.allglobal_mod.nt,
            self.allglobal_mod.nz,
            irbc,
            izbs,
            irbs,
            izbc,
            "U",
            False,
            True,
        )
        return irbc, izbs, irbs, izbc

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

    def compute_coordinates(self, lvol, s, lcurvature, nt, nz, mn):
        """Compute coordinates and metric arrays with caller-owned storage.

        Args:
            lvol (int): One-based volume index.
            s (float): Radial coordinate in ``[-1, 1]``.
            lcurvature (int): Highest derivative order to compute.
            nt (int): Number of poloidal grid points.
            nz (int): Number of toroidal grid points.
            mn (int): Number of Fourier harmonics.

        Returns:
            tuple[numpy.ndarray, ...]: ``rij``, ``zij``, ``sg``, ``inv_jac``,
            and ``guvij``.
        """
        ntz = nt * nz
        rij = np.zeros((ntz, 4, 4), dtype=np.float64, order="F")
        zij = np.zeros((ntz, 4, 4), dtype=np.float64, order="F")
        sg = np.zeros((ntz, 4), dtype=np.float64, order="F")
        inv_jac = np.zeros(ntz, dtype=np.float64, order="F")
        guvij = np.zeros((ntz, 4, 4, 5), dtype=np.float64, order="F")
        coordinate_singularity = lvol == 1 and self.input_list_mod.igeometry != 1

        self.geometry_mod.compute_coordinates(
            lvol,
            s,
            lcurvature,
            nt,
            nz,
            mn,
            coordinate_singularity,
            rij,
            zij,
            sg,
            inv_jac,
            guvij,
        )
        return rij, zij, sg, inv_jac, guvij

    @property
    def field_is_current(self) -> bool:
        """Whether the Beltrami field in Fortran memory matches the current geometry.

        Compares the geometry used by the last tracked field solve against the geometry
        currently in memory, using the packed interface vector as a fingerprint.

        Returns ``False`` when no solve has been tracked through the Python API,
        or when the live interfaces have changed since the tracked solve.

        Returns:
            bool: ``True`` if a tracked solve matches the current geometry.
        """
        return self._field_fingerprint is not None and np.array_equal(
            self._field_fingerprint, get_xinit_specwrap(self)
        )

    def _record_field_solve(self, xin=None, recompute_axis=None):
        """Record that the field matches the live interface geometry.

        Args:
            xin: The exact packed vector the solve was called with, recorded for the
                :meth:`compute_fields` skip. The geometry fingerprint cannot serve there:
                the pack/unpack round trip is not bit-stable against a caller's vector.
            recompute_axis: The axis flag of that solve (part of the skip key).
        """
        self._field_fingerprint = self.xin.copy()
        if xin is not None:
            self._field_solved_at = (np.array(xin, copy=True), bool(recompute_axis))

    def compute_fields(self, recompute_axis=True, xin=None):
        """Solve the Beltrami field for an interface geometry.

        Populates the magnetic vector potential in Fortran memory for the current ``xin``
        and records the geometry as the tracked solve, so :attr:`field_is_current` becomes
        ``True``. This is the lightweight alternative to :meth:`create_spectreout` (which
        additionally builds a full ``SPECTREout`` postprocess object) for when you only need
        the in-memory field, e.g. before evaluating Beltrami errors.

        The solve is skipped when the field in memory was already solved by this method
        (or :meth:`compute_field_derivatives`) for bitwise the same ``xin`` and axis flag,
        the live geometry still matches it (:attr:`field_is_current`) and nothing
        invalidated it since: any :meth:`dforce` (it re-solves the field, possibly at
        another ``dtflux``), an ``update_bns`` refresh moving ``iBns``, or a geometry
        reset. Direct edits of ``input_list_mod`` arrays are not seen (the blind spot
        :attr:`field_is_current` has). Repeated force evaluations at one iterate therefore
        pay for one solve.

        Args:
            recompute_axis (bool): Recompute the coordinate axis as part of the solve
                (matches the ``force_real`` default). Defaults to ``True``.
            xin (numpy.ndarray, optional): Packed interface degrees of freedom.
                Defaults to the live :attr:`xin` state.
        """
        xin = self.xin if xin is None else xin
        solved = self._field_solved_at
        if (
            solved is not None
            and solved[1] == bool(recompute_axis)
            and np.array_equal(solved[0], xin)
            and self.field_is_current
        ):
            return
        self.field_mod.solve_field(pad_dead_slot(xin), recompute_axis)
        self._record_field_solve(xin, recompute_axis)

    def compute_field_derivatives(self, mpol_rz, ntor_rz, recompute_axis=True, xin=None):
        """Solve the Beltrami field and its interface derivatives.

        Args:
            mpol_rz (int): Maximum poloidal mode for the derivatives.
            ntor_rz (int): Maximum toroidal mode for the derivatives.
            recompute_axis (bool): Recompute the coordinate axis as part of the solve.
                Defaults to ``True``.
            xin (numpy.ndarray, optional): Packed interface degrees of freedom.
                Defaults to the live :attr:`xin` state.
        """
        xin = self.xin if xin is None else xin
        self.field_mod.solve_field_deriv(pad_dead_slot(xin), mpol_rz, ntor_rz, recompute_axis)
        self._record_field_solve(xin, recompute_axis)

    def dforce(self, xin=None, compute_derivatives=False, compute_axis=True):
        """Evaluate the spectral force with ``forces_mod.dforce``, re-solving the field at ``xin``.

        The Fortran position/force layout (a dead slot 0 ahead of the ``ngdof`` entries) is
        handled here; callers see plain ``ngdof``-length vectors. Because the field is
        re-solved (possibly at another ``dtflux``), the :meth:`compute_fields` skip is
        dropped.

        Args:
            xin (numpy.ndarray, optional): Packed interface degrees of freedom. Defaults to
                the live :attr:`xin` state.
            compute_derivatives (bool): ``LComputeDerivatives`` of ``dforce`` (fills the
                hessian storage allocated by ``alloc_for_hessian``). Defaults to ``False``.
            compute_axis (bool): ``LComputeAxis`` of ``dforce``. Defaults to ``True``.

        Returns:
            numpy.ndarray: The ``ngdof`` force harmonics.
        """
        xin = self.xin if xin is None else xin
        ngdof = int(self.allglobal_mod.ngdof)
        force = np.zeros(ngdof + 1, dtype=np.float64, order="F")
        self.forces_mod.dforce(ngdof, pad_dead_slot(xin), force, compute_derivatives, compute_axis)
        self._field_solved_at = None
        return force[1:]

    def update_bns(
        self,
        mode="converge",
        *,
        damping=None,
        tol=None,
        max_sweeps=20,
        tol_dtflux=1e-10,
        verbose=False,
    ):
        """Converge the free-boundary state ``(iBns, dtflux(Mvol))`` by damped Picard sweeps.

        At fixed plasma geometry the coupled state obeys an affine sweep map (the vacuum
        Beltrami solve is linear in it), so a single sweep cannot reach the fixed point,
        while the damped iteration contracts reliably: measured spectral radius ~2 undamped
        at fresh geometries, ~0.4 near the fixed point, robust damping band
        ``gbnbld in [0.3, 0.7]`` (``docs/notes/ibns_picard_investigation.md`` on the
        ``optimizer_overhaul`` branch). Every mode is MPI-collective: all ranks call
        together, and the identical numpy blend on each rank keeps the state consistent.

        Modes:
            ``"flux"``: repeat ``dforce`` until ``|delta dtflux(Mvol)| < tol_dtflux`` (the
                linking-current constraint is linear, so this settles in 2-3 passes), then
                write the cumulative ``tflux``/``pflux`` back. Leaves ``iBns`` untouched.
            ``"refresh"``: one damped sweep: flux-converge, extrapolate the plasma field
                onto the computational boundary by virtual casing, measure the pre-blend
                residual ``bnserr = mean|iBns - ofm|``, then blend
                ``iBns := damping*iBns + (1-damping)*ofm``.
            ``"converge"`` (default): ``"refresh"`` sweeps until ``bnserr < tol`` or
                ``max_sweeps``, bailing out with a warning if the residual grows.
            ``"initialize"``: bootstrap replacing the legacy ``LautoinitBn``: one
                full-replacement sweep (``damping=0``) integrating the plasma-side surface
                current (``first_free_bound``), then ``"converge"``.

        Args:
            mode: One of ``"flux"``, ``"refresh"``, ``"converge"``, ``"initialize"``.
            damping: Blend factor in ``iBns := damping*iBns + (1-damping)*ofm``. Defaults
                to ``global.gbnbld``; ``1.0`` freezes ``iBns``, ``0.0`` replaces it.
            tol: ``"converge"``/``"initialize"`` stop once the pre-blend ``bnserr`` falls
                below this. Defaults to ``global.gbntol``.
            max_sweeps: Cap on the refresh sweeps. ``0`` with ``mode="initialize"`` is the
                bootstrap sweep alone, without the converge tail.
            tol_dtflux: Convergence tolerance on ``|delta dtflux(Mvol)|``.
            verbose: Print per-sweep diagnostics on rank 0.

        Returns:
            The last pre-blend ``bnserr`` (independent of ``damping``); ``nan`` for
            ``"flux"`` and for the declared no-op cases (``lvcvacuum = 1``, ``lbnszero``).

        Raises:
            ValueError: Unknown ``mode``.
            RuntimeError: Not a free-boundary case (``lfreebound != 1``).
        """
        if mode not in ("flux", "refresh", "converge", "initialize"):
            raise ValueError(
                f"mode must be 'flux', 'refresh', 'converge' or 'initialize', got {mode!r}"
            )
        if int(self.input_list_mod.lfreebound) != 1:
            raise RuntimeError("update_bns requires a free-boundary case (lfreebound = 1)")

        # The direct Laplace virtual-casing solve carries the plasma field in the vacuum
        # operator itself, so it never reads iBns -- updating it would be a silent no-op.
        if int(self.input_list_mod.lvcvacuum) == 1:
            if verbose and self.comm.rank == 0:
                print("update_bns: lvcvacuum=1 ignores iBns; skipping the update.")
            return float("nan")

        # lbnszero pins iBns = 0 (the vacuum-verification hack); there is nothing to converge.
        if bool(self.input_list_mod.lbnszero):
            if verbose and self.comm.rank == 0:
                print("update_bns: lbnszero pins iBns = 0; skipping the update.")
            return float("nan")

        lam = float(self.input_list_mod.gbnbld) if damping is None else float(damping)
        tol = float(self.input_list_mod.gbntol) if tol is None else float(tol)

        if mode == "flux":
            self._converge_dtflux(tol_dtflux, verbose)
            return float("nan")

        if mode == "refresh":
            return self._refresh_sweep(lam, False, tol_dtflux, verbose)

        if mode == "initialize":
            # Full replacement from the plasma-side surface current: the incoming iBns is
            # not trusted, so nothing of it may survive the first sweep.
            bnserr = self._refresh_sweep(0.0, True, tol_dtflux, verbose)
            if verbose and self.comm.rank == 0:
                print(f"update_bns: initialized iBns (incoming residual bnserr={bnserr:.3e})")
        else:
            bnserr = float("nan")

        # "converge" proper; also the tail of "initialize".
        prev = None
        for sweep in range(max_sweeps):
            bnserr = self._refresh_sweep(lam, False, tol_dtflux, verbose)
            if verbose and self.comm.rank == 0:
                print(f"update_bns: sweep {sweep} bnserr={bnserr:.6e} (tol {tol:.1e})")
            if bnserr < tol:
                break
            if prev is not None and bnserr > prev:
                warnings.warn(
                    f"update_bns: bnserr grew ({prev:.3e} -> {bnserr:.3e}); the damped "
                    f"Picard map is not contracting at damping={lam}. Bailing out; "
                    "consider a damping closer to 1.",
                    stacklevel=2,
                )
                break
            prev = bnserr
        return bnserr

    def _converge_dtflux(self, tol_dtflux, verbose=False):
        """Repeat ``dforce`` until ``dtflux(Mvol)`` settles, then write back ``tflux``/``pflux``.

        ``dforce`` adjusts the vacuum-volume ``dtflux`` to hold the linking current
        ``curtor``; the constraint is linear in ``dtflux``, so this settles in a few
        passes (capped at ``_MAX_FLUX_PASSES``).
        """
        x = self.xin
        ivac = int(self.allglobal_mod.mvol) - 1  # dtflux is allocated (1:Mvol)
        for _ in range(_MAX_FLUX_PASSES):
            dt_before = float(self.allglobal_mod.dtflux[ivac])
            self.dforce(x)
            if abs(float(self.allglobal_mod.dtflux[ivac]) - dt_before) < tol_dtflux:
                break
        self._sync_tflux_pflux_from_dt_dp()
        if verbose and self.comm.rank == 0:
            tf_vac = float(np.asarray(self.input_list_mod.tflux)[ivac])
            print(f"update_bns: vacuum-volume tflux={tf_vac:.6e} after flux convergence")

    def _refresh_sweep(self, lam, first_free_bound, tol_dtflux, verbose):
        """One damped Picard sweep at fixed geometry; returns the pre-blend ``bnserr``."""
        self._converge_dtflux(tol_dtflux, verbose)

        mn = int(self.allglobal_mod.mn)
        efm = np.zeros(mn, dtype=np.float64, order="F")
        ofm = np.zeros(mn, dtype=np.float64, order="F")
        # first_free_bound selects the virtual-casing surface current: the plasma-side
        # current for a bootstrap (no vacuum solution exists yet), the outer face otherwise.
        # Call-scoped: a later lvcvacuum=1 instance would trip a fatal if it leaked.
        try:
            self.allglobal_mod.first_free_bound = first_free_bound
            self.wrapper_funcs_mod.get_bnorm_plasmab(mn, efm, ofm)
        finally:
            self.allglobal_mod.first_free_bound = False

        stellsym = bool(self.input_list_mod.enforce_stell_sym)
        # A NaN here (degenerate trial interface, failed cubature) would poison iBns for
        # good and defeat every stopping test, so it is an error, not a warning.
        if not (np.all(np.isfinite(ofm[1:mn])) and (stellsym or np.all(np.isfinite(efm[:mn])))):
            raise FloatingPointError("update_bns: virtual casing returned non-finite Bn harmonics")
        ibns = self.allglobal_mod.ibns
        ibnc = self.allglobal_mod.ibnc

        # Fixed-point residual of the iBns the field was just solved at, measured before
        # the blend so it is independent of the damping factor.
        bnserr = 0.0 if mn == 1 else float(np.sum(np.abs(ibns[1:mn] - ofm[1:mn]))) / (mn - 1)
        if not stellsym:
            bnserr += float(np.sum(np.abs(ibnc[:mn] - efm[:mn]))) / mn

        # Damped blend, in place: prepare_endfile_writing and /output/Bns read these
        # arrays. The m = n = 0 sine mode is identically zero and is never blended.
        ibns[1:mn] = lam * ibns[1:mn] + (1.0 - lam) * ofm[1:mn]
        if not stellsym:
            ibnc[:mn] = lam * ibnc[:mn] + (1.0 - lam) * efm[:mn]

        # Nothing on the Fortran side sets bnserr; the output writer reads it from here.
        self.allglobal_mod.bnserr = bnserr

        # iBns moved: the Beltrami field in memory no longer matches the free-boundary
        # state, even though the geometry is unchanged.
        self._field_fingerprint = None
        self._field_solved_at = None
        return bnserr

    def _sync_tflux_pflux_from_dt_dp(self):
        """Rebuild cumulative-normalized ``tflux``/``pflux`` from the per-volume increments.

        Inverse of the diff-and-scale in ``preset_mod.initialize_internal_arrays``:
        ``tflux[v] = sum_{k<=v} dtflux[k] * 2*pi/phiedge`` and likewise for ``pflux``,
        with the Igeometry 2/3 convention that ``pflux[0]`` stays zero. ``tflux`` is
        allocated ``(1:MNvol+1)`` with only the first ``Mvol`` entries real, so the
        volumes are indexed explicitly rather than from the end. Call this whenever
        ``dtflux``/``dpflux`` have moved by code that does not go through a Fortran
        resync.
        """
        phiedge = float(self.input_list_mod.phiedge)
        if phiedge == 0.0:
            return  # no scaling possible; nothing sensible to write

        pi2 = 2.0 * np.pi
        mvol = int(self.allglobal_mod.mvol)
        igeometry = int(self.input_list_mod.igeometry)

        dtflux = np.asarray(self.allglobal_mod.dtflux)
        dpflux = np.asarray(self.allglobal_mod.dpflux)

        # Mirror the Fortran resync arithmetic exactly (cumulate the raw increments, then
        # scale the sums by pi2/phiedge) so the write-back is bit-identical to the values
        # spectre() used to produce.
        t = np.empty(mvol)
        p = np.empty(mvol)
        t[0] = dtflux[0]
        p[0] = dpflux[0] if igeometry == 1 else 0.0
        for v in range(1, mvol):
            t[v] = t[v - 1] + dtflux[v]
            p[v] = p[v - 1] + dpflux[v]

        self.input_list_mod.tflux[:mvol] = t * pi2 / phiedge
        self.input_list_mod.pflux[:mvol] = p * pi2 / phiedge

    def save_output(self, evaluate_force=False):
        """Write the HDF5 output file and the ``_end.toml`` restart file.

        Owns the output duties of a completed calculation: the collective final
        diagnostics, the flux write-back, the HDF5 file (input mirror, grid, output
        groups and vector potential) and the rank-0 end-TOML. The Beltrami field for the
        current geometry must be in memory (see :meth:`compute_fields`).

        The volumes are always recomputed (a geometry integral, not a solve). With
        ``evaluate_force=False`` the field is never solved here, and ``ForceErr``, ``BBe``
        and friends are whatever the last ``dforce`` left in memory; the ``h5`` is only
        complete after a force evaluation at the written geometry.

        Args:
            evaluate_force (bool): Run one ``dforce`` at the live geometry first (the call
                the removed Fortran driver made before writing), so the field, ``ForceErr``
                and the flux diagnostics all describe the written state.

        MPI-collective: all ranks must call together.
        """
        if evaluate_force:
            self.dforce()
            self._record_field_solve()
        elif not self.field_is_current:
            warnings.warn(
                "Field in Fortran memory is not known to match the current state (stale: "
                "no solve is tracked, or geometry / free-boundary inputs changed since the "
                "last one). Call compute_fields() (or pass evaluate_force=True) before "
                "save_output().",
                stacklevel=2,
            )

        # Collective; fills vvolume and the diagnostic arrays hdfint writes.
        self.wrapper_funcs_mod.compute_volumes()
        self.lib.simulation_mod.final_diagnostics()

        # dforce moves the per-volume dtflux/dpflux in place but leaves the cumulative
        # tflux/pflux at the input guess; without this the mirrored input group, the
        # /output flux datasets and the end-TOML would go stale.
        self._sync_tflux_pflux_from_dt_dp()

        # Copy the live geometry and free-boundary state into the input_list_mod arrays
        # that both the HDF5 input mirror and the end-TOML are written from (process-local
        # copies, so every rank stays consistent). Must follow the flux write-back.
        self.wrapper_funcs_mod.prepare_endfile_writing()

        self.allglobal_mod.skip_write = False
        self.lib.sphdf5_mod.init_outfile()
        self.lib.sphdf5_mod.mirror_input_to_outfile()
        self.lib.sphdf5_mod.write_grid()
        self.lib.sphdf5_mod.hdfint()
        self.lib.vector_potential_writer_mod.write_vector_potentials("W")
        self.lib.sphdf5_mod.finish_outfile()

        if self.comm.rank == 0:
            write_input_parameters_to_toml(self.parameters, f"{self.name}_end.toml")

    def create_spectreout(self, solve_field=False):
        """Returns a ``SPECTREout`` object from the current ``SPECTRE`` instance.

        Creates a persistent SPECTREout object containing the geometry and field
        information of the current instance. The field must be computed
        before calling this method.

        Args:
            solve_field (bool): If ``True``, (re)compute the Beltrami field for
                the geometry currently in memory before building the
                ``SPECTREout``, guaranteeing the field matches the interfaces. If
                ``False`` (default), the field already in Fortran memory is used;
                a warning is emitted if it is detectably stale (see
                :attr:`field_is_current`).

        Returns:
            SPECTREout: A new object containing the geometry and field
                information of the current SPECTRE instance.
        """

        from spectre.postprocess import SPECTREout

        if solve_field:
            self.compute_fields()
        elif not self.field_is_current:
            warnings.warn(
                "Field in Fortran memory is not known to match the current state (stale: "
                "no solve is tracked, or geometry / free-boundary inputs changed since the "
                "last one). Pass solve_field=True to recompute.",
                stacklevel=2,
            )

        return SPECTREout.from_spectre(self)
