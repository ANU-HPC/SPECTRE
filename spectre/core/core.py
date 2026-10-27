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

import spectre.angle_targets as angle_targets
import spectre.force_targets as force_targets
from spectre.file_io import (
    InputParameters,
    read_input_parameters_from_toml,
    write_input_parameters_to_toml,
)
from spectre.input_dict import input_dict
from spectre.utils import _set_interfaces_from_xin, get_xinit_specwrap

from .helpers import allrzrz_from_interface_arrays

logger = getLogger(f"[{MPI.COMM_WORLD.Get_rank()}]" + __name__)


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
    # None means "no solve tracked on the python side yet".
    obj._field_fingerprint = None


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
        self._record_field_solve()

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

        # initialize_internal_arrays() zeros the field, so any tracked solve is now stale
        self._field_fingerprint = None

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
            xin,
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

    def _record_field_solve(self):
        """Record that the field matches the live interface geometry."""
        self._field_fingerprint = self.xin.copy()

    def compute_fields(self, recompute_axis=True, xin=None):
        """Solve the Beltrami field for an interface geometry.

        Populates the magnetic vector potential in Fortran memory for the current ``xin``
        and records the geometry as the tracked solve, so :attr:`field_is_current` becomes
        ``True``. This is the lightweight alternative to :meth:`create_spectreout` (which
        additionally builds a full ``SPECTREout`` postprocess object) for when you only need
        the in-memory field, e.g. before evaluating Beltrami errors.

        Args:
            recompute_axis (bool): Recompute the coordinate axis as part of the solve
                (matches the ``force_real`` default). Defaults to ``True``.
            xin (numpy.ndarray, optional): Packed interface degrees of freedom.
                Defaults to the live :attr:`xin` state.
        """
        xin = self.xin if xin is None else xin
        self.field_mod.solve_field(xin, recompute_axis)
        self._record_field_solve()

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
        self.field_mod.solve_field_deriv(xin, mpol_rz, ntor_rz, recompute_axis)
        self._record_field_solve()

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
        elif self._field_fingerprint is not None and not self.field_is_current:
            import warnings

            warnings.warn(
                "Field in Fortran memory is stale: the geometry changed since the last "
                "solve. Pass solve_field=True to recompute.",
                stacklevel=2,
            )

        return SPECTREout.from_spectre(self)
