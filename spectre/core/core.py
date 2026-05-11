# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

from logging import getLogger
from os import PathLike
from pathlib import Path
from mpi4py import MPI
import spectre4py_wrap

from spectre.input_dict import input_dict
from spectre.utils import get_vec_pot, unpack_interfaces
from spectre.field_tracer import Field_Tracer
from spectre.file_io import (
    read_input_parameters_from_toml,
    InputParameters,
    write_input_parameters_to_toml,
)
from .helpers import allrzrz_from_interface_arrays
from spectre.postprocess._get_geometry import get_rz_coords_helper

logger = getLogger("[{}]".format(MPI.COMM_WORLD.Get_rank()) + __name__)


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
    if obj.comm.rank == 0:
        min_params_temp = obj.parameters.minimization.model_dump()
    else:
        min_params_temp = None
    obj.min_params = input_dict(obj.comm.bcast(min_params_temp, root=0))

    # Set .h5 output filename
    obj.allglobal_mod.filename_h5 = obj.name + ".h5"

class SPECTRE(object):
    FORTRAN_INPUTS = ("physics", "numeric", "global_", "diagnostics")
    PYTHON_INPUTS = ("minimization",)

    def __init__(self, input_file: str | PathLike, name=None, verbose=True, comm=MPI.COMM_WORLD):
        """Initialize a SPECTRE object from a TOML file

        Args:
            input_file (str | PathLike): Path to the SPECTRE TOML input file.
            name (str, optional): Name for the SPECTRE run. Defaults to None, in which case the name is derived from input_file.
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
    def from_input_params(cls, input_params: InputParameters, name="spectre", verbose=True, comm=MPI.COMM_WORLD):
        """
        Initialize a SPECTRE instance from an InputParameters instance.
        
        Args:
            input_params(InputParameters): An instance of InputParameters containing the SPECTRE input parameters.
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
    def from_deprecated_input_file(cls, input_file: str | PathLike, name=None, verbose=True, comm=MPI.COMM_WORLD):
        """
        Initialize a SPECTRE instance from a SPECTRE input file.

        Args:
            input_file (str | PathLike): Path to the SPECTRE input file in TOML format.
            name (str, optional): Name for the SPECTRE run. Defaults to None, in which case the name is derived from input_file.
            verbose (bool): Whether or not to enable verbose output. Defaults to True.
            comm (MPI communicator): MPI communicator to use. Defaults to MPI.COMM_WORLD.
        
        Returns:
            SPECTRE instance initialized with the provided input file.
        """

        # print a deprecation error:
        import warnings
        from spectre.file_io.specfile_to_toml import _parameters_from_spec

        warnings.warn(
            "The 'from_deprecated_input_file' method is deprecated. Please convert your SPECTRE input files to TOML format and initialize with this.",
            DeprecationWarning,
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

    def trace_fieldlines(
        self,
        init_pts,
        num_phi_planes=32,
        num_transits=1,
        num_threads=1,
        batch_size=1,
        method="RK45",
        rtol=1.0e-7,
        atol=1.0e-7,
    ):
        """Generate Poincare using custom field line tracing in Python. 
        Sets the self.tracer to the generated Field_Tracer object.

        Args:
            init_pts (List[mvol]): list of ndarray[num_trajectories, 2] with initial (s, theta) points for tracing
            num_phi_planes (int, optional): Number of phi planes to record the poincare.
                Defaults to 32.
            num_transits (int, optional): Number of toroidal transits to perform. Defaults to 1.
            num_threads (int, optional): Number of threads for multithreaded integration. Defaults to 1.
            batch_size (int, optional): Batch size for multithreaded integration. Defaults to 1.
            method (str, optional): Method for integrating the ode. Defaults to 'RK45'.
                Options: 'RK23', 'RK45', 'DOP853'
            rtol (float, optional): Relative tolerance of the integrator. Defaults to 1.0e-6.
            atol (float, optional): Absolute tolerance of the integrator. Defaults to 1.0e-6.
        """

        mvol = self.allglobal_mod.mvol
        igeometry = self.input_list_mod.igeometry
        mpol = self.input_list_mod.mpol
        im = self.allglobal_mod.im
        in_ = self.allglobal_mod.in_
        nfp = self.input_list_mod.nfp
        lrad_arr = self.input_list_mod.lrad

        ate, aze, ato, azo = get_vec_pot(self)

        rbc = self.allglobal_mod.irbc.T
        zbs = self.allglobal_mod.izbs.T
        rbs = self.allglobal_mod.irbs.T
        zbc = self.allglobal_mod.izbc.T

        tracer = Field_Tracer(
            mvol,
            igeometry,
            nfp,
            mpol,
            im,
            in_,
            ate,
            aze,
            ato,
            azo,
            rbs,
            rbc,
            zbs,
            zbc,
            lrad_arr,
            num_threads=num_threads,
            batch_size=batch_size,
            method=method,
            rtol=rtol,
            atol=atol,
        )

        tracer.trace_field(init_pts, num_phi_planes=num_phi_planes, num_transits=num_transits)

        self._tracer = tracer

    def get_rz_coords(self, lvol, s, theta, phi):
        """Return R, Z coordinates for given (s, theta, zeta) arrays in volume lvol
        
        Returns:
                R
                Z
        """

        Rac, Rbc = self.allglobal_mod.irbc.T[lvol : lvol + 2]
        Zas, Zbs = self.allglobal_mod.izbs.T[lvol : lvol + 2]

        return get_rz_coords_helper(
            self.input_list_mod.igeometry,
            Rac,
            Rbc,
            Zas,
            Zbs,
            self.allglobal_mod.im,
            self.allglobal_mod.in_,
            s,
            theta,
            phi,
            lvol,
        )

    def trace_one_fieldline(
        self,
        sth_init,
        ivol,
        num_phi_planes=32,
        num_transits=1,
        method="RK45",
        rtol=1.0e-7,
        atol=1.0e-7,
    ):
        """Generate field line trace for a single point in Python. 
        Sets the self.tracer to the generated Field_Tracer object, for the first call.

        Args:
            sth_init (List[float]): initial (s, theta) of the point.
            ivol (int): Index of the volume where tracing is done.
            num_phi_planes (int, optional): Number of phi planes to record the poincare.
                Defaults to 32.
            num_transits (int, optional): Number of toroidal transits to perform. Defaults to 1.
            method (str, optional): Method for integrating the ode. Defaults to 'RK45'.
                Options: 'RK23', 'RK45', 'DOP853'
            rtol (float, optional): Relative tolerance of the integrator. Defaults to 1.0e-6.
            atol (float, optional): Absolute tolerance of the integrator. Defaults to 1.0e-6.

        Returns:
            List[float]: Final (s, theta) coordinates.
            List[float]: Final (R, Z) coordinates.
            float: Final iota.
            bool: True if integration did not reach the end, False otherwise.

        """

        if not hasattr(self, "_tracer_single"):
            mvol = self.allglobal_mod.mvol
            igeometry = self.input_list_mod.igeometry
            mpol = self.input_list_mod.mpol
            im = self.allglobal_mod.im
            in_ = self.allglobal_mod.in_
            nfp = self.input_list_mod.nfp
            lrad_arr = self.input_list_mod.lrad

            ate, aze, ato, azo = get_vec_pot(self)

            rbc = self.allglobal_mod.irbc.T
            zbs = self.allglobal_mod.izbs.T
            rbs = self.allglobal_mod.irbs.T
            zbc = self.allglobal_mod.izbc.T

            tracer = Field_Tracer(
                mvol,
                igeometry,
                nfp,
                mpol,
                im,
                in_,
                ate,
                aze,
                ato,
                azo,
                rbs,
                rbc,
                zbs,
                zbc,
                lrad_arr,
                method=method,
                rtol=rtol,
                atol=atol,
            )

            self._tracer_single = tracer

        sth, RZ, iota, flag = self._tracer_single.trace_field_single(
            sth_init, ivol, num_phi_planes=num_phi_planes, num_transits=num_transits
        )

        return sth, RZ, iota, flag

    @property
    def tracer(self):
        if not hasattr(self, "_tracer"):
            raise AttributeError("Field tracing has not been done, run 'trace_fieldlines()' first")
        return self._tracer

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

    def find_axis(self, max_repeats=0, guess=None, verbose=False):
        """Finds the magnetic axis using root-finding and _return_map from custom field line tracing. 
        If intial guess doesn't work, retries max_repeats times with random guesses."""
        from spectre.fixed_point import find_axis

        rax, zax, sax, thax = find_axis(
            self, max_repeats, guess, plot=False, verbose=verbose, name=self.name
        )

        self.r_magaxis = rax
        self.z_magaxis = zax

        return rax, zax, sax, thax
