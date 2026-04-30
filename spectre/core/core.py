# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

from logging import getLogger
from mpi4py import MPI
import spectre4py_wrap

from spectre.input_dict import input_dict
from spectre.utils import unpack_interfaces
from spectre.file_io import (
    read_input_parameters_from_toml,
    InputParameters,
    write_input_parameters_to_toml,
)
from .helpers import allrzrz_from_interface_arrays

logger = getLogger("[{}]".format(MPI.COMM_WORLD.Get_rank()) + __name__)


class SPECTRE(object):
    FORTRAN_INPUTS = ("physics", "numeric", "global_", "diagnostics")
    PYTHON_INPUTS = ("minimization",)

    def __init__(self, input_params: InputParameters, name=None, verbose=True, comm=MPI.COMM_WORLD):
        """Initialization of SPECTRE runs

        Args:
            input_params (InputParameters): Input parameters for the SPECTRE run.
            verbose (bool): If wants screen outputs. (default: True).
            comm (MPI communicator): the comunicator assigned to SPECTRE. Default: MPI.COMM_WORLD
        Returns:
            None
        """

        self.comm = comm
        assert isinstance(verbose, bool), "verbose is either True or False."
        self.verbose = verbose
        # Fortran libaries accessed via self.lib
        self.lib = spectre4py_wrap
        if name is None:
            self.name = "SPECTRE_run"
        else:
            self.name = name

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
            setattr(self, key, getattr(self.lib, key))

        # py2f converts the Python object to the Fortran integer identifying an MPI communicator.
        self.allglobal_mod.set_mpi_comm(self.comm.py2f())

        # Make the Fortran ouput according to the verbose option
        self.io_mod.mute_output(not self.verbose)

        if self.comm.rank == 0:
            self.last_set_params = input_params

        for group_name in self.PYTHON_INPUTS:
            setattr(self, f"{group_name}_params", input_params.__getattribute__(group_name))
        self.parameters = input_params  # use the setter, initializes fortran

        # Broadcast minimization params into a 'min_params' attribute
        if self.comm.rank == 0:
            min_params_temp = self.parameters.minimization.model_dump()
        else:
            min_params_temp = None
        self.min_params = input_dict(self.comm.bcast(min_params_temp, root=0))

        # Set .h5 output filename
        self.allglobal_mod.filename_h5 = self.name + ".h5"

        return

    @classmethod
    def from_input_file(cls, input_file: str, **kwargs):
        """
        Initialize a SPECTRE instance from a SPECTRE input file.
        Args:
            input_file: Path to the SPECTRE input file in TOML format.
            verbose: Whether or not to enable verbose output.
            comm: MPI communicator to use.

        Returns:
            SPECTRE instance initialized with the provided input file.
        """
        input_params = read_input_parameters_from_toml(input_file)
        if kwargs.get("name") is None:
            name = input_file.rsplit(".", 1)[0]  # first part of the string
        return cls(input_params, name=name, **kwargs)

    @classmethod
    def from_deprecated_input_file(cls, input_file: str, **kwargs):
        """
        Initialize a SPECTRE instance from a SPECTRE input file.
        Args:
            input_file: Path to the SPECTRE input file in TOML format.
            verbose: Whether or not to enable verbose output.
            comm: MPI communicator to use.

        Returns:
            SPECTRE instance initialized with the provided input file.
        """
        # print a deprecation error:
        import warnings
        from spectre.file_io.specfile_to_toml import _parameters_from_spec

        warnings.warn(
            "The 'from_deprecated_input_file' method is deprecated. Please convert your SPECTRE input files to TOML format and use 'from_input_file' instead.",
            DeprecationWarning,
        )

        input_params = _parameters_from_spec(input_file)
        if kwargs.get("name") is None:
            name = input_file.rsplit(".", 1)[0]  # first part of the string
        return cls(input_params, name=name, **kwargs)

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
        """
        Returns a SPECTREout object that is created from the current SPECTRE object, containing geometry and field information. 
        Note that the field has to be computer prior to calling this function.
        """
        
        from spectre.postprocess import SPECTREout

        return SPECTREout.from_spectre(self)