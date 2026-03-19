from logging import getLogger
from mpi4py import MPI
import spectre4py_wrap
import numpy as np

from spectre.input_dict import input_dict
from spectre.misc import get_vec_pot, unpack_interfaces
from spectre.field_tracer import Field_Tracer
from spectre.file_io import read_input_parameters_from_toml, InputParameters, write_input_parameters_to_toml
from .helpers import allrzrz_from_interface_arrays
from spectre.analysis import PyocFields, PyocHelpers

logger = getLogger("[{}]".format(MPI.COMM_WORLD.Get_rank()) + __name__)


class SPECTRE(object):

    FORTRAN_INPUTS = ('physics', 'numeric', 'global_', 'diagnostics')
    PYTHON_INPUTS = ('minimization',)

    def __init__(self, input_params: InputParameters, name = None, verbose=True, comm=MPI.COMM_WORLD):
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
            "hessian_mod"
        ]

        for key in modules:
            setattr(self, key, getattr(self.lib, key))

        # py2f converts the Python object to the Fortran integer identifying an MPI communicator.
        self.allglobal_mod.set_mpi_comm(self.comm.py2f())

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
        self.allglobal_mod.filename_h5 = self.name + '.h5'

        self.pyoculus_helpers = PyocHelpers(self, self.allglobal_mod.mvol) # make cached property when we implement
        self.pyoculus_fields = PyocFields(self, self.allglobal_mod.mvol)
        
        # Make the Fortran ouput according to the verbose option
        self.io_mod.mute_output(self.verbose)

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
            name = input_file.rsplit('.', 1)[0]  # first part of the string
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
        #print a deprecation error:
        import warnings
        from spectre.file_io.specfile_to_toml import _parameters_from_spec
        warnings.warn("The 'from_deprecated_input_file' method is deprecated. Please convert your SPECTRE input files to TOML format and use 'from_input_file' instead.", DeprecationWarning)

        input_params = _parameters_from_spec(input_file)
        if kwargs.get("name") is None:
            name = input_file.rsplit('.', 1)[0]  # first part of the string
        return cls(input_params, name=name, **kwargs)

    
    def run(self, save_output: bool = False, infaces = None):
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
            _ = unpack_interfaces(infaces, self) # pack the interfaces

        self.lib.simulation_mod.spectre()

        self.lib.simulation_mod.final_diagnostics()

        if save_output:
            self.lib.sphdf5_mod.write_grid()
            self.lib.sphdf5_mod.hdfint()
            self.lib.sphdf5_mod.finish_outfile()

        if self.comm.rank == 0:
            # generate parameters from fortran memory and write to toml:
            self.wrapper_funcs_mod.prepare_endfile_writing()
            write_input_parameters_to_toml(self.parameters, f"{self.name}_end.toml") # write the final parameters to a toml file
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
            for group_name in self.FORTRAN_INPUTS: # iterate over groups, extract from fortran memory
                #print(f"Retrieving input group: {group_name}")
                this_dict = dict()
                newdict[group_name] = this_dict
                for field_name, _ in self.last_set_params.__getattribute__(group_name):
                    try:
                        field_value = self.input_list_mod.__getattribute__(field_name)
                        this_dict[field_name] = field_value
                    except AttributeError:
                        print(f"Unknown input key {field_name} in section {group_name}")
            # now fix the allRZRZ 
            allrzrz = allrzrz_from_interface_arrays(self.allglobal_mod.irbc, self.allglobal_mod.izbs,
                                                    self.allglobal_mod.irbs, self.allglobal_mod.izbc,
                                                    self.allglobal_mod.mn, self.allglobal_mod.im,
                                                    self.allglobal_mod.in_, self.allglobal_mod.mvol, 
                                                    self.input_list_mod.nfp)
            newdict['physics']['allrzrz'] = allrzrz
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

            #first allocate all variables on the Fortran side:
            self.input_list_mod.initialize_inputs()
            self.wrapper_funcs_mod.alloc_allrzrz(new_params.physics.nvol, 128, 128)  # nasty hack cause allRZRZ is not fixed shape
            # then set all the variables by the appropriate keys in the dictionary:
            for group_name, parameter_group in new_params:
                if group_name in self.FORTRAN_INPUTS:
                    #print(f"Setting input group: {group_name}")
                    for field_name, field_value in parameter_group:
                        try:
                            self.input_list_mod.__setattr__(field_name, field_value)
                        except AttributeError:
                            print(f"Unknown input key {field_name} in section {group_name}")
            self.allglobal_mod.check_inputs()

        self.allglobal_mod.broadcast_inputs()
        self.lib.preset_mod.initialize_internal_arrays()

        return
    
    def trace_fieldlines(self, init_pts, num_phi_planes=32, num_transits=1, num_threads=1, batch_size=1, method='RK45', rtol=1.0e-7, atol=1.0e-7):
        """Generate Poincare using custom field line tracing in Python. Sets the self.tracer to the generated Field_Tracer object.

        Args:
            init_pts (List[mvol]): list of ndarray[num_trajectories, 2] with initial (s, theta) points for tracing
            num_phi_planes (int, optional): Number of phi planes to record the poincare. Defaults to 32.
            num_transits (int, optional): Number of toroidal transits to perform. Defaults to 1.
            num_threads (int, optional): Number of threads for multithreaded integration. Defaults to 1.
            batch_size (int, optional): Batch size for multithreaded integration. Defaults to 1.
            method (str, optional): Method for integrating the ode. Defaults to 'RK45'. Options: 'RK23', 'RK45', 'DOP853'
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

        tracer = Field_Tracer(mvol, igeometry, nfp, mpol, im, in_, ate, aze, ato, azo, rbs, rbc, zbs, zbc, lrad_arr, 
            num_threads=num_threads, batch_size=batch_size, method=method, rtol=rtol, atol=atol)

        tracer.trace_field(init_pts, num_phi_planes=num_phi_planes, num_transits=num_transits)

        self._tracer = tracer

    def trace_one_fieldline(self, sth_init, ivol, num_phi_planes=32, num_transits=1, method='RK45', rtol=1.0e-7, atol=1.0e-7):
        """Generate field line trace for a single point in Python. Sets the self.tracer to the generated Field_Tracer object, for the first call.

        Args:
            sth_init (List[float]): initial (s, theta) of the point.
            ivol (int): Index of the volume where tracing is done.
            num_phi_planes (int, optional): Number of phi planes to record the poincare. Defaults to 32.
            num_transits (int, optional): Number of toroidal transits to perform. Defaults to 1.
            method (str, optional): Method for integrating the ode. Defaults to 'RK45'. Options: 'RK23', 'RK45', 'DOP853'
            rtol (float, optional): Relative tolerance of the integrator. Defaults to 1.0e-6.
            atol (float, optional): Absolute tolerance of the integrator. Defaults to 1.0e-6.

        Returns:
            List[float]: Final (s, theta) coordinates.
            List[float]: Final (R, Z) coordinates.
            float: Final iota.
            bool: True if integration did not reach the end, False otherwise.

        """
        
        if not hasattr(self, '_tracer_single'):
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

            tracer = Field_Tracer(mvol, igeometry, nfp, mpol, im, in_, ate, aze, ato, azo, rbs, rbc, zbs, zbc, lrad_arr, method=method, rtol=rtol, atol=atol)

            self._tracer_single = tracer

        sth, RZ, iota, flag = self._tracer_single.trace_field_single(sth_init, ivol, num_phi_planes=num_phi_planes, num_transits=num_transits)

        return sth, RZ, iota, flag
    
    @property
    def tracer(self):
        if not hasattr(self, '_tracer'):
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
    
    def get_pyoculus_helper(self, lvol):
        """Returns a SPECTRE_pyoculus_helper object for the current SPECTRE instance"""
        from spectre.pyoculus_helper import SPECTRE_pyoculus_helper
        from spectre.misc import get_xinit_specwrap
        # if the field is just loaded, then the solution has not yet been calculated. 
        self.field_mod.solve_field(get_xinit_specwrap(self), recompute_axis=False)
        return SPECTRE_pyoculus_helper(self, lvol=lvol)


    def print_inputs(self):
        min_params = self.min_params
        physics = self.parameters.physics.model_dump()
        for k in ['rac', 'zas', 'ras', 'zac', 'rbc', 'zbs', 'rbs', 'zbc', 'zws', 'rws', 'zwc', 'vns', 'bns', 'vnc', 'bnc', 'rwc', 'pl', 'ql', 'pr', 'qr', 'lp', 'lq', 'rp', 'rq', 'allrzrz']:
            physics.pop(k)

        print('Minimization params:')
        for k in min_params:
            print(f"{k:23} {min_params[k]}")
        print('\nPhysics params:')
        for k in physics:
            print(f"{k:23} {physics[k]}")
        print()

    def find_island_fixed_point(self, vol=0, n=0, m=1, guess=None, nrestart=1, phi = 0.0, plot=False, ax=None, finderargs ={}):
        """
        Use pyoculus to find the fixed point of an island chain with given n and m. 
        
        Because field lines do not cross MRxMHD surfaces, one must specify the volume to search in (0 for the volume containing
        the axis).

        The method uses the SPECTRE poloidal angle to determine poloidal winding, and counts the number of windings around this axis if the region
        is annular (lvol=0)
        Configurations with strongly writhing geometry may have a helical coordinate axis, and produce unreliable results in this volume. 
        In this case use the find_fixed_point method which simply finds fixed points of n applications of the Poincare map, without counting windings. 

        Note: for fields with negiative rotational transform, one must specify either m to be negative. 
        Run trace_fieldlines and investigate tracer.iotas to determine the sign of the rotational transform.

        Args:
            vol (int, optional): Volume to search in. Defaults to 0.
            n (int, optional): Toroidal mode number of the island. Defaults to 0.
            m (int, optional): Poloidal mode number of the island. Defaults to 1
            guess (List[float], optional): Initial guess for the fixed point in (s, theta) coordinates. Defaults to None, which uses a random guess
            nrestart (int, optional): Number of restarts to perform. Defaults to 1.
            phi (float, optional): Toroidal angle to search for the fixed point. Defaults to 0.0.
            plot (bool, optional): Whether to plot the fixed point on the given axis. Defaults to False.
            ax (Matplotlib axis, optional): Matplotlib axis to plot the fixed point on, if plot is True. Defaults to None.
        Returns:
            tuple: (r, z, s, theta) coordinates of the fixed point. Returns (None, None, None, None) if fixed point is not found.
        """

        if n ==0 and m==1 and vol==0:
            print("Looks like you are trying to find the axis. Use the 'find_axis' method instead for more robust results.")

        from pyoculus.maps.toroidal_bfield_section import ToroidalBfieldSection
        from pyoculus.solvers import FixedPoint

        field = self.pyoculus_fields[vol]

        section = ToroidalBfieldSection(field, phi0=phi)

        fixed_point = FixedPoint(section)

        guess = [-0.989, 0.1] if guess is None else guess
        fixed_point.find_with_iota(n=n, m=m, guess=guess, nrestart=nrestart, method='scipy.root', **finderargs)
        
        if not fixed_point.successful:
            print("Fixed point not found")
            return None, None, None, None

        s = fixed_point.coords[0,0]
        theta = fixed_point.coords[0,1]


        r, phi, z = self.pyoculus_helpers[vol].convert_coords([s, theta, phi])  #  move to spectre internal routine

        if plot:
            ax.plot(r, z, 'rx', ms=4, label="Fixed point")

        return r, z, s, theta

    def find_axis(self, guess=None, nrestart=1, phi = 0.0, plot=False, ax=None, finderargs ={}):
        """
        Use pyoculus to find the fixed point of an island chain with given n and m. 
        
        Because field lines do not cross MRxMHD surfaces, one must specify the volume to search in (0 for the volume containing
        the axis).

        The method uses the SPECTRE poloidal angle to determine poloidal winding, and counts the number of windings around this axis if the region
        is annular (lvol=0)
        Configurations with strongly writhing geometry may have a helical coordinate axis, and produce unreliable results in this volume. 
        In this case use the find_fixed_point method which simply finds fixed points of n applications of the Poincare map, without counting windings. 
        Args:
            guess (List[float], optional): Initial guess for the fixed point in (s, theta) coordinates. Defaults to None, which uses a random guess
            nrestart (int, optional): Number of restarts to perform. Defaults to 1.
            phi (float, optional): Toroidal angle to search for the fixed point. Defaults to 0.0.
            plot (bool, optional): Whether to plot the fixed point on the given axis. Defaults to False.
            ax (Matplotlib axis, optional): Matplotlib axis to plot the fixed point on, if plot is True. Defaults to None.
        Returns:
            tuple: (r, z, s, theta) coordinates of the fixed point. Returns (None, None, None, None) if fixed point is not found.
        """

        from pyoculus.maps.toroidal_bfield_section import ToroidalBfieldSection
        from pyoculus.solvers import FixedPoint

        vol=0

        field = self.pyoculus_fields[vol]

        section = ToroidalBfieldSection(field, phi0=phi)

        fixed_point = FixedPoint(section)

        guess = [-0.989, np.pi +0.1] if guess is None else guess
        options = finderargs.pop('options', {})
        fixed_point.find(1, guess=guess, nrestart=nrestart, method='scipy.root', options={'factor':1e-3}, **finderargs)

        if not fixed_point.successful:
            print("Fixed point not found")
            return None, None, None, None
        
        s = fixed_point.coords[0,0]
        theta = fixed_point.coords[0,1]


        r, phi, z = self.pyoculus_helpers[vol].convert_coords([s, theta, phi])  #  move to spectre internal routine

        if plot:
            ax.plot(r, z, 'rx', ms=4, label="Fixed point")

        return r, z, s, theta
