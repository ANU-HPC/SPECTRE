#########################
Free-boundary W7-X vacuum
#########################

This example builds and runs a free-boundary vacuum calculation for the W7-X standard
configuration. The input-generation workflow is the notebook
``examples/utils/make_freeboundary/spectre_set_vns_w7x.ipynb``, which:

1. loads the W7-X coil set (a simsopt ``BiotSavart`` object) from the simsopt zoo;
2. builds a fixed-boundary SPECTRE input from the W7-X standard-configuration VMEC input with
   :func:`spectre.converters.simsvmec2spectre` (two plasma volumes, ``mpol=9``, ``ntor=8``);
3. constructs the computational boundary by extending the plasma boundary 0.2 m outward along
   its normal (simsopt's ``SurfaceRZFourier.extend_via_normal``);
4. converts the input to free boundary with :func:`spectre.converters.fix2free.fix2free`
   (``vacuum=True``, i.e. pressureless with the toroidal-current constraint
   ``lconstraint = 3``), which evaluates the ``vns`` spectrum of the coil field on the
   computational boundary and anchors the ``curpol``/``phiedge`` signs to the coil field, and
   writes the result to a TOML input;
5. runs SPECTRE on that input and traces a Poincaré section of the initial state — which is
   not yet force-balanced, so the interfaces are not yet flux surfaces of the total field.

The physics of the free-boundary setup — the vacuum region, the boundary condition on the
computational boundary, and the two available vacuum solvers — is described in
:doc:`/numerics/free_boundary`. To converge the equilibrium, run the generated TOML through the
usual force minimization (``calc_spectre_eq.py``, see :doc:`index`), preferably with the direct
vacuum solve enabled (``lvcvacuum = 1`` in the ``[global]`` section), which is documented in
:doc:`/numerics/direct_vacuum_solve`; the classic Picard iteration (``lvcvacuum = 0``) is the
default.

The W7-X free-boundary regression inputs of the test suite
(``tests/wrapper/test_freeb_w7x*.toml``) are lower-resolution fixtures derived from this same
configuration and are stored in the repository as numerical fixtures; the direct-solve variants
among them use ``lvcvacuum = 1``. The optional NCSX comparison test obtains the public NCSX
coil configuration through SIMSOPT at test time; SIMSOPT is a test dependency and no SIMSOPT
implementation is copied into SPECTRE. The virtual-casing implementation is an independent
discretization of the equations cited in :doc:`/numerics/direct_vacuum_solve`, using SPECTRE's
existing geometry, Fourier, and virtual-casing conventions.
