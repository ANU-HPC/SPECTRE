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
vacuum solve enabled (``lvcvacuum = 1`` in the ``[physics]`` section), which is documented in
:doc:`/numerics/direct_vacuum_solve`; the classic Picard iteration (``lvcvacuum = 0``) is the
default.

Force minimization with Picard free boundary
============================================

When the force minimizer drives a ``lvcvacuum = 0`` case, *when* ``bns`` is refreshed matters
as much as how. The Jacobian is built once per accepted iteration at a fixed ``bns``, so
refreshing before every objective evaluation — including the rejected trial steps of the
trust-region search — blends ``bns`` toward the extrapolation from bad geometries and moves
the residual surface out from under that Jacobian. Near convergence the trust radius then
collapses.

:func:`spectre.force_minimizer.optimizer_picard` (used by ``scripts/run/calc_spectre_eq.py``)
alternates instead. Each inner least-squares solve runs at a **frozen** ``bns``; the field is
refreshed exactly once between solves, via :meth:`spectre.SPECTRE.update_bns`. The run ends
when the boundary field is consistent (``bnserr < freeb_bnorm_tol``) and the inner solve has
either converged or stalled. Fixed-boundary cases and ``lvcvacuum = 1`` take a single solve,
since neither needs the alternation.

Those cycles are the only Picard stepping in the run: ``update_bns`` is the sole mutator of
``bns``, so the ``_end.toml`` a run writes is a faithful restart point — its ``bns`` is the
one the run finished on. Geometry checkpoints (``xs_fixb/``) carry no ``bns``, so restarting
from one alone recovers only half of the free-boundary state.

``bnserr`` is the fixed-point residual ``mean|bns - ofmn|`` of the ``bns`` the field was
solved at, measured before the update is blended in. It therefore grades the inner solve that
just finished and does not depend on ``gbnbld``.

The ``[minimization]`` inputs controlling this are ``inner_iter_cap`` (iterations per inner
solve; ``max_niter`` remains the total budget across all of them), ``step_tol`` and
``cost_red_frac`` (the stall test), ``freeb_bnorm_tol`` (the ``bnserr`` target) and ``gbnbld``
(the blend factor, overriding the ``global`` namelist value).
``initialize_bns_from_current`` (default ``true``) bootstraps ``bns`` from the starting
geometry with ``update_bns(mode="initialize")`` before the first solve, which a fresh
fixed- to free-boundary conversion generally needs.

If ``bnserr`` grows from cycle to cycle the damped Picard map is not contracting; the driver
warns and a larger ``gbnbld`` usually helps.

Regression fixtures
===================

The W7-X free-boundary regression inputs of the test suite
(``tests/wrapper/test_freeb_w7x*.toml``) are lower-resolution fixtures derived from this same
configuration and are stored in the repository as numerical fixtures; the direct-solve variants
among them use ``lvcvacuum = 1``. The optional NCSX comparison test obtains the public NCSX
coil configuration through SIMSOPT at test time; SIMSOPT is a test dependency and no SIMSOPT
implementation is copied into SPECTRE. The virtual-casing implementation is an independent
discretization of the equations cited in :doc:`/numerics/direct_vacuum_solve`, using SPECTRE's
existing geometry, Fourier, and virtual-casing conventions.
