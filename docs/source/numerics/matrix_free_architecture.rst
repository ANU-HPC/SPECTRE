#####################################
Matrix-free feature architecture
#####################################

This page records the ownership and policy boundaries of the matrix-free
features. It complements the numerical details in
:doc:`beltrami_matrix_free` and :doc:`field_derivatives`.

Three distinct layers
=====================

The word "matrix-free" applies to three separate layers:

1. **Beltrami volume backend.** ``beltrami_operator_mod`` applies and solves
   one volume's linear Beltrami system without assembling its dense operator.
   This is an inner linear-solver choice; a run may mix dense and iterative
   volumes.
2. **Force Jacobian-vector product.** ``force_real_jvp`` contracts the
   directional geometry response and performs one linearized volume solve per
   direction. It does not assemble the outer force Jacobian.
3. **Lagged reduced-space iteration.** ``LaggedJacobianController`` reuses an
   outer force Jacobian for a bounded number of accepted iterates and refreshes
   each reduced model with force JVPs.

A dense Beltrami volume solve and an outer dense force-Jacobian rebuild are
therefore different recovery actions. Each is selected at its own layer.

Protected dense boundary
========================

The established dense force-Jacobian algorithm remains a separate path in
``field_mod::solve_field_deriv``. Its interface loop, MPI ownership,
constraint response, derivative assembly, and output layout are not shared
with arbitrary-direction JVP orchestration. The only shared inner seam is
``op_solve_derivative_rhs_batch``, an allocation-free dispatcher over an
already prepared volume operator.

The directional path remains in ``field_mod::solve_field_deriv_dir``. Its
coordinator validates a direction batch, owns its JVP context and workspace,
prepares each volume operator, evaluates the directional constraint response,
solves the batch, assembles force products, and releases all owned state.

Ownership and lifetime
======================

.. list-table::
   :header-rows: 1
   :widths: 28 36 36

   * - Component
     - Owner
     - Lifetime
   * - JVP direction, axis response, active volume and status
     - ``force_jvp_context_mod``
     - One scoped JVP batch; diagnostics survive only until copied at the
       Python boundary.
   * - Dense factors or iterative preconditioner
     - One opaque ``beltrami_operator``
     - From bind/prepare until move or release.
   * - Reusable volume operators
     - ``factor_cache_manager`` in ``beltrami_factor_cache_mod``
     - Process-local optimizer cache scope. Insert and acquire move ownership;
       invalidation advances the geometry generation.
   * - Banded normal matrix and reduced model
     - ``LaggedJacobianController``
     - From a fresh Jacobian until a rebuild, disable transition, or controller
       release.
   * - Numerical reference matrices and operator workspaces used by tests
     - ``matrix_free_test_support_mod``
     - One self-test call, through centralized setup/teardown helpers.

``beltrami_operator`` has private dense and iterative payloads. Callers bind,
prepare, apply, solve, move, observe identity, or release the handle; they do
not inspect its allocations. The cache key is
``(volume, NAdof, mu, geometry generation)`` and cache operations move complete
handles rather than copying nested factors.

Lagged-controller transitions
=============================

The controller begins with a fresh Jacobian and owns Jacobian age, the
previous accepted step, dirty trial state, the banded model, the current
reduced model, rebuild requests, and JVP diagnostics. ``trf_no_bounds`` only:

1. asks whether to use a fresh model, build a lagged model, or rebuild;
2. evaluates the objective or Jacobian requested by that action;
3. reports JVP, trial, acceptance, rejection, and convergence outcomes; and
4. consumes controller diagnostics for reporting.

A rejected stale-model step, an unusable JVP, or a convergence test based on
stale quantities requests a dense force-Jacobian rebuild. If a rejected trial
left the wrapped objective at another geometry, the controller also requests
one objective evaluation at the accepted point before rebuilding. Convergence
is declared only after a fresh-Jacobian recheck.

Numerical policy locations
==========================

``spectre/least_squares/banded.py`` separates equal-block validation, banded
normal-matrix assembly/storage, and the Cholesky/alpha numerical kernel. The
kernel returns a typed step-availability result with residual, refinement, and
failure diagnostics. It does not select another algorithm.

``trf_no_bounds`` owns the optimizer fallback policy: invalid block structure,
excess out-of-band mass, or an unavailable banded step selects the existing
exact-SVD path and emits the warning. Exact and LSMR control flow otherwise
remains unchanged.

The public ``tr_options`` mapping is translated once into ``BandedTrOptions``.
JVP values and backend status, banded solve outcomes, and controller actions
then use focused internal result types. The public least-squares inputs and
results remain compatible.

Test-only boundaries
====================

The numerical self-test implementations live in
``matrix_free_test_support_mod``; the wrapper tests call its f90wrap entry
points directly. Non-numerical operator and cache ownership checks live in
``beltrami_operator_test_support_mod``. These modules are test support;
production callers use the operator, cache, JVP, and optimizer interfaces
directly.
