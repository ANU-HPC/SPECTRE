#########################
Free-boundary W7-X vacuum
#########################

.. note::
    A full W7-X walkthrough is coming soon. The section below documents the free-boundary
    solver options that such a run uses.

Free-boundary solvers
=====================

In a free-boundary calculation the plasma is surrounded by a vacuum region bounded by a fixed
*computational boundary*, on which the normal magnetic field has two contributions: ``vns``
(the coil field, computed by :func:`spectre.converters.fix2free.fix2free` from a simsopt
``BiotSavart`` object) and the field produced by the plasma itself.

SPECTRE offers two ways to make the plasma contribution self-consistent:

Picard iteration (default, ``lvcvacuum = 0``)
    The plasma contribution ``bns`` is evaluated by a virtual-casing integral after each
    equilibrium solve and blended into the boundary condition (inputs ``mfreeits``, ``gbntol``,
    ``gbnbld``). Convergence is linear and damped by the blend factor.

Direct solve (``lvcvacuum = 1``)
    The virtual-casing field is a linear operator on the vacuum-region vector-potential
    coefficients. Setting ``lvcvacuum = 1`` embeds that operator (the *Laplace-virtual-casing
    matrix* of Henneberg, Hudson, Pfefferlé & Helander, J. Plasma Phys. **87**, 905870226
    (2021)) directly in the vacuum Beltrami system, so every field solve is self-consistent —
    no Picard iteration and no ``bns`` input are used. The inputs ``cteta`` and ``czeta`` set
    the poloidal and per-field-period toroidal resolution of the virtual-casing quadrature grid
    (defaults 128; the operator converges spectrally with them). The analytic force Jacobian
    and Hessian include the geometry derivative of the operator, so the Gauss–Newton force
    minimizer converges as for a fixed-boundary case. Requires ``igeometry = 3``.

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

Those cycles are the only Picard stepping in the run. SPECTRE's own free-boundary loop takes
a step on every field solve, including the ones the minimizer makes before the first
iteration and after the last, so the minimizer switches it off (``mfreeits`` and
``lautoinitbn`` are forced to 0) and drives the iteration itself. Two consequences worth
knowing: a ``lautoinitbn = 1`` input is honoured as an equivalent
``update_bns(mode="initialize")`` bootstrap, and the ``_end.toml`` a run writes is a faithful
restart point — its ``bns`` is the one the run finished on, and it carries the flags that
stop a reload from overwriting it. Geometry checkpoints (``xs_fixb/``) carry no ``bns``, so
restarting from one alone recovers only half of the free-boundary state.

``bnserr`` is the fixed-point residual ``mean|bns - ofmn|`` of the ``bns`` the field was
solved at, measured before the update is blended in. It therefore grades the inner solve that
just finished and does not depend on ``gbnbld``.

The ``[minimization]`` inputs controlling this are ``inner_iter_cap`` (iterations per inner
solve; ``max_niter`` remains the total budget across all of them), ``step_tol`` and
``cost_red_frac`` (the stall test), ``freeb_bnorm_tol`` (the ``bnserr`` target) and ``gbnbld``
(the blend factor, overriding the ``global`` namelist value). Setting
``initialize_bns_from_current = true`` bootstraps ``bns`` from the starting geometry before
the first solve, which a fresh fixed- to free-boundary conversion generally needs.

If ``bnserr`` grows from cycle to cycle the damped Picard map is not contracting; the driver
warns and a larger ``gbnbld`` usually helps.

The derivative of the vacuum field with respect to the plasma-boundary harmonics can also be
evaluated pointwise: ``wrapper_funcs_mod.solve_evaluate_vacuum_field_deriv(...)`` returns
:math:`\sqrt{g}\,B` and cylindrical
:math:`B` (or their derivatives at fixed :math:`(s,\theta,\zeta)` and fixed fluxes) at
arbitrary points of the vacuum region.

The W7-X test inputs used for this solver are derived from SPECTRE's existing W7-X
free-boundary regression configuration and are stored in the repository as numerical fixtures.
The optional NCSX comparison obtains the public NCSX coil configuration through SIMSOPT at test
time; SIMSOPT is a test dependency and no SIMSOPT implementation is copied into SPECTRE. The
virtual-casing implementation is an independent discretization of the equations cited above,
using SPECTRE's existing geometry, Fourier, and virtual-casing conventions.
