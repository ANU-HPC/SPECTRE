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

The derivative of the vacuum field with respect to the plasma-boundary harmonics can also be
evaluated pointwise: ``wrapper_funcs_mod.solve_vacuum_field_deriv(ii, irz, issym)`` followed by
``wrapper_funcs_mod.get_field_at_points(...)`` returns :math:`\sqrt{g}\,B` and cylindrical
:math:`B` (or their derivatives at fixed :math:`(s,\theta,\zeta)` and fixed fluxes) at
arbitrary points of the vacuum region.
