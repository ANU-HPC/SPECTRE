##########################
Free-boundary equilibria
##########################

In a fixed-boundary calculation the shape of the plasma boundary is prescribed. In a
free-boundary calculation (``physics.lfreebound = 1``) the plasma boundary is itself an unknown:
it is the outermost *free* interface of the MRxMHD equilibrium, and its shape is determined by
force balance with the magnetic field produced by external coils.

Geometry of the problem
=======================

A free-boundary case contains one volume more than the corresponding fixed-boundary case,
:math:`M_{vol} = N_{vol} + 1`:

* volumes :math:`1 \dots N_{vol}` are the plasma volumes, exactly as in a fixed-boundary run;
* volume :math:`M_{vol}` is a *vacuum region* bounded on the inside by the plasma boundary
  (interface :math:`N_{vol}`, free to move) and on the outside by the *computational boundary*
  (interface :math:`M_{vol}`, prescribed in the input as ``physics.rwc``/``physics.zws`` and
  never moved).

In the vacuum region the field is current-free. In each volume, including the vacuum one, the
field is represented by a vector potential in the gauge of SPEC
:footcite:p:`Hudson2012pop,Hudson2020`,

.. math::

   \mathbf{A} = A_\theta(s,\theta,\zeta)\,\nabla\theta + A_\zeta(s,\theta,\zeta)\,\nabla\zeta,

with a Fourier representation in the angles and a Chebyshev representation in the radial
coordinate :math:`s \in [-1, 1]` (:math:`s=-1` on the plasma boundary of the vacuum region,
:math:`s=+1` on the computational boundary), so that

.. math::

   \sqrt{g}\, B^s = \partial_\theta A_\zeta - \partial_\zeta A_\theta, \qquad
   \sqrt{g}\, B^\theta = -\partial_s A_\zeta, \qquad
   \sqrt{g}\, B^\zeta = \partial_s A_\theta .

The boundary condition on the computational boundary
====================================================

On the computational boundary the normal component of the total field must match the field
produced by all currents, external *and* internal. Writing a scalar field on the boundary as a
Fourier series :math:`F(\theta,\zeta) = \sum_i F^c_i \cos\alpha_i + F^s_i \sin\alpha_i`, with
:math:`\alpha_i = m_i\theta - n_i\zeta`, the vacuum Beltrami system enforces for every mode
:math:`(m_i, n_i) \neq (0,0)`

.. math::
   :label: freeb-bc

   \bigl[\sqrt{g}\,\mathbf{B}\cdot\nabla s\bigr]^s_i \Big|_{s=+1}
   \;=\; V^s_i + B^s_i ,

(and the analogous cosine equation when stellarator symmetry is not enforced), where

* :math:`V^s_i` (input ``physics.vns``) are the harmonics of the Jacobian-weighted normal field
  :math:`\sqrt{g}\,\mathbf{B}_{coil}\cdot\nabla s = \mathbf{B}_{coil}\cdot
  (\mathbf{e}_\theta \times \mathbf{e}_\zeta)` produced by the *coils* on the computational
  boundary — fixed data, computed once from a coil model (this Jacobian-weighted normal field
  is the *differential flux* :math:`D = D_C + D_P` of :footcite:t:`Hudson2020`, with ``vns``
  and ``bns`` corresponding to the external part :math:`D_C` and the plasma part :math:`D_P`);
* :math:`B^s_i` (``bns``) are the harmonics of the same quantity produced by the *plasma*
  (volume currents and interface current sheets, including the discontinuity of the tangential
  field across the plasma boundary itself). This contribution depends on the equilibrium being
  solved for, which makes the free-boundary problem self-consistent (nonlinear).

The :math:`(0,0)` mode carries no such condition: the net flux of the plasma field through the
closed computational boundary vanishes by :math:`\nabla\cdot\mathbf{B}=0`, and the
:math:`(0,0)` constraint slots instead fix the toroidal and poloidal fluxes enclosed by the
vacuum region.

The plasma contribution is evaluated with the virtual-casing principle
:footcite:p:`ShafranovZakharov1972,Hudson2020`: outside the plasma boundary, the field produced
by every current flowing inside it equals the field of a fictitious surface current
:math:`\mathbf{K} = \hat{\mathbf{n}} \times \mathbf{B}` on the plasma boundary, where
:math:`\mathbf{B}` is the total field on the vacuum side of that boundary.

Preparing a free-boundary input
===============================

The converter :func:`spectre.converters.fix2free.fix2free` turns a fixed-boundary SPECTRE input
into a free-boundary one. The caller supplies a computational boundary (e.g. built with
simsopt's ``SurfaceRZFourier.extend_via_normal`` from the plasma boundary) and a simsopt
``BiotSavart`` object for the coils; the converter evaluates the ``vns`` spectrum on the
computational boundary and anchors the signs of ``curpol`` and ``phiedge`` to the actual coil
field. See the docstring of :func:`~spectre.converters.fix2free.fix2free` for the details and
caveats (in particular the computation of ``curpol``, the net poloidal linking current). A
complete worked example — from the W7-X coil set to a runnable free-boundary input — is
described in :doc:`/user_guide/equilibrium_calculation/w7x_freeb_vac`.

Two ways to close the self-consistency loop
===========================================

The plasma term :math:`B^s_i` in :eq:`freeb-bc` depends on the field, which depends on
:math:`B^s_i`. SPECTRE offers two strategies:

Picard iteration (``global.lvcvacuum = 0``, default)
    The classic SPEC scheme :footcite:p:`Hudson2020`. The field is solved with the current
    ``bns``; the plasma contribution is then re-evaluated by numerical quadrature of the
    virtual-casing integral and blended into the boundary data,

    .. math::

       B^s_i \;\leftarrow\; \lambda\, B^s_i + (1-\lambda)\, B^{s,new}_i ,

    with blend factor :math:`\lambda` = ``global.gbnbld`` (default 0.666), and the solve is
    repeated until the update falls below ``global.gbntol`` or ``global.mfreeits`` iterations
    are exhausted. The quadrature is controlled by ``vcnt``, ``vcnz`` (surface-current grid),
    ``vcasingtol``, ``vcasingits``, ``vcasingper`` and ``numeric.lvcgrid``;
    ``numeric.lautoinitbn`` bootstraps an initial ``bns`` from a preliminary fixed-boundary
    solve. Convergence is linear and damped by the blend factor, and the force Jacobian treats
    ``bns`` as frozen.

Direct solve (``global.lvcvacuum = 1``)
    The virtual-casing field is a *linear* operator on the vacuum-region vector-potential
    coefficients. Embedding that operator (the Laplace-virtual-casing matrix of
    :footcite:t:`Henneberg2021`) in the vacuum Beltrami matrix makes every single linear solve
    self-consistent: no Picard iteration, no ``bns`` input, and analytic geometry derivatives
    for the force Jacobian. This is the recommended mode for toroidal (``igeometry = 3``)
    free-boundary calculations; it is documented in full in
    :doc:`direct_vacuum_solve`.

Running and output
==================

A free-boundary equilibrium is run exactly like a fixed-boundary one, e.g. with
``scripts/run/calc_spectre_eq.py input.toml`` (see
:doc:`/user_guide/equilibrium_calculation/index`). The interface geometry
returned by the force minimization now includes the plasma boundary itself. The output HDF5
file records the solver inputs (including ``lvcvacuum``) and the converged ``bns`` spectrum
(identically zero in direct-solve mode, where the plasma contribution lives inside the matrix
instead).

.. toctree::
   :maxdepth: 1

   direct_vacuum_solve

An example free-boundary configuration is described in
:doc:`/user_guide/equilibrium_calculation/w7x_freeb_vac`.

.. rubric:: References

.. footbibliography::
