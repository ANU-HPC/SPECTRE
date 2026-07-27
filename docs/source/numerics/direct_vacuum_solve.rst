#####################################################
Direct vacuum solve (Laplace-virtual-casing matrix)
#####################################################

Setting ``global.lvcvacuum = 1`` replaces the Picard iteration of a
:doc:`free-boundary calculation <free_boundary>` by a *direct* solve: the virtual-casing
response of the plasma is embedded, as a dense linear operator, in the Beltrami matrix of the
vacuum region. This is the *Laplace-virtual-casing matrix*
:math:`\mathcal{L}_{vc} \equiv \mathcal{A} - \mathcal{B}` defined in eq. (4.4) of
:footcite:t:`Henneberg2021`, whose geometry variation is their eq. (4.5); that paper formulates
the operator for a scalar-potential representation of the vacuum field, while SPECTRE applies
the same construction to its vector-potential representation (the alternative anticipated in
their §2 and used by free-boundary SPEC :footcite:p:`Hudson2020`). Every linear solve of the
vacuum field is then self-consistent with the boundary condition on the computational boundary,
and the geometry derivative of the operator is computed analytically, so the force Jacobian and
Hessian used by the Gauss–Newton force minimizer are exact.

The implementation lives in ``fortran_src/vc_matrix_mod.F90``. This page derives what is
computed, states the discretization exactly as implemented, and explains how and when to run in
this mode.

The vacuum system and its constraint rows
=========================================

In the vacuum volume (:math:`s\in[-1,1]`, plasma boundary at :math:`s=-1`, computational
boundary at :math:`s=+1`) the field is represented by

.. math::

   \mathbf{A} = A_\theta \nabla\theta + A_\zeta \nabla\zeta, \qquad
   A_\theta = \sum_{i,l} \Bigl( A^{e}_{\theta,l,i} \cos\alpha_i
            + A^{o}_{\theta,l,i} \sin\alpha_i \Bigr)\, \bar T_l(s),

with :math:`\alpha_i = m_i\theta - n_i\zeta`, :math:`\bar T_l` the (rescaled) Chebyshev basis,
and analogously for :math:`A_\zeta`; the coefficients are the code arrays
``Ate``/``Aze`` (``Ato``/``Azo`` without stellarator symmetry). The field components are

.. math::

   \sqrt{g}\,B^s = \partial_\theta A_\zeta - \partial_\zeta A_\theta, \qquad
   \sqrt{g}\,B^\theta = -\partial_s A_\zeta, \qquad
   \sqrt{g}\,B^\zeta = \partial_s A_\theta .

Collecting all vacuum-volume coefficients (and the Lagrange multipliers of the gauge, flux and
boundary constraints) in a vector :math:`\mathbf{a}`, the vacuum Beltrami system is a linear
system whose *normal-field constraint rows* (code: the ``Lme``, and without stellarator
symmetry ``Lmf``, rows of ``dMA``) enforce, for every mode :math:`(m_i,n_i)\neq(0,0)`,

.. math::
   :label: dvs-constraint-picard

   \underbrace{\sum_l \bigl(-n_i A^{e}_{\theta,l,i} - m_i A^{e}_{\zeta,l,i}\bigr)
   \bar T_l(+1)}_{\textstyle \bigl[\sqrt{g}\,\mathbf{B}\cdot\nabla s\bigr]^{s}_{i}\big|_{s=+1}}
   \;=\; V^s_i + B^s_i ,

where :math:`[F]^{c}_i, [F]^{s}_i` denote the cosine/sine harmonics of a field on the boundary,
:math:`F = \sum_i [F]^c_i\cos\alpha_i + [F]^s_i\sin\alpha_i`. In the Picard scheme the plasma
contribution :math:`B^s_i` on the right-hand side is data, updated between solves. The direct
solve instead expresses it *exactly* as a linear operator acting on :math:`\mathbf{a}` itself.

The virtual-casing operator
===========================

Virtual-casing principle
------------------------

Every interface of an MRxMHD equilibrium is a flux surface, so the total field on the vacuum
side of the plasma boundary is purely tangential. By the virtual-casing principle
:footcite:p:`ShafranovZakharov1972`, the field produced at a point :math:`\mathbf{x}` *outside*
the plasma boundary :math:`\Gamma` by all currents flowing inside it (volume currents and
interface current sheets, including the sheet on :math:`\Gamma` itself) is the Biot–Savart
field of the surface current :math:`\mathbf{K} = \hat{\mathbf{n}}\times\mathbf{B}`:

.. math::
   :label: dvs-vc

   \mathbf{B}_{pl}(\mathbf{x}) = \frac{1}{4\pi} \oint_\Gamma
   \bigl[\hat{\mathbf{n}}(\mathbf{x}')\times\mathbf{B}(\mathbf{x}')\bigr] \times
   \frac{\mathbf{x}-\mathbf{x}'}{|\mathbf{x}-\mathbf{x}'|^3}\, \mathrm{d}S' ,

with :math:`\mathbf{B}` the total field on the outer (vacuum) side of :math:`\Gamma` and
:math:`\hat{\mathbf{n}}` the outward unit normal.

Parametrizing :math:`\Gamma` by the :math:`s=-1` surface of the vacuum volume,
:math:`\mathbf{x}'(\theta,\zeta) = (R\cos\zeta,\, R\sin\zeta,\, Z)`, and using
:math:`\hat{\mathbf{n}}\,\mathrm{d}S = (\mathbf{e}_\theta\times\mathbf{e}_\zeta)\,
\mathrm{d}\theta\,\mathrm{d}\zeta` together with
:math:`\sqrt{g}\,\nabla s\times\mathbf{B} = B_\theta\mathbf{e}_\zeta - B_\zeta\mathbf{e}_\theta`
(valid on the surface, where :math:`\mathbf{B}\cdot\nabla s = 0`), the surface-current element
becomes :math:`\hat{\mathbf{n}}\times\mathbf{B}\,\mathrm{d}S = -\,\mathbf{j}\,
\mathrm{d}\theta\,\mathrm{d}\zeta` with

.. math::
   :label: dvs-j

   \mathbf{j}(\theta,\zeta) = B_\zeta\, \mathbf{e}_\theta - B_\theta\, \mathbf{e}_\zeta ,

where :math:`B_\theta, B_\zeta` are the covariant components of the vacuum-region field at
:math:`s=-1` and :math:`\mathbf{e}_\theta = \partial_\theta\mathbf{x}',
\mathbf{e}_\zeta = \partial_\zeta\mathbf{x}'`. What the solver needs is the Jacobian-weighted
normal component of :eq:`dvs-vc` on the computational boundary, evaluated at the target points
:math:`\mathbf{x}_t` with (non-unit) normal :math:`\mathbf{N}(\mathbf{x}_t) =
\mathbf{e}_\theta\times\mathbf{e}_\zeta\big|_{s=+1} = \sqrt{g}\,\nabla s`:

.. math::
   :label: dvs-gbn

   \sqrt{g}\,\mathbf{B}_{pl}\cdot\nabla s\,(\mathbf{x}_t)
   = -\frac{1}{4\pi} \int_0^{2\pi}\!\!\int_0^{2\pi}
   \bigl[\mathbf{j}(\theta',\zeta') \times \mathbf{r}\bigr]\cdot\mathbf{N}(\mathbf{x}_t)\,
   f_\varepsilon(|\mathbf{r}|)\; \mathrm{d}\theta'\,\mathrm{d}\zeta' ,
   \qquad \mathbf{r} = \mathbf{x}_t - \mathbf{x}' ,

with the regularized kernel

.. math::

   f_\varepsilon(r) = \frac{1 + 3\varepsilon^2/(r^2+\varepsilon^2)}{(r^2+\varepsilon^2)^{3/2}}
   \;\xrightarrow{\;\varepsilon\to 0\;}\; \frac{1}{r^3},

where :math:`\varepsilon` = ``global.vcasingeps`` (default :math:`10^{-12}`) guards against a
vanishing denominator; since source and target surfaces are separated, the regularization is
inconsequential at the default value. Kernel, regularization and normalization replicate the
Picard-mode quadrature (``freeb_mod``) exactly.

Linearity in the vector potential
---------------------------------

The tangential field entering :eq:`dvs-j` is linear in :math:`\mathbf{a}`. At :math:`s=-1`,

.. math::

   \sqrt{g}\,B^\theta = -\sum_{i,l} A^{e}_{\zeta,l,i}\, \bar T_l'(-1) \cos\alpha_i
   \;(+\,\text{odd terms}), \qquad
   \sqrt{g}\,B^\zeta = +\sum_{i,l} A^{e}_{\theta,l,i}\, \bar T_l'(-1) \cos\alpha_i
   \;(+\,\text{odd terms}),

and the covariant components follow through the (geometry-only) metric of the surface,

.. math::

   B_\theta = \frac{g_{\theta\theta}\,\sqrt{g}B^\theta + g_{\theta\zeta}\,\sqrt{g}B^\zeta}
   {\sqrt{g}}, \qquad
   B_\zeta = \frac{g_{\theta\zeta}\,\sqrt{g}B^\theta + g_{\zeta\zeta}\,\sqrt{g}B^\zeta}
   {\sqrt{g}} ,

with :math:`g_{\theta\theta} = R_\theta^2 + Z_\theta^2`,
:math:`g_{\theta\zeta} = R_\theta R_\zeta + Z_\theta Z_\zeta`,
:math:`g_{\zeta\zeta} = R_\zeta^2 + Z_\zeta^2 + R^2` and
:math:`\sqrt{g} = R\,(Z_s R_\theta - R_s Z_\theta)`. Grouping the integrand of :eq:`dvs-gbn`
per source point as

.. math::

   \bigl[\mathbf{j}\times\mathbf{r}\bigr]\cdot\mathbf{N}\, f_\varepsilon
   = \bigl(\sqrt{g}B^\theta\bigr) w_\theta + \bigl(\sqrt{g}B^\zeta\bigr) w_\zeta ,

.. math::

   w_\theta = \frac{ g_{\theta\zeta} (\mathbf{e}_\theta\times\mathbf{r})\cdot\mathbf{N}
            - g_{\theta\theta} (\mathbf{e}_\zeta\times\mathbf{r})\cdot\mathbf{N}}{\sqrt{g}}
            \, f_\varepsilon , \qquad
   w_\zeta = \frac{ g_{\zeta\zeta} (\mathbf{e}_\theta\times\mathbf{r})\cdot\mathbf{N}
           - g_{\theta\zeta} (\mathbf{e}_\zeta\times\mathbf{r})\cdot\mathbf{N}}{\sqrt{g}}
           \, f_\varepsilon ,

makes the whole map :math:`\mathbf{a} \mapsto \bigl[\sqrt{g}\,\mathbf{B}_{pl}\cdot\nabla s
\bigr]^{c,s}_i` manifestly linear: it is a dense matrix, the *Laplace-virtual-casing matrix*
:math:`\mathsf{V}` (code: ``dVC``).

Discretization
--------------

The integral :eq:`dvs-gbn` is evaluated by the trapezoidal rule on a uniform tensor grid of the
plasma boundary covering the **full torus**: :math:`C_\theta` = ``global.cteta`` points in
:math:`\theta` and :math:`C_\zeta = {}` ``global.czeta`` :math:`\times\, n_{fp}` points in
:math:`\zeta`,

.. math::

   \oint\!\!\oint \mathrm{d}\theta'\,\mathrm{d}\zeta'\,(\cdot)
   \;\approx\; \frac{(2\pi)^2}{C_\theta C_\zeta} \sum_{\text{sources}} (\cdot) .

Because the integrand is smooth (the surfaces are separated) and periodic, this converges
geometrically ("spectrally") in :math:`C_\theta, C_\zeta`; the rate degrades as the gap between
plasma boundary and computational boundary shrinks. The targets are the :math:`N_t\times N_z`
collocation points of the computational boundary on one field period (per-field-period targets
suffice by the :math:`n_{fp}` symmetry), and the harmonics are extracted by discrete Fourier
projection with the standard transform normalization. The resulting matrix elements are, e.g.
for the columns of the coefficients :math:`A^{e}_{\theta,l,j}` and :math:`A^{e}_{\zeta,l,j}`,

.. math::

   \mathsf{V}\bigl[(i,s),(\theta,e,l,j)\bigr] = \nu\, \bar T_l'(-1)\;
   P^s_i \Bigl[ \sum_{\text{sources}} w_\zeta(\cdot,\text{src}) \cos\alpha_j(\text{src})
   \Bigr], \quad
   \mathsf{V}\bigl[(i,s),(\zeta,e,l,j)\bigr] = -\nu\, \bar T_l'(-1)\;
   P^s_i \Bigl[ \sum_{\text{sources}} w_\theta(\cdot,\text{src}) \cos\alpha_j(\text{src})
   \Bigr],

with quadrature normalization :math:`\nu = -\pi / (C_\theta C_\zeta)`
(:math:`= (2\pi)^2 \cdot (-\tfrac{1}{4\pi}) / (C_\theta C_\zeta)`) and :math:`P^{c,s}_i` the
cosine/sine projection over the target grid. Odd (``Ato``/``Azo``) columns replace
:math:`\cos\alpha_j` by :math:`\sin\alpha_j`; cosine rows :math:`(i,c)` exist only without
stellarator symmetry.

Only the constraint rows of :math:`\mathsf{V}` are nonzero, so only those rows are ever formed
and stored; a dense :math:`(N_{dof})^2` version of the operator never exists.

The modified linear system
--------------------------

The direct solve subtracts :math:`\mathsf{V}` from the vacuum matrix and drops the plasma term
from the right-hand side. The constraint rows :eq:`dvs-constraint-picard` become

.. math::
   :label: dvs-constraint-direct

   \bigl[\sqrt{g}\,\mathbf{B}\cdot\nabla s\bigr]^{s}_{i}\Big|_{s=+1}
   \;-\; \bigl[\sqrt{g}\,\mathbf{B}_{pl}\cdot\nabla s\bigr]^{s}_{i}
   \;=\; V^s_i ,
   \qquad (m_i,n_i)\neq(0,0),

(and the cosine analogue without stellarator symmetry): the normal field left over after
removing the plasma's own contribution must equal the coil field. One LU solve of
:math:`(\mathsf{A} - \mathsf{V})\,\mathbf{a} = \text{rhs}` — the vector-potential form of
:math:`\mathcal{L}_{vc}\,\boldsymbol{\phi} = \boldsymbol{B}_E`, eq. (4.4) of
:footcite:t:`Henneberg2021` — therefore returns the self-consistent free-boundary vacuum field
directly: the ``mfreeits`` Picard loop, the ``bns`` input and the ``lautoinitbn`` bootstrap are
not used (any ``bns`` present in the input is zeroed on read so it cannot leak into the solve
or the output).

The :math:`(0,0)` mode needs no row in :math:`\mathsf{V}`: the plasma field has zero net flux
through the closed computational boundary (:math:`\nabla\cdot\mathbf{B}=0`), and that slot of
the system constrains the enclosed fluxes instead.

Geometry derivative — the analytic force Jacobian
=================================================

The force minimizer needs :math:`\partial\mathbf{a}/\partial x_k` for every geometry degree of
freedom :math:`x_k`. For the vacuum volume, differentiating the modified system at fixed fluxes
gives the perturbed solve

.. math::
   :label: dvs-perturbed

   (\mathsf{A} - \mathsf{V})\, \frac{\partial\mathbf{a}}{\partial x_k}
   = -\left( \frac{\partial\mathsf{A}}{\partial x_k}
   - \frac{\partial\mathsf{V}}{\partial x_k} \right) \mathbf{a}_0 ,

where :math:`\mathbf{a}_0` is the converged solution — the vector-potential form of eq. (4.5)
of :footcite:t:`Henneberg2021`,
:math:`\mathcal{L}_{vc}\,\delta\boldsymbol{\phi} = -(\delta\mathcal{A} - \delta\mathcal{B})
\boldsymbol{\phi} + \delta\boldsymbol{B}_E`. The coil-field term
:math:`\delta\boldsymbol{B}_E` does not appear: the computational boundary never moves, so
:math:`V^s_i` is constant. The operator
:math:`\mathsf{V}` depends on the plasma boundary both as the *source surface* of the
virtual-casing integral and through the metric factors in :eq:`dvs-j`; only perturbations of
the plasma boundary (the inner interface of the vacuum volume) contribute
:math:`\partial\mathsf{V}/\partial x_k \neq 0`.

The term :math:`(\partial\mathsf{V}/\partial x_k)\,\mathbf{a}_0` is assembled analytically
(cf. eq. (4.5) of :footcite:t:`Henneberg2021`), not by finite differences. Because the vacuum
volume interpolates the geometry linearly in :math:`s`,

.. math::

   R(s,\theta,\zeta) = \tfrac{1-s}{2}\, R_{pb}(\theta,\zeta)
                     + \tfrac{1+s}{2}\, R_{cb}(\theta,\zeta),

a perturbation of one plasma-boundary harmonic, say :math:`x_k = R^{c}_{m_k n_k}`, perturbs the
source surface at :math:`s=-1` by

.. math::

   \delta R = \cos\alpha_k, \quad
   \delta R_s = -\tfrac{1}{2}\cos\alpha_k, \quad
   \delta R_\theta = -m_k \sin\alpha_k, \quad
   \delta R_\zeta = +n_k \sin\alpha_k,

(and the corresponding sine-based quartet for :math:`Z^{s}`, :math:`R^{s}`, :math:`Z^{c}`
harmonics). Every factor of the integrand in :eq:`dvs-gbn` depends on the source geometry only
through the eight local quantities :math:`(R, R_s, R_\theta, R_\zeta, Z, Z_s, Z_\theta,
Z_\zeta)`, so the perturbed integrand is a linear combination of these eight perturbations with
coefficient fields that are *independent of which harmonic is perturbed*. The contributions
are, at fixed :math:`\mathbf{a}_0`:

* the covariant components :math:`B_\theta, B_\zeta` through :math:`\delta g_{\alpha\beta}` and
  :math:`\delta\sqrt{g}`;
* the tangent vectors :math:`\mathbf{e}_\theta, \mathbf{e}_\zeta` in :math:`\mathbf{j}`;
* the source position :math:`\mathbf{x}'` in :math:`\mathbf{r}`, entering both
  :math:`\mathbf{j}\times\mathbf{r}` and the kernel :math:`f_\varepsilon` through
  :math:`\delta(|\mathbf{r}|^2)`.

The implementation precomputes the projected harmonics of
:math:`(\partial\mathsf{V}/\partial x_k)\,\mathbf{a}_0` for *all* plasma-boundary harmonics at
once (eight DOF-independent kernel fields contracted against per-DOF weights with blocked
``DGEMM``), and the perturbed right-hand sides of :eq:`dvs-perturbed` are then served by
lookup.

This term is what makes the analytic force Jacobian (and the Hessian built from it) *exact* in
free-boundary mode: in Picard mode the Jacobian is computed with ``bns`` frozen, i.e. it
neglects the change of the plasma's own contribution when the boundary moves. Every
plasma-boundary degree of freedom of the Jacobian is validated against finite differences in
``tests/wrapper/test_force_jac_findiff_vc.py``.

Symmetry, caching and cost
==========================

* **Stellarator symmetry.** When ``enforce_stell_sym`` is set,
  :math:`\sqrt{g}\,\mathbf{B}_{pl}\cdot\nabla s` is odd under
  :math:`(\theta,\zeta)\to(-\theta,-\zeta)`; only about half of the target planes are
  evaluated and the rest are filled by mirroring, and only the sine rows of
  :math:`\mathsf{V}` exist.
* **Caching.** :math:`\mathsf{V}` depends only on the geometry of the two interfaces bounding
  the vacuum region. The assembled rows are cached keyed on those harmonics: a force
  evaluation followed by a Jacobian evaluation at the same geometry assembles the matrix once.
* **Cost.** Assembly is :math:`\mathcal{O}(mn \times C_\theta C_\zeta \times N_t N_z)`
  floating-point work, organized as blocked ``DGEMM`` calls and OpenMP-parallel loops; storage
  is limited to the constraint rows, :math:`(mn-1)\times(N_{dof}+1)` numbers (twice that
  without stellarator symmetry).

How to run
==========

Start from a working free-boundary input (see
:doc:`free_boundary` for how to generate one with
:func:`~spectre.converters.fix2free.fix2free`) and set in the TOML input:

.. code-block:: toml

   [global]
   lvcvacuum = 1
   cteta = 128    # poloidal resolution of the virtual-casing source grid
   czeta = 128    # toroidal resolution per field period

Then run the equilibrium calculation exactly as usual, e.g.
``calc_spectre_eq.py input.toml``. Requirements, enforced both by the Fortran input check and
the Python-side validator:

* ``physics.lfreebound = 1`` and toroidal geometry ``physics.igeometry = 3``;
* ``cteta`` :math:`\geq \max(4,\, 4\,m_{pol})` and ``czeta`` :math:`\geq \max(4,\, 4\,n_{tor})`
  so the source grid resolves the boundary harmonics.

The defaults ``cteta = czeta = 128`` are ample for typical resolutions; on the W7-X regression
case they reproduce a fully converged Picard reference to a few parts in :math:`10^4` of the
largest ``bns`` harmonic. Verify convergence, if needed, by increasing them and watching the
plasma-induced normal-field spectrum settle.

In this mode the Picard-related inputs are inactive: ``bns``/``bnc`` (zeroed on read),
``mfreeits``, ``gbnbld``, ``gbntol``, ``numeric.lautoinitbn``, and the Picard quadrature
controls ``vcnt``, ``vcnz``, ``lvcgrid``, ``vcasingtol``, ``vcasingits``, ``vcasingper``.
``vcasingeps`` *is* still used (kernel regularization, :eq:`dvs-gbn`). The output HDF5 file
records ``lvcvacuum``, ``cteta`` and ``czeta``, and writes a zero ``bns`` spectrum.

When to use it
==============

Use ``lvcvacuum = 1`` as the default for toroidal free-boundary equilibrium calculations,
in particular whenever the Gauss–Newton force minimizer with the analytic Jacobian is used:

* every force evaluation is self-consistent — there is no partially converged ``bns`` entering
  the force residual, which matters inside line searches and trust-region steps;
* the force Jacobian is exact, including the plasma-response of the vacuum field, so the
  minimizer converges as it does for a fixed-boundary case;
* there is no blend factor (``gbnbld``) to tune and no risk of a slowly converging or
  oscillating Picard loop when the plasma response is strong.

Keep the Picard scheme (``lvcvacuum = 0``) for slab geometry (``igeometry = 1``, where the
direct solve is not implemented), for cross-validation of the two discretizations, or to
reproduce legacy SPEC behaviour.

Diagnostics and validation
==========================

Pointwise evaluation of the vacuum field and of its derivative with respect to a
plasma-boundary harmonic is available through the wrapper:
``wrapper_funcs_mod.evaluate_vacuum_field(...)`` returns :math:`\sqrt{g}\,\mathbf{B}` and
cylindrical :math:`\mathbf{B}` at arbitrary :math:`(s,\theta,\zeta)` points of the vacuum
region, and ``wrapper_funcs_mod.solve_evaluate_vacuum_field_deriv(...)`` their derivatives (at
fixed :math:`(s,\theta,\zeta)` and fixed fluxes) for one boundary harmonic, using the same
perturbed-solve machinery as the Jacobian. ``wrapper_funcs_mod.get_vc_consistency`` returns the
plasma-induced normal-field harmonics computed both matrix-free and as
:math:`\mathsf{V}\mathbf{a}`, and ``wrapper_funcs_mod.get_bnorm_plasmab`` evaluates the same
integral with the independent Picard-mode machinery.

The implementation is validated by:

* ``tests/wrapper/test_vc_matrix.py`` — :math:`\mathsf{V}\mathbf{a}` against the matrix-free
  quadrature (machine precision) and against the independent Picard-mode virtual-casing
  integration (quadrature error);
* ``tests/wrapper/test_vc_direct_vs_picard.py`` — at fixed geometry, the direct solve against
  the converged fixed point of the Picard iteration;
* ``tests/wrapper/test_vc_vacuum_biotsavart.py`` — ground truth: a current-free free-boundary
  equilibrium in the NCSX coil field must reproduce the coils' Biot–Savart field (evaluated
  with simsopt) with a vanishing virtual-casing contribution;
* ``tests/wrapper/test_force_jac_findiff_vc.py`` — the analytic force Jacobian, including
  :math:`(\partial\mathsf{V}/\partial x_k)\,\mathbf{a}_0`, against finite differences;
* ``tests/wrapper/test_vacuum_pointwise_db.py`` — the pointwise derivative API against
  finite-difference re-solves.

.. rubric:: References

.. footbibliography::
