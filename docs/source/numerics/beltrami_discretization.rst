########################################################
The Beltrami system: formulation and discretization
########################################################

This document derives the linear system that SPECTRE solves for the magnetic
field in each volume. Both solver backends — the dense LU path
(:doc:`beltrami_dense`) and the matrix-free GMRES path
(:doc:`beltrami_matrix_free`) — solve *exactly this system*; they differ only
in how the operator is represented and inverted. The notation introduced here
is used throughout the Numerics section.

The presentation follows the SPEC papers
:footcite:p:`Hudson2012ppcf,Hudson2012pop,Hudson2020`, specialised to the
conventions actually implemented in the SPECTRE sources
(``fortran_src/matrices_mod.F90``, ``fortran_src/chebyshev_mod.F90``,
``fortran_src/beltrami_solver_mod.F90``, ``fortran_src/preset_mod.F90``).

The variational problem
=======================

In MRxMHD the plasma is partitioned into :math:`N_V` nested volumes
:math:`\mathcal{V}_v` separated by ideal interfaces :math:`\mathcal{I}_v`.
Equilibria are extrema of the constrained energy functional
(:footcite:t:`Hudson2012ppcf`, Eq. (1); :footcite:t:`Hudson2012pop`,
Eq. (11); :footcite:t:`Hudson2020`, Eq. (3))

.. math::

   \mathcal{F} \;=\; \sum_{v=1}^{N_V}
   \left[\, W_v \;-\; \frac{\mu_v}{2}\bigl( K_v - K_{v,o} \bigr) \right],

.. math::

   W_v = \int_{\mathcal{V}_v}\!\Bigl( \frac{p}{\gamma-1}
       + \frac{B^2}{2} \Bigr)\,dv,
   \qquad
   K_v = \int_{\mathcal{V}_v}\! \mathbf{A}\cdot\mathbf{B}\;dv ,

where :math:`\mu_v` is the Lagrange multiplier for the helicity constraint
:math:`K_v = K_{v,o}`. Setting the first variation with respect to the vector
potential to zero in each volume yields the *Beltrami equation*

.. math::

   \nabla\times\mathbf{B} \;=\; \mu_v\,\mathbf{B}
   \qquad \text{in } \mathcal{V}_v ,

and stationarity with respect to the interface geometry yields the
force-balance condition :math:`[[\,p + B^2/2\,]] = 0` across each interface
(the outer iteration, see :doc:`field_derivatives`). Because the field in each
volume depends only on the geometry of its two bounding interfaces, its
enclosed fluxes, and :math:`\mu_v`, the volumes decouple: SPECTRE solves one
independent Beltrami boundary-value problem per volume (in parallel over MPI
ranks).

In a free-boundary calculation the outermost volume is a vacuum region,
treated as the special case :math:`\mu = 0` of the same formulation
:footcite:p:`Hudson2020`.

How the physics becomes matrix algebra
======================================

Everything on this page follows from three mechanical translations;
keeping them in mind makes every formula below readable.

**1. A field is a list of numbers.** The vector potential in a volume is
expanded in basis functions — Chebyshev/Zernike polynomials
:math:`\bar{T}_l(s)` radially, Fourier harmonics :math:`\cos\alpha_i`
(:math:`\alpha_i = m_i\theta - n_i\zeta`) in the angles. Once the basis
is fixed, "the field" *is* the coefficient vector :math:`\mathbf{a}` (its
``nadof`` entries :math:`A_{\theta,i,l}, A_{\zeta,i,l}`): evaluating
:math:`\mathbf{A}` anywhere means summing basis functions times
coefficients — nothing more.

**2. Derivatives are index swapping.** :math:`\mathbf{B} =
\nabla\times\mathbf{A}` needs derivatives of :math:`\mathbf{A}`, but a
derivative of a known basis function is just another known function:
:math:`\partial_s` swaps :math:`\bar{T}_l \to \bar{T}'_l`,
:math:`\partial_\theta` swaps :math:`\cos\alpha_i \to
-m_i\sin\alpha_i`, and :math:`\partial_\zeta` swaps
:math:`\cos\alpha_i \to +n_i\sin\alpha_i`. No new unknowns ever appear —
the *same* coefficients get attached to a different basis function with a
mode-number factor. That is why :math:`\sqrt{g}\,B^\zeta = \partial_s
A_\theta` is not an equation to solve but a relabeling: the tensors named
``TT…``, ``TD…``, ``DD…`` below record exactly which combination
(:math:`T` or its derivative :math:`D`, sine or cosine) each term of
:math:`\mathbf{B}` picked up, and the :math:`m_i, n_j` prefactors in the
assembly formulas are those angular derivatives.

**3. Integrals are sums.** The energy :math:`\int B^2\,dv` is done in two
different ways at once: the *radial* integral by Gauss quadrature
(:math:`\int f\,ds \to \sum_q w_q f(s_q)`, the ``iquad`` points), and the
*angular* integrals exactly — a product of two harmonics is rewritten as
the sum/difference harmonics :math:`\alpha_i \mp \alpha_j`, so the
:math:`\theta\zeta`-integral collapses to reading off two Fourier
coefficients of the metric :math:`\bar{g}_{\mu\nu}`. The metric is where
all the geometry (the shape of the volume) enters; the basis functions
never change.

Put together: :math:`\mathbf{B}` is *linear* in :math:`\mathbf{a}` (moves
1–2), so the energy :math:`\int \mathbf{B}\cdot\mathbf{B}\,dv` is
*quadratic* — it is :math:`\tfrac12\,\mathbf{a}^{\!\top}\mathcal{A}\,
\mathbf{a}`, where the matrix entry :math:`\mathcal{A}_{(i,l),(j,p)}` is
simply the number move 3 produces for the basis-function pair
:math:`(i,l)` and :math:`(j,p)`: "how much energy these two modes share
in this geometry." The giant matrices are nothing but the complete table
of these pairwise overlaps. Extremizing the energy at fixed helicity then
turns the physics statement :math:`\nabla\times\mathbf{B} =
\mu\mathbf{B}` into the linear system :math:`(\mathcal{A} -
\mu\mathcal{D})\,\mathbf{a} = \text{rhs}` — solving the field *means*
solving that matrix equation. The dense solver writes the whole table out
and factorizes it; the matrix-free solver (:doc:`beltrami_matrix_free`)
evaluates the same sums on the quadrature grid on the fly, without ever
storing the table — same arithmetic, different order.

Representation of the vector potential
======================================

Coordinates
-----------

Each volume carries coordinates :math:`(s,\theta,\zeta)` with
:math:`s\in[-1,1]`: the radial coordinate interpolates between the inner and
outer bounding interfaces, and :math:`(\theta,\zeta)` are the poloidal and
toroidal angles of the interface representation. Fourier harmonics are
indexed by :math:`i = 1,\dots,\texttt{mn}` with poloidal/toroidal mode numbers
:math:`(m_i, n_i)` (:math:`n_i` includes the field periodicity factor) and
angle argument

.. math::

   \alpha_i \;=\; m_i\theta - n_i\zeta ,
   \qquad (m_1, n_1) = (0,0).

In the innermost volume the inner "interface" degenerates to the coordinate
axis; the interpolation is regularized so that Fourier harmonics of the
geometry behave as :math:`\bar{s}^{\,m/2}` near the axis, where
:math:`\bar{s} = (1+s)/2` (:footcite:t:`Hudson2020`, Eq. (8)). The metric
elements :math:`g_{\mu\nu}(s,\theta,\zeta)` and Jacobian :math:`\sqrt{g}`
follow from the coordinate map (``coordinate_mod.F90``).

Gauge
-----

The gauge freedom is used to remove :math:`A_s` entirely and to set both
tangential components to zero on the inner boundary
(:footcite:t:`Hudson2012pop`, Eqs. (20)–(22); :footcite:t:`Hudson2020`,
Sec. III):

.. math::

   \mathbf{A} \;=\; A_\theta\,\nabla\theta + A_\zeta\,\nabla\zeta,
   \qquad
   A_\theta(-1,\theta,\zeta) = 0, \quad A_\zeta(-1,\theta,\zeta) = 0 .

With this gauge the magnetic field components are

.. math::

   \sqrt{g}\,B^s      = \partial_\theta A_\zeta - \partial_\zeta A_\theta,
   \qquad
   \sqrt{g}\,B^\theta = -\,\partial_s A_\zeta,
   \qquad
   \sqrt{g}\,B^\zeta  = \partial_s A_\theta ,

(:footcite:t:`Hudson2012pop`, Eq. (36)), and

.. math::

   \sqrt{g}\;\mathbf{A}\cdot\mathbf{B}
   \;=\; -\,A_\theta\,\partial_s A_\zeta \;+\; A_\zeta\,\partial_s A_\theta .

Spectral expansion
------------------

In the stellarator-symmetric case (``enforce_stell_sym = true``) the two
components are expanded as

.. math::

   \begin{aligned}
   A_\theta(s,\theta,\zeta) &= \sum_{i=1}^{\texttt{mn}} \sum_{l}
      A_{\theta,i,l}\; \bar{T}_l(s)\, \cos\alpha_i , \\
   A_\zeta(s,\theta,\zeta) &= \sum_{i=1}^{\texttt{mn}} \sum_{l}
      A_{\zeta,i,l}\; \bar{T}_l(s)\, \cos\alpha_i ,
   \end{aligned}

with radial resolution :math:`l \le L \equiv \texttt{lrad}(v)` per volume.
Without stellarator symmetry, sine partners
:math:`A^o_{\theta,i,l},\,A^o_{\zeta,i,l}` (arrays ``Ato``, ``Azo``) are
added for :math:`i > 1`.

The radial basis :math:`\bar{T}_l` differs between annular volumes and the
innermost volume (``base_functions_mod.F90``):

**Annular volumes — recombined, scaled Chebyshev** (``get_cheby``):

.. math::

   \bar{T}_0(s) = 1, \qquad
   \bar{T}_l(s) = \frac{T_l(s) - (-1)^l}{l+1} \quad (l \ge 1),

where :math:`T_l` is the Chebyshev polynomial of the first kind. The
subtraction ("basis recombination") makes :math:`\bar{T}_l(-1) = 0` for every
:math:`l \ge 1`, so the inner-boundary gauge condition is satisfied
identically by dropping the :math:`l = 0` coefficient from the unknowns; the
:math:`1/(l+1)` scaling improves the conditioning of the resulting matrix
(:footcite:t:`Hudson2020` discusses the conditioning of the raw basis).

**Innermost volume — recombined, scaled Zernike** (``get_zernike``): the basis
attached to harmonic :math:`i` is
:math:`\bar{R}^{m_i}_l(\bar{s})` with :math:`\bar{s} = (1+s)/2`, where
:math:`R^m_l` is the radial Zernike polynomial (nonzero only for
:math:`l \ge m` with :math:`l - m` even), recombined for :math:`m = 0`
(:math:`R^0_l \to R^0_l - (-1)^{l/2}`) and :math:`m = 1`
(:math:`R^1_l \to R^1_l - (-1)^{(l-1)/2}\tfrac{l+1}{2}\bar{s}`), and scaled by
:math:`1/(l+1)`. Zernike polynomials behave as :math:`\bar{s}^{\,m}` at the
axis and resolve the ill-conditioning of the earlier
:math:`\bar{s}^{\,m/2}`-regularized Chebyshev basis
:footcite:p:`Hudson2020,Qu2020`. Chain-rule factors from
:math:`d\bar{s}/ds = 1/2` are applied wherever radial derivatives of the
Zernike basis are used.

Degrees of freedom and constraints
==================================

The unknown vector :math:`\mathbf{a}` of one volume (length
:math:`N \equiv \texttt{nadof}(v)`, layout defined in ``preset_mod.F90`` and
packed/unpacked by ``packing_mod.packab``) contains:

1. the retained radial coefficients :math:`A_{\theta,i,l}, A_{\zeta,i,l}` of
   every Fourier harmonic (index arrays ``Ate(v,0,i)%i(l)``,
   ``Aze(v,0,i)%i(l)``; coefficients removed by basis recombination carry
   index 0 and are identically zero), and
2. one Lagrange multiplier per remaining explicit constraint.

The explicit constraints and their multipliers, in the stellarator-symmetric
case, are (:footcite:t:`Hudson2020`, Eq. (10) — the multipliers
:math:`a_i, b_i, e_i, c_1, d_1` there map to the code's
``Lma, Lmb, Lme, Lmg, Lmh``):

.. list-table::
   :header-rows: 1
   :widths: 12 20 68

   * - Multiplier
     - Harmonics
     - Constraint (as implemented)
   * - ``Lma``
     - innermost volume, :math:`m_i \ge 2`
     - axis regularity of :math:`A_\theta`: the leading
       :math:`\mathcal{O}(\bar{s}^{\,m_i})` coefficient vanishes,
       :math:`\sum_l A_{\theta,i,l}\,
       \bigl[\bar{R}^{m_i}_l/\bar{s}^{\,m_i}\bigr]_{\bar{s}=0} = 0`
       (array ``RTM``). For :math:`m_i \in \{0,1\}` the condition is already
       built into the recombined Zernike basis.
   * - ``Lmb``
     - innermost volume, :math:`m_i = 0`
     - for :math:`n_i \neq 0`: :math:`A_\zeta` harmonic vanishes at the axis,
       :math:`\sum_l A_{\zeta,i,l}\,\bar{R}^0_l(0) = 0`. For
       :math:`(m,n) = (0,0)`: the condition is applied at the *outer* boundary,
       :math:`\sum_l A_{\zeta,1,l}\,\bar{R}^0_l(1) = 0`, fixing the
       poloidal-flux gauge of the innermost volume (:math:`\Delta\psi_{p,1}`
       is not an input there).
   * - ``Lme``
     - all volumes, :math:`i > 1`
     - outer-boundary normal field, harmonic by harmonic:
       :math:`\sum_l \bigl( -n_i A_{\theta,i,l} - m_i A_{\zeta,i,l} \bigr)
       \bar{T}_l(+1) = V^s_i + B^s_i`, where :math:`V^s_i + B^s_i`
       (``iVns + iBns``) are the prescribed sine harmonics of
       :math:`\mathbf{B}\cdot\mathbf{x}_\theta\times\mathbf{x}_\zeta` on the
       outer boundary — zero for an ideal interface, nonzero on the
       computational boundary of a free-boundary run (the :math:`D^n_i` of
       :footcite:t:`Hudson2020`, Eq. (10)). Stored in the vector
       :math:`\mathcal{G}` (``dMG``).
   * - ``Lmg``
     - all volumes, :math:`i = 1`
     - enclosed toroidal flux:
       :math:`\sum_l A_{\theta,1,l}\,\bar{T}_l(+1) = \Delta\psi_t`.
   * - ``Lmh``
     - annular volumes, :math:`i = 1`
     - enclosed poloidal flux:
       :math:`\sum_l A_{\zeta,1,l}\,\bar{T}_l(+1) = -\,\Delta\psi_p`.

(Without stellarator symmetry the sine partners ``Lmc``, ``Lmd``, ``Lmf``
appear analogously.) For a stellarator-symmetric annular volume the count is

.. math::

   \texttt{nadof} \;=\; \underbrace{2\,\texttt{mn}\,L}_{A_{\theta},A_{\zeta}}
   \;+\; \underbrace{(\texttt{mn}-1)}_{\texttt{Lme}}
   \;+\; \underbrace{2}_{\texttt{Lmg},\,\texttt{Lmh}} .

The discrete quadratic forms
============================

Substituting the expansion into the energy and helicity integrals gives
quadratic forms in :math:`\mathbf{a}` (:footcite:t:`Hudson2020`, Eq. (11);
:footcite:t:`Hudson2012pop`, Eqs. (37)–(42)). SPECTRE stores

.. math::

   \begin{aligned}
   \int_{\mathcal{V}_v} \mathbf{B}\cdot\mathbf{B}\;dv
     &\;=\; \tfrac{1}{2}\,\mathbf{a}^{\!\top} \mathcal{A}\, \mathbf{a}
     \Big|_{\text{interior block}} , \\
   \int_{\mathcal{V}_v} \mathbf{A}\cdot\mathbf{B}\;dv
     &\;=\; \tfrac{1}{2}\,\mathbf{a}^{\!\top} \mathcal{D}\, \mathbf{a}
     \Big|_{\text{interior block}} ,
   \end{aligned}

with :math:`\mathcal{A} = \texttt{dMA}` (which additionally carries the
constraint rows and columns of the table above) and
:math:`\mathcal{D} = \texttt{dMD}`. The flux right-hand side is encoded in
:math:`\mathcal{B} = \texttt{dMB}` — a two-column matrix that is nonzero only
in the ``Lmg``/``Lmh`` rows,
:math:`\mathcal{B}_{\texttt{Lmg},1} = -1`,
:math:`\mathcal{B}_{\texttt{Lmh},2} = +1` — and the free-boundary normal-field
data in :math:`\mathcal{G} = \texttt{dMG}` (nonzero only in the ``Lme`` rows,
:math:`\mathcal{G}_{\texttt{Lme}(i)} = -(V^s_i + B^s_i)`). Only
:math:`\mathcal{A}` depends on the interface geometry (through the metric);
:math:`\mathcal{D}`, :math:`\mathcal{B}` and :math:`\mathcal{G}` do not
(:footcite:t:`Hudson2020`, Sec. III).

Metric-integral tensors
-----------------------

The entries of :math:`\mathcal{A}` and :math:`\mathcal{D}` are volume
integrals of products of two radial basis functions (value :math:`T` or
:math:`s`-derivative :math:`D`), two trigonometric factors
(:math:`c = \cos\alpha`, :math:`s = \sin\alpha`), and one Jacobian-normalized
metric element :math:`\bar{g}_{\mu\nu} \equiv g_{\mu\nu}/\sqrt{g}`
(``chebyshev_mod.volume_integrate_chebyshev``). In the
stellarator-symmetric case seven tensors are needed:

.. math::

   \begin{aligned}
   \texttt{TTssss}_{l p i j} &= \int_{-1}^{1}\!\!ds \iint d\theta\, d\zeta\;
       \bar{T}_l \bar{T}_p \;\bar{g}_{ss}\; \sin\alpha_i \sin\alpha_j , \\
   \texttt{TDstsc}_{l p i j} &= \int \bar{T}_l \bar{T}'_p \;\bar{g}_{s\theta}\;
       \sin\alpha_i \cos\alpha_j , \\
   \texttt{TDszsc}_{l p i j} &= \int \bar{T}_l \bar{T}'_p \;\bar{g}_{s\zeta}\;
       \sin\alpha_i \cos\alpha_j , \\
   \texttt{DDttcc}_{l p i j} &= \int \bar{T}'_l \bar{T}'_p \;\bar{g}_{\theta\theta}\;
       \cos\alpha_i \cos\alpha_j , \\
   \texttt{DDtzcc}_{l p i j} &= \int \bar{T}'_l \bar{T}'_p \;\bar{g}_{\theta\zeta}\;
       \cos\alpha_i \cos\alpha_j , \\
   \texttt{DDzzcc}_{l p i j} &= \int \bar{T}'_l \bar{T}'_p \;\bar{g}_{\zeta\zeta}\;
       \cos\alpha_i \cos\alpha_j , \\
   \texttt{DToocc}_{l p i j} &= \int \bar{T}'_l \bar{T}_p \;
       \cos\alpha_i \cos\alpha_j
   \end{aligned}

(the last one, with no metric factor, feeds the helicity matrix; the
non-symmetric case adds the remaining :math:`c/s` parity combinations). The
angular integrals are evaluated exactly: the product of two trig factors is
expanded into the sum/difference harmonics
:math:`\alpha_i \mp \alpha_j`, so each tensor entry reads just two Fourier
harmonics of :math:`\bar{g}_{\mu\nu}` — the :math:`(0,0)` harmonic and the
:math:`(m_i \pm m_j,\, n_i \pm n_j)` harmonic, with the corresponding
:math:`\tfrac{1}{2}` factors (index maps ``kijs``/``kija``). The harmonics of
:math:`\bar{g}_{\mu\nu}` themselves are obtained by FFT on each quadrature
surface, and the :math:`s`-integral by Gauss–Legendre quadrature with
``iquad(v)`` nodes (:footcite:t:`Hudson2020`, Eq. (12);
:footcite:t:`Hudson2012pop`, Eqs. (37)–(38)).

Assembled blocks
----------------

With row index :math:`(i,l)` and column index :math:`(j,p)`,
``matrices_mod.matrix`` assembles (stellarator-symmetric case; the
:math:`\theta\theta`, :math:`\theta\zeta`, :math:`\zeta\theta`,
:math:`\zeta\zeta` component blocks of :math:`\mathcal{A}`):

.. math::

   \begin{aligned}
   \mathcal{A}^{\theta\theta}_{(i,l),(j,p)} &=
     2 n_i n_j\,\texttt{TTssss}_{lpij}
     - 2 n_i\,\texttt{TDszsc}_{lpij}
     - 2 n_j\,\texttt{TDszsc}_{plji}
     + 2\,\texttt{DDzzcc}_{lpij} , \\
   \mathcal{A}^{\theta\zeta}_{(i,l),(j,p)} &=
     2 n_i m_j\,\texttt{TTssss}_{lpij}
     + 2 n_i\,\texttt{TDstsc}_{lpij}
     - 2 m_j\,\texttt{TDszsc}_{plji}
     - 2\,\texttt{DDtzcc}_{plji} , \\
   \mathcal{A}^{\zeta\theta}_{(i,l),(j,p)} &=
     2 m_i n_j\,\texttt{TTssss}_{lpij}
     + 2 n_j\,\texttt{TDstsc}_{plji}
     - 2 m_i\,\texttt{TDszsc}_{lpij}
     - 2\,\texttt{DDtzcc}_{lpij} , \\
   \mathcal{A}^{\zeta\zeta}_{(i,l),(j,p)} &=
     2 m_i m_j\,\texttt{TTssss}_{lpij}
     + 2 m_i\,\texttt{TDstsc}_{lpij}
     + 2 m_j\,\texttt{TDstsc}_{plji}
     + 2\,\texttt{DDttcc}_{lpij} ,
   \end{aligned}

which is the discrete form of :math:`\int \mathbf{B}\cdot\mathbf{B}\,dv`
under the field representation above (each term traceable to one product of
two field components and a metric element). The helicity matrix has only
mixed blocks,

.. math::

   \begin{aligned}
   \mathcal{D}^{\theta\zeta}_{(i,l),(j,p)} &=
     -\,\texttt{DToocc}_{plji} + \texttt{DToocc}_{lpij} , \\
   \mathcal{D}^{\zeta\theta}_{(i,l),(j,p)} &=
     +\,\texttt{DToocc}_{plji} - \texttt{DToocc}_{lpij} , \\
   \mathcal{D}^{\theta\theta} &= \mathcal{D}^{\zeta\zeta} = 0 ,
   \end{aligned}

the discrete form of
:math:`\int (\!-A_\theta\,\partial_s A_\zeta + A_\zeta\,\partial_s A_\theta)\,
ds\,d\theta\,d\zeta`. The constraint rows and columns of the table above are
added symmetrically to :math:`\mathcal{A}` (values of the boundary basis
functions :math:`\bar{T}_l(\pm 1)`, ``TT``/``RTT``/``RTM`` arrays), so that
:math:`\mathcal{A} - \mu\mathcal{D}` is a symmetric *saddle-point* (KKT)
matrix: an interior block bordered by constraint rows/columns whose diagonal
is exactly zero.

The linear Beltrami system
==========================

Stationarity of
:math:`\tfrac12\mathbf{a}^{\!\top}\mathcal{A}\mathbf{a}
+ \mathbf{a}^{\!\top}\mathcal{B}\boldsymbol{\psi}
+ \mathbf{a}^{\!\top}\mathcal{G}
- \tfrac{\mu}{2}\,\mathbf{a}^{\!\top}\mathcal{D}\mathbf{a}`
with respect to :math:`\mathbf{a}`, at fixed
:math:`\boldsymbol{\psi} = (\Delta\psi_t,\ \Delta\psi_p)^{\!\top}` and
:math:`\mu`, gives the linear system solved in
``beltrami_solver_mod.solve_beltrami_system``:

.. math::
   :name: eq-beltrami-linear

   \bigl( \mathcal{A} - \mu\,\mathcal{D} \bigr)\,\mathbf{a}
   \;=\; -\,\mathcal{B}\,\boldsymbol{\psi} \;-\; \mathcal{G} .

This is Eq. (13) of :footcite:t:`Hudson2020` and Eq. (42) of
:footcite:t:`Hudson2012pop`. Special cases as implemented:

- **Plasma volume** (``lplasmaregion``): system as written;
  :math:`\mathcal{G} = 0` except in free-boundary constraint variants.
- **Vacuum volume** (``lvacuumregion``): :math:`\mu = 0`, i.e.
  :math:`\mathcal{A}\,\mathbf{a} = -\mathcal{G} - \mathcal{B}\boldsymbol{\psi}`,
  with both fluxes :math:`(\Delta\psi_t, \Delta\psi_p)` potentially unknowns
  of the constraint iteration below.
- **Helicity constraint** (``lconstraint = 2``): :math:`\mu` is eliminated as
  an unknown of a *bordered* :math:`(N{+}1)\times(N{+}1)` system instead
  (dense path only; see :doc:`beltrami_dense`).

Derivative systems
------------------

The constraint iteration and the geometry derivatives need
:math:`\partial\mathbf{a}/\partial\mu` and
:math:`\partial\mathbf{a}/\partial\Delta\psi_p` (or
:math:`\partial\mathbf{a}/\partial\Delta\psi_t` in the vacuum volume).
Differentiating :ref:`the linear system <eq-beltrami-linear>` gives two more
solves *with the same operator*:

.. math::

   \begin{aligned}
   \bigl( \mathcal{A} - \mu\mathcal{D} \bigr)\,
       \frac{\partial\mathbf{a}}{\partial\mu}
     &\;=\; \mathcal{D}\,\mathbf{a} , \\
   \bigl( \mathcal{A} - \mu\mathcal{D} \bigr)\,
       \frac{\partial\mathbf{a}}{\partial\Delta\psi_p}
     &\;=\; -\,\mathcal{B}\,\mathbf{e}_2 .
   \end{aligned}

In the code these are the ``ideriv = 1, 2`` right-hand sides; the solutions
are stored in slots 1 and 2 of ``solution`` and of the ``Ate``/``Aze``
coefficient store. Derivatives with respect to the *geometry* lead to the
same operator with perturbed-metric right-hand sides and are covered in
:doc:`field_derivatives`.

Energy and helicity outputs
---------------------------

After the solve, the per-volume integrals are evaluated directly from the
quadratic forms (``solve_beltrami_system``):

.. math::

   \begin{aligned}
   \texttt{lBBintegral}(v) &=
     \tfrac{1}{2}\,\mathbf{a}^{\!\top}\mathcal{A}\,\mathbf{a}
     + \mathbf{a}^{\!\top}\mathcal{B}\,\boldsymbol{\psi} , \\
   \texttt{lABintegral}(v) &=
     \tfrac{1}{2}\,\mathbf{a}^{\!\top}\mathcal{D}\,\mathbf{a} ,
   \end{aligned}

the discrete magnetic energy and helicity of the volume (the reported
``helicity(v)`` is ``lABintegral``). The matrix-free backend evaluates the
products :math:`\mathcal{A}\mathbf{a}` and :math:`\mathcal{D}\mathbf{a}`
without the matrices (see :doc:`beltrami_matrix_free`), so these outputs are
available on both paths.

The constraint iteration on :math:`(\mu, \Delta\psi_p)`
=======================================================

The linear system determines :math:`\mathbf{a}` for *given*
:math:`(\mu, \Delta\psi_t, \Delta\psi_p)`. Which of these are known a priori
and which are adjusted is selected by ``lconstraint``
(``beltrami_field_mod.construct_beltrami_field``):

.. list-table::
   :header-rows: 1
   :widths: 12 30 58

   * - ``lconstraint``
     - Unknowns per volume
     - Residuals driven to zero
   * - ``-2``
     - :math:`\Delta\psi_t` (outermost volume)
     - poloidal linking current :math:`G - \texttt{curpol}`
   * - ``-1``
     - none
     - none (fields from prescribed :math:`\mu,\ \Delta\psi_t,\ \Delta\psi_p`)
   * - ``0``
     - plasma: none; vacuum: :math:`(\Delta\psi_t, \Delta\psi_p)`
     - vacuum: enclosed currents
       :math:`(I - \texttt{curtor},\ G - \texttt{curpol})`
   * - ``1``
     - plasma: :math:`(\mu, \Delta\psi_p)` (only :math:`\mu` in the innermost
       volume); vacuum: :math:`(\Delta\psi_t, \Delta\psi_p)`
     - rotational transform on the inner/outer side of each interface,
       :math:`\iota_{\mathrm{in}} - \texttt{oita}`,
       :math:`\iota_{\mathrm{out}} - \texttt{iota}` (vacuum: transform and
       :math:`G - \texttt{curpol}`)
   * - ``2``
     - :math:`\mu`
     - helicity :math:`K - K_o` (bordered system, dense path)
   * - ``3``
     - none locally; :math:`\Delta\psi_p(2{:}M_{\mathrm{vol}})` globally
     - interface surface currents :math:`\texttt{IPDt} - \texttt{isurf}`,
       solved by a global Newton iteration over volumes
       (``forces_mod.dfp100``)

For the local constraints the nonlinear system is solved per volume by a
modified-Powell Newton method with analytic Jacobian (``hybrj2``, a MINPACK
derivative): each residual evaluation is one solve of
:ref:`the linear system <eq-beltrami-linear>`, and the Jacobian entries come
from the ``ideriv = 1, 2`` derivative solves through the chain rule (e.g.
:math:`\partial\iota/\partial\mu` from
:math:`\partial\mathbf{a}/\partial\mu`; the transform itself is computed from
the straight-field-line equation, ``magnetic_field_mod.
compute_rotational_transform``; cf. :footcite:t:`Hudson2012pop`,
Eqs. (44)–(45)). The iteration stops when the mean residual falls below
``mupftol`` or after ``mupfits`` evaluations. This is the iteration on
:math:`(\mu_v, \Delta\psi_{p,v})` described in :footcite:t:`Hudson2012pop`,
Sec. IV.B and :footcite:t:`Hudson2020`, Sec. III.

Every evaluation inside this loop reuses the geometry-dependent integrals:
the metric tensors are computed once per volume per geometry
(``forces_mod.dfp100``), and only the :math:`\mu`-dependent combination
:math:`\mathcal{A} - \mu\mathcal{D}` (dense path) or its preconditioner
(matrix-free path) is rebuilt per iteration.

References
==========

.. footbibliography::
