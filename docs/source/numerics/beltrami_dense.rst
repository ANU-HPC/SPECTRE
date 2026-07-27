#################################
The dense Beltrami solver
#################################

The dense path is the default backend (``lbeltramisolver = 0``) and the
reference implementation: it forms the operator
:math:`\mathcal{K}(\mu) \equiv \mathcal{A} - \mu\,\mathcal{D}` of
:doc:`beltrami_discretization` explicitly as an
:math:`N\times N` matrix (:math:`N = \texttt{nadof}(v)`) and solves it by LU
factorization. It is used for every volume unless the matrix-free backend is
explicitly enabled *and* the volume is eligible
(:doc:`beltrami_matrix_free`), and it is the only backend for the bordered
helicity constraint (``lconstraint = 2``) and for
non-stellarator-symmetric cases.

Assembly
========

Per volume and per geometry, in ``forces_mod.dfp100``:

1. ``chebyshev_mod.volume_integrate_chebyshev`` computes the full
   metric-integral tensors
   (:math:`\texttt{TTssss}, \dots, \texttt{DToocc}`) for **all** mode pairs
   :math:`(i,j)`:
   time :math:`\mathcal{O}(\texttt{mn}^2\, \texttt{iquad}\, L^2)`, memory
   :math:`\mathcal{O}(\texttt{mn}^2 L^2)` for the tensors.
2. ``matrices_mod.matrix`` assembles :math:`\mathcal{A}` (``dMA``) and
   :math:`\mathcal{D}` (``dMD``) from the tensors, including the
   Lagrange-multiplier rows and columns, and
   ``matrices_mod.matrixBG`` fills the flux/boundary data
   :math:`\mathcal{B}` (``dMB``) and :math:`\mathcal{G}` (``dMG``).

Both matrices are stored at size :math:`(N{+}1)\times(N{+}1)` (row/column 0
is a scratch slot for eliminated coefficients). Their memory is the dominant
cost at high resolution: :math:`8 (N{+}1)^2` bytes each, and the solve adds
two more arrays of the same size (the combination
:math:`\mathcal{A} - \mu\mathcal{D}` and its LU factor). At the
``m21n17_L12`` benchmark
resolution (``mpol = 21``, ``ntor = 17``, ``lrad = 12``,
:math:`N = 36\,898` in the vacuum volume) each matrix is ~10.9 GB, and the
dense force evaluation exceeds 69 GB resident — the practical motivation for
the matrix-free backend.

Factorization and solve
=======================

``solve_beltrami_system`` forms
:math:`\mathcal{K} = \mathcal{A} - \mu\,\mathcal{D}` for the current
:math:`\mu` and delegates to the ``beltrami_operator`` handle
(``beltrami_operator_mod``), which wraps LAPACK:

1. **Factor + condition estimate** (``op_factor_copy``): copy
   :math:`\mathcal{K}` into the handle's storage, LU-factorize with partial
   pivoting (``DGETRF``), then estimate the infinity-norm condition number
   (``DGECON``). A reciprocal condition number below machine precision is
   reported as "ill conditioned" and the volume is flagged failed
   (``ImagneticOK(v) = .false.``).
2. **Solve + iterative refinement** (``op_solve_refine``): back-substitute
   the right-hand sides (``DGETRS``) and refine each solution against the
   *unfactored* copy of :math:`\mathcal{K}` (``DGERFS``), which improves the
   backward error of the LU solve for the cost of a few extra
   matrix–vector products per right-hand side.

Three right-hand sides are solved per outer evaluation (see
:doc:`beltrami_discretization`):

.. math::

   \mathcal{K}\,\mathbf{a} = -\mathcal{B}\boldsymbol{\psi} - \mathcal{G},
   \qquad
   \mathcal{K}\,\frac{\partial \mathbf{a}}{\partial \mu} = \mathcal{D}\mathbf{a},
   \qquad
   \mathcal{K}\,\frac{\partial \mathbf{a}}{\partial \Delta\psi_p} =
      -\mathcal{B}\mathbf{e}_2 ,

the second and third sharing the factorization of the first (``ideriv = 0``
then ``ideriv = 1`` with two columns). Within one constraint iteration
(:math:`\mu` changes, geometry fixed) the matrices
:math:`\mathcal{A},\ \mathcal{D}` are reused and only the combination and its
factorization are redone.

The bordered helicity system (``lconstraint = 2``)
==================================================

When the helicity itself is constrained, :math:`\mu` is an unknown coupled to
:math:`\mathbf{a}`, and the system is augmented by one row and column into an
:math:`(N{+}1)\times(N{+}1)` *bordered* matrix (handle flag
``bordered = .true.``, factored by ``op_factor_prepared`` over the full
:math:`0{:}N` block). This variant exists only on the dense path;
``op_select_backend`` never chooses the iterative backend when
``lconstraint = 2``.

Cost profile
============

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Step
     - Time
     - Memory
   * - metric-integral tensors
     - :math:`\mathcal{O}(\texttt{mn}^2\,\texttt{iquad}\,L^2)`
     - :math:`\mathcal{O}(\texttt{mn}^2 L^2)`
   * - assembly of :math:`\mathcal{A}, \mathcal{D}`
     - :math:`\mathcal{O}(\texttt{mn}^2 L^2)`
     - :math:`2\times 8(N{+}1)^2` B
   * - LU factorization (``DGETRF``)
     - :math:`\mathcal{O}(N^3)`
     - :math:`8(N{+}1)^2` B
   * - per extra right-hand side (``DGETRS``)
     - :math:`\mathcal{O}(N^2)`
     - —

Since :math:`N \approx 2\,\texttt{mn}\,L`, both time and memory grow rapidly
with Fourier resolution (:math:`\texttt{mn} \approx (M{+}1)(2N_{\mathrm{tor}}{+}1)`).
The dense path is preferred at small and moderate resolution — where the LU
factorization is cheap and unconditionally robust — and as the reference
against which the matrix-free path is validated: the matrix-free operator
application and preconditioner are tested against the dense
:math:`\mathcal{A}, \mathcal{D}` extraction
(``tests/wrapper/test_beltrami_matvec.py``,
``tests/wrapper/test_beltrami_precond.py``), and enabling the matrix-free
backend leaves the dense branch bit-identical to the legacy behaviour.
