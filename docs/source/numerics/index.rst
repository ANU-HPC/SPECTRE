########
Numerics
########

This section documents the numerical methods at the core of SPECTRE: how the
Beltrami (linear force-free) field in each volume is discretized and solved —
with a dense direct solver or with the matrix-free iterative solver — and how
derivatives of the field with respect to the interface geometry are obtained
for the outer force-balance iteration.

SPECTRE computes a multi-region relaxed MHD (MRxMHD) equilibrium through three
nested loops:

1. **Innermost — the linear Beltrami solve.** For fixed interface geometry,
   fluxes and helicity multiplier :math:`\mu`, the field in one volume is the
   solution of a single linear system. Two interchangeable backends exist: a
   dense LU factorization and a matrix-free preconditioned GMRES solver.
2. **Middle — the constraint iteration.** Per volume, :math:`\mu` and the
   enclosed poloidal flux :math:`\Delta\psi_p` are adjusted by a Newton method
   so that the field satisfies the requested constraints (interface rotational
   transform, enclosed currents, or helicity), selected by ``lconstraint``.
3. **Outermost — force balance.** The interface geometry is adjusted by a
   least-squares/Newton iteration until the total-pressure jump
   :math:`[[p + B^2/2]]` vanishes on every interface. The Jacobian of this
   iteration is obtained from linearized Beltrami solves, either column by
   column (dense) or as matrix-free Jacobian-vector products.

.. toctree::
   :maxdepth: 2

   beltrami_discretization
   beltrami_dense
   beltrami_matrix_free
   field_derivatives
   matrix_free_architecture
   performance

Where to start
==============

- :doc:`beltrami_discretization` derives the linear system that both solver
  backends solve: the representation of the vector potential, the constraints
  and their Lagrange multipliers, and the matrices
  :math:`\mathcal{A}, \mathcal{B}, \mathcal{D}, \mathcal{G}`. Read this first;
  the other documents build on its notation.
- :doc:`beltrami_dense` describes the dense direct path (the default,
  ``lbeltramisolver = 0``): assembly, LU factorization, iterative refinement,
  and its cost and memory scaling.
- :doc:`beltrami_matrix_free` describes the matrix-free path
  (``lbeltramisolver = 1`` or ``2``): the operator application without forming
  the matrix, the exact same-:math:`(m,n)` block preconditioner, the GMRES
  kernel, warm starts, when to use it and how to interpret its diagnostics.
- :doc:`field_derivatives` describes the derivative machinery on top of the
  field solve: the dense force Jacobian, matrix-free Jacobian columns,
  Jacobian-vector products (JVP), and the lagged-banded matrix-free outer
  iteration (``min_params.jvp_lag``).
- :doc:`matrix_free_architecture` summarizes the ownership, lifetime, and
  numerical-policy boundaries between the new matrix-free layers.
- :doc:`performance` gives the measured cost and memory scaling of both
  backends, guidance for splitting a node between MPI ranks and OMP
  threads, and recommended configurations per system size.

Solver-selection summary
========================

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - What
     - Input flag(s)
     - Documented in
   * - Beltrami solve, dense LU
     - ``lbeltramisolver = 0`` (default)
     - :doc:`beltrami_dense`
   * - Beltrami solve, matrix-free GMRES
     - ``lbeltramisolver = 1`` (volumes with ``nadof >= nadofgmres``) or
       ``2`` (all eligible volumes)
     - :doc:`beltrami_matrix_free`
   * - Force Jacobian, per-column
     - always available (``force_real_jac``)
     - :doc:`field_derivatives`
   * - Force Jacobian-vector products
     - ``force_real_jvp`` (single MPI rank)
     - :doc:`field_derivatives`
   * - Lagged-banded matrix-free outer iteration
     - ``[minimization] tr_solver = "banded"``, ``jvp_lag = N``
     - :doc:`field_derivatives`
