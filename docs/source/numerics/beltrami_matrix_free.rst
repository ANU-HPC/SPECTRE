###############################
The matrix-free Beltrami solver
###############################

The matrix-free backend solves the same linear system as the dense path,

.. math::

   \bigl( \mathcal{A} - \mu\,\mathcal{D} \bigr)\,\mathbf{a}
   \;=\; -\,\mathcal{B}\,\boldsymbol{\psi} \;-\; \mathcal{G} ,

(:doc:`beltrami_discretization`), but never forms
:math:`\mathcal{A}` or :math:`\mathcal{D}`. The operator is *applied* to a
vector by evaluating the underlying quadrature directly, the system is solved
by restarted, right-preconditioned GMRES, and the preconditioner is the exact
same-:math:`(m,n)` block of the operator, built from
:math:`\mathcal{O}(\texttt{mn}\,L^2)` data. Memory drops from
:math:`\mathcal{O}(N^2)` to :math:`\mathcal{O}(\texttt{mn}\,L^2)` and the
per-solve time from :math:`\mathcal{O}(N^3)` to
:math:`k \times` (cost of one operator application), with measured iteration
counts :math:`k` of a few tens.

.. important::

   **The matrix-free backend is a memory technology, not a run-speed
   technology.** The isolated force/forward solve *is* faster than dense
   (increasingly so with size), but the full Jacobian build — which
   dominates a conventional rebuild-per-iteration equilibrium solve —
   remains ~10× faster on the dense backend at every measured size, with
   no crossover in reach (:doc:`performance`). Switching a conventional
   run to the matrix-free backend therefore does not make it faster.
   What the backend does buy:

   1. it removes the :math:`\mathcal{O}(N^2)` dense memory wall, making
      resolutions runnable that the dense path cannot even allocate; and
   2. its operator, factor-cache, and collective-JVP machinery power the
      **lagged-banded optimizer** (:doc:`field_derivatives`), where the
      cheap force evaluations and factor-cached JVP batches do translate
      into faster outer iterations.

   See :ref:`sec-mf-outlook` for what would be required to make the
   matrix-free Jacobian itself competitive.

Implementation seams: backend selection and solve orchestration in
``fortran_src/beltrami_operator_mod.F90`` and
``fortran_src/beltrami_solver_mod.F90``; operator application in
``fortran_src/intghs_mod.F90`` + ``fortran_src/matrices_mod.F90``
(``mtrxhs_full``); preconditioner construction in
``fortran_src/beltrami_sparse_mod.F90``; GMRES and block-LU kernels in
``fortran_src/gmres_mod.F90``.

The operator handle is opaque outside ``beltrami_operator_mod`` and owns
exactly one private payload: dense LAPACK factors or the iterative CSR/block-LU
preconditioner. Solves borrow that handle; cache insertion and acquisition
move the complete payload through ``beltrami_factor_cache_mod``.

How to run
==========

Set in the ``[physics]`` block of the input TOML:

.. code-block:: toml

   [physics]
   lbeltramisolver = 1      # 0 = dense (default), 1 = threshold, 2 = always
   # nadofgmres = 5000      # NAdof threshold for lbeltramisolver = 1
   # nadofgmres_sing = ...  # opt-in for the innermost volume
   #                        # (default: excluded)
   # nitergmres = 200       # max GMRES iterations per solve
   # ngmresrestart = 30     # Krylov restart length
   # epsgmres = 1e-12       # relative residual tolerance (forward solves)
   # epsgmres_jac = 1e-9    # relative residual tolerance (Jacobian/JVP solves)

.. list-table::
   :header-rows: 1
   :widths: 22 12 66

   * - Parameter
     - Default
     - Meaning
   * - ``lbeltramisolver``
     - ``0``
     - Backend selection: ``0`` always dense (bit-identical to legacy);
       ``1`` matrix-free GMRES for volumes with
       :math:`\texttt{nadof} \ge \texttt{nadofgmres}`; ``2`` matrix-free for
       every eligible volume (testing). These are the only accepted values
       (``spectre.file_io.input_parameters.BeltramiSolver`` names them
       ``DENSE``, ``THRESHOLD`` and ``ITERATIVE``); anything else is rejected
       by the input validation rather than silently leaving volumes dense.
   * - ``nadofgmres``
     - ``5000``
     - Volume-size threshold for ``lbeltramisolver = 1``. Below it the dense
       LU is faster; above it GMRES wins on both time and memory.
   * - ``nadofgmres_sing``
     - ``huge``
     - Separate threshold for the innermost (coordinate-singularity) volume,
       which is kept dense by default — see
       :ref:`the section below <sec-coordinate-singularity>`. Lower it to
       opt that volume in.
   * - ``nitergmres``
     - ``200``
     - Total inner-iteration budget per solve. A solve that exhausts it
       returns its best iterate and raises a warning.
   * - ``ngmresrestart``
     - ``30``
     - Restart length :math:`m` of GMRES(:math:`m`) — the Krylov-basis
       memory is :math:`\mathcal{O}(m N)`. Must be :math:`\ge 1`.
   * - ``epsgmres``
     - ``1e-12``
     - Relative residual tolerance
       :math:`\lVert \mathbf{b} - \mathcal{K}\mathbf{x} \rVert \le
       \texttt{epsgmres}\,\lVert\mathbf{b}\rVert` for the forward field
       solves.
   * - ``epsgmres_jac``
     - ``1e-9``
     - Tolerance for the Jacobian-column and JVP solves
       (:doc:`field_derivatives`), which need less accuracy than the field
       itself.

Eligibility (per volume)
------------------------

``op_select_backend`` picks the backend per volume. The dense path is used
whenever any of the following holds:

- ``lbeltramisolver = 0`` (default);
- the case is not stellarator-symmetric (the same-:math:`(m,n)`
  preconditioner is implemented for the symmetric block structure only);
- the volume is the innermost (coordinate-singularity) volume with
  :math:`\texttt{nadof} < \texttt{nadofgmres\_sing}` (excluded by default;
  does not apply in slab geometry, ``igeometry = 1``, which has no
  coordinate singularity);
- ``lbeltramisolver = 1`` and :math:`\texttt{nadof} < \texttt{nadofgmres}`.

Mixed configurations are normal: at production resolution the interior and
vacuum volumes run matrix-free while the innermost volume stays dense.
Multi-rank MPI runs are supported for the forward solve, Jacobian, and
collective JVP. Each rank prepares and solves the volumes it owns; the JVP
combines directional constraint and force responses so every rank returns
identical products and diagnostics (:doc:`field_derivatives`).

When to use it
--------------

- **The forward/force solve of any production-size volume.** Measured on
  the 7-volume NCSX ladder, the iterative force solve is faster than dense
  from the smallest measured volumes (``nadof`` ≲ 10³) and the advantage
  grows with size — ~8× at ``nadof`` ~10⁴ (:doc:`performance`). The
  ``nadofgmres = 5000`` default is deliberately conservative: below it the
  dense LU is unconditionally robust and cheap enough that the choice
  barely matters.
- **Memory-limited runs.** The dense path needs
  :math:`3 \times 8(N{+}1)^2` bytes per volume being solved; the matrix-free
  path needs the preconditioner blocks and the Krylov basis, orders of
  magnitude less. At the ``m21n17_L12`` benchmark the dense force
  evaluation exceeds desktop-class memory and cannot run; matrix-free
  peaks at ~15 GB and makes the case routinely runnable.
- **Not the full dense-Jacobian build**: one dense LU amortized over all
  ``ngdof`` right-hand sides beats per-column GMRES by a roughly
  size-independent ~10× (measured across the whole 7-volume ladder). Use
  the dense backend for Jacobian rebuilds whenever they fit in memory, and
  the JVP-lagged optimizer to avoid most rebuilds altogether
  (:doc:`field_derivatives`).
- Vacuum-volume solves converge in ~46 iterations, interior-plasma solves
  in ~88, largely independent of resolution.
- The results agree with the dense path to the GMRES tolerance (e.g. the
  benchmark force norm matches to all printed digits) but are *not* bitwise
  identical — the dense path remains the bit-exact reference.

The matrix-free operator application
====================================

One application :math:`\mathbf{x} \mapsto
(\mathcal{A}\mathbf{x},\ \mathcal{D}\mathbf{x})` reproduces the dense matrix
action to round-off by evaluating the same quadrature the assembler uses,
in five steps (``op_apply`` / ``op_apply_ad``):

1. **Unpack** (``packing_mod.packab_matvec``): scatter the augmented vector
   :math:`\mathbf{x}` into radial/Fourier coefficients
   :math:`x^\theta_{i,l},\ x^\zeta_{i,l}` (a scratch slot of the
   ``Ate``/``Aze`` store) and the Lagrange multipliers
   :math:`\lambda_a, \lambda_b, \lambda_e, \lambda_g, \lambda_h`.

2. **Contravariant field at the quadrature nodes** (``intghs``): at each
   Gauss node :math:`s_q` build the Fourier harmonics of the field the
   coefficients induce,

   .. math::

      \bigl[\sqrt{g}B^\zeta\bigr]^{\cos}_i(s_q) =
          \sum_l x^\theta_{i,l}\,\bar{T}'_l(s_q), \qquad
      \bigl[\sqrt{g}B^\theta\bigr]^{\cos}_i(s_q) =
          -\sum_l x^\zeta_{i,l}\,\bar{T}'_l(s_q),

   .. math::

      \bigl[\sqrt{g}B^s\bigr]^{\sin}_i(s_q) =
          -\sum_l \bigl( n_i\,x^\theta_{i,l} + m_i\,x^\zeta_{i,l} \bigr)
          \bar{T}_l(s_q),

   and inverse-FFT them to the real-space angular grid.

3. **Lower the index**: with the Jacobian-normalized metric
   :math:`\bar{g}_{\mu\nu} = g_{\mu\nu}/\sqrt{g}` evaluated at the same
   nodes,

   .. math::

      B_\mu(s_q,\theta,\zeta) \;=\; \sum_\nu
      \bar{g}_{\mu\nu}\,\bigl(\sqrt{g}B^\nu\bigr) ,

   then forward-FFT :math:`B_s, B_\theta, B_\zeta` back to harmonics. The
   metric is computed **once per volume per solve** into a workspace
   (``op_prepare_metric`` → ``pws%guvijsave``) and reused by every matvec of
   that solve; the per-node FFTs are the dominant matvec cost and are
   OpenMP-parallel over quadrature nodes.

4. **Radial projections** (``intghs``): Gauss-quadrature the harmonics
   against the basis,

   .. math::

      \begin{aligned}
      \texttt{Tss}_{l,i} &= \int\! ds \iint\! d\theta\, d\zeta\;
          \bar{T}_l\, B_s\, \sin\alpha_i , &
      \texttt{Dtc}_{l,i} &= \int \bar{T}'_l\, B_\theta\, \cos\alpha_i , \\
      \texttt{Dzc}_{l,i} &= \int \bar{T}'_l\, B_\zeta\, \cos\alpha_i , &
      \texttt{Ttc}_{l,i} &= \int \bigl( \bar{T}_l\,(-\partial_s A_\zeta)
          + \bar{T}'_l\, A_\zeta \bigr) \cos\alpha_i , \\
      & &
      \texttt{Tzc}_{l,i} &= \int \bigl( \bar{T}_l\,\partial_s A_\theta
          - \bar{T}'_l\, A_\theta \bigr) \cos\alpha_i ,
      \end{aligned}

   where :math:`A_\theta, A_\zeta` here denote the fields induced by
   :math:`\mathbf{x}` (the ``Ttc``/``Tzc`` projections feed
   :math:`\mathcal{D}\mathbf{x}` and involve no metric).

5. **Assemble the result rows** (``matrices_mod.mtrxhs_full``):

   .. math::

      (\mathcal{A}\mathbf{x})_{A_\theta,i,l} =
        -2 n_i\,\texttt{Tss}_{l,i} + 2\,\texttt{Dzc}_{l,i}
        + [\text{multiplier columns}] ,
      \qquad
      (\mathcal{A}\mathbf{x})_{A_\zeta,i,l} =
        -2 m_i\,\texttt{Tss}_{l,i} - 2\,\texttt{Dtc}_{l,i}
        + [\text{multiplier columns}] ,

   .. math::

      (\mathcal{D}\mathbf{x})_{A_\theta,i,l} = \texttt{Ttc}_{l,i} ,
      \qquad
      (\mathcal{D}\mathbf{x})_{A_\zeta,i,l} = \texttt{Tzc}_{l,i} ,

   plus, exactly as in the dense matrix, the Lagrange coupling: the interior
   rows receive the multiplier values times the boundary basis values
   (columns of the constraint table in :doc:`beltrami_discretization`), and
   the multiplier rows receive the constraint expressions evaluated on the
   interior part of :math:`\mathbf{x}`. The GMRES operator is then

   .. math::

      \mathcal{K}(\mu)\,\mathbf{x} \;=\;
      \mathcal{A}\mathbf{x} - \mu\,\mathcal{D}\mathbf{x} ,

   with :math:`\mu = 0` in the vacuum volume.

Correctness of this pipeline is enforced by
``tests/wrapper/test_beltrami_matvec.py``, which compares the matrix-free
application against explicit dense :math:`\mathcal{A}, \mathcal{D}` products.

An operator may also own a compact immutable correction to selected rows of
:math:`\mathcal{A}`. ``op_set_row_correction`` snapshots the row indices and
values into the opaque handle; apply subtracts those row products after the
ordinary action, while :math:`\mathcal{D}` and the same-mode preconditioner
remain unchanged. The correction follows move/cache ownership and is cleared
by release or rebind. Consequently GMRES tests convergence with the corrected
operator even though its preconditioner is built from the uncorrected block.

One application costs
:math:`\mathcal{O}(\texttt{iquad}\cdot N_{tz}\log N_{tz})` for the FFTs plus
:math:`\mathcal{O}(\texttt{mn}\,L\,\texttt{iquad})` for the sums —
linear in the resolution parameters, versus :math:`\mathcal{O}(N^2)` for a
dense matrix-vector product.

The same-:math:`(m,n)` preconditioner
=====================================

GMRES on the raw operator converges slowly; the implemented preconditioner
:math:`\mathcal{M}` is the **exact same-mode block** of
:math:`\mathcal{K}(\mu)`: all entries coupling degrees of freedom of the same
Fourier harmonic :math:`(m_i, n_i)` — including that harmonic's Lagrange
multipliers — and nothing else. Because each harmonic appears exactly once,
:math:`\mathcal{M}` is block-diagonal with one block per mode of size
:math:`\mathcal{O}(L)` (radial coefficients of both components plus the
mode's multipliers).

Two properties make this preconditioner effective and cheap:

- **It is exact, not approximated.** ``build_samemn_csr_diag``
  (``beltrami_sparse_mod``) evaluates the *same* tensor expressions the
  dense assembler uses, restricted to mode pairs :math:`j = i`, from
  diagonal-only metric-integral tensors produced by
  ``volume_integrate_chebyshev(..., diag_only=.true.)`` at
  :math:`\mathcal{O}(\texttt{mn}\,\texttt{iquad}\,L^2)` cost — the diagonal
  mode pair pulls exactly two metric harmonics,
  :math:`\bar g^{(0,0)} \pm \tfrac12 \bar g^{(2m_i,\,2n_i)}`. The result is
  bitwise identical to extracting the block from the assembled dense matrix
  (``tests/wrapper/test_diag_integrals.py``,
  ``tests/wrapper/test_beltrami_precond.py``; the dense extraction
  ``build_samemn_csr`` is retained as the independent test oracle).
- **Each block is factored with partial pivoting** (``gmres_mod.block_lu``,
  one ``DGETRF`` per mode block). The operator is a saddle-point matrix —
  the Lagrange rows have genuinely zero diagonals — which an unpivoted
  incomplete factorization cannot handle without diagonal regularization;
  pivoted per-block LU handles it exactly, and no regularization is applied
  (measured on the ``m13n11`` vacuum volume: boosting the diagonal costs
  953 iterations versus 44 without). A block that is exactly singular is
  shifted by :math:`10^{-10}` of its diagonal scale and refactored, affecting
  only the preconditioner, never the operator.

The preconditioner depends on :math:`\mu`, so it is rebuilt from the (fixed,
geometry-dependent) diagonal tensors at each step of the
:math:`(\mu, \Delta\psi_p)` constraint iteration
(``op_bind_iter_sparse``) — a cheap resummation, not a re-integration.

The GMRES kernel
================

``gmres_mod.gmres_solve`` is a self-contained restarted GMRES(:math:`m`)
with right preconditioning :footcite:p:`Saad1986`: it solves

.. math::

   \mathcal{K}\,\mathcal{M}^{-1}\,\mathbf{y} = \mathbf{b},
   \qquad \mathbf{x} = \mathcal{M}^{-1}\mathbf{y} ,

building the Krylov basis by Arnoldi with modified Gram–Schmidt and updating
the least-squares residual with Givens rotations. Right preconditioning keeps
the convergence test in the *true* residual norm: the solve stops when

.. math::

   \lVert \mathbf{b} - \mathcal{K}\mathbf{x} \rVert
   \;\le\; \texttt{epsgmres}\cdot\lVert\mathbf{b}\rVert ,

recomputed from :math:`\mathbf{x}` at every restart. The Givens estimate is
used only to leave a restart cycle early — after a loss of Arnoldi
orthogonality it can drift below the tolerance while the true residual has
not — so success is never declared on it.

The residual itself is formed in floating point and therefore carries an
uncertainty of order
:math:`\varepsilon\,(\lVert\mathcal{K}\rVert\lVert\mathbf{x}\rVert + \lVert\mathbf{b}\rVert)`,
which for these saddle-point systems sits well above
:math:`\varepsilon\lVert\mathbf{b}\rVert`. A tolerance below that floor cannot
be measured, so reaching the floor counts as convergence (the iterate is
correct to working precision); :math:`\lVert\mathcal{K}\rVert` is estimated
from the matvecs the iteration performs anyway. A residual that stops
improving while still above both the tolerance and the floor is a stalled
solve: it stops there with a nonzero status rather than spending the rest of
the ``nitergmres`` budget. Exhausting that budget likewise returns the best
iterate with a nonzero status.

The restart length is ``ngmresrestart`` (Krylov storage
:math:`\mathcal{O}(\texttt{ngmresrestart}\cdot N)`); it must be at least 1,
since a restart length of 0 would leave the solve without a single inner
iteration. The kernel takes the operator and preconditioner as procedure
arguments and is unit-tested standalone
(``tests/wrapper/test_gmres_kernel.py``).

Warm starts
-----------

Every iterative solve is seeded from the persistent coefficient store: the
converged solution (and its two flux/:math:`\mu` derivatives) of the previous
solve of the same volume are kept in ``Ate``/``Aze`` slots 0–2 and packed
into the initial guess (``packab('P')``). Seeding therefore acts both across
:math:`\mu`-iterations within one force evaluation and across force
evaluations of an optimization. The semantics are *seed-only*: GMRES judges
convergence on the true residual of the current system, so a stale seed can
only cost iterations, never correctness, and a zero store reduces to a cold
start. Measured effect at ``m21n17``: 11–23 % fewer iterations at typical
optimizer step sizes, growing to ~50 % for very small steps
(``tests/wrapper/test_beltrami_warmstart.py``; the global counter
``niters_gmres_total`` in ``allglobal_mod`` exposes iteration counts to
Python).

Derivative right-hand sides and integrals
-----------------------------------------

The :math:`\mu`-derivative system needs
:math:`\mathcal{D}\mathbf{a}` as its right-hand side and the
energy/helicity outputs need :math:`\mathbf{a}^{\!\top}\mathcal{A}\mathbf{a}`
and :math:`\mathbf{a}^{\!\top}\mathcal{D}\mathbf{a}`; the dense path reads
these off the stored matrices. The iterative path computes both products once
per converged solve with the split application ``op_apply_ad``
(:math:`\mathcal{A}\mathbf{a}` and :math:`\mathcal{D}\mathbf{a}`
separately, matrix-free) and evaluates

.. math::

   \begin{aligned}
   \texttt{lBBintegral} &=
     \tfrac12\,\mathbf{a}\cdot(\mathcal{A}\mathbf{a})
     + \mathbf{a}\cdot(\mathcal{B}\boldsymbol{\psi}) , \\
   \texttt{lABintegral} &=
     \tfrac12\,\mathbf{a}\cdot(\mathcal{D}\mathbf{a}) .
   \end{aligned}

Diagnostics and failure modes
=============================

A GMRES solve that reaches ``nitergmres`` without meeting the tolerance
**does not abort the run**: the best iterate is kept, the volume remains
usable, and the situation is reported in two ways:

- an explicit warning naming the volume::

     solve_beltrami_system : WARNING : GMRES did not converge
     in volume   1 (NAdof=  13296) after 4000 iterations; the
     field for this volume is not converged. Raise nitergmres,
     or keep this volume on the dense path.

- per-volume status arrays in ``allglobal_mod``, readable from Python:
  ``beltrami_gmres_status(v)`` (0 = converged, 1 = iteration limit),
  ``beltrami_gmres_iterations(v)`` (iterations accumulated this evaluation)
  and ``beltrami_gmres_residual(v)`` (worst relative residual).

An unconverged field degrades the computed force rather than failing loudly
downstream, so these statuses should be checked when force values look
suspect. Remedies, in order: raise ``nitergmres``; raise ``ngmresrestart``;
return the offending volume to the dense path (raise ``nadofgmres``, or do
not lower ``nadofgmres_sing``).

.. _sec-coordinate-singularity:

The coordinate-singularity volume
=================================

The innermost volume is excluded from the iterative backend by default
(``nadofgmres_sing = huge``), based on measured structure, not caution
(analysis in ``benchmarks/matrixfree_preconditioner_research.md``):

- Near the axis the mapping is polar-like and fully shaped, so the Fourier
  content of the metric is broad: the mode-space coupling of the operator
  does **not** decay with mode separation (flat at 10–22 % of the diagonal
  out to distance 4 in :math:`m`), where a regular annular volume falls to
  5–8 % at distance one.
- The Zernike basis functions scale as :math:`\bar{s}^{\,m}`, concentrating
  high-:math:`m` blocks at the outer edge: the block-diagonal dominance
  :math:`\lVert A_{mm}\rVert/\lVert A_{m,\mathrm{rest}}\rVert` falls from
  0.49 at :math:`m = 0` to 0.13 at :math:`m = 12`, so any block-Jacobi-type
  preconditioner amplifies the neglected coupling.

Same-:math:`(m,n)`, same-:math:`m`, same-:math:`n`, banded and two-grid
preconditioners all stall on this volume (:math:`\ge 1200` iterations), and a
stalled solve *degrades the force silently* (measured at ``m21n17``: force
norm 2.5947 instead of 2.4638) — hence the default exclusion and the loud
warning if an opted-in singular volume stalls. Freezing a previous dense
factorization as preconditioner across geometry steps was also measured
infeasible: the axis volume's operator changes by :math:`\mathcal{O}(1)` for
relative geometry steps of :math:`10^{-4}`.

The volume can be opted in by lowering ``nadofgmres_sing`` below its
``nadof``; this is only useful in combination with work that changes the
axis-volume conditioning itself (e.g. the conformal interior-mesh study in
``benchmarks/matrixfree_preconditioner_research.md``, where the same
preconditioner then converges in ~50 iterations).

Tests
=====

``tests/wrapper/`` contains the matrix-free validation suite:

- ``test_gmres_kernel.py`` — GMRES and block-LU kernels standalone;
- ``test_beltrami_matvec.py`` — matrix-free operator application against
  dense matrix products;
- ``test_diag_integrals.py`` — diagonal-only metric-integral tensors against
  the full integration;
- ``test_beltrami_precond.py`` — sparse preconditioner against dense
  extraction (bitwise);
- ``test_beltrami_solve.py``, ``test_beltrami_forward.py`` — full solves
  against the dense path;
- ``test_beltrami_warmstart.py`` — warm-start iteration savings and
  correctness;
- ``test_beltrami_jac.py``, ``test_force_jac_jvp.py`` — the derivative paths
  (:doc:`field_derivatives`).

.. _sec-mf-outlook:

Outlook: what would make the matrix-free Jacobian fast
======================================================

The Jacobian build's ~10× dense advantage is structural, not a tuning
matter: the dense path amortizes one :math:`\mathcal{O}(N^3)`
factorization over every one of the :math:`\mathcal{O}(10^3\text{–}10^4)`
columns (each marginal column is a BLAS-3 multi-right-hand-side
triangular solve, near machine peak), while restarted GMRES amortizes
*nothing* across columns — every column pays :math:`k \sim 40\text{–}90`
bandwidth-bound matvecs from scratch, and warm starts recover only a
constant factor. Three routes would change that balance, in increasing
order of invasiveness:

1. **Amortize across columns: block and recycled Krylov methods.** Solve
   the perturbed systems of a Jacobian slab *simultaneously* with a block
   GMRES (one operator application per iteration serves every
   right-hand side, restoring BLAS-3 arithmetic intensity), or recycle
   the Krylov subspace between consecutive columns (GCRO-DR-type
   deflation — adjacent geometry perturbations produce closely related
   systems). This attacks the amortization asymmetry directly and fits
   the existing chunked column engine.

2. **Never build the matrix: Krylov methods on the outer subproblem.**
   The trust-region step needs the action of :math:`J` and
   :math:`J^\top`, not the entries of :math:`J`: an LSMR/CG-on-normal-
   equations subproblem solver driven by Jacobian-vector and
   transpose-vector products would remove the full build from the
   iteration entirely. The forward product exists (``force_real_jvp``).
   The missing piece is the **adjoint**: :math:`J^\top \mathbf{v}`
   requires the adjoint of the directional linearised solve — adjoints
   of the metric/quadrature integral contractions of the operator
   application chain (``intghs``/``mtrxhs``), adjoint Beltrami solves
   (the self-adjointness of :math:`\mathcal{A} - \mu\mathcal{D}` means
   the factor caches and preconditioners can be reused), and adjoints of
   the geometry-perturbation right-hand-side assembly and of the
   constraint coupling (:math:`d\mu/d\mathbf{x}`). Hand-deriving and
   validating that adjoint chain is the substantial part of this route.

3. **Cut the iteration count.** Every unit of :math:`k` multiplies the
   per-column cost; a preconditioner that brought :math:`k` from ~50
   toward ~5–10 (two-grid across radial resolution, or coupling-aware
   extensions of the same-:math:`(m,n)` block) would shrink the gap
   proportionally, and would compound with either route above.

Until one of these exists, the supported answer to Jacobian cost is the
one measured in :doc:`performance`: build full Jacobians on the dense
backend where they fit, and use the lagged-banded optimizer so that most
iterations pay only for a handful of JVP directions rather than a build.

References
==========

.. footbibliography::
