#####################################################
Field derivatives, JVPs and the outer force iteration
#####################################################

The outer equilibrium iteration adjusts the interface geometry until force
balance holds. This document describes how SPECTRE differentiates the
Beltrami field with respect to the geometry — as a full Jacobian, column by
column, or as matrix-free Jacobian-vector products (JVPs) — and how the
lagged-banded outer iteration (``min_params.jvp_lag``) uses JVPs to avoid
rebuilding the dense Jacobian at every step.

The outer problem
=================

Let :math:`\mathbf{x}` collect the geometry degrees of freedom of the
internal interfaces (Fourier harmonics of :math:`R` and :math:`Z`, or a
reduced parametrization, ``interface_dof``). The residual minimized by the
Python driver (``spectre.force_minimizer``, objective ``freal``) is the
*real-space* total-pressure jump evaluated on a :math:`(n_t, n_z)` grid of
every internal interface:

.. math::

   \mathbf{f}(\mathbf{x}) \;=\;
   \Bigl\{\, [[\, p + B^2/2 \,]]\,(\theta_k, \zeta_k;\ \mathcal{I}_v)
   \,\Bigr\}_{v = 1..M_\mathrm{vol}-1,\ k = 1..n_t n_z} ,

computed by ``force_real`` (a field solve followed by evaluating
:math:`B^2` on both sides of each interface). The equilibrium is found by a
trust-region least-squares iteration
:math:`\min_{\mathbf{x}} \tfrac12\lVert\mathbf{f}(\mathbf{x})\rVert^2`
(``spectre.least_squares``), which needs
:math:`J = \partial\mathbf{f}/\partial\mathbf{x}` or products
:math:`J\mathbf{d}`.

Linearized field response
=========================

Each residual entry depends on :math:`\mathbf{x}` through the geometry
directly *and* through the Beltrami field of the two volumes adjacent to the
perturbed interface. Differentiating the Beltrami system
(:doc:`beltrami_discretization`) at fixed :math:`(\mu, \boldsymbol{\psi})`
gives the *matrix perturbation* equation
(:footcite:t:`Hudson2012pop`, Eq. (43)):

.. math::
   :name: eq-matrix-perturbation

   \bigl( \mathcal{A} - \mu\,\mathcal{D} \bigr)\,
   \frac{\partial\mathbf{a}}{\partial x_j}
   \;=\; -\,\frac{\partial\mathcal{A}}{\partial x_j}\,\mathbf{a} ,

since :math:`\mathcal{D}`, :math:`\mathcal{B}` and :math:`\mathcal{G}` do not
depend on the geometry. The right-hand side is evaluated matrix-free in all
cases: the perturbed metric enters the same quadrature pipeline as the
operator application, with metric *derivatives* in place of the metric
(``get_perturbed_rhs`` → ``intghs``/``mtrxhs`` with ``dBdX`` active). When
the constraints tie :math:`(\mu, \Delta\psi_p)` to the geometry
(``lconstraint = 1, 3``), the total derivative adds the constraint response

.. math::

   \frac{d\mathbf{a}}{dx_j} = \frac{\partial\mathbf{a}}{\partial x_j}
   + \frac{\partial\mathbf{a}}{\partial\mu}\,\frac{d\mu}{dx_j}
   + \frac{\partial\mathbf{a}}{\partial\Delta\psi_p}\,
     \frac{d\Delta\psi_p}{dx_j} ,

with :math:`d\mu/dx_j,\ d\Delta\psi_p/dx_j` from differentiating the
constraint equations (``evaluate_dmupfdx``); the factors
:math:`\partial\mathbf{a}/\partial\mu,\
\partial\mathbf{a}/\partial\Delta\psi_p` are the ``ideriv = 1, 2`` solutions
already computed by the forward solve.

The helicity constraint (``lconstraint = 2``) ties :math:`\mu` to the
geometry through :math:`{\cal K} = \tfrac{1}{2}\mathbf{a}^T\mathcal{D}\,
\mathbf{a} = \text{const}`; since :math:`\mathcal{D}` carries no geometry
dependence, the constraint row is simply
:math:`\langle\mathcal{D}\mathbf{a},\, d\mathbf{a}\rangle = 0`. The dense
Jacobian path solves the resulting bordered system directly
(:doc:`beltrami_dense`); the JVP eliminates it through the Schur
complement over the unbordered operator: with
:math:`\mathbf{w} = \mathcal{D}\mathbf{a}` and
:math:`\mathbf{z} = (\mathcal{A} - \mu\mathcal{D})^{-1}\mathbf{w}` (both
direction-independent, built once per batch),

.. math::

   \frac{d\mathbf{a}}{dx_j} = \mathbf{y}_j + \frac{d\mu}{dx_j}\,\mathbf{z},
   \qquad
   \frac{d\mu}{dx_j}
   = -\,\frac{\langle\mathbf{w}, \mathbf{y}_j\rangle}
            {\langle\mathbf{w}, \mathbf{z}\rangle} ,

where :math:`\mathbf{y}_j` is the unconstrained directional solve. The
constrained response then conserves each volume's helicity by
construction, and no separate ``dmupfdx`` term appears downstream for the
plasma volumes (:math:`B` depends on the geometry and :math:`\mathbf{a}`
alone). In free-boundary mode the vacuum volume carries no helicity
constraint — :math:`\mu = 0` leaves no freedom to impose one — and instead
varies :math:`(\Delta\psi_t, \Delta\psi_p)` to hold the prescribed currents
:math:`(I_{\rm tor}, G_{\rm pol})`, exactly as ``lconstraint = 0`` does;
its flux response enters through the ordinary ``dmupfdx``/``dmupf_dir``
contraction while its helicity floats.

The dense Jacobian (per-column)
-------------------------------

``field_mod.solve_field_deriv`` (Python: ``force_real_jac``) builds the full
:math:`J` one geometry degree of freedom at a time. Per volume the operator
is factorized **once** and every column is a cheap back-substitution
(``op_solve_derivative_rhs_batch`` dispatches to ``op_solve_multi``); the
column count is
:math:`\mathcal{O}(\texttt{mn}\cdot M_\mathrm{vol})`, so the factorizations
amortize but the per-column right-hand-side builds and solves dominate the
Jacobian cost.

For volumes on the matrix-free backend there is no factorization to
back-substitute against; each column is solved by preconditioned GMRES at the
relaxed tolerance ``epsgmres_jac`` (default :math:`10^{-9}` — Jacobian
columns need less accuracy than the field). Two reuse mechanisms keep this
affordable:

- the metric and preconditioner are prepared once per volume and shared by
  all columns;
- each column warm-starts from the previous column's solution
  (``op_solve_multi_iter``): spectrally neighbouring right-hand sides are
  similar, so after the first (cold) column the chain converges in far fewer
  iterations. In the non-batched path (multiple MPI ranks) the seed is the
  volume's previous perturbed solution (``Ate``/``Aze`` slot −1,
  ``get_perturbed_solution``).

A stalled column solve prints a warning naming the volume ("this Jacobian
column is not converged") and continues. Multi-rank runs are supported; each
rank solves the columns of the volumes it owns.

Jacobian-vector products
========================

``field_mod.solve_field_deriv_dir`` (Python: ``force_real_jvp``) computes
:math:`J\,\mathbf{d}_k` for a batch of directions :math:`\mathbf{d}_k`
without building :math:`J`. Because
:ref:`the linearized system <eq-matrix-perturbation>` is linear in the
perturbation direction, contracting it with a full direction
:math:`\mathbf{d}` (a simultaneous displacement of *all* interfaces) needs
just **one** linearized solve per volume per direction:

1. load the direction's displacement field into the private
   ``force_jvp_context_mod`` state (``load_direction_coeffs``);
2. build the direction-contracted right-hand side
   :math:`-(\partial_{\mathbf{d}}\mathcal{A})\,\mathbf{a}` and solve it with
   the volume's operator (dense back-substitution or warm-started GMRES at
   ``epsgmres_jac``); under the helicity constraint the solve is closed by
   the Schur-complement :math:`\mu` response above, so the perturbed field
   conserves the volume's helicity;
3. add the direction-contracted constraint response
   (``evaluate_dmupfdx_dir``);
4. evaluate the perturbed :math:`B^2` response on the force grid and
   assemble the rows exactly as the per-column path does
   (``force_real_evaldbb_dir``).

The public routine is a thin coordinator around batch initialization,
per-volume directional solves, constraint evaluation, row assembly, and
owned cleanup. Dense and directional force/constraint implementations remain
separate; only the allocation-free derivative-RHS backend dispatcher is
shared with the dense Jacobian path.

The volume factorizations (or preconditioner binds) are set up once per
batch and shared by all directions, so a batch of :math:`k` directions costs
the setup once plus :math:`k` cheap applications — at 6 interfaces a
marginal direction is ~90× cheaper than the dense Jacobian build.

Contract and status
-------------------

``solve_field_deriv_dir`` requires that ``solve_field`` has already run at
the same geometry (slots 0–2 of the coefficient store hold the converged
field and its flux/:math:`\mu` derivatives). It is restricted to a single
MPI rank. The context owns the active-volume
identity, direction and axis-response buffers, and diagnostics. Its work
storage is reserved once per batch and released by a scope guard, including
on an early return; only the diagnostics persist long enough for the Python
boundary to copy them.

Outcomes are reported per volume in the private JVP context: 0 = success;
positive = localized degradation (1 =
GMRES column stall, 2 = dense back-substitution failure, 3 = degenerate
volume in the pressure term, 4 = singular local constraint response,
5 = degenerate geometry response, 6 = degenerate volume derivative,
7 = singular global (``lconstraint = 3``) constraint system); negative =
unsupported configuration (−1 = multiple ranks).
The Python wrapper ``force_real_jvp(..., return_status=True)`` exposes a
copy through a compatibility accessor. A product is only usable when every
entry is 0 — a nonzero status means the product was assembled from an
unconverged or singular solve, and consumers (the lagged optimizer in
``force_minimizer``) must discard it.

The dense Jacobian build reports the same way in ``force_jac_status``
(same codes, filled by ``solve_field_deriv``), exposed by
``force_real_jac(..., return_status=True)``, which reduces it over MPI
ranks. A Jacobian whose status is nonzero was built from degraded linear
solves and must not be trusted either.

The Beltrami factor cache
-------------------------

A JVP batch at the geometry of the preceding force evaluation would
otherwise rebuild each volume's factorization from scratch — the dominant
fixed cost of a batch. With ``allglobal_mod.lbeltrami_factor_cache = true``
(set automatically by the lagged-banded driver below), the forward solve
deposits each volume's final factorization — built at the accepted
:math:`(\text{geometry}, \mu)` — into the process-owned manager in
``beltrami_factor_cache_mod`` (``factor_cache_insert``), and the JVP batch
takes it back (``factor_cache_acquire``) instead of refactoring. The explicit
key is
:math:`(\text{volume}, \texttt{nadof}, \mu, \text{geometry generation})`;
``dfp100`` invalidates a volume before re-solving it and thereby advances
that generation, so a stale-geometry factor cannot survive into a batch.
Insert, acquire, and return move the opaque operator handle rather than
copying it. The lagged optimizer owns the cache flag and cleanup through one
scoped adapter, including retry and exception paths. The cache costs resident
memory for the dense factors, hence opt-in.

The banded trust-region solver
==============================

The force Jacobian is block-tridiagonal in the interface index: the residual
rows of interface :math:`v` respond only to the geometry of interfaces
:math:`v-1, v, v+1`, because only the Beltrami fields of the two adjacent
volumes change (measured; the only out-of-band content is the
:math:`\sim 10^{-8}`-relative constraint coupling through
:math:`d\mu/d\mathbf{x}`). The normal matrix :math:`N = J^\top J` is then
block-pentadiagonal, and the trust-region subproblem

.. math::

   (N + \alpha I)\,\mathbf{p} = -\mathbf{g},
   \qquad \lVert\mathbf{p}\rVert \le \Delta ,

is solved with a banded Cholesky factorization per :math:`\alpha` (Moré's
safeguarded Newton iteration on :math:`\alpha`
:footcite:p:`More1977`) instead of one dense SVD of :math:`J` per outer
iteration (``spectre/least_squares/banded.py``). Enabled with
``tr_solver = "banded"`` in ``[minimization]``; requires
``objective = "freal"`` and a per-interface DOF representation. The assembly
measures the Jacobian's actual out-of-band mass and each step is validated
against the in-band normal matrix; on either check failing, the driver falls
back to the exact SVD with a printed warning, so the banded solver cannot
silently change the optimization.

The lagged-banded matrix-free outer iteration (``jvp_lag``)
===========================================================

Even with the banded subproblem solver, the dominant cost per outer
iteration is building the dense Jacobian. The lagged-banded mode reuses a
stale ("lagged") Jacobian for up to ``jvp_lag`` iterations and corrects it
each iterate with a small batch of *fresh* JVPs
(``spectre/least_squares/trf.py``, helpers in
``spectre/least_squares/lagged.py``).

At a stale iterate with residual :math:`\mathbf{f}` and trust radius
:math:`\Delta`:

1. **Subspace.** Candidate directions are generated from the lagged
   factorization: the banded trust-region direction computed from the lagged
   :math:`N` and the current gradient, the (lagged) scaled gradient, and the
   previous accepted step. Their orthonormalized span is
   :math:`Q \in \mathbb{R}^{n\times k}` (:math:`k \le 3`; rank-deficient
   candidates are dropped).
2. **Fresh reduced model.** One JVP batch gives :math:`W = J\,Q` with the
   *current* Jacobian, and the step solves the reduced problem

   .. math::

      \min_{\mathbf{y}\in\mathbb{R}^k}\ \tfrac12\,
      \lVert \mathbf{f} + W\,\mathbf{y} \rVert^2
      \quad\text{s.t.}\quad \lVert\mathbf{y}\rVert \le \Delta ,
      \qquad
      \mathbf{p} = Q\,\mathbf{y}

   (:math:`\lVert Q\mathbf{y}\rVert = \lVert\mathbf{y}\rVert` since
   :math:`Q` is orthonormal), via the same Moré :math:`\alpha`-iteration on
   the small :math:`(m, k)` SVD as the exact solver.
3. **Honest radius control.** The predicted reduction — and therefore the
   trust-region gain ratio that grows or shrinks :math:`\Delta` — is
   evaluated with the fresh :math:`W`, not the lagged Jacobian, so the
   radius control remains valid even when the lagged subspace is poor.

Safeguards, all designed so the mode can degrade only to the exact method:

- convergence is **never** declared from lagged quantities: any termination
  test that fires while the Jacobian is stale forces a dense rebuild and a
  fresh recheck;
- a rejected step under a lagged model signals model quality and forces a
  rebuild;
- an unusable subspace (all candidates dropped) forces a rebuild;
- a JVP batch is consumed only if every volume reports context status 0. A
  batch reporting a failed or degraded solve is discarded, retried once
  with freshly built factorizations, and — if it fails again — dropped in
  favour of a dense rebuild. That rebuild goes through the ordinary
  Jacobian path, so each volume keeps the backend it is configured for; an
  iterative volume is never quietly moved to dense LU. If the rebuild
  itself reports degraded solves (``force_jac_status`` nonzero) the run
  prints a warning and continues, so a minimization drifting on an
  untrustworthy Jacobian is attributable;
- after ``jvp_lag`` stale iterations the Jacobian is rebuilt regardless.

How to run
----------

.. code-block:: toml

   [minimization]
   method = "trf"          # banded solver is implemented for trf
   tr_solver = "banded"
   jvp_lag = 6             # reuse the dense J for up to 6 iterations
   objective = "freal"
   inface_repr = "fourier" # per-interface DOFs; "henn" is not supported

Requirements checked at startup (each falls back with a printed warning
rather than failing): ``tr_solver = "banded"`` with ``objective = "freal"``
and a per-interface DOF representation; a single MPI rank (the JVP is
single-rank); fixed-boundary mode (in free-boundary runs the
``bnorm`` updates change the residual between iterations, invalidating a
lagged model); the full perturbation spectrum (no ``mpol_rz``/``ntor_rz``
truncation — the JVP has none). Enabling ``jvp_lag`` also switches on the
Beltrami factor cache automatically; the previous setting of that flag is
restored and the cached factorizations are released when the optimization
returns, including on an exception.

When to use
-----------

Use the lagged mode when the dense Jacobian build dominates the wall time of
the optimization — high Fourier resolution and/or many volumes — and the
iteration is in a regime of steady progress, where consecutive Jacobians are
similar and the lagged subspace is informative. Every accepted lagged step
replaces a full Jacobian build (cost
:math:`\mathcal{O}(\texttt{mn}\cdot M_\mathrm{vol})` linearized solves) with
at most three JVP directions (:math:`3\,M_\mathrm{vol}` linearized solves
plus cached factorizations). The safeguards force dense rebuilds near
convergence, so final-answer quality is that of the exact method; observed
iteration counts to a given force level can differ, which is the intended
trade of iterations for per-iteration cost.

Origins and related methods
---------------------------

The lagged-banded iteration is a recombination of three classical
ingredients rather than a new method; the references anchor each design
choice.

- **Jacobian lagging.** Reusing a stale Jacobian across Newton steps is the
  *chord* (modified Newton) method; refreshing it every :math:`m` steps is
  the Shamanskii method :footcite:p:`Shamanskii1967`, with convergence
  theory in :footcite:t:`Kelley1995`. The optimal lag depends on the cost
  ratio of a Jacobian build to a step — the trade ``jvp_lag`` exposes as a
  knob. The same amortization is default practice elsewhere: stiff ODE/DAE
  integrators (VODE, CVODE/IDA, DASSL) reuse the Jacobian and its
  factorization across many time steps, rebuilding only when the Newton
  iteration's convergence degrades — the same "rebuild on evidence of
  staleness" trigger as the rejected-step rule here. Quasi-Newton methods
  :footcite:p:`Broyden1965` are the complementary classical answer (cheap
  updates instead of rebuilds); MINPACK's ``hybrj`` — the solver SPECTRE
  uses for the per-volume :math:`(\mu, \Delta\psi_p)` iteration
  (:doc:`beltrami_discretization`) — applies Broyden rank-1 updates between
  Jacobian evaluations, so a lagged Jacobian has been in the code's inner
  loops all along.
- **Subspace trust-region steps.** Solving the trust-region subproblem over
  a low-dimensional subspace spanned by the gradient and a (quasi-)Newton
  direction is Powell's dogleg idea, generalized by
  :footcite:t:`Byrd1988` — the same construction used by the 2-D subspace
  mode of the underlying ``trf`` solver. The lagged mode's subspace adds
  the previous accepted step as a third, momentum-like direction.
- **Matrix-free products with the fresh Jacobian.** Evaluating
  :math:`J\mathbf{v}` exactly without forming :math:`J` while an expensive
  factorization is frozen from an earlier iterate is the *lagged
  preconditioner* arrangement of Jacobian-free Newton–Krylov methods
  :footcite:p:`KnollKeyes2004`, standard in large CFD and extended-MHD
  codes — and the source of the "lagging" terminology.

The distinctive choice here is the inversion of the usual quasi-Newton
arrangement: the *model* is fresh (built from current-Jacobian JVPs) and
only the *subspace* is lagged. The predicted reduction — and hence the
trust-region gain ratio — is therefore exact restricted to the subspace, so
the standard radius control and convergence safeguards remain valid without
a separate convergence theory, in the spirit of inexact-Newton analysis
:footcite:p:`Dembo1982`, which controls model error rather than assuming it
away. What makes the lag pay off at much shorter lags than in the classical
ODE setting is the SPECTRE-specific cost ratio: a JVP direction is ~90×
cheaper than a dense Jacobian build.

Tests
=====

- ``tests/wrapper/test_force_jac_findiff.py`` — Jacobian against finite
  differences;
- ``tests/wrapper/test_beltrami_jac.py`` — matrix-free Jacobian columns
  against the dense path;
- ``tests/wrapper/test_force_jac_jvp.py`` — JVPs against dense Jacobian
  products;
- ``tests/wrapper/test_banded_trust_region.py`` — banded subproblem solver
  against the exact SVD path;
- ``tests/wrapper/test_lagged_banded_trf.py`` — the lagged iteration,
  including its fallback behaviour.

References
==========

.. footbibliography::
