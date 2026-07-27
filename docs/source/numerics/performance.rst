###################################################
Performance: scaling, parallelism, parameter choice
###################################################

This document explains what the field and Jacobian solves cost as a function
of resolution, how that cost responds to MPI ranks and OpenMP threads, and
which solver options to choose for a given system size — for the dense path
(:doc:`beltrami_dense`) and the matrix-free/sparse-preconditioned path
(:doc:`beltrami_matrix_free`).

.. attention::

   **Snapshot — this guidance will be revised.** Several performance
   improvements are in flight on separate branches and are *not* reflected
   in the measurements below:

   - a merge-request stack that threads the Jacobian right-hand-side
     assembly (measured 2–2.75× on the Jacobian solve; it moves the optimal
     core split toward **1 rank × 8–12 threads**, and MPI ranks compose
     poorly with the threaded Jacobian);
   - the matrix-free Beltrami backend and the lagged-banded outer iteration
     (documented in this section; developed on their own branch);
   - axis-volume formulation work (conformal interior mesh) that would make
     the innermost volume eligible for the iterative backend.

   The concrete numbers below were measured on one 16-core hybrid desktop
   (6 P-cores + 8 E-cores, 62 GB, OpenBLAS) on an NCSX free-boundary case
   ladder — treat the *structure* (models, ratios, saturation points) as
   transferable and the constants as machine-specific. This page will be
   re-measured and rewritten once the pending branches land.

   Measurement provenance: ``benchmarks/report/report.pdf`` (dense scaling
   campaign, 68 runs, fitted constants in ``benchmarks/analysis/fit.json``)
   and ``benchmarks/matrixfree_preconditioner_research.md`` (matrix-free
   benchmarks).

The cost structure
==================

Size measures
-------------

For Fourier resolution :math:`(M_{\mathrm{pol}}, N_{\mathrm{tor}})` and
radial resolution :math:`L_v = \texttt{lrad}(v)`:

.. math::

   \texttt{mn} = 1 + N_{\mathrm{tor}} + M_{\mathrm{pol}}(2N_{\mathrm{tor}}+1),
   \qquad
   \texttt{nadof}_v \approx 2\,\texttt{mn}\,L_v + \texttt{mn} + 1 ,

with the free-boundary convention often putting the vacuum volume at
:math:`L_{\mathrm{vac}} = 2L` (so its ``nadof`` is ~2× and its dense-LU cost
~8× an interior volume). Aggregates that appear in the fitted models:

.. math::

   S_2 = \sum_v \texttt{nadof}_v^2, \qquad
   S_3 = \max_{\text{rank } r} \sum_{v \in r} \texttt{nadof}_v^3, \qquad
   S_{1m} = \texttt{mn} \sum_v \texttt{nadof}_v ,

where :math:`S_3` sums over the volumes owned by the *slowest* rank
(volumes are distributed round-robin over MPI ranks), and :math:`S_{1m}`
tracks the grid/metric workspace arrays.

Dense time and memory models
----------------------------

The measured dense-path campaign is reproduced across three and a half
decades of :math:`S_3` by an affine model per solve kind,

.. math::

   t \;\approx\; t_0(T) \;+\; c(T)\, S_3 ,

(:math:`T` = OMP threads): a per-call overhead :math:`t_0` plus LU work
linear in :math:`S_3`, i.e. cubic in :math:`\texttt{mn}\cdot L`. The
overhead dominates below :math:`S_3 \approx 4\times 10^{10}` (roughly
``mpol=9, ntor=7, lrad=12``) — small cases do not benefit from
parallelization. Peak memory follows

.. math::

   \mathrm{mem} \;\approx\; 0.36\,\mathrm{GB}\cdot R
   \;+\; 26\,\mathrm{B}\cdot S_2
   \;+\; 270\,\mathrm{B}\cdot R\, S_{1m}

to 14 % relative RMS: ~26 B per dense-matrix element (the
:math:`\mathcal{A}/\mathcal{D}`/LU copies) plus **per-rank replicated**
grid/workspace arrays — the same case needs less memory at fewer ranks
(measured at ``m13n11 L12``: 18.0 GB at 3 ranks, 12.2 GB at 1 rank).

``benchmarks/scripts/estimate_cost.py`` evaluates these models with the
fitted constants:

.. code-block:: bash

   python benchmarks/scripts/estimate_cost.py --mpol 21 --ntor 17 --lrad 12 \
       [--nvol 2] [--ranks 3] [--threads 4]

Use it before launching a large case: it predicts the Jacobian and
force-evaluation times and, more importantly, whether the case fits in
memory at the planned rank count.

Fixed cost ratios worth knowing
-------------------------------

- One accepted outer iteration averages **1 Jacobian + ~2.2 force
  evaluations**, and the Jacobian solve costs **~3.6×** a force evaluation
  at the same size — the Jacobian is the first thing to optimize.
- Within a volume, the ``ideriv = 0`` factorization dominates: the
  derivative solves (``ideriv ≥ 1``) reuse the LU and are 4–5× cheaper.
- The **vacuum volume gates everything** when it runs at
  :math:`L_{\mathrm{vac}} = 2L` (~8× an interior volume). Its ``lrad`` is
  the single largest dense-cost lever — reducing it trades cost against
  vacuum-field accuracy, which must be validated per case.
- The innermost (Zernike) volume has roughly half the ``nadof`` of an
  interior volume at equal ``lrad`` and is 4–5× cheaper in absolute terms.
  It never gates a collective solve — which is also why keeping it on the
  dense path when the rest runs matrix-free costs little
  (:doc:`beltrami_matrix_free`).

MPI ranks × OMP threads
=======================

What parallelizes over what:

- **MPI** distributes *volumes* round-robin
  (``allglobal_mod.IsMyVolume``); the useful rank count is bounded by
  :math:`M_{\mathrm{vol}}`, and the collective solve waits for the slowest
  rank (the :math:`S_3` gating picture above). The forward solve and the
  Jacobian both run multi-rank on either backend; the JVP path is
  single-rank (:doc:`field_derivatives`).
- **OMP** threads work *within* a volume: the threaded BLAS inside the
  dense LU, the OpenMP loops of matrix assembly and metric integration, the
  per-node FFTs of the matrix-free matvec, and the per-DOF loop of the
  Jacobian right-hand-side builds.

Measured behaviour (dense path, current baseline):

- **Ranks help up to ~2** with the standard :math:`2L` vacuum volume: rank 2
  moves the plasma volumes off the gating rank's critical path; rank 3
  merely relocates the cheap innermost volume while adding ~3.5 GB of
  per-rank memory. Even with **eight equal-size volumes**, rank scaling
  saturates at 4 ranks (~1.8× on the Jacobian, ~3.5× on the force solve
  against an ideal 8×), and 8 ranks is *slower* than 4 for the force solve
  — beyond load imbalance there is a communication/serial floor in the
  volume-parallel solve.
- **Threads saturate at ~4** on the benchmark machine: the force solve
  (LU-dominated) gains 2.2× from :math:`T = 1 \to 4`; the Jacobian gains
  only 1.2× — its many-right-hand-side triangular solves are
  memory-bandwidth-bound. 16 threads is consistently slower than 4 (E-core
  and thread-pool effects on hybrid CPUs).
- **Best measured split: 2 ranks × 4 threads.** At ``m13n11 L12`` this
  gives fjac 136 s / f 31 s versus 206 s / 73 s serial. At a fixed 8-core
  budget, (2 ranks × 4 threads) beats (1 × 8) by 27 % on the Jacobian, and
  using all 16 cores matches the 8-core optimum to within a few percent —
  **half the machine achieves the optimum**.

Practical rules:

- Keep :math:`R \times T` within the *physical performance* cores; do not
  count SMT or efficiency cores.
- Prefer fewer ranks when memory is tight (both memory-model terms grow
  with :math:`R`).
- The dense path's throughput *is* the BLAS/LAPACK: link a well-threaded
  implementation (the reference campaign used OpenBLAS) and make sure the
  BLAS thread pool and ``OMP_NUM_THREADS`` agree rather than multiply.
- Total available parallel speedup on a desktop-class node is modest
  (1.5× Jacobian, 2.4× force in the campaign) — **resolution increases are
  paid at nearly serial cost**, so the scaling laws, not the parallelism,
  decide what is affordable.

.. note::

   The pending threaded-Jacobian stack changes this picture: with the
   Jacobian right-hand-side builds threaded, the measured optimum moves to
   **1 rank × 8–12 threads** and adding MPI ranks *degrades* the threaded
   Jacobian. Re-measure the split after that stack lands.

The matrix-free path: when it wins and how to set it
====================================================

Scaling difference
------------------

Per volume, the dense path costs
:math:`\mathcal{O}(\texttt{nadof}^3)` time and
:math:`\mathcal{O}(\texttt{nadof}^2)` memory; the matrix-free path costs
:math:`k \times` (one matvec: FFTs,
:math:`\mathcal{O}(\texttt{iquad}\cdot N_{tz}\log N_{tz})`) time and
:math:`\mathcal{O}(\texttt{mn}\,L^2)` memory (preconditioner blocks), with
measured iteration counts :math:`k \sim` 40–90 that are largely
resolution-independent. The crossover on the benchmark machine is around
:math:`\texttt{nadof} \approx 5000` — the shipped ``nadofgmres`` default.
Below it, dense LU is faster and unconditionally robust; above it the gap
grows quickly:

.. list-table:: Measured at ``m21n17_L12`` (1 rank, 8 threads)
   :header-rows: 1
   :widths: 40 30 30

   * -
     - dense
     - matrix-free
   * - force evaluation
     - 516 s (extrapolated; exceeds memory)
     - **39.1 s**
   * - peak RSS
     - 69 GB (killed on a 62 GB node)
     - **15.0 GB**
   * - full Jacobian
     - not runnable at 1 rank
     - 79 min at 15.0 GB

Settings for performance
------------------------

- ``lbeltramisolver = 1`` with the default ``nadofgmres = 5000`` is the
  intended production setting: small volumes stay dense, large ones go
  matrix-free, the innermost volume stays dense
  (:ref:`why <sec-coordinate-singularity>`).
- **Tolerances are a direct cost lever**: GMRES iterations scale with
  :math:`\log(1/\varepsilon)`. The defaults (``epsgmres = 1e-12`` forward,
  ``epsgmres_jac = 1e-9`` for Jacobian/JVP columns) are chosen so solver
  error stays below the force-iteration noise floor; relaxing
  ``epsgmres_jac`` further is the first thing to try if Jacobian cost
  dominates, since the trust-region iteration tolerates inexact Jacobians
  far better than an inexact residual.
- ``ngmresrestart = 30`` bounds the Krylov memory
  (:math:`\mathcal{O}(\texttt{ngmresrestart}\cdot N)` per solve). Raising
  it helps volumes that converge slowly more than raising ``nitergmres``
  does, at proportional memory cost.
- Warm starts are automatic and free (11–23 % fewer iterations at typical
  optimizer steps, up to ~50 % for small steps); nothing to configure.
- The Beltrami factor cache (``lbeltrami_factor_cache``, switched on
  automatically by ``jvp_lag``) removes the per-batch factorization/bind
  cost of JVP batches at the price of keeping each volume's factors
  resident — a memory-for-time trade that pays off whenever more than one
  JVP batch runs per geometry.

Threads and ranks on the matrix-free path:

- The matvec is OMP-parallel over quadrature nodes (FFT-dominated), and the
  preconditioner build is OMP-parallel over modes; thread scaling of the
  iterative solve is therefore FFT/bandwidth-limited, similar in character
  to the dense force solve (saturating around the physical-core memory
  bandwidth), not LU-limited.
- MPI still distributes volumes; with the vacuum volume iterative, the
  gating-rank imbalance shrinks (its solve no longer costs 8× an interior
  volume), which *reduces* the payoff of adding ranks further.
- If the run will use the lagged-banded outer iteration, plan for **1 rank
  × many threads** from the start: ``jvp_lag`` requires a single rank.

Choosing a configuration by system size
=======================================

The table condenses the current measured guidance. Sizes are per-volume
``nadof`` (printed at startup as ``NAdof=…``; or compute
:math:`2\,\texttt{mn}\,L + \texttt{mn} + 1`).

.. list-table::
   :header-rows: 1
   :widths: 20 26 27 27

   * - Regime
     - Example
     - Backend
     - Ranks × threads
   * - Small (``nadof`` ≲ 5 k)
     - ``mpol/ntor`` ≲ 9/7, ``L`` = 12
     - dense everywhere (``lbeltramisolver = 0``)
     - 1 × 4 — overhead-dominated; parallelization buys little
   * - Medium (5–20 k)
     - ~13/11, ``L`` = 12
     - ``lbeltramisolver = 1`` (defaults); largest volumes go iterative
     - 2 × 4 (current baseline); 1 × 8 if memory-tight or heading for
       ``jvp_lag``
   * - Large (≳ 20 k)
     - ≳ 17/14, ``L`` = 12
     - matrix-free for all annular + vacuum volumes; innermost stays dense
     - 1 × 8 — memory, not time, is the binding constraint for dense here
   * - Optimization runs at scale
     - ≳ 13/11, many iterations
     - matrix-free + ``tr_solver = "banded"`` + ``jvp_lag``
     - 1 × (all physical cores) — ``jvp_lag`` is single-rank

Cost-efficiency notes
---------------------

- **Half the machine reaches the optimum** on desktop-class nodes: two
  cases side by side use the hardware better than one case on all cores.
- **Trade ranks for memory** when a case does not fit: dropping from 3
  ranks to 1 saved a third of peak memory at ``m13n11`` with limited time
  penalty, and the matrix-free backend saves far more.
- **Optimize the Jacobian first** (3.6× a force evaluation, once per
  iteration): relax ``epsgmres_jac``, enable the factor cache, and — for
  long optimizations — ``jvp_lag``, which replaces most dense Jacobian
  builds with ≤ 3 JVP directions each.
- **Check the vacuum volume's ``lrad``**: at :math:`2L` it is ~8× an
  interior volume and gates the whole solve on the dense path.
- Use ``estimate_cost.py`` (dense) before committing cluster time; there is
  no fitted matrix-free cost model yet (see below).

Planned revisions
=================

To state it explicitly: **the recommendations on this page are tied to the
current state of the code and will be revisited** once the following land,
at which point the campaign numbers should be re-measured:

1. **Threaded Jacobian assembly** (merge-request stack, measured
   2–2.75× on the Jacobian): shifts the optimal split to 1 rank × 8–12
   threads and makes MPI ranks counterproductive for the Jacobian.
2. **Matrix-free backend and lagged-banded outer iteration** (this
   section's own branch): moves the dense/matrix-free crossover into
   routine use, changes the memory model entirely for large volumes, and
   adds the single-rank ``jvp_lag`` mode to the configuration space. A
   fitted cost model for the matrix-free path (analogue of
   ``estimate_cost.py``) does not exist yet and should be added with the
   re-measurement.
3. **Axis-volume formulation work** (conformal interior mesh, research
   stage): would make the innermost volume iterative-eligible and remove
   the last dense factorization from large runs.

Until then, when this page and a measurement disagree, trust the
measurement — and record it in ``benchmarks``.
