#######################
Minimization parameters
#######################

The equilibrium solve is driven by the ``[minimization]`` table of the input
TOML. These parameters are parsed into
:class:`~spectre.file_io.input_parameters.MinimizationParameters` and consumed by
``Force_Minimizer`` and the driver ``scripts/run/calc_spectre_eq.py``.

Solver
======

``method`` : ``"trf"`` | ``"dogbox"`` (default ``"dogbox"``)
   Least-squares method. ``trf`` (trust-region reflective) is the robust default
   for the free-boundary inner solve; ``dogbox`` is also available.

``gtol``, ``xtol``, ``ftol`` : float
   Convergence tolerances on the gradient, step, and cost reduction.

``max_nfev`` : int (default ``10000``)
   Cap on objective evaluations within a single least-squares call.

``max_niter`` : int (default ``10000``)
   Total inner-iteration budget across all outer cycles.

``rcond_lstsq`` : float | null
   Cutoff for small singular values in the Gauss-Newton solve (null = library
   default).

Representation
==============

``inface_repr`` : ``"fourier"`` | ``"fourier red"`` | ``"henn"`` (default ``"fourier"``)
   Interface degree-of-freedom representation: raw Fourier DOFs, truncated to
   ``mpol_rz``/``ntor_rz`` (``fourier red``), or a Henneberg parametrization
   sized by ``mmax``/``nmax`` (``henn``).

``mmax``, ``nmax`` : int (default ``8``)
   Poloidal/toroidal mode caps for the ``henn`` representation.

Objective
=========

``objective`` : ``"freal"`` | ``"freal,len"`` (default ``"freal"``)
   Residual definition. ``freal`` is the real-space force imbalance;
   ``freal,len`` adds a per-interface length constraint weighted by
   ``len_factor``.

``mpol_rz``, ``ntor_rz`` : int | null
   Poloidal/toroidal truncation of the geometry DOFs (null = use the TOML
   mpol/ntor).

``len_factor`` : float (default ``1.0``)
   Weight of the length-constraint block in the ``freal,len`` objective.

Free-boundary / Bns
===================

``initialize_bns_from_current`` : bool (default ``true``)
   If true and ``lfreebound = 1``, ``Force_Minimizer`` runs
   ``SPECTRE.update_bns()`` (dtflux(Mvol) Newton + one iBns refresh) on the
   constructed object. No-op for fixed-boundary cases.

``freeb_bnorm_tol`` : float (default ``1e-5``)
   Outer-loop convergence tolerance on ``bnserr`` (the iBns/ofmn mismatch).

``gbnbld`` : float | null (default ``null``)
   iBns relaxation blend factor. When set, it is copied into the Fortran
   ``global_`` input so the free-boundary iteration and ``refresh_bns`` use it:
   ``iBns := gbnbld·iBns + (1 - gbnbld)·ofmn``. ``null`` leaves the ``global_``
   value (whose own default is ``0.666``) untouched. 0.5–0.7 is robust; ``1.0``
   freezes iBns, ``0.0`` is full replacement.

Free-boundary driver
====================

These knobs are read by ``scripts/run/calc_spectre_eq.py``.

``bns_trigger`` : ``"outer"`` | ``"inloop"`` (default ``"outer"``)
   When to refresh iBns. ``outer``: freeze iBns, run an inner force solve to
   convergence/stall, refresh once, repeat. ``inloop`` (prototype): a single
   solve that refreshes mid-run when the step norm drops below ``step_tol``.

``inner_iter_cap`` : int (default ``5``)
   Max inner iterations per outer cycle before forcing a refresh (``outer``
   trigger only).

``step_tol``, ``cost_red_frac`` : float
   An inner solve is "stalled" (triggering a refresh) when the accepted step
   norm is below ``step_tol`` and the cost reduction is below ``cost_red_frac``
   of the current cost.

``loss`` : ``"linear"`` | ``"soft_l1"`` (default ``"linear"``)
   Inner least-squares loss; ``f_scale`` is the soft_l1 transition scale.

``f_scale`` : float (default ``1.0``)
   Soft-loss transition scale (only used for ``loss = "soft_l1"``).

``x_scale``, ``x_scale2`` : float | ``"jac"`` | null
   Variable scaling for the inner solve. Two-stage: outer cycles at/after
   ``switch_iter`` use ``x_scale2`` instead of ``x_scale`` when it is set.

``switch_iter`` : int (default ``10``)
   Outer-cycle index at which ``x_scale2`` takes over.

IO
==

``save_xin_fixb`` : bool (default ``true``)
   Save per-iteration interface snapshots to ``xs_fixb/``.

``nt_dbb``, ``nz_dbb`` : int | null
   Real-space (theta, zeta) grid for the force evaluation (null = derive from
   the resolution).

``filter_stellsym`` : bool (default ``true``)
   Keep only the unique stellarator-symmetric grid points in the residual.

``resume`` : bool (default ``false``)
   Resume an interrupted run from the latest ``xs_fixb/`` snapshot.
