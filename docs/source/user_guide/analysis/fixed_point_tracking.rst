#####################
Fixed-point tracking
#####################

A magnetic island appears where the rotational transform passes through a
rational value :math:`\iota = n/m`. On a Poincare section the island chain shows
up as a ring of :math:`m` fixed points of the :math:`m`-fold return map,
alternating **O-points** (elliptic, the island centres) and **X-points**
(hyperbolic, the separatrix crossings). The magnetic axis is the degenerate
fixed point at the core.

SPECTRE can **locate** these fixed points in a single equilibrium and **track**
them as the equilibrium changes across a force-minimization run, recording how
the axis and one island chain move and how the island's stability evolves. This
is useful for watching an island open or heal during an optimization, or for
measuring an island's width and O/X character at convergence.

The machinery lives in :mod:`spectre.fixed_points`. It is driven in two ways:

* **In-loop**, from :class:`~spectre.force_minimizer.Force_Minimizer` during a
  minimization, configured entirely from the input TOML. Results are written to
  the ``/iterations/fp`` group of the run's HDF5 output.
* **Offline**, with ``scripts/minimization_monitoring/track_fp.py``, which
  replays the ``xs_fixb/`` checkpoints a minimization saved and writes an
  ``.npz`` of the same per-step metrics.

Both use the same stepper, :class:`~spectre.fixed_points.FixedPointTracker`.

Conventions
===========

Resonance
---------

A chain is named by the transform on it, ``iota = n/m``:

* ``m`` is the denominator -- the number of islands in the chain and the fold
  of the Poincare return map.
* ``n`` is the numerator. For a stellarator with :math:`N_{fp}` field periods
  it must be a positive multiple of :math:`N_{fp}`, and is usually coprime with
  ``m`` (the name in lowest terms). Coprimality is not fundamentally required:
  a non-coprime pair (e.g. 5/5) simply views a shorter chain through a
  longer-period map, and is accepted when ``n`` is given explicitly.

For example, the 8/9 chain of a four-field-period device (:math:`N_{fp}=4`) has
``m = 9`` and ``n = 8`` (:math:`8 = 2 N_{fp}`).

O/X classification
------------------

A located fixed point is classified from the trace of the return-map tangent
map :math:`J` via Greene's residue :math:`R = (2 - \mathrm{tr}\,J)/4` and the
discriminant :math:`\mathrm{disc} = \mathrm{tr}^2 J - 4\,\det J`:

* :math:`\mathrm{disc} < 0` -- **O-point** (elliptic), :math:`0 < R < 1`.
* :math:`\mathrm{disc} > 0` -- **X-point** (hyperbolic), :math:`R < 0` or
  :math:`R > 1`.

The tangent map is obtained analytically by integrating the variational
equation :math:`\mathrm{d}J/\mathrm{d}\phi = A(\phi)\,J` alongside the field
line, which resolves the O/X sign even for near-parabolic chains. Only
stellarator-symmetric equilibria are supported.

In-loop tracking
================

Add the tracking keys to the ``[minimization]`` table of the input TOML. All
default off, so tracking has no effect unless you switch it on.

.. code-block:: toml

    [minimization]
    # ... the usual minimization keys ...
    track_axis = true              # follow the magnetic axis
    track_islands = true           # follow one island chain
    island_volume = 8              # volume hosting the chain (0 = axis volume)
    island_m = 9                   # denominator of iota (island count)
    island_n = 8                   # numerator of iota; pins the chain identity

With no explicit seed the chain is self-discovered by a fixed-point scan.
Alternatively, give the seed as ``island_guess_rz = [R, Z]`` (the
:math:`(R, Z)` of the tracked point at :math:`\phi = 0` on the first iterate).
Setting ``track_axis = true`` alone, with no island keys, runs **axis-only**
mode. The ``island_*`` keys default to ``None`` and mean anything only when
``track_islands`` is on -- and are then required: an incomplete or invalid
island configuration is rejected when the TOML is loaded, so a run can never
silently degrade to axis-only tracking.

.. note::

   When discovering (no ``island_guess_rz``) it is strongly recommended to set
   ``island_n``. Discovery locks onto the first resonant chain it finds; giving
   the numerator pins the intended chain and avoids locking onto a neighbouring
   resonance.

During the iterations the results accumulate in memory only. The
``/iterations/fp`` group is written once, at postprocess time, into the final
``<stem>_opt.h5`` (nothing is written to the intermediate ``<stem>.h5``).

Reading the output
------------------

.. code-block:: python

    from spectre import SPECTREout

    fp = SPECTREout("<stem>_opt.h5").iterations.fp
    fp.axis_rz          # (n_iter, 2) magnetic axis (R, Z)
    fp.island_rz        # (n_iter, 2) committed fixed point (R, Z)
    fp.greene_residue   # (n_iter,) Greene residue of the committed point
    fp.kind             # (n_iter,) b"o" / b"x"
    fp.iota             # (n_iter,) empirical transform
    fp.displacement     # (n_iter,) (R, Z) move per anchor-follow step [m]

Per-step arrays (first axis is the iteration) include ``axis_rz``,
``island_rz``, ``island_rz_half_period``, ``residual``, ``iota``, ``iota_rel``,
``greene_residue``, ``greene_trace``, ``greene_det``, ``greene_disc``, ``kind``,
``provenance``, ``displacement``, ``greene_residue_delta`` and ``no_fp``. The
two deltas are the step convergence metrics and are finite only on anchor-follow
steps (a cold re-acquire may land on a different chain member). Scalar metadata
is stored as attributes on the group: ``island_m``, ``island_n``,
``volume_start``, ``nfp`` and the acceptance tolerance ``root_ftol``
(attributes are not mapped by ``SPECTREout``; read them with ``h5py`` if
needed). Rows for iterations where no fixed point was found are left NaN with
``no_fp = True``.

The tracker prefers an O-point: a follow step that lands on an X is treated as a
miss and falls back to discovery, and discovery scans for an O-point first. When
no O-point survives the scan it commits the nearest remaining member instead, so
``island_rz`` and ``greene_residue`` describe whichever point was committed on
that iterate, and ``kind`` (``b"o"`` / ``b"x"``) records which it was.

A worked example, plotting how the axis and an 8/9 chain evolve over an
optimization, ships in ``examples/analysis/finding_fixed_points/``.

Offline tracking
================

To analyse a finished (or running) minimization that saved ``xs_fixb/``
checkpoints, replay them with ``track_fp.py``:

.. code-block:: bash

    python scripts/minimization_monitoring/track_fp.py case.toml \
        -v 8 -m 9 -n 8 \
        --data-files xs_fixb/*.npy \
        --output fp_metrics.npz

The only positional argument is the input TOML; ``-v`` (long form
``--island-volume``), ``-m`` (``--island-m``) and ``-n`` (``--island-n``) name
the volume and the chain, and with no ``--guess`` the tracker self-seeds by
discovery. Each checkpoint is re-solved and stepped through the same tracker,
and the per-step metrics are written to the ``--output`` ``.npz``. This needs
only the saved checkpoints -- no re-optimization -- so it can run alongside a
live minimization.

Interactive use
===============

Both finders are also methods of :class:`~spectre.postprocess.main.SPECTREout`, so a
single equilibrium can be inspected without any tracking setup:

.. code-block:: python

    from spectre import SPECTREout

    out = SPECTREout("case.h5")
    rax, zax, sax, thax = out.find_axis(guess=[1.4, 0.0])

    # Discover the fixed points of the m-fold map in a volume (no seed needed) ...
    members = out.find_fixed_point(2, 11, return_all=True)
    # ... or polish a single point from an (R, Z) seed.
    fp = out.find_fixed_point(2, 11, guess_rz=(members[0].R, members[0].Z))
    print(fp.R, fp.Z, fp.residual, fp.converged)

``find_fixed_point(volume, m, ...)`` mirrors the TOML behaviour: with no ``guess_rz``
it discovers by a fixed-point scan; results are
:class:`~spectre.fixed_points.FixedPointResult` objects and are cached in
``out.fixed_points[(volume, m)]``.

Model parameters
================

:class:`~spectre.fixed_points.FixedPointTracker` carries a set of tuned
parameters, grouped by what fixes their value:

* **Numerical tolerances** (integrator and root-finder settings) are
  device-independent and fixed.
* **Search resolution** (the number of radial and poloidal samples used to
  locate candidates) only has to seed the Newton polish: a sample must land
  inside a fixed point's basin of attraction, which is much wider than the
  island, so the scan never has to resolve the resonance itself. Changing it
  changes which candidates are seeded, and so the census of discovered members.
* **Absolute length scales** (the acceptance residual, the deduplication
  distance and the step-continuity limits, all in metres) are tuned for a
  major-radius :math:`\sim 20\,\mathrm{m}` device and should be scaled with the
  machine size.

The length scales, together with the two scan sample counts ``disc_n_s`` and
``n_s``, are exposed as keyword arguments of
:class:`~spectre.fixed_points.FixedPointTracker` so a different device can retune
them without editing the source; the defaults reproduce the reference case. The
remaining parameters -- the follow-box geometry, the drift and window factors and
the iota-mismatch warning threshold -- are module-level constants of
:mod:`spectre.fixed_points.tracker` and require a source edit to change.
