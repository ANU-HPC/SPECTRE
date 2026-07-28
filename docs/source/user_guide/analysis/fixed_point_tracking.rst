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

A chain is named by the transform on it, ``iota = poloidal_occurrences /
repeat_period``, written in lowest terms:

* ``repeat_period`` is the denominator :math:`m` -- the number of islands in the
  chain and the fold of the Poincare return map.
* ``poloidal_occurrences`` is the numerator :math:`n`. For a stellarator with
  :math:`N_{fp}` field periods it must be a positive multiple of :math:`N_{fp}`,
  and coprime with ``repeat_period``.

For example, the 8/9 chain of a four-field-period device (:math:`N_{fp}=4`) has
``repeat_period = 9`` and ``poloidal_occurrences = 8`` (:math:`8 = 2 N_{fp}`).

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
    island_repeat_period = 9       # denominator of iota (island count)
    island_poloidal_occurrences = 8  # numerator of iota; pins the chain identity
    island_discover = true         # self-seed by a fixed-point scan
    max_axis_repeats = 8           # random-restart budget for the axis finder

Instead of ``island_discover``, an explicit seed may be given as
``island_guess_rz = [R, Z]`` (the :math:`(R, Z)` of the tracked point at
:math:`\phi = 0` on the first iterate). Setting ``track_axis = true`` alone,
with no island keys, runs **axis-only** mode.

.. note::

   With ``island_discover`` it is strongly recommended to also set
   ``island_poloidal_occurrences``. Discovery locks onto the first resonant
   chain it finds; giving the numerator pins the intended chain and avoids
   locking onto a neighbouring resonance.

The run then writes, after every accepted iterate, the ``/iterations/fp`` group
into the HDF5 output (both the intermediate ``<stem>.h5`` and the final
``<stem>_opt.h5``). Rewriting every step keeps the file current if the job is
cancelled.

Reading the output
------------------

.. code-block:: python

    import h5py
    import numpy as np

    with h5py.File("<stem>_opt.h5", "r") as f:
        fp = f["iterations/fp"]
        axis_rz = fp["axis_rz"][:]        # (n_iter, 2) magnetic axis (R, Z)
        island_rz = fp["island_rz"][:]    # (n_iter, 2) tracked O-point (R, Z)
        residue = fp["residue"][:]        # (n_iter,) Greene residue
        kind = fp["kind"][:]              # (n_iter,) b"o" / b"x"
        iota = fp["iota"][:]              # (n_iter,) empirical transform

Per-step arrays (first axis is the iteration) include ``axis_rz``,
``island_rz``, ``island_rz_half_period``, ``residual``, ``iota``, ``iota_rel``,
``residue``, ``greene_trace``, ``greene_det``, ``greene_disc``, ``kind``,
``provenance`` and ``no_fp``. Scalar metadata is stored as attributes on the
group: ``repeat_period``, ``poloidal_occurrences``, ``volume_start``, ``nfp``
and the acceptance tolerance ``root_ftol``. Rows for iterations where no fixed
point was found are left NaN with ``no_fp = True``.

A worked example, plotting how the axis and an 8/9 chain evolve over an
optimization, ships in ``examples/analysis/finding_fixed_points/``.

Offline tracking
================

To analyse a finished (or running) minimization that saved ``xs_fixb/``
checkpoints, replay them with ``track_fp.py``:

.. code-block:: bash

    python scripts/minimization_monitoring/track_fp.py case.toml 9 8 \
        --data-files xs_fixb/*.npy \
        --poloidal-occurrences 8 \
        --discover \
        --output fp_metrics.npz

The positional arguments are the input TOML, the ``repeat_period`` and the
``volume``. Each checkpoint is re-solved and stepped through the same tracker,
and the per-step metrics are written to the ``--output`` ``.npz``. This needs
only the saved checkpoints -- no re-optimization -- so it can run alongside a
live minimization.

Model parameters
================

:class:`~spectre.fixed_points.FixedPointTracker` carries a set of tuned
parameters, grouped by what fixes their value:

* **Numerical tolerances** (integrator and root-finder settings) are
  device-independent and fixed.
* **Search resolution** (the number of radial and poloidal samples used to
  locate candidates) is set by the width of the resonant dip in radius; a
  narrower resonance needs finer sampling.
* **Absolute length scales** (the acceptance residual, the deduplication
  distance and the step-continuity limits, all in metres) are tuned for a
  major-radius :math:`\sim 20\,\mathrm{m}` device and should be scaled with the
  machine size.

The search-resolution and length parameters are exposed as keyword arguments of
:class:`~spectre.fixed_points.FixedPointTracker` so a different device can retune
them without editing the source; the defaults reproduce the reference case.
