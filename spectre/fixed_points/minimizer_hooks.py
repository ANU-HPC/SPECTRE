# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT
"""``Force_Minimizer``'s fixed-point tracking hooks.

Everything the minimizer needs for in-loop axis/island tracking lives here, so
``spectre.force_minimizer`` stays focused on the minimization itself: building the tracker from
the ``[minimization]`` parameters, stepping it on each accepted iterate, and writing its results
into the output HDF5.
"""

import numpy as np

from spectre.fixed_points.tracker import FixedPointTracker


def initialize_fp_tracker(min_params):
    """Build the in-loop ``FixedPointTracker`` from the ``[minimization]`` parameters.

    Off by default (zero overhead): returns ``None`` unless ``track_axis`` or ``track_islands``
    is set. ``track_islands`` configurations are validated at TOML load time
    (``MinimizationParameters``), so the island keys are complete here.

    Args:
        min_params: The ``MinimizationParameters`` of the run.

    Returns:
        FixedPointTracker | None: One stateful tracker reused across the whole run, or ``None``
        when tracking is off.
    """
    mp = min_params
    if not (mp.track_axis or mp.track_islands):
        return None

    island_mode = mp.track_islands
    tracker = FixedPointTracker(
        m=mp.island_m if island_mode else None,
        volume=mp.island_volume,
        n_steps=mp.max_niter + 1,
        guess=mp.island_guess_rz if island_mode else None,
        # no island_guess_rz -> self-seed by discovery (the tracker applies the same default)
        discover=(mp.island_guess_rz is None) if island_mode else False,
        n=mp.island_n if island_mode else None,
    )
    if island_mode:
        seed = (
            "self-discover"
            if mp.island_guess_rz is None
            else f"guess (R,Z)={tuple(mp.island_guess_rz)}"
        )
        print(
            f"In-loop FP tracking ON: period-{mp.island_m} chain in "
            f"volume {mp.island_volume}, {seed} -> h5:/iterations/fp",
            flush=True,
        )
        if mp.island_guess_rz is None and mp.island_n is None:
            print(
                "  ** WARNING: discovery without island_n "
                "- the first cold lock pins the chain; set "
                "island_n to avoid a wrong-chain lock.",
                flush=True,
            )
    else:
        print("In-loop FP tracking ON: axis-only mode -> h5:/iterations/fp", flush=True)
    return tracker


def track_fixed_points(it, obj):
    """Record axis + island fixed-point metrics for the accepted iterate.

    Runs the shared ``FixedPointTracker`` on the live, already-solved in-memory field (the analytic
    Jacobian just solved it at this accepted ``x``), so ``create_spectreout()`` is a free wrapper -
    no disk load, no re-solve. Rank-0 only (the field-line tracing is single-process; workers are
    parked) and fully guarded: any failure is logged and the optimisation continues untouched.
    Results stay in the tracker's in-memory arrays; :func:`write_fp_group` stores them once at
    postprocess time.

    Args:
        it: The optimizer iteration, used only to label a skipped step.
        obj: The ``Force_Minimizer`` instance carrying ``fp_tracker`` and the live field.
    """
    tracker = getattr(obj, "fp_tracker", None)
    if tracker is None or obj.rank != 0:
        return
    try:
        specout = obj.field_obj.create_spectreout()
        report = tracker.step(specout)
        tracker.log(report)  # one terse fp: line to stdout (-> opt.txt)
    except Exception as exc:
        print(f"[fp-track] iter {it} skipped - {exc}", flush=True)


def write_fp_group(tracker, grp, iteration_start):
    """Write the tracker's results as an ``fp`` group under ``grp`` (``/iterations`` in the h5).

    Scalars go to group attributes; unicode arrays (kind, provenance) are stored as bytes.

    Args:
        tracker: The run's ``FixedPointTracker`` (``n_done > 0``).
        grp: The open ``/iterations`` ``h5py.Group`` to write into.
        iteration_start: First global optimizer iteration of this run (labels the rows).
    """
    res = tracker.result_dict(
        iteration_start + np.arange(tracker.n_done), iteration_start, 1, False
    )
    fp_grp = grp.create_group("fp")
    for name, data in res.items():
        if np.ndim(data) == 0:
            fp_grp.attrs[name] = data
        else:
            arr = np.asarray(data)
            fp_grp.create_dataset(name, data=arr.astype("S") if arr.dtype.kind == "U" else arr)
