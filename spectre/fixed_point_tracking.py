# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT
"""Track the magnetic axis and one island fixed point across SPECTRE optimisation checkpoints.

For each ``xs_fixb/x_<a>_<b>.npy`` checkpoint the equilibrium is rebuilt and the ``Z=0`` line of
the island volume is scanned for return-map residual minima (no root-finder, so nothing stalls).
Under stellarator symmetry ``Z=0`` corresponds to ``theta in {0, pi}`` (``theta=0`` outboard,
``theta=pi`` inboard at ``phi=0``), so only those two poloidal angles are scanned in
``(s, theta)``. The candidate nearest in ``(R, Z)`` to the previously tracked point, within
``r_window`` in ``R`` and ``s_window`` in ``s``, is followed; its per-step displacement is the
convergence metric. The tracker stays on one side (``theta=0`` or ``pi``, locked from the first
selection) and in one volume.

Results are returned as a dict of per-checkpoint arrays and, optionally, saved as a ``.npz`` after
every checkpoint; :func:`track_islands_and_axis` documents the full key list.
"""

from __future__ import annotations

import re
import signal
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter

import numpy as np

from spectre.fixed_point import _return_map

_XS_FIXB_RE = re.compile(
    r"^x_(\d+)_(\d+)\.npy$"
)  # matches x_<a>_<b>.npy; b is the iteration number
_IOTA_REL_TOL = (
    0.01  # if iota is within 1% of the nearest k*Nfp/mpol resonance, it is not flagged as far
)
_S_CLAMP = 0.999  # hard clamp on the seed s-range (s in (-1, 1))
_THETA_SIDES = (
    0.0,
    float(np.pi),
)  # the two Z=0 poloidal angles (phi=0): theta=0 outboard, theta=pi inboard


def track_islands_and_axis(
    xs_fixb_dir: str | Path,
    toml_path: str | Path,
    mpol: int,
    volume: int,
    *,
    stride: int = 1,
    start_at_iter: int = 0,
    n_z0_points: int = 1000,
    guess: Sequence[float] | None = None,
    r_window: float = 0.1,
    s_window: float = 0.3,
    max_axis_repeats: int = 8,
    verbose: bool = False,
    checkpoint_path: str | Path | None = None,
) -> dict:
    """Track the magnetic axis and one island fixed point across optimisation checkpoints.

    Stays on one side (``theta=0`` outboard / ``theta=pi`` inboard, locked from ``guess``) and in
    one volume, selecting the candidate nearest in ``(R, Z)`` to the previous point within
    ``r_window`` (in ``R``) and ``s_window`` (in the flux label ``s``).

    Args:
        xs_fixb_dir: Folder of ``x_<a>_<b>.npy`` checkpoints (``b`` is the iteration number).
        toml_path: SPECTRE TOML matching the checkpoints.
        mpol: Island count = denominator of ``iota = ntor/mpol``.
        volume: Volume index hosting the chain (``>= 1``).
        stride: Process every ``stride``-th checkpoint.
        start_at_iter: Skip checkpoints with iteration ``< start_at_iter``.
        n_z0_points: Seeds per ``theta`` side per scan.
        guess: ``(R, Z)`` at ``phi=0`` selecting which fixed point / side to lock on the first
            checkpoint. ``None`` selects the most-fixed candidate.
        r_window: Candidate continuity window in ``R`` [m].
        s_window: Candidate continuity window in the flux label ``s``.
        max_axis_repeats: Random-restart budget for the axis finder.
        verbose: Print a per-checkpoint table.
        checkpoint_path: If given, re-save the ``.npz`` after every checkpoint (send ``SIGUSR1`` to
            request a clean stop after the current checkpoint).

    Returns:
        A dict with one entry per output. Per-checkpoint arrays (length = number of processed
        checkpoints) and scalar metadata:

        ``iteration_indices`` (int), ``axis_rz`` ``(n, 2)``, ``island_rz`` ``(n, 2)``,
        ``island_sth`` ``(n, 2)``, ``residual`` (return-map norm), ``iota``, ``ntor`` (int),
        ``iota_far`` (bool), ``no_fp`` (bool), ``leaving`` (``''``/``'inner'``/``'outer'``),
        ``displacement`` (``||step||`` [m] vs the previous located point), ``n_candidates`` (int),
        ``best_residual``, ``select_dist``, ``t_total``, ``t_field_reconstruction``,
        ``t_fieldline_tracing``, ``mpol``, ``volume``, ``nfp``, ``start_at_iter``, ``n_z0_points``,
        ``r_window``, ``s_window``.
    """
    # SPECTRE imports are deferred to call time (heavy core; keeps module import cheap).
    from spectre import SPECTRE
    from spectre.fixed_point import find_axis
    from spectre.force_targets import force_real
    from spectre.utils import get_xinit_specwrap, unpack_interfaces

    xs_fixb_dir = Path(xs_fixb_dir)
    toml_path = Path(toml_path)
    if not xs_fixb_dir.is_dir():
        raise FileNotFoundError(f"xs_fixb directory not found: {xs_fixb_dir}")
    if not toml_path.is_file():
        raise FileNotFoundError(f"TOML file not found: {toml_path}")
    checkpoint_path = None if checkpoint_path is None else Path(checkpoint_path)

    mpol, volume, stride = int(mpol), int(volume), max(1, int(stride))

    # -- List every x_***_***.npy checkpoint, sorted by iteration (group 2 = iter) ----------
    all_paths: list[tuple[int, Path]] = []
    for p in xs_fixb_dir.iterdir():
        m = _XS_FIXB_RE.match(p.name)
        if m:
            all_paths.append((int(m.group(2)), p))
    all_paths.sort(key=lambda t: t[0])
    if not all_paths:
        raise FileNotFoundError(f"No x_***_***.npy files found in {xs_fixb_dir}")

    paths = [(it, p) for (it, p) in all_paths if it >= int(start_at_iter)][::stride]
    if not paths:
        raise ValueError(
            f"No checkpoints with iter >= start_at_iter={start_at_iter} "
            f"(available iters {all_paths[0][0]}..{all_paths[-1][0]})"
        )
    n_iter = len(paths)

    # -- Pre-allocate outputs ----------------------------------------------------------------
    iter_indices = np.array([it for it, _ in paths], dtype=int)
    axis_rz = np.full((n_iter, 2), np.nan)
    island_rz = np.full((n_iter, 2), np.nan)
    island_sth = np.full((n_iter, 2), np.nan)
    residual = np.full(n_iter, np.nan)
    iota = np.full(n_iter, np.nan)
    ntor = np.zeros(n_iter, dtype=int)
    iota_far = np.zeros(n_iter, dtype=bool)
    no_fp = np.zeros(n_iter, dtype=bool)
    leaving = np.full(n_iter, "", dtype="<U6")  # ''/'inner'/'outer'
    displacement = np.full(n_iter, np.nan)  # ||step|| [m] vs previous located point
    # -- Confirmation diagnostics ------------------------------------------------------------
    n_candidates = np.zeros(n_iter, dtype=int)  # fixed-point candidates in the window
    best_residual = np.full(n_iter, np.nan)  # min residual among candidates
    select_dist = np.full(n_iter, np.nan)  # (R,Z) dist from ref to the selected one
    # -- Timing breakdown --------------------------------------------------------------------
    t_total = np.full(n_iter, np.nan)
    t_field_reconstruction = np.full(n_iter, np.nan)
    t_fieldline_tracing = np.full(n_iter, np.nan)

    guess_rz = None if guess is None else np.array([float(guess[0]), float(guess[1])])
    nfp: int | None = None

    print(
        f"track_islands_and_axis: {n_iter} checkpoints "
        f"(iters {iter_indices[0]}..{iter_indices[-1]}, stride={stride}, "
        f"start_at_iter={start_at_iter}, mpol={mpol}, volume={volume}, n_z0={n_z0_points}, "
        f"r_window={r_window}, s_window={s_window}, "
        f"guess={None if guess is None else list(guess_rz)})"
    )

    # _make_outputs / _save are reused (every-checkpoint save + final return), so kept as
    # closures over the arrays above rather than re-listing every key at each call site.
    def _make_outputs() -> dict:
        return {
            "iteration_indices": iter_indices,
            "axis_rz": axis_rz,
            "island_rz": island_rz,
            "island_sth": island_sth,
            "residual": residual,
            "iota": iota,
            "ntor": ntor,
            "iota_far": iota_far,
            "no_fp": no_fp,
            "leaving": leaving,
            "displacement": displacement,
            "n_candidates": n_candidates,
            "best_residual": best_residual,
            "select_dist": select_dist,
            "t_total": t_total,
            "t_field_reconstruction": t_field_reconstruction,
            "t_fieldline_tracing": t_fieldline_tracing,
            "mpol": mpol,
            "volume": volume,
            "nfp": -1 if nfp is None else nfp,
            "start_at_iter": int(start_at_iter),
            "n_z0_points": int(n_z0_points),
            "r_window": float(r_window),
            "s_window": float(s_window),
        }

    def _save(path: Path) -> None:
        np.savez(str(path), **_make_outputs())

    # -- SIGUSR1 -> clean stop ---------------------------------------------------------------
    stop_flag = [False]

    def _on_sigusr1(signum, frame):
        stop_flag[0] = True
        print(
            "\n  [SIGUSR1] Stop requested - finishing current checkpoint then saving.", flush=True
        )

    old_handler = signal.signal(signal.SIGUSR1, _on_sigusr1)

    obj = SPECTRE(str(toml_path), verbose=False)

    # Carry-state for continuity:
    locked_theta: float | None = None  # side (0 or pi), locked from the first selection
    anchor_rz: np.ndarray | None = None  # (R,Z) of last located point
    anchor_s: float | None = None  # s of last located point (s-continuity gate)
    prev_located_rz: np.ndarray | None = None  # for the displacement metric

    if verbose:
        print(
            "\n  iter | axis (R,Z)          | s       | th |  island (R,Z)        | residual | "
            "  iota   | ntor | disp[mm] | t tot/rec/tra [s]   | note"
        )
        print("  " + "-" * 118)

    try:
        for i, (it_idx, npy_path) in enumerate(paths):
            if stop_flag[0]:
                print(f"  Stopping before iter {it_idx} (SIGUSR1).", flush=True)
                break

            # -- Rebuild the equilibrium from the checkpoint: npy -> unpack -> force_real --------
            t0 = perf_counter()
            try:
                x = np.load(str(npy_path))
                unpack_interfaces(x, obj)
                xin = get_xinit_specwrap(obj)
                force_real(xin, obj)
                specout = obj.create_spectreout()
            except Exception as exc:
                print(f"  WARNING iter {it_idx}: checkpoint load failed - {exc}")
                continue
            t_field_reconstruction[i] = perf_counter() - t0

            if nfp is None:
                nfp = int(specout.input.physics.Nfp)

            # -- Magnetic axis (volume 0; failure leaves the row NaN but does not abort) --------
            try:
                rax, zax, _, _ = find_axis(specout, max_repeats=max_axis_repeats, verbose=False)
                axis_rz[i] = [rax, zax]
            except Exception as exc:
                print(f"  WARNING iter {it_idx}: axis finding failed - {exc}")

            # -- Scan the Z=0 line for return-map residual minima (the core; no solver) ---------
            # Scan both theta sides until one is locked, then only the locked side. On each
            # theta the residual ||return map|| is a 1-D function of s whose interior local minima
            # are candidate fixed points; ``end_pinned[theta]`` flags a minimum stuck at a scan
            # endpoint (leaving). trace_one_fieldline records phi=0 at every transit including t=0,
            # so the mpol-fold return map needs num_transits = mpol+1 (the last puncture is then
            # after mpol transits).
            thetas = _THETA_SIDES if locked_theta is None else (locked_theta,)
            s_grid = np.linspace(-_S_CLAMP, _S_CLAMP, int(n_z0_points))

            t_scan = perf_counter()
            try:
                cands: list[dict] = []
                end_pinned: dict[float, str] = {}
                for theta in thetas:
                    res = np.array(
                        [
                            np.hypot(*_return_map([s, theta], specout, volume, mpol + 1))
                            for s in s_grid
                        ]
                    )
                    # A residual minimised at an endpoint (s -> -_S_CLAMP inner /
                    # s -> +_S_CLAMP outer) means the fixed point has slid past the interface and
                    # is leaving the volume; the interior local-min search below deliberately
                    # ignores the endpoints.
                    amin = int(np.argmin(res))
                    end_pinned[theta] = (
                        "inner" if amin == 0 else ("outer" if amin == len(res) - 1 else "")
                    )
                    for k in range(1, len(s_grid) - 1):
                        if res[k] < res[k - 1] and res[k] < res[k + 1]:
                            s_best, r_best = s_grid[k], res[k]
                            # Parabolic sub-grid refinement (uniform spacing; still no solver).
                            denom = res[k - 1] - 2.0 * res[k] + res[k + 1]
                            if denom > 0.0:
                                h = s_grid[k] - s_grid[k - 1]
                                s_v = s_grid[k] + 0.5 * (res[k - 1] - res[k + 1]) / denom * h
                                s_v = min(max(s_v, s_grid[k - 1]), s_grid[k + 1])
                                r_v = np.hypot(
                                    *_return_map([s_v, theta], specout, volume, mpol + 1)
                                )
                                if r_v < r_best:
                                    s_best, r_best = s_v, r_v
                            r_c, z_c = specout.get_rz_coords(volume, s_best, theta, 0.0)
                            cands.append(
                                {
                                    "s": s_best,
                                    "theta": theta,
                                    "R": r_c,
                                    "Z": z_c,
                                    "residual": r_best,
                                }
                            )
            except Exception as exc:
                print(f"  WARNING iter {it_idx}: Z=0 scan failed - {exc}")
                t_total[i] = perf_counter() - t0
                if checkpoint_path is not None:
                    _save(checkpoint_path)
                continue
            t_fieldline_tracing[i] = perf_counter() - t_scan

            # -- Build the candidate pool, then select within it --------------------------------
            # Reference = last anchor, else the first-checkpoint guess. The pool is the set the
            # selector considers and is also the scan-health diagnostic
            # (n_candidates, best_residual): all candidates when there is no reference yet, else
            # those within both continuity
            # windows -- |R-R_prev| <= r_window (no jump across the axis) and |s-s_prev| <= s_window
            # (no teleport along the flux label onto a shallow feature).
            ref = anchor_rz if anchor_rz is not None else guess_rz
            if not cands:
                pool = []
            elif ref is None:
                pool = cands
            else:
                pool = [
                    c
                    for c in cands
                    if abs(c["R"] - ref[0]) <= r_window
                    and (anchor_s is None or abs(c["s"] - anchor_s) <= s_window)
                ]
            n_candidates[i] = len(pool)
            if pool:
                best_residual[i] = min(c["residual"] for c in pool)

            # Select: most-fixed when there is no reference yet, else nearest in (R,Z).
            if not pool:
                sel = None
            elif ref is None:
                sel = min(pool, key=lambda c: c["residual"])
            else:
                sel = min(pool, key=lambda c: np.hypot(c["R"] - ref[0], c["Z"] - ref[1]))

            note = ""
            if sel is None:
                # Nothing in range: distinguish a chain that left the volume (residual pinned
                # past an interface on the locked side) from a genuine no-fixed-point row.
                no_fp[i] = True
                side = end_pinned.get(locked_theta, "") if locked_theta is not None else ""
                leaving[i] = side
                note = f"LEAVING_TO_{side.upper()}" if side else "NO_FP"
            else:
                if ref is not None:
                    select_dist[i] = np.hypot(sel["R"] - ref[0], sel["Z"] - ref[1])
                island_rz[i] = [sel["R"], sel["Z"]]
                island_sth[i] = [sel["s"], sel["theta"]]
                residual[i] = sel["residual"]

                # Measured iota (physical sign) by tracing mpol+1 transits from the selected seed.
                _, _, iota_trace, flag = specout.trace_one_fieldline(
                    np.array([sel["s"], sel["theta"]]),
                    volume,
                    num_phi_planes=1,
                    num_transits=mpol + 1,
                )
                iota[i] = float("nan") if flag else -iota_trace  # tracer sign -> physical iota

                # Nearest resonance ntor = k*Nfp and the iota_far flag (rel. error > 1%).
                if not np.isfinite(iota[i]):
                    ntor[i], iota_far[i] = 0, True
                else:
                    ntor_i = int(round(abs(iota[i]) * mpol / nfp)) * nfp
                    if ntor_i == 0:
                        ntor[i], iota_far[i] = 0, True
                    else:
                        iota_theo = ntor_i / mpol
                        ntor[i] = ntor_i
                        iota_far[i] = bool(
                            abs(abs(iota[i]) - iota_theo) / iota_theo > _IOTA_REL_TOL
                        )

                if iota_far[i]:
                    note = "IOTA_FAR"
                if prev_located_rz is not None:
                    displacement[i] = np.linalg.norm(island_rz[i] - prev_located_rz)

                # Update carry-state.
                locked_theta = sel["theta"]
                anchor_rz = np.array([sel["R"], sel["Z"]])
                anchor_s = sel["s"]
                prev_located_rz = island_rz[i].copy()

            t_total[i] = perf_counter() - t0

            if verbose:
                rax, zax = axis_rz[i]
                ax_str = f"{rax:8.4f} {zax:+8.4f}" if not np.isnan(rax) else "      nan        "
                tim = (
                    f"{t_total[i]:6.1f}/{t_field_reconstruction[i]:5.1f}/"
                    f"{t_fieldline_tracing[i]:6.1f}"
                )
                if sel is None:
                    row = (
                        f"  {it_idx:4d} | {ax_str} |   ---   | -  |        ---          |"
                        f"    ---   |   ---  |  --- |    ---   | "
                        f"{tim} | {note}"
                    )
                else:
                    th_lbl = "0" if sel["theta"] == 0.0 else "pi"
                    io_str = f"{iota[i]:+.4f}" if np.isfinite(iota[i]) else "  nan  "
                    d_str = (
                        f"{displacement[i] * 1e3:8.2f}"
                        if np.isfinite(displacement[i])
                        else "     ---"
                    )
                    row = (
                        f"  {it_idx:4d} | {ax_str} | {sel['s']:+.4f} | {th_lbl:>2} | "
                        f"{sel['R']:8.4f} {sel['Z']:+8.4f} | {sel['residual']:.2e} | "
                        f"{io_str} | {ntor[i]:4d} | {d_str} | {tim} | {note}"
                    )
                print(row, flush=True)

            if checkpoint_path is not None:
                _save(checkpoint_path)
    finally:
        signal.signal(signal.SIGUSR1, old_handler)

    n_ok = int(np.sum(~np.isnan(island_rz[:, 0])))
    res_ok = residual[~np.isnan(residual)]
    res_rng = f"{res_ok.min():.2e}..{res_ok.max():.2e}" if res_ok.size else "n/a"

    def _mean(a):
        m = np.nanmean(a) if np.any(np.isfinite(a)) else float("nan")
        return f"{m:.1f}" if np.isfinite(m) else "n/a"

    print(
        f"\nDone. island: {n_ok}/{n_iter} located  |  axis: "
        f"{int(np.sum(~np.isnan(axis_rz[:, 0])))}/{n_iter}  |  "
        f"no_fp: {int(np.sum(no_fp))}/{n_iter}  |  "
        f"leaving: {int(np.sum(leaving != ''))}/{n_iter}  |  "
        f"iota_far: {int(np.sum(iota_far))}/{n_iter}  |  residual range: {res_rng}"
    )
    print(
        f"Timing (mean s/checkpoint): total {_mean(t_total)}  |  "
        f"field_reconstruction {_mean(t_field_reconstruction)}  |  "
        f"fieldline_tracing {_mean(t_fieldline_tracing)}"
    )

    return _make_outputs()
