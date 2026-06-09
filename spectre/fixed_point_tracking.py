# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT
"""Track the magnetic axis and one island fixed-point chain across SPECTRE optimisation checkpoints.

Built on the ``spectre.fixed_point`` primitives (``scan_fixed_points``, ``root_fixed_point``,
``find_axis``) and the robust O/X classifier in ``spectre.fixed_point_classification``. Two regimes,
the same two acceptance conditions in both:

  * **Discovery** (no anchor: cold start, or just lost). FP-FIRST then iota-filter. Scan the whole
    radial extent of the volume(s) ``s in (-0.999, 0.999)`` at ``2*mpol+1`` theta, polish every
    candidate (return-map residual gate ``< ftol``), then keep only those whose empirical iota gives
    the target ``ntor = k*Nfp`` within ``iota_rtol``. Finding fixed points first and checking iota
    at them avoids the bias of measuring iota near a forming island. A cold start prefers an O-point
    (island centre) via the classifier.

  * **Anchor-follow** (have an anchor). Cheap narrow scan around the previous fixed point's
    ``(s, theta)``; the window half-width is drift-sized but distrusts an implausible ``(R,Z)->s``
    inversion and is capped. Two gates confirm "same island": ``(R,Z)`` continuity (drift-adaptive)
    and iota. Cross-volume handoff (with hysteresis) when an interface moves across the anchor.

Per checkpoint the result carries the axis and island ``(R,Z)``, the step displacement (the
geometric convergence metric), the robust O/X verdict, an island-size proxy, timings, and how the
point was acquired. Results are a dict of per-checkpoint arrays, optionally saved as ``.npz`` every
checkpoint. With ``verbose`` a compact labelled block is printed per checkpoint.
"""

from __future__ import annotations

import contextlib
import os
import re
import signal
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter

import numpy as np

_XS_FIXB_RE = re.compile(r"^x_(\d+)_(\d+)\.npy$")  # x_<a>_<b>.npy; group(2)=<b>=iteration number
_S_CLAMP = 0.999  # hard clamp on the seed s-range; s in (-1, 1) spans one annular volume

# Log glyphs, centralised so switching to ASCII is a one-line change.
_D, _OK, _NO, _WARN, _DASH, _TIMES = "Δ", "✓", "✗", "⚠", "—", "×"


def _fmt(x, w=8, p=4, sci=False):
    """Render a fixed-width float, ``---`` for non-finite, to keep the verbose block aligned."""
    if x is None or not np.isfinite(x):
        return f"{'---':>{w}}"
    return f"{x:{w}.{p}e}" if sci else f"{x:{w}.{p}f}"


def track_islands_and_axis(
    xs_fixb_dir: str | Path,
    toml_path: str | Path,
    mpol: int,
    volume: int,
    *,
    guess: Sequence[float] | None = None,
    discover: bool = False,
    target_ntor: int | None = None,
    discover_volumes: Sequence[int] | None = None,
    discover_n_theta: int | None = None,
    rediscover_after: int = 5,
    start_at_iter: int = 0,
    stride: int = 1,
    limit: int = 0,
    reverse: bool = False,
    ds_local: float = 0.05,
    follow_ds: float | None = None,
    follow_ds_max: float = 0.15,
    n_s: int = 31,
    disc_n_s: int = 121,
    dtheta: float = 0.3,
    n_theta: int = 7,
    follow_window: float | None = None,
    max_jump_floor: float = 0.02,
    handoff_margin: float = 0.05,
    no_handoff: bool = False,
    root_maxfev: int = 100,
    root_rtol: float = 1e-8,
    root_ftol: float = 1e-6,
    iota_rtol: float = 0.01,
    dedup_tol: float = 1e-3,
    near_parabolic_margin: float = 10.0,
    classify_rtol: float = 1e-12,
    fast_classify: bool = False,
    max_axis_repeats: int = 8,
    verbose: bool = False,
    checkpoint_path: str | Path | None = None,
) -> dict:
    """Track the magnetic axis and one island fixed-point chain across optimisation checkpoints.

    Either pass ``guess`` (an ``(R, Z)`` seed for the first checkpoint) or ``discover=True`` to
    self-seed by an FP-first scan filtered to the target resonance.

    Args:
        xs_fixb_dir: Folder of ``x_<a>_<b>.npy`` checkpoints (``b`` is the iteration number).
        toml_path: SPECTRE TOML matching the checkpoints.
        mpol: Island count = denominator of ``iota = ntor/mpol``.
        volume: Volume index hosting the chain at iter 0 (``>= 1``).
        guess: ``(R, Z)`` at ``phi=0`` of the tracked fixed point on the first checkpoint. Required
            unless ``discover``.
        discover: Self-seed (no ``guess``): FP-first scan, filter by iota, re-run whenever lost.
        target_ntor: Toroidal mode number of the chain (``iota = ntor/mpol``); the iota gate keeps
            only fixed points with this ntor. Default: any positive multiple of Nfp.
        discover_volumes: Volumes to search in discovery (default: just ``volume``).
        discover_n_theta: theta samples for the discovery scan (default ``max(2*mpol+1, 16)``).
        rediscover_after: Drop a stale anchor and re-discover after this many lock-free checkpoints.
        start_at_iter: Skip checkpoints with iteration ``< start_at_iter``.
        stride: Process every ``stride``-th checkpoint.
        limit: Process at most this many checkpoints (0 = all).
        reverse: Process high->low iter (track backward from the converged state).
        ds_local: Floor half-width in ``s`` of the steady-state anchor-follow scan.
        follow_ds: Fixed anchor-follow half-width in ``s`` (default AUTO from the measured drift).
        follow_ds_max: Hard ceiling on the AUTO anchor-follow half-width in ``s``.
        n_s: ``s`` samples per theta for the narrow anchor-follow scan.
        disc_n_s: ``s`` samples per theta for the full-volume discovery scan.
        dtheta: Half-width in theta of the localized anchor-follow scan.
        n_theta: theta samples in the anchor-follow scan (1 = on-anchor line only).
        follow_window: Continuity gate, max ``(R,Z)`` move [m] from anchor (default AUTO from the
            island geometry).
        max_jump_floor: ``(R,Z)`` continuity floor [m]; a candidate may move up to
            ``max(this, 10*prev_disp)``.
        handoff_margin: Hysteresis [in ``s``] before switching volumes at an interface.
        no_handoff: Disable cross-volume handoff at interfaces.
        root_maxfev: Max function evaluations of the Newton polish.
        root_rtol: Integrator tolerance of the Newton polish.
        root_ftol: Return-map residual acceptance gate ``||(dR,dZ)|| < ftol`` [m].
        iota_rtol: iota gate, keep a point only if ``|iota-ntor/mpol|/(ntor/mpol) <= this``.
        dedup_tol: ``(R,Z)`` tolerance [m] for collapsing duplicate discovery members.
        near_parabolic_margin: Trust gate of the O/X classifier (see ``classify_point``).
        classify_rtol: Integrator tolerance for the O/X classification traces.
        fast_classify: Use the cheap library ``classify_residue`` instead of the robust classifier.
        max_axis_repeats: Random-restart budget for the axis finder.
        verbose: Print a compact labelled block per checkpoint.
        checkpoint_path: If given, re-save the ``.npz`` after every checkpoint (send ``SIGUSR1`` to
            request a clean stop after the current checkpoint).

    Returns:
        A dict of per-checkpoint arrays and scalar metadata.
    """
    # SPECTRE imports are deferred to call time (heavy core; keeps module import cheap).
    from spectre import SPECTRE
    from spectre.fixed_point import (
        _return_map,
        classify_residue,
        find_axis,
        root_fixed_point,
        scan_fixed_points,
    )
    from spectre.fixed_point_classification import FixedPointClassifier, classify_point
    from spectre.force_targets import force_real
    from spectre.utils import get_xinit_specwrap, unpack_interfaces

    xs_fixb_dir, toml_path = Path(xs_fixb_dir), Path(toml_path)
    if not xs_fixb_dir.is_dir():
        raise FileNotFoundError(f"xs_fixb directory not found: {xs_fixb_dir}")
    if not toml_path.is_file():
        raise FileNotFoundError(f"TOML file not found: {toml_path}")
    if guess is None and not discover:
        raise ValueError("guess (R, Z) is required unless discover=True.")
    # A genuine period-mpol chain has iota in lowest terms, so the target must be coprime with mpol.
    if target_ntor is not None and int(np.gcd(target_ntor, mpol)) != 1:
        g = int(np.gcd(target_ntor, mpol))
        raise ValueError(
            f"target_ntor {target_ntor} is not coprime with mpol {mpol} (gcd={g}); that is a "
            f"period-{mpol // g} chain, not period-{mpol}; pass a coprime mpol/target_ntor."
        )

    out = None if checkpoint_path is None else Path(checkpoint_path)
    n_transits = (
        mpol + 1
    )  # trace_one_fieldline logs phi=0 at t=0 too, so the mpol-fold map needs mpol+1
    stride = max(1, int(stride))
    # The poloidal discovery scan must resolve the mpol islands around the circle: sample 2*mpol+1
    # theta (just above the 2*mpol Nyquist floor).
    disc_n_theta = discover_n_theta or max(2 * mpol + 1, 16)
    root_kw = dict(maxfev=root_maxfev, rtol=root_rtol, atol=root_rtol, ftol=root_ftol)
    # Boxed so the nested helpers can write them.
    nfp = [-1]
    follow_box = [follow_window]

    # ── List checkpoints, sorted by iteration ──────────────────────────────────
    all_paths = sorted(
        ((int(m.group(2)), q) for q in xs_fixb_dir.iterdir() if (m := _XS_FIXB_RE.match(q.name))),
        key=lambda t: t[0],
    )
    if not all_paths:
        raise FileNotFoundError(f"No x_***_***.npy files found in {xs_fixb_dir}")
    paths = [(it, q) for it, q in all_paths if it >= start_at_iter][::stride]
    if reverse:
        paths = paths[::-1]  # high->low iter: track backward from the converged state
    if limit > 0:
        paths = paths[:limit]
    if not paths:
        raise ValueError(
            f"No checkpoints with iter >= {start_at_iter} "
            f"(have {all_paths[0][0]}..{all_paths[-1][0]})"
        )
    n = len(paths)

    # ── Lean header (always printed) ───────────────────────────────────────────
    seed_str = "DISCOVER" if discover else f"guess{tuple(guess)}"
    ntor_str = f"{target_ntor} ({target_ntor}/{mpol})" if target_ntor else "any k*Nfp"
    gate_res = f"{target_ntor}/{mpol}" if target_ntor else "nearest k*Nfp/mpol"
    print("track_FP — island + axis tracker")
    print(f"  tracking : period-{mpol} · ntor {ntor_str} · volume {volume} · seed {seed_str}")
    print(
        f"  run      : {n} checkpoints (iters {paths[0][0]}..{paths[-1][0]}, "
        f"{'reverse' if reverse else 'forward'}, stride {stride})"
    )
    print(f"  gates    : FP residual < {root_ftol:.0e} · iota within {iota_rtol:.0e} of {gate_res}")
    if out is not None:
        print(f"  output   : {out.name}")
    if sid := os.environ.get("SLURM_JOB_ID"):
        print(f"  stop     : scancel --signal=USR1 {sid}")
    if discover and target_ntor is None:
        print(
            "  ** WARNING: discover without target_ntor — the first cold lock pins the chain; "
            "pass target_ntor to avoid a silent wrong-chain lock."
        )
    print(flush=True)

    # ── Pre-allocate every per-checkpoint diagnostic ───────────────────────────
    iter_indices = np.array([it for it, _ in paths], dtype=int)
    nan = lambda shape=n: np.full(shape, np.nan)  # noqa: E731
    vol_tracked = np.full(n, volume, dtype=int)
    axis_rz, axis_disp = nan((n, 2)), nan()
    island_rz, island_sth = nan((n, 2)), nan((n, 2))
    residual, residual_scan, polished = nan(), nan(), np.zeros(n, dtype=bool)
    iota, ntor, iota_far = nan(), np.zeros(n, dtype=int), np.zeros(n, dtype=bool)
    iota_rel = nan()
    kind = np.full(n, "", dtype="<U10")
    residue, rtrace, rdet, rdet_err = nan(), nan(), nan(), nan()
    residue_stable = np.zeros(n, dtype=bool)
    disc, classify_method, orbit_growth = nan(), np.full(n, "", dtype="<U8"), nan()
    in_window, select_dist = np.zeros(n, dtype=bool), nan()
    displacement, disp_s = nan(), nan()
    best_rz, best_sth, best_residual = nan((n, 2)), nan((n, 2)), nan()
    best_nfev, best_converged = np.zeros(n, dtype=int), np.zeros(n, dtype=bool)
    width_s, width_R = nan(), nan()
    width_floored = np.zeros(n, dtype=bool)
    n_cand_total, n_cand_inwin = np.zeros(n, dtype=int), np.zeros(n, dtype=int)
    n_iota_pass = np.zeros(n, dtype=int)
    end_pinned_s = np.full(n, "", dtype="<U6")
    leaving, no_fp = np.full(n, "", dtype="<U6"), np.zeros(n, dtype=bool)
    rediscovered = np.zeros(n, dtype=bool)
    provenance = np.full(n, "", dtype="<U10")  # how the FP was acquired: cold / re-lock / follow
    handoff = np.full(n, "", dtype="<U12")
    seed_sth = nan((n, 2))
    t_total, t_recon, t_axis = nan(), nan(), nan()
    t_scan, t_polish, t_residue = nan(), nan(), nan()

    def outputs() -> dict:
        return {
            "iteration_indices": iter_indices,
            "vol_tracked": vol_tracked,
            "axis_rz": axis_rz,
            "axis_disp": axis_disp,
            "island_rz": island_rz,
            "island_sth": island_sth,
            "residual": residual,
            "residual_scan": residual_scan,
            "polished": polished,
            "iota": iota,
            "ntor": ntor,
            "iota_far": iota_far,
            "iota_rel": iota_rel,
            "kind": kind,
            "residue": residue,
            "trace": rtrace,
            "det": rdet,
            "det_err": rdet_err,
            "residue_stable": residue_stable,
            "disc": disc,
            "classify_method": classify_method,
            "orbit_growth": orbit_growth,
            "in_window": in_window,
            "select_dist": select_dist,
            "displacement": displacement,
            "disp_s": disp_s,
            "best_rz": best_rz,
            "best_sth": best_sth,
            "best_residual": best_residual,
            "best_nfev": best_nfev,
            "best_converged": best_converged,
            "width_s": width_s,
            "width_R": width_R,
            "width_floored": width_floored,
            "n_candidates": n_cand_total,
            "n_candidates_inwin": n_cand_inwin,
            "n_iota_pass": n_iota_pass,
            "end_pinned": end_pinned_s,
            "leaving": leaving,
            "no_fp": no_fp,
            "rediscovered": rediscovered,
            "provenance": provenance,
            "handoff": handoff,
            "seed_sth": seed_sth,
            "t_total": t_total,
            "t_field_reconstruction": t_recon,
            "t_axis": t_axis,
            "t_scan": t_scan,
            "t_polish": t_polish,
            "t_residue": t_residue,
            "mpol": mpol,
            "volume_start": volume,
            "nfp": nfp[0],
            "start_at_iter": int(start_at_iter),
            "stride": int(stride),
            "n_s": int(n_s),
            "disc_n_s": int(disc_n_s),
            "n_theta": int(n_theta),
            "dtheta": float(dtheta),
            "root_ftol": float(root_ftol),
            "iota_rtol": float(iota_rtol),
            "dedup_tol": float(dedup_tol),
            "near_parabolic_margin": float(near_parabolic_margin),
            "classify_rtol": float(classify_rtol),
            "fast_classify": bool(fast_classify),
            "target_ntor": int(target_ntor) if target_ntor else -1,
            "follow_window": float(follow_box[0]) if follow_box[0] is not None else -1.0,
            "reverse": bool(reverse),
            "discover": bool(discover),
            "guess_rz": np.array([np.nan, np.nan] if guess is None else guess, dtype=float),
        }

    # ── SIGUSR1 → finish current checkpoint then stop ──────────────────────────
    stop = [False]
    old_handler = signal.signal(
        signal.SIGUSR1,
        lambda *_: (
            stop.__setitem__(0, True),
            print("\n  [SIGUSR1] stop after this checkpoint.", flush=True),
        ),
    )

    obj = SPECTRE(str(toml_path), verbose=False)
    clf_cache: dict = {}  # per-checkpoint FixedPointClassifier per volume (reset each iteration)

    def find_fp(
        specout,
        vol,
        s_lo,
        s_hi,
        theta_grid,
        anchor_rz=None,
        max_move=None,
        return_all=False,
        require_ntor=None,
        n_s_scan=None,
    ):
        """FP-first then iota-filter: scan, polish each minimum, keep points passing both gates.

        Gates are the return-map residual (``root_fixed_point`` converged, i.e. residual < ftol) and
        the iota gate (a genuine period-mpol resonance with ``ntor`` a positive multiple of Nfp, in
        lowest terms, ``== require_ntor`` when given, and within ``iota_rtol``). With ``return_all``
        returns the residual-sorted list of accepted members (deduplicated by ``(R,Z)``), else the
        first. Returns ``(result, end_pinned, best_polish, stats)``. The ``(R,Z)`` proximity gate is
        applied only against ``anchor_rz`` (continuity), never the scan seed.
        """
        s_lo, s_hi = max(-_S_CLAMP, s_lo), min(_S_CLAMP, s_hi)
        try:
            cands, end_pinned = scan_fixed_points(
                specout,
                vol,
                n_transits,
                theta_grid=theta_grid,
                s_range=(s_lo, s_hi),
                n_s=(n_s_scan or n_s),
            )
        except Exception as exc:
            print(f"      scan failed (vol {vol}) — {exc}", flush=True)
            return ([] if return_all else None), {}, None, {"n_cands": 0, "n_rm": 0, "n_iota": 0}
        confirmed, best_polish, n_rm = [], None, 0
        for c in sorted(cands, key=lambda c: c["residual"]):
            try:
                pol = root_fixed_point(specout, vol, n_transits, (c["s"], c["theta"]), **root_kw)
            except Exception:
                continue
            if np.isfinite(pol.residual) and (
                best_polish is None or pol.residual < best_polish["residual"]
            ):
                best_polish = {
                    "s": pol.s,
                    "theta": pol.theta,
                    "R": pol.R,
                    "Z": pol.Z,
                    "residual": pol.residual,
                    "nfev": int(pol.nfev),
                    "converged": bool(pol.converged),
                    "scan_res": c["residual"],
                }
            if not pol.converged:
                continue
            if (
                anchor_rz is not None
                and np.hypot(pol.R - anchor_rz[0], pol.Z - anchor_rz[1]) > max_move
            ):
                continue  # polished to a different member than tracked — keep continuity
            n_rm += 1
            # iota gate: measure iota AT the fixed point (well-posed there) and keep only a genuine
            # period-mpol resonance — ntor>0, Nfp-compatible, lowest terms, == require_ntor.
            _, _, iota_tr, flag = specout.trace_one_fieldline(
                np.array([pol.s, pol.theta]), vol, num_phi_planes=1, num_transits=n_transits
            )
            if flag:
                continue
            iota_emp = -float(iota_tr)  # physical (VMEC) sign convention
            ntor_p = int(round(abs(iota_emp) * mpol)) if np.isfinite(iota_emp) else 0
            ok = (
                ntor_p > 0
                and ntor_p % nfp[0] == 0
                and int(np.gcd(ntor_p, mpol)) == 1
                and (require_ntor is None or ntor_p == require_ntor)
            )
            if not ok:
                continue
            rel = abs(abs(iota_emp) - ntor_p / mpol) / (ntor_p / mpol)
            if rel > iota_rtol:
                continue
            entry = {
                "s": pol.s,
                "theta": pol.theta,
                "R": pol.R,
                "Z": pol.Z,
                "residual": pol.residual,
                "nfev": int(pol.nfev),
                "scan_res": c["residual"],
                "ntor": ntor_p,
                "iota": iota_emp,
                "iota_rel": rel,
            }
            if not return_all:
                return (
                    entry,
                    end_pinned,
                    best_polish,
                    {"n_cands": len(cands), "n_rm": n_rm, "n_iota": 1},
                )
            confirmed.append(entry)
        # Collapse duplicates: the same FP is polished from several adjacent theta seeds. Keep the
        # lowest-residual member of each (R,Z) cluster so counts/selection see distinct members.
        deduped = []
        for e in sorted(confirmed, key=lambda e: e["residual"]):
            if all(np.hypot(e["R"] - k["R"], e["Z"] - k["Z"]) > dedup_tol for k in deduped):
                deduped.append(e)
        return (
            (deduped if return_all else None),
            end_pinned,
            best_polish,
            {"n_cands": len(cands), "n_rm": n_rm, "n_iota": len(deduped)},
        )

    def island_width(specout, vol, s0, th0):
        """Estimate the island radial size as the return-map residual-basin half-width.

        Adaptive: grow the window until the small-residual basin is bracketed on both sides, up to a
        cap, then convert the half-width in ``s`` to ``dR`` via local ``dR/ds``. Returns
        ``(width_s, width_R, floored)`` — NaN widths where the basin can't be cleanly bracketed, and
        ``(NaN, NaN, True)`` where it brackets only at the grid-resolution floor.
        """
        cap, half = min(0.6, 20.0 * ds_local), max(4.0 * ds_local, 0.15)
        w_s, floored = np.nan, False
        while True:
            npts = max(41, int(2.0 * half / 7.5e-3))  # ~constant s-resolution as the window grows
            sg = np.linspace(max(-_S_CLAMP, s0 - half), min(_S_CLAMP, s0 + half), npts)
            res = np.full(npts, np.nan)
            for j, s in enumerate(sg):
                with contextlib.suppress(Exception):
                    rm = _return_map(
                        [s, th0],
                        specout,
                        vol,
                        n_transits,
                        method="RK45",
                        rtol=root_rtol,
                        atol=root_rtol,
                    )
                    res[j] = float(np.hypot(*rm))
            fin = res[np.isfinite(res)]
            if fin.size < 3:
                return np.nan, np.nan, False
            r_floor, r_plateau = float(np.nanmin(res)), float(np.nanpercentile(fin, 90))
            i0 = int(np.argmin(np.abs(sg - s0)))
            lo = hi = None
            if r_plateau > r_floor > 0:
                thresh = np.sqrt(r_floor * r_plateau)  # log-midpoint floor<->off-island plateau
                lo = next(
                    (sg[j] for j in range(i0, -1, -1) if np.isfinite(res[j]) and res[j] > thresh),
                    None,
                )
                hi = next(
                    (sg[j] for j in range(i0, npts) if np.isfinite(res[j]) and res[j] > thresh),
                    None,
                )
            if lo is not None and hi is not None:
                dg = float(sg[1] - sg[0])  # grid spacing
                floored = (hi - lo) <= 2.0 * dg * (
                    1.0 + 1e-9
                )  # lo,hi at i0+-1 -> basin below resolution
                w_s = 0.5 * (hi - lo)
                break
            if half >= cap:
                return np.nan, np.nan, False  # still saturated at the cap -> honest NaN
            half = min(cap, 2.0 * half)  # grow and rescan
        if floored:
            return np.nan, np.nan, True
        try:
            dds = 1e-3
            rp = specout.get_rz_coords(vol, min(_S_CLAMP, s0 + dds), th0, 0.0)
            rm = specout.get_rz_coords(vol, max(-_S_CLAMP, s0 - dds), th0, 0.0)
            drds = abs(
                float(np.asarray(rp[0]).ravel()[0]) - float(np.asarray(rm[0]).ravel()[0])
            ) / (2 * dds)
            return w_s, w_s * drds, False
        except Exception:
            return w_s, np.nan, False

    def classify_fp(specout, vol, r, z):
        """Robust O/X of a located fixed point (reuses a warmed classifier; None on failure)."""
        if fast_classify:
            try:
                info = classify_residue(specout, vol, r, z, n_transits)
                return dict(
                    kind=info["kind"],
                    residue=info["residue"],
                    disc=np.nan,
                    trace=info["trace"],
                    det=info["det"],
                    det_err=info["meta"]["det_err_best"],
                    stable=bool(info["stable"]),
                    method="residue_fast",
                    growth=np.nan,
                )
            except Exception as exc:
                print(f"    classify (fast) error - {exc}", flush=True)
                return None
        try:
            clf = clf_cache.get(vol)
            if clf is None:
                clf = FixedPointClassifier(
                    specout, vol, mpol, method="DOP853", rtol=classify_rtol, atol=classify_rtol
                )
                clf_cache[vol] = clf
            return classify_point(clf, (r, z), near_parabolic_margin=near_parabolic_margin)
        except Exception as exc:
            print(f"    classify error - {exc}", flush=True)
            return None

    cur_vol = volume
    anchor_rz = None if discover else np.array(guess, dtype=float)
    anchor_th: float | None = None
    prev_axis: np.ndarray | None = None
    prev_island: np.ndarray | None = None
    prev_s: float | None = None
    prev_ntor: int | None = target_ntor
    last_member_th: float | None = None
    prev_disp = np.nan
    n_since_lock = 0

    try:
        for i, (it, npy) in enumerate(paths):
            if stop[0]:
                print(f"  Stopping before iter {it} (SIGUSR1).", flush=True)
                break
            t0 = perf_counter()
            n_since_lock += 1
            warns: list[str] = []  # verbose ⚠ exception lines for this checkpoint
            # Drop a stale anchor so discovery re-triggers instead of scanning a dead point forever.
            if discover and anchor_rz is not None and n_since_lock > rediscover_after:
                warns.append(
                    f"stale anchor ({n_since_lock - 1} checkpoints w/o lock) — re-discovering"
                )
                anchor_rz = anchor_th = None

            # 1) Rebuild the equilibrium from the checkpoint.
            try:
                unpack_interfaces(np.load(str(npy)), obj)
                force_real(get_xinit_specwrap(obj), obj)
                specout = obj.create_spectreout()
            except Exception as exc:
                print(f"[iter {it:3d}]  ! checkpoint load FAILED — {exc}; skipping.", flush=True)
                continue
            t_recon[i] = perf_counter() - t0
            if nfp[0] < 0:
                nfp[0] = int(specout.input.physics.Nfp)

            # 2) Magnetic axis. Failure leaves the row NaN, does not abort.
            ta = perf_counter()
            rax = zax = np.nan
            try:
                rax, zax, _, _ = find_axis(specout, max_repeats=max_axis_repeats, verbose=False)
                axis_rz[i] = [rax, zax]
                if prev_axis is not None:
                    axis_disp[i] = np.linalg.norm(axis_rz[i] - prev_axis)
                prev_axis = axis_rz[i].copy()
            except Exception as exc:
                warns.append(f"axis finding failed — {exc}")
            t_axis[i] = perf_counter() - ta

            # 3) Acquire the fixed point: anchor-follow (narrow scan) else discovery (full-volume).
            fp = None
            from_discovery = False
            do_discover = discover and anchor_rz is None
            lost_rz = None
            end_pinned: dict = {}
            best_diag = None
            committed_class = None
            acq_msg = ""
            clf_cache.clear()
            t_scan[i] = 0.0

            if anchor_rz is not None:
                # Anchor (R,Z) -> (s,theta) on THIS field; hand off to a neighbour volume only with
                # hysteresis (the interface itself moves, so an island near a boundary would flap).
                seed_s = seed_th = None
                for v in (cur_vol, cur_vol - 1, cur_vol + 1):
                    if v < 1:
                        continue
                    try:
                        cs, ct = specout.get_st_coords(v, anchor_rz[0], anchor_rz[1], 0.0)
                        cand_s = float(np.asarray(cs).ravel()[0])
                        cand_th = float(np.asarray(ct).ravel()[0])
                    except Exception:
                        continue
                    if v != cur_vol:
                        if no_handoff or abs(cand_s) >= _S_CLAMP - handoff_margin:
                            continue  # not clearly inside the neighbour — don't flap
                        handoff[i] = f"vol{cur_vol}->vol{v}"
                        warns.append(f"handoff {handoff[i]} (s={cand_s:+.3f}, inside margin)")
                        cur_vol = v
                    seed_s, seed_th = cand_s, cand_th
                    break
                if seed_s is not None:
                    # Re-centre theta on the previous member — but NOT across a handoff, where
                    # the fresh inversion theta in the new volume is the trustworthy seed.
                    if anchor_th is not None and not handoff[i]:
                        seed_th = anchor_th
                    # Centre the scan on prev_s (the committed surface predicts the next far better
                    # than the noisy (R,Z)->s inversion). Size the window from the measured drift,
                    # but distrust an implausibly large inversion and always cap at follow_ds_max.
                    use_prev = prev_s is not None and np.isfinite(prev_s) and not handoff[i]
                    seed_s_used = prev_s if use_prev else seed_s
                    if follow_ds is not None:
                        follow_ds_i = follow_ds
                    elif use_prev:
                        gap = abs(seed_s - prev_s)
                        if gap > follow_ds_max:
                            follow_ds_i = ds_local
                            warns.append(
                                f"distrust inversion s={seed_s:+.4f} vs prev_s={prev_s:+.4f} "
                                f"off by {gap:.3f} -> box=ds_local"
                            )
                        else:
                            follow_ds_i = min(max(ds_local, 2.0 * gap), follow_ds_max)
                    else:
                        follow_ds_i = ds_local
                    geom_ceiling = follow_box[0] if follow_box[0] is not None else 0.30
                    drift_gate = (
                        max_jump_floor
                        if not np.isfinite(prev_disp)
                        else max(max_jump_floor, 10.0 * prev_disp)
                    )
                    max_move = min(geom_ceiling, drift_gate)
                    seed_sth[i] = [seed_s_used, seed_th]
                    th_grid = (
                        (float(seed_th),)
                        if n_theta <= 1 or dtheta <= 0
                        else np.linspace(seed_th - dtheta, seed_th + dtheta, n_theta)
                    )
                    ts = perf_counter()
                    fp, end_pinned, bp, st = find_fp(
                        specout,
                        cur_vol,
                        seed_s_used - follow_ds_i,
                        seed_s_used + follow_ds_i,
                        th_grid,
                        anchor_rz=anchor_rz,
                        max_move=max_move,
                        require_ntor=prev_ntor,
                    )
                    n_cand_total[i] = st["n_cands"]
                    if bp is not None and (
                        best_diag is None or bp["residual"] < best_diag["residual"]
                    ):
                        best_diag = bp
                    t_scan[i] += perf_counter() - ts
                    acq_msg = (
                        f"[follow] tracked from previous anchor ({n_cand_total[i]} cand, "
                        f"window {max_move * 1e3:.0f} mm)"
                    )
                if fp is None and discover:
                    # Anchor-follow lost the member -> re-discover, but remember where it was so
                    # discovery re-locks THE SAME island (nearest), not a far member.
                    lost_rz = anchor_rz.copy()
                    do_discover = True

            if fp is None and do_discover:
                # FP-first: scan each volume's full radial extent, polish, keep iota-gate passers.
                vols = list(discover_volumes) if discover_volumes else [volume]
                th_full = np.linspace(0.0, 2.0 * np.pi, disc_n_theta, endpoint=False)
                confirmed = []  # (|v - volume|, v, entry) across all searched volumes
                ts = perf_counter()
                for v in vols:
                    confs, _ep, bp, st = find_fp(
                        specout,
                        v,
                        -_S_CLAMP,
                        _S_CLAMP,
                        th_full,
                        return_all=True,
                        require_ntor=prev_ntor,
                        n_s_scan=disc_n_s,
                    )
                    n_cand_total[i] += st["n_cands"]
                    n_iota_pass[i] += st["n_iota"]
                    if bp is not None and (
                        best_diag is None or bp["residual"] < best_diag["residual"]
                    ):
                        best_diag = bp
                    confirmed += [(abs(v - volume), v, e) for e in confs]
                confirmed.sort(
                    key=lambda t: (t[0], t[2]["residual"])
                )  # nearest volume, then lowest residual
                n_members = len(confirmed)
                res_str = f"{prev_ntor}/{mpol}" if prev_ntor else f"k*Nfp (period-{mpol})"
                if confirmed and lost_rz is not None:
                    # Re-acquiring a tracked member: pick the accepted member nearest the lost
                    # anchor and accept ONLY within the continuity window — never hop.
                    tol = follow_box[0] if follow_box[0] is not None else 0.30
                    _, v, best = min(
                        confirmed,
                        key=lambda t: np.hypot(t[2]["R"] - lost_rz[0], t[2]["Z"] - lost_rz[1]),
                    )
                    dist = np.hypot(best["R"] - lost_rz[0], best["Z"] - lost_rz[1])
                    if dist <= tol:
                        fp, cur_vol, from_discovery = best, v, True
                        acq_msg = (
                            f"[re-lock] follow lost it -> re-discovered {n_members} members; "
                            f"nearest member {dist * 1e3:.1f} mm <= {tol * 1e3:.0f} mm window"
                        )
                    else:
                        acq_msg = (
                            f"[no_fp] nearest re-lock member {dist * 1e3:.1f} mm > "
                            f"{tol * 1e3:.0f} mm window — not hopping"
                        )
                elif confirmed:
                    # Cold start: prefer an O-point (island centre); order by member stability
                    # (theta nearest the last member, else lowest residual) and commit the first O.
                    if last_member_th is not None:

                        def wrap(th, _m=last_member_th):  # _m default binds the loop value
                            return abs((th - _m + np.pi) % (2 * np.pi) - np.pi)

                        order = sorted(
                            confirmed, key=lambda t: (wrap(t[2]["theta"]), t[2]["residual"])
                        )
                    else:
                        order = confirmed
                    for _, v, e in order:
                        ci = classify_fp(specout, v, e["R"], e["Z"])
                        if ci is not None and ci["kind"] == "o":
                            cur_vol, fp, committed_class = v, e, ci
                            acq_msg = f"[cold] {n_members} members on {res_str}; locked O-point"
                            break
                    else:
                        _, cur_vol, fp = order[0]
                        committed_class = classify_fp(specout, cur_vol, fp["R"], fp["Z"])
                        acq_msg = (
                            f"[cold] discovered {n_members} members on {res_str}; "
                            f"no O found -> nearest-seed member"
                        )
                    from_discovery = True
                t_scan[i] += perf_counter() - ts

            # -- Commit (polish-gated: fp set only if a Newton polish confirmed it) --
            vol_tracked[i] = cur_vol
            done = i + 1
            if fp is None:
                no_fp[i] = True
                leaving[i] = next((sd for sd in end_pinned.values() if sd), "")
                nofp_reason = "no polishable scan candidate"
                if best_diag is not None:
                    best_rz[i] = [best_diag["R"], best_diag["Z"]]
                    best_sth[i] = [best_diag["s"], best_diag["theta"]]
                    best_residual[i] = best_diag["residual"]
                    best_nfev[i] = best_diag["nfev"]
                    best_converged[i] = best_diag["converged"]
                    why = (
                        "passed ftol; failed iota/continuity"
                        if best_diag["converged"]
                        else "failed ftol gate"
                    )
                    nofp_reason = (
                        f"best polish {best_diag['residual']:.1e} @ s={best_diag['s']:+.4f} "
                        f"th={best_diag['theta']:.2f} ({why})"
                    )
                t_total[i] = perf_counter() - t0
                elapsed = float(np.nansum(t_total[:done]))
                if verbose:
                    print(f"\n[iter {it:3d}]  {done}/{n}  vol{cur_vol}")
                    print(
                        f"  axis     R={rax:.5f} Z={zax:+.5f}            {_D} "
                        f"{_fmt(axis_disp[i] * 1e3, 6, 2)} mm"
                    )
                    for w in warns:
                        print(f"  {_WARN} {w}")
                    print(f"  {_WARN} no_fp   {nofp_reason}")
                    print(
                        f"  timing  recon {_fmt(t_recon[i], 5, 0)}s scan {_fmt(t_scan[i], 5, 0)}s "
                        f"total {_fmt(t_total[i], 5, 0)}s  | {elapsed / 60:.1f}m elapsed",
                        flush=True,
                    )
                if out is not None:
                    np.savez(str(out), **outputs())
                continue

            s_sel, th_sel, r_sel, z_sel = fp["s"], fp["theta"], fp["R"], fp["Z"]
            residual[i], residual_scan[i], polished[i] = fp["residual"], fp["scan_res"], True
            in_window[i] = True
            island_rz[i] = [r_sel, z_sel]
            island_sth[i] = [s_sel, th_sel]
            best_rz[i], best_sth[i] = [r_sel, z_sel], [s_sel, th_sel]
            best_residual[i], best_nfev[i], best_converged[i] = (
                fp["residual"],
                int(fp.get("nfev", 0)),
                True,
            )

            # iota / ntor were measured at the fixed point by the iota gate — reuse (no re-trace).
            iota[i], ntor[i], iota_rel[i] = fp["iota"], fp["ntor"], fp["iota_rel"]
            iota_far[i] = bool(fp["iota_rel"] > 0.5 * iota_rtol)
            if iota_far[i]:
                warns.append(f"marginal: iota_rel {iota_rel[i]:.1e} in upper half of gate band")

            # Robust O/X classification (reuse discovery's verdict for this member if available).
            tr = perf_counter()
            ci = (
                committed_class
                if committed_class is not None
                else classify_fp(specout, cur_vol, r_sel, z_sel)
            )
            if ci is not None:
                kind[i], residue[i] = ci["kind"], ci["residue"]
                rtrace[i], rdet[i] = ci["trace"], ci["det"]
                rdet_err[i], residue_stable[i] = ci["det_err"], ci["stable"]
                disc[i], classify_method[i], orbit_growth[i] = (
                    ci["disc"],
                    ci["method"],
                    ci["growth"],
                )
            t_residue[i] = perf_counter() - tr

            try:
                width_s[i], width_R[i], width_floored[i] = island_width(
                    specout, cur_vol, s_sel, th_sel
                )
            except Exception as exc:
                warns.append(f"width proxy error - {exc}")

            # Auto-derive the continuity gate from the island geometry (unless fixed): the mpol
            # islands lie at minor radius r = ||FP-axis||, so 0.35*2*pi*r/mpol ~ 0.7x O-X spacing.
            if follow_window is None and np.isfinite(axis_rz[i, 0]):
                r = np.hypot(r_sel - axis_rz[i, 0], z_sel - axis_rz[i, 1])
                follow_box[0] = 0.35 * 2.0 * np.pi * r / mpol

            # Displacement is the convergence metric, valid only across an anchor-follow step (a
            # handoff still follows the same member). ANY discovery-sourced lock breaks continuity
            # (the nearest member is not guaranteed to be the same one), so leave displacement NaN
            # — which also stops a falsified re-lock distance from poisoning prev_disp.
            rediscovered[i] = from_discovery
            continuity_broken = from_discovery
            provenance[i] = (
                ("cold" if lost_rz is None else "re-lock") if from_discovery else "follow"
            )
            rz_valid = not continuity_broken
            s_valid = (not continuity_broken) and (not handoff[i])
            if prev_island is not None and rz_valid:
                displacement[i] = np.linalg.norm(
                    island_rz[i] - prev_island
                )  # (R,Z) survives a handoff
            if prev_s is not None and np.isfinite(prev_s) and s_valid:
                disp_s[i] = s_sel - prev_s
            prev_island = island_rz[i].copy()
            prev_s = s_sel
            prev_ntor = int(ntor[i])
            last_member_th = th_sel
            prev_disp = displacement[i] if np.isfinite(displacement[i]) else prev_disp
            anchor_rz = island_rz[i].copy()
            anchor_th = th_sel
            n_since_lock = 0

            t_total[i] = perf_counter() - t0
            elapsed = float(np.nansum(t_total[:done]))
            if verbose:
                kindword = "O-point (elliptic)" if kind[i] == "o" else "X-point (hyperbolic)"
                if classify_method[i] == "orbit":
                    how = (
                        f"by orbit-topology (near-parabolic, growth {orbit_growth[i]:.1f}{_TIMES})"
                    )
                elif classify_method[i]:
                    how = f"by Greene residue (R={residue[i]:.1e}, |det-1|={rdet_err[i]:.0e})"
                else:
                    how, kindword = "classification unavailable", "?"
                disp_mm = displacement[i] * 1e3 if np.isfinite(displacement[i]) else np.nan
                gate_ok = _OK if residual[i] < root_ftol else _NO
                iota_ok = _OK if iota_rel[i] < iota_rtol else _NO
                print(f"\n[iter {it:3d}]  {done}/{n}  vol{cur_vol}")
                print(
                    f"  axis     R={rax:.5f} Z={zax:+.5f}            {_D} "
                    f"{_fmt(axis_disp[i] * 1e3, 6, 2)} mm"
                )
                print(
                    f"  island   R={r_sel:.4f} Z={z_sel:+.4f}             {_D} "
                    f"{_fmt(disp_mm, 6, 2)} mm"
                )
                print(f"  acquire  {acq_msg}")
                for w in warns:
                    print(f"  {_WARN} {w}")
                print(
                    f"  gates    FP res {residual[i]:.1e} < {root_ftol:.0e} {gate_ok}   "
                    f"iota {iota[i]:+.4f} = {ntor[i]}/{mpol}  rel {iota_rel[i]:.1e} < "
                    f"{iota_rtol:.0e} {iota_ok}"
                )
                print(f"  class    {kindword} {_DASH} {how}")
                print(
                    f"  timing   recon {_fmt(t_recon[i], 5, 0)}s  scan {_fmt(t_scan[i], 5, 0)}s  "
                    f"total {_fmt(t_total[i], 5, 0)}s  | {elapsed / 60:.1f}m elapsed",
                    flush=True,
                )
            if out is not None:
                np.savez(str(out), **outputs())
    finally:
        signal.signal(signal.SIGUSR1, old_handler)

    # ── Summary ────────────────────────────────────────────────────────────────
    found = ~np.isnan(island_rz[:, 0])
    n_cold = int(np.sum(provenance == "cold"))
    n_relock = int(np.sum(provenance == "re-lock"))
    n_follow = int(np.sum(provenance == "follow"))
    res_rng = (
        f"{residual[found].min():.1e}..{np.nanmax(residual[found]):.1e}" if found.any() else "n/a"
    )
    print(
        f"\nDone. {int(found.sum())}/{n} loc · cold {n_cold}/re-lock {n_relock}/follow {n_follow} "
        f"· no_fp {int(no_fp.sum())} · handoffs {int((handoff != '').sum())} · res {res_rng}"
    )
    if found.any() and volume not in np.unique(vol_tracked[found]):
        print(
            f"  ** never locked in the requested volume {volume} "
            f"(locked in {list(np.unique(vol_tracked[found]))}) — is volume correct?"
        )
    print(flush=True)
    return outputs()
