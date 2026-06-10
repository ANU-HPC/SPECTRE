# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT
"""Track the magnetic axis and one island fixed-point chain across equilibria.

The algorithm lives in :class:`FixedPointTracker`, a stateful stepper whose ``step(specout, i)``
processes one equilibrium (a ``SPECTREout``) and records the axis and island fixed point, the step
displacement (the geometric convergence metric), the robust O/X verdict, an island-size proxy,
timings, and how the point was acquired. The carry-state (anchor, previous surface, drift) lives on
the instance, so successive ``step`` calls follow the same island across a sequence of equilibria.

Two drivers feed it the same algorithm:

  * :func:`track_islands_and_axis` (here) — **post-processing**: replays ``x_<a>_<b>.npy``
    checkpoints from disk, re-solves the field (``force_real``), and calls ``step`` per checkpoint.
  * ``spectre.force_minimizer`` — **in-loop**: builds a tracker once and calls ``step`` on the live,
    already-solved in-memory field each optimisation iteration (no disk, no re-solve).

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


class FixedPointTracker:
    """Stateful tracker for the magnetic axis and one island fixed-point chain.

    Construct once with the resonance (``mpol``, ``volume``, optional ``target_ntor``) and either a
    ``guess`` ``(R, Z)`` seed or ``discover=True``; then call :meth:`step` on each equilibrium in
    sequence. Per-checkpoint diagnostics accumulate in pre-allocated arrays of length ``n`` (indexed
    by the ``i`` passed to :meth:`step`), exposed as attributes and assembled by :meth:`result_dict`.

    The tuning knobs mirror :func:`track_islands_and_axis`; see that function's docstring for their
    meaning.
    """

    def __init__(
        self,
        mpol: int,
        volume: int,
        n: int,
        *,
        guess: Sequence[float] | None = None,
        discover: bool = False,
        target_ntor: int | None = None,
        discover_volumes: Sequence[int] | None = None,
        discover_n_theta: int | None = None,
        rediscover_after: int = 5,
        ds_local: float = 0.05,
        follow_ds: float | None = None,
        follow_ds_max: float = 0.15,
        n_s: int = 31,
        disc_n_s: int = 121,
        relock_ds: float = 0.12,
        relock_n_s: int = 31,
        relock_dtheta: float = 0.6,
        relock_n_theta: int = 7,
        dtheta: float = 0.3,
        n_theta: int = 7,
        follow_window: float | None = None,
        max_jump_floor: float = 0.02,
        handoff_margin: float = 0.05,
        no_handoff: bool = False,
        root_maxfev: int = 100,
        root_rtol: float = 1e-8,
        root_ftol: float = 1e-6,
        scan_rtol: float = 1e-6,
        iota_rtol: float = 0.01,
        dedup_tol: float = 1e-3,
        near_parabolic_margin: float = 10.0,
        classify_rtol: float = 1e-12,
        fast_classify: bool = False,
        max_axis_repeats: int = 8,
        verbose: bool = False,
    ):
        if guess is None and not discover:
            raise ValueError("guess (R, Z) is required unless discover=True.")
        # A genuine period-mpol chain has iota in lowest terms, so the target must be coprime with mpol.
        if target_ntor is not None and int(np.gcd(target_ntor, mpol)) != 1:
            g = int(np.gcd(target_ntor, mpol))
            raise ValueError(
                f"target_ntor {target_ntor} is not coprime with mpol {mpol} (gcd={g}); that is a "
                f"period-{mpol // g} chain, not period-{mpol}; pass a coprime mpol/target_ntor."
            )

        # SPECTRE imports are deferred to construction time (heavy core; keeps module import cheap).
        from spectre.fixed_point import (
            _return_map,
            classify_residue,
            find_axis,
            root_fixed_point,
            scan_fixed_points,
        )
        from spectre.fixed_point_classification import FixedPointClassifier, classify_point

        self._return_map = _return_map
        self._classify_residue = classify_residue
        self._find_axis = find_axis
        self._root_fixed_point = root_fixed_point
        self._scan_fixed_points = scan_fixed_points
        self._FixedPointClassifier = FixedPointClassifier
        self._classify_point = classify_point

        # ── Config ──────────────────────────────────────────────────────────────
        self.mpol = mpol
        self.volume = volume
        self.guess = guess
        self.discover = discover
        self.target_ntor = target_ntor
        self.discover_volumes = discover_volumes
        self.rediscover_after = rediscover_after
        self.ds_local = ds_local
        self.follow_ds = follow_ds
        self.follow_ds_max = follow_ds_max
        self.n_s = n_s
        self.disc_n_s = disc_n_s
        self.relock_ds = relock_ds
        self.relock_n_s = relock_n_s
        self.relock_dtheta = relock_dtheta
        self.relock_n_theta = relock_n_theta
        self.dtheta = dtheta
        self.n_theta = n_theta
        self.max_jump_floor = max_jump_floor
        self.handoff_margin = handoff_margin
        self.no_handoff = no_handoff
        self.root_rtol = root_rtol
        self.root_ftol = root_ftol
        self.iota_rtol = iota_rtol
        self.dedup_tol = dedup_tol
        self.near_parabolic_margin = near_parabolic_margin
        self.classify_rtol = classify_rtol
        self.fast_classify = fast_classify
        self.max_axis_repeats = max_axis_repeats
        self.verbose = verbose

        # ── Derived ─────────────────────────────────────────────────────────────
        self.n_transits = (
            mpol + 1
        )  # trace_one_fieldline logs phi=0 at t=0 too, so the mpol-fold map needs mpol+1
        # The poloidal discovery scan must resolve the mpol islands around the circle: sample
        # 2*mpol+1 theta (just above the 2*mpol Nyquist floor).
        self.disc_n_theta = discover_n_theta or max(2 * mpol + 1, 16)
        self.root_kw = dict(maxfev=root_maxfev, rtol=root_rtol, atol=root_rtol, ftol=root_ftol)
        # The scan only LOCATES residual minima to seed the polish, so it traces looser than the
        # 1e-8 polish (benchmarked ~2.6x faster at 1e-6, same candidate set within 0.17 mm).
        self.scan_rtol = scan_rtol
        self.nfp = -1  # learned from the first equilibrium
        # ``follow_window`` is the live continuity-gate half-width [m]; AUTO-derived from the island
        # geometry when the user left it None. ``_follow_window_user`` records the original request.
        self._follow_window_user = follow_window
        self.follow_window = follow_window
        self.clf_cache: dict = {}  # per-checkpoint FixedPointClassifier per volume (reset each step)

        # ── Carry-state (persists across step calls — the heart of the follow) ───
        self.cur_vol = volume
        self.anchor_rz: np.ndarray | None = None if discover else np.array(guess, dtype=float)
        self.anchor_th: float | None = None
        self.prev_axis: np.ndarray | None = None
        self.prev_island: np.ndarray | None = None
        self.prev_s: float | None = None
        self.prev_ntor: int | None = target_ntor
        self.last_member_th: float | None = None
        # O/X kind of the chain, pinned at the first successful lock; re-locks and follows are held
        # to it so a lost O-point never silently re-acquires (or drifts onto) the adjacent X-point.
        self.tracked_kind: str | None = None
        self.prev_disp = np.nan
        self.n_since_lock = 0

        # ── Pre-allocate every per-checkpoint diagnostic ─────────────────────────
        self.n = n
        nan = lambda shape=n: np.full(shape, np.nan)  # noqa: E731
        self.vol_tracked = np.full(n, volume, dtype=int)
        self.axis_rz, self.axis_disp = nan((n, 2)), nan()
        self.island_rz, self.island_sth = nan((n, 2)), nan((n, 2))
        self.residual, self.residual_scan, self.polished = nan(), nan(), np.zeros(n, dtype=bool)
        self.iota, self.ntor, self.iota_far = nan(), np.zeros(n, dtype=int), np.zeros(n, dtype=bool)
        self.iota_rel = nan()
        self.kind = np.full(n, "", dtype="<U10")
        self.residue, self.rtrace, self.rdet, self.rdet_err = nan(), nan(), nan(), nan()
        self.residue_stable = np.zeros(n, dtype=bool)
        self.disc, self.classify_method, self.orbit_growth = nan(), np.full(n, "", dtype="<U8"), nan()
        self.in_window, self.select_dist = np.zeros(n, dtype=bool), nan()
        self.displacement, self.disp_s = nan(), nan()
        self.best_rz, self.best_sth, self.best_residual = nan((n, 2)), nan((n, 2)), nan()
        self.best_nfev, self.best_converged = np.zeros(n, dtype=int), np.zeros(n, dtype=bool)
        self.width_s, self.width_R = nan(), nan()
        self.width_floored = np.zeros(n, dtype=bool)
        self.n_cand_total, self.n_cand_inwin = np.zeros(n, dtype=int), np.zeros(n, dtype=int)
        self.n_iota_pass = np.zeros(n, dtype=int)
        self.end_pinned_s = np.full(n, "", dtype="<U6")
        self.leaving, self.no_fp = np.full(n, "", dtype="<U6"), np.zeros(n, dtype=bool)
        self.rediscovered = np.zeros(n, dtype=bool)
        self.provenance = np.full(n, "", dtype="<U10")  # how the FP was acquired: cold/re-lock/follow
        self.handoff = np.full(n, "", dtype="<U12")
        self.seed_sth = nan((n, 2))
        self.t_total, self.t_recon, self.t_axis = nan(), nan(), nan()
        self.t_scan, self.t_polish, self.t_residue = nan(), nan(), nan()

    # ──────────────────────────────────────────────────────────────────────────
    def find_fp(
        self,
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
            cands, end_pinned = self._scan_fixed_points(
                specout,
                vol,
                self.n_transits,
                theta_grid=theta_grid,
                s_range=(s_lo, s_hi),
                n_s=(n_s_scan or self.n_s),
                method="RK45",
                rtol=self.scan_rtol,
                atol=self.scan_rtol,
            )
        except Exception as exc:
            print(f"      scan failed (vol {vol}) — {exc}", flush=True)
            return ([] if return_all else None), {}, None, {"n_cands": 0, "n_rm": 0, "n_iota": 0}
        confirmed, best_polish, n_rm = [], None, 0
        for c in sorted(cands, key=lambda c: c["residual"]):
            try:
                pol = self._root_fixed_point(
                    specout, vol, self.n_transits, (c["s"], c["theta"]), **self.root_kw
                )
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
                np.array([pol.s, pol.theta]), vol, num_phi_planes=1, num_transits=self.n_transits
            )
            if flag:
                continue
            iota_emp = -float(iota_tr)  # physical (VMEC) sign convention
            ntor_p = int(round(abs(iota_emp) * self.mpol)) if np.isfinite(iota_emp) else 0
            ok = (
                ntor_p > 0
                and ntor_p % self.nfp == 0
                and int(np.gcd(ntor_p, self.mpol)) == 1
                and (require_ntor is None or ntor_p == require_ntor)
            )
            if not ok:
                continue
            rel = abs(abs(iota_emp) - ntor_p / self.mpol) / (ntor_p / self.mpol)
            if rel > self.iota_rtol:
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
            if all(np.hypot(e["R"] - k["R"], e["Z"] - k["Z"]) > self.dedup_tol for k in deduped):
                deduped.append(e)
        return (
            (deduped if return_all else None),
            end_pinned,
            best_polish,
            {"n_cands": len(cands), "n_rm": n_rm, "n_iota": len(deduped)},
        )

    # ──────────────────────────────────────────────────────────────────────────
    def island_width(self, specout, vol, s0, th0):
        """Estimate the island radial size as the return-map residual-basin half-width.

        Adaptive: grow the window until the small-residual basin is bracketed on both sides, up to a
        cap, then convert the half-width in ``s`` to ``dR`` via local ``dR/ds``. Returns
        ``(width_s, width_R, floored)`` — NaN widths where the basin can't be cleanly bracketed, and
        ``(NaN, NaN, True)`` where it brackets only at the grid-resolution floor.
        """
        cap, half = min(0.6, 20.0 * self.ds_local), max(4.0 * self.ds_local, 0.15)
        w_s, floored = np.nan, False
        while True:
            npts = max(41, int(2.0 * half / 7.5e-3))  # ~constant s-resolution as the window grows
            sg = np.linspace(max(-_S_CLAMP, s0 - half), min(_S_CLAMP, s0 + half), npts)
            res = np.full(npts, np.nan)
            for j, s in enumerate(sg):
                with contextlib.suppress(Exception):
                    rm = self._return_map(
                        [s, th0],
                        specout,
                        vol,
                        self.n_transits,
                        method="RK45",
                        rtol=self.root_rtol,
                        atol=self.root_rtol,
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

    # ──────────────────────────────────────────────────────────────────────────
    def classify_fp(self, specout, vol, r, z):
        """Robust O/X of a located fixed point (reuses a warmed classifier; None on failure)."""
        if self.fast_classify:
            try:
                info = self._classify_residue(specout, vol, r, z, self.n_transits)
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
            clf = self.clf_cache.get(vol)
            if clf is None:
                clf = self._FixedPointClassifier(
                    specout, vol, self.mpol, method="DOP853", rtol=self.classify_rtol,
                    atol=self.classify_rtol,
                )
                self.clf_cache[vol] = clf
            return self._classify_point(
                clf, (r, z), near_parabolic_margin=self.near_parabolic_margin
            )
        except Exception as exc:
            print(f"    classify error - {exc}", flush=True)
            return None

    # ──────────────────────────────────────────────────────────────────────────
    def _pick_relock_member(self, specout, confirmed, lost_rz, tol):
        """Nearest accepted member to the lost anchor, preferring the tracked O/X kind, inside ``tol``.

        Classifies candidates nearest-first and returns the first whose kind matches the pinned
        ``tracked_kind`` — so a lost O-point re-locks the O-point, never the spatially-adjacent
        X-point of the same chain. Stops once outside the continuity window. Returns
        ``(v, entry, dist, classify_info)`` or ``None`` when no member of the tracked kind lies inside
        the window (the caller then reports no_fp rather than hopping to the wrong topology).
        """
        ranked = sorted(
            confirmed, key=lambda t: np.hypot(t[2]["R"] - lost_rz[0], t[2]["Z"] - lost_rz[1])
        )
        for _, v, e in ranked:
            dist = np.hypot(e["R"] - lost_rz[0], e["Z"] - lost_rz[1])
            if dist > tol:
                break  # ranked ascending — nothing closer remains inside the window
            if self.tracked_kind is None:
                return v, e, dist, None
            ci = self.classify_fp(specout, v, e["R"], e["Z"])
            if ci is not None and ci["kind"] == self.tracked_kind:
                return v, e, dist, ci
        return None

    # ──────────────────────────────────────────────────────────────────────────
    def log_row(self, i, ctx, *, it=None, done=None, n=None, elapsed=None):
        """Print the per-checkpoint verbose block for row ``i``.

        Shared by the post-processing driver (:func:`track_islands_and_axis`) and the in-loop
        ``Force_Minimizer`` caller, so both render identically. ``done``/``n`` add a progress count
        to the header; the timing line is printed only when ``elapsed`` is given (driver context) —
        in-loop tracking pays no reconstruction and sets no driver timing, so it is skipped there.
        """
        it_lbl = i if it is None else it
        rax, zax, warns = ctx["rax"], ctx["zax"], ctx["warns"]
        hdr = f"\n[iter {it_lbl:3d}]"
        if done is not None and n is not None:
            hdr += f"  {done}/{n}"
        hdr += f"  vol{self.cur_vol}"
        print(hdr)
        print(
            f"  axis     R={rax:.5f} Z={zax:+.5f}            {_D} "
            f"{_fmt(self.axis_disp[i] * 1e3, 6, 2)} mm"
        )
        if ctx["no_fp"]:
            for w in warns:
                print(f"  {_WARN} {w}")
            print(f"  {_WARN} no_fp   {ctx['nofp_reason']}")
        else:
            r_sel, z_sel = self.island_rz[i]
            kindword = "O-point (elliptic)" if self.kind[i] == "o" else "X-point (hyperbolic)"
            if self.classify_method[i] == "orbit":
                how = f"by orbit-topology (near-parabolic, growth {self.orbit_growth[i]:.1f}{_TIMES})"
            elif self.classify_method[i]:
                how = f"by Greene residue (R={self.residue[i]:.1e}, |det-1|={self.rdet_err[i]:.0e})"
            else:
                how, kindword = "classification unavailable", "?"
            disp_mm = self.displacement[i] * 1e3 if np.isfinite(self.displacement[i]) else np.nan
            gate_ok = _OK if self.residual[i] < self.root_ftol else _NO
            iota_ok = _OK if self.iota_rel[i] < self.iota_rtol else _NO
            print(
                f"  island   R={r_sel:.4f} Z={z_sel:+.4f}             {_D} "
                f"{_fmt(disp_mm, 6, 2)} mm"
            )
            print(f"  acquire  {ctx['acq_msg']}")
            for w in warns:
                print(f"  {_WARN} {w}")
            print(
                f"  gates    FP res {self.residual[i]:.1e} < {self.root_ftol:.0e} {gate_ok}   "
                f"iota {self.iota[i]:+.4f} = {self.ntor[i]}/{self.mpol}  rel {self.iota_rel[i]:.1e} < "
                f"{self.iota_rtol:.0e} {iota_ok}"
            )
            print(f"  class    {kindword} {_DASH} {how}")
        if elapsed is not None:
            print(
                f"  timing   recon {_fmt(self.t_recon[i], 5, 0)}s  scan {_fmt(self.t_scan[i], 5, 0)}s  "
                f"total {_fmt(self.t_total[i], 5, 0)}s  | {elapsed / 60:.1f}m elapsed",
                flush=True,
            )
        else:
            print("", end="", flush=True)   # flush the block even without the timing line (in-loop)

    def step(self, specout, i):
        """Process one equilibrium ``specout``, recording diagnostics into row ``i``.

        Updates the carry-state (anchor, previous surface/island, drift) so the next call follows the
        same chain. Driver-side timings (``t_recon``, ``t_total``) and verbose printing are left to
        the caller. Returns a small context dict for the caller's verbose block:
        ``{rax, zax, acq_msg, warns, nofp_reason, no_fp}``.
        """
        self.n_since_lock += 1
        warns: list[str] = []  # verbose ⚠ exception lines for this checkpoint
        # Drop a stale anchor so discovery re-triggers instead of scanning a dead point forever.
        if self.discover and self.anchor_rz is not None and self.n_since_lock > self.rediscover_after:
            warns.append(
                f"stale anchor ({self.n_since_lock - 1} checkpoints w/o lock) — re-discovering"
            )
            self.anchor_rz = self.anchor_th = None

        if self.nfp < 0:
            self.nfp = int(specout.input.physics.Nfp)

        # 2) Magnetic axis. Failure leaves the row NaN, does not abort.
        ta = perf_counter()
        rax = zax = np.nan
        try:
            rax, zax, _, _ = self._find_axis(specout, max_repeats=self.max_axis_repeats, verbose=False)
            self.axis_rz[i] = [rax, zax]
            if self.prev_axis is not None:
                self.axis_disp[i] = np.linalg.norm(self.axis_rz[i] - self.prev_axis)
            self.prev_axis = self.axis_rz[i].copy()
        except Exception as exc:
            warns.append(f"axis finding failed — {exc}")
        self.t_axis[i] = perf_counter() - ta

        # 3) Acquire the fixed point: anchor-follow (narrow scan) else discovery (full-volume).
        fp = None
        from_discovery = False
        do_discover = self.discover and self.anchor_rz is None
        lost_rz = None
        end_pinned: dict = {}
        best_diag = None
        committed_class = None
        acq_msg = ""
        self.clf_cache.clear()
        self.t_scan[i] = 0.0

        if self.anchor_rz is not None:
            # Anchor (R,Z) -> (s,theta) on THIS field; hand off to a neighbour volume only with
            # hysteresis (the interface itself moves, so an island near a boundary would flap).
            seed_s = seed_th = None
            for v in (self.cur_vol, self.cur_vol - 1, self.cur_vol + 1):
                if v < 1:
                    continue
                try:
                    cs, ct = specout.get_st_coords(v, self.anchor_rz[0], self.anchor_rz[1], 0.0)
                    cand_s = float(np.asarray(cs).ravel()[0])
                    cand_th = float(np.asarray(ct).ravel()[0])
                except Exception:
                    continue
                if v != self.cur_vol:
                    if self.no_handoff or abs(cand_s) >= _S_CLAMP - self.handoff_margin:
                        continue  # not clearly inside the neighbour — don't flap
                    self.handoff[i] = f"vol{self.cur_vol}->vol{v}"
                    warns.append(f"handoff {self.handoff[i]} (s={cand_s:+.3f}, inside margin)")
                    self.cur_vol = v
                seed_s, seed_th = cand_s, cand_th
                break
            if seed_s is not None:
                # Re-centre theta on the previous member — but NOT across a handoff, where
                # the fresh inversion theta in the new volume is the trustworthy seed.
                if self.anchor_th is not None and not self.handoff[i]:
                    seed_th = self.anchor_th
                # Centre the scan on prev_s (the committed surface predicts the next far better
                # than the noisy (R,Z)->s inversion). Size the window from the measured drift,
                # but distrust an implausibly large inversion and always cap at follow_ds_max.
                use_prev = self.prev_s is not None and np.isfinite(self.prev_s) and not self.handoff[i]
                seed_s_used = self.prev_s if use_prev else seed_s
                if self.follow_ds is not None:
                    follow_ds_i = self.follow_ds
                elif use_prev:
                    gap = abs(seed_s - self.prev_s)
                    if gap > self.follow_ds_max:
                        follow_ds_i = self.ds_local
                        warns.append(
                            f"distrust inversion s={seed_s:+.4f} vs prev_s={self.prev_s:+.4f} "
                            f"off by {gap:.3f} -> box=ds_local"
                        )
                    else:
                        follow_ds_i = min(max(self.ds_local, 2.0 * gap), self.follow_ds_max)
                else:
                    follow_ds_i = self.ds_local
                geom_ceiling = self.follow_window if self.follow_window is not None else 0.30
                drift_gate = (
                    self.max_jump_floor
                    if not np.isfinite(self.prev_disp)
                    else max(self.max_jump_floor, 10.0 * self.prev_disp)
                )
                max_move = min(geom_ceiling, drift_gate)
                self.seed_sth[i] = [seed_s_used, seed_th]
                th_grid = (
                    (float(seed_th),)
                    if self.n_theta <= 1 or self.dtheta <= 0
                    else np.linspace(seed_th - self.dtheta, seed_th + self.dtheta, self.n_theta)
                )
                ts = perf_counter()
                fp, end_pinned, bp, st = self.find_fp(
                    specout,
                    self.cur_vol,
                    seed_s_used - follow_ds_i,
                    seed_s_used + follow_ds_i,
                    th_grid,
                    anchor_rz=self.anchor_rz,
                    max_move=max_move,
                    require_ntor=self.prev_ntor,
                )
                self.n_cand_total[i] = st["n_cands"]
                if bp is not None and (
                    best_diag is None or bp["residual"] < best_diag["residual"]
                ):
                    best_diag = bp
                self.t_scan[i] += perf_counter() - ts
                acq_msg = (
                    f"[follow] tracked from previous anchor ({self.n_cand_total[i]} cand, "
                    f"window {max_move * 1e3:.0f} mm)"
                )
                # Hold the follow to the pinned topology: continuity alone can slide onto the
                # adjacent X-point when O and X are close, so reject a wrong-kind landing and let
                # the re-lock below re-acquire the right one. The classify here is reused at commit
                # (committed_class), so this costs no extra trace on a clean follow.
                if fp is not None and self.tracked_kind is not None:
                    ci_follow = self.classify_fp(specout, self.cur_vol, fp["R"], fp["Z"])
                    if ci_follow is not None and ci_follow["kind"] != self.tracked_kind:
                        warns.append(
                            f"follow landed on {ci_follow['kind'].upper()}-point but tracking "
                            f"{self.tracked_kind.upper()}-point — re-locking"
                        )
                        fp = None
                    else:
                        committed_class = ci_follow
            if fp is None and self.discover:
                # Anchor-follow lost the member -> re-discover, but remember where it was so
                # discovery re-locks THE SAME island (nearest), not a far member.
                lost_rz = self.anchor_rz.copy()
                do_discover = True

        if fp is None and do_discover:
            # FP-first: scan, polish, keep iota-gate passers. A re-lock (we know roughly where the
            # member was) localizes the scan to a box around the lost surface/member at discovery-
            # grade s-density — far cheaper than the whole-volume sweep a cold start needs.
            relock = lost_rz is not None and self.prev_s is not None and np.isfinite(self.prev_s)
            if relock:
                vols = [self.cur_vol]
                s_lo, s_hi = self.prev_s - self.relock_ds, self.prev_s + self.relock_ds
                n_s_scan = self.relock_n_s
                th_full = (
                    np.linspace(
                        self.last_member_th - self.relock_dtheta,
                        self.last_member_th + self.relock_dtheta,
                        self.relock_n_theta,
                    )
                    if self.last_member_th is not None and self.relock_dtheta > 0
                    else np.linspace(0.0, 2.0 * np.pi, self.disc_n_theta, endpoint=False)
                )
            else:
                vols = list(self.discover_volumes) if self.discover_volumes else [self.volume]
                s_lo, s_hi, n_s_scan = -_S_CLAMP, _S_CLAMP, self.disc_n_s
                th_full = np.linspace(0.0, 2.0 * np.pi, self.disc_n_theta, endpoint=False)
            confirmed = []  # (|v - volume|, v, entry) across all searched volumes
            ts = perf_counter()
            for v in vols:
                confs, _ep, bp, st = self.find_fp(
                    specout,
                    v,
                    s_lo,
                    s_hi,
                    th_full,
                    return_all=True,
                    require_ntor=self.prev_ntor,
                    n_s_scan=n_s_scan,
                )
                self.n_cand_total[i] += st["n_cands"]
                self.n_iota_pass[i] += st["n_iota"]
                if bp is not None and (
                    best_diag is None or bp["residual"] < best_diag["residual"]
                ):
                    best_diag = bp
                confirmed += [(abs(v - self.volume), v, e) for e in confs]
            confirmed.sort(
                key=lambda t: (t[0], t[2]["residual"])
            )  # nearest volume, then lowest residual
            n_members = len(confirmed)
            res_str = f"{self.prev_ntor}/{self.mpol}" if self.prev_ntor else f"k*Nfp (period-{self.mpol})"
            if lost_rz is not None:
                # Re-acquiring a tracked member: pick the nearest accepted member of the pinned
                # O/X kind within the continuity window — never hop, and never swap topology.
                tol = self.follow_window if self.follow_window is not None else 0.30
                pick = self._pick_relock_member(specout, confirmed, lost_rz, tol) if confirmed else None
                kind_str = f"{self.tracked_kind}-point " if self.tracked_kind else ""
                if pick is not None:
                    v, best, dist, ci = pick
                    fp, self.cur_vol, from_discovery, committed_class = best, v, True, ci
                    acq_msg = (
                        f"[re-lock] follow lost it -> {'localized ' if relock else ''}re-discovered "
                        f"{n_members} members; nearest {kind_str}{dist * 1e3:.1f} mm "
                        f"<= {tol * 1e3:.0f} mm window"
                    )
                else:
                    acq_msg = (
                        f"[no_fp] no {kind_str}member within {tol * 1e3:.0f} mm window "
                        f"({n_members} candidate{'s' if n_members != 1 else ''} found)"
                    )
            elif confirmed:
                # Cold start: prefer an O-point (island centre); order by member stability
                # (theta nearest the last member, else lowest residual) and commit the first O.
                if self.last_member_th is not None:

                    def wrap(th, _m=self.last_member_th):  # _m default binds the loop value
                        return abs((th - _m + np.pi) % (2 * np.pi) - np.pi)

                    order = sorted(
                        confirmed, key=lambda t: (wrap(t[2]["theta"]), t[2]["residual"])
                    )
                else:
                    order = confirmed
                for _, v, e in order:
                    ci = self.classify_fp(specout, v, e["R"], e["Z"])
                    if ci is not None and ci["kind"] == "o":
                        self.cur_vol, fp, committed_class = v, e, ci
                        acq_msg = f"[cold] {n_members} members on {res_str}; locked O-point"
                        break
                else:
                    _, self.cur_vol, fp = order[0]
                    committed_class = self.classify_fp(specout, self.cur_vol, fp["R"], fp["Z"])
                    acq_msg = (
                        f"[cold] discovered {n_members} members on {res_str}; "
                        f"no O found -> nearest-seed member"
                    )
                from_discovery = True
            self.t_scan[i] += perf_counter() - ts

        # -- Commit (polish-gated: fp set only if a Newton polish confirmed it) --
        self.vol_tracked[i] = self.cur_vol
        if fp is None:
            self.no_fp[i] = True
            self.leaving[i] = next((sd for sd in end_pinned.values() if sd), "")
            nofp_reason = "no polishable scan candidate"
            if best_diag is not None:
                self.best_rz[i] = [best_diag["R"], best_diag["Z"]]
                self.best_sth[i] = [best_diag["s"], best_diag["theta"]]
                self.best_residual[i] = best_diag["residual"]
                self.best_nfev[i] = best_diag["nfev"]
                self.best_converged[i] = best_diag["converged"]
                why = (
                    "passed ftol; failed iota/continuity"
                    if best_diag["converged"]
                    else "failed ftol gate"
                )
                nofp_reason = (
                    f"best polish {best_diag['residual']:.1e} @ s={best_diag['s']:+.4f} "
                    f"th={best_diag['theta']:.2f} ({why})"
                )
            return {
                "rax": rax,
                "zax": zax,
                "acq_msg": acq_msg,
                "warns": warns,
                "nofp_reason": nofp_reason,
                "no_fp": True,
            }

        s_sel, th_sel, r_sel, z_sel = fp["s"], fp["theta"], fp["R"], fp["Z"]
        self.residual[i], self.residual_scan[i], self.polished[i] = fp["residual"], fp["scan_res"], True
        self.in_window[i] = True
        self.island_rz[i] = [r_sel, z_sel]
        self.island_sth[i] = [s_sel, th_sel]
        self.best_rz[i], self.best_sth[i] = [r_sel, z_sel], [s_sel, th_sel]
        self.best_residual[i], self.best_nfev[i], self.best_converged[i] = (
            fp["residual"],
            int(fp.get("nfev", 0)),
            True,
        )

        # iota / ntor were measured at the fixed point by the iota gate — reuse (no re-trace).
        self.iota[i], self.ntor[i], self.iota_rel[i] = fp["iota"], fp["ntor"], fp["iota_rel"]
        self.iota_far[i] = bool(fp["iota_rel"] > 0.5 * self.iota_rtol)
        if self.iota_far[i]:
            warns.append(f"marginal: iota_rel {self.iota_rel[i]:.1e} in upper half of gate band")

        # Robust O/X classification (reuse discovery's verdict for this member if available).
        tr = perf_counter()
        ci = (
            committed_class
            if committed_class is not None
            else self.classify_fp(specout, self.cur_vol, r_sel, z_sel)
        )
        if ci is not None:
            if self.tracked_kind is None:
                self.tracked_kind = ci["kind"]  # pin the chain topology at the first lock
            self.kind[i], self.residue[i] = ci["kind"], ci["residue"]
            self.rtrace[i], self.rdet[i] = ci["trace"], ci["det"]
            self.rdet_err[i], self.residue_stable[i] = ci["det_err"], ci["stable"]
            self.disc[i], self.classify_method[i], self.orbit_growth[i] = (
                ci["disc"],
                ci["method"],
                ci["growth"],
            )
        self.t_residue[i] = perf_counter() - tr

        try:
            self.width_s[i], self.width_R[i], self.width_floored[i] = self.island_width(
                specout, self.cur_vol, s_sel, th_sel
            )
        except Exception as exc:
            warns.append(f"width proxy error - {exc}")

        # Auto-derive the continuity gate from the island geometry (unless fixed): the mpol
        # islands lie at minor radius r = ||FP-axis||, so 0.35*2*pi*r/mpol ~ 0.7x O-X spacing.
        if self._follow_window_user is None and np.isfinite(self.axis_rz[i, 0]):
            r = np.hypot(r_sel - self.axis_rz[i, 0], z_sel - self.axis_rz[i, 1])
            self.follow_window = 0.35 * 2.0 * np.pi * r / self.mpol

        # Displacement is the convergence metric, valid only across an anchor-follow step (a
        # handoff still follows the same member). ANY discovery-sourced lock breaks continuity
        # (the nearest member is not guaranteed to be the same one), so leave displacement NaN
        # — which also stops a falsified re-lock distance from poisoning prev_disp.
        self.rediscovered[i] = from_discovery
        continuity_broken = from_discovery
        self.provenance[i] = (
            ("cold" if lost_rz is None else "re-lock") if from_discovery else "follow"
        )
        rz_valid = not continuity_broken
        s_valid = (not continuity_broken) and (not self.handoff[i])
        if self.prev_island is not None and rz_valid:
            self.displacement[i] = np.linalg.norm(
                self.island_rz[i] - self.prev_island
            )  # (R,Z) survives a handoff
        if self.prev_s is not None and np.isfinite(self.prev_s) and s_valid:
            self.disp_s[i] = s_sel - self.prev_s
        self.prev_island = self.island_rz[i].copy()
        self.prev_s = s_sel
        self.prev_ntor = int(self.ntor[i])
        self.last_member_th = th_sel
        self.prev_disp = self.displacement[i] if np.isfinite(self.displacement[i]) else self.prev_disp
        self.anchor_rz = self.island_rz[i].copy()
        self.anchor_th = th_sel
        self.n_since_lock = 0

        return {
            "rax": rax,
            "zax": zax,
            "acq_msg": acq_msg,
            "warns": warns,
            "nofp_reason": "",
            "no_fp": False,
        }

    # ──────────────────────────────────────────────────────────────────────────
    def result_dict(self, iter_indices, start_at_iter, stride, reverse) -> dict:
        """Assemble the per-checkpoint arrays + scalar metadata into the output ``.npz`` schema."""
        return {
            "iteration_indices": iter_indices,
            "vol_tracked": self.vol_tracked,
            "axis_rz": self.axis_rz,
            "axis_disp": self.axis_disp,
            "island_rz": self.island_rz,
            "island_sth": self.island_sth,
            "residual": self.residual,
            "residual_scan": self.residual_scan,
            "polished": self.polished,
            "iota": self.iota,
            "ntor": self.ntor,
            "iota_far": self.iota_far,
            "iota_rel": self.iota_rel,
            "kind": self.kind,
            "residue": self.residue,
            "trace": self.rtrace,
            "det": self.rdet,
            "det_err": self.rdet_err,
            "residue_stable": self.residue_stable,
            "disc": self.disc,
            "classify_method": self.classify_method,
            "orbit_growth": self.orbit_growth,
            "in_window": self.in_window,
            "select_dist": self.select_dist,
            "displacement": self.displacement,
            "disp_s": self.disp_s,
            "best_rz": self.best_rz,
            "best_sth": self.best_sth,
            "best_residual": self.best_residual,
            "best_nfev": self.best_nfev,
            "best_converged": self.best_converged,
            "width_s": self.width_s,
            "width_R": self.width_R,
            "width_floored": self.width_floored,
            "n_candidates": self.n_cand_total,
            "n_candidates_inwin": self.n_cand_inwin,
            "n_iota_pass": self.n_iota_pass,
            "end_pinned": self.end_pinned_s,
            "leaving": self.leaving,
            "no_fp": self.no_fp,
            "rediscovered": self.rediscovered,
            "provenance": self.provenance,
            "handoff": self.handoff,
            "seed_sth": self.seed_sth,
            "t_total": self.t_total,
            "t_field_reconstruction": self.t_recon,
            "t_axis": self.t_axis,
            "t_scan": self.t_scan,
            "t_polish": self.t_polish,
            "t_residue": self.t_residue,
            "mpol": self.mpol,
            "volume_start": self.volume,
            "nfp": self.nfp,
            "start_at_iter": int(start_at_iter),
            "stride": int(stride),
            "n_s": int(self.n_s),
            "disc_n_s": int(self.disc_n_s),
            "n_theta": int(self.n_theta),
            "dtheta": float(self.dtheta),
            "root_ftol": float(self.root_ftol),
            "scan_rtol": float(self.scan_rtol),
            "iota_rtol": float(self.iota_rtol),
            "dedup_tol": float(self.dedup_tol),
            "near_parabolic_margin": float(self.near_parabolic_margin),
            "classify_rtol": float(self.classify_rtol),
            "fast_classify": bool(self.fast_classify),
            "target_ntor": int(self.target_ntor) if self.target_ntor else -1,
            "follow_window": float(self.follow_window) if self.follow_window is not None else -1.0,
            "reverse": bool(reverse),
            "discover": bool(self.discover),
            "guess_rz": np.array([np.nan, np.nan] if self.guess is None else self.guess, dtype=float),
        }


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
    relock_ds: float = 0.12,
    relock_n_s: int = 31,
    relock_dtheta: float = 0.6,
    relock_n_theta: int = 7,
    dtheta: float = 0.3,
    n_theta: int = 7,
    follow_window: float | None = None,
    max_jump_floor: float = 0.02,
    handoff_margin: float = 0.05,
    no_handoff: bool = False,
    root_maxfev: int = 100,
    root_rtol: float = 1e-8,
    root_ftol: float = 1e-6,
    scan_rtol: float = 1e-6,
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

    Post-processing driver around :class:`FixedPointTracker`: replays the ``x_<a>_<b>.npy``
    checkpoints, re-solves the field from disk, and calls ``tracker.step`` per checkpoint. Either
    pass ``guess`` (an ``(R, Z)`` seed for the first checkpoint) or ``discover=True`` to self-seed by
    an FP-first scan filtered to the target resonance.

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
        relock_ds: Half-width in ``s`` of the localized re-lock scan, centred on the lost surface.
        relock_n_s: ``s`` samples per theta for the localized re-lock scan.
        relock_dtheta: Half-width in theta of the localized re-lock scan (0 = full 2pi).
        relock_n_theta: theta samples for the localized re-lock scan.
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
        scan_rtol: Integrator tolerance of the candidate-locating scan (RK45). Looser than ``root_rtol``
            since the scan only seeds the polish; ``1e-6`` is ~2.6x faster than ``1e-8`` with the same
            candidate set.
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
    from spectre.force_targets import force_real
    from spectre.utils import get_xinit_specwrap, unpack_interfaces

    xs_fixb_dir, toml_path = Path(xs_fixb_dir), Path(toml_path)
    if not xs_fixb_dir.is_dir():
        raise FileNotFoundError(f"xs_fixb directory not found: {xs_fixb_dir}")
    if not toml_path.is_file():
        raise FileNotFoundError(f"TOML file not found: {toml_path}")

    out = None if checkpoint_path is None else Path(checkpoint_path)
    stride = max(1, int(stride))

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

    # ── Build the tracker (owns the algorithm + per-checkpoint diagnostics) ─────
    tk = FixedPointTracker(
        mpol,
        volume,
        n,
        guess=guess,
        discover=discover,
        target_ntor=target_ntor,
        discover_volumes=discover_volumes,
        discover_n_theta=discover_n_theta,
        rediscover_after=rediscover_after,
        ds_local=ds_local,
        follow_ds=follow_ds,
        follow_ds_max=follow_ds_max,
        n_s=n_s,
        disc_n_s=disc_n_s,
        relock_ds=relock_ds,
        relock_n_s=relock_n_s,
        relock_dtheta=relock_dtheta,
        relock_n_theta=relock_n_theta,
        dtheta=dtheta,
        n_theta=n_theta,
        follow_window=follow_window,
        max_jump_floor=max_jump_floor,
        handoff_margin=handoff_margin,
        no_handoff=no_handoff,
        root_maxfev=root_maxfev,
        root_rtol=root_rtol,
        root_ftol=root_ftol,
        scan_rtol=scan_rtol,
        iota_rtol=iota_rtol,
        dedup_tol=dedup_tol,
        near_parabolic_margin=near_parabolic_margin,
        classify_rtol=classify_rtol,
        fast_classify=fast_classify,
        max_axis_repeats=max_axis_repeats,
        verbose=verbose,
    )

    # ── Lean header (always printed) ───────────────────────────────────────────
    seed_str = "DISCOVER" if discover else f"guess{tuple(guess)}"
    ntor_str = f"{target_ntor} ({target_ntor}/{mpol})" if target_ntor else "any k*Nfp"
    gate_res = f"{target_ntor}/{mpol}" if target_ntor else "nearest k*Nfp/mpol"
    print("track_FP — island + axis tracker")
    print(f"  toml     : {toml_path}")
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

    iter_indices = np.array([it for it, _ in paths], dtype=int)

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

    try:
        for i, (it, npy) in enumerate(paths):
            if stop[0]:
                print(f"  Stopping before iter {it} (SIGUSR1).", flush=True)
                break
            t0 = perf_counter()

            # 1) Rebuild the equilibrium from the checkpoint.
            try:
                unpack_interfaces(np.load(str(npy)), obj)
                force_real(get_xinit_specwrap(obj), obj)
                specout = obj.create_spectreout()
            except Exception as exc:
                print(f"[iter {it:3d}]  ! checkpoint load FAILED — {exc}; skipping.", flush=True)
                continue
            tk.t_recon[i] = perf_counter() - t0

            # 2-3) Axis + island fixed point (the shared algorithm).
            ctx = tk.step(specout, i)

            tk.t_total[i] = perf_counter() - t0
            done = i + 1
            elapsed = float(np.nansum(tk.t_total[:done]))
            if verbose:
                tk.log_row(i, ctx, it=it, done=done, n=n, elapsed=elapsed)

            if out is not None:
                np.savez(str(out), **tk.result_dict(iter_indices, start_at_iter, stride, reverse))
    finally:
        signal.signal(signal.SIGUSR1, old_handler)

    # ── Summary ────────────────────────────────────────────────────────────────
    found = ~np.isnan(tk.island_rz[:, 0])
    n_cold = int(np.sum(tk.provenance == "cold"))
    n_relock = int(np.sum(tk.provenance == "re-lock"))
    n_follow = int(np.sum(tk.provenance == "follow"))
    res_rng = (
        f"{tk.residual[found].min():.1e}..{np.nanmax(tk.residual[found]):.1e}"
        if found.any()
        else "n/a"
    )
    print(
        f"\nDone. {int(found.sum())}/{n} loc · cold {n_cold}/re-lock {n_relock}/follow {n_follow} "
        f"· no_fp {int(tk.no_fp.sum())} · handoffs {int((tk.handoff != '').sum())} · res {res_rng}"
    )
    if found.any() and volume not in np.unique(tk.vol_tracked[found]):
        print(
            f"  ** never locked in the requested volume {volume} "
            f"(locked in {list(np.unique(tk.vol_tracked[found]))}) — is volume correct?"
        )
    print(flush=True)
    return tk.result_dict(iter_indices, start_at_iter, stride, reverse)
