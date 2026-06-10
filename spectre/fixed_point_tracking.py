# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT
"""Track the magnetic axis and one island fixed-point chain across equilibria.

The algorithm lives in :class:`FixedPointTracker`, a stateful stepper whose ``step(specout, i)``
processes one equilibrium (a ``SPECTREout``) and records the axis and island fixed point, the step
displacement (the geometric convergence metric), the robust O/X verdict, and how the point was
acquired. The carry-state (anchor, previous surface, drift) lives on the instance, so successive
``step`` calls follow the same island across a sequence of equilibria.

Two drivers feed it the same algorithm:

  * :func:`track_islands_and_axis` (here) — **post-processing**: replays ``x_<a>_<b>.npy``
    checkpoints from disk, re-solves the field (``force_real``), and calls ``step`` per checkpoint.
  * ``spectre.force_minimizer`` — **in-loop**: builds a tracker once and calls ``step`` on the live,
    already-solved in-memory field each optimisation iteration (no disk, no re-solve).

Built on the ``spectre.fixed_point`` primitives (``scan_fixed_points``, ``root_fixed_point``,
``find_axis``) and the robust O/X classifier in ``spectre.fixed_point_classification``. Two regimes,
the same two acceptance conditions in both:

  * **Discovery** (no anchor: cold start, or the follow just missed). FP-FIRST then iota-filter.
    Scan the whole radial extent of the host volume ``s in (-0.999, 0.999)`` at ``2*mpol+1`` theta,
    polish every candidate (return-map residual gate ``< ftol``), then keep only those whose
    empirical iota gives the target ``ntor = k*Nfp`` within ``iota_rtol``. Finding fixed points
    first and checking iota at them avoids the bias of measuring iota near a forming island.
    Discovery prefers an O-point (island centre) via the classifier.

  * **Anchor-follow** (have an anchor). Cheap narrow scan around the previous fixed point's
    ``(s, theta)`` in the fixed host volume. Two gates confirm "same island": ``(R,Z)`` continuity
    (drift-adaptive) and iota. A miss falls back to the full discovery scan above.

The tracker is deliberately simple: it tracks one chain in one volume, with no cross-volume
handoff, no localized re-lock, and no O/X kind-pinning — a chain that migrates across an interface
shows up as ``no_fp`` rows, and a drift onto the adjacent X-point stays visible in the recorded
``kind``/``residue`` rather than being auto-corrected.
"""

from __future__ import annotations

import os
import re
import signal
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter

import numpy as np

_XS_FIXB_RE = re.compile(r"^x_(\d+)_(\d+)\.npy$")  # x_<a>_<b>.npy; group(2)=<b>=iteration number
_S_CLAMP = 0.999  # hard clamp on the seed s-range; s in (-1, 1) spans one annular volume
_MAX_JUMP_FLOOR = 0.02  # (R,Z) continuity floor [m]; a candidate may move up to max(this, 10*prev_disp)

# Tuning constants (settled values; were keyword knobs once — kept fixed for simplicity).
FOLLOW_DS = 0.05  # anchor-follow box half-width in s
N_S = 31  # s samples per theta in a follow scan
DISC_N_S = 121  # s samples per theta in discovery; resonant dip width_s ~ 0.008 — do not coarsen
DTHETA = 0.3  # follow theta-grid half-width
N_THETA = 7  # follow theta samples
ROOT_MAXFEV = 100  # Newton-polish function-evaluation budget
ROOT_RTOL = 1e-8  # polish integrator tolerance
ROOT_FTOL = 1e-6  # return-map residual acceptance gate [m]
SCAN_RTOL = 1e-6  # candidate-locating scan tolerance (looser: only seeds the polish, ~2.6x faster)
DEDUP_TOL = 1e-3  # (R,Z) tolerance [m] for collapsing duplicate discovery members
NEAR_PARABOLIC_MARGIN = 10.0  # classifier trust gate (|disc| vs |det-1|)
CLASSIFY_RTOL = 1e-12  # O/X classifier integrator tolerance (must be tight near resonances)

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
        iota_rtol: float = 0.01,
        max_axis_repeats: int = 8,
        verbose: bool = False,
    ):
        if guess is None and not discover:
            discover = True  # no explicit guess -> self-seed by an FP-first discovery scan
        # A genuine period-mpol chain has iota in lowest terms, so the target must be coprime with mpol.
        if target_ntor is not None and int(np.gcd(target_ntor, mpol)) != 1:
            g = int(np.gcd(target_ntor, mpol))
            raise ValueError(
                f"target_ntor {target_ntor} is not coprime with mpol {mpol} (gcd={g}); that is a "
                f"period-{mpol // g} chain, not period-{mpol}; pass a coprime mpol/target_ntor."
            )

        # SPECTRE imports are deferred to construction time (heavy core; keeps module import cheap).
        from spectre.fixed_point import find_axis, root_fixed_point, scan_fixed_points
        from spectre.fixed_point_classification import FixedPointClassifier, classify_point

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
        self.iota_rtol = iota_rtol
        self.max_axis_repeats = max_axis_repeats
        self.verbose = verbose

        # ── Derived ─────────────────────────────────────────────────────────────
        self.n_transits = (
            mpol + 1
        )  # trace_one_fieldline logs phi=0 at t=0 too, so the mpol-fold map needs mpol+1
        # The poloidal discovery scan must resolve the mpol islands around the circle: sample
        # 2*mpol+1 theta (just above the 2*mpol Nyquist floor).
        self.disc_n_theta = max(2 * mpol + 1, 16)
        self.root_kw = dict(maxfev=ROOT_MAXFEV, rtol=ROOT_RTOL, atol=ROOT_RTOL, ftol=ROOT_FTOL)
        self.nfp = -1  # learned from the first equilibrium
        # ``follow_window`` is the live continuity-gate half-width [m]; always AUTO-derived from the
        # island geometry at each step (filled in by ``step``; None until the axis is known).
        self.follow_window = None
        self.clf_cache: dict = {}  # per-checkpoint FixedPointClassifier per volume (reset each step)

        # ── Carry-state (persists across step calls — the heart of the follow) ───
        self.anchor_rz: np.ndarray | None = None if discover else np.array(guess, dtype=float)
        self.anchor_th: float | None = None
        self.prev_axis: np.ndarray | None = None
        self.prev_island: np.ndarray | None = None
        self.prev_s: float | None = None
        self.prev_ntor: int | None = target_ntor
        self.last_member_th: float | None = None
        self.prev_disp = np.nan

        # ── Pre-allocate every per-checkpoint diagnostic ─────────────────────────
        self.n = n
        nan = lambda shape=n: np.full(shape, np.nan)  # noqa: E731
        self.vol_tracked = np.full(n, volume, dtype=int)  # constant; fp_metrics.npz reads it
        self.axis_rz, self.axis_disp = nan((n, 2)), nan()
        self.island_rz, self.island_sth = nan((n, 2)), nan((n, 2))
        self.residual = nan()
        self.iota, self.ntor, self.iota_rel = nan(), np.zeros(n, dtype=int), nan()
        self.kind = np.full(n, "", dtype="<U10")
        self.residue, self.rdet_err = nan(), nan()
        self.classify_method, self.orbit_growth = np.full(n, "", dtype="<U8"), nan()
        self.displacement = nan()
        self.provenance = np.full(n, "", dtype="<U10")  # how the FP was acquired: cold/follow
        self.no_fp = np.zeros(n, dtype=bool)
        self.t_total, self.t_recon, self.t_scan = nan(), nan(), nan()

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
        first. Returns ``(result, best_polish, stats)``. The ``(R,Z)`` proximity gate is applied
        only against ``anchor_rz`` (continuity), never the scan seed.
        """
        s_lo, s_hi = max(-_S_CLAMP, s_lo), min(_S_CLAMP, s_hi)
        try:
            cands, _ = self._scan_fixed_points(
                specout,
                vol,
                self.n_transits,
                theta_grid=theta_grid,
                s_range=(s_lo, s_hi),
                n_s=(n_s_scan or N_S),
                method="RK45",
                rtol=SCAN_RTOL,
                atol=SCAN_RTOL,
            )
        except Exception as exc:
            print(f"      scan failed (vol {vol}) — {exc}", flush=True)
            return ([] if return_all else None), None, {"n_cands": 0, "n_rm": 0, "n_iota": 0}
        confirmed, best_polish, n_rm, min_iota_rel = [], None, 0, None
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
            if min_iota_rel is None or rel < min_iota_rel:
                min_iota_rel = rel  # lowest relative iota error seen (reported if the gate rejects all)
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
                return entry, best_polish, {"n_cands": len(cands), "n_rm": n_rm, "n_iota": 1}
            confirmed.append(entry)
        # Collapse duplicates: the same FP is polished from several adjacent theta seeds. Keep the
        # lowest-residual member of each (R,Z) cluster so counts/selection see distinct members.
        deduped = []
        for e in sorted(confirmed, key=lambda e: e["residual"]):
            if all(np.hypot(e["R"] - k["R"], e["Z"] - k["Z"]) > DEDUP_TOL for k in deduped):
                deduped.append(e)
        if best_polish is not None:
            best_polish["iota_rel"] = min_iota_rel  # None if no candidate reached the iota check
        return (
            (deduped if return_all else None),
            best_polish,
            {"n_cands": len(cands), "n_rm": n_rm, "n_iota": len(deduped)},
        )

    # ──────────────────────────────────────────────────────────────────────────
    def classify_fp(self, specout, vol, r, z):
        """Robust O/X of a located fixed point (reuses a warmed classifier; None on failure)."""
        try:
            clf = self.clf_cache.get(vol)
            if clf is None:
                clf = self._FixedPointClassifier(
                    specout, vol, self.mpol, method="DOP853", rtol=CLASSIFY_RTOL,
                    atol=CLASSIFY_RTOL,
                )
                self.clf_cache[vol] = clf
            return self._classify_point(
                clf, (r, z), near_parabolic_margin=NEAR_PARABOLIC_MARGIN
            )
        except Exception as exc:
            print(f"    classify error - {exc}", flush=True)
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
        hdr += f"  vol{self.volume}"
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
            gate_ok = _OK if self.residual[i] < ROOT_FTOL else _NO
            iota_ok = _OK if self.iota_rel[i] < self.iota_rtol else _NO
            print(
                f"  island   R={r_sel:.4f} Z={z_sel:+.4f}             {_D} "
                f"{_fmt(disp_mm, 6, 2)} mm"
            )
            print(f"  acquire  {ctx['acq_msg']}")
            for w in warns:
                print(f"  {_WARN} {w}")
            print(
                f"  gates    FP res {self.residual[i]:.1e} < {ROOT_FTOL:.0e} {gate_ok}   "
                f"iota {self.iota[i]:+.4f} = {self.ntor[i]}/{self.mpol}  "
                f"relative iota error {self.iota_rel[i]:.1e} < "
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
        warns: list[str] = []  # verbose ⚠ exception lines for this checkpoint

        if self.nfp < 0:
            self.nfp = int(specout.input.physics.Nfp)

        # 2) Magnetic axis. Failure leaves the row NaN, does not abort.
        rax = zax = np.nan
        try:
            rax, zax, _, _ = self._find_axis(specout, max_repeats=self.max_axis_repeats, verbose=False)
            self.axis_rz[i] = [rax, zax]
            if self.prev_axis is not None:
                self.axis_disp[i] = np.linalg.norm(self.axis_rz[i] - self.prev_axis)
            self.prev_axis = self.axis_rz[i].copy()
        except Exception as exc:
            warns.append(f"axis finding failed — {exc}")

        # 3) Acquire the fixed point: anchor-follow (narrow scan), full discovery on a miss.
        fp = None
        from_discovery = False
        follow_missed = False
        best_diag = None
        committed_class = None
        acq_msg = ""
        self.clf_cache.clear()
        self.t_scan[i] = 0.0

        if self.anchor_rz is not None:
            # Anchor (R,Z) -> (s,theta) on THIS field, in the fixed host volume. An inversion
            # failure (e.g. the chain migrated across an interface) is just a miss.
            seed_s = seed_th = None
            try:
                cs, ct = specout.get_st_coords(
                    self.volume, self.anchor_rz[0], self.anchor_rz[1], 0.0
                )
                seed_s = float(np.asarray(cs).ravel()[0])
                seed_th = float(np.asarray(ct).ravel()[0])
            except Exception as exc:
                warns.append(f"anchor inversion failed in vol{self.volume} — {exc}")
            if seed_s is not None:
                # Re-centre theta on the previous member.
                if self.anchor_th is not None:
                    seed_th = self.anchor_th
                # Centre the scan on prev_s (the committed surface predicts the next far better than
                # the noisy (R,Z)->s inversion), with a fixed follow-box half-width.
                use_prev = self.prev_s is not None and np.isfinite(self.prev_s)
                seed_s_used = self.prev_s if use_prev else seed_s
                geom_ceiling = self.follow_window if self.follow_window is not None else 0.30
                drift_gate = (
                    _MAX_JUMP_FLOOR
                    if not np.isfinite(self.prev_disp)
                    else max(_MAX_JUMP_FLOOR, 10.0 * self.prev_disp)
                )
                max_move = min(geom_ceiling, drift_gate)
                th_grid = np.linspace(seed_th - DTHETA, seed_th + DTHETA, N_THETA)
                ts = perf_counter()
                fp, bp, st = self.find_fp(
                    specout,
                    self.volume,
                    seed_s_used - FOLLOW_DS,
                    seed_s_used + FOLLOW_DS,
                    th_grid,
                    anchor_rz=self.anchor_rz,
                    max_move=max_move,
                    require_ntor=self.prev_ntor,
                )
                if bp is not None and (
                    best_diag is None or bp["residual"] < best_diag["residual"]
                ):
                    best_diag = bp
                self.t_scan[i] += perf_counter() - ts
                acq_msg = (
                    f"[follow] tracked from previous anchor ({st['n_cands']} cand, "
                    f"window {max_move * 1e3:.0f} mm)"
                )
            if fp is None:
                follow_missed = True

        if fp is None and self.discover:
            # FP-first: full-volume scan, polish every minimum, keep iota-gate passers. A follow
            # miss pays the same full sweep as a cold start (no localized re-lock — simplicity).
            th_full = np.linspace(0.0, 2.0 * np.pi, self.disc_n_theta, endpoint=False)
            ts = perf_counter()
            confirmed, bp, st = self.find_fp(
                specout,
                self.volume,
                -_S_CLAMP,
                _S_CLAMP,
                th_full,
                return_all=True,
                require_ntor=self.prev_ntor,
                n_s_scan=DISC_N_S,
            )
            if bp is not None and (best_diag is None or bp["residual"] < best_diag["residual"]):
                best_diag = bp
            n_members = len(confirmed)
            res_str = f"{self.prev_ntor}/{self.mpol}" if self.prev_ntor else f"k*Nfp (period-{self.mpol})"
            lbl = "[cold] follow missed -> re-discovered" if follow_missed else "[cold] discovered"
            if confirmed:
                # Prefer an O-point (island centre); order by member stability (theta nearest the
                # last member, else lowest residual) and commit the first O.
                if self.last_member_th is not None:

                    def wrap(th, _m=self.last_member_th):  # _m default binds the current value
                        return abs((th - _m + np.pi) % (2 * np.pi) - np.pi)

                    order = sorted(confirmed, key=lambda e: (wrap(e["theta"]), e["residual"]))
                else:
                    order = confirmed
                for e in order:
                    ci = self.classify_fp(specout, self.volume, e["R"], e["Z"])
                    if ci is not None and ci["kind"] == "o":
                        fp, committed_class = e, ci
                        acq_msg = f"{lbl} {n_members} members on {res_str}; locked O-point"
                        break
                else:
                    fp = order[0]
                    committed_class = self.classify_fp(specout, self.volume, fp["R"], fp["Z"])
                    acq_msg = (
                        f"{lbl} {n_members} members on {res_str}; "
                        f"no O found -> nearest-seed member"
                    )
                from_discovery = True
            self.t_scan[i] += perf_counter() - ts

        # -- Commit (polish-gated: fp set only if a Newton polish confirmed it) --
        if fp is None:
            self.no_fp[i] = True
            nofp_reason = "no polishable scan candidate"
            if best_diag is not None:
                if not best_diag["converged"]:
                    why = "failed ftol gate"
                elif best_diag.get("iota_rel") is not None:
                    why = (
                        f"passed ftol; failed iota gate (lowest relative iota error "
                        f"{best_diag['iota_rel']:.1e}, gate {self.iota_rtol:.0e})"
                    )
                else:
                    why = "passed ftol; failed continuity gate"
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
        self.residual[i] = fp["residual"]
        self.island_rz[i] = [r_sel, z_sel]
        self.island_sth[i] = [s_sel, th_sel]

        # iota / ntor were measured at the fixed point by the iota gate — reuse (no re-trace).
        self.iota[i], self.ntor[i], self.iota_rel[i] = fp["iota"], fp["ntor"], fp["iota_rel"]
        if fp["iota_rel"] > 0.5 * self.iota_rtol:
            warns.append(f"marginal: iota_rel {self.iota_rel[i]:.1e} in upper half of gate band")

        # Robust O/X classification (reuse discovery's verdict for this member if available).
        ci = (
            committed_class
            if committed_class is not None
            else self.classify_fp(specout, self.volume, r_sel, z_sel)
        )
        if ci is not None:
            self.kind[i], self.residue[i] = ci["kind"], ci["residue"]
            self.rdet_err[i] = ci["det_err"]
            self.classify_method[i], self.orbit_growth[i] = ci["method"], ci["growth"]

        # Auto-derive the continuity gate from the island geometry: the mpol islands lie at minor
        # radius r = ||FP-axis||, so 0.35*2*pi*r/mpol ~ 0.7x O-X spacing.
        if np.isfinite(self.axis_rz[i, 0]):
            r = np.hypot(r_sel - self.axis_rz[i, 0], z_sel - self.axis_rz[i, 1])
            self.follow_window = 0.35 * 2.0 * np.pi * r / self.mpol

        # Displacement is the convergence metric, valid only across an anchor-follow step. ANY
        # discovery-sourced lock breaks continuity (the committed member is not guaranteed to be
        # the same one), so leave displacement NaN — which also stops a falsified re-acquire
        # distance from poisoning prev_disp.
        self.provenance[i] = "cold" if from_discovery else "follow"
        if self.prev_island is not None and not from_discovery:
            self.displacement[i] = np.linalg.norm(self.island_rz[i] - self.prev_island)
        self.prev_island = self.island_rz[i].copy()
        self.prev_s = s_sel
        self.prev_ntor = int(self.ntor[i])
        self.last_member_th = th_sel
        self.prev_disp = self.displacement[i] if np.isfinite(self.displacement[i]) else self.prev_disp
        self.anchor_rz = self.island_rz[i].copy()
        self.anchor_th = th_sel

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
            "iota": self.iota,
            "ntor": self.ntor,
            "iota_rel": self.iota_rel,
            "kind": self.kind,
            "residue": self.residue,
            "det_err": self.rdet_err,
            "classify_method": self.classify_method,
            "orbit_growth": self.orbit_growth,
            "displacement": self.displacement,
            "provenance": self.provenance,
            "no_fp": self.no_fp,
            "t_total": self.t_total,
            "t_field_reconstruction": self.t_recon,
            "t_scan": self.t_scan,
            "mpol": self.mpol,
            "volume_start": self.volume,
            "nfp": self.nfp,
            "start_at_iter": int(start_at_iter),
            "stride": int(stride),
            "root_ftol": float(ROOT_FTOL),
            "iota_rtol": float(self.iota_rtol),
            "target_ntor": int(self.target_ntor) if self.target_ntor else -1,
            "reverse": bool(reverse),
            "discover": bool(self.discover),
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
    start_at_iter: int = 0,
    stride: int = 1,
    limit: int = 0,
    reverse: bool = False,
    iota_rtol: float = 0.01,
    max_axis_repeats: int = 8,
    verbose: bool = False,
    checkpoint_path: str | Path | None = None,
) -> dict:
    """Track the magnetic axis and one island fixed-point chain across optimisation checkpoints.

    Post-processing driver around :class:`FixedPointTracker`: replays the ``x_<a>_<b>.npy``
    checkpoints, re-solves the field from disk, and calls ``tracker.step`` per checkpoint. Either
    pass ``guess`` (an ``(R, Z)`` seed for the first checkpoint) or ``discover=True`` to self-seed by
    an FP-first scan filtered to the target resonance. The scan/polish/classifier tuning lives in
    the module constants at the top of this file.

    Args:
        xs_fixb_dir: Folder of ``x_<a>_<b>.npy`` checkpoints (``b`` is the iteration number).
        toml_path: SPECTRE TOML matching the checkpoints.
        mpol: Island count = denominator of ``iota = ntor/mpol``.
        volume: Volume index hosting the chain (``>= 1``).
        guess: ``(R, Z)`` at ``phi=0`` of the tracked fixed point on the first checkpoint. Required
            unless ``discover``.
        discover: Self-seed (no ``guess``): FP-first scan, filter by iota, re-run whenever lost.
        target_ntor: Toroidal mode number of the chain (``iota = ntor/mpol``); the iota gate keeps
            only fixed points with this ntor. Default: any positive multiple of Nfp.
        start_at_iter: Skip checkpoints with iteration ``< start_at_iter``.
        stride: Process every ``stride``-th checkpoint.
        limit: Process at most this many checkpoints (0 = all).
        reverse: Process high->low iter (track backward from the converged state).
        iota_rtol: iota gate, keep a point only if ``|iota-ntor/mpol|/(ntor/mpol) <= this``.
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

    auto_discover = guess is None and not discover
    if auto_discover:
        discover = True  # no guess given -> self-seed by discovery (noted in the header below)

    # Resolve the .npz next to the TOML: a bare filename (no directory component) lands in the
    # TOML's folder, not the cwd, so concurrent runs on different cases never collide.
    if checkpoint_path is None:
        out = None
    else:
        out = Path(checkpoint_path)
        if out.parent == Path("."):
            out = toml_path.parent / out.name
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
        iota_rtol=iota_rtol,
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
    if auto_discover:
        print("  note     : no guess given -> DISCOVER (self-seed by an FP-first scan)")
    print(
        f"  run      : {n} checkpoints (iters {paths[0][0]}..{paths[-1][0]}, "
        f"{'reverse' if reverse else 'forward'}, stride {stride})"
    )
    print(f"  gates    : FP residual < {ROOT_FTOL:.0e} · iota within {iota_rtol:.0e} of {gate_res}")
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
    n_follow = int(np.sum(tk.provenance == "follow"))
    res_rng = (
        f"{tk.residual[found].min():.1e}..{np.nanmax(tk.residual[found]):.1e}"
        if found.any()
        else "n/a"
    )
    print(
        f"\nDone. {int(found.sum())}/{n} loc · cold {n_cold}/follow {n_follow} "
        f"· no_fp {int(tk.no_fp.sum())} · res {res_rng}"
    )
    print(flush=True)
    return tk.result_dict(iter_indices, start_at_iter, stride, reverse)
