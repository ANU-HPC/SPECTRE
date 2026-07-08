# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT
"""Track the magnetic axis and one island fixed-point chain across equilibria.

The algorithm lives in :class:`FixedPointTracker`, a stateful stepper whose ``step(specout, i)``
processes one equilibrium (a ``SPECTREout``) and records the axis and island fixed point, the
step displacement (the geometric convergence metric), the robust O/X verdict, and how the point
was acquired. The carry-state (anchor, previous surface, drift) lives on the instance, so
successive ``step`` calls follow the same island across a sequence of equilibria.

When ``repeat_period=None`` the tracker operates in axis-only mode: only the magnetic axis is
located; all island arrays remain NaN and ``no_fp=True`` every step.

Two drivers feed it the same algorithm:

  * ``scripts/minimization_monitoring/track_fp.py`` — **post-processing**: replays
    ``x_<a>_<b>.npy`` checkpoints from disk, re-solves the field (``force_real``), and calls
    ``step`` per checkpoint.
  * ``spectre.force_minimizer`` — **in-loop**: builds a tracker once and calls ``step`` on the
    live, already-solved in-memory field each optimisation iteration.

Built on the ``spectre.fixed_points.primitives`` functions and the robust O/X classifier in
``spectre.fixed_points.classification`` (analytic variational tangent map). Two regimes, the
same two acceptance conditions in both:

  * **Discovery** (no anchor). FP-FIRST then resonance-filter.
    Scan the whole radial extent of the host volume at ``2*repeat_period+1`` theta, polish every
    candidate (return-map residual gate ``< ftol``), then keep only those whose empirical iota
    gives the target ``poloidal_occurrences`` a positive multiple of Nfp, in lowest terms.
    Discovery prefers an O-point (island centre) via the classifier.

  * **Anchor-follow** (have an anchor). Cheap narrow scan around the previous fixed point's
    ``(s, theta)`` in the fixed host volume. Two gates confirm "same island": ``(R,Z)``
    continuity (drift-adaptive) and resonance identity. A miss falls back to discovery.

The tracker is deliberately simple: it tracks one chain in one volume with no cross-volume
handoff. A chain that migrates across an interface shows up as ``no_fp`` rows.
"""

from __future__ import annotations

from collections.abc import Sequence
from time import perf_counter

import numpy as np

_S_CLAMP = 0.999  # hard clamp on the seed s-range; s in (-1, 1) spans one annular volume
# (R,Z) continuity floor [m]; a candidate may move up to max(this, 10*prev_disp).
_MAX_JUMP_FLOOR = 0.02

# Tuning constants (settled values; kept fixed for simplicity).
FOLLOW_DS = 0.05  # anchor-follow box half-width in s
N_S = 61  # s samples per theta in a follow scan
DISC_N_S = 121  # s samples per theta in discovery; resonant dip width_s ~ 0.008 - do not coarsen
DTHETA = 0.3  # follow theta-grid half-width
N_THETA = 7  # follow theta samples
ROOT_MAXFEV = 100  # Newton-polish function-evaluation budget
ROOT_RTOL = 1e-8  # polish integrator tolerance
ROOT_FTOL = 1e-6  # return-map residual acceptance gate [m]
SCAN_RTOL = 1e-6  # candidate-locating scan tolerance (looser: only seeds the polish, ~2.6x faster)
DEDUP_TOL = 1e-3  # (R,Z) tolerance [m] for collapsing duplicate discovery members
CLASSIFY_RTOL = 1e-12  # O/X classifier integrator tolerance (must be tight near resonances)

_D, _OK, _NO, _WARN, _DASH = "D", "ok", "no", "!", "-"


def _fmt(x, w=8, p=4):
    """Render a fixed-width float, ``---`` for non-finite, to keep the verbose block aligned."""
    if x is None or not np.isfinite(x):
        return f"{'---':>{w}}"
    return f"{x:{w}.{p}f}"


class FixedPointTracker:
    """Stateful tracker for the magnetic axis and one island fixed-point chain.

    Construct once with the resonance (``repeat_period``, ``volume``, optional
    ``poloidal_occurrences``) and either a ``guess`` ``(R, Z)`` seed or ``discover=True``; then
    call :meth:`step` on each equilibrium in sequence. Per-checkpoint diagnostics accumulate in
    pre-allocated arrays of length ``n`` (indexed by the ``i`` passed to :meth:`step`), exposed
    as attributes and assembled by :meth:`result_dict`.

    Pass ``repeat_period=None`` for axis-only mode (no island tracking).
    """

    def __init__(
        self,
        repeat_period: int | None,
        volume: int | None,
        n: int,
        *,
        guess: Sequence[float] | None = None,
        discover: bool = False,
        poloidal_occurrences: int | None = None,
        max_axis_repeats: int = 8,
    ):
        """Initialize the tracker.

        Args:
            repeat_period: Island count = denominator of
                ``iota = poloidal_occurrences/repeat_period`` (the Poincare map period).
                Pass ``None`` for axis-only mode.
            volume: Volume index hosting the chain (0 = the singular axis volume). Unused
                when ``repeat_period=None``.
            n: Number of equilibria (pre-allocates all diagnostic arrays).
            guess: ``(R, Z)`` seed at ``phi=0`` of the tracked fixed point on the first
                checkpoint. May be combined with ``discover``.
            discover: Self-seed by FP-first scan when no anchor (or after a follow miss).
            poloidal_occurrences: Toroidal mode number of the chain
                (``iota = poloidal_occurrences/repeat_period``).
            max_axis_repeats: Random-restart budget for the axis finder.
        """
        island_mode = repeat_period is not None
        if island_mode and guess is None and not discover:
            discover = True  # no explicit guess -> self-seed by an FP-first discovery scan
        # A genuine period-repeat_period chain has iota in lowest terms.
        if (
            island_mode
            and poloidal_occurrences is not None
            and int(np.gcd(poloidal_occurrences, repeat_period)) != 1
        ):
            g = int(np.gcd(poloidal_occurrences, repeat_period))
            raise ValueError(
                f"poloidal_occurrences {poloidal_occurrences} is not coprime with "
                f"repeat_period {repeat_period} (gcd={g}); that is a "
                f"period-{repeat_period // g} chain, not period-{repeat_period}; "
                f"pass a coprime repeat_period/poloidal_occurrences."
            )

        # SPECTRE imports are deferred to construction time (heavy core; keeps module import cheap).
        from spectre.fixed_points.classification import FixedPointClassifier, classify_point
        from spectre.fixed_points.primitives import find_axis, root_fixed_point, scan_fixed_points

        self._find_axis = find_axis
        self._root_fixed_point = root_fixed_point
        self._scan_fixed_points = scan_fixed_points
        self._FixedPointClassifier = FixedPointClassifier
        self._classify_point = classify_point

        # -- Config --
        self.repeat_period = repeat_period
        self.volume = int(volume) if volume is not None else 0
        self.island_mode = island_mode
        self.discover = discover
        self.poloidal_occurrences = poloidal_occurrences
        self.max_axis_repeats = max_axis_repeats

        # -- Derived (island-mode only) --
        if island_mode:
            # trace_one_fieldline logs phi=0 at t=0 too, so the repeat_period-fold map needs +1
            self.n_transits = repeat_period + 1
            # Poloidal scan must resolve the repeat_period islands: sample 2*repeat_period+1 theta
            self.disc_n_theta = max(2 * repeat_period + 1, 16)
            self.root_kw = dict(maxfev=ROOT_MAXFEV, rtol=ROOT_RTOL, atol=ROOT_RTOL, ftol=ROOT_FTOL)
        self.nfp = -1  # learned from the first equilibrium
        # ``follow_window`` is the live continuity-gate half-width [m]; always AUTO-derived from the
        # island geometry at each step.
        self.follow_window = None
        # per-checkpoint FixedPointClassifier per volume (reset each step)
        self.clf_cache: dict = {}

        # -- Carry-state (persists across step calls) --
        self.anchor_rz: np.ndarray | None = (
            np.array(guess, dtype=float) if guess is not None else None
        )
        self.anchor_th: float | None = None
        self.prev_axis: np.ndarray | None = None
        self.prev_island: np.ndarray | None = None
        self.prev_s: float | None = None
        self.prev_poloidal_occurrences: int | None = poloidal_occurrences
        self.last_member_th: float | None = None
        self.prev_disp = np.nan
        self.prev_residue = np.nan

        # -- Pre-allocate every per-checkpoint diagnostic --
        self.n = n
        self.vol_tracked = np.full(n, self.volume, dtype=int)
        self.axis_rz = np.full((n, 2), np.nan)
        self.island_rz = np.full((n, 2), np.nan)
        self.island_sth = np.full((n, 2), np.nan)
        self.island_rz_half_period = np.full((n, 2), np.nan)
        self.residual = np.full(n, np.nan)
        self.iota = np.full(n, np.nan)
        self.ntor = np.zeros(n, dtype=int)
        self.iota_rel = np.full(n, np.nan)
        self.kind = np.full(n, "", dtype="<U10")
        self.residue = np.full(n, np.nan)
        self.greene_trace = np.full(n, np.nan)
        self.greene_det = np.full(n, np.nan)
        self.greene_disc = np.full(n, np.nan)
        # how the FP was acquired: cold/follow
        self.provenance = np.full(n, "", dtype="<U10")
        self.no_fp = np.zeros(n, dtype=bool)
        # t_total/t_recon are driver-side (post-processing only); the rest are set by step().
        self.t_total = np.full(n, np.nan)
        self.t_recon = np.full(n, np.nan)
        self.t_scan = np.full(n, np.nan)
        self.t_axis = np.full(n, np.nan)
        self.t_classify = np.full(n, np.nan)
        self.t_step = np.full(n, np.nan)

    # ---
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
        require_poloidal_occurrences=None,
        n_s_scan=None,
    ):
        """FP-first then resonance-filter: scan, polish each minimum, keep passing points.

        Gates are the return-map residual (``root_fixed_point`` converged, i.e. residual < ftol)
        and the resonance identity (integer ``poloidal_occurrences`` match and Nfp-compatibility).
        With ``return_all`` returns the residual-sorted list of accepted members (deduplicated by
        ``(R,Z)``), else the first. Returns ``(result, best_polish, stats)``. The ``(R,Z)``
        proximity gate is applied only against ``anchor_rz`` (continuity), never the scan seed.
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
            print(f"      scan failed (vol {vol}) - {exc}", flush=True)
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
                continue  # polished to a different member than tracked - keep continuity
            n_rm += 1
            # Resonance filter: measure iota AT the fixed point and keep only a genuine
            # period-repeat_period resonance in lowest terms, matching require_poloidal_occurrences.
            # A true fixed point is exactly rational; the integer round identifies the resonance.
            _, _, iota_tr, flag = specout.trace_one_fieldline(
                np.array([pol.s, pol.theta]), vol, num_phi_planes=1, num_transits=self.n_transits
            )
            if flag:
                continue
            iota_emp = -float(iota_tr)  # physical (VMEC) sign convention
            ntor_p = int(round(abs(iota_emp) * self.repeat_period)) if np.isfinite(iota_emp) else 0
            ok = (
                ntor_p > 0
                and ntor_p % self.nfp == 0
                and int(np.gcd(ntor_p, self.repeat_period)) == 1
                and (require_poloidal_occurrences is None or ntor_p == require_poloidal_occurrences)
            )
            if not ok:
                continue
            rel = abs(abs(iota_emp) - ntor_p / self.repeat_period) / (ntor_p / self.repeat_period)
            if min_iota_rel is None or rel < min_iota_rel:
                min_iota_rel = rel
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
        # Collapse duplicates: the same FP polished from several adjacent theta seeds.
        deduped = []
        for e in sorted(confirmed, key=lambda e: e["residual"]):
            if all(np.hypot(e["R"] - k["R"], e["Z"] - k["Z"]) > DEDUP_TOL for k in deduped):
                deduped.append(e)
        if best_polish is not None:
            best_polish["iota_rel"] = min_iota_rel
        return (
            (deduped if return_all else None),
            best_polish,
            {"n_cands": len(cands), "n_rm": n_rm, "n_iota": len(deduped)},
        )

    # ---
    def classify_fp(self, specout, vol, sth):
        """Robust O/X of a located fixed point (reuses a warmed classifier; None on failure)."""
        try:
            clf = self.clf_cache.get(vol)
            if clf is None:
                clf = self._FixedPointClassifier(
                    specout,
                    vol,
                    self.repeat_period,
                    method="DOP853",
                    rtol=CLASSIFY_RTOL,
                    atol=CLASSIFY_RTOL,
                )
                self.clf_cache[vol] = clf
            return self._classify_point(clf, sth)
        except Exception as exc:
            print(f"    classify error - {exc}", flush=True)
            return None

    # ---
    def log_row(self, i, ctx, *, it=None, done=None, n=None, elapsed=None):
        """Print the per-checkpoint verbose block for row ``i`` (post-processing driver).

        ``done``/``n`` add a progress count to the header; the timing line is printed when
        ``elapsed`` is given. The in-loop caller uses :meth:`log_compact` instead.
        """
        it_lbl = i if it is None else it
        warns = ctx["warns"]
        hdr = f"\n[iter {it_lbl:3d}]"
        if done is not None and n is not None:
            hdr += f"  {done}/{n}"
        hdr += f"  vol{self.volume}"
        print(hdr)
        axis = f"  axis     ({ctx['rax']:.5f}, {ctx['zax']:+.5f})"
        if np.isfinite(ctx["axis_disp"]):
            axis += f"   {_D} {ctx['axis_disp']:.1e} m"
        print(axis)
        if ctx["no_fp"]:
            for w in warns:
                print(f"  {_WARN} {w}")
            print(f"  {_WARN} no_fp   {ctx['nofp_reason']}")
        else:
            r_sel, z_sel = self.island_rz[i]
            isl = f"  island   ({r_sel:.4f}, {z_sel:+.4f})"
            if np.isfinite(ctx["displacement"]):
                isl += f"   {_D} {ctx['displacement']:.1e} m"
            print(isl)
            print(f"  acquire  {ctx['acq_msg']}")
            for w in warns:
                print(f"  {_WARN} {w}")
            gate_ok = _OK if self.residual[i] < ROOT_FTOL else _NO
            print(
                f"  gates    FP res {self.residual[i]:.1e} < {ROOT_FTOL:.0e} {gate_ok}   "
                f"iota {self.iota[i]:+.4f} (= {self.ntor[i]}/{self.repeat_period})"
            )
            if self.kind[i]:
                kindword = "O-point (elliptic)" if self.kind[i] == "o" else "X-point (hyperbolic)"
                cls = f"  class    {kindword} {_DASH} Greene={self.residue[i]:.3e}"
                if np.isfinite(ctx["residue_delta"]):
                    cls += f" ({ctx['residue_delta']:+.1e})"
            else:
                cls = f"  class    ? {_DASH} classification unavailable"
            print(cls)
        if elapsed is not None:
            print(
                f"  timing   recon {_fmt(self.t_recon[i], 5, 0)}s"
                f"  scan {_fmt(self.t_scan[i], 5, 0)}s"
                f"  classify {_fmt(self.t_classify[i], 5, 1)}s"
                f"  total {_fmt(self.t_total[i], 5, 0)}s"
                f"  | {elapsed / 60:.1f}m elapsed",
                flush=True,
            )

    def log_compact(self, i, ctx):
        """Print row ``i`` as one terse ``fp:`` line (in-loop log, amid the optimiser output)."""
        for w in ctx["warns"]:
            print(f"fp: {_WARN} {w}")
        line = f"fp: axis ({ctx['rax']:.5f}, {ctx['zax']:+.5f})"
        if np.isfinite(ctx["axis_disp"]):
            line += f" d {ctx['axis_disp']:.1e} m"
        if ctx["no_fp"]:
            print(f"{line} | no_fp {_DASH} {ctx['nofp_reason']}", flush=True)
            return
        r_sel, z_sel = self.island_rz[i]
        line += f" | isl vol{self.volume} ({r_sel:.4f}, {z_sel:+.4f})"
        if np.isfinite(ctx["displacement"]):
            line += f" d {ctx['displacement']:.1e} m"
        line += f" [{self.provenance[i]}]"
        if self.kind[i]:
            line += f" | Greene={self.residue[i]:.3e}"
            if np.isfinite(ctx["residue_delta"]):
                line += f" ({ctx['residue_delta']:+.1e})"
        print(line, flush=True)

    def step(self, specout, i):
        """Process one equilibrium ``specout``, recording diagnostics into row ``i``.

        Updates the carry-state (anchor, previous surface/island, drift) so the next call
        follows the same chain. Driver-side timings (``t_recon``, ``t_total``) and verbose
        printing are left to the caller. Returns a small context dict for the caller's verbose
        block: ``{rax, zax, acq_msg, warns, nofp_reason, no_fp, axis_disp, displacement,
        residue_delta}``.
        """
        t0 = perf_counter()
        warns: list[str] = []

        if self.nfp < 0:
            self.nfp = int(specout.input.physics.Nfp)

        # 1) Magnetic axis. Failure leaves the row NaN, does not abort.
        rax = zax = np.nan
        axis_disp = np.nan
        ta = perf_counter()
        try:
            rax, zax, _, _ = self._find_axis(
                specout, max_repeats=self.max_axis_repeats, verbose=False
            )
            self.axis_rz[i] = [rax, zax]
            if self.prev_axis is not None:
                axis_disp = float(np.linalg.norm(self.axis_rz[i] - self.prev_axis))
            self.prev_axis = self.axis_rz[i].copy()
        except Exception as exc:
            warns.append(f"axis finding failed - {exc}")
        self.t_axis[i] = perf_counter() - ta

        # Axis-only mode: skip island finding entirely.
        if not self.island_mode:
            self.no_fp[i] = True
            self.t_step[i] = perf_counter() - t0
            return {
                "rax": rax,
                "zax": zax,
                "acq_msg": "",
                "warns": warns,
                "nofp_reason": "axis-only mode",
                "no_fp": True,
                "axis_disp": axis_disp,
                "displacement": np.nan,
                "residue_delta": np.nan,
            }

        # 2) Acquire the fixed point: anchor-follow (narrow scan), full discovery on a miss.
        fp = None
        from_discovery = False
        follow_missed = False
        best_diag = None
        committed_class = None
        acq_msg = ""
        self.clf_cache.clear()
        self.t_scan[i] = 0.0
        self.t_classify[i] = 0.0

        if self.anchor_rz is not None:
            # Anchor (R,Z) -> (s,theta) on THIS field.
            seed_s = seed_th = None
            try:
                cs, ct = specout.get_st_coords(
                    self.volume, self.anchor_rz[0], self.anchor_rz[1], 0.0
                )
                seed_s = float(np.asarray(cs).ravel()[0])
                seed_th = float(np.asarray(ct).ravel()[0])
            except Exception as exc:
                warns.append(f"anchor inversion failed in vol{self.volume} - {exc}")
            if seed_s is not None:
                if self.anchor_th is not None:
                    seed_th = self.anchor_th
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
                    require_poloidal_occurrences=self.prev_poloidal_occurrences,
                )
                if bp is not None and (best_diag is None or bp["residual"] < best_diag["residual"]):
                    best_diag = bp
                self.t_scan[i] += perf_counter() - ts
                acq_msg = (
                    f"[follow] tracked from previous anchor ({st['n_cands']} cand, "
                    f"window {max_move * 1e3:.0f} mm)"
                )
            if fp is None:
                follow_missed = True
            else:
                # An X landing is a follow miss — we track the O, so fall back to discovery.
                tc = perf_counter()
                committed_class = self.classify_fp(specout, self.volume, (fp["s"], fp["theta"]))
                self.t_classify[i] += perf_counter() - tc
                if committed_class is not None and committed_class["kind"] == "x":
                    warns.append("follow landed on an X-point - re-discovering (O preferred)")
                    fp, committed_class, follow_missed = None, None, True

        if fp is None and self.discover:
            # FP-first: full-volume scan, polish every minimum, keep resonance-gate passers.
            th_full = np.linspace(0.0, 2.0 * np.pi, self.disc_n_theta, endpoint=False)
            if self.anchor_th is not None:
                # Guarantee the last known O-point theta is always seeded.
                th_full = np.unique(np.append(th_full, float(self.anchor_th) % (2.0 * np.pi)))
            ts = perf_counter()
            confirmed, bp, st = self.find_fp(
                specout,
                self.volume,
                -_S_CLAMP,
                _S_CLAMP,
                th_full,
                return_all=True,
                require_poloidal_occurrences=self.prev_poloidal_occurrences,
                n_s_scan=DISC_N_S,
            )
            if bp is not None and (best_diag is None or bp["residual"] < best_diag["residual"]):
                best_diag = bp
            self.t_scan[i] += perf_counter() - ts
            n_members = len(confirmed)
            res_str = (
                f"{self.prev_poloidal_occurrences}/{self.repeat_period}"
                if self.prev_poloidal_occurrences
                else f"k*Nfp (period-{self.repeat_period})"
            )
            lbl = "[cold] follow missed -> re-discovered" if follow_missed else "[cold] discovered"
            if confirmed:
                # Prefer an O-point; order by member stability then commit the first O.
                if self.last_member_th is not None:
                    _m = self.last_member_th
                    order = sorted(
                        confirmed,
                        key=lambda e: (
                            abs((e["theta"] - _m + np.pi) % (2 * np.pi) - np.pi),
                            e["residual"],
                        ),
                    )
                else:
                    order = confirmed
                tc = perf_counter()
                for e in order:
                    ci = self.classify_fp(specout, self.volume, (e["s"], e["theta"]))
                    if ci is not None and ci["kind"] == "o":
                        fp, committed_class = e, ci
                        acq_msg = f"{lbl} {n_members} members on {res_str}; locked O-point"
                        break
                else:
                    fp = order[0]
                    committed_class = self.classify_fp(specout, self.volume, (fp["s"], fp["theta"]))
                    acq_msg = (
                        f"{lbl} {n_members} members on {res_str}; no O found -> nearest-seed member"
                    )
                self.t_classify[i] += perf_counter() - tc
                from_discovery = True

        # -- Commit --
        if fp is None:
            self.no_fp[i] = True
            nofp_reason = "no polishable scan candidate"
            if best_diag is not None:
                if not best_diag["converged"]:
                    if best_diag["residual"] < ROOT_FTOL and abs(best_diag["s"]) >= _S_CLAMP:
                        why = (
                            "converged at the volume edge - chain likely in the neighbouring volume"
                        )
                    else:
                        why = "failed ftol gate"
                elif best_diag.get("iota_rel") is not None:
                    why = (
                        f"passed ftol; failed resonance filter (lowest relative iota error "
                        f"{best_diag['iota_rel']:.1e})"
                    )
                else:
                    why = "passed ftol; failed continuity gate"
                nofp_reason = (
                    f"best polish {best_diag['residual']:.1e} @ s={best_diag['s']:+.4f} "
                    f"th={best_diag['theta']:.2f} ({why})"
                )
            self.t_step[i] = perf_counter() - t0
            return {
                "rax": rax,
                "zax": zax,
                "acq_msg": acq_msg,
                "warns": warns,
                "nofp_reason": nofp_reason,
                "no_fp": True,
                "axis_disp": axis_disp,
                "displacement": np.nan,
                "residue_delta": np.nan,
            }

        s_sel, th_sel, r_sel, z_sel = fp["s"], fp["theta"], fp["R"], fp["Z"]
        self.residual[i] = fp["residual"]
        self.island_rz[i] = [r_sel, z_sel]
        self.island_sth[i] = [s_sel, th_sel]
        _, RZ_mp, _, _ = specout.trace_one_fieldline(
            [s_sel, th_sel], self.volume, num_phi_planes=2, num_transits=1
        )
        self.island_rz_half_period[i] = [float(RZ_mp[0, 1]), float(RZ_mp[1, 1])]

        self.iota[i], self.ntor[i], self.iota_rel[i] = fp["iota"], fp["ntor"], fp["iota_rel"]
        if fp["iota_rel"] > 0.005:
            warns.append(f"iota_rel {self.iota_rel[i]:.1e} unusually large for a true fixed point")

        # Robust O/X classification (reuse discovery's verdict for this member if available).
        if committed_class is not None:
            ci = committed_class
        else:
            tc = perf_counter()
            ci = self.classify_fp(specout, self.volume, (s_sel, th_sel))
            self.t_classify[i] += perf_counter() - tc
        if ci is not None:
            self.kind[i], self.residue[i] = ci["kind"], ci["residue"]
            self.greene_trace[i] = ci["trace"]
            self.greene_det[i] = ci["det"]
            self.greene_disc[i] = ci["disc"]

        # Auto-derive the continuity gate from the island geometry.
        if np.isfinite(self.axis_rz[i, 0]):
            r = np.hypot(r_sel - self.axis_rz[i, 0], z_sel - self.axis_rz[i, 1])
            self.follow_window = 0.35 * 2.0 * np.pi * r / self.repeat_period

        # Displacement is the convergence metric, valid only for anchor-follow steps; same for
        # the Greene-residue change (a cold re-acquire may be a different member).
        displacement = np.nan
        residue_delta = np.nan
        self.provenance[i] = "cold" if from_discovery else "follow"
        if self.prev_island is not None and not from_discovery:
            displacement = float(np.linalg.norm(self.island_rz[i] - self.prev_island))
            if np.isfinite(self.residue[i]) and np.isfinite(self.prev_residue):
                residue_delta = self.residue[i] - self.prev_residue
        self.prev_island = self.island_rz[i].copy()
        self.prev_s = s_sel
        self.prev_poloidal_occurrences = int(self.ntor[i])
        self.last_member_th = th_sel
        self.prev_disp = displacement if np.isfinite(displacement) else self.prev_disp
        self.prev_residue = self.residue[i]
        self.anchor_rz = self.island_rz[i].copy()
        self.anchor_th = th_sel

        self.t_step[i] = perf_counter() - t0
        return {
            "rax": rax,
            "zax": zax,
            "acq_msg": acq_msg,
            "warns": warns,
            "nofp_reason": "",
            "no_fp": False,
            "axis_disp": axis_disp,
            "displacement": displacement,
            "residue_delta": residue_delta,
        }

    # ---
    def result_dict(self, iter_indices, start_at_iter, stride, reverse) -> dict:
        """Assemble the per-checkpoint arrays + scalar metadata into the output schema."""
        return {
            "iteration_indices": iter_indices,
            "vol_tracked": self.vol_tracked,
            "axis_rz": self.axis_rz,
            "island_rz": self.island_rz,
            "island_sth": self.island_sth,
            "island_rz_half_period": self.island_rz_half_period,
            "residual": self.residual,
            "iota": self.iota,
            "ntor": self.ntor,
            "iota_rel": self.iota_rel,
            "kind": self.kind,
            "residue": self.residue,
            "greene_trace": self.greene_trace,
            "greene_det": self.greene_det,
            "greene_disc": self.greene_disc,
            "provenance": self.provenance,
            "no_fp": self.no_fp,
            "t_total": self.t_total,
            "t_field_reconstruction": self.t_recon,
            "t_scan": self.t_scan,
            "t_axis": self.t_axis,
            "t_classify": self.t_classify,
            "t_step": self.t_step,
            "repeat_period": self.repeat_period if self.repeat_period is not None else -1,
            "volume_start": self.volume,
            "nfp": self.nfp,
            "start_at_iter": int(start_at_iter),
            "stride": int(stride),
            "root_ftol": float(ROOT_FTOL),
            "poloidal_occurrences": int(self.poloidal_occurrences)
            if self.poloidal_occurrences
            else -1,
            "reverse": bool(reverse),
            "discover": bool(self.discover),
        }

    def flush_to_h5(self, h5_path, n_done):
        """Write the first ``n_done`` steps to ``/iterations/fp`` in the H5 file.

        Opens the file in append mode and recreates the ``/iterations/fp`` group so data is
        up-to-date after each step — surviving a scancel on the enclosing SLURM job.
        """
        import h5py as _h5py

        res = self.result_dict(np.arange(n_done, dtype=int), 0, 1, False)
        try:
            with _h5py.File(h5_path, "a") as f:
                grp = f.require_group("iterations")
                if "fp" in grp:
                    del grp["fp"]
                fp_grp = grp.create_group("fp")
                string_dt = _h5py.string_dtype()
                for k, v in res.items():
                    if np.ndim(v):
                        arr = np.asarray(v)
                        if arr.dtype == object or arr.dtype.kind == "U":
                            fp_grp.create_dataset(
                                k, data=arr[:n_done].astype(object), dtype=string_dt
                            )
                        else:
                            fp_grp.create_dataset(k, data=arr[:n_done])
                    else:
                        fp_grp.attrs[k] = v
        except Exception as exc:
            print(f"[fp-track] flush_to_h5 failed - {exc}", flush=True)
