# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT
"""Track the magnetic axis and one island fixed-point chain across a sequence of equilibria.

:class:`FixedPointTracker` is a stateful stepper: :meth:`step` processes one equilibrium and
records the axis and island fixed point, the step displacement (the geometric convergence
metric), the O/X verdict, and how the point was acquired. The location found at one step primes
the search at the next, so successive steps follow the *same* island. Diagnostics accumulate in
pre-allocated per-step arrays exposed as attributes and assembled by :meth:`result_dict`.

Pass ``repeat_period=None`` for axis-only mode (island arrays stay NaN, ``no_fp=True`` every step).

Two drivers feed it the same algorithm: ``scripts/minimization_monitoring/track_fp.py`` replays
saved checkpoints, and ``spectre.force_minimizer`` calls :meth:`step` on the live field each
optimisation iteration.

Each step acquires the island in one of two regimes, both gated on the same two acceptance
conditions (return-map residual below ``root_ftol`` and a resonance-identity match):

* **Anchor-follow** (an anchor from the previous step exists): a narrow scan around the previous
  ``(s, theta)`` in the host volume, accepted only if it stays continuous in ``(R, Z)`` and keeps
  the same resonance. A miss falls back to discovery.
* **Discovery** (no anchor, or a follow miss): a full-volume scan, polishing every candidate and
  keeping resonance-gate passers, preferring the O-point (island centre).

The tracker follows one chain in one volume with no cross-volume handoff; a chain that migrates
across an interface shows up as ``no_fp`` rows.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from spectre.fixed_points.primitives import S_CLAMP

# -- Model parameters -----------------------------------------------------------------------
# Three groups, by what fixes the value. Group A is device-independent and permanent. Groups B
# and C are problem-dependent: the values below are defaults measured on CIEMAT QI4X, and are
# overridable per instance via the matching FixedPointTracker.__init__ arguments.

# A. Numerical tolerances -- integrator/solver settings, independent of the device. Permanent.
ROOT_MAXFEV = 100  # Newton-polish function-evaluation budget
ROOT_RTOL = 1e-8  # polish integrator tolerance
SCAN_RTOL = 1e-6  # candidate-locating scan tolerance (looser; only seeds the polish)
CLASSIFY_RTOL = 1e-12  # O/X classifier integrator tolerance. Pinned by a measured criterion:
# it drives the area-preservation diagnostic |det-1| to ~1e-9, which is what resolves the
# discriminant sign for near-parabolic points (see classification.py's module docstring).

# B. Search resolution -- set by the width of the resonant residual dip, which is ~0.008 in s
# for QI4X. A chain with a narrower dip needs finer sampling; a wider one can be coarsened.
DISC_N_S = 121  # s samples per theta in discovery; resolves the ~0.008-wide dip, do not coarsen
N_S = 61  # s samples per theta in a follow scan (narrower window, so fewer samples suffice)
FOLLOW_DS = 0.05  # anchor-follow box half-width in s
DTHETA = 0.3  # follow theta-grid half-width
N_THETA = 7  # follow theta samples
# (the discovery theta count is derived, not tuned: max(2*repeat_period + 1, 16) -- enough
# samples to resolve repeat_period islands around the poloidal circuit.)

# C. Absolute lengths [m] -- tuned on a major-radius ~20 m device (CIEMAT QI4X). These do NOT
# scale themselves: on a smaller machine (e.g. W7-X, ~5.5 m) they should be scaled down with it.
ROOT_FTOL = 1e-6  # return-map residual acceptance gate [m]
DEDUP_TOL = 1e-3  # (R,Z) tolerance [m] for collapsing duplicate discovery members
MAX_JUMP_FLOOR = 0.02  # (R,Z) continuity floor [m]; a step may move up to
DRIFT_FACTOR = 10.0  # max(MAX_JUMP_FLOOR, DRIFT_FACTOR * prev_disp)
GEOM_CEILING = 0.30  # hard (R,Z) move ceiling [m] before the follow window is known
FOLLOW_WINDOW_FRAC = 0.35  # follow window as a fraction of the inter-island arc 2*pi*r/period
IOTA_REL_WARN = 0.005  # relative iota mismatch above which a resonance match is flagged


@dataclass
class StepReport:
    """Display-only summary of one :meth:`FixedPointTracker.step`, consumed by :meth:`log`."""

    rax: float
    zax: float
    axis_disp: float
    no_fp: bool
    reason: str = ""  # why no fixed point was committed (empty on success)
    acq_msg: str = ""  # how the point was acquired
    warns: list = field(default_factory=list)
    displacement: float = np.nan  # (R,Z) move since the previous island (follow steps only)
    residue_delta: float = np.nan  # Greene-residue change since the previous step


@dataclass
class _Acq:
    """Internal carry between the follow and discovery acquisition passes of one step."""

    fp: dict | None = None
    cls: dict | None = None  # committed O/X classification of ``fp``
    msg: str = ""
    best_diag: dict | None = None  # lowest-residual polish attempt, for the no-fp reason
    from_discovery: bool = False
    follow_missed: bool = False


class FixedPointTracker:
    """Stateful tracker for the magnetic axis and one island fixed-point chain.

    Construct once with the resonance (``repeat_period``, ``volume``, optional
    ``poloidal_occurrences``) and either a ``guess`` ``(R, Z)`` or ``discover=True``, then call
    :meth:`step` on each equilibrium in sequence. Pass ``repeat_period=None`` for axis-only mode.
    """

    def __init__(
        self,
        repeat_period: int | None,
        volume: int | None,
        n: int,
        guess: Sequence[float] | None = None,
        discover: bool = False,
        poloidal_occurrences: int | None = None,
        max_axis_repeats: int = 8,
        root_ftol: float = ROOT_FTOL,
        dedup_tol: float = DEDUP_TOL,
        max_jump_floor: float = MAX_JUMP_FLOOR,
        geom_ceiling: float = GEOM_CEILING,
        disc_n_s: int = DISC_N_S,
        n_s: int = N_S,
    ):
        """Initialise the tracker and pre-allocate ``n`` rows of per-step diagnostics.

        The last six arguments are the problem-dependent model parameters (see the module-level
        "Model parameters" block). Their defaults were measured on CIEMAT QI4X; the four lengths
        are absolute metres and should be scaled with the machine size.

        Args:
            repeat_period: Denominator of ``iota = poloidal_occurrences/repeat_period`` (the
                Poincare-map period, i.e. the island count). ``None`` selects axis-only mode.
            volume: Volume index hosting the chain (0 = the singular axis volume).
            n: Number of equilibria; sizes every diagnostic array.
            guess: ``(R, Z)`` of the tracked fixed point at ``phi=0`` on the first step.
            discover: Self-seed by a full-volume scan when there is no anchor.
            poloidal_occurrences: Numerator of iota; pins the chain identity.
            max_axis_repeats: Random-restart budget for the axis finder.
            root_ftol: Return-map residual [m] below which a point is accepted.
            dedup_tol: ``(R, Z)`` tolerance [m] for collapsing duplicate discovery members.
            max_jump_floor: Floor [m] on how far the point may move between steps.
            geom_ceiling: Hard ceiling [m] on a step's move, used until the follow window is known.
            disc_n_s: ``s`` samples per theta in a discovery scan; must resolve the resonant dip.
            n_s: ``s`` samples per theta in a follow scan.
        """
        island_mode = repeat_period is not None
        if island_mode and guess is None and not discover:
            discover = True  # no explicit guess -> self-seed by discovery
        if (
            island_mode
            and poloidal_occurrences is not None
            and int(np.gcd(poloidal_occurrences, repeat_period)) != 1
        ):
            g = int(np.gcd(poloidal_occurrences, repeat_period))
            print(
                f"WARNING: poloidal_occurrences {poloidal_occurrences} and repeat_period "
                f"{repeat_period} are not coprime (gcd={g}); tracking a period-"
                f"{repeat_period // g} chain with the {repeat_period}-fold map.",
                flush=True,
            )

        # Bind heavy SPECTRE imports here (not at module load) to keep this module cheap to import.
        from spectre.fixed_points.classification import FixedPointClassifier, classify_point
        from spectre.fixed_points.primitives import find_axis, root_fixed_point, scan_fixed_points

        self._find_axis = find_axis
        self._root_fixed_point = root_fixed_point
        self._scan_fixed_points = scan_fixed_points
        self._FixedPointClassifier = FixedPointClassifier
        self._classify_point = classify_point

        # Config.
        self.repeat_period = repeat_period
        self.volume = int(volume) if volume is not None else 0
        self.island_mode = island_mode
        self.discover = discover
        self.poloidal_occurrences = poloidal_occurrences
        self.max_axis_repeats = max_axis_repeats
        self.nfp = -1  # learned from the first equilibrium

        # Problem-dependent model parameters (module defaults unless overridden).
        self.root_ftol = float(root_ftol)
        self.dedup_tol = float(dedup_tol)
        self.max_jump_floor = float(max_jump_floor)
        self.geom_ceiling = float(geom_ceiling)
        self.disc_n_s = int(disc_n_s)
        self.n_s = int(n_s)

        # Derived (island mode only).
        if island_mode:
            # trace_one_fieldline logs phi=0 at t=0 too, so the repeat_period-fold map needs +1.
            self.n_transits = repeat_period + 1
            # Resolve the repeat_period islands: at least 2*repeat_period+1 theta samples.
            self.disc_n_theta = max(2 * repeat_period + 1, 16)
            self.root_kw = dict(
                maxfev=ROOT_MAXFEV, rtol=ROOT_RTOL, atol=ROOT_RTOL, ftol=self.root_ftol
            )
        self.follow_window = None  # continuity-gate half-width [m], auto-derived each step
        self.clf_cache: dict = {}  # per-volume classifier, reset each step

        # Carry-state (persists across steps).
        self.anchor_rz = np.array(guess, dtype=float) if guess is not None else None
        self.anchor_th: float | None = None
        self.prev_axis: np.ndarray | None = None
        self.prev_island: np.ndarray | None = None
        self.prev_s: float | None = None
        self.prev_poloidal_occurrences = poloidal_occurrences
        self.last_member_th: float | None = None
        self.prev_disp = np.nan
        self.prev_residue = np.nan

        # Per-step diagnostics.
        self.n = n
        self.vol_tracked = np.full(n, self.volume, dtype=int)
        for name in ("axis_rz", "island_rz", "island_sth", "island_rz_half_period"):
            setattr(self, name, np.full((n, 2), np.nan))
        for name in ("residual", "iota", "iota_rel", "residue", "greene_trace",
                     "greene_det", "greene_disc"):  # fmt: skip
            setattr(self, name, np.full(n, np.nan))
        self.ntor = np.zeros(n, dtype=int)
        self.kind = np.full(n, "", dtype="<U10")
        self.provenance = np.full(n, "", dtype="<U10")  # 'cold' (discovered) or 'follow'
        self.no_fp = np.zeros(n, dtype=bool)

    # -- Fixed-point primitives -------------------------------------------------------------

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
        """Scan, polish every residual minimum, and keep the resonance-gate passers.

        A member is accepted when the polish converges (return-map residual below ``root_ftol``)
        and the empirical iota is a genuine ``poloidal_occurrences/repeat_period`` resonance in
        lowest terms with ``poloidal_occurrences`` a positive multiple of Nfp.

        Args:
            specout: A SPECTRE ``SPECTREout`` equilibrium.
            vol: Volume index to search (0 = the singular axis volume).
            s_lo: Lower ``s`` scan bound; clamped to ``-S_CLAMP``.
            s_hi: Upper ``s`` scan bound; clamped to ``+S_CLAMP``.
            theta_grid: Poloidal angles to scan.
            anchor_rz: ``(R, Z)`` of the previously tracked member. When given, a candidate
                further than ``max_move`` from it is rejected as a different member. The
                continuity gate is applied only against this anchor.
            max_move: Continuity-gate radius [m]. Ignored when ``None`` or without ``anchor_rz``.
            return_all: Return every accepted member instead of just the first.
            require_poloidal_occurrences: If set, accept only this iota numerator, pinning the
                chain identity.
            n_s_scan: ``s`` samples per theta; defaults to ``self.n_s``.

        Returns:
            tuple: ``(result, best_polish, stats)``.

                - **result** -- the first accepted member as a dict, or (with ``return_all``) the
                  residual-sorted, ``(R,Z)``-deduplicated list of accepted members. ``None`` (or
                  ``[]``) if nothing passed the gates.
                - **best_polish** -- the lowest-residual polish attempt regardless of acceptance,
                  used to explain a miss; ``None`` if no candidate was polished.
                - **stats** -- ``{n_cands, n_rm, n_iota}``: candidates scanned, candidates that
                  passed the residual+continuity gates, and members that passed the resonance gate.
        """
        s_lo, s_hi = max(-S_CLAMP, s_lo), min(S_CLAMP, s_hi)
        cands, _ = self._scan_fixed_points(
            specout, vol, self.n_transits, theta_grid=theta_grid, s_range=(s_lo, s_hi),
            n_s=(n_s_scan or self.n_s), method="RK45", rtol=SCAN_RTOL, atol=SCAN_RTOL,
        )  # fmt: skip

        confirmed, best_polish, n_rm, min_iota_rel = [], None, 0, None
        for c in sorted(cands, key=lambda c: c["residual"]):
            pol = self._root_fixed_point(
                specout, vol, self.n_transits, (c["s"], c["theta"]), **self.root_kw
            )
            if np.isfinite(pol.residual) and (
                best_polish is None or pol.residual < best_polish["residual"]
            ):
                best_polish = {"s": pol.s, "theta": pol.theta, "residual": pol.residual,
                               "converged": bool(pol.converged), "iota_rel": None}  # fmt: skip
            if not pol.converged:
                continue
            if (
                anchor_rz is not None
                and max_move is not None
                and np.hypot(pol.R - anchor_rz[0], pol.Z - anchor_rz[1]) > max_move
            ):
                continue  # polished to a different member than the one we track
            n_rm += 1
            # Resonance filter: a true fixed point is exactly rational; the integer round of
            # |iota|*repeat_period is the resonance numerator, which must be a positive multiple
            # of Nfp and coprime with repeat_period.
            _, _, iota_tr, success = specout.trace_one_fieldline(
                np.array([pol.s, pol.theta]), vol, num_phi_planes=1, num_transits=self.n_transits
            )
            if not success:
                continue
            iota_emp = -float(iota_tr)  # physical (VMEC) sign convention
            ntor_p = int(round(abs(iota_emp) * self.repeat_period)) if np.isfinite(iota_emp) else 0
            if not (
                ntor_p > 0
                and ntor_p % self.nfp == 0
                and (
                    int(np.gcd(ntor_p, self.repeat_period)) == 1
                    # non-coprime pair explicitly requested (e.g. W7-X 5/5): accept it
                    or ntor_p == require_poloidal_occurrences
                )
                and (require_poloidal_occurrences is None or ntor_p == require_poloidal_occurrences)
            ):
                continue
            rel = abs(abs(iota_emp) - ntor_p / self.repeat_period) / (ntor_p / self.repeat_period)
            min_iota_rel = rel if min_iota_rel is None else min(min_iota_rel, rel)
            entry = {
                "s": pol.s,
                "theta": pol.theta,
                "R": pol.R,
                "Z": pol.Z,
                "residual": pol.residual,
                "ntor": ntor_p,
                "iota": iota_emp,
                "iota_rel": rel,
            }
            if not return_all:
                return entry, best_polish, {"n_cands": len(cands), "n_rm": n_rm, "n_iota": 1}
            confirmed.append(entry)

        # Collapse duplicates: the same fixed point polished from adjacent theta seeds.
        deduped = []
        for e in sorted(confirmed, key=lambda e: e["residual"]):
            if all(np.hypot(e["R"] - k["R"], e["Z"] - k["Z"]) > self.dedup_tol for k in deduped):
                deduped.append(e)
        if best_polish is not None:
            best_polish["iota_rel"] = min_iota_rel
        return (
            (deduped if return_all else None),
            best_polish,
            {"n_cands": len(cands), "n_rm": n_rm, "n_iota": len(deduped)},
        )

    def classify_fp(self, specout, vol, sth):
        """O/X verdict for a fixed point, reusing a warmed per-volume classifier.

        Returns ``None`` if the point could not be classified. A non-stellarator-symmetric
        equilibrium raises from the classifier constructor: that is a precondition error, not a
        per-point failure, so it is deliberately not swallowed here.
        """
        clf = self.clf_cache.get(vol)
        if clf is None:
            clf = self._FixedPointClassifier(
                specout, vol, self.repeat_period, method="DOP853",
                rtol=CLASSIFY_RTOL, atol=CLASSIFY_RTOL,
            )  # fmt: skip
            self.clf_cache[vol] = clf
        return self._classify_point(clf, sth)

    # -- The stepper ------------------------------------------------------------------------

    def step(self, specout, i) -> StepReport:
        """Process one equilibrium into row ``i`` and advance the carry-state.

        Returns a :class:`StepReport` for the caller to :meth:`log`.
        """
        warns: list[str] = []
        if self.nfp < 0:
            self.nfp = int(specout.input.physics.Nfp)

        rax, zax, axis_disp = self._locate_axis(specout, i, warns)

        if not self.island_mode:
            self.no_fp[i] = True
            return StepReport(rax, zax, axis_disp, no_fp=True, reason="axis-only mode", warns=warns)

        self.clf_cache.clear()
        acq = self._acquire_follow(specout, i, warns) if self.anchor_rz is not None else _Acq()
        if acq.fp is None and self.discover:
            acq = self._acquire_discovery(specout, i, warns, acq)

        if acq.fp is None:
            self.no_fp[i] = True
            return StepReport(
                rax,
                zax,
                axis_disp,
                no_fp=True,
                reason=self._no_fp_reason(acq.best_diag),
                acq_msg=acq.msg,
                warns=warns,
            )

        self._record_fp(specout, i, acq.fp, acq.cls, warns)
        displacement, residue_delta = self._update_carry_state(i, acq.fp, acq.from_discovery)
        return StepReport(rax, zax, axis_disp, no_fp=False, acq_msg=acq.msg, warns=warns,
                          displacement=displacement, residue_delta=residue_delta)  # fmt: skip

    def _locate_axis(self, specout, i, warns):
        """Find the magnetic axis into row ``i``; a failure leaves the row NaN and continues.

        Returns ``(rax, zax, axis_disp)``, ``axis_disp`` being the move from the previous axis.
        """
        axis_disp = np.nan
        rax, zax, _, _ = self._find_axis(specout, max_repeats=self.max_axis_repeats)
        if rax is None:
            warns.append("axis finding did not converge")
            return np.nan, np.nan, axis_disp
        self.axis_rz[i] = [rax, zax]
        if self.prev_axis is not None:
            axis_disp = float(np.linalg.norm(self.axis_rz[i] - self.prev_axis))
        self.prev_axis = self.axis_rz[i].copy()
        return rax, zax, axis_disp

    def _acquire_follow(self, specout, i, warns) -> _Acq:
        """Follow the chain from the previous anchor with a narrow scan.

        A miss (or landing on an X-point) leaves ``fp=None`` so the caller falls back to discovery.
        """
        acq = _Acq(follow_missed=True)
        try:
            cs, ct = specout.get_st_coords(self.volume, self.anchor_rz[0], self.anchor_rz[1], 0.0)
            seed_s = float(np.asarray(cs).ravel()[0])
            seed_th = (
                self.anchor_th if self.anchor_th is not None else float(np.asarray(ct).ravel()[0])
            )
        except Exception as exc:
            warns.append(f"anchor inversion failed in vol{self.volume} - {exc}")
            return acq

        have_prev = self.prev_s is not None and np.isfinite(self.prev_s)
        seed_s_used = self.prev_s if have_prev else seed_s
        geom_ceiling = self.follow_window if self.follow_window is not None else self.geom_ceiling
        drift_gate = (
            self.max_jump_floor if not np.isfinite(self.prev_disp)
            else max(self.max_jump_floor, DRIFT_FACTOR * self.prev_disp)
        )  # fmt: skip
        max_move = min(geom_ceiling, drift_gate)
        th_grid = np.linspace(seed_th - DTHETA, seed_th + DTHETA, N_THETA)
        fp, best_diag, st = self.find_fp(
            specout, self.volume, seed_s_used - FOLLOW_DS, seed_s_used + FOLLOW_DS, th_grid,
            anchor_rz=self.anchor_rz, max_move=max_move,
            require_poloidal_occurrences=self.prev_poloidal_occurrences,
        )  # fmt: skip
        acq.best_diag = best_diag
        if fp is None:
            return acq
        acq.msg = f"[follow] from anchor ({st['n_cands']} cand, window {max_move * 1e3:.0f} mm)"
        # An X landing is a follow miss -- we track the O-point, so fall back to discovery.
        cls = self.classify_fp(specout, self.volume, (fp["s"], fp["theta"]))
        if cls is not None and cls["kind"] == "x":
            warns.append("follow landed on an X-point - re-discovering (O preferred)")
            return acq
        acq.fp, acq.cls, acq.follow_missed = fp, cls, False
        return acq

    def _acquire_discovery(self, specout, i, warns, prior: _Acq) -> _Acq:
        """Full-volume scan for the chain, preferring an O-point (island centre)."""
        acq = _Acq(best_diag=prior.best_diag)
        th_full = np.linspace(0.0, 2.0 * np.pi, self.disc_n_theta, endpoint=False)
        if self.anchor_th is not None:  # always seed the last known O-point theta
            th_full = np.unique(np.append(th_full, float(self.anchor_th) % (2.0 * np.pi)))
        confirmed, best_diag, _ = self.find_fp(
            specout, self.volume, -S_CLAMP, S_CLAMP, th_full, return_all=True,
            require_poloidal_occurrences=self.prev_poloidal_occurrences, n_s_scan=self.disc_n_s,
        )  # fmt: skip
        if best_diag is not None and (
            acq.best_diag is None or best_diag["residual"] < acq.best_diag["residual"]
        ):
            acq.best_diag = best_diag
        if not confirmed:
            return acq

        res_str = (
            f"{self.prev_poloidal_occurrences}/{self.repeat_period}"
            if self.prev_poloidal_occurrences else f"k*Nfp (period-{self.repeat_period})"
        )  # fmt: skip
        lbl = "[cold] re-discovered" if prior.follow_missed else "[cold] discovered"
        # Prefer an O-point; order by theta-stability (nearest the last member) then residual.
        if self.last_member_th is not None:
            m = self.last_member_th
            order = sorted(confirmed, key=lambda e: (
                abs((e["theta"] - m + np.pi) % (2 * np.pi) - np.pi), e["residual"]))  # fmt: skip
        else:
            order = confirmed
        for e in order:
            cls = self.classify_fp(specout, self.volume, (e["s"], e["theta"]))
            if cls is not None and cls["kind"] == "o":
                acq.fp, acq.cls = e, cls
                acq.msg = f"{lbl} {len(confirmed)} members on {res_str}; locked O-point"
                break
        else:
            acq.fp = order[0]
            acq.cls = self.classify_fp(specout, self.volume, (acq.fp["s"], acq.fp["theta"]))
            acq.msg = f"{lbl} {len(confirmed)} members on {res_str}; no O -> nearest-seed member"
        acq.from_discovery = True
        return acq

    def _no_fp_reason(self, best_diag) -> str:
        """One-line diagnostic explaining why no fixed point was committed this step."""
        if best_diag is None:
            return "no polishable scan candidate"
        if not best_diag["converged"]:
            edge = best_diag["residual"] < self.root_ftol and abs(best_diag["s"]) >= S_CLAMP
            why = (
                "converged at the volume edge - chain likely in the neighbouring volume"
                if edge
                else "failed ftol gate"
            )
        elif best_diag.get("iota_rel") is not None:
            why = f"passed ftol; failed resonance filter (iota error {best_diag['iota_rel']:.1e})"
        else:
            why = "passed ftol; failed continuity gate"
        return (
            f"best polish {best_diag['residual']:.1e} @ s={best_diag['s']:+.4f} "
            f"th={best_diag['theta']:.2f} ({why})"
        )

    def _record_fp(self, specout, i, fp, cls, warns):
        """Record the located fixed point (position, half-period, iota, O/X) into row ``i``."""
        s_sel, th_sel, r_sel, z_sel = fp["s"], fp["theta"], fp["R"], fp["Z"]
        self.residual[i] = fp["residual"]
        self.island_rz[i] = [r_sel, z_sel]
        self.island_sth[i] = [s_sel, th_sel]
        _, RZ_mp, _, _ = specout.trace_one_fieldline(
            [s_sel, th_sel], self.volume, num_phi_planes=2, num_transits=1
        )
        self.island_rz_half_period[i] = [float(RZ_mp[0, 1]), float(RZ_mp[1, 1])]
        self.iota[i], self.ntor[i], self.iota_rel[i] = fp["iota"], fp["ntor"], fp["iota_rel"]
        if fp["iota_rel"] > IOTA_REL_WARN:
            warns.append(f"iota_rel {self.iota_rel[i]:.1e} unusually large for a true fixed point")
        if cls is None:  # follow may not have classified; discovery always does
            cls = self.classify_fp(specout, self.volume, (s_sel, th_sel))
        if cls is not None:
            self.kind[i], self.residue[i] = cls["kind"], cls["residue"]
            self.greene_trace[i], self.greene_det[i], self.greene_disc[i] = (
                cls["trace"],
                cls["det"],
                cls["disc"],
            )

    def _update_carry_state(self, i, fp, from_discovery):
        """Advance the carry-state and return ``(displacement, residue_delta)`` for row ``i``.

        Both are the convergence metric and are meaningful only on anchor-follow steps; a cold
        re-acquire may land on a different member, so they stay NaN there.
        """
        r_sel, z_sel, s_sel, th_sel = fp["R"], fp["Z"], fp["s"], fp["theta"]
        if np.isfinite(self.axis_rz[i, 0]):  # continuity gate from the island geometry
            r = np.hypot(r_sel - self.axis_rz[i, 0], z_sel - self.axis_rz[i, 1])
            self.follow_window = FOLLOW_WINDOW_FRAC * 2.0 * np.pi * r / self.repeat_period

        displacement = residue_delta = np.nan
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
        return displacement, residue_delta

    # -- Output -----------------------------------------------------------------------------

    def log(self, i, report: StepReport, *, verbose=False, it=None):
        """Print a one-line summary of step ``i`` (a short extra block when ``verbose``)."""
        tag = i if it is None else it
        for w in report.warns:
            print(f"fp[{tag}]: ! {w}", flush=True)
        line = f"fp[{tag}]: axis ({report.rax:.5f}, {report.zax:+.5f})"
        if np.isfinite(report.axis_disp):
            line += f" d={report.axis_disp:.1e}"
        if report.no_fp:
            print(f"{line} | no_fp - {report.reason}", flush=True)
            return
        r_sel, z_sel = self.island_rz[i]
        line += f" | isl v{self.volume} ({r_sel:.4f}, {z_sel:+.4f}) [{self.provenance[i]}]"
        if np.isfinite(report.displacement):
            line += f" d={report.displacement:.1e}"
        if self.kind[i]:
            line += f" | {self.kind[i].upper()} Greene={self.residue[i]:.3e}"
            if np.isfinite(report.residue_delta):
                line += f" ({report.residue_delta:+.1e})"
        print(line, flush=True)
        if verbose:
            gate = "ok" if self.residual[i] < self.root_ftol else "NO"
            print(
                f"         res {self.residual[i]:.1e}<{self.root_ftol:.0e} {gate} | "
                f"iota {self.iota[i]:+.4f}={self.ntor[i]}/{self.repeat_period} | {report.acq_msg}",
                flush=True,
            )

    def result_dict(self, iter_indices, start_at_iter, stride, reverse) -> dict:
        """Assemble the per-step arrays and scalar metadata into the output schema."""
        arrays = ("vol_tracked", "axis_rz", "island_rz", "island_sth", "island_rz_half_period",
                  "residual", "iota", "ntor", "iota_rel", "kind", "residue", "greene_trace",
                  "greene_det", "greene_disc", "provenance", "no_fp")  # fmt: skip
        out = {"iteration_indices": iter_indices}
        out.update({name: getattr(self, name) for name in arrays})
        out.update(
            repeat_period=self.repeat_period if self.repeat_period is not None else -1,
            volume_start=self.volume,
            nfp=self.nfp,
            start_at_iter=int(start_at_iter),
            stride=int(stride),
            root_ftol=float(self.root_ftol),
            poloidal_occurrences=(
                int(self.poloidal_occurrences) if self.poloidal_occurrences else -1
            ),
            reverse=bool(reverse),
            discover=bool(self.discover),
        )
        return out

    def flush_to_h5(self, h5_path, n_done):
        """Write the first ``n_done`` steps to ``/iterations/fp``, recreating the group each call.

        Rewriting after every step keeps the file current so it survives a scancel of the SLURM job.
        """
        import h5py

        res = self.result_dict(np.arange(n_done, dtype=int), 0, 1, False)
        try:
            with h5py.File(h5_path, "a") as f:
                grp = f.require_group("iterations")
                if "fp" in grp:
                    del grp["fp"]
                fp_grp = grp.create_group("fp")
                string_dt = h5py.string_dtype()
                for k, v in res.items():
                    if not np.ndim(v):
                        fp_grp.attrs[k] = v
                        continue
                    arr = np.asarray(v)
                    if arr.dtype == object or arr.dtype.kind == "U":
                        fp_grp.create_dataset(k, data=arr[:n_done].astype(object), dtype=string_dt)
                    else:
                        fp_grp.create_dataset(k, data=arr[:n_done])
        except Exception as exc:
            print(f"[fp-track] flush_to_h5 failed - {exc}", flush=True)
