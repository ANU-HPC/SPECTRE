# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT
"""Reliable O/X (elliptic/hyperbolic) classification of SPECTRE island fixed points.

Implements the procedure validated in ``fixed_point_classification_findings.md`` §6.2:

  1. Refine the fixed point at TIGHT tolerance (``DOP853, rtol=1e-12``), driving
     ``trace_field_single`` DIRECTLY -- this bypasses the ``trace_one_fieldline`` tracer-cache bug
     that otherwise silently traces at ``rtol~1e-7`` and makes near-parabolic O/X labels noise.
  2. Greene residue ``R=(2-tr)/4`` + eigenvalue discriminant ``disc=tr**2-4*det`` via a central-FD
     tangent map over an h-ladder; keep the h with ``|det-1|`` smallest; classify by ``sign(disc)``.

At tight tolerance ``|det-1|`` drops to ~1e-8, making the discriminant sign resolved in the common
case. When ``stable=False`` (h-ladder rungs disagree or ``|det-1|`` is large) the best-h verdict is
still used — this is an accepted failure mode for genuinely near-parabolic points.
"""

import numpy as np
from scipy.optimize import root


class FixedPointClassifier:
    """Classify SPECTRE island fixed points as O (elliptic) or X (hyperbolic)."""

    def __init__(self, specout, vol, mpol, *, method="DOP853", rtol=1e-12, atol=1e-12):
        """Build a classifier for one volume, warming and caching the single-field-line tracer.

        Args:
            specout: A SPECTRE ``SPECTREout`` equilibrium.
            vol: Volume index hosting the chain (``>= 1``).
            mpol: Island count = denominator of ``iota = ntor/mpol`` (the return map is mpol-fold).
            method: Integrator for the tight-tolerance traces.
            rtol: Relative integrator tolerance.
            atol: Absolute integrator tolerance.
        """
        self.specout, self.vol, self.mpol = specout, int(vol), int(mpol)
        self.ntr = self.mpol + 1
        self.tg = dict(method=method, rtol=rtol, atol=atol)
        # Warm up + cache the Field_Tracer, then we drive trace_field_single DIRECTLY so our
        # per-call method/rtol/atol take effect (the trace_one_fieldline cache bug, findings §2).
        specout.trace_one_fieldline(
            np.array([-0.9, 0.0]), self.vol, num_phi_planes=1, num_transits=2)
        self.tracer = specout._tracer_single

    def refine(self, guess_rz, tol=1e-12, guess_sth=None):
        """Newton-refine a fixed point from an ``(R,Z)`` guess (``guess_sth`` skips the inversion)."""
        s0, th0 = guess_sth if guess_sth is not None else self.specout.get_st_coords(self.vol, *guess_rz, 0.0)

        def _disp(x):
            s, th = x
            if not (-0.999 < s < 0.999):
                return np.array([1e2, 1e2])
            _, RZ, _, flag = self.tracer.trace_field_single(
                np.array([s, th % (2 * np.pi)]), self.vol,
                num_phi_planes=1, num_transits=self.ntr, **self.tg)
            if flag:
                return np.array([1e2, 1e2])
            r0, z0 = self.specout.get_rz_coords(self.vol, s, th, 0.0)
            return (np.array([RZ[0][-1], RZ[1][-1]])
                    - np.array([float(np.ravel(r0)[0]), float(np.ravel(z0)[0])]))

        sol = root(_disp, [s0, th0], tol=tol)
        sF, thF = float(sol.x[0]), float(sol.x[1]) % (2 * np.pi)
        rF, zF = self.specout.get_rz_coords(self.vol, sF, thF, 0.0)
        return dict(R=float(np.ravel(rF)[0]), Z=float(np.ravel(zF)[0]),
                    s=sF, theta=thF, residual=float(np.linalg.norm(sol.fun)))

    def greene(self, rF, zF, *, h_list=(5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4), guess_sth=None,
               verbose=False):
        """Greene residue + discriminant via an h-ladder; keep the h with smallest ``|det-1|``."""
        rows = []
        for h in h_list:
            # Central-FD tangent (monodromy) map of the return map at step h. Each offset is
            # inverted to (s, th) warm-started at the fixed point itself; an offset that cannot
            # be inverted (outside the volume near an edge, or a failed root) drops this h rung
            # only — the smaller rungs of the ladder still classify.
            cols = []
            for dR, dZ in ((h, 0.0), (-h, 0.0), (0.0, h), (0.0, -h)):
                try:
                    s, th = self.specout.get_st_coords(self.vol, rF + dR, zF + dZ, 0.0, *(guess_sth or (0.0, 0.0)))
                except RuntimeError:
                    break
                _, RZ, _, flag = self.tracer.trace_field_single(
                    np.array([s, th % (2 * np.pi)]), self.vol,
                    num_phi_planes=1, num_transits=self.ntr, **self.tg)
                if flag:
                    break
                cols.append(np.array([RZ[0][-1], RZ[1][-1]]))
            if len(cols) < 4:
                continue
            J = np.column_stack([(cols[0] - cols[1]) / (2 * h), (cols[2] - cols[3]) / (2 * h)])
            tr, det = float(np.trace(J)), float(np.linalg.det(J))
            disc = tr * tr - 4 * det
            rows.append(dict(h=h, trace=tr, det=det, residue=(2 - tr) / 4, disc=disc,
                             eig=np.linalg.eigvals(J), det_err=abs(det - 1.0),
                             kind=("O" if disc < 0 else "X")))
        if not rows:
            raise RuntimeError("tangent map could not be evaluated at any h (out of volume?)")
        best = min(rows, key=lambda r: r["det_err"])
        clean = [r for r in rows if r["det_err"] < 1e-3]
        stable = bool(clean) and len({r["kind"] for r in clean}) == 1
        if verbose:
            print(f"  {'h':>8} | {'trace':>11} | {'|det-1|':>9} | "
                  f"{'residue':>11} | {'disc':>11} | kind")
            for r in rows:
                print(f"  {r['h']:8.1e} | {r['trace']:11.6f} | {r['det_err']:9.2e} | "
                      f"{r['residue']:+11.3e} | {r['disc']:+11.3e} | {r['kind']}")
        return dict(trace=best["trace"], det=best["det"], residue=best["residue"],
                    disc=best["disc"], eigvals=best["eig"], kind=best["kind"], stable=stable,
                    h_best=best["h"], det_err=best["det_err"], rows=rows)


def classify_point(clf, guess_rz, *, refine=True, guess_sth=None):
    """Classify one fixed point O/X with a pre-built ``FixedPointClassifier``.

    Reuses the warmed tracer across many points in a checkpoint. Tight-tol Greene residue gives the
    sign. When ``stable=False`` (h-ladder rungs disagree or noisy ``|det-1|``) the best-h verdict is
    still used; this is an accepted failure mode for genuinely near-parabolic points.

    Args:
        clf: A warmed :class:`FixedPointClassifier`.
        guess_rz: ``(R, Z)`` guess of the fixed point.
        refine: Newton-refine the guess first (else classify at ``guess_rz`` as given).
        guess_sth: Known ``(s, theta)`` of the point; warm-starts every (R,Z)->(s,th) inversion
            (they are stochastic and can fail near a volume edge without it).

    Returns:
        A flat dict with ``kind`` ('o'/'x'), ``residue``, ``disc``, ``trace``, ``det``, ``det_err``,
        ``stable``, and the refined ``(R, Z, s, theta)``.
    """
    fp = clf.refine(guess_rz, guess_sth=guess_sth) if refine else dict(
        R=guess_rz[0], Z=guess_rz[1], s=np.nan, theta=np.nan, residual=np.nan)
    sth = (fp["s"], fp["theta"]) if np.isfinite(fp["s"]) else guess_sth
    g = clf.greene(fp["R"], fp["Z"], guess_sth=sth)
    return dict(R=fp["R"], Z=fp["Z"], s=fp["s"], theta=fp["theta"], kind=g["kind"].lower(),
                residue=g["residue"], disc=g["disc"], trace=g["trace"], det=g["det"],
                det_err=g["det_err"], stable=bool(g["stable"]))


