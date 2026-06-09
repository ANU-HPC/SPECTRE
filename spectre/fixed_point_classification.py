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
  3. TRUST gate: accept the residue verdict only if the clean-det steps agree AND
     ``|disc| >> |det-1|`` (near-parabolic guard). Otherwise fall back to:
  4. Model-free ORBIT TOPOLOGY -- seed just inside the island, trace ~1200 same-island returns,
     label by radial growth (escape -> X, bounded -> O).

At tight tolerance ``|det-1|`` drops to ~1e-8, so the trust margin is large and the cheap residue
verdict is used in the common case; the expensive orbit fallback fires only for genuinely marginal
points.
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

    def rz(self, s, th):
        """Return the ``(R, Z)`` at ``phi=0`` of the flux coordinate ``(s, th)``."""
        r, z = self.specout.get_rz_coords(self.vol, s, th, 0.0)
        return float(np.ravel(r)[0]), float(np.ravel(z)[0])

    def st(self, r, z):
        """Return the flux coordinate ``(s, th)`` at ``phi=0`` of the point ``(r, z)``."""
        s, th = self.specout.get_st_coords(self.vol, r, z, 0.0)
        return float(np.ravel(s)[0]), float(np.ravel(th)[0])

    def image_rz(self, s, th, **over):
        """Return the ``(R,Z)`` image of the mpol-fold return map from ``(s,th)``."""
        tg = {**self.tg, **over}
        _, RZ, _, flag = self.tracer.trace_field_single(
            np.array([s, th % (2 * np.pi)]), self.vol,
            num_phi_planes=1, num_transits=self.ntr, **tg)
        return None if flag else np.array([RZ[0][-1], RZ[1][-1]])

    def return_disp_sth(self, x, **over):
        """Return the ``(R,Z)`` return-map displacement at flux coordinate ``x=(s, th)``."""
        s, th = x
        if not (-0.999 < s < 0.999):
            return np.array([1e2, 1e2])
        img = self.image_rz(s, th, **over)
        if img is None:
            return np.array([1e2, 1e2])
        r0, z0 = self.rz(s, th)
        return img - np.array([r0, z0])

    def refine(self, guess_rz, tol=1e-12):
        """Newton-refine a fixed point from an ``(R,Z)`` guess."""
        s0, th0 = self.st(*guess_rz)
        sol = root(self.return_disp_sth, [s0, th0], tol=tol)
        sF, thF = float(sol.x[0]), float(sol.x[1]) % (2 * np.pi)
        rF, zF = self.rz(sF, thF)
        return dict(R=rF, Z=zF, s=sF, theta=thF, residual=float(np.linalg.norm(sol.fun)))

    def tangent(self, rF, zF, h):
        """Central-FD tangent (monodromy) map of the return map at ``(rF, zF)`` with step ``h``."""
        cols = []
        for dR, dZ in ((h, 0.0), (-h, 0.0), (0.0, h), (0.0, -h)):
            s, th = self.st(rF + dR, zF + dZ)
            img = self.image_rz(s, th)
            if img is None:
                return None
            cols.append(img)
        return np.column_stack([(cols[0] - cols[1]) / (2 * h), (cols[2] - cols[3]) / (2 * h)])

    def greene(self, rF, zF, *, h_list=(5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4), verbose=False):
        """Greene residue + discriminant via an h-ladder; keep the h with smallest ``|det-1|``."""
        rows = []
        for h in h_list:
            J = self.tangent(rF, zF, h)
            if J is None:
                continue
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

    def orbit_topology(self, rF, zF, *, ds=0.006, n_transits=1200, rtol=1e-11, atol=1e-11):
        """Sample same-island returns next to ``(rF,zF)``: X escapes, O librates."""
        sO, thO = self.st(rF, zF)
        s_seed = min(max(sO + ds, -0.99), 0.99)
        _, RZ, _, _ = self.tracer.trace_field_single(
            np.array([s_seed, thO]), self.vol, num_phi_planes=1, num_transits=n_transits,
            method="DOP853", rtol=rtol, atol=atol)
        R = np.asarray(RZ[0])[::self.mpol] - rF
        Z = np.asarray(RZ[1])[::self.mpol] - zF
        rad = np.hypot(R, Z)
        ang = np.unwrap(np.arctan2(Z, R))
        w = max(len(rad) // 5, 1)
        win_max = [float(rad[i * w:(i + 1) * w].max()) for i in range(5)]
        growth = win_max[-1] / max(win_max[0], 1e-12)
        return dict(net_angle=float(ang[-1] - ang[0]), growth=growth, n_returns=len(R),
                    max_r_windows=win_max,
                    verdict=("X (escapes)" if growth > 2.0 else "O (librates)"))

    def orbit(self, rF, zF):
        """Return the mpol orbit punctures through ``(rF,zF)`` plus the measured iota."""
        s, th = self.st(rF, zF)
        _, RZ, iraw, _ = self.tracer.trace_field_single(
            np.array([s, th % (2 * np.pi)]), self.vol,
            num_phi_planes=1, num_transits=self.mpol, **self.tg)
        pts = [(float(np.asarray(RZ[0])[k]), float(np.asarray(RZ[1])[k])) for k in range(self.mpol)]
        return pts, -float(iraw)


def classify_point(clf, guess_rz, *, near_parabolic_margin=10.0, refine=True):
    """Classify one fixed point O/X with a pre-built ``FixedPointClassifier``.

    Reuses the warmed tracer across many points in a checkpoint. Tight-tol Greene gives the sign,
    cross-checked by orbit topology when near-parabolic.

    Args:
        clf: A warmed :class:`FixedPointClassifier`.
        guess_rz: ``(R, Z)`` guess of the fixed point.
        near_parabolic_margin: Trust gate -- accept the residue verdict only if ``|disc| >`` this
            ``* |det-1|``; else fall back to orbit topology.
        refine: Newton-refine the guess first (else classify at ``guess_rz`` as given).

    Returns:
        A flat dict with ``kind`` ('o'/'x'), ``residue``, ``disc``, ``trace``, ``det``, ``det_err``,
        ``stable``, ``method`` ('residue'/'orbit'), ``growth`` (NaN unless fallback fired), and the
        refined ``(R, Z, s, theta)``.
    """
    fp = clf.refine(guess_rz) if refine else dict(R=guess_rz[0], Z=guess_rz[1],
                                                  s=np.nan, theta=np.nan, residual=np.nan)
    g = clf.greene(fp["R"], fp["Z"])
    det_err = abs(g["det"] - 1.0)
    if g["stable"] and abs(g["disc"]) > near_parabolic_margin * max(det_err, 1e-12):
        kind, method, growth = g["kind"], "residue", np.nan
    else:
        o = clf.orbit_topology(fp["R"], fp["Z"])
        kind, method, growth = ("X" if o["growth"] > 2.0 else "O"), "orbit", float(o["growth"])
    return dict(R=fp["R"], Z=fp["Z"], s=fp["s"], theta=fp["theta"], kind=kind.lower(),
                residue=g["residue"], disc=g["disc"], trace=g["trace"], det=g["det"],
                det_err=det_err, stable=bool(g["stable"]), method=method, growth=growth)


def classify(specout, vol, mpol, guess_rz, *, near_parabolic_margin=10.0, verbose=True, **clf_kw):
    """Build a :class:`FixedPointClassifier` and classify one point (standalone convenience).

    For repeated use within a checkpoint, build the classifier once and call :func:`classify_point`.

    Args:
        specout: A SPECTRE ``SPECTREout`` equilibrium.
        vol: Volume index hosting the chain.
        mpol: Island count = denominator of ``iota``.
        guess_rz: ``(R, Z)`` guess of the fixed point.
        near_parabolic_margin: Trust gate for the residue verdict (see :func:`classify_point`).
        verbose: Print a one-line summary of the verdict.
        **clf_kw: Forwarded to :class:`FixedPointClassifier` (``method``/``rtol``/``atol``).

    Returns:
        The :func:`classify_point` dict.
    """
    clf = FixedPointClassifier(specout, vol, mpol, **clf_kw)
    out = classify_point(clf, guess_rz, near_parabolic_margin=near_parabolic_margin)
    if verbose:
        print(f"(R,Z)=({out['R']:.5f},{out['Z']:+.5f})  kind={out['kind'].upper()}  "
              f"residue={out['residue']:+.2e}  disc={out['disc']:+.2e}  decided-by={out['method']}"
              + (f" (orbit growth {out['growth']:.1f}x)" if out["method"] == "orbit" else ""))
    return out
