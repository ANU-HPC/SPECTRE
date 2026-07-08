# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT
"""Reliable O/X (elliptic/hyperbolic) classification of SPECTRE island fixed points.

Implements Greene's residue ``R=(2-tr)/4`` + eigenvalue discriminant ``disc=tr**2-4*det`` via a
central-FD tangent map at step size ``h = H_REL * R00`` (``R00`` = major radius of the hosting
volume's outer interface, so the step tracks the machine scale); classify by ``sign(disc)``.

At tight integrator tolerance (``DOP853, rtol=1e-12``) ``|det-1|`` drops to ~1e-8, making the
discriminant sign resolved in the common case. The classifier drives ``trace_field_single``
directly to bypass the ``trace_one_fieldline`` tracer-cache bug that otherwise silently traces at
``rtol~1e-7`` and makes near-parabolic O/X labels noise.
"""

import numpy as np
from scipy.optimize import root

# Central-FD step for the tangent map, as a fraction of the volume's major radius R00
# (h = H_REL * R00 metres; reproduces the legacy hardcoded h=1e-5 m on R00 = 1 m machines).
H_REL = 1e-5


class FixedPointClassifier:
    """Classify SPECTRE island fixed points as O (elliptic) or X (hyperbolic)."""

    def __init__(self, specout, vol, repeat_period, *, method="DOP853", rtol=1e-12, atol=1e-12):
        """Build a classifier for one volume, warming and caching the single-field-line tracer.

        Args:
            specout: A SPECTRE ``SPECTREout`` equilibrium.
            vol: Volume index hosting the chain (``>= 1``).
            repeat_period: Island count = denominator of ``iota = po/repeat_period``
                where ``po`` = ``poloidal_occurrences`` (the Poincare map is repeat_period-fold).
            method: Integrator for the tight-tolerance traces.
            rtol: Relative integrator tolerance.
            atol: Absolute integrator tolerance.
        """
        self.specout, self.vol, self.repeat_period = specout, int(vol), int(repeat_period)
        self.ntr = self.repeat_period + 1
        self.tg = dict(method=method, rtol=rtol, atol=atol)
        # Warm up + cache the Field_Tracer, then drive trace_field_single DIRECTLY so our
        # per-call method/rtol/atol take effect (the trace_one_fieldline cache bug).
        specout.trace_one_fieldline(
            np.array([-0.9, 0.0]), self.vol, num_phi_planes=1, num_transits=2
        )
        self.tracer = specout._tracer_single
        im, in_ = specout.output.im, specout.output.in_
        i00 = int(np.flatnonzero((im == 0) & (in_ == 0))[0])
        r00 = abs(float(specout.output.Rbc[self.vol + 1][i00]))
        self.h = H_REL * r00

    def refine(self, guess_sth, tol=1e-12):
        """Newton-refine a fixed point from an ``(s, theta)`` guess."""
        s0, th0 = guess_sth

        def _disp(x):
            s, th = x
            if not (-0.999 < s < 0.999):
                return np.array([1e2, 1e2])
            _, RZ, _, flag = self.tracer.trace_field_single(
                np.array([s, th % (2 * np.pi)]),
                self.vol,
                num_phi_planes=1,
                num_transits=self.ntr,
                **self.tg,
            )
            if flag:
                return np.array([1e2, 1e2])
            r0, z0 = self.specout.get_rz_coords(self.vol, s, th, 0.0)
            return np.array([RZ[0][-1], RZ[1][-1]]) - np.array(
                [float(np.ravel(r0)[0]), float(np.ravel(z0)[0])]
            )

        sol = root(_disp, [s0, th0], tol=tol)
        sF, thF = float(sol.x[0]), float(sol.x[1]) % (2 * np.pi)
        rF, zF = self.specout.get_rz_coords(self.vol, sF, thF, 0.0)
        return dict(
            R=float(np.ravel(rF)[0]),
            Z=float(np.ravel(zF)[0]),
            s=sF,
            theta=thF,
            residual=float(np.linalg.norm(sol.fun)),
        )

    def greene(self, rF, zF, *, h=None, guess_sth=None, verbose=False):
        """Greene residue + discriminant via a central-FD tangent map at step size ``h``.

        ``h`` is in metres and defaults to ``H_REL * R00`` computed at init.
        """
        h = self.h if h is None else h
        cols = []
        for dR, dZ in ((h, 0.0), (-h, 0.0), (0.0, h), (0.0, -h)):
            try:
                seed = guess_sth or (0.0, 0.0)
                s, th = self.specout.get_st_coords(self.vol, rF + dR, zF + dZ, 0.0, *seed)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"tangent map: get_st_coords failed at (R,Z)=({rF + dR:.4f},{zF + dZ:.4f})"
                    " - point may be outside the volume"
                ) from exc
            _, RZ, _, flag = self.tracer.trace_field_single(
                np.array([s, th % (2 * np.pi)]),
                self.vol,
                num_phi_planes=1,
                num_transits=self.ntr,
                **self.tg,
            )
            if flag:
                raise RuntimeError("tangent map: field-line tracer flagged failure")
            cols.append(np.array([RZ[0][-1], RZ[1][-1]]))
        J = np.column_stack([(cols[0] - cols[1]) / (2 * h), (cols[2] - cols[3]) / (2 * h)])
        tr, det = float(np.trace(J)), float(np.linalg.det(J))
        disc = tr * tr - 4 * det
        kind = "O" if disc < 0 else "X"
        if verbose:
            print(
                f"  h={h:.1e} | trace={tr:.6f} | det={det:.6f} | "
                f"residue={(2 - tr) / 4:+.3e} | disc={disc:+.3e} | {kind}"
            )
        return dict(
            trace=tr,
            det=det,
            residue=(2 - tr) / 4,
            disc=disc,
            eigvals=np.linalg.eigvals(J),
            kind=kind,
        )


def classify_point(clf, guess_sth, *, refine=True):
    """Classify one fixed point O/X with a pre-built ``FixedPointClassifier``.

    Reuses the warmed tracer across many points in a checkpoint. Tight-tol Greene residue gives the
    sign. ``stable=False`` (near-parabolic) is an accepted failure mode.

    Args:
        clf: A warmed :class:`FixedPointClassifier`.
        guess_sth: ``(s, theta)`` of the point.
        refine: Newton-refine the guess first (else classify at ``guess_sth`` as given).

    Returns:
        A flat dict with ``kind`` ('o'/'x'), ``residue``, ``disc``, ``trace``, ``det``,
        and the refined ``(R, Z, s, theta)``.
    """
    if refine:
        fp = clf.refine(guess_sth)
        sth = (fp["s"], fp["theta"])
    else:
        s, th = guess_sth
        R, Z = clf.specout.get_rz_coords(clf.vol, s, th, 0.0)
        fp = dict(R=float(np.ravel(R)[0]), Z=float(np.ravel(Z)[0]), s=s, theta=th, residual=np.nan)
        sth = guess_sth
    g = clf.greene(fp["R"], fp["Z"], guess_sth=sth)
    return dict(
        R=fp["R"],
        Z=fp["Z"],
        s=fp["s"],
        theta=fp["theta"],
        kind=g["kind"].lower(),
        residue=g["residue"],
        disc=g["disc"],
        trace=g["trace"],
        det=g["det"],
    )
