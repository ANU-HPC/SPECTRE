# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT
"""Reliable O/X (elliptic/hyperbolic) classification of SPECTRE island fixed points.

Implements Greene's residue ``R=(2-tr)/4`` + eigenvalue discriminant ``disc=tr**2-4*det`` from
the ANALYTIC tangent map: the variational equation ``dJ/dphi = A(phi) J``, ``J(0)=I``, is
integrated alongside the field line in flux coordinates (``Field_Tracer.trace_tangent_single``,
``A = d(Bs/Bz, Bt/Bz)/d(s, theta)`` from the spectral representation); classify by
``sign(disc)``. No finite-difference step, no fragile ``get_st_coords`` inversions; at
``DOP853, rtol=1e-12`` the accuracy diagnostic ``|det-1|`` is integrator-limited (~1e-9, vs
~1e-6 for central FD), so the discriminant sign is resolved even near-parabolic. Both radial
bases are supported (Chebyshev volumes and the Zernike axis volume, ``vol = 0``).

Restrictions: stellarator-symmetric equilibria only. The classifier drives the traces directly
to bypass the ``trace_one_fieldline`` tracer-cache bug that otherwise silently traces at
``rtol~1e-7`` and makes near-parabolic O/X labels noise.
"""

import numpy as np
from scipy.optimize import root


class FixedPointClassifier:
    """Classify SPECTRE island fixed points as O (elliptic) or X (hyperbolic)."""

    def __init__(self, specout, vol, repeat_period, *, method="DOP853", rtol=1e-12, atol=1e-12):
        """Build a classifier for one volume, warming and caching the single-field-line tracer.

        Args:
            specout: A SPECTRE ``SPECTREout`` equilibrium.
            vol: Volume index hosting the chain (0 = the singular axis volume).
            repeat_period: Island count = denominator of ``iota = po/repeat_period``
                where ``po`` = ``poloidal_occurrences`` (the Poincare map is repeat_period-fold).
            method: Integrator for the tight-tolerance traces.
            rtol: Relative integrator tolerance.
            atol: Absolute integrator tolerance.
        """
        self.specout, self.vol, self.repeat_period = specout, int(vol), int(repeat_period)
        self.ntr = self.repeat_period + 1
        self.tg = dict(method=method, rtol=rtol, atol=atol)
        if np.any(specout.vector_potential.Azo[self.vol]) or np.any(
            specout.vector_potential.Ato[self.vol]
        ):
            raise NotImplementedError(
                "FixedPointClassifier: the analytic tangent map supports stellarator-symmetric "
                "equilibria only (Azo/Ato are nonzero here)"
            )
        # Warm up + cache the Field_Tracer, then drive trace_field_single DIRECTLY so our
        # per-call method/rtol/atol take effect (the trace_one_fieldline cache bug).
        specout.trace_one_fieldline(
            np.array([-0.9, 0.0]), self.vol, num_phi_planes=1, num_transits=2
        )
        self.tracer = specout._tracer_single

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

    def greene(self, s, theta, *, verbose=False):
        """Greene residue + discriminant via the analytic (variational) tangent map.

        Integrates ``dJ/dphi = A(phi) J`` with ``J(0) = I`` alongside the field line over
        ``repeat_period`` field periods from the refined fixed point ``(s, theta)``, at the
        classifier's method/rtol/atol. ``|det-1|`` is the accuracy diagnostic (the return map
        is area-preserving at a fixed point).
        """
        _, J, flag = self.tracer.trace_tangent_single(
            np.array([s, theta % (2 * np.pi)]),
            self.vol,
            num_transits=self.repeat_period,
            **self.tg,
        )
        if flag:
            raise RuntimeError(
                "tangent map: variational integration flagged failure"
                " - point may have left the volume"
            )
        tr, det = float(np.trace(J)), float(np.linalg.det(J))
        disc = tr * tr - 4 * det
        kind = "O" if disc < 0 else "X"
        if verbose:
            print(
                f"  trace={tr:.6f} | det={det:.6f} | "
                f"residue={(2 - tr) / 4:+.3e} | disc={disc:+.3e} | {kind}"
            )
        return dict(trace=tr, det=det, residue=(2 - tr) / 4, disc=disc, kind=kind)


def classify_point(clf, guess_sth, *, refine=True):
    """Classify one fixed point O/X with a pre-built ``FixedPointClassifier``.

    Reuses the warmed tracer across many points in a checkpoint. The analytic tangent-map Greene
    residue gives the sign.

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
    else:
        s, th = guess_sth
        R, Z = clf.specout.get_rz_coords(clf.vol, s, th, 0.0)
        fp = dict(R=float(np.ravel(R)[0]), Z=float(np.ravel(Z)[0]), s=s, theta=th, residual=np.nan)
    g = clf.greene(fp["s"], fp["theta"])
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
