# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT
"""O/X (elliptic/hyperbolic) classification of SPECTRE island fixed points.

Classifies by the sign of the return-map tangent-map discriminant ``disc = tr**2 - 4*det`` and
reports Greene's residue ``R = (2 - tr)/4``. The tangent map is obtained analytically: the
variational equation ``dJ/dphi = A(phi) J`` with ``J(0) = I`` is integrated alongside the field
line in flux coordinates (``Field_Tracer.trace_tangent_single``), where
``A = d(Bs/Bz, Bt/Bz)/d(s, theta)`` comes from the spectral field representation. At
``DOP853, rtol=1e-12`` the area-preservation diagnostic ``|det - 1|`` reaches ~1e-9, resolving the
discriminant sign even for near-parabolic points. Both radial bases are supported (Chebyshev
volumes and the Zernike axis volume, ``vol=0``).

Traces run at the classifier's own tolerances, so the O/X label does not depend on the tracer's
default settings. Stellarator-symmetric equilibria only.
"""

import warnings

import numpy as np
from scipy.optimize import root

from spectre.fixed_points.primitives import LARGE, S_CLAMP


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
        # Warm up and cache the single-field-line tracer, then drive it directly at our tolerances.
        specout.trace_one_fieldline(
            np.array([-0.9, 0.0]), self.vol, num_phi_planes=1, num_transits=2
        )
        self.tracer = specout._tracer_single

    def refine(self, guess_sth, tol=1e-12):
        """Newton-refine a fixed point from an ``(s, theta)`` guess.

        Args:
            guess_sth: ``(s, theta)`` starting guess.
            tol: Tolerance handed to :func:`scipy.optimize.root`.

        Returns:
            dict | None: ``{R, Z, s, theta, residual}`` of the refined point, or ``None`` on an
            unexpected tracing/geometry error (a :class:`UserWarning` is issued) -- never raises.
        """
        s0, th0 = guess_sth

        def _disp(x):
            s, th = x
            if not (-S_CLAMP < s < S_CLAMP):
                return np.array([LARGE, LARGE])
            _, RZ, _, success = self.tracer.trace_field_single(
                np.array([s, th % (2 * np.pi)]),
                self.vol,
                num_phi_planes=1,
                num_transits=self.ntr,
                **self.tg,
            )
            if not success:
                return np.array([LARGE, LARGE])
            r0, z0 = self.specout.get_rz_coords(self.vol, s, th, 0.0)
            return np.array([RZ[0][-1], RZ[1][-1]]) - np.array(
                [float(np.ravel(r0)[0]), float(np.ravel(z0)[0])]
            )

        try:
            sol = root(_disp, [s0, th0], tol=tol)
            sF, thF = float(sol.x[0]), float(sol.x[1]) % (2 * np.pi)
            rF, zF = self.specout.get_rz_coords(self.vol, sF, thF, 0.0)
        except Exception as exc:  # noqa: BLE001 - one bad point must not abort a chain
            warnings.warn(f"FixedPointClassifier.refine failed in volume {self.vol}: {exc}",
                          stacklevel=2)  # fmt: skip
            return None
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

        Args:
            s: Radial coordinate of the refined fixed point.
            theta: Poloidal angle of the refined fixed point [rad].
            verbose: Print the trace/det/residue/discriminant line.

        Returns:
            dict | None: ``{trace, det, residue, disc, kind}``, or ``None`` if the variational
            integration did not complete (the point has likely left the volume) -- never raises.
        """
        try:
            _, J, success = self.tracer.trace_tangent_single(
                np.array([s, theta % (2 * np.pi)]),
                self.vol,
                num_transits=self.repeat_period,
                **self.tg,
            )
        except Exception as exc:  # noqa: BLE001 - one bad point must not abort a chain
            warnings.warn(f"tangent map failed in volume {self.vol}: {exc}", stacklevel=2)
            return None
        if not success:
            warnings.warn(
                f"tangent map: variational integration flagged failure in volume {self.vol}"
                " - the point may have left the volume",
                stacklevel=2,
            )
            return None
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
        dict | None: A flat dict with ``kind`` ('o'/'x'), ``residue``, ``disc``, ``trace``,
        ``det``, and the refined ``(R, Z, s, theta)``; ``None`` if the point could not be
        refined or its tangent map could not be integrated -- never raises.
    """
    if refine:
        fp = clf.refine(guess_sth)
        if fp is None:
            return None
    else:
        s, th = guess_sth
        R, Z = clf.specout.get_rz_coords(clf.vol, s, th, 0.0)
        fp = dict(R=float(np.ravel(R)[0]), Z=float(np.ravel(Z)[0]), s=s, theta=th, residual=np.nan)
    g = clf.greene(fp["s"], fp["theta"])
    if g is None:
        return None
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
