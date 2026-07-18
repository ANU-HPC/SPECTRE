# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT
"""Primitives for locating magnetic-axis and island fixed points on the return map.

Every routine here is total: none raises when a trace or a root-find fails. Failure is reported
in-band instead, so callers need no exception handling -- ``_return_map`` returns ``LARGE``,
``root_fixed_point`` returns a :class:`FixedPointResult` with ``converged=False``,
``scan_fixed_points`` returns ``([], {})``, and ``find_axis`` warns and returns all-``None``.
"""

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.optimize import root

# Numerical constants shared across the fixed_points package. Both are device-independent:
# S_CLAMP is a coordinate bound and LARGE is a sentinel magnitude, not a physical length.
S_CLAMP = 0.999  # |s| bound for a seed or solution; s in (-1, 1) spans one annular volume
LARGE = 1.0e2  # return-map residual reported when a point leaves the volume or a trace fails


@dataclass
class FixedPointResult:
    """One fixed-point location.

    The field defaults describe a *failed* search, so a bare ``FixedPointResult()`` is the
    failure sentinel returned by :func:`root_fixed_point`.

    Attributes:
        R: Cylindrical :math:`R` of the point [m], NaN if not found.
        Z: Cylindrical :math:`Z` of the point [m], NaN if not found.
        s: Radial coordinate of the point in its volume, NaN if not found.
        theta: Poloidal angle of the point [rad], NaN if not found.
        residual: Return-map residual :math:`|(R,Z)_{end} - (R,Z)_{start}|` [m], NaN if not found.
        converged: True only if the solver succeeded, ``s`` is in volume, and ``residual < ftol``.
        nfev: Number of return-map evaluations the root-finder used.
    """

    R: float = np.nan
    Z: float = np.nan
    s: float = np.nan
    theta: float = np.nan
    residual: float = np.nan
    converged: bool = False
    nfev: int = 0


def _phi(obj, zetan):
    r"""Convert ``zetan`` (in units of :math:`2\pi/N_{fp}`) to a toroidal angle in radians."""
    return zetan * (2.0 * np.pi / obj.input.physics.Nfp)


def _return_map(
    sth, obj, vol=0, tracer_num_transits=2, zetan=0.0, method=None, rtol=None, atol=None
):
    r"""Return-map residual: ``(R,Z)`` after ``tracer_num_transits`` circuits, minus the start.

    Args:
        sth: ``(s, theta)`` start point in volume ``vol``.
        obj: A SPECTRE ``SPECTREout`` equilibrium.
        vol: Volume index hosting the point (0 = the singular axis volume).
        tracer_num_transits: Field-period circuits to trace before closing the map.
        zetan: Toroidal plane of the map, in units of :math:`2\pi/N_{fp}`.
        method: Integrator override; ``None`` keeps the tracer's default.
        rtol: Relative tolerance override; ``None`` keeps the tracer's default.
        atol: Absolute tolerance override; ``None`` keeps the tracer's default.

    Returns:
        numpy.ndarray: The 2-vector ``(dR, dZ)`` [m]. ``(LARGE, LARGE)`` if ``s`` is outside
        the volume or the trace failed to complete -- never raises.
    """
    s, th = sth

    if s > 1.0 or s < -1:
        return np.array([LARGE, LARGE])

    tracer_kw = {}
    if method is not None:
        tracer_kw["method"] = method
    if rtol is not None:
        tracer_kw["rtol"] = rtol
    if atol is not None:
        tracer_kw["atol"] = atol

    sth, RZ, iota, success = obj.trace_one_fieldline(
        np.array([s, th]),
        vol,
        num_phi_planes=1,
        num_transits=tracer_num_transits,
        zetan_start=zetan,
        **tracer_kw,
    )

    if not success:
        return np.array([LARGE, LARGE])

    r_start, z_start = obj.get_rz_coords(vol, s, th, _phi(obj, zetan))

    return np.array([RZ[0][-1] - r_start, RZ[1][-1] - z_start], dtype=float)


def find_axis(obj, max_repeats=0, guess=None, plot=False, verbose=False, name="", zetan=0.0):
    r"""Find the magnetic axis in volume 0 (a 2-transit return map).

    Root-finds from ``guess``, retrying from random ``(s, theta)`` seeds up to ``max_repeats``
    times.

    Args:
        obj: A SPECTRE ``SPECTREout`` equilibrium.
        max_repeats: Additional random-restart attempts if the first seed fails.
        guess: Initial ``(s, theta)`` seed. Defaults to ``[-0.9, 0.0]``.
        plot: Draw the located axis over the interfaces and a Poincare section.
        verbose: Print convergence information.
        name: Label used in the ``verbose`` output.
        zetan: Toroidal plane to find the axis on, in units of :math:`2\pi/N_{fp}`.

    Returns:
        tuple: ``(R, Z, s, theta)`` of the axis, or ``(None, None, None, None)`` if no attempt
        converged -- in which case a :class:`UserWarning` is issued rather than an exception.
    """
    user_guess = [-0.9, 0.0] if guess is None else list(guess)

    res = None
    for it in range(max_repeats + 1):
        small = 1e-7
        r = np.random.rand(2)
        guess_try = (
            user_guess
            if it == 0
            else [
                (r[0] * (1 - 2 * small) + small) * 2.0 - 1.0,
                r[1] * 2.0 * np.pi,
            ]
        )
        res = root_fixed_point(obj, 0, 2, guess_try, zetan=zetan, maxfev=100)
        if res.converged:
            break

    label = f" for {name}" if name else ""
    if not res.converged:
        warnings.warn(
            f"find_axis{label} did not converge after {max_repeats + 1} tries; "
            f"last residual {res.residual:.3e}",
            stacklevel=2,
        )
        return None, None, None, None

    if verbose:
        print(
            f"find_axis{label}: residual {res.residual:.3e} after {res.nfev} evals | "
            f"R {res.R:5.6f}  Z {res.Z:5.6f}  s {res.s:5.6f}  th {res.theta:5.6f}"
        )

    if plot:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()
        obj.plot_infaces(plot_axis=True, ax=ax, zetan=zetan)
        obj.plot_poincare(ax=ax, zetan=zetan)
        ax.plot(res.R, res.Z, "bX", ms=2)

    return res.R, res.Z, res.s, res.theta


def root_fixed_point(
    obj,
    vol,
    n_transits,
    guess_sth,
    *,
    zetan=0.0,
    method="RK45",
    rtol=1e-8,
    atol=1e-8,
    maxfev=30,
    root_tol=1e-9,
    ftol=1e-6,
):
    r"""Locate a fixed point by 2-D root-finding on the return map, warm-started from ``guess_sth``.

    Bounded by ``maxfev`` at a moderate integrator tolerance, so well-conditioned points converge
    cheaply while near-parabolic ones fail fast.

    Args:
        obj: A SPECTRE ``SPECTREout`` equilibrium.
        vol: Volume index hosting the point (0 = the singular axis volume).
        n_transits: Field-period circuits per return map (``repeat_period + 1`` for an island
            chain, 2 for the axis).
        guess_sth: ``(s, theta)`` warm start.
        zetan: Toroidal plane of the map, in units of :math:`2\pi/N_{fp}`.
        method: Integrator for the traces.
        rtol: Relative integrator tolerance.
        atol: Absolute integrator tolerance.
        maxfev: Return-map evaluation budget for the root-finder.
        root_tol: Tolerance handed to :func:`scipy.optimize.root`.
        ftol: Residual [m] below which the point is accepted as converged.

    Returns:
        FixedPointResult: The located point. ``converged`` is True only if the solver succeeded,
        ``|s| < S_CLAMP``, and ``residual < ftol``. On an unexpected solver/geometry error a bare
        (all-NaN, ``converged=False``) result is returned and a :class:`UserWarning` is issued --
        never raises.
    """
    try:
        sol = root(
            lambda x: _return_map(x, obj, vol, n_transits, zetan, method, rtol, atol),
            x0=[float(guess_sth[0]), float(guess_sth[1])],
            tol=root_tol,
            options=dict(maxfev=int(maxfev)),
        )
        s_f = float(sol.x[0])
        th_f = float(sol.x[1]) % (2.0 * np.pi)
        in_range = -S_CLAMP < s_f < S_CLAMP
        residual = float(np.linalg.norm(sol.fun))
        r_f, z_f = (
            obj.get_rz_coords(vol, s_f, th_f, _phi(obj, zetan)) if in_range else (np.nan, np.nan)
        )
    except Exception as exc:  # noqa: BLE001 - one failed point must not abort a whole scan
        warnings.warn(f"root_fixed_point failed in volume {vol}: {exc}", stacklevel=2)
        return FixedPointResult()

    return FixedPointResult(
        R=float(r_f),
        Z=float(z_f),
        s=s_f,
        theta=th_f,
        residual=residual,
        converged=bool(sol.success and in_range and residual < ftol),
        nfev=int(sol.nfev),
    )


def scan_fixed_points(
    obj,
    vol,
    n_transits,
    *,
    zetan=0.0,
    theta_grid=(0.0, np.pi),
    s_range=(-S_CLAMP, S_CLAMP),
    n_s=200,
    method="RK45",
    rtol=1e-8,
    atol=1e-8,
):
    r"""Grid-scan ``(s, theta)`` and return the interior local minima of the return-map residual.

    For each ``theta`` the residual is a 1-D function of ``s``; its interior minima are candidate
    fixed points, to be polished by :func:`root_fixed_point`.

    Args:
        obj: A SPECTRE ``SPECTREout`` equilibrium.
        vol: Volume index to scan (0 = the singular axis volume).
        n_transits: Field-period circuits per return map.
        zetan: Toroidal plane of the map, in units of :math:`2\pi/N_{fp}`.
        theta_grid: Poloidal angles to scan. ``(0, pi)`` scans the stellarator-symmetry midplane;
            a finer grid reaches off-symmetry-line points.
        s_range: Inclusive ``(s_lo, s_hi)`` scan bounds.
        n_s: Number of ``s`` samples per ``theta``.
        method: Integrator for the traces.
        rtol: Relative integrator tolerance.
        atol: Absolute integrator tolerance.

    Returns:
        tuple: ``(candidates, end_pinned)``. ``candidates`` is a list of
        ``{s, theta, R, Z, residual}`` dicts. ``end_pinned[theta]`` is
        ``'inner'``/``'outer'``/``''`` when that theta's minimum sits at an ``s`` endpoint (the
        point has left the volume). Returns ``([], {})`` on an unexpected geometry/tracing
        error, with a :class:`UserWarning` -- never raises.
    """
    s_grid = np.linspace(s_range[0], s_range[1], int(n_s))
    candidates = []
    end_pinned = {}
    try:
        for theta in np.atleast_1d(np.asarray(theta_grid, dtype=float)):
            theta = float(theta)
            res = np.array(
                [
                    np.hypot(
                        *_return_map([s, theta], obj, vol, n_transits, zetan, method, rtol, atol)
                    )
                    for s in s_grid
                ]
            )
            amin = int(np.argmin(res))
            end_pinned[theta] = "inner" if amin == 0 else ("outer" if amin == len(res) - 1 else "")
            for k in range(1, len(s_grid) - 1):
                if res[k] < res[k - 1] and res[k] < res[k + 1]:
                    r_c, z_c = obj.get_rz_coords(vol, s_grid[k], theta, _phi(obj, zetan))
                    candidates.append(
                        {
                            "s": float(s_grid[k]),
                            "theta": theta,
                            "R": float(r_c),
                            "Z": float(z_c),
                            "residual": float(res[k]),
                        }
                    )
    except Exception as exc:  # noqa: BLE001 - a failed scan must not abort the caller
        warnings.warn(f"scan_fixed_points failed in volume {vol}: {exc}", stacklevel=2)
        return [], {}
    return candidates, end_pinned
