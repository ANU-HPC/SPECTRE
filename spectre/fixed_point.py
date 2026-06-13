# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

from dataclasses import dataclass

import numpy as np
from scipy.optimize import root


@dataclass
class FixedPointResult:
    """One fixed-point location."""

    R: float = np.nan
    Z: float = np.nan
    s: float = np.nan
    theta: float = np.nan
    residual: float = np.nan
    converged: bool = False
    nfev: int = 0


def _return_map(sth, obj, vol=0, tracer_num_transits=2, *, method=None, rtol=None, atol=None):
    """Return-map residual: ``(R,Z)`` at phi=0 after ``tracer_num_transits`` circuits, minus start.

    ``method``/``rtol``/``atol`` override the integrator tolerance (``None`` keeps the
    ``trace_one_fieldline`` defaults, so existing callers are unchanged).
    """
    s, th = sth

    large = 1.0e2

    if s > 1.0 or s < -1:
        return np.array([large, large])

    tracer_kw = {}
    if method is not None:
        tracer_kw["method"] = method
    if rtol is not None:
        tracer_kw["rtol"] = rtol
    if atol is not None:
        tracer_kw["atol"] = atol

    sth, RZ, iota, flag = obj.trace_one_fieldline(
        np.array([s, th]),
        vol,
        num_phi_planes=1,
        num_transits=tracer_num_transits,
        **tracer_kw,
    )

    if flag:
        return np.array([large, large])

    r_start, z_start = obj.get_rz_coords(vol, s, th, 0.0)

    return np.array([RZ[0][-1] - r_start, RZ[1][-1] - z_start], dtype=float)


def _random_guess(small=1e-7):
    r_rand = np.random.rand(2)
    return [
        (r_rand[0] * (1 - 2 * small) + small) * 2.0 - 1.0,
        r_rand[1] * 2.0 * np.pi,
    ]


def find_axis(obj, max_repeats=0, guess=None, verbose=False):
    """Find the magnetic axis in volume 0 (``tracer_num_transits=2`` return map).

    Root-finds from ``guess`` (default ``[-0.9, 0.0]``), retrying from random ``(s, theta)``
    seeds up to ``max_repeats`` times; returns ``(R, Z, s, theta)``.
    """
    user_guess = [-0.9, 0.0] if guess is None else list(guess)

    res = None
    for it in range(max_repeats + 1):
        guess_try = user_guess if it == 0 else _random_guess()
        res = root_fixed_point(obj, 0, 2, guess_try, maxfev=100)
        if res.converged:
            if verbose:
                print(
                    f"find_axis: residual {res.residual:.3e} after {res.nfev} evals | "
                    f"R {res.R:5.6f}  Z {res.Z:5.6f}  s {res.s:5.6f}  th {res.theta:5.6f}"
                )
            return res.R, res.Z, res.s, res.theta

    raise RuntimeError(
        f"find_axis failed after {max_repeats + 1} tries; last residual {res.residual:.3e}"
    )


def root_fixed_point(
    obj,
    vol,
    n_transits,
    guess_sth,
    *,
    method="RK45",
    rtol=1e-8,
    atol=1e-8,
    maxfev=30,
    root_tol=1e-9,
    ftol=1e-6,
):
    """Locate a fixed point by 2-D root-finding on the return map, warm-started from ``guess_sth``.

    Bounded by ``maxfev`` at a moderate integrator tolerance, so well-conditioned points converge
    cheaply while near-parabolic ones fail fast. ``converged`` requires solver success, ``s`` in
    volume, and residual ``< ftol``.
    """
    sol = root(
        lambda x: _return_map(x, obj, vol, n_transits, method=method, rtol=rtol, atol=atol),
        x0=[float(guess_sth[0]), float(guess_sth[1])],
        tol=root_tol,
        options=dict(maxfev=int(maxfev)),
    )
    s_f = float(sol.x[0])
    th_f = float(sol.x[1]) % (2.0 * np.pi)
    in_range = -0.999 < s_f < 0.999
    residual = float(np.linalg.norm(sol.fun))
    r_f, z_f = obj.get_rz_coords(vol, s_f, th_f, 0.0) if in_range else (np.nan, np.nan)

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
    theta_grid=(0.0, np.pi),
    s_range=(-0.999, 0.999),
    n_s=200,
    method="RK45",
    rtol=1e-8,
    atol=1e-8,
):
    """Seed a grid in ``(s, theta)`` and return return-map-residual local minima (no solver).

    For each ``theta`` the residual is a 1-D function of ``s`` whose interior local minima are
    candidate fixed points. ``theta_grid=(0, pi)`` is the stellarator-symmetry midplane scan; a
    finer grid makes it a 2-D scan reaching off-symmetry-line points. Returns
    ``(candidates, end_pinned)``: each candidate is a ``{s, theta, R, Z, residual}`` dict, and
    ``end_pinned[theta]`` is ``'inner'``/``'outer'``/``''`` flagging a minimum stuck at an ``s``
    endpoint (the point leaving the volume).
    """
    s_grid = np.linspace(s_range[0], s_range[1], int(n_s))
    candidates = []
    end_pinned = {}
    for theta in np.atleast_1d(np.asarray(theta_grid, dtype=float)):
        theta = float(theta)
        res = np.array(
            [
                np.hypot(
                    *_return_map(
                        [s, theta], obj, vol, n_transits, method=method, rtol=rtol, atol=atol
                    )
                )
                for s in s_grid
            ]
        )
        amin = int(np.argmin(res))
        end_pinned[theta] = "inner" if amin == 0 else ("outer" if amin == len(res) - 1 else "")
        for k in range(1, len(s_grid) - 1):
            if res[k] < res[k - 1] and res[k] < res[k + 1]:
                r_c, z_c = obj.get_rz_coords(vol, s_grid[k], theta, 0.0)
                candidates.append(
                    {
                        "s": float(s_grid[k]),
                        "theta": theta,
                        "R": float(r_c),
                        "Z": float(z_c),
                        "residual": float(res[k]),
                    }
                )
    return candidates, end_pinned
