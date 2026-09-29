# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT

"""Tests for the ``delta_prev`` trust-region warm start and global iteration numbering.

Pure Python: exercises the vendored solvers on an analytic residual, no MPI/Fortran needed.
"""

import numpy as np
import pytest

from spectre.least_squares.common import blend_delta_init
from spectre.least_squares.least_squares import least_squares


def fun_rosenbrock(x, *args):
    return np.array([10 * (x[1] - x[0] ** 2), (1 - x[0])])


def jac_rosenbrock(x, *args):
    return np.array([[-20 * x[0], 10.0], [-1.0, 0.0]])


X0 = np.array([2.0, 2.0])

# Gauss-Newton step at X0 has norm sqrt(5) ~ 2.236, while the naive initial radius is
# norm(X0) ~ 2.828 -- so a cold start takes the full step and a warm start blended with a
# tiny previous radius (~1.414) is genuinely constrained by the trust region.
UNUSABLE = [None, 0.0, -1.0, np.inf, np.nan]

# Rosenbrock has an exact root that Gauss-Newton reaches in two iterations, which is too
# fast to observe iteration numbering across cycles. This decay fit is deliberately
# inconsistent (the cosine term is not in the model), so it converges slowly instead.
T_DATA = np.linspace(0.0, 3.0, 10)
Y_DATA = 2.0 * np.exp(-1.3 * T_DATA) + 0.05 * np.cos(5.0 * T_DATA)
X0_DECAY = np.array([1.0, -0.1])


def fun_decay(x, *args):
    return x[0] * np.exp(x[1] * T_DATA) - Y_DATA


def jac_decay(x, *args):
    e = np.exp(x[1] * T_DATA)
    return np.stack([e, x[0] * T_DATA * e], axis=1)


@pytest.mark.parametrize("delta_prev, expected", [(None, 2.0), (4.0, 3.0), (1.0, 1.5)])
def test_blend_delta_init_averages_usable_radius(delta_prev, expected):
    assert blend_delta_init(2.0, delta_prev) == pytest.approx(expected)


@pytest.mark.parametrize("bad", UNUSABLE)
def test_blend_delta_init_ignores_unusable_radius(bad):
    assert blend_delta_init(2.0, bad) == 2.0


@pytest.mark.parametrize("method", ["dogbox", "trf"])
def test_result_reports_final_tr_radius(method):
    res = least_squares(fun_rosenbrock, X0, jac=jac_rosenbrock, method=method, args=[None])
    assert res.tr_radius is not None
    assert np.isfinite(res.tr_radius)
    assert res.tr_radius > 0


@pytest.mark.parametrize("method", ["dogbox", "trf"])
def test_delta_prev_constrains_the_first_step(method):
    """A small carried radius must shorten the first step relative to a cold start."""
    cold = least_squares(
        fun_rosenbrock, X0, jac=jac_rosenbrock, method=method, max_niter=1, args=[None]
    )
    warm = least_squares(
        fun_rosenbrock,
        X0,
        jac=jac_rosenbrock,
        method=method,
        max_niter=1,
        delta_prev=1e-6,
        args=[None],
    )
    assert np.linalg.norm(warm.x - X0) < np.linalg.norm(cold.x - X0)


@pytest.mark.parametrize("method", ["dogbox", "trf"])
@pytest.mark.parametrize("bad", UNUSABLE)
def test_unusable_delta_prev_matches_a_cold_start(method, bad):
    cold = least_squares(
        fun_rosenbrock, X0, jac=jac_rosenbrock, method=method, max_niter=1, args=[None]
    )
    warm = least_squares(
        fun_rosenbrock,
        X0,
        jac=jac_rosenbrock,
        method=method,
        max_niter=1,
        delta_prev=bad,
        args=[None],
    )
    np.testing.assert_array_equal(warm.x, cold.x)


@pytest.mark.parametrize("method", ["dogbox", "trf"])
def test_max_niter_is_enforced(method):
    res = least_squares(
        fun_rosenbrock, X0, jac=jac_rosenbrock, method=method, max_niter=2, args=[None]
    )
    assert res.status == 6


@pytest.mark.parametrize("method", ["dogbox", "trf"])
def test_iteration_start_continues_the_global_numbering(method):
    """A continued solve resumes the iteration count without re-reporting its first iterate."""
    rows = []

    def cb(it, nfev, cost, cost_red, step_norm, opt, x, obj):
        rows.append(it)

    first = least_squares(
        fun_decay,
        X0_DECAY,
        jac=jac_decay,
        method=method,
        max_niter=2,
        verbose=2,
        cb=cb,
        args=[None],
    )
    cycle_one = list(rows)
    rows.clear()

    least_squares(
        fun_decay,
        first.x,
        jac=jac_decay,
        method=method,
        max_niter=4,
        iteration_start=2,
        delta_prev=first.tr_radius,
        verbose=2,
        cb=cb,
        args=[None],
    )
    cycle_two = list(rows)

    assert first.status == 6, "the first cycle must stop on max_niter, not converge"
    assert cycle_one == [0, 1, 2]
    # Iterate 2 was already reported by the first cycle, so the second must not repeat it.
    assert cycle_two == [3, 4]
