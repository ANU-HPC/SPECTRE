# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT

"""Tests for the ``delta_prev`` trust-region warm start in the vendored least_squares.

Pure Python: exercises the solver on an analytic residual, no MPI/Fortran needed.
"""

import numpy as np
import pytest

from spectre.least_squares.least_squares import least_squares


def fun_rosenbrock(x):
    return np.array([10 * (x[1] - x[0] ** 2), (1 - x[0])])


def jac_rosenbrock(x):
    return np.array([[-20 * x[0], 10], [-1, 0]])


X0 = np.array([2.0, 2.0])


@pytest.mark.parametrize("method", ["dogbox", "trf"])
def test_result_reports_final_tr_radius(method):
    res = least_squares(fun_rosenbrock, X0, jac=jac_rosenbrock, method=method)
    assert res.tr_radius is not None
    assert np.isfinite(res.tr_radius)
    assert res.tr_radius > 0


@pytest.mark.parametrize("method", ["dogbox", "trf"])
def test_delta_prev_blends_initial_radius(method, capsys):
    delta_prev = 0.05
    least_squares(fun_rosenbrock, X0, jac=jac_rosenbrock, method=method, delta_prev=delta_prev)

    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if ln.startswith("initial Delta blended")]
    assert len(lines) == 1
    naive = float(lines[0].split("naive")[1].split()[0])
    prev = float(lines[0].split("delta_prev")[1].split()[0])

    # The log prints %.4e, so compare at that precision.
    assert prev == pytest.approx(delta_prev, rel=1e-4)
    # naive radius for x_scale=1: norm(x0) for trf (no bounds), max|x0| for dogbox
    expected_naive = np.linalg.norm(X0) if method == "trf" else np.max(np.abs(X0))
    assert naive == pytest.approx(expected_naive, rel=1e-4)


@pytest.mark.parametrize("method", ["dogbox", "trf"])
@pytest.mark.parametrize("bad", [None, 0.0, -1.0, np.inf, np.nan])
def test_delta_prev_unusable_is_ignored(method, bad, capsys):
    res = least_squares(fun_rosenbrock, X0, jac=jac_rosenbrock, method=method, delta_prev=bad)

    out = capsys.readouterr().out
    assert "initial Delta blended" not in out
    assert res.success
