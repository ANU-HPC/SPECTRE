# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

"""Lagged-Jacobian trf mode (jac_lag): same solution, fewer Jacobian builds."""

import numpy as np

from spectre import SPECTRE, force_real, force_real_jac, get_xinit_specwrap, least_squares


def _rosenbrock_like():
    # smooth nonlinear least-squares with a long, nearly-linear tail
    a = np.linspace(0.5, 2.0, 8)

    def f(x):
        return np.concatenate([(x[1:] - x[:-1] ** 2), a * (x - 1.0)])

    def jac(x):
        n = x.size
        J = np.zeros((2 * n - 1, n))
        for i in range(n - 1):
            J[i, i] = -2.0 * x[i]
            J[i, i + 1] = 1.0
        J[n - 1:, :] = np.diag(a)
        return J

    return f, jac, np.full(8, 3.0)


def test_lagged_matches_standard_numpy_problem():
    f, jac, x0 = _rosenbrock_like()
    res0 = least_squares(f, x0, jac=jac, method="trf", ftol=1e-12, xtol=1e-14, gtol=1e-12)
    res1 = least_squares(f, x0, jac=jac, method="trf", ftol=1e-12, xtol=1e-14, gtol=1e-12,
                         jac_lag=6, jac_lag_rho=0.9)
    assert res1.cost <= res0.cost * (1 + 1e-6) + 1e-14
    assert np.allclose(res1.x, res0.x, atol=1e-6)
    assert res1.njev < res0.njev


def test_lagged_convergence_not_declared_stale_numpy_problem():
    # gtol convergence must be confirmed with a fresh Jacobian
    f, jac, x0 = _rosenbrock_like()
    res = least_squares(f, x0, jac=jac, method="trf", ftol=1e-14, xtol=1e-15, gtol=1e-10,
                        jac_lag=50, jac_lag_rho=0.0)
    g = jac(res.x).T @ f(res.x)
    assert np.linalg.norm(g, ord=np.inf) < 1e-8


def test_lagged_matches_standard_spectre_qa():
    def run(jac_lag):
        test = SPECTRE("tests/wrapper/qa_7vol_test.toml", verbose=False)
        x0 = get_xinit_specwrap(test)
        force_real(x0, test)

        def f(x, t):
            return force_real(x, t).flatten(order="C")

        def jac(x, t, *a):
            return force_real_jac(x, t).reshape(-1, t.allglobal_mod.ngdof + 1)

        return least_squares(f, x0, jac=jac, method="trf", args=[test], max_niter=10,
                             ftol=1e-10, xtol=1e-22, gtol=1e-13, jac_lag=jac_lag)

    res0 = run(0)
    res1 = run(6)
    assert res1.cost <= res0.cost * 1.05
    assert res1.njev < res0.njev
