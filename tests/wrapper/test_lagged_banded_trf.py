# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

"""Verify the lagged-banded matrix-free trust-region iteration (tr_options 'jvp'/'lag').

Between dense Jacobian rebuilds the iteration steps from a fresh-JVP reduced model over
a lagged subspace (least_squares/lagged.py, wired in trf.trf_no_bounds). It must reach
the same minimizer as the exact SVD path with strictly fewer dense Jacobian builds, and
must never declare convergence from lagged quantities.
"""

import os.path

import numpy as np
import pytest
from numpy.linalg import norm

from spectre.least_squares.trf import trf_no_bounds

curr_path = os.path.dirname(__file__) + "/"

N_BLOCKS = 6
RPB = 40  # residual rows per interface block
CPB = 7  # geometry DOFs per interface block


def _block_tridiagonal_lsq_problem(seed):
    """Mildly nonlinear least squares with an exactly block-tridiagonal Jacobian."""
    rng = np.random.default_rng(seed)
    m, n = N_BLOCKS * RPB, N_BLOCKS * CPB
    A = np.zeros((m, n))
    for r in range(N_BLOCKS):
        c_lo = max(r - 1, 0) * CPB
        c_hi = min(r + 2, N_BLOCKS) * CPB
        A[r * RPB : (r + 1) * RPB, c_lo:c_hi] = rng.standard_normal((RPB, c_hi - c_lo))
    # Small residual at the minimizer, so the gradient can be driven to gtol cleanly.
    x_star = 0.3 * rng.standard_normal(n)
    b = A @ (x_star + 0.05 * x_star**3) + 1e-3 * rng.standard_normal(m)
    x0 = x_star + 0.5 * rng.standard_normal(n)

    def fun(x):
        return A @ (x + 0.05 * x**3) - b

    def jac(x, f=None):
        return A * (1.0 + 0.15 * x**2)  # column scaling keeps the block structure

    def jvp(x, dirs):
        return jac(x) @ dirs

    return fun, jac, jvp, x0


def _run_trf(fun, jac, x0, tr_solver, tr_options):
    f0 = fun(x0)
    J0 = jac(x0, f0)
    counter = {"njev": 1}

    def jac_counted(x, f):
        counter["njev"] += 1
        return jac(x, f)

    result = trf_no_bounds(
        fun,
        jac_counted,
        x0,
        f0,
        J0,
        ftol=1e-15,
        xtol=1e-15,
        gtol=1e-8,
        max_nfev=400,
        max_niter=100,
        x_scale=1.0,
        loss_function=None,
        tr_solver=tr_solver,
        tr_options=tr_options,
        verbose=0,
        cb=None,
    )
    assert result.njev == counter["njev"]
    return result


@pytest.mark.parametrize("seed", [3, 17])
def test_lagged_reaches_exact_minimizer_with_fewer_jacobians(seed):
    fun, jac, jvp, x0 = _block_tridiagonal_lsq_problem(seed)

    res_exact = _run_trf(fun, jac, x0, "exact", {})
    res_lag = _run_trf(fun, jac, x0, "banded", {"n_blocks": N_BLOCKS, "jvp": jvp, "lag": 3})

    assert res_exact.status == 1 and res_lag.status == 1  # both converged on the gradient
    rel = norm(res_lag.x - res_exact.x) / norm(res_exact.x)
    assert rel < 1e-6, f"lagged minimizer deviates from exact by rel {rel:.3e}"
    assert res_lag.njvp > 0
    assert res_lag.njev < res_exact.njev, (
        f"lagged mode did not save dense Jacobian builds "
        f"(njev {res_lag.njev} vs exact {res_exact.njev})"
    )


def test_lag_without_jvp_warns_and_falls_back(capsys):
    fun, jac, jvp, x0 = _block_tridiagonal_lsq_problem(3)

    res_exact = _run_trf(fun, jac, x0, "exact", {})
    res_lag = _run_trf(fun, jac, x0, "banded", {"n_blocks": N_BLOCKS, "lag": 3})

    assert "disabled" in capsys.readouterr().out
    assert res_lag.njvp == 0
    rel = norm(res_lag.x - res_exact.x) / norm(res_exact.x)
    assert rel < 1e-10  # plain banded: identical trajectory to the SVD path


def test_trf_lagged_on_qa_7vol():
    """Four trf iterations of the real force minimization, lagged-banded vs exact SVD."""
    from spectre import (
        SPECTRE,
        force_real,
        force_real_jac,
        force_real_jvp,
        get_xinit_specwrap,
        remove_file,
    )
    from spectre.least_squares.least_squares import least_squares as lstsq

    fname = f"{curr_path}qa_7vol_test.toml"

    def run(tr_solver, lag):
        test = SPECTRE(fname, verbose=False)
        test.input_list_mod.lcons = 3
        ngdof = int(test.allglobal_mod.ngdof)
        n_blocks = int(test.allglobal_mod.mvol) - 1

        def obj_fn(xdof):
            return force_real(np.concatenate(([0.0], xdof)), test).flatten()

        def jac_fn(xdof):
            fjac = force_real_jac(np.concatenate(([0.0], xdof)), test)
            return fjac.reshape(-1, ngdof + 1)[:, 1:]

        def jvp_fn(xdof, dirs):
            dirs_full = np.vstack([np.zeros((1, dirs.shape[1])), dirs])
            jvp = force_real_jvp(np.concatenate(([0.0], xdof)), dirs_full, test)
            return jvp.reshape(-1, dirs.shape[1])

        tr_options = {}
        if tr_solver == "banded":
            tr_options = {"n_blocks": n_blocks}
            if lag:
                tr_options.update({"jvp": jvp_fn, "lag": lag})

        x0 = np.asarray(get_xinit_specwrap(test))[1:]
        result = lstsq(
            obj_fn,
            x0,
            jac=jac_fn,
            method="trf",
            tr_solver=tr_solver,
            tr_options=tr_options,
            max_niter=4,
            gtol=1e-15,
            ftol=1e-15,
            xtol=1e-22,
            verbose=0,
        )
        return result

    res_exact = run("exact", 0)
    res_lag = run("banded", 2)

    # Different (fresh vs reduced-model) steps, so trajectories differ; the lagged run
    # must still make comparable progress with strictly fewer dense Jacobian builds.
    assert res_lag.njev < res_exact.njev
    assert res_lag.njvp > 0
    assert res_lag.cost < 10.0 * res_exact.cost, (
        f"lagged cost {res_lag.cost:.3e} fell far behind exact {res_exact.cost:.3e}"
    )

    remove_file(fname + ".end")
