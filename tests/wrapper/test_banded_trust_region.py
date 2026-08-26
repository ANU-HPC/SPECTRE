# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT

"""Verify the banded normal-equation trust-region solver against the exact SVD path.

The banded solver (spectre/least_squares/banded.py) exploits the block-tridiagonal
interface structure of the force Jacobian. Its step must reproduce the exact SVD
trust-region step: both run the same More alpha-iteration, so on a well-conditioned
problem the steps agree to numerical precision in every regime (interior Gauss-Newton
and boundary-constrained).
"""

import os.path

import numpy as np
import pytest
from helpers import spectre_with_lconstraint
from numpy.linalg import norm
from scipy.linalg import LinAlgError, svd

from spectre.least_squares.banded import (
    BandedSolveStatus,
    BandedTrustRegionSolver,
    assemble_banded_normal_equations,
    validate_block_layout,
)
from spectre.least_squares.common import solve_lsq_trust_region

curr_path = os.path.dirname(__file__) + "/"

N_BLOCKS = 6
RPB = 40  # residual rows per interface block
CPB = 7  # geometry DOFs per interface block


def _block_tridiagonal_problem(seed):
    rng = np.random.default_rng(seed)
    m, n = N_BLOCKS * RPB, N_BLOCKS * CPB
    J = np.zeros((m, n))
    for r in range(N_BLOCKS):
        c_lo = max(r - 1, 0) * CPB
        c_hi = min(r + 2, N_BLOCKS) * CPB
        J[r * RPB : (r + 1) * RPB, c_lo:c_hi] = rng.standard_normal((RPB, c_hi - c_lo))
    f = rng.standard_normal(m)
    return J, f


def _svd_step(J, f, Delta):
    m, n = J.shape
    U, s, V = svd(J, full_matrices=False)
    return solve_lsq_trust_region(n, m, U.T @ f, s, V.T, Delta)


@pytest.mark.parametrize("seed", [3, 17])
def test_banded_step_matches_svd_step(seed):
    J, f = _block_tridiagonal_problem(seed)
    g = J.T @ f

    assembly = assemble_banded_normal_equations(J, N_BLOCKS)
    assert assembly.out_of_band_ratio == 0.0
    bn = BandedTrustRegionSolver(assembly)

    p_gn = _svd_step(J, f, Delta=1e9)[0]  # interior Gauss-Newton regime
    deltas = [1e9, norm(p_gn) / 3.0, norm(p_gn) / 100.0]  # + boundary-constrained regimes

    for Delta in deltas:
        p_svd, alpha_svd, _ = _svd_step(J, f, Delta)
        res = bn.solve(g, Delta)
        assert res.available, f"banded solver declined a well-conditioned problem (Delta={Delta})"
        p_band, alpha_band, _ = res.step

        assert norm(p_band) <= Delta * (1 + 1e-12)
        rel = norm(p_band - p_svd) / norm(p_svd)
        assert rel < 1e-6, f"banded step deviates from SVD step by rel {rel:.3e} at Delta={Delta}"
        assert abs(alpha_band - alpha_svd) <= 1e-6 * max(alpha_svd, 1.0)


def test_out_of_band_mass_is_measured():
    J, _ = _block_tridiagonal_problem(11)
    _, layout = validate_block_layout(J, N_BLOCKS)
    assert (layout.rows_per_block, layout.columns_per_block) == (RPB, CPB)
    rng = np.random.default_rng(12)
    scale = np.sqrt(np.mean(J**2))
    J_noisy = J + 1e-3 * scale * rng.standard_normal(J.shape)

    assembly = assemble_banded_normal_equations(J_noisy, N_BLOCKS)
    assert assembly.out_of_band_ratio > 1e-4  # the guard in trf trips at 1e-6

    with pytest.raises(ValueError):
        assemble_banded_normal_equations(J[:, 1:], N_BLOCKS)  # columns not divisible into blocks


def test_banded_solver_validates_inputs_and_handles_zero_gradient():
    J, _ = _block_tridiagonal_problem(5)
    bn = BandedTrustRegionSolver(assemble_banded_normal_equations(J, N_BLOCKS))

    result = bn.solve(np.zeros(J.shape[1]), Delta=1.0)
    assert result.available
    assert result.diagnostics.status is BandedSolveStatus.SUCCESS
    assert result.diagnostics.relative_residual == 0.0
    step, alpha, iterations = result.step
    assert np.array_equal(step, np.zeros(J.shape[1]))
    assert alpha == 0.0
    assert iterations == 0

    with pytest.raises(ValueError, match="positive"):
        bn.solve(np.ones(J.shape[1]), Delta=0.0)
    with pytest.raises(ValueError, match="gradient"):
        bn.solve(np.ones(J.shape[1] - 1), Delta=1.0)
    with pytest.raises(ValueError, match="finite"):
        assemble_banded_normal_equations(np.full_like(J, np.nan), N_BLOCKS)


def test_numerical_failure_is_reported_without_choosing_fallback(monkeypatch):
    J, _ = _block_tridiagonal_problem(5)
    bn = BandedTrustRegionSolver(assemble_banded_normal_equations(J, N_BLOCKS))

    def fail_factorization(alpha):
        raise LinAlgError("synthetic failure")

    monkeypatch.setattr(bn, "_factor", fail_factorization)
    result = bn.solve(np.ones(J.shape[1]), Delta=1.0)
    assert not result.available
    assert result.diagnostics.status is BandedSolveStatus.FACTORIZATION_FAILED


def test_solve_newton_matches_lstsq_gauss_newton():
    J, f = _block_tridiagonal_problem(7)
    g = J.T @ f
    bn = BandedTrustRegionSolver(assemble_banded_normal_equations(J, N_BLOCKS))

    result = bn.solve_newton(g)
    assert result.available
    assert result.diagnostics.status is BandedSolveStatus.SUCCESS
    step, alpha, iterations = result.step
    assert alpha == 0.0
    assert iterations == 0

    p_ref = np.linalg.lstsq(J, -f, rcond=None)[0]
    assert norm(step - p_ref) < 1e-8 * norm(p_ref)

    zero = bn.solve_newton(np.zeros(J.shape[1]))
    assert zero.available
    assert np.array_equal(zero.step.step, np.zeros(J.shape[1]))

    with pytest.raises(ValueError, match="gradient"):
        bn.solve_newton(np.ones(J.shape[1] - 1))


def _nonlinear_block_problem(seed):
    """Mildly nonlinear residual whose Jacobian keeps the block-tridiagonal layout."""
    J0, _ = _block_tridiagonal_problem(seed)
    rng = np.random.default_rng(seed + 1)
    x_target = rng.standard_normal(J0.shape[1])
    y = J0 @ (x_target + 0.05 * x_target**2)

    def fun(x):
        return J0 @ (x + 0.05 * x**2) - y

    def jac(x):
        return J0 * (1.0 + 0.1 * x)[None, :]

    return fun, jac, np.zeros(J0.shape[1])


def test_dogbox_banded_matches_exact():
    from spectre.least_squares.least_squares import least_squares as lstsq

    fun, jac, x0 = _nonlinear_block_problem(23)

    def run(tr_solver):
        return lstsq(
            fun,
            x0,
            jac=jac,
            method="dogbox",
            tr_solver=tr_solver,
            tr_options={"n_blocks": N_BLOCKS} if tr_solver == "banded" else {},
            ftol=1e-14,
            xtol=1e-14,
            gtol=1e-14,
            verbose=0,
        )

    res_exact = run("exact")
    res_banded = run("banded")

    assert res_exact.status > 0 and res_banded.status > 0
    x_scale = max(norm(res_exact.x), 1.0)
    assert norm(res_banded.x - res_exact.x) < 1e-8 * x_scale, (
        "dogbox banded and lstsq Gauss-Newton steps diverged"
    )
    assert res_banded.cost < 1e-12


def test_dogbox_banded_falls_back_when_structure_is_violated(capsys):
    from spectre.least_squares.least_squares import least_squares as lstsq

    J, _ = _block_tridiagonal_problem(11)
    rng = np.random.default_rng(12)
    J_noisy = J + 1e-3 * np.sqrt(np.mean(J**2)) * rng.standard_normal(J.shape)
    y = J_noisy @ rng.standard_normal(J.shape[1])

    res = lstsq(
        lambda x: J_noisy @ x - y,
        np.zeros(J.shape[1]),
        jac=lambda x: J_noisy,
        method="dogbox",
        tr_solver="banded",
        tr_options={"n_blocks": N_BLOCKS},
        verbose=0,
    )
    assert res.status > 0
    out = capsys.readouterr().out
    assert "out-of-band" in out


def test_dogbox_banded_requires_n_blocks(capsys):
    from spectre.least_squares.least_squares import least_squares as lstsq

    fun, jac, x0 = _nonlinear_block_problem(29)
    res = lstsq(fun, x0, jac=jac, method="dogbox", tr_solver="banded", tr_options={}, verbose=0)
    assert res.status > 0
    out = capsys.readouterr().out
    assert "banded trust-region solver disabled" in out


def test_trf_banded_matches_exact_on_qa_7vol():
    """Three trf iterations of the real force minimization, banded vs exact SVD."""
    from spectre.least_squares.least_squares import least_squares as lstsq

    fname = f"{curr_path}qa_7vol_test.toml"

    def run(tr_solver):
        test = spectre_with_lconstraint(fname, 3)
        ngdof = int(test.allglobal_mod.ngdof)
        n_blocks = int(test.allglobal_mod.mvol) - 1

        def obj_fn(xdof):
            return test.force_real(xdof).ravel()

        def jac_fn(xdof):
            return test.force_real_jac(xdof).reshape(-1, ngdof)

        x0 = test.xin
        result = lstsq(
            obj_fn,
            x0,
            jac=jac_fn,
            method="trf",
            tr_solver=tr_solver,
            tr_options={"n_blocks": n_blocks} if tr_solver == "banded" else {},
            max_niter=3,
            gtol=1e-15,
            ftol=1e-15,
            xtol=1e-22,
            verbose=0,
        )
        return result

    res_exact = run("exact")
    res_banded = run("banded")

    x_scale = norm(res_exact.x)
    assert np.max(np.abs(res_banded.x - res_exact.x)) < 1e-5 * x_scale, (
        "banded and exact trust-region solvers diverged within 3 iterations"
    )
    rel_cost = abs(res_banded.cost - res_exact.cost) / res_exact.cost
    assert rel_cost < 1e-5
