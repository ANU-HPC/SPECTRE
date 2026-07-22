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
from numpy.linalg import norm
from scipy.linalg import svd

from spectre.least_squares.banded import BandedNormalEquations
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

    bn = BandedNormalEquations(J, N_BLOCKS)
    assert bn.oob_ratio == 0.0

    p_gn = _svd_step(J, f, Delta=1e9)[0]  # interior Gauss-Newton regime
    deltas = [1e9, norm(p_gn) / 3.0, norm(p_gn) / 100.0]  # + boundary-constrained regimes

    for Delta in deltas:
        p_svd, alpha_svd, _ = _svd_step(J, f, Delta)
        res = bn.solve_trust_region(g, Delta)
        assert res is not None, f"banded solver declined a well-conditioned problem (Delta={Delta})"
        p_band, alpha_band, _ = res

        assert norm(p_band) <= Delta * (1 + 1e-12)
        rel = norm(p_band - p_svd) / norm(p_svd)
        assert rel < 1e-6, f"banded step deviates from SVD step by rel {rel:.3e} at Delta={Delta}"
        assert abs(alpha_band - alpha_svd) <= 1e-6 * max(alpha_svd, 1.0)


def test_out_of_band_mass_is_measured():
    J, _ = _block_tridiagonal_problem(11)
    rng = np.random.default_rng(12)
    scale = np.sqrt(np.mean(J**2))
    J_noisy = J + 1e-3 * scale * rng.standard_normal(J.shape)

    bn = BandedNormalEquations(J_noisy, N_BLOCKS)
    assert bn.oob_ratio > 1e-4  # the guard in trf trips at 1e-6

    with pytest.raises(ValueError):
        BandedNormalEquations(J[:, 1:], N_BLOCKS)  # columns not divisible into blocks


def test_trf_banded_matches_exact_on_qa_7vol():
    """Three trf iterations of the real force minimization, banded vs exact SVD."""
    from spectre import SPECTRE, force_real, force_real_jac, get_xinit_specwrap, remove_file
    from spectre.least_squares.least_squares import least_squares as lstsq

    fname = f"{curr_path}qa_7vol_test.toml"

    def run(tr_solver):
        test = SPECTRE(fname, verbose=False)
        test.input_list_mod.lcons = 3
        ngdof = int(test.allglobal_mod.ngdof)
        n_blocks = int(test.allglobal_mod.mvol) - 1

        def obj_fn(xdof):
            return force_real(np.concatenate(([0.0], xdof)), test).flatten()

        def jac_fn(xdof):
            fjac = force_real_jac(np.concatenate(([0.0], xdof)), test)
            return fjac.reshape(-1, ngdof + 1)[:, 1:]

        x0 = np.asarray(get_xinit_specwrap(test))[1:]
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

    remove_file(fname + ".end")
