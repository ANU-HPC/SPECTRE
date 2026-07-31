# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT

"""Standalone test of the independent GMRES(m) + block-LU pair (fortran_src/gmres_mod.F90).

``gmres_selftest`` solves a 2D 5-point Laplacian on a k x k grid preconditioned by the
exact block LU of the same-grid-row restriction (block Jacobi) -- the same kernel pair
production uses for the Beltrami solve, with the grid row standing in for the Fourier
mode. The restriction is approximate, so real GMRES iterations run and the kernel is
exercised end to end: Arnoldi with modified Gram-Schmidt, the Givens QR of the
Hessenberg, the restart, and the block gather/scatter triangular solves. The kernel
needs no equilibrium, so we call the wrapped module directly.
"""

import pytest
import spectre4py_wrap


@pytest.mark.parametrize("k", [10, 24, 40])
def test_gmres_laplacian(k):
    restart = 30
    maxiter = 2000
    its, relres, relres_reported, ierr = (
        spectre4py_wrap.matrix_free_test_support_mod.gmres_selftest(k, restart, maxiter)
    )

    # block-LU-preconditioned GMRES must drive the true residual to the requested tolerance
    assert relres < 1e-9, f"k={k}: GMRES did not converge, relres={relres:.3e}"
    assert 0 < its < maxiter, f"k={k}: unexpected iteration count {its}"

    # success is declared on the recomputed residual, not on the Givens estimate of it: what the
    # solver reports must be the residual an independent evaluation of b - A x finds
    assert ierr == 0, f"k={k}: GMRES reported status {ierr}"
    assert relres_reported == pytest.approx(relres, rel=1e-6, abs=1e-16), (
        f"k={k}: reported residual {relres_reported:.3e} is not the true residual {relres:.3e}"
    )


def test_shared_block_lu_is_thread_safe():
    """One factor can serve concurrent solves when each caller owns its scratch."""
    max_absdiff = spectre4py_wrap.matrix_free_test_support_mod.block_lu_concurrency_selftest(200)

    assert max_absdiff == 0.0
