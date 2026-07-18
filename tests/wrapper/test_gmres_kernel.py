# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

"""Standalone test of the clean-room GMRES(m) + ILU(0) kernel (fortran_src/gmres_mod.F90).

``gmres_selftest`` solves a 2D 5-point Laplacian on a k x k grid with an ILU(0)
preconditioner. ILU(0) is an approximate factorization of this matrix (exact LU fills
in), so real GMRES iterations run and the kernel is exercised end to end: Arnoldi with
modified Gram-Schmidt, the Givens QR of the Hessenberg, the restart, and the CSR
triangular solves. The kernel needs no equilibrium, so we call the wrapped module directly.
"""

import pytest
import spectre4py_wrap


@pytest.mark.parametrize("k", [10, 24, 40])
def test_gmres_laplacian(k):
    restart = 30
    maxiter = 2000
    its, relres = spectre4py_wrap.wrapper_funcs_mod.gmres_selftest(k, restart, maxiter)

    # ILU(0)-preconditioned GMRES must drive the true residual to the requested tolerance
    assert relres < 1e-9, f"k={k}: GMRES did not converge, relres={relres:.3e}"
    assert 0 < its < maxiter, f"k={k}: unexpected iteration count {its}"
