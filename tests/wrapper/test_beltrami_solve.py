# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT

"""End-to-end validation of the matrix-free Beltrami solve against a direct solve.

``beltrami_solve_selftest`` builds the dense operator (dMA - mu*dMD) for a volume, solves
A x = b directly with LAPACK, then solves the same system with matrix-free GMRES
(``op_apply``) preconditioned by the exact block LU of the same-(m,n) block, and returns
the relative error between the two solutions and the GMRES iteration count. This exercises
the full pipeline — the matrix-free matvec, the block-diagonal preconditioner, and the
Krylov kernel — and asserts it reproduces the direct solve.

``beltrami_row_correction_solve_selftest`` repeats the exercise on a genuinely row-corrected
operator. The corrections are fractions of real rows of the dense operator (5% of row 1,
-3% of row NN), so subtracting them yields a system with a different solution; that corrected
matrix is direct-solved as the oracle. The matrix-free side attaches the correction to the
operator while the block-LU preconditioner is still built from the uncorrected same-(m,n)
blocks — converging to the direct solution proves GMRES iterates on the corrected residual
rather than on the operator the preconditioner was built from.
"""

import os.path

import numpy as np
import pytest
from helpers import spectre_with_lconstraint

from spectre import force_real, get_xinit_specwrap

curr_path = os.path.dirname(__file__) + "/"

TEST_MUS = [0.0, 0.37]


@pytest.mark.parametrize(
    "fname", [f"{curr_path}qa_7vol_test.toml", f"{curr_path}cyl_manyvol_test.toml"]
)
def test_beltrami_solve(fname):
    test = spectre_with_lconstraint(fname, 1)
    test.input_list_mod.lfindzero = 0
    test.input_list_mod.lhmatrix = False
    test.input_list_mod.lhevalues = False

    xinit = get_xinit_specwrap(test)
    force_real(xinit, test)

    mvol = test.allglobal_mod.mvol

    worst = 0.0
    for vol in range(1, mvol + 1):
        for testmu in TEST_MUS:
            # use_sparse=0: preconditioner extracted from the dense dMA/dMD (oracle path)
            # use_sparse=1: production build from the diagonal metric integrals
            for use_sparse in (0, 1):
                its, relerr = test.lib.matrix_free_test_support_mod.beltrami_solve_selftest(
                    vol, testmu, 50, 4000, 1e-10, use_sparse
                )
                worst = max(worst, relerr)
                assert 0 < its < 4000, (
                    f"vol={vol} mu={testmu} sparse={use_sparse}: GMRES used {its} iters"
                )
                assert relerr < 1e-6, (
                    f"{os.path.basename(fname)} vol={vol} mu={testmu} sparse={use_sparse}: "
                    f"matrix-free GMRES differs from direct solve by rel {relerr:.3e} ({its} iters)"
                )

    assert np.isfinite(worst)


@pytest.mark.parametrize("testmu", TEST_MUS)
def test_row_corrected_operator_solve_uses_corrected_residual(testmu):
    """GMRES converges against the corrected operator, not its unmodified preconditioner."""
    fname = f"{curr_path}qa_7vol_test.toml"
    test = spectre_with_lconstraint(fname, 1)
    test.input_list_mod.lfindzero = 0
    test.input_list_mod.lhmatrix = False
    test.input_list_mod.lhevalues = False

    xinit = get_xinit_specwrap(test)
    force_real(xinit, test)

    its, relerr, ierr = (
        test.lib.matrix_free_test_support_mod.beltrami_row_correction_solve_selftest(
            2, testmu, 50, 4000, 1e-10
        )
    )

    assert ierr == 0
    assert 0 < its < 4000
    assert relerr < 1e-6
