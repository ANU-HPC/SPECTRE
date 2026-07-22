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
"""

import os.path

import numpy as np
import pytest

from spectre import SPECTRE, force_real, get_xinit_specwrap, remove_file

curr_path = os.path.dirname(__file__) + "/"

TEST_MUS = [0.0, 0.37]


@pytest.mark.parametrize(
    "fname", [f"{curr_path}qa_7vol_test.toml", f"{curr_path}cyl_manyvol_test.toml"]
)
def test_beltrami_solve(fname):
    test = SPECTRE(fname, verbose=False)
    test.input_list_mod.lcons = 1
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
                its, relerr = test.wrapper_funcs_mod.beltrami_solve_selftest(
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
    remove_file(fname + ".end")
