# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT

"""Verify the matrix-free Beltrami operator apply against the dense operator.

``op_apply`` (fortran_src/beltrami_operator_mod.F90) computes ``(dMA - mu*dMD) . x``
without forming the matrix, via ``packab_matvec`` -> ``intghs`` -> ``mtrxhs_full``.
The Fortran helper ``beltrami_matvec_selftest`` builds the dense ``dMA``/``dMD`` for a
volume and returns the max relative difference between the matrix-free apply and
``matmul(dMA - testmu*dMD, x)`` over several input vectors. This must be at the level
of floating-point accumulation noise for every volume (coordinate-singularity, interior
plasma, and — via a free-boundary case — vacuum), stellarator-symmetric and not, and
for mu = 0 as well as mu != 0.
"""

import os.path

import numpy as np
import pytest

from spectre import force_real, get_xinit_specwrap, remove_file

from helpers import spectre_with_lconstraint

curr_path = os.path.dirname(__file__) + "/"

# mu = 0 exercises the vacuum-region operator (dMA only); mu != 0 exercises the full
# (dMA - mu*dMD). testmu is independent of the volume's physical mu — the identity
# op_apply(mu) == (dMA - mu*dMD).x must hold for any mu.
TEST_MUS = [0.0, 0.37, 1.5]


@pytest.mark.parametrize(
    "fname", [f"{curr_path}qa_7vol_test.toml", f"{curr_path}cyl_manyvol_test.toml"]
)
@pytest.mark.parametrize("lc", [1, 3])
def test_beltrami_matvec(fname, lc):
    test = spectre_with_lconstraint(fname, lc)

    test.input_list_mod.lfindzero = 0
    test.input_list_mod.lhmatrix = False
    test.input_list_mod.lhevalues = False

    xinit = get_xinit_specwrap(test)
    force_real(xinit, test)

    mvol = test.allglobal_mod.mvol

    worst = 0.0
    for vol in range(1, mvol + 1):
        for testmu in TEST_MUS:
            reldiff = test.lib.matrix_free_test_support_mod.beltrami_matvec_selftest(vol, 4, testmu)
            worst = max(worst, reldiff)
            assert reldiff < 1e-11, (
                f"{os.path.basename(fname)} lc={lc} vol={vol} mu={testmu}: "
                f"matrix-free apply differs from dense by rel {reldiff:.3e}"
            )

    assert np.isfinite(worst)
    remove_file(fname + ".end")
