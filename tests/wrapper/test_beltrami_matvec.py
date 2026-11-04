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

``beltrami_row_correction_selftest`` covers the compact row correction on the same apply:
it attaches a synthetic two-row correction (rows 1 and NN), then clobbers the caller's
array with ``huge()`` to prove the operator snapshotted the values at attachment, and
compares ``op_apply`` against the dense matvec with the two row dot products subtracted
by hand. The correction rows are not scaled by mu — they modify only the A-action.
"""

import os.path

import numpy as np
import pytest
from helpers import spectre_freeb_w7x_reduced, spectre_with_lconstraint

from spectre import force_real, get_xinit_specwrap, remove_file

curr_path = os.path.dirname(__file__) + "/"

# mu = 0 exercises the vacuum-region operator (dMA only); mu != 0 exercises the full
# (dMA - mu*dMD). testmu is independent of the volume's physical mu — the identity
# op_apply(mu) == (dMA - mu*dMD).x must hold for any mu.
TEST_MUS = [0.0, 0.37, 1.5]


def _assert_matvec_matches_dense(test, label, mus=TEST_MUS):
    worst = 0.0
    for vol in range(1, test.allglobal_mod.mvol + 1):
        for testmu in mus:
            reldiff = test.lib.matrix_free_test_support_mod.beltrami_matvec_selftest(vol, 4, testmu)
            worst = max(worst, reldiff)
            assert reldiff < 1e-11, (
                f"{label} vol={vol} mu={testmu}: "
                f"matrix-free apply differs from dense by rel {reldiff:.3e}"
            )
    assert np.isfinite(worst)


@pytest.mark.parametrize(
    "fname", [f"{curr_path}qa_3vol_light.toml", f"{curr_path}cyl_manyvol_test.toml"]
)
@pytest.mark.parametrize("lc", [1, 3])
def test_beltrami_matvec(fname, lc):
    test = spectre_with_lconstraint(fname, lc)

    test.input_list_mod.lfindzero = 0
    test.input_list_mod.lhmatrix = False
    test.input_list_mod.lhevalues = False

    xinit = get_xinit_specwrap(test)
    force_real(xinit, test)

    _assert_matvec_matches_dense(test, f"{os.path.basename(fname)} lc={lc}")
    remove_file(fname + ".end")


@pytest.mark.parametrize("lc", [1, 3])
def test_beltrami_matvec_vacuum_volume(lc):
    """Free boundary (reduced-resolution W7-X): the last volume is a real vacuum volume."""
    test = spectre_freeb_w7x_reduced(lc)

    test.input_list_mod.lfindzero = 0
    test.input_list_mod.lhmatrix = False
    test.input_list_mod.lhevalues = False

    xinit = get_xinit_specwrap(test)
    force_real(xinit, test)

    assert test.allglobal_mod.mvol == test.input_list_mod.nvol + 1, "no vacuum volume present"
    # a reduced sweep: mu = 0 is the actual vacuum operator, one mu != 0 covers the dMD term
    _assert_matvec_matches_dense(test, f"freeb_w7x lc={lc}", mus=[0.0, 0.37])


@pytest.mark.parametrize("testmu", TEST_MUS)
def test_beltrami_row_correction_matches_dense_subtraction(testmu):
    """The generic correction snapshots compact rows and modifies only the A-action."""
    fname = f"{curr_path}qa_3vol_light.toml"
    test = spectre_with_lconstraint(fname, 1)

    test.input_list_mod.lfindzero = 0
    test.input_list_mod.lhmatrix = False
    test.input_list_mod.lhevalues = False
    force_real(get_xinit_specwrap(test), test)

    reldiff = test.lib.matrix_free_test_support_mod.beltrami_row_correction_selftest(2, testmu)
    assert reldiff < 1e-11
    remove_file(fname + ".end")
