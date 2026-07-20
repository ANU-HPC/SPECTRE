# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

"""Verify the diagonal-restricted metric-integral build against the full build.

``volume_integrate_chebyshev`` (fortran_src/chebyshev_mod.F90) integrates the metric
elements over every mode pair (ii, jj). With ``diag_only=.true.`` it visits only the
diagonal pairs (jj == ii), which form the exact same-(m,n) block used as the strong
preconditioner -- at O(mn*lquad*Lrad^2) instead of O(mn^2*lquad*Lrad^2).

The diagonal restriction changes *which* mode pairs are visited, not how any one of them
is integrated: the loop body, the kijs/kija harmonic lookup and divisors, and the Zernike
parity/ll<m skips are all shared with the full pass. The diagonal entries must therefore
agree *exactly* -- this is a structural identity, not a tolerance. A nonzero difference
means the index decode or the tensor slot convention (``tensor_jslot``) is wrong.
"""

import os.path

import pytest

from spectre import SPECTRE, force_real, get_xinit_specwrap, remove_file

curr_path = os.path.dirname(__file__) + "/"


@pytest.mark.parametrize(
    "fname", [f"{curr_path}qa_7vol_test.toml", f"{curr_path}cyl_manyvol_test.toml"]
)
def test_diag_integrals_match_full(fname):
    test = SPECTRE(fname, verbose=False)

    test.input_list_mod.lcons = 1
    test.input_list_mod.lfindzero = 0
    test.input_list_mod.lhmatrix = False
    test.input_list_mod.lhevalues = False

    xinit = get_xinit_specwrap(test)
    force_real(xinit, test)

    mvol = test.allglobal_mod.mvol

    for vol in range(1, mvol + 1):
        absdiff = test.wrapper_funcs_mod.diag_integrals_selftest(vol)
        assert absdiff == 0.0, (
            f"{os.path.basename(fname)} vol={vol}: diagonal-restricted metric integrals "
            f"differ from the full build by {absdiff:.3e} (must be exactly 0)"
        )

    remove_file(fname + ".end")
