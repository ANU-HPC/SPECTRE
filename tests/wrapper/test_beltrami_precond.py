# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT

"""Cross-check the production preconditioner builder against the dense-extraction oracle.

``build_samemn_csr_diag`` assembles the exact same-(m,n) block of (dMA - mu*dMD) from the
diagonal metric-integral tensors, without ever forming the dense operator (the memory win
for the large volumes). ``build_samemn_csr`` extracts the same block from the dense
dMA/dMD and serves as the independent oracle. ``samemn_precond_compare`` densifies both
and returns the max entrywise difference relative to the largest entry.
"""

import os.path

import pytest
from helpers import spectre_with_lconstraint

from spectre import force_real, get_xinit_specwrap, remove_file

curr_path = os.path.dirname(__file__) + "/"

# The builder evaluates the same expressions as the dense assembler on the same diagonal
# metric-integral tensors, so it must agree with the dense extraction to round-off for ANY
# geometry -- shaped stellarator included. A loose tolerance here would hide an
# approximation sneaking into the preconditioner rather than absorb noise.
CASES = [
    (f"{curr_path}cyl_manyvol_test.toml", 1e-12),
    (f"{curr_path}qa_7vol_test.toml", 1e-12),
]


@pytest.mark.parametrize("fname,tol", CASES)
def test_samemn_precond_vs_dense(fname, tol):
    test = spectre_with_lconstraint(fname, 1)
    test.input_list_mod.lfindzero = 0
    test.input_list_mod.lhmatrix = False
    test.input_list_mod.lhevalues = False

    if not test.input_list_mod.enforce_stell_sym:
        pytest.skip("sparse preconditioner builder is stellarator-symmetric only")

    xinit = get_xinit_specwrap(test)
    force_real(xinit, test)

    mvol = test.allglobal_mod.mvol
    for vol in range(1, mvol + 1):
        for testmu in (0.0, 0.37):
            reldiff = test.lib.matrix_free_test_support_mod.samemn_precond_compare(vol, testmu)
            assert reldiff < tol, (
                f"{os.path.basename(fname)} vol={vol} mu={testmu}: sparse preconditioner "
                f"differs from dense extraction by rel {reldiff:.3e} (tol {tol})"
            )

    remove_file(fname + ".end")
