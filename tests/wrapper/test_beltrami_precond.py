# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

"""Cross-check the memory-optimal sparse preconditioner against the dense-extraction one.

``op_bind_iter`` extracts the same-(m,n) preconditioner block from the dense dMA/dMD.
``build_samemn_csr_sparse`` builds a block from the flux-surface-averaged (0,0) metric
harmonic, so the dense matrix need never be formed for the large vacuum volume (the memory
win). For an axisymmetric configuration the two are identical (there are no other
harmonics); for a shaped stellarator the sparse one is the "tokamak" approximation and
differs by the higher-harmonic self-coupling (a few percent) — that is the intended
preconditioner, and GMRES converges with it regardless (see test_beltrami_solve.py).
``samemn_precond_compare`` returns the max relative off-diagonal difference.
"""

import os.path

import pytest

from spectre import SPECTRE, force_real, get_xinit_specwrap, remove_file

curr_path = os.path.dirname(__file__) + "/"

# axisymmetric -> sparse == dense to round-off; shaped stellarator -> a few-percent
# tokamak-approximation gap is expected and bounded.
CASES = [
    (f"{curr_path}cyl_manyvol_test.toml", 1e-10),
    (f"{curr_path}qa_7vol_test.toml", 0.1),
]


@pytest.mark.parametrize("fname,tol", CASES)
def test_samemn_precond_vs_dense(fname, tol):
    test = SPECTRE(fname, verbose=False)
    test.input_list_mod.lcons = 1
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
            reldiff = test.wrapper_funcs_mod.samemn_precond_compare(vol, testmu)
            assert reldiff < tol, (
                f"{os.path.basename(fname)} vol={vol} mu={testmu}: sparse preconditioner "
                f"differs from dense extraction by rel {reldiff:.3e} (tol {tol})"
            )

    remove_file(fname + ".end")
