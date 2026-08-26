# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT

"""End-to-end check that the matrix-free forward field solve matches the dense one.

With ``Lbeltramisolver`` enabled, ``solve_beltrami_system`` replaces the dense LU factor and
solve with a matrix-free preconditioned GMRES (op_bind_iter_sparse + op_solve_iter), while
the (mu, dpflux) constraint root-find and everything downstream is unchanged. A full
``force_real`` evaluation must therefore return the same real-space force as the default
dense solver, to the GMRES tolerance. ``Lbeltramisolver=2`` forces the iterative path on
every eligible volume so small test cases exercise it.
"""

import os.path

import numpy as np
import pytest
from helpers import spectre_with_lconstraint

from spectre import force_real, get_xinit_specwrap

curr_path = os.path.dirname(__file__) + "/"


def _force(fname, lbeltramisolver):
    test = spectre_with_lconstraint(fname, 1)
    test.input_list_mod.lfindzero = 0
    test.input_list_mod.lhmatrix = False
    test.input_list_mod.lhevalues = False
    test.input_list_mod.lbeltramisolver = lbeltramisolver
    if lbeltramisolver:
        test.input_list_mod.nitergmres = 3000
        test.input_list_mod.ngmresrestart = 60
        test.input_list_mod.epsgmres = 1e-13
    xin = get_xinit_specwrap(test)
    f = np.asarray(force_real(xin, test))
    return f


@pytest.mark.parametrize(
    "fname", [f"{curr_path}qa_7vol_test.toml", f"{curr_path}cyl_manyvol_test.toml"]
)
def test_iterative_forward_matches_dense(fname):
    f_dense = _force(fname, 0)
    f_iter = _force(fname, 2)

    rel = np.linalg.norm(f_iter - f_dense) / max(np.linalg.norm(f_dense), 1e-300)
    assert rel < 1e-8, (
        f"{os.path.basename(fname)}: matrix-free forward solve force differs from the "
        f"dense solve by rel {rel:.3e}"
    )
