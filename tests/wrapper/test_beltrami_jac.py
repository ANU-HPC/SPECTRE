# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT

"""End-to-end check that the matrix-free Jacobian matches the dense one.

With ``Lbeltramisolver`` enabled, the global-constraint Jacobian path
(``solve_field_deriv``) replaces the dense per-volume LU factor + batched back-substitution
with a matrix-free preconditioned GMRES sweep (``op_bind_iter_sparse`` +
``op_solve_multi_iter``, warm-starting each of the ~4*mn columns from the previous). A full
``force_real_jac`` must therefore return the same real-space Jacobian as the default dense
path, to the GMRES tolerance. Uses Lconstraint=3 (the global path that is wired) and
``Lbeltramisolver=2`` to force the iterative path on every eligible volume.
"""

import os.path

import numpy as np
import pytest

from spectre import force_real, force_real_jac, get_xinit_specwrap, remove_file

from helpers import spectre_with_lconstraint

curr_path = os.path.dirname(__file__) + "/"


def _jac(fname, lbeltramisolver):
    test = spectre_with_lconstraint(fname, 3)
    test.input_list_mod.lfindzero = 0
    test.input_list_mod.lhmatrix = False
    test.input_list_mod.lhevalues = False
    test.input_list_mod.lbeltramisolver = lbeltramisolver
    if lbeltramisolver:
        test.input_list_mod.nadofgmres = 1
        test.input_list_mod.nitergmres = 5000
        test.input_list_mod.ngmresrestart = 60
        test.input_list_mod.epsgmres_jac = 1e-12
    xin = get_xinit_specwrap(test)
    force_real(xin, test)
    jac = np.asarray(force_real_jac(xin, test))
    remove_file(fname + ".end")
    return jac


@pytest.mark.parametrize(
    "fname", [f"{curr_path}qa_7vol_test.toml", f"{curr_path}cyl_manyvol_test.toml"]
)
def test_iterative_jac_matches_dense(fname):
    j_dense = _jac(fname, 0)
    j_iter = _jac(fname, 2)

    rel = np.linalg.norm(j_iter - j_dense) / max(np.linalg.norm(j_dense), 1e-300)
    assert rel < 1e-7, (
        f"{os.path.basename(fname)}: matrix-free Jacobian differs from the dense "
        f"Jacobian by rel {rel:.3e}"
    )
