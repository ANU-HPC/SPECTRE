# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

import os.path

import numpy as np
import pytest
from helpers import spectre_with_lconstraint

from spectre import (
    force_modes_jac_speclike,
    force_real,
    get_fmn_force_mask,
    get_xinit_specwrap,
)

curr_path = os.path.dirname(__file__) + "/"


@pytest.mark.parametrize(
    "fname", [f"{curr_path}qa_7vol_test.toml", f"{curr_path}cyl_manyvol_test.toml"]
)
@pytest.mark.parametrize("lc", [1, 3])
def test_jac_analytic_spec(fname, lc):

    test = spectre_with_lconstraint(fname, lc)

    # test.read(lcons=lc)
    test.input_list_mod.lfindzero = 0
    test.input_list_mod.lhmatrix = False
    test.input_list_mod.lhevalues = False

    xinit = get_xinit_specwrap(test)

    force_real(xinit, test)

    fmn_jac_custom = force_modes_jac_speclike(xinit, test)

    ntz = test.allglobal_mod.ntz
    nt = test.allglobal_mod.nt
    nz = test.allglobal_mod.nz
    test.wrapper_funcs_mod.alloc_for_hessian(ntz)
    test.field_mod.alloc_ntz_arrays(nt, nz)

    test.dforce(xinit, compute_derivatives=True)

    # get SPECTRE jacobian
    fmn_jac_spec = test.allglobal_mod.force_ft_trunc_grad[:, :]
    mask_force_bsqrd = get_fmn_force_mask(test)
    fmn_jac_spec_bsqrd = fmn_jac_spec[mask_force_bsqrd]

    test.wrapper_funcs_mod.dealloc_hessian()

    maxelem = max(np.max(np.abs(fmn_jac_custom)), np.max(np.abs(fmn_jac_spec_bsqrd)))

    assert np.allclose(fmn_jac_custom, fmn_jac_spec_bsqrd.T, atol=1e-11 * maxelem, rtol=1e-11)
