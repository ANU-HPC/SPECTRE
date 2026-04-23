# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

import pytest
import numpy as np
import os.path

from spectre import (
    SPECTRE,
    get_xinit_specwrap,
    force_real,
    force_modes_jac_speclike,
    get_fmn_force_mask,
    remove_file,
)

curr_path = os.path.dirname(__file__) + "/"

@pytest.mark.parametrize("fname", [f"{curr_path}qa_7vol_test.toml", f"{curr_path}cyl_manyvol_test.toml"])
@pytest.mark.parametrize("lc", [1, 3])
def test_jac_analytic_spec(fname, lc):

    test = SPECTRE.from_input_file(fname, verbose=False)

    #test.read(lcons=lc)
    test.input_list_mod.lcons = lc
    test.input_list_mod.lfindzero = 0
    test.input_list_mod.lhmatrix = False
    test.input_list_mod.lhevalues = False

    xinit = get_xinit_specwrap(test)

    force_real(xinit, test)

    fmn_jac_custom = force_modes_jac_speclike(xinit, test)

    ngdof = test.allglobal_mod.ngdof
    ntz = test.allglobal_mod.ntz
    nt = test.allglobal_mod.nt
    nz = test.allglobal_mod.nz
    test.wrapper_funcs_mod.alloc_for_hessian(ntz)
    test.field_mod.alloc_ntz_arrays(nt, nz)

    f = np.empty([ngdof + 1], dtype=np.float64, order="F")
    test.forces_mod.dforce(ngdof, xinit, f, True, True)

    # get SPECTRE jacobian
    fmn_jac_spec = test.allglobal_mod.force_ft_trunc_grad[:, :]
    mask_force_bsqrd = get_fmn_force_mask(test)
    fmn_jac_spec_bsqrd = fmn_jac_spec[mask_force_bsqrd]

    test.wrapper_funcs_mod.dealloc_hessian()

    maxelem = max(np.max(np.abs(fmn_jac_custom)), np.max(np.abs(fmn_jac_spec_bsqrd)))

    assert np.allclose(fmn_jac_custom, fmn_jac_spec_bsqrd.T, atol=1e-11 * maxelem, rtol=1e-11)

    remove_file(fname + ".end")
