# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

import pytest
import numpy as np
import os.path

from spectre import (
    SPECTRE,
    angular_force_likespec,
    get_xinit_specwrap,
    get_fmn_force_mask,
    force_modes_dforce,
    angular_force_jac_likespec,
    remove_file,
)

curr_path = os.path.dirname(__file__) + "/"


@pytest.mark.parametrize("fname", ["qa_3vol_light.toml"])
def test_ang_force_tok(fname):

    test = SPECTRE(curr_path + fname, verbose=False)
    test.input_list_mod.lcons = 0

    xin = get_xinit_specwrap(test)

    force_custom = angular_force_likespec(xin, test)

    force_spec_full = force_modes_dforce(xin, test)
    mask = ~get_fmn_force_mask(test)
    force_spec = force_spec_full[1:][mask]

    maxelem = max(np.max(force_custom), np.max(force_spec))
    assert np.allclose(force_custom, force_spec, atol=4e-5 * maxelem, rtol=4e-5)

    remove_file(curr_path + fname + ".end")


@pytest.mark.parametrize("fname", ["qa_3vol_light.toml"])
def test_ang_force_jac_tok(fname):

    test = SPECTRE(curr_path + fname, verbose=False)
    test.input_list_mod.lcons = 0

    test.input_list_mod.lfindzero = 0
    test.input_list_mod.lhmatrix = False
    test.input_list_mod.lhevalues = False

    xin = get_xinit_specwrap(test)

    ngdof = test.allglobal_mod.ngdof
    ntz = test.allglobal_mod.ntz

    test.wrapper_funcs_mod.alloc_for_hessian(ntz)

    f = np.empty([ngdof + 1], dtype=np.float64, order="F")
    test.forces_mod.dforce(ngdof, xin, f, True, False)

    # get SPECTRE jacobian
    fmn_jac_spec = test.allglobal_mod.force_ft_trunc_grad[:, :]
    mask_force_bsqrd = ~get_fmn_force_mask(test)
    angular_constr_jac_spec = fmn_jac_spec[mask_force_bsqrd]

    test.wrapper_funcs_mod.dealloc_hessian()

    # get custom jacobian
    angular_constr_jac_custom = angular_force_jac_likespec(xin, test, refind_fields=False)

    maxelem = max(np.max(angular_constr_jac_custom), np.max(angular_constr_jac_spec.T))
    assert np.allclose(
        angular_constr_jac_custom, angular_constr_jac_spec.T, atol=4e-5 * maxelem, rtol=4e-5
    )

    remove_file(curr_path + fname + ".end")
