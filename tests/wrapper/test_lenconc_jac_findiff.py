# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

import os.path

import numpy as np
import pytest
from helpers import spectre_with_lconstraint

from spectre import (
    approx_derivative,
    get_xinit_specwrap,
    input_dict,
    len_cons_real,
    len_cons_real_jac,
    remove_file,
)
from spectre.utils import pad_dead_slot

curr_path = os.path.dirname(__file__) + "/"


@pytest.fixture(scope="module")
def tor():

    return input_dict(
        dict(
            dofs=np.array([1, 5, 37]),
            method="3-point",
            abs_step=np.ones(3) * 1e-8,
            sp_fname=f"{curr_path}qa_7vol_test.toml",
        )
    )


@pytest.mark.parametrize("fixture", ["tor"])
def test_lenconc_jac(fixture, request):

    fixture = request.getfixturevalue(fixture)

    dofs = fixture.dofs
    method = fixture.method
    abs_step = fixture.abs_step
    sp_fname = fixture.sp_fname

    test = spectre_with_lconstraint(sp_fname, 0)
    # test.read(lcons=0)
    test.input_list_mod.lfindzero = 0
    test.input_list_mod.lhmatrix = False
    test.input_list_mod.lhevalues = False

    ngdof = test.allglobal_mod.ngdof
    ntz = test.allglobal_mod.ntz

    test.wrapper_funcs_mod.alloc_for_hessian(ntz)

    xinit = get_xinit_specwrap(test)

    f = np.empty([ngdof + 1], dtype=np.float64, order="F")
    test.forces_mod.dforce(ngdof, pad_dead_slot(xinit), f, True, True)

    jac_analytic_full = len_cons_real_jac(xinit, test, add_psifactor=True, add_sweightfactor=True)
    fjac_analytic = jac_analytic_full[:, dofs]

    test.wrapper_funcs_mod.dealloc_hessian()

    jac_numeric = approx_derivative(
        lambda x: len_cons_real(x, test, refind_fields=True),
        xinit,
        method=method,
        dofs=dofs + 0,
        abs_step=abs_step,
    )

    maxelem = max(np.max(np.abs(jac_numeric)), np.max(np.abs(fjac_analytic)))

    assert np.allclose(jac_numeric, fjac_analytic, atol=1e-7 * maxelem, rtol=1e-7)

    remove_file(sp_fname + ".end")
