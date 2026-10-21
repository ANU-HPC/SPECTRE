# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

import pytest
import numpy as np
import os.path

from spectre import (
    SPECTRE,
    get_xinit_specwrap,
    input_dict,
    force_real,
    force_real_jac,
    approx_derivative,
    remove_file,
)

curr_path = os.path.dirname(__file__) + "/"


@pytest.fixture(scope="module")
def cyl():

    return input_dict(
        dict(
            dofs=np.array([1, 5, 14]),
            method="3-point",
            abs_step=np.ones(3) * 1e-8,
            sp_fname=f"{curr_path}cyl_manyvol_test.toml",
        )
    )


@pytest.fixture(scope="module")
def tor():

    return input_dict(
        dict(
            dofs=np.array([1, 5, 37]),
            method="3-point",
            abs_step=np.ones(3) * 1e-8,
            sp_fname=f"{curr_path}qa_3vol_light.toml",
        )
    )


@pytest.mark.parametrize("lcons", [0, 1, 3])
@pytest.mark.parametrize("fixture", ["tor", "cyl"])
def test_force_real_jac(fixture, lcons, request):

    fixture = request.getfixturevalue(fixture)

    dofs = fixture.dofs
    method = fixture.method
    abs_step = fixture.abs_step
    sp_fname = fixture.sp_fname

    test = SPECTRE(sp_fname, verbose=False)
    test.input_list_mod.lcons = lcons

    xinit = get_xinit_specwrap(test)

    force_real(xinit, test)

    freal_jac_analytic_full = force_real_jac(xinit, test)

    freal_jac_numeric = approx_derivative(
        lambda x: force_real(x, test).flatten(order="C"),
        xinit,
        method=method,
        dofs=dofs + 0,
        abs_step=abs_step,
    )

    freal_jac_analytic = freal_jac_analytic_full[:, :, dofs]

    freal_jac_analytic = freal_jac_analytic.reshape(-1, len(dofs))

    maxelem = max(np.max(np.abs(freal_jac_numeric)), np.max(np.abs(freal_jac_analytic)))
    assert np.allclose(freal_jac_numeric, freal_jac_analytic, atol=5e-6 * maxelem, rtol=5e-6)

    remove_file(sp_fname + ".end")
